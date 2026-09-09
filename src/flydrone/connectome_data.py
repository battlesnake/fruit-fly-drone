"""Build a compact, auditable MaleCNS sensor-to-foreleg graph.

The raw release is an Arrow IPC/Feather file with more than 151 million rows.  This
module deliberately scans it in record batches; loading the whole table just to select
the hover scaffold takes several gigabytes and obscures the selection procedure.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy.sparse import csr_matrix

ANNOTATIONS_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS_FILE = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# These are fixed motor-to-stick synergies, not a learned output decoder.  Each name
# denotes the population whose increased activity pushes the corresponding stick axis.
OUTPUT_POOLS: dict[str, tuple[str, str]] = {
    "roll_pos": ("R", "Tergopleural/Pleural promotor MN"),
    "roll_neg": ("R", "Pleural remotor/abductor MN"),
    "pitch_pos": ("R", "Ti extensor MN"),
    "pitch_neg": ("R", "Ti flexor MN"),
    "yaw_pos": ("L", "Tergopleural/Pleural promotor MN"),
    "yaw_neg": ("L", "Pleural remotor/abductor MN"),
    "throttle_pos": ("L", "Ti extensor MN"),
    "throttle_neg": ("L", "Ti flexor MN"),
}

ATTITUDE_CHANNELS = ("roll_pos", "roll_neg", "pitch_pos", "pitch_neg")
ACCELERATION_CHANNELS = ("specific_force_z_above_1g", "specific_force_z_below_1g")
ACCELERATION_OUTPUT_POOLS = ("throttle_pos", "throttle_neg")

# Presynaptic transmitter hypotheses.  The raw labels and counts remain in the derived
# manifest so these signs can be ablated.  "unclear" is kept excitatory rather than
# silently dropping an anatomical edge.
TRANSMITTER_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": 1.0,
    "octopamine": 1.0,
    "serotonin": 1.0,
    "tyramine": 1.0,
    "unclear": 1.0,
}


@dataclass(frozen=True)
class BuildConfig:
    path_weight_threshold: int = 50
    induced_weight_threshold: int = 10
    max_path_hops: int = 10
    visual_per_eye: int = 24
    attitude_per_channel: int = 8
    acceleration_per_channel: int = 0


def _read_annotations(path: Path) -> dict[str, list[Any]]:
    columns = (
        "bodyId",
        "type",
        "instance",
        "superclass",
        "subclass",
        "somaNeuromere",
        "status",
        "assignedOlHex1",
        "assignedOlHex2",
        "rootSide",
        "somaSide",
    )
    return feather.read_table(path, columns=columns, memory_map=True).to_pydict()


def _read_transmitters(path: Path) -> dict[int, str]:
    table = feather.read_table(path, columns=("body", "consensus_nt"), memory_map=True)
    bodies = table["body"].to_numpy(zero_copy_only=False)
    labels = table["consensus_nt"].to_pylist()
    return {
        int(body): (label or "unclear").lower() for body, label in zip(bodies, labels, strict=True)
    }


def stream_edges(
    path: Path, threshold: int, eligible_ids: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read qualifying edges without materializing the complete Feather table."""

    source_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    eligible = None if eligible_ids is None else np.sort(eligible_ids)

    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            weights = batch.column(2).to_numpy(zero_copy_only=True)
            keep = weights >= threshold
            if not np.any(keep):
                continue
            presynaptic = batch.column(0).to_numpy(zero_copy_only=True)[keep]
            postsynaptic = batch.column(1).to_numpy(zero_copy_only=True)[keep]
            selected_weights = weights[keep]
            if eligible is not None:
                keep_eligible = np.isin(presynaptic, eligible) & np.isin(postsynaptic, eligible)
                presynaptic = presynaptic[keep_eligible]
                postsynaptic = postsynaptic[keep_eligible]
                selected_weights = selected_weights[keep_eligible]
            source_chunks.append(presynaptic.copy())
            target_chunks.append(postsynaptic.copy())
            weight_chunks.append(selected_weights.copy())

    if not source_chunks:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    return (
        np.concatenate(source_chunks),
        np.concatenate(target_chunks),
        np.concatenate(weight_chunks),
    )


def farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    """Return deterministic, spatially spread row indices."""

    if count < 1 or len(points) == 0:
        return np.empty(0, dtype=np.int64)
    count = min(count, len(points))
    span = np.ptp(points, axis=0)
    normalized = (points - points.min(axis=0)) / np.where(span > 0, span, 1.0)
    centroid = normalized.mean(axis=0)
    chosen = [int(np.argmax(np.square(normalized - centroid).sum(axis=1)))]
    min_distance = np.square(normalized - normalized[chosen[0]]).sum(axis=1)
    for _ in range(1, count):
        next_index = int(np.argmax(min_distance))
        chosen.append(next_index)
        distance = np.square(normalized - normalized[next_index]).sum(axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[chosen] = -1.0
    return np.asarray(chosen, dtype=np.int64)


def reverse_bfs_next(adjacency: csr_matrix, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return distance and one deterministic next hop from every node to targets."""

    reverse = adjacency.transpose().tocsr()
    distance = np.full(adjacency.shape[0], -1, dtype=np.int16)
    next_hop = np.full(adjacency.shape[0], -1, dtype=np.int64)
    queue: deque[int] = deque()
    for target in np.sort(np.unique(targets)):
        distance[target] = 0
        queue.append(int(target))
    while queue:
        current = queue.popleft()
        begin, end = reverse.indptr[current : current + 2]
        for predecessor in reverse.indices[begin:end]:
            if distance[predecessor] >= 0:
                continue
            distance[predecessor] = distance[current] + 1
            next_hop[predecessor] = current
            queue.append(int(predecessor))
    return distance, next_hop


def trace_path(start: int, distance: np.ndarray, next_hop: np.ndarray, max_hops: int) -> list[int]:
    if distance[start] < 0 or distance[start] > max_hops:
        return []
    path = [start]
    while distance[path[-1]] > 0:
        following = int(next_hop[path[-1]])
        if following < 0:
            raise RuntimeError("BFS path terminated before reaching its motor pool")
        path.append(following)
    return path


def _instance_side(instance: str | None) -> str | None:
    if instance and instance.endswith("_L"):
        return "L"
    if instance and instance.endswith("_R"):
        return "R"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_hover_scaffold(raw_dir: Path, output: Path, config: BuildConfig) -> dict[str, Any]:
    """Create the selected graph and return its JSON-serializable manifest."""

    annotation_path = raw_dir / ANNOTATIONS_FILE
    transmitter_path = raw_dir / NEUROTRANSMITTERS_FILE
    weights_path = raw_dir / WEIGHTS_FILE
    for path in (annotation_path, transmitter_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"required MaleCNS file is missing: {path}")

    annotations = _read_annotations(annotation_path)
    body_ids = np.asarray(annotations["bodyId"], dtype=np.int64)
    traced_mask = np.asarray([status == "Traced" for status in annotations["status"]])
    eligible_ids = np.sort(body_ids[traced_mask])
    body_to_annotation = {int(body): row for row, body in enumerate(body_ids)}

    path_pre, path_post, path_count = stream_edges(
        weights_path, config.path_weight_threshold, eligible_ids
    )
    pre_index = np.searchsorted(eligible_ids, path_pre)
    post_index = np.searchsorted(eligible_ids, path_post)
    adjacency = csr_matrix(
        (np.ones(len(path_pre), dtype=np.int8), (pre_index, post_index)),
        shape=(len(eligible_ids), len(eligible_ids)),
    )
    adjacency.sum_duplicates()
    adjacency.data[:] = 1

    output_body_ids: dict[str, np.ndarray] = {}
    output_indices: dict[str, np.ndarray] = {}
    for name, (side, neuron_type) in OUTPUT_POOLS.items():
        mask = np.asarray(
            [
                status == "Traced"
                and superclass == "vnc_motor"
                and subclass == "fl"
                and neuromere == "T1"
                and cell_type == neuron_type
                and _instance_side(instance) == side
                for status, superclass, subclass, neuromere, cell_type, instance in zip(
                    annotations["status"],
                    annotations["superclass"],
                    annotations["subclass"],
                    annotations["somaNeuromere"],
                    annotations["type"],
                    annotations["instance"],
                    strict=True,
                )
            ]
        )
        pool_ids = np.sort(body_ids[mask])
        if len(pool_ids) == 0:
            raise RuntimeError(f"no annotated cells found for output pool {name}")
        output_body_ids[name] = pool_ids
        output_indices[name] = np.searchsorted(eligible_ids, pool_ids)

    routes = {
        name: reverse_bfs_next(adjacency, indices) for name, indices in output_indices.items()
    }

    visual_mask = np.asarray(
        [
            status == "Traced" and cell_type == "L1" and hex1 is not None and hex2 is not None
            for status, cell_type, hex1, hex2 in zip(
                annotations["status"],
                annotations["type"],
                annotations["assignedOlHex1"],
                annotations["assignedOlHex2"],
                strict=True,
            )
        ]
    )
    visual_candidates = body_ids[visual_mask]
    visual_rows = np.flatnonzero(visual_mask)
    visual_global_indices = np.searchsorted(eligible_ids, visual_candidates)
    reachable_visual = np.ones(len(visual_candidates), dtype=bool)
    for distance, _ in routes.values():
        reachable_visual &= (distance[visual_global_indices] >= 0) & (
            distance[visual_global_indices] <= config.max_path_hops
        )
    visual_candidates = visual_candidates[reachable_visual]
    visual_rows = visual_rows[reachable_visual]

    selected_visual_rows: list[int] = []
    for side in ("L", "R"):
        side_positions = np.asarray(
            [row for row in visual_rows if annotations["somaSide"][row] == side],
            dtype=np.int64,
        )
        points = np.asarray(
            [
                (annotations["assignedOlHex1"][row], annotations["assignedOlHex2"][row])
                for row in side_positions
            ],
            dtype=np.float32,
        )
        selected_visual_rows.extend(
            side_positions[farthest_point_sample(points, config.visual_per_eye)].tolist()
        )
    selected_visual_rows_array = np.asarray(selected_visual_rows, dtype=np.int64)
    selected_visual_ids = body_ids[selected_visual_rows_array]

    attitude_mask = np.asarray(
        [
            status == "Traced" and subclass == "wind_gravity"
            for status, subclass in zip(annotations["status"], annotations["subclass"], strict=True)
        ]
    )
    attitude_candidates = np.sort(body_ids[attitude_mask])
    attitude_global_indices = np.searchsorted(eligible_ids, attitude_candidates)
    reachable_attitude = np.ones(len(attitude_candidates), dtype=bool)
    for distance, _ in routes.values():
        reachable_attitude &= (distance[attitude_global_indices] >= 0) & (
            distance[attitude_global_indices] <= config.max_path_hops
        )
    attitude_candidates = attitude_candidates[reachable_attitude]
    attitude_global_indices = np.searchsorted(eligible_ids, attitude_candidates)
    required_attitude = config.attitude_per_channel * len(ATTITUDE_CHANNELS)
    if len(attitude_candidates) < required_attitude:
        raise RuntimeError(
            f"only {len(attitude_candidates)} attitude cells reach every output pool; "
            f"need {required_attitude}"
        )
    # Interleave the sorted candidates so each push/pull channel spans the ID range.
    positions = np.linspace(0, len(attitude_candidates) - 1, required_attitude, dtype=int)
    selected_attitude_ids = attitude_candidates[positions]
    attitude_channels = np.tile(
        np.arange(len(ATTITUDE_CHANNELS), dtype=np.int64), config.attitude_per_channel
    )

    selected_input_ids = np.concatenate((selected_visual_ids, selected_attitude_ids))
    selected_input_indices = np.searchsorted(eligible_ids, selected_input_ids)
    selected_nodes: set[int] = set()
    route_lengths: list[int] = []
    for input_index in selected_input_indices:
        for distance, next_hop in routes.values():
            path = trace_path(int(input_index), distance, next_hop, config.max_path_hops)
            if not path:
                raise RuntimeError("selected sensory cell does not reach every motor pool")
            selected_nodes.update(path)
            route_lengths.append(len(path) - 1)
    for pool in output_indices.values():
        selected_nodes.update(int(index) for index in pool)

    # Additional Johnston's-organ cells carry the optional accelerometer interface.
    # They remain disjoint from the complete baseline graph, not merely its established
    # attitude inputs, so zeroing new-to-old boundaries reproduces the warm start exactly.
    # They need real paths only to the two throttle pools used by this first experiment.
    acceleration_routes = {name: routes[name] for name in ACCELERATION_OUTPUT_POOLS}
    reachable_acceleration = np.ones(len(attitude_candidates), dtype=bool)
    for distance, _ in acceleration_routes.values():
        reachable_acceleration &= (distance[attitude_global_indices] >= 0) & (
            distance[attitude_global_indices] <= config.max_path_hops
        )
    acceleration_candidates = attitude_candidates[reachable_acceleration]
    acceleration_candidates = np.setdiff1d(
        acceleration_candidates,
        eligible_ids[np.asarray(sorted(selected_nodes), dtype=np.int64)],
        assume_unique=True,
    )
    required_acceleration = config.acceleration_per_channel * len(ACCELERATION_CHANNELS)
    if len(acceleration_candidates) < required_acceleration:
        raise RuntimeError(
            f"only {len(acceleration_candidates)} unused acceleration cells reach both "
            f"throttle pools; need {required_acceleration}"
        )
    if required_acceleration:
        positions = np.linspace(
            0, len(acceleration_candidates) - 1, required_acceleration, dtype=int
        )
        selected_acceleration_ids = acceleration_candidates[positions]
    else:
        selected_acceleration_ids = np.empty(0, dtype=np.int64)
    acceleration_channels = np.tile(
        np.arange(len(ACCELERATION_CHANNELS), dtype=np.int64),
        config.acceleration_per_channel,
    )

    acceleration_input_indices = np.searchsorted(eligible_ids, selected_acceleration_ids)
    for input_index in acceleration_input_indices:
        for distance, next_hop in acceleration_routes.values():
            path = trace_path(int(input_index), distance, next_hop, config.max_path_hops)
            if not path:
                raise RuntimeError("selected acceleration cell does not reach both throttle pools")
            selected_nodes.update(path)
            route_lengths.append(len(path) - 1)
    selected_global_indices = np.asarray(sorted(selected_nodes), dtype=np.int64)
    selected_body_ids = eligible_ids[selected_global_indices]
    graph_index = {int(body): index for index, body in enumerate(selected_body_ids)}

    induced_pre, induced_post, induced_count = stream_edges(
        weights_path, config.induced_weight_threshold, selected_body_ids
    )
    induced_pre_index = np.asarray([graph_index[int(body)] for body in induced_pre], dtype=np.int64)
    induced_post_index = np.asarray(
        [graph_index[int(body)] for body in induced_post], dtype=np.int64
    )
    transmitters = _read_transmitters(transmitter_path)
    presynaptic_labels = [transmitters.get(int(body), "unclear") for body in induced_pre]
    induced_sign = np.asarray(
        [TRANSMITTER_SIGN.get(label, 1.0) for label in presynaptic_labels], dtype=np.float32
    )

    visual_graph_indices = np.asarray(
        [graph_index[int(body)] for body in selected_visual_ids], dtype=np.int64
    )
    visual_hex = np.asarray(
        [
            (
                annotations["assignedOlHex1"][row],
                annotations["assignedOlHex2"][row],
            )
            for row in selected_visual_rows_array
        ],
        dtype=np.float32,
    )
    visual_eye = np.asarray(
        [-1 if annotations["somaSide"][row] == "L" else 1 for row in selected_visual_rows_array],
        dtype=np.int8,
    )
    attitude_graph_indices = np.asarray(
        [graph_index[int(body)] for body in selected_attitude_ids], dtype=np.int64
    )
    acceleration_graph_indices = np.asarray(
        [graph_index[int(body)] for body in selected_acceleration_ids], dtype=np.int64
    )

    pool_names = tuple(OUTPUT_POOLS)
    pool_members: list[int] = []
    pool_offsets = [0]
    for name in pool_names:
        pool_members.extend(
            graph_index[int(body)] for body in output_body_ids[name] if int(body) in graph_index
        )
        pool_offsets.append(len(pool_members))

    output.parent.mkdir(parents=True, exist_ok=True)
    graph_arrays = {
        "node_ids": selected_body_ids,
        "edge_pre": induced_pre_index,
        "edge_post": induced_post_index,
        "edge_count": induced_count.astype(np.int32),
        "edge_sign": induced_sign,
        "visual_node_indices": visual_graph_indices,
        "visual_hex": visual_hex,
        "visual_eye": visual_eye,
        "attitude_node_indices": attitude_graph_indices,
        "attitude_channels": attitude_channels,
        "output_pool_offsets": np.asarray(pool_offsets, dtype=np.int64),
        "output_pool_indices": np.asarray(pool_members, dtype=np.int64),
    }
    if required_acceleration:
        graph_arrays.update(
            acceleration_node_indices=acceleration_graph_indices,
            acceleration_channels=acceleration_channels,
        )
    np.savez_compressed(output, **graph_arrays)

    node_types = [annotations["type"][body_to_annotation[int(body)]] for body in selected_body_ids]
    transmitter_counts: dict[str, int] = {}
    for body in selected_body_ids:
        label = transmitters.get(int(body), "unclear")
        transmitter_counts[label] = transmitter_counts.get(label, 0) + 1
    manifest: dict[str, Any] = {
        "format": (
            "flydrone-connectome-v2"
            if required_acceleration
            else "flydrone-hover-connectome-v1"
        ),
        "source": "MaleCNS v1.0",
        "source_url": "https://male-cns.janelia.org/download/",
        "source_license": "CC BY 4.0",
        "source_license_url": "https://creativecommons.org/licenses/by/4.0/",
        "transformation_notice": (
            "This project selected a task-specific subgraph, converted synapse counts "
            "to normalized initial magnitudes, and added engineered sensor/output mappings."
        ),
        "config": {
            "path_weight_threshold": config.path_weight_threshold,
            "induced_weight_threshold": config.induced_weight_threshold,
            "max_path_hops": config.max_path_hops,
            "visual_per_eye": config.visual_per_eye,
            "attitude_per_channel": config.attitude_per_channel,
            "acceleration_per_channel": config.acceleration_per_channel,
        },
        "raw_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (annotation_path, transmitter_path, weights_path)
        },
        "selection": {
            "eligible_traced_nodes": int(len(eligible_ids)),
            "path_edges": int(adjacency.nnz),
            "nodes": int(len(selected_body_ids)),
            "edges": int(len(induced_pre_index)),
            "visual_nodes": int(len(visual_graph_indices)),
            "attitude_nodes": int(len(attitude_graph_indices)),
            "acceleration_nodes": int(len(acceleration_graph_indices)),
            "route_hops_min": int(min(route_lengths)),
            "route_hops_max": int(max(route_lengths)),
            "node_types": len({cell_type for cell_type in node_types if cell_type}),
            "transmitters": dict(sorted(transmitter_counts.items())),
        },
        "visual_interface": {
            "injection_layer": "MaleCNS L1 cells with assignedOlHex1/2",
            "reason": "MaleCNS ol_sensory photoreceptor rows do not carry assigned hex coordinates",
            "eyes": {"L": config.visual_per_eye, "R": config.visual_per_eye},
        },
        "attitude_interface": {
            "injection_layer": "MaleCNS wind_gravity mechanosensory cells",
            "channels": list(ATTITUDE_CHANNELS),
            "cells_per_channel": config.attitude_per_channel,
        },
        "acceleration_interface": {
            "injection_layer": "additional MaleCNS wind_gravity mechanosensory cells",
            "measurement": "body-frame specific force from the completed physics interval",
            "normalization": "(body_z - 1g) / 0.25g, clamped to +/-2, push/pull",
            "channels": list(ACCELERATION_CHANNELS),
            "cells_per_channel": config.acceleration_per_channel,
            "reachable_output_pools": list(ACCELERATION_OUTPUT_POOLS),
            "engineering_mapping_not_claimed_physiology": True,
        },
        "output_pools": {
            name: {
                "side": OUTPUT_POOLS[name][0],
                "type": OUTPUT_POOLS[name][1],
                "body_ids": [
                    int(body) for body in output_body_ids[name] if int(body) in graph_index
                ],
            }
            for name in pool_names
        },
        "output_pool_order": list(pool_names),
        "transmitter_sign_hypothesis": TRANSMITTER_SIGN,
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(
        f"{json.dumps(manifest, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    return manifest
