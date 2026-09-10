"""Build a module-preserving MaleCNS graph with a fixed RGB retina interface.

Unlike the compact hover scaffold, this builder does not select shortest routes.  It
retains every traced MaleCNS v1.0 neuron and every directed edge above a declared
synapse-count threshold.  Photoreceptor hex locations are inferred from their direct
connections to annotated retinotopic optic-lobe partners; no flight score or task image
is used in that inference.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from flydrone.connectome_data import (
    ANNOTATIONS_FILE,
    ATTITUDE_CHANNELS,
    NEUROTRANSMITTERS_FILE,
    OUTPUT_POOLS,
    TRANSMITTER_SIGN,
    WEIGHTS_FILE,
    _instance_side,
    _read_annotations,
    _read_transmitters,
    _sha256,
    stream_edges,
)

DRIVEN_PHOTORECEPTOR_TYPES = ("R1-R6", "R8p", "R8y")
RGB_CHANNEL_NAMES = ("red", "green", "blue")


@dataclass(frozen=True)
class FullConnectomeConfig:
    edge_weight_threshold: int = 10
    attitude_per_channel: int = 10


DEFAULT_FULL_CONNECTOME_CONFIG = FullConnectomeConfig()


@dataclass(frozen=True)
class RetinotopyAssignment:
    body_id: int
    hex1: int
    hex2: int
    total_weight: int
    winning_weight: int
    unique_coordinates: int

    @property
    def confidence(self) -> float:
        return self.winning_weight / self.total_weight


def infer_photoreceptor_retinotopy(
    weights_path: Path,
    receptor_ids: np.ndarray,
    partner_coordinates: dict[int, tuple[int, int]],
) -> dict[int, RetinotopyAssignment]:
    """Infer each receptor's column from direct outgoing anatomical connectivity.

    All released edge weights are considered, including edges below the recurrent graph
    threshold.  For each receptor, synapse counts are summed by the assigned hex of its
    postsynaptic partner.  The maximum-weight coordinate wins, with lexicographic tie
    breaking.  This is deterministic and independent of the drone task.
    """

    receptors = np.sort(np.asarray(receptor_ids, dtype=np.int64))
    partner_ids = np.sort(np.fromiter(partner_coordinates, dtype=np.int64))
    evidence: dict[int, dict[tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))

    with pa.memory_map(str(weights_path), "r") as source:
        reader = ipc.open_file(source)
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            presynaptic = batch.column(0).to_numpy(zero_copy_only=True)
            postsynaptic = batch.column(1).to_numpy(zero_copy_only=True)
            keep = np.isin(presynaptic, receptors) & np.isin(postsynaptic, partner_ids)
            if not np.any(keep):
                continue
            weights = batch.column(2).to_numpy(zero_copy_only=True)
            for pre, post, weight in zip(
                presynaptic[keep], postsynaptic[keep], weights[keep], strict=True
            ):
                evidence[int(pre)][partner_coordinates[int(post)]] += int(weight)

    assignments: dict[int, RetinotopyAssignment] = {}
    for body_id, coordinate_weights in evidence.items():
        ranked = sorted(coordinate_weights.items(), key=lambda item: (-item[1], item[0]))
        (hex1, hex2), winning_weight = ranked[0]
        total_weight = sum(coordinate_weights.values())
        assignments[body_id] = RetinotopyAssignment(
            body_id=body_id,
            hex1=hex1,
            hex2=hex2,
            total_weight=total_weight,
            winning_weight=winning_weight,
            unique_coordinates=len(coordinate_weights),
        )
    return assignments


def fixed_visual_grid(
    assignments: list[RetinotopyAssignment],
    receptor_sides: list[str],
    full_hex_bounds: dict[str, tuple[int, int, int, int]],
) -> np.ndarray:
    """Map inferred full-eye hex coordinates to normalized camera coordinates.

    The same frontal FPV image is projected to both virtual eyes.  Each eye uses bounds
    established from every traced, hex-assigned cell on that side, rather than bounds of
    the driven subset.  MaleCNS hex axes are treated as an affine camera grid; this is a
    declared engineering approximation, not a reconstruction of ommatidial optics.
    """

    if len(assignments) != len(receptor_sides):
        raise ValueError("each retinotopy assignment must have one receptor side")
    grid = np.empty((len(assignments), 2), dtype=np.float32)
    for row, (assignment, side) in enumerate(zip(assignments, receptor_sides, strict=True)):
        if side not in full_hex_bounds:
            raise ValueError(f"no full-eye hex bounds are available for side {side!r}")
        min_hex1, max_hex1, min_hex2, max_hex2 = full_hex_bounds[side]
        span1 = max(max_hex1 - min_hex1, 1)
        span2 = max(max_hex2 - min_hex2, 1)
        grid[row, 0] = 2.0 * (assignment.hex1 - min_hex1) / span1 - 1.0
        grid[row, 1] = -(2.0 * (assignment.hex2 - min_hex2) / span2 - 1.0)
    return np.clip(grid, -1.0, 1.0)


def fixed_spectral_weights(receptor_types: list[str]) -> np.ndarray:
    """Return a conservative, fixed linear RGB transduction for supported receptors."""

    mapping = {
        "R1-R6": np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32),
        "R8p": np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
        "R8y": np.asarray((0.0, 1.0, 0.0), dtype=np.float32),
    }
    try:
        return np.stack([mapping[receptor_type] for receptor_type in receptor_types])
    except KeyError as error:
        raise ValueError(f"unsupported driven photoreceptor type: {error.args[0]}") from error


def _annotation_side(annotations: dict[str, list[Any]], row: int) -> str | None:
    for field in ("rootSide", "somaSide"):
        side = annotations[field][row]
        if side in ("L", "R"):
            return side
    return _instance_side(annotations["instance"][row])


def _full_hex_bounds(
    annotations: dict[str, list[Any]], traced_mask: np.ndarray
) -> dict[str, tuple[int, int, int, int]]:
    points: dict[str, list[tuple[int, int]]] = {"L": [], "R": []}
    for row in np.flatnonzero(traced_mask):
        hex1 = annotations["assignedOlHex1"][row]
        hex2 = annotations["assignedOlHex2"][row]
        side = _annotation_side(annotations, int(row))
        if hex1 is not None and hex2 is not None and side in points:
            points[side].append((int(hex1), int(hex2)))
    bounds: dict[str, tuple[int, int, int, int]] = {}
    for side, side_points in points.items():
        if not side_points:
            continue
        values = np.asarray(side_points, dtype=np.int64)
        bounds[side] = (
            int(values[:, 0].min()),
            int(values[:, 0].max()),
            int(values[:, 1].min()),
            int(values[:, 1].max()),
        )
    return bounds


def build_full_visual_connectome(
    raw_dir: Path,
    output: Path,
    config: FullConnectomeConfig = DEFAULT_FULL_CONNECTOME_CONFIG,
) -> dict[str, Any]:
    """Build the full traced graph and its auditable camera-to-receptor interface."""

    annotation_path = raw_dir / ANNOTATIONS_FILE
    transmitter_path = raw_dir / NEUROTRANSMITTERS_FILE
    weights_path = raw_dir / WEIGHTS_FILE
    for path in (annotation_path, transmitter_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"required MaleCNS file is missing: {path}")
    if config.edge_weight_threshold < 1:
        raise ValueError("edge_weight_threshold must be positive")
    if config.attitude_per_channel < 1:
        raise ValueError("attitude_per_channel must be positive")

    annotations = _read_annotations(annotation_path)
    body_ids = np.asarray(annotations["bodyId"], dtype=np.int64)
    traced_mask = np.asarray([status == "Traced" for status in annotations["status"]])
    traced_ids = np.sort(body_ids[traced_mask])
    body_to_row = {int(body): row for row, body in enumerate(body_ids)}
    graph_index = {int(body): index for index, body in enumerate(traced_ids)}

    recurrent_pre, recurrent_post, recurrent_count = stream_edges(
        weights_path, config.edge_weight_threshold, traced_ids
    )
    recurrent_pre_index = np.searchsorted(traced_ids, recurrent_pre)
    recurrent_post_index = np.searchsorted(traced_ids, recurrent_post)
    transmitters = _read_transmitters(transmitter_path)
    presynaptic_labels = [transmitters.get(int(body), "unclear") for body in recurrent_pre]
    recurrent_sign = np.asarray(
        [TRANSMITTER_SIGN.get(label, 1.0) for label in presynaptic_labels], dtype=np.float32
    )

    receptor_rows = np.asarray(
        [
            row
            for row in np.flatnonzero(traced_mask)
            if annotations["superclass"][row] == "ol_sensory"
        ],
        dtype=np.int64,
    )
    receptor_ids = body_ids[receptor_rows]
    partner_coordinates = {
        int(body_ids[row]): (
            int(annotations["assignedOlHex1"][row]),
            int(annotations["assignedOlHex2"][row]),
        )
        for row in np.flatnonzero(traced_mask)
        if annotations["assignedOlHex1"][row] is not None
        and annotations["assignedOlHex2"][row] is not None
    }
    assignments_by_id = infer_photoreceptor_retinotopy(
        weights_path, receptor_ids, partner_coordinates
    )
    driven_rows = [
        int(row)
        for row in receptor_rows
        if annotations["type"][row] in DRIVEN_PHOTORECEPTOR_TYPES
        and int(body_ids[row]) in assignments_by_id
        and _annotation_side(annotations, int(row)) in ("L", "R")
    ]
    driven_rows.sort(key=lambda row: int(body_ids[row]))
    driven_ids = body_ids[driven_rows]
    assignments = [assignments_by_id[int(body)] for body in driven_ids]
    receptor_sides = [_annotation_side(annotations, row) for row in driven_rows]
    if any(side is None for side in receptor_sides):
        raise RuntimeError("driven photoreceptor lacks an anatomical side")
    side_strings = [str(side) for side in receptor_sides]
    hex_bounds = _full_hex_bounds(annotations, traced_mask)
    visual_grid = fixed_visual_grid(assignments, side_strings, hex_bounds)
    receptor_types = [str(annotations["type"][row]) for row in driven_rows]
    visual_channel_weights = fixed_spectral_weights(receptor_types)

    attitude_ids = np.sort(
        body_ids[
            np.asarray(
                [
                    traced and subclass == "wind_gravity"
                    for traced, subclass in zip(traced_mask, annotations["subclass"], strict=True)
                ]
            )
        ]
    )
    required_attitude = config.attitude_per_channel * len(ATTITUDE_CHANNELS)
    if len(attitude_ids) < required_attitude:
        raise RuntimeError(
            f"only {len(attitude_ids)} traced wind/gravity cells; need {required_attitude}"
        )
    attitude_ids = attitude_ids[:required_attitude]
    attitude_channels = np.tile(
        np.arange(len(ATTITUDE_CHANNELS), dtype=np.int64), config.attitude_per_channel
    )

    output_body_ids: dict[str, np.ndarray] = {}
    for name, (side, neuron_type) in OUTPUT_POOLS.items():
        mask = np.asarray(
            [
                traced
                and superclass == "vnc_motor"
                and subclass == "fl"
                and neuromere == "T1"
                and cell_type == neuron_type
                and _instance_side(instance) == side
                for traced, superclass, subclass, neuromere, cell_type, instance in zip(
                    traced_mask,
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

    pool_members: list[int] = []
    pool_offsets = [0]
    for name in OUTPUT_POOLS:
        pool_members.extend(graph_index[int(body)] for body in output_body_ids[name])
        pool_offsets.append(len(pool_members))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        node_ids=traced_ids,
        edge_pre=recurrent_pre_index.astype(np.int64),
        edge_post=recurrent_post_index.astype(np.int64),
        edge_count=recurrent_count.astype(np.int32),
        edge_sign=recurrent_sign,
        visual_node_indices=np.searchsorted(traced_ids, driven_ids).astype(np.int64),
        visual_hex=np.asarray([(item.hex1, item.hex2) for item in assignments], dtype=np.float32),
        visual_eye=np.asarray([-1 if side == "L" else 1 for side in side_strings], dtype=np.int8),
        visual_grid=visual_grid,
        visual_channel_weights=visual_channel_weights,
        attitude_node_indices=np.searchsorted(traced_ids, attitude_ids).astype(np.int64),
        attitude_channels=attitude_channels,
        output_pool_offsets=np.asarray(pool_offsets, dtype=np.int64),
        output_pool_indices=np.asarray(pool_members, dtype=np.int64),
    )

    receptor_type_counts = Counter(str(annotations["type"][row]) for row in receptor_rows)
    mapped_type_counts = Counter(receptor_types)
    superclass_counts = Counter(
        str(annotations["superclass"][body_to_row[int(body)]]) for body in traced_ids
    )
    transmitter_counts = Counter(transmitters.get(int(body), "unclear") for body in traced_ids)
    confidences = np.asarray([item.confidence for item in assignments], dtype=np.float64)
    unique_coordinates = np.asarray(
        [item.unique_coordinates for item in assignments], dtype=np.int64
    )
    manifest: dict[str, Any] = {
        "format": "flydrone-full-visual-connectome-v1",
        "source": "MaleCNS v1.0",
        "source_url": "https://male-cns.janelia.org/download/",
        "source_license": "CC BY 4.0",
        "source_license_url": "https://creativecommons.org/licenses/by/4.0/",
        "transformation_notice": (
            "This project retained the full traced graph above a declared edge threshold, "
            "converted synapse counts to normalized initial magnitudes at load time, and "
            "added fixed engineered camera/sensor/output mappings."
        ),
        "config": {
            "edge_weight_threshold": config.edge_weight_threshold,
            "attitude_per_channel": config.attitude_per_channel,
        },
        "raw_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (annotation_path, transmitter_path, weights_path)
        },
        "selection": {
            "nodes": int(len(traced_ids)),
            "edges": int(len(recurrent_pre_index)),
            "superclasses": dict(sorted(superclass_counts.items())),
            "transmitters": dict(sorted(transmitter_counts.items())),
            "visual_sensory_nodes_retained": int(len(receptor_rows)),
            "visual_sensory_nodes_driven": int(len(driven_ids)),
            "attitude_nodes_driven": int(len(attitude_ids)),
        },
        "visual_interface": {
            "camera": {
                "width": 320,
                "height": 200,
                "horizontal_fov_degrees": 125.0,
                "projection": "rectangular pinhole with square pixels",
            },
            "projection": (
                "The same frontal camera is affinely projected to both eyes using full-eye "
                "MaleCNS hex bounds; hex axes are an engineering approximation to image axes."
            ),
            "full_hex_bounds": {side: list(bounds) for side, bounds in hex_bounds.items()},
            "retinotopy_inference": (
                "Maximum aggregate released synapse weight from each ol_sensory receptor "
                "to direct postsynaptic partners sharing an assigned optic-lobe hex."
            ),
            "receptors_by_type": dict(sorted(receptor_type_counts.items())),
            "driven_by_type": dict(sorted(mapped_type_counts.items())),
            "supported_types": list(DRIVEN_PHOTORECEPTOR_TYPES),
            "spectral_mapping": {
                "R1-R6": "linear RGB luminance (0.2126, 0.7152, 0.0722)",
                "R8p": "blue camera channel",
                "R8y": "green camera channel",
            },
            "rgb_channels": list(RGB_CHANNEL_NAMES),
            "r7_policy": (
                "R7 and its downstream circuitry are retained but receive no invented UV "
                "drive because an RGB camera has no ultraviolet channel."
            ),
            "other_exclusions": (
                "Unclear and dorsal-specialized receptor types remain in the recurrent graph "
                "but are not directly driven in v1."
            ),
            "mapped_receptors": int(len(assignments)),
            "all_type_receptors_with_coordinate_evidence": int(len(assignments_by_id)),
            "confidence": {
                "median_winning_weight_fraction": float(np.median(confidences)),
                "p10_winning_weight_fraction": float(np.quantile(confidences, 0.1)),
                "p95_unique_coordinates": float(np.quantile(unique_coordinates, 0.95)),
                "single_coordinate_fraction": float(np.mean(unique_coordinates == 1)),
            },
        },
        "attitude_interface": {
            "injection_layer": "MaleCNS wind_gravity mechanosensory cells",
            "measurement": "estimated roll and pitch angles",
            "channels": list(ATTITUDE_CHANNELS),
            "cells_per_channel": config.attitude_per_channel,
            "can_be_zeroed_for_visual_only_ablation": True,
        },
        "accelerometer_interface": {"enabled": False},
        "output_pool_order": list(OUTPUT_POOLS),
        "output_pools": {
            name: {
                "side": OUTPUT_POOLS[name][0],
                "type": OUTPUT_POOLS[name][1],
                "body_ids": [int(body) for body in output_body_ids[name]],
            }
            for name in OUTPUT_POOLS
        },
        "transmitter_sign_hypothesis": TRANSMITTER_SIGN,
    }
    output.with_suffix(".json").write_text(
        f"{json.dumps(manifest, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    return manifest
