#!/usr/bin/env python3
"""Freeze the anatomy and stimuli for vertical T4/T5 commissioning.

This is deliberately a no-response manifest: it reads annotations and graph topology,
and it creates stimulus *specifications*, but it never renders a pixel, constructs a
controller, loads neural parameters, or evaluates a neural response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flydrone.connectome_data import ANNOTATIONS_FILE, _read_annotations  # noqa: E402

EXPERIMENT = "vertical-t4t5-local-commissioning-manifest-v1"
PROTOCOL_COMMIT = "461ca9d"

EXPECTED_GRAPH_SHA256 = "8c6ba28d149e9ac4a5223c5919657a1734f2c2cac9828c51114fd0174a383665"
EXPECTED_CHECKPOINT_SHA256 = "7238b0e3ca39dc1a8bfd35dcf9f8b6fc9e8c8881ff989f64cb135fa6fed4e572"
EXPECTED_ANNOTATIONS_SHA256 = "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"
EXPECTED_FROZEN_AUDIT_REPORT_SHA256 = (
    "11305380a565651c25f7d0b8e0a564b42bddc1e457b7e7b575fb3839583e5af8"
)

TARGET_SUBTYPES = ("T4c", "T4d", "T5c", "T5d")
T4_SOURCES = (("Mi1", 1), ("Mi4", -1), ("Mi9", -1), ("Tm3", 1))
T5_SOURCES = (("Tm1", 1), ("Tm2", 1), ("Tm4", 1), ("Tm9", 1))
GROUPS = tuple(
    (target, source, sign)
    for target, sources in (
        ("T4c", T4_SOURCES),
        ("T4d", T4_SOURCES),
        ("T5c", T5_SOURCES),
        ("T5d", T5_SOURCES),
    )
    for source, sign in sources
)

EXPECTED_GROUP_EDGE_COUNTS = (
    4036,
    317,
    1435,
    1335,
    3388,
    250,
    1238,
    1108,
    1390,
    2458,
    788,
    2743,
    1364,
    2013,
    697,
    2221,
)
EXPECTED_GROUP_SYNAPSE_COUNTS = (
    81286,
    3659,
    19304,
    16346,
    70642,
    2827,
    16597,
    13394,
    17996,
    39041,
    9356,
    40495,
    17819,
    32185,
    8314,
    32963,
)
EXPECTED_EDGE_COUNT = 26_781
EXPECTED_SYNAPSE_COUNT = 422_224
EXPECTED_EDGE_INDICES_SHA256 = "38145c1032f1f8533f1e238cf6625d83b2e6651ffa163826a96f5e9e1aa808c1"
EXPECTED_EDGE_GROUP_PAIRS_SHA256 = (
    "4ed47bbb3fe9e5f97d20b94b9ad21fd94c646ce32baecac6fb89bce995cde527"
)
EXPECTED_TARGET_COUNT = 6_827
EXPECTED_TARGET_INDICES_SHA256 = "5abd5dba0452da06b7439f88cd726cac524a271d912cd012fec3c026c65f7145"
EXPECTED_TARGET_BODY_IDS_SHA256 = "6cd9debc539446d73b4a86cf63d171d58e14736358ebb7ffb06841b526c3286c"
EXPECTED_TARGET_SUBTYPE_PAIRS_SHA256 = (
    "1447b05b32cd08c033a5988c6850e102d9ff4ac208dbad8e24a1be0687f29db9"
)

WIDTH = 320
HEIGHT = 200
HFOV_DEGREES = 125.0
CAMERA_HZ = 50
PREFIX_FRAMES = 25
MOTION_FRAMES = 16
TERMINAL_FRAMES = 1
CNS_SUBSTEPS_PER_FRAME = 32
PHASE_DENOMINATOR = 1 << 53
EXPECTED_NUMPY_VERSION = "2.5.3"

SPLITS: dict[str, dict[str, Any]] = {
    "training": {
        "seed": 481_101,
        "edge_speeds": (1, 2, 4, 6),
        "edge_phases": 6,
        "edge_phase_bins": (4, 16, 28, 40, 52, 64),
        "texture_speeds": (1, 2, 4, 6),
        "texture_realizations": 12,
        "expected_pairs": 96,
        "pixel_access": "may_render_for_training_after_preflight_authorization",
    },
    "development": {
        "seed": 481_211,
        "edge_speeds": (1, 4, 6),
        "edge_phases": 4,
        "edge_phase_bins": (10, 30, 50, 70),
        "texture_speeds": (1, 4, 6),
        "texture_realizations": 8,
        "expected_pairs": 48,
        "pixel_access": "may_render_only_at_updates_25_50_100_after_numerical_gate",
    },
    "acceptance": {
        "seed": 481_307,
        "edge_speeds": (1, 2, 3, 4),
        "edge_phases": 8,
        "edge_phase_bins": (1, 11, 21, 31, 41, 51, 61, 71),
        "texture_speeds": (1, 2, 3, 4),
        "texture_realizations": 16,
        "expected_pairs": 128,
        "pixel_access": "sealed_until_a_development_snapshot_passes",
    },
}

# Filled from specification-only generation. They intentionally cover no rendered pixels
# and no neural response. The constants are replaced exactly once before this manifest lands.
EXPECTED_SPLIT_SPEC_SHA256 = {
    "training": "df7c27579ac4874fce28e00090b239fd126996dfe9b55d8d4e5970f3ba93c60d",
    "development": "ffc568742e57f95aa09dc38015cbff612f3d7fa5fdda011b7885350f40115642",
    "acceptance": "024e4fee8e3756dc00413687fc7472c53fb1f86ae1bc162a8259d4e3e6fccec8",
}
EXPECTED_ALL_SPECS_SHA256 = "b61aa15d0b8d1c60d0de5414c411b96dc26431c1bc1695528c18aa6b48647ceb"
EXPECTED_MANIFEST_SEMANTIC_SHA256 = (
    "8d40ae087c1552f0473be47ede74a11f220c1d7474e4b4025a251839f1dcdfef"
)
EXPECTED_SCHEMA_SEMANTIC_SHA256 = "1d91507a56f30a3984e6f78b86f81dedab7a92fdddfc3a005aec3817274b94fc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
    )
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw/malecns-v1.0")
    parser.add_argument(
        "--frozen-audit-report",
        type=Path,
        default=REPO_ROOT / "runs/optic-motion/frozen-t4t5-audit-002/report.json",
    )
    parser.add_argument("--write", type=Path)
    parser.add_argument(
        "--compute-hashes",
        action="store_true",
        help="print generated semantic hashes without comparing placeholder locks",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def semantic_sha256(value: Any) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def int64_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<i8")
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def uint64_array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<u8")
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def stable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def validate_locked_inputs(args: argparse.Namespace) -> dict[str, str]:
    expected = {
        args.graph: EXPECTED_GRAPH_SHA256,
        args.checkpoint: EXPECTED_CHECKPOINT_SHA256,
        args.raw_dir / ANNOTATIONS_FILE: EXPECTED_ANNOTATIONS_SHA256,
        args.frozen_audit_report: EXPECTED_FROZEN_AUDIT_REPORT_SHA256,
    }
    observed: dict[str, str] = {}
    for path, digest in expected.items():
        if not path.is_file() or file_sha256(path) != digest:
            raise SystemExit(f"locked commissioning input is missing or changed: {path}")
        observed[stable_path(path)] = digest

    with args.frozen_audit_report.open() as stream:
        report = json.load(stream)
    if (
        report.get("classification") != "frozen_optic_motion_module_validly_uncommissioned"
        or report.get("passed")
        or not report.get("local_motion_commissioning_preregistration_authorized")
        or report.get("motion_output_routing_preregistration_authorized")
        or not report.get("source_restored")
    ):
        raise SystemExit("frozen audit report is not the registered commissioning authority")
    return observed


def anatomy_manifest(graph_path: Path, annotations_path: Path) -> dict[str, Any]:
    graph = np.load(graph_path)
    node_ids = graph["node_ids"]
    edge_pre = graph["edge_pre"]
    edge_post = graph["edge_post"]
    edge_count = graph["edge_count"]
    edge_sign = graph["edge_sign"]
    annotations = _read_annotations(annotations_path)
    rows = {int(body): row for row, body in enumerate(annotations["bodyId"])}
    node_types = np.asarray(
        [str(annotations["type"][rows[int(body)]]) for body in node_ids], dtype=object
    )

    edge_group = np.full(len(edge_pre), -1, dtype=np.int64)
    group_rows = []
    for group_index, (target, source, transmitter_sign) in enumerate(GROUPS):
        selected = (node_types[edge_pre] == source) & (node_types[edge_post] == target)
        selected_signs = np.unique(edge_sign[selected])
        if not np.array_equal(
            selected_signs, np.asarray([transmitter_sign], dtype=edge_sign.dtype)
        ):
            raise SystemExit(f"transmitter sign changed for commissioning group {source}->{target}")
        edge_group[selected] = group_index
        indices = np.flatnonzero(selected).astype(np.int64)
        group_rows.append(
            {
                "group_index": group_index,
                "source_type": source,
                "target_subtype": target,
                "fixed_transmitter_sign": transmitter_sign,
                "edge_count": int(len(indices)),
                "synapse_count": int(edge_count[selected].sum(dtype=np.int64)),
                "edge_indices_sha256": int64_array_sha256(indices),
            }
        )

    edge_indices = np.flatnonzero(edge_group >= 0).astype(np.int64)
    edge_group_pairs = np.column_stack((edge_indices, edge_group[edge_indices])).astype(np.int64)
    target_indices = np.flatnonzero(np.isin(node_types, TARGET_SUBTYPES)).astype(np.int64)
    target_subtypes = np.asarray(
        [TARGET_SUBTYPES.index(str(node_types[index])) for index in target_indices],
        dtype=np.int64,
    )
    target_body_ids = node_ids[target_indices].astype(np.int64)
    target_subtype_pairs = np.column_stack((target_indices, target_subtypes)).astype(np.int64)

    observed = {
        "group_edge_counts": tuple(row["edge_count"] for row in group_rows),
        "group_synapse_counts": tuple(row["synapse_count"] for row in group_rows),
        "edge_count": int(len(edge_indices)),
        "synapse_count": int(edge_count[edge_indices].sum(dtype=np.int64)),
        "edge_indices_sha256": int64_array_sha256(edge_indices),
        "edge_group_pairs_sha256": int64_array_sha256(edge_group_pairs),
        "target_count": int(len(target_indices)),
        "target_indices_sha256": int64_array_sha256(target_indices),
        "target_body_ids_sha256": int64_array_sha256(target_body_ids),
        "target_subtype_pairs_sha256": int64_array_sha256(target_subtype_pairs),
    }
    expected = {
        "group_edge_counts": EXPECTED_GROUP_EDGE_COUNTS,
        "group_synapse_counts": EXPECTED_GROUP_SYNAPSE_COUNTS,
        "edge_count": EXPECTED_EDGE_COUNT,
        "synapse_count": EXPECTED_SYNAPSE_COUNT,
        "edge_indices_sha256": EXPECTED_EDGE_INDICES_SHA256,
        "edge_group_pairs_sha256": EXPECTED_EDGE_GROUP_PAIRS_SHA256,
        "target_count": EXPECTED_TARGET_COUNT,
        "target_indices_sha256": EXPECTED_TARGET_INDICES_SHA256,
        "target_body_ids_sha256": EXPECTED_TARGET_BODY_IDS_SHA256,
        "target_subtype_pairs_sha256": EXPECTED_TARGET_SUBTYPE_PAIRS_SHA256,
    }
    if observed != expected:
        wrong = [name for name, value in expected.items() if observed[name] != value]
        raise SystemExit(f"vertical-motion anatomy mask changed: {', '.join(wrong)}")
    return {
        "selection": (
            "existing graph edges whose exact annotated presynaptic type and exact "
            "postsynaptic subtype match one ordered group"
        ),
        "excluded": [
            "untyped or other external afferents",
            "intra-T4/T5 edges",
            "T4a/T4b/T5a/T5b",
            "all downstream edges and motor readouts",
        ],
        "groups": group_rows,
        **{
            name: list(value) if isinstance(value, tuple) else value
            for name, value in observed.items()
        },
    }


def _random_uint64(rng: np.random.Generator) -> int:
    return int(rng.bit_generator.random_raw())


def _rng_state_sha256(rng: np.random.Generator) -> str:
    return semantic_sha256(rng.bit_generator.state)


def _phase_records(
    rng: np.random.Generator, phase_bins: tuple[int, ...]
) -> list[dict[str, int | str]]:
    records = []
    for phase_bin in phase_bins:
        raw = _random_uint64(rng)
        # Keep the phase well inside its unique one-pixel bin. This makes the
        # row-crossing history stable and lets split disjointness be checked exactly.
        jitter_code = PHASE_DENOMINATOR // 8 + (raw >> 11) % (PHASE_DENOMINATOR // 4)
        records.append(
            {
                "phase_bin": phase_bin,
                "subpixel_jitter_code": jitter_code,
                "phase_row_numerator": phase_bin * PHASE_DENOMINATOR + jitter_code,
                "phase_row_denominator": PHASE_DENOMINATOR,
                "raw_draw_uint64_hex": f"0x{raw:016x}",
            }
        )
    return records


def _edge_row_crossing_signature(phase_row_numerator: int, speed: int) -> tuple[int, ...]:
    """Return exact hard-edge boundary rows without constructing image pixels."""

    denominator = 2 * PHASE_DENOMINATOR
    signature = []
    # The base center is 60 + phase_row_numerator / 2^53. Branch +1 moves
    # downward and uses row <= center; branch -1 moves upward and uses row >= center.
    for branch_sign in (1, -1):
        for frame in range(MOTION_FRAMES):
            numerator = (
                2 * (60 * PHASE_DENOMINATOR + phase_row_numerator)
                + branch_sign * speed * (2 * frame - (MOTION_FRAMES - 1)) * PHASE_DENOMINATOR
            )
            if branch_sign == 1:
                boundary = numerator // denominator
            else:
                boundary = -((-numerator) // denominator)
            signature.append(int(boundary))
    return tuple(signature)


def split_specs(name: str, config: dict[str, Any]) -> dict[str, Any]:
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise SystemExit(f"NumPy changed: {np.__version__!r} != {EXPECTED_NUMPY_VERSION!r}")
    rng = np.random.Generator(np.random.PCG64(config["seed"]))
    initial_rng_state_sha256 = _rng_state_sha256(rng)
    phase_records = _phase_records(rng, config["edge_phase_bins"])
    texture_seeds = [_random_uint64(rng) for _ in range(config["texture_realizations"])]
    final_rng_state_sha256 = _rng_state_sha256(rng)
    specs: list[dict[str, Any]] = []
    case = 0
    for polarity in ("ON", "OFF"):
        for speed in config["edge_speeds"]:
            for phase_index, phase in enumerate(phase_records):
                specs.append(
                    {
                        "case": case,
                        "family_id": f"{name}:edge-phase-{phase_index}",
                        "family": "polarity_preserving_edge",
                        "polarity": polarity,
                        "speed_pixels_per_frame": speed,
                        "phase_index": phase_index,
                        **phase,
                    }
                )
                case += 1
    for speed in config["texture_speeds"]:
        for realization, realization_seed in enumerate(texture_seeds):
            specs.append(
                {
                    "case": case,
                    "family_id": f"{name}:texture-realization-{realization}",
                    "family": "band_limited_texture",
                    "polarity": "mixed",
                    "speed_pixels_per_frame": speed,
                    "realization": realization,
                    "realization_seed_uint64_hex": f"0x{realization_seed:016x}",
                }
            )
            case += 1
    if len(specs) != config["expected_pairs"]:
        raise RuntimeError(f"{name} stimulus factorial is incomplete")
    return {
        "seed": config["seed"],
        "numpy_bit_generator": "PCG64",
        "numpy_version": EXPECTED_NUMPY_VERSION,
        "raw_draw_order": "edge phases in phase-bin order, then texture realizations",
        "raw_draw_count": len(phase_records) + len(texture_seeds),
        "initial_rng_state_semantic_sha256": initial_rng_state_sha256,
        "final_rng_state_semantic_sha256": final_rng_state_sha256,
        "pixel_access": config["pixel_access"],
        "opposite_direction_pairs": len(specs),
        "independent_family_count": len(phase_records) + len(texture_seeds),
        "edge_phase_records": phase_records,
        "edge_phase_identity_pairs_sha256": int64_array_sha256(
            np.asarray(
                [(row["phase_bin"], row["subpixel_jitter_code"]) for row in phase_records],
                dtype=np.int64,
            )
        ),
        "texture_realization_seeds_uint64_sha256": uint64_array_sha256(
            np.asarray(texture_seeds, dtype=np.uint64)
        ),
        "edge_row_crossing_signatures_sha256": int64_array_sha256(
            np.asarray(
                [
                    (
                        speed,
                        phase_index,
                        *_edge_row_crossing_signature(int(phase["phase_row_numerator"]), speed),
                    )
                    for speed in config["edge_speeds"]
                    for phase_index, phase in enumerate(phase_records)
                ],
                dtype=np.int64,
            )
        ),
        "specs": specs,
        "specs_semantic_sha256": semantic_sha256(specs),
    }


def stimulus_manifest(*, verify_hashes: bool) -> dict[str, Any]:
    splits = {name: split_specs(name, config) for name, config in SPLITS.items()}
    all_specs_hash = semantic_sha256({name: split["specs"] for name, split in splits.items()})
    phase_sets = {
        name: {
            (row["phase_bin"], row["subpixel_jitter_code"]) for row in split["edge_phase_records"]
        }
        for name, split in splits.items()
    }
    phase_disjoint = all(
        phase_sets[first].isdisjoint(phase_sets[second])
        for position, first in enumerate(phase_sets)
        for second in tuple(phase_sets)[position + 1 :]
    )
    texture_sets = {
        name: {
            spec["realization_seed_uint64_hex"]
            for spec in split["specs"]
            if spec["family"] == "band_limited_texture"
        }
        for name, split in splits.items()
    }
    texture_disjoint = all(
        texture_sets[first].isdisjoint(texture_sets[second])
        for position, first in enumerate(texture_sets)
        for second in tuple(texture_sets)[position + 1 :]
    )
    signature_sets = {
        name: {
            (
                speed,
                _edge_row_crossing_signature(int(phase["phase_row_numerator"]), speed),
            )
            for speed in SPLITS[name]["edge_speeds"]
            for phase in split["edge_phase_records"]
        }
        for name, split in splits.items()
    }
    signature_disjoint = all(
        signature_sets[first].isdisjoint(signature_sets[second])
        for position, first in enumerate(signature_sets)
        for second in tuple(signature_sets)[position + 1 :]
    )
    if not phase_disjoint or not texture_disjoint or not signature_disjoint:
        raise SystemExit("stimulus phase or texture identities overlap between splits")
    if verify_hashes:
        observed = {name: split["specs_semantic_sha256"] for name, split in splits.items()}
        if observed != EXPECTED_SPLIT_SPEC_SHA256:
            wrong = [
                name
                for name, digest in EXPECTED_SPLIT_SPEC_SHA256.items()
                if observed[name] != digest
            ]
            raise SystemExit(f"vertical-motion stimulus specs changed: {', '.join(wrong)}")
        if all_specs_hash != EXPECTED_ALL_SPECS_SHA256:
            raise SystemExit("combined vertical-motion stimulus specs changed")
    return {
        "image": {
            "width": WIDTH,
            "height": HEIGHT,
            "linear_pixels": True,
            "horizontal_fov_degrees": HFOV_DEGREES,
            "attitude": {"roll": 0.0, "pitch": 0.0},
        },
        "timing": {
            "camera_hz": CAMERA_HZ,
            "neutral_prefix_frames": PREFIX_FRAMES,
            "motion_frames": MOTION_FRAMES,
            "common_terminal_frames": TERMINAL_FRAMES,
            "cns_solver": "deterministic_exponential_euler_K32",
            "cns_substeps_per_camera_frame": CNS_SUBSTEPS_PER_FRAME,
            "neural_state_updates_hz": CAMERA_HZ * CNS_SUBSTEPS_PER_FRAME,
            "image_held_during_neural_substeps": True,
        },
        "pair_semantics": {
            "branch_order": ["down", "up"],
            "normal_motion_pixel_signs": [1, -1],
            "stationary_branch": "the corresponding branch's first moving frame",
            "literal_reverse": (
                "reverse the 16 moving frames and use the corresponding normal final "
                "moving frame as the stationary baseline"
            ),
            "terminal": "one shared uniform linear-gray frame at 0.5",
            "edge_center": ("60 + phase_row_numerator / 2^53 at the middle of the motion window"),
            "pixel_centers": "integer row and column coordinates starting at zero",
            "antialiasing": False,
            "hard_edge_threshold_equality": (
                "down branch selects row <= center; up branch selects row >= center"
            ),
            "edge_contrast": {"dark": 0.2, "bright": 0.8},
            "edge_orientation": (
                "orientation reverses with branch so every crossed pixel increases for "
                "ON and decreases for OFF"
            ),
            "texture_renderer_sealed_specification": {
                "version": "pcg64-rfft2-annulus-periodic-v1",
                "white_field": "PCG64(realization_seed).standard_normal(200x320,float64)",
                "transform": "numpy rfft2 and irfft2",
                "passband_cycles_per_pixel": [1.0 / 96.0, 1.0 / 8.0],
                "passband_boundary": "inclusive Euclidean radial frequency",
                "dc_removed": True,
                "normalization": "divide by maximum absolute spatial sample",
                "linear_range": [0.18, 0.82],
                "translation": "integer cyclic image-row shifts with no interpolation",
            },
        },
        "labels_available_only_to_loss": ["direction", "speed", "polarity", "family"],
        "rendered_pixel_hashes": None,
        "neural_response_hashes": None,
        "cross_split_edge_phases_disjoint": phase_disjoint,
        "cross_split_effective_edge_histories_disjoint": signature_disjoint,
        "cross_split_texture_seeds_disjoint": texture_disjoint,
        "split_specs_semantic_sha256": {
            name: split["specs_semantic_sha256"] for name, split in splits.items()
        },
        "all_specs_semantic_sha256": all_specs_hash,
        "splits": splits,
    }


def manifest(args: argparse.Namespace, *, verify_hashes: bool = True) -> dict[str, Any]:
    locked_inputs = validate_locked_inputs(args)
    anatomy = anatomy_manifest(args.graph, args.raw_dir / ANNOTATIONS_FILE)
    stimuli = stimulus_manifest(verify_hashes=verify_hashes)
    schema = {
        "version": 1,
        "top_level_fields": [
            "experiment",
            "protocol_commit",
            "purpose",
            "no_response_manifest",
            "schema",
            "locked_inputs",
            "anatomy",
            "parameters",
            "stimuli",
            "execution_authority",
        ],
        "canonical_semantic_encoding": (
            "UTF-8 JSON, sorted keys, compact separators, finite values only"
        ),
    }
    schema_digest = semantic_sha256(schema)
    if verify_hashes and schema_digest != EXPECTED_SCHEMA_SEMANTIC_SHA256:
        raise SystemExit("vertical-motion commissioning manifest schema changed")
    payload = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "purpose": "freeze topology and stimulus specifications before outcome-bearing work",
        "no_response_manifest": True,
        "schema": {**schema, "semantic_sha256": schema_digest},
        "locked_inputs": locked_inputs,
        "anatomy": anatomy,
        "parameters": {
            "total": 24,
            "edge_gain_multipliers": {
                "count": 16,
                "group_sharing": "source type x target subtype, bilateral and column-shared",
                "bounds": [0.25, 4.0],
                "source_identity": 1.0,
            },
            "target_bias_offsets": {
                "count": 4,
                "subtype_order": list(TARGET_SUBTYPES),
                "bounds": [-0.25, 0.25],
                "source_identity": 0.0,
            },
            "target_physical_tau_multipliers": {
                "count": 4,
                "subtype_order": list(TARGET_SUBTYPES),
                "ratio_bounds": [0.5, 2.0],
                "native_physical_tau_bounds_seconds": [0.010, 0.250],
                "source_identity": 1.0,
            },
            "all_other_parameters_frozen": True,
        },
        "stimuli": stimuli,
        "execution_authority": {
            "pixel_rendering_performed": False,
            "neural_response_evaluation_performed": False,
            "training_execution_authorized": False,
            "candidate_retained": False,
            "development_opened": False,
            "acceptance_opened": False,
            "hover_gate_or_promotion_authorized": False,
            "next_authorized_action": "implement and run the disposable one-update preflight",
        },
    }
    digest = semantic_sha256(payload)
    if verify_hashes and digest != EXPECTED_MANIFEST_SEMANTIC_SHA256:
        raise SystemExit("vertical-motion commissioning manifest semantics changed")
    return payload


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> None:
    args = parse_args()
    payload = manifest(args, verify_hashes=not args.compute_hashes)
    if args.compute_hashes:
        print(
            json.dumps(
                {
                    "split_specs_semantic_sha256": payload["stimuli"][
                        "split_specs_semantic_sha256"
                    ],
                    "all_specs_semantic_sha256": payload["stimuli"]["all_specs_semantic_sha256"],
                    "manifest_semantic_sha256": semantic_sha256(payload),
                    "schema_semantic_sha256": payload["schema"]["semantic_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"manifest semantic SHA-256: {semantic_sha256(payload)}")
    if args.write is not None:
        write_exclusive(args.write, payload)
        print(f"wrote {args.write}")


if __name__ == "__main__":
    main()
