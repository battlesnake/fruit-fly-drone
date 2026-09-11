from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import preregister_vertical_motion_commissioning as manifest  # noqa: E402


def _args() -> Namespace:
    return Namespace(
        graph=manifest.REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
        checkpoint=manifest.REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
        raw_dir=manifest.REPO_ROOT / "data/raw/malecns-v1.0",
        frozen_audit_report=manifest.REPO_ROOT
        / "runs/optic-motion/frozen-t4t5-audit-002/report.json",
    )


def test_real_anatomy_regenerates_exact_preregistered_mask() -> None:
    payload = manifest.manifest(_args())
    anatomy = payload["anatomy"]

    assert anatomy["edge_count"] == 26_781
    assert anatomy["synapse_count"] == 422_224
    assert anatomy["edge_indices_sha256"] == manifest.EXPECTED_EDGE_INDICES_SHA256
    assert anatomy["edge_group_pairs_sha256"] == (manifest.EXPECTED_EDGE_GROUP_PAIRS_SHA256)
    assert anatomy["target_count"] == 6_827
    assert [row["fixed_transmitter_sign"] for row in anatomy["groups"]] == [
        1,
        -1,
        -1,
        1,
        1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]


def test_stimulus_factorials_are_sealed_and_disjoint() -> None:
    stimuli = manifest.stimulus_manifest(verify_hashes=True)

    assert {
        name: split["opposite_direction_pairs"] for name, split in stimuli["splits"].items()
    } == {"training": 96, "development": 48, "acceptance": 128}
    assert stimuli["cross_split_edge_phases_disjoint"] is True
    assert stimuli["cross_split_effective_edge_histories_disjoint"] is True
    assert stimuli["cross_split_texture_seeds_disjoint"] is True
    assert stimuli["rendered_pixel_hashes"] is None
    assert stimuli["neural_response_hashes"] is None
    assert stimuli["splits"]["acceptance"]["pixel_access"] == (
        "sealed_until_a_development_snapshot_passes"
    )
    acceptance = stimuli["splits"]["acceptance"]["specs"]
    assert {
        item["speed_pixels_per_frame"]
        for item in acceptance
        if item["family"] == "band_limited_texture"
    } == {1, 2, 3, 4}
    assert not any(
        item["speed_pixels_per_frame"] == 3
        for split_name in ("training", "development")
        for item in stimuli["splits"][split_name]["specs"]
    )
    assert all(
        isinstance(item["realization_seed_uint64_hex"], str)
        for split in stimuli["splits"].values()
        for item in split["specs"]
        if item["family"] == "band_limited_texture"
    )


def test_manifest_grants_only_preflight_implementation() -> None:
    payload = manifest.manifest(_args())

    assert payload["no_response_manifest"] is True
    assert payload["parameters"]["total"] == 24
    assert payload["execution_authority"] == {
        "pixel_rendering_performed": False,
        "neural_response_evaluation_performed": False,
        "training_execution_authorized": False,
        "candidate_retained": False,
        "development_opened": False,
        "acceptance_opened": False,
        "hover_gate_or_promotion_authorized": False,
        "next_authorized_action": "implement and run the disposable one-update preflight",
    }
