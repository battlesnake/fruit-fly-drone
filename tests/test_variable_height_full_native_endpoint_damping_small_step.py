from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_variable_height_full_native_endpoint_damping_small_step import (  # noqa: E402
    EXPECTED_CACHE,
    EXPECTED_SCALARS,
    SCALES,
    protocol_manifest,
    reproduction_report,
)


def _cache_integrity() -> dict:
    return {
        "manifest_sha256": EXPECTED_CACHE["manifest_sha256"],
        "training": {
            "factorial_tensor_sha256": EXPECTED_CACHE[
                "training_factorial_tensor_sha256"
            ],
            "attitude_tensor_sha256": EXPECTED_CACHE["training_attitude_tensor_sha256"],
        },
        "development": {
            "factorial_tensor_sha256": EXPECTED_CACHE[
                "development_factorial_tensor_sha256"
            ],
            "attitude_tensor_sha256": EXPECTED_CACHE[
                "development_attitude_tensor_sha256"
            ],
        },
    }


def _metrics(endpoint_nrmse: float) -> dict:
    return {"endpoint_damping_nrmse": endpoint_nrmse}


def test_reproduction_report_requires_exact_cache_and_tolerant_scalars() -> None:
    result = reproduction_report(
        cache_integrity=_cache_integrity(),
        endpoint_scale=EXPECTED_SCALARS["endpoint_damping_scale"],
        baseline=_metrics(EXPECTED_SCALARS["baseline_endpoint_damping_nrmse"]),
        full_candidate=_metrics(EXPECTED_SCALARS["full_scale_endpoint_damping_nrmse"]),
        derivative=EXPECTED_SCALARS["autograd_directional_derivative"],
        displacement_rms={
            "edge_magnitude": EXPECTED_SCALARS["edge_magnitude_rms"],
            "bias": EXPECTED_SCALARS["bias_rms"],
            "raw_time_constant": EXPECTED_SCALARS["raw_time_constant_rms"],
        },
    )

    assert result["pass"] is True

    bad_cache = _cache_integrity()
    bad_cache["development"]["attitude_tensor_sha256"] = "wrong"
    result = reproduction_report(
        cache_integrity=bad_cache,
        endpoint_scale=EXPECTED_SCALARS["endpoint_damping_scale"],
        baseline=_metrics(EXPECTED_SCALARS["baseline_endpoint_damping_nrmse"]),
        full_candidate=_metrics(EXPECTED_SCALARS["full_scale_endpoint_damping_nrmse"]),
        derivative=EXPECTED_SCALARS["autograd_directional_derivative"],
        displacement_rms={
            "edge_magnitude": EXPECTED_SCALARS["edge_magnitude_rms"],
            "bias": EXPECTED_SCALARS["bias_rms"],
            "raw_time_constant": EXPECTED_SCALARS["raw_time_constant_rms"],
        },
    )
    assert result["pass"] is False


def test_protocol_freezes_only_two_new_scales_and_one_development_trial() -> None:
    protocol = protocol_manifest()

    assert SCALES == (0.0625, 0.03125)
    assert protocol["training_scales_descending"] == [0.0625, 0.03125]
    assert protocol["source_cache_regeneration_allowed"] is False
    assert protocol["development_candidates_evaluated"] == 1
    assert protocol["no_smaller_scale_after_development_failure"] is True
    assert "original source" in protocol["pass_authorizes"]
