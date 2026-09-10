"""Shared matched cases for the harder diverse annular-gate protocol."""

from __future__ import annotations

import torch
from search_gate_acceleration_path_es import sample_matched_cases
from search_gate_motor_interface_es import BalancedCases

from flydrone.gate import AnnularGate, GateConfig
from flydrone.hover import HoverConfig


def diverse_matched_cases(
    episodes: int,
    *,
    seed: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    extreme_fraction: float,
) -> BalancedCases:
    """Make adjacent mass pairs with distinct, broad gate geometry."""

    cases = sample_matched_cases(
        episodes,
        seed=seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    pairs = episodes // 2
    generator = torch.Generator(device="cpu").manual_seed(seed + 31)
    distance = 4.2 + torch.rand(pairs, generator=generator)
    lateral_magnitude = 0.35 + 0.90 * torch.rand(pairs, generator=generator)
    height = 0.95 + 0.30 * torch.rand(pairs, generator=generator)
    obliquity_magnitude = torch.deg2rad(10.0 + 20.0 * torch.rand(pairs, generator=generator))
    geometry = cases.stratum_code[0::2].cpu()
    lateral_sign = torch.where(geometry.bitwise_and(2).bool(), 1.0, -1.0)
    obliquity_sign = torch.where(geometry.bitwise_and(4).bool(), 1.0, -1.0)
    pair_center = torch.stack((distance, lateral_sign * lateral_magnitude, height), dim=1).to(
        device
    )
    bearing = torch.atan2(pair_center[:, 1], pair_center[:, 0])
    pair_yaw = bearing + (obliquity_sign * obliquity_magnitude).to(device)
    gate = AnnularGate(
        center=pair_center.repeat_interleave(2, dim=0),
        yaw=pair_yaw.repeat_interleave(2),
    )
    mass_delta = 0.005 + 0.075 * torch.rand(pairs, generator=generator)
    extreme_pairs = round(extreme_fraction * pairs)
    mass_delta[:extreme_pairs] = 0.08
    mass_delta = mass_delta.to(device)
    mass = torch.stack((1.0 - mass_delta, 1.0 + mass_delta), dim=1).flatten()
    if torch.unique(pair_center, dim=0).shape[0] != pairs:
        raise RuntimeError("diverse cases require distinct geometry for every pair")
    return BalancedCases(cases.state, gate, mass, cases.stratum_code)
