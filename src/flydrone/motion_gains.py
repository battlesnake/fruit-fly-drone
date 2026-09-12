"""Training-only type-shared gains, compiled into existing visual synapses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch
from torch import nn
from torch.nn.utils import parametrize

MOTION_GROUPS = tuple(
    (source, f"{pathway}{subtype}")
    for pathway, sources in (
        ("T4", ("Mi1", "Mi4", "Mi9", "Tm3")),
        ("T5", ("Tm1", "Tm2", "Tm4", "Tm9")),
    )
    for subtype in "abcd"
    for source in sources
)


def motion_edge_groups(node_types, edge_pre, edge_post, groups=MOTION_GROUPS):
    """Include every existing edge of each type pair, with no eye/hop filtering."""
    types, codes = np.unique(np.asarray(node_types, dtype=str), return_inverse=True)
    lookup = {name: index for index, name in enumerate(types)}
    pre, post = codes[edge_pre], codes[edge_post]
    assignments = np.full(len(edge_pre), -1, dtype=np.int64)
    counts = []
    for index, (source, target) in enumerate(groups):
        selected = (pre == lookup.get(source, -2)) & (post == lookup.get(target, -2))
        if not selected.any():
            raise ValueError(f"no existing edges for {source} -> {target}")
        if np.any(assignments[selected] >= 0):
            raise ValueError("motion connection groups overlap")
        assignments[selected] = index
        counts.append(int(selected.sum()))
    indices = np.flatnonzero(assignments >= 0)
    return indices, assignments[indices], counts


def load_motion_edge_groups(graph_path: Path, annotations_path: Path, device):
    annotations = feather.read_table(
        annotations_path, columns=("bodyId", "type"), memory_map=True
    ).to_pydict()
    types_by_id = dict(zip(annotations["bodyId"], annotations["type"], strict=True))
    with np.load(graph_path) as graph:
        node_types = [types_by_id[int(body)] or "" for body in graph["node_ids"]]
        indices, groups, counts = motion_edge_groups(
            node_types, graph["edge_pre"], graph["edge_post"]
        )
        if np.isin(graph["edge_post"][indices], graph["output_pool_indices"]).any():
            raise ValueError("motion groups unexpectedly enter motor pools")
    manifest = dict(
        trainable_shared_gains=len(MOTION_GROUPS),
        selected_edges=len(indices),
        groups=[
            dict(source=source, target=target, edges=count)
            for (source, target), count in zip(MOTION_GROUPS, counts, strict=True)
        ],
        all_existing_edges_per_type_pair=True,
        eye_or_hop_filter=False,
        gain_bounds=[0.5, 2.0],
        signs_topology_biases_taus_and_motor_readout_frozen=True,
    )
    return (
        torch.tensor(indices, device=device),
        torch.tensor(groups, device=device),
        manifest,
    )


class SharedMotionGains(nn.Module):
    """Multiply selected magnitudes; the original full edge vector stays frozen."""

    def __init__(self, edge_indices, edge_groups):
        super().__init__()
        if (
            edge_indices.ndim != 1
            or edge_groups.shape != edge_indices.shape
            or not len(edge_indices)
            or len(torch.unique(edge_indices)) != len(edge_indices)
            or bool((edge_indices < 0).any())
        ):
            raise ValueError("selected edges must be a nonempty unique index vector")
        group_count = int(edge_groups.max()) + 1
        if not torch.equal(
            torch.unique(edge_groups), torch.arange(group_count, device=edge_groups.device)
        ):
            raise ValueError("group indices must be contiguous and start at zero")
        self.register_buffer("edge_indices", edge_indices.clone())
        self.register_buffer("edge_groups", edge_groups.clone())
        self.gains = nn.Parameter(torch.ones(group_count, device=edge_indices.device))

    def forward(self, original):
        selected = original[self.edge_indices] * self.gains[self.edge_groups]
        return original.index_copy(0, self.edge_indices, selected)

    @torch.no_grad()
    def project_parameters(self):
        if not bool(torch.isfinite(self.gains).all()):
            raise RuntimeError("nonfinite motion gains")
        self.gains.clamp_(0.5, 2.0)


def attach_motion_gains(controller, edge_indices, edge_groups):
    """The optimizer must receive only the returned module's gains."""
    controller.requires_grad_(False)
    if bool((controller.edge_magnitude[edge_indices] < 0).any()) or bool(
        (controller.edge_magnitude[edge_indices] * 2 > 8).any()
    ):
        raise ValueError("source motion magnitudes cannot honor the [0,8] edge bound")
    module = SharedMotionGains(edge_indices, edge_groups)
    parametrize.register_parametrization(controller, "edge_magnitude", module)
    return module


@torch.no_grad()
def compiled_controller_state(controller):
    """Ordinary controller state: no parametrization, gain module or extra decoder."""
    state = {
        name: value.detach().cpu().clone()
        for name, value in controller.state_dict().items()
        if not name.startswith("parametrizations.edge_magnitude.")
    }
    state["edge_magnitude"] = controller.edge_magnitude.detach().cpu().clone()
    return state
