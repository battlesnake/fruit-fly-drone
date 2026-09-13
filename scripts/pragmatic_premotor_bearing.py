"""Training-only bearing supervision on existing direct premotor neurons.

The decoder, visibility mask and labels never enter deployed control. Changed
premotor cells are not sinks: every supervised window replays a full, fresh
current-weight sensory prefix, then differentiates only its short window.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch
import train_pragmatic_course_replay as replay
from pragmatic_policy_rollout import CourseRewardTracker, stick_values

from flydrone.hover import StickState, rotation_matrix

COMMAND_SIGMAS = (0.006, 0.002, 0.001, 0.0025)
PHASES = ("launch", "first-approach", "later-gates")


def premotor_mask(graph_path, annotations_path, device):
    """Official superclass selection; freeze ALL outgoing edges, including internal ones."""
    with np.load(graph_path) as graph:
        nodes, pre, post = (graph[k] for k in ("node_ids", "edge_pre", "edge_post"))
        motors = graph["output_pool_indices"][: graph["output_pool_offsets"][2]]
        parents = np.unique(pre[np.isin(post, motors)])
    annotations = feather.read_table(
        Path(annotations_path),
        columns=["bodyId", "superclass"],
        memory_map=True,
    ).to_pydict()
    classes = dict(zip(annotations["bodyId"], annotations["superclass"], strict=True))
    selected = np.array(
        [p for p in parents if classes[int(nodes[p])] in ("descending_neuron", "vnc_intrinsic")],
        dtype=np.int64,
    )
    incoming, outgoing = np.isin(post, selected), np.isin(pre, selected)
    mask = incoming & ~outgoing
    if not len(selected) or not mask.any():
        raise ValueError("no eligible annotated direct premotor inputs")
    manifest = dict(
        direct_roll_parents=len(parents),
        selected_neurons=len(selected),
        selected_superclasses=dict(Counter(classes[int(nodes[p])] for p in selected)),
        selected_body_ids=nodes[selected].tolist(),
        selected_node_indices=selected.tolist(),
        incoming_edges=int(incoming.sum()),
        frozen_internal_edges=int((incoming & outgoing).sum()),
        trainable_edges=int(mask.sum()),
        annotations=str(annotations_path),
        outgoing_edges_biases_and_time_constants_frozen=True,
        actor_inputs="RGB and roll/pitch only; labels/head/visibility training-only",
    )
    return torch.as_tensor(mask, device=device), torch.as_tensor(selected, device=device), manifest


def bearing_labels(state, gates, current):
    centers = replay.active_gate(gates, current).center
    body = torch.einsum("bji,bj->bi", rotation_matrix(state.euler), centers - state.position)
    angle = torch.atan2(body[:, 1], body[:, 0])
    return torch.stack((angle.sin(), angle.cos()), -1)


def visible_current_gate(image):
    """Training-only actual pixel evidence, including partial rings/occlusion.

    This threshold is specific to the renderer's fixed current-gate turquoise.
    Floor grey and next/later red/blue do not qualify; four pixels are required.
    """
    red, green, blue = image.unbind(1)
    return (((green - red) > 0.15) & ((green - blue) > 0.08)).flatten(1).sum(1) >= 4


@dataclass
class BearingBank:
    replay: replay.ReplayBank
    source_features: torch.Tensor  # tanh of selected cells AFTER the current frame
    labels: torch.Tensor
    visible: torch.Tensor  # already masked by task-active at command time
    metrics: dict


@torch.no_grad()
def collect_bearing_bank(
    controller,
    source_controller,
    nodes,
    *,
    pairs,
    seed,
    kind,
    camera,
    config,
    gate_config,
    seconds=30,
):
    if kind not in ("native", "roll-assisted"):
        raise ValueError("native or explicitly training-only roll-assisted collection required")
    device = controller.bias.device
    cases, gates = replay.sample_two_gate_cases(
        pairs,
        seed=seed,
        device=device,
        hover_config=config,
        **replay.GEOMETRY,
    )
    state, sticks = cases.state, cases.sticks
    current = torch.zeros(2 * pairs, dtype=torch.long, device=device)
    tracker = CourseRewardTracker(2 * pairs, len(gates), device)
    neural = controller.initial_state(2 * pairs, device=device, dtype=torch.float32)
    source_neural = source_controller.initial_state(2 * pairs, device=device, dtype=torch.float32)
    image = replay.render_annular_gates_rgb(
        state, gates, current_gate_index=current, camera=camera, gate_config=gate_config
    )
    for _ in range(10):
        _, neural = controller(image, state.euler[:, :2], neural)
        _, source_neural = source_controller(image, state.euler[:, :2], source_neural)
    quad, legs = (
        replay.DifferentiableQuad(config).to(device),
        replay.ForelegStickPlant(config).to(device),
    )
    histories = [[] for _ in state.as_tuple()]
    roles, actives, references, features, labels, visibility = ([] for _ in range(6))
    for _ in range(round(seconds * 50)):
        active = ~tracker.failed & (current < len(gates))
        if not bool(active.any()):
            break
        image = replay.render_annular_gates_rgb(
            state, gates, current_gate_index=current, camera=camera, gate_config=gate_config
        )
        reference, source_neural = source_controller(image, state.euler[:, :2], source_neural)
        native, neural = controller(image, state.euler[:, :2], neural)
        for history, value in zip(histories, state.as_tuple(), strict=True):
            history.append(value.cpu().clone())
        roles.append(current.cpu().clone())
        actives.append(active.cpu())
        consistency = reference.clone()
        # Preserve previous sink-PPO roll learning instead of teaching upstream
        # cells to cancel it. P/Y/T stay anchored to the original source.
        consistency[:, 0] = native[:, 0]
        references.append(consistency.cpu())  # No teacher motor targets.
        features.append(source_neural[:, nodes].tanh().cpu())
        labels.append(bearing_labels(state, gates, current).cpu())
        visibility.append((active & visible_current_gate(image)).cpu())
        if kind == "native":
            motor = native
        else:
            motor = reference.clone()
            motor[:, 0] = replay.current_gate_roll_motor(
                state,
                replay.active_gate(gates, current),
                config,
                active=active,
            )
        if not bool(
            motor.isfinite().all() & source_neural.isfinite().all() & neural.isfinite().all()
        ):
            raise FloatingPointError("nonfinite representation collection")
        for _ in range(2):
            live = ~tracker.failed & (current < len(gates))
            # This collection stops at first failure/completion, unlike the PPO
            # full-flight collector. Synthetic padding must not create events.
            tracker.absorbed |= ~live
            rc, proposed_sticks = legs(motor, sticks)
            proposed = quad(rc, state, cases.mass_scale)
            events = replay.classify_course_step(
                state.position, proposed.position, gates, current, gate_config
            )
            tracker.step(events, proposed, gate_config)
            current = torch.where(live, events.next_gate_index, current)
            # Training records stop at first failure/completion; finite old state
            # pads inactive rows. Outside-aperture plane crossings remain recoverable.
            keep = (live & ~tracker.failed)[:, None]
            state = replay.QuadState(
                *(
                    torch.where(keep, new, old)
                    for new, old in zip(proposed.as_tuple(), state.as_tuple(), strict=True)
                )
            )
            sticks = StickState(
                *(
                    torch.where(keep, new, old)
                    for new, old in zip(
                        stick_values(proposed_sticks), stick_values(sticks), strict=True
                    )
                )
            )
    if not roles:
        raise ValueError("empty representation bank")
    reference_tensor = torch.stack(references)
    bank = replay.ReplayBank(
        tuple(torch.stack(h) for h in histories),
        tuple(replay.AnnularGate(g.center.cpu(), g.yaw.cpu()) for g in gates),
        torch.stack(roles),
        torch.stack(actives),
        reference_tensor,
        kind,
        seed,
        reference_outputs=reference_tensor,
        roll_teacher="current-gate",
        roll_from_start=True,
    )
    result = BearingBank(
        bank,
        torch.stack(features),
        torch.stack(labels),
        torch.stack(visibility),
        dict(
            seed=seed,
            kind=kind,
            episodes=2 * pairs,
            training_only=True,
            recorded_frames=len(roles),
            active_frames_by_gate=[
                int(((bank.current == g) & bank.active).sum()) for g in range(len(gates))
            ],
            ground_contacts=int(tracker.ground.sum()),
            invalid_episodes=int(tracker.invalid.sum()),
            failures=int(tracker.failed.sum()),
            clean_prefix_completions=int(tracker.clean().sum()),
            full_flight_success_is_not_assessed=True,
        ),
    )
    for value in (*bank.states, result.source_features, result.labels, bank.target):
        if not bool(value.isfinite().all()):
            raise FloatingPointError("nonfinite representation archive")
    return result


def phase_masks(bank):
    time = torch.arange(len(bank.current))[:, None]
    launch = (time < 50) & (bank.current == 0)
    return (launch, (bank.current == 0) & ~launch, bank.current >= 1)


def strata(bank, *, heldout=False, unroll=1):
    """Last mirrored pair is episode-held-out; balance kind, side and flight phase."""
    episodes = bank.replay.current.shape[1]
    if episodes < 4 or unroll < 1:
        raise ValueError("need at least two mirrored pairs and positive unroll")
    rows = range(episodes - 2, episodes) if heldout else range(episodes - 2)
    result = {}
    for phase, phase_mask in zip(PHASES, phase_masks(bank.replay), strict=True):
        eligible = bank.replay.active & phase_mask
        if len(eligible) < unroll:
            valid = torch.zeros((0, episodes), dtype=torch.bool)
        else:
            valid = eligible.unfold(0, unroll, 1).all(-1) & bank.visible.unfold(0, unroll, 1).any(
                -1
            )
        for side in (0, 1):
            chosen_rows = [r for r in rows if r % 2 == side]
            points = torch.nonzero(valid[:, chosen_rows])
            if len(points):
                points[:, 1] = torch.tensor(chosen_rows)[points[:, 1]]
            result[f"{bank.replay.kind}/{phase}/{side}"] = points  # time, row
    return result


class FrozenBearingHead(torch.nn.Module):
    """A frozen training loss, never a control output or deployed module."""

    def __init__(self, center, scale, weight, offset, source_mse):
        super().__init__()
        for name, value in dict(
            center=center, scale=scale, weight=weight, offset=offset, source_mse=source_mse
        ).items():
            self.register_buffer(name, value.detach())

    def forward(self, activity):
        return ((activity - self.center) / self.scale) @ self.weight + self.offset


def balanced_examples(banks, *, heldout=False, seed=1, limit=512):
    rng = torch.Generator().manual_seed(seed)
    x, y, weights, coverage = [], [], [], {}
    for bank in banks:
        for key, points in strata(bank, heldout=heldout).items():
            coverage[key] = len(points)
            if not len(points):
                continue
            chosen = points[torch.randperm(len(points), generator=rng)[:limit]]
            t, r = chosen.unbind(1)
            x.append(bank.source_features[t, r])
            y.append(bank.labels[t, r])
            weights.append(torch.full((len(chosen),), 1 / len(chosen)))
    if not x:
        raise ValueError("no visible active bearing examples")
    weight = torch.cat(weights)
    return torch.cat(x), torch.cat(y), weight / weight.sum(), coverage


@torch.no_grad()
def fit_bearing_head(banks, *, device, seed):
    x, y, weights, coverage = balanced_examples(banks, seed=seed)
    x, y, weights = x.to(device), y.to(device), weights.to(device)
    center = (weights[:, None] * x).sum(0)
    scale = (weights[:, None] * (x - center).square()).sum(0).sqrt().clamp_min(0.01)
    z = (x - center) / scale
    offset = (weights[:, None] * y).sum(0)
    ridge = 0.01 * torch.eye(x.shape[1], device=device)
    weight = torch.linalg.solve(
        z.T @ (weights[:, None] * z) + ridge, z.T @ (weights[:, None] * (y - offset))
    )
    mse = (weights[:, None] * (z @ weight + offset - y).square()).sum(0)
    head = FrozenBearingHead(center, scale, weight, offset, mse.mean().clamp_min(1e-4))
    hx, hy, hw, heldout = balanced_examples(banks, heldout=True, seed=seed)
    hmse = (hw.to(device)[:, None] * (head(hx.to(device)) - hy.to(device)).square()).sum(0)
    return head, dict(
        training_frames=len(x),
        training_coverage=coverage,
        heldout_coverage=heldout,
        source_train_mse=mse.tolist(),
        source_heldout_mse=hmse.tolist(),
        normalization=head.source_mse.tolist(),
        ridge=0.01,
        activity_std_floor=0.01,
        source_mse_floor=1e-4,
        source_features="post-tick tanh activity",
        decoder_is_deployed=False,
    )


def choose_windows(banks, rng, update, *, unroll=20, heldout=False):
    """Cycle available kind/phase combinations; sample fresh windows for BOTH sides."""
    candidates, coverage = [], {}
    for bank in banks:
        groups = strata(bank, heldout=heldout, unroll=unroll)
        coverage.update({k: len(v) for k, v in groups.items()})
        for phase in PHASES:
            sides = [groups[f"{bank.replay.kind}/{phase}/{s}"] for s in (0, 1)]
            if all(len(side) for side in sides):
                candidates.append((bank, phase, sides))
    if not candidates:
        raise ValueError("no paired visible active representation windows")
    bank, phase, sides = candidates[update % len(candidates)]
    chosen = [
        points[int(torch.randint(len(points), (), generator=rng))].tolist() for points in sides
    ]
    starts, rows = zip(*chosen, strict=True)
    return (
        bank,
        list(rows),
        list(starts),
        dict(
            kind=bank.replay.kind,
            phase=phase,
            starts=list(starts),
            rows=list(rows),
            coverage=coverage,
        ),
    )


def bearing_window_loss(
    controller, head, nodes, bank, rows, starts, *, camera, gate_config, unroll=20
):
    device = controller.bias.device
    window = replay.prepare_window(bank.replay, rows, starts, unroll, device)
    neural = replay.replay_prefix_state(controller, window, camera, gate_config).detach()
    indices = torch.arange(len(rows), device=device)
    predicted, commands = [], []
    for frame in range(unroll):
        times = window[4] + frame
        motor, neural = controller(
            *replay.window_observations(window, times, camera, gate_config), neural
        )
        predicted.append(head(neural[:, nodes].tanh()))
        commands.append(torch.atanh(motor.clamp(-0.999999, 0.999999)))
    times_cpu = torch.tensor(starts)[None, :] + torch.arange(unroll)[:, None]
    labels = bank.labels[times_cpu, rows].to(device)
    visible = bank.visible[times_cpu, rows].to(device)
    if not bool(visible.any(0).all()):
        raise ValueError("each supervised branch needs visible active bearing labels")
    error = (torch.stack(predicted) - labels).square() / head.source_mse
    # Equal branch/side weight despite different visible-frame counts.
    bearing = ((error * visible[..., None]).sum((0, 2)) / (2 * visible.sum(0))).mean()
    times = window[4][None, :] + torch.arange(unroll, device=device)[:, None]
    reference = torch.atanh(window[3][times, indices].clamp(-0.999999, 0.999999))
    command_error = (torch.stack(commands) - reference) / reference.new_tensor(COMMAND_SIGMAS)
    axes = command_error.square().mean((0, 1))
    loss = 0.1 * bearing + axes.mean()
    if not bool(loss.isfinite()):
        raise FloatingPointError("nonfinite representation loss")
    return loss, dict(
        loss=float(loss.detach()),
        normalized_bearing_mse=float(bearing.detach()),
        command_mse_in_sigma=axes.detach().tolist(),
        visible_frames=int(visible.sum()),
        supervised_frames=unroll,
        prefix_is_current_weight=True,
        gradient_is_window_truncated=True,
    )


def representation_update(controller, optimizer, mask, compute_loss):
    """Only mask-selected magnitudes can change; Adam has no weight decay."""
    optimizer.zero_grad(set_to_none=True)
    loss, stats = compute_loss()
    loss.backward()
    gradient = controller.edge_magnitude.grad
    gradient.mul_(mask)
    if not bool(gradient.isfinite().all()):
        raise FloatingPointError("nonfinite premotor gradient")
    norm = torch.nn.utils.clip_grad_norm_([controller.edge_magnitude], 1.0)
    with torch.no_grad():
        before = controller.edge_magnitude.clone()
    optimizer.step()
    with torch.no_grad():
        controller.edge_magnitude.copy_(
            torch.where(
                mask,
                controller.edge_magnitude.clamp(0, 8),
                before,
            )
        )
        delta = controller.edge_magnitude - before
    stats.update(
        gradient_norm=float(norm),
        edge_delta_l2=float(delta.norm()),
        changed_edges=int((delta != 0).sum()),
    )
    optimizer.zero_grad(set_to_none=True)
    return stats
