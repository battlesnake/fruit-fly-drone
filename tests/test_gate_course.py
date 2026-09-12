from __future__ import annotations

import math

import pytest
import torch

from flydrone.gate import AnnularGate
from flydrone.gate_course import classify_course_step


def gate_at(x: float, batch: int = 1, yaw: float = 0.0) -> AnnularGate:
    return AnnularGate(
        center=torch.tensor([[x, 0.0, 1.0]]).expand(batch, -1),
        yaw=torch.full((batch,), yaw),
    )


def test_multiple_crossings_in_one_step_follow_physical_time_order() -> None:
    previous = torch.tensor([[0.0, 0.0, 1.0]])
    position = torch.tensor([[3.0, 0.0, 1.0]])
    index = torch.tensor([0])
    ordered = classify_course_step(previous, position, (gate_at(1), gate_at(2)), index)
    reversed_layout = classify_course_step(previous, position, (gate_at(2), gate_at(1)), index)

    assert ordered.passed.tolist() == [[True, True]]
    assert ordered.next_gate_index.tolist() == [2]
    assert ordered.expected_forward_crossing.all()
    assert ordered.crossing_lateral.eq(0).all()
    assert not ordered.failed.any()
    assert reversed_layout.passed.tolist() == [[True, False]]
    assert reversed_layout.wrong_order.tolist() == [[False, True]]
    assert reversed_layout.next_gate_index.tolist() == [1]
    assert index.tolist() == [0]  # caller owns progression


def test_reverse_crossing_uses_interpolated_aperture_and_ring_geometry() -> None:
    previous = torch.tensor([[3.0, 0.0, 1.0], [3.0, 0.0, 1.0], [3.0, 2.0, 1.0]])
    position = torch.tensor([[1.0, 0.8, 1.0], [1.0, 1.4, 1.0], [1.0, 2.0, 1.0]])
    events = classify_course_step(previous, position, (gate_at(2, 3),), torch.zeros(3).long())

    assert events.wrong_direction.tolist() == [[True], [False], [False]]
    assert events.ring_collision.tolist() == [[False], [True], [False]]
    assert events.failed.tolist() == [True, True, False]
    assert events.next_gate_index.tolist() == [0, 0, 0]


def test_wrong_order_and_direction_are_counted_once_per_illegal_traversal() -> None:
    events = classify_course_step(
        torch.tensor([[3.0, 0.0, 1.0]]),
        torch.tensor([[1.5, 0.0, 1.0]]),
        (gate_at(1), gate_at(2)),
        torch.tensor([0]),
    )
    assert events.wrong_direction.tolist() == [[False, True]]
    assert events.wrong_order.tolist() == [[False, True]]
    assert events.penalty().tolist() == [-1.0]
    with pytest.raises(ValueError, match="nonnegative"):
        events.penalty(traversal_cost=-1.0)


def test_dark_and_future_gate_rings_remain_physical_in_both_directions() -> None:
    events = classify_course_step(
        torch.tensor([[0.0, 0.7, 1.0], [3.0, 0.7, 1.0]]),
        torch.tensor([[3.0, 0.7, 1.0], [0.0, 0.7, 1.0]]),
        (gate_at(1, 2), gate_at(2, 2)),
        torch.tensor([1, 0]),
    )
    assert events.ring_collision.all()
    assert not events.illegal_traversal.any()
    assert events.penalty().tolist() == [-4.0, -4.0]


def test_exterior_plane_crossings_are_legal_for_circling_maneuvers() -> None:
    events = classify_course_step(
        torch.tensor([[0.0, 2.0, 1.0]]),
        torch.tensor([[3.0, 2.0, 1.0]]),
        (gate_at(1), gate_at(2)),
        torch.tensor([0]),
    )
    assert not events.failed.any()
    assert not events.passed.any()
    assert events.penalty().tolist() == [0.0]


def test_gate_normal_defines_direction_not_world_x_axis() -> None:
    gate = AnnularGate(center=torch.tensor([[0.0, 0.0, 1.0]]), yaw=torch.tensor([math.pi / 2]))
    events = classify_course_step(
        torch.tensor([[0.0, -1.0, 1.0]]),
        torch.tensor([[0.0, 1.0, 1.0]]),
        (gate,),
        torch.tensor([0]),
    )
    assert events.passed.tolist() == [[True]]


def test_gate_event_not_repeated_while_flying_away_from_plane() -> None:
    gate = gate_at(1)
    before = torch.tensor([[0.0, 0.0, 1.0]])
    on_plane = torch.tensor([[1.0, 0.0, 1.0]])
    after = torch.tensor([[2.0, 0.0, 1.0]])
    first = classify_course_step(before, on_plane, (gate,), torch.tensor([0]))
    second = classify_course_step(on_plane, after, (gate,), first.next_gate_index)
    assert first.passed.all()
    assert not second.passed.any()
    assert not second.failed.any()


def test_traversing_a_passed_aperture_is_wrong_order() -> None:
    events = classify_course_step(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[1.5, 0.0, 1.0]]),
        (gate_at(1), gate_at(2)),
        torch.tensor([1]),
    )
    assert events.wrong_order.tolist() == [[True, False]]
    assert events.next_gate_index.tolist() == [1]


def test_touch_plane_then_return_counts_backwards_traversal() -> None:
    gate = gate_at(1)
    before = torch.tensor([[0.0, 0.0, 1.0]])
    on_plane = torch.tensor([[1.0, 0.0, 1.0]])
    first = classify_course_step(before, on_plane, (gate,), torch.tensor([0]))
    second = classify_course_step(on_plane, before, (gate,), first.next_gate_index)
    assert first.passed.all()
    assert second.wrong_direction.all()
    assert second.penalty().tolist() == [-1.0]
