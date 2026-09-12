"""Batch independent native-parameter candidates, without changing actor inputs."""

from __future__ import annotations

from dataclasses import replace

import torch
from search_gate_motor_interface_es import controller_step_with_interface

from flydrone.gate import AnnularGate
from flydrone.hover import QuadState, StickState


def repeat_course_bank(bank, policies):
    if policies < 1:
        raise ValueError("policies must be positive")
    cases, gates = bank
    repeated_gates = tuple(
        AnnularGate(gate.center.repeat(policies, 1), gate.yaw.repeat(policies)) for gate in gates
    )
    return replace(
        cases,
        state=QuadState(*(value.repeat(policies, 1) for value in cases.state.as_tuple())),
        sticks=StickState(
            cases.sticks.joint_position.repeat(policies, 1),
            cases.sticks.joint_velocity.repeat(policies, 1),
            cases.sticks.position.repeat(policies, 1),
            cases.sticks.velocity.repeat(policies, 1),
        ),
        gate=repeated_gates[0],
        mass_scale=cases.mass_scale.repeat(policies),
        side=cases.side.repeat(policies),
        pair=torch.cat([cases.pair + i * (len(cases.side) // 2) for i in range(policies)]),
    ), repeated_gates


class ParameterBatchController:
    """The same controller, with one fixed anatomical parameter vector per flight.

    Vectors are candidate weights, not observations or dynamically chosen actions.
    A selected vector is compiled into ordinary checkpoint weights for deployment.
    """

    def __init__(self, controller, vectors, spec, episodes):
        if controller.uses_accelerometer or controller.uses_proprioception:
            raise ValueError("this course batch supports only RGB plus roll/pitch actors")
        self.controller = controller
        self.interface = vectors.repeat_interleave(episodes, dim=0)
        self.spec = spec

    def initial_state(self, *args, **kwargs):
        return self.controller.initial_state(*args, **kwargs)

    def __call__(self, image, attitude, neural):
        if len(image) != len(self.interface):
            raise ValueError("candidate vectors and flight rows differ")
        return controller_step_with_interface(
            self.controller, image, attitude, neural, None, None, self.interface, self.spec
        )
