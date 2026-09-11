#!/usr/bin/env python3
"""Run the preregistered source-restarted beta1=0 assisted-throttle curriculum."""

from __future__ import annotations

import train_variable_height_native_throttle_assisted as train

EXPERIMENT = "variable-height-native-throttle-assisted-beta1-zero-v1"
PROTOCOL_COMMIT = "ddd3c5c"
OPTIMIZER_BETAS = (0.0, 0.999)


def configure_protocol() -> None:
    train.EXPERIMENT = EXPERIMENT
    train.PROTOCOL_COMMIT = PROTOCOL_COMMIT
    train.OPTIMIZER_BETAS = OPTIMIZER_BETAS


def main() -> int:
    configure_protocol()
    return train.main()


if __name__ == "__main__":
    raise SystemExit(main())
