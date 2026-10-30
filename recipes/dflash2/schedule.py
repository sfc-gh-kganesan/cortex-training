# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Client-driven schedules for DFlash2 training."""

from __future__ import annotations

import math


def learning_rate(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    warmup_ratio: float = 0.04,
    scheduler: str = "cosine",
    min_lr: float = 0.0,
) -> float:
    """Learning rate used for the optimizer step with 0-based index ``step``."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = int(warmup_ratio * total_steps)
    if step < warmup_steps:
        return (step + 1) / warmup_steps * base_lr
    if scheduler == "constant":
        return base_lr
    if scheduler != "cosine":
        raise ValueError(f"unknown scheduler {scheduler!r}; expected 'cosine' or 'constant'")
    horizon = max(total_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0) / horizon, 1.0)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def selector_loss_alpha(
    step: int,
    *,
    target: float,
    total_steps: int,
    warmup_ratio: float = 0.0,
    ramp_ratio: float = 0.0,
) -> float:
    """Weight of the selector cross-entropy at optimizer step ``step``.

    Zero during warmup, then a linear ramp to ``target``. A configured target of exactly zero stays zero.
    """
    if target <= 0 or total_steps <= 0:
        return max(target, 0.0)
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        return 0.0
    ramp_steps = int(total_steps * ramp_ratio)
    if ramp_steps <= 0:
        return target
    progress = min(max((step - warmup_steps + 1) / ramp_steps, 0.0), 1.0)
    return target * progress
