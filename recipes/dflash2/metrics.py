# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ratio metrics from the additive terms the ``dflash2`` loss returns."""

from __future__ import annotations

from collections.abc import Mapping

TERM_PREFIX = "loss_term_"
NUM, DEN = "__num", "__den"

SELECTOR_ACCEPTED_LENGTH = "dflash2/selector/greedy_accepted_length"
UNARY_ACCEPTED_LENGTH = "dflash/hard_label/unary_greedy_accepted_length"


def summarize(metrics: Mapping[str, float]) -> dict[str, float]:
    """Pooled ratio metrics from a forward-backward result's ``metrics`` mapping."""
    out = {}
    for key, numerator in metrics.items():
        if not (key.startswith(TERM_PREFIX) and key.endswith(NUM)):
            continue
        name = key[len(TERM_PREFIX) : -len(NUM)]
        denominator = metrics.get(f"{TERM_PREFIX}{name}{DEN}")
        if denominator is not None:
            out[name] = float(numerator) / float(denominator) if float(denominator) > 0 else 0.0
    return out
