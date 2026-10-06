# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The forward-backward body is built here, then handed to arctic-platform."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("arctic_platform")
pytest.importorskip("httpx")
pytest.importorskip("tinker")


torch = pytest.importorskip("torch")

from cortex_training.integrations.tinker.payload import to_cortex_fwd_bwd_payload  # noqa: E402


def test_explicit_old_log_probs_ride_the_alignment():
    """Sampler log-probs move with the tokens they scored, and batch log-probs are dropped."""
    ids = torch.tensor([[1, 2, 3, 4], [0, 0, 7, 8]])
    attn = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    resp_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    old = torch.tensor([[0.0, 0.0, -0.1, -0.2], [0.0, 0.0, -0.7, -0.8]], dtype=torch.float64)
    out = to_cortex_fwd_bwd_payload(
        {
            "batch": {
                "input_ids": ids,
                "attention_mask": attn,
                "advantages": resp_mask.to(torch.float32),
                "response_mask": resp_mask,
                "old_log_probs": torch.full((2, 4), -9.0),
            },
            "meta": {},
        },
        old_log_probs_shifted=old,
    )
    sent = out["context"]["old_log_probs_shifted"]
    assert sent.dtype == torch.float32
    loss_mask = out["context"]["loss_mask"]
    assert sent[1][loss_mask[1]].tolist() == pytest.approx([-0.7, -0.8])
    assert sent[0][loss_mask[0]].tolist() == pytest.approx([-0.1, -0.2])
    assert "old_log_probs" not in out["kwargs"]
