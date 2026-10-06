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
"""Build the forward-backward body the Arctic Platform client sends as-is.

``arctic_platform`` forwards the caller's batch verbatim. Cortex scores a
left-aligned ``{args, kwargs, context, processing}`` body, so this module is
the translation from Tinker's token rows into that body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import torch

_DEFAULT_PROC_CONFIG: dict[str, Any] = {
    "eps_clip": 0.2,
    "loss_agg_mode": "token-mean",
    "entropy_coeff": 0.0,
}

# `labels` uses the ignore index so a padded column contributes no loss.
_PAD_VALUE: dict[str, Any] = {"labels": -100}


def _left_align_batch(tensors: dict, attention_mask, extra: dict) -> tuple[dict, dict, Any]:
    """Move every row's real tokens into its leading columns, padding at the tail.

    Cortex's packer rejects a left-padded row. A stable sort on validity gives
    each row's real columns in order, then its pads. Every full-width tensor is
    gathered through that one index so advantages and the loss mask stay on the
    tokens they scored.
    """
    import torch

    mask = attention_mask.to(torch.bool)
    width = mask.shape[-1]
    lengths = mask.sum(dim=1)
    valid = torch.arange(width, device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)

    if torch.equal(mask, valid):
        return tensors, extra, attention_mask

    order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)

    def move(name: str, t):
        if not torch.is_tensor(t) or t.dim() != 2 or t.shape[-1] != width:
            return t
        pad = _PAD_VALUE.get(name, False if t.dtype == torch.bool else 0)
        return torch.where(valid, t.gather(1, order), torch.full_like(t, pad))

    return (
        {k: move(k, v) for k, v in tensors.items()},
        {k: move(k, v) for k, v in extra.items()},
        valid.to(attention_mask.dtype),
    )


def to_cortex_fwd_bwd_payload(
    batch: dict,
    *,
    processing: dict | None = None,
    old_log_probs_shifted: torch.Tensor | None = None,
) -> dict:
    """Reshape token rows into the body ``AsyncArcticRLClient.fwd_bwd`` forwards.

    Log-probs already in the batch are dropped. Cortex then treats the live
    forward as π_old, which is right for a single on-policy step. A ratio
    against another policy, such as the sampler, passes
    ``old_log_probs_shifted``: ``[B, S]`` in the frame Cortex scores, entry
    ``i`` the log-prob of token ``i + 1``.

    Requires ``loss_mask`` or ``response_mask``. Falling back to
    ``attention_mask`` would train on prompt tokens.
    """
    import torch

    payload = dict(batch)
    processing_in = processing or payload.pop("processing", None)
    payload.pop("router_replay", None)

    if "batch" in payload and isinstance(payload["batch"], dict):
        tensors, meta = dict(payload["batch"]), dict(payload.get("meta") or {})
    else:
        tensors = dict(payload)
        meta = dict(tensors.pop("context", None) or {})

    input_ids = tensors.get("input_ids")
    attention_mask = tensors.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise ValueError("cortex fwd_bwd requires 'input_ids' and 'attention_mask'")

    loss_mask = tensors.pop("loss_mask", None)
    if loss_mask is None:
        loss_mask = tensors.pop("response_mask", None)
    if loss_mask is None:
        raise ValueError(
            "cortex fwd_bwd requires either 'loss_mask' or 'response_mask' in the "
            "batch. Falling back to 'attention_mask' would include prompt tokens "
            "in the loss and silently corrupt the gradient."
        )
    if torch.is_tensor(loss_mask):
        loss_mask = loss_mask.to(torch.bool)
    advantages = tensors.pop("advantages", None)
    if advantages is None:
        raise ValueError("cortex fwd_bwd requires 'advantages' [B, S]")
    tensors.pop("old_log_probs", None)

    forwarded = {"input_ids": input_ids}
    for k in ("position_ids", "labels"):
        if k in tensors:
            forwarded[k] = tensors[k]
    scored = {"advantages": advantages, "loss_mask": loss_mask}
    if old_log_probs_shifted is not None:
        scored["old_log_probs_shifted"] = old_log_probs_shifted.to(torch.float32)
    forwarded, scored, attention_mask = _left_align_batch(forwarded, attention_mask, scored)
    input_ids = forwarded["input_ids"]
    context: dict[str, Any] = {"input_ids": input_ids, **scored}

    kwargs_out: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    for k in ("position_ids", "labels"):
        if k in forwarded:
            kwargs_out[k] = forwarded[k]

    caller_config = dict((processing_in or {}).get("config") or {})
    proc_config: dict[str, Any] = {**_DEFAULT_PROC_CONFIG, **caller_config}
    for k in ("global_batch_size", "batch_num_tokens"):
        if k not in proc_config and k in meta:
            proc_config[k] = int(meta[k])

    return {
        "args": (),
        "kwargs": kwargs_out,
        "context": context,
        "processing": {"post": ["compute_logprobs"], "loss_fn": "grpo", "config": proc_config},
    }
