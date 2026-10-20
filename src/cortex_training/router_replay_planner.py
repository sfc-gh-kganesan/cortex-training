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

"""GPU-free capacity planning for router-replay caches."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from numbers import Integral, Rational, Real
from typing import Any

GIB_BYTES = 1024**3
ROUTER_REPLAY_DTYPE_BYTES = 1
ROUTER_REPLAY_WIDE_DTYPE_BYTES = 2
MAX_ROUTER_REPLAY_EXPERTS = 1 << (8 * ROUTER_REPLAY_WIDE_DTYPE_BYTES)

DEFAULT_ROUTING_SKEW = 2.0
DEFAULT_SAFETY_FACTOR = 1.2

DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.9
DEFAULT_MAX_RESERVED_UTILIZATION_FRACTION = 0.5
MIN_VLLM_GPU_MEMORY_UTILIZATION = 0.05

_EXPERT_COUNT_FIELDS = (
    "num_local_experts",
    "n_routed_experts",
    "num_experts",
    "moe_num_experts",
)

__all__ = [
    "GIB_BYTES",
    "ROUTER_REPLAY_DTYPE_BYTES",
    "ROUTER_REPLAY_WIDE_DTYPE_BYTES",
    "DEFAULT_ROUTING_SKEW",
    "DEFAULT_SAFETY_FACTOR",
    "DEFAULT_VLLM_GPU_MEMORY_UTILIZATION",
    "DEFAULT_MAX_RESERVED_UTILIZATION_FRACTION",
    "MIN_VLLM_GPU_MEMORY_UTILIZATION",
    "MoEGeometry",
    "CachePlan",
    "VllmUtilizationPlan",
    "gib_to_bytes",
    "moe_geometry_from_config",
    "resolve_model_moe_geometry",
    "derive_backlog_rollouts",
    "derive_resident_rows",
    "derive_dp_size",
    "derive_num_sampling_workers",
    "plan_sampling_cache_bytes",
    "plan_training_cache_bytes",
    "plan_vllm_gpu_memory_utilization",
]


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return result


def _finite_fraction(
    name: str,
    value: Any,
    *,
    exact_float: bool = False,
) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, got {value!r}")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{name} must be finite, got {value!r}")
        return Fraction(value)
    if isinstance(value, Rational):
        return Fraction(value)
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric, got {value!r}")
    result_float = float(value)
    if not math.isfinite(result_float):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if exact_float:
        return Fraction.from_float(result_float)
    return Fraction(str(result_float))


def _factor(name: str, value: Any) -> Fraction:
    result = _finite_fraction(name, value)
    if result < 1.0:
        raise ValueError(f"{name} must be at least 1.0, got {value!r}")
    return result


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _factor_breakdown_value(value: Fraction) -> int | float:
    return value.numerator if value.denominator == 1 else float(value)


def _float_ratio_or_none(numerator: int, denominator: int) -> float | None:
    try:
        return numerator / denominator
    except OverflowError:
        return None


def _format_fraction(value: Fraction) -> str:
    try:
        return f"{float(value):.12g}"
    except OverflowError:
        return f"{value.numerator}/{value.denominator}"


def _previous_float(value: float) -> float:
    """Return the adjacent IEEE-754 double below a positive finite value."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"value must be positive and finite, got {value!r}")
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return struct.unpack(">d", struct.pack(">Q", bits - 1))[0]


def gib_to_bytes(gib: float | int | Fraction | Decimal) -> int:
    """Convert a positive, potentially fractional GiB quantity to bytes."""
    quantity = _finite_fraction("gib", gib)
    if quantity <= 0:
        raise ValueError(f"gib must be positive, got {gib!r}")
    return _ceil_fraction(quantity * GIB_BYTES)


@dataclass(frozen=True)
class MoEGeometry:
    """Per-layer routing geometry read from a model config."""

    num_moe_layers: int
    topk: int
    dtype_bytes: int = ROUTER_REPLAY_DTYPE_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "num_moe_layers",
            _positive_int("num_moe_layers", self.num_moe_layers),
        )
        topk = _positive_int("topk", self.topk)
        if topk > MAX_ROUTER_REPLAY_EXPERTS:
            raise ValueError(
                f"topk ({topk}) must not exceed {MAX_ROUTER_REPLAY_EXPERTS}; "
                "router-replay expert IDs are stored as uint8 or uint16"
            )
        object.__setattr__(self, "topk", topk)
        dtype_bytes = _positive_int("dtype_bytes", self.dtype_bytes)
        if dtype_bytes not in {
            ROUTER_REPLAY_DTYPE_BYTES,
            ROUTER_REPLAY_WIDE_DTYPE_BYTES,
        }:
            raise ValueError(
                "dtype_bytes must match a supported router-replay capture width "
                f"({ROUTER_REPLAY_DTYPE_BYTES} or {ROUTER_REPLAY_WIDE_DTYPE_BYTES}), "
                f"got {dtype_bytes}"
            )
        object.__setattr__(self, "dtype_bytes", dtype_bytes)

    def entry_bytes(
        self,
        seq_len: int,
        *,
        dtype_bytes: int | None = None,
    ) -> int:
        """Return bytes in one full-sequence router-replay capture."""
        sequence_length = _positive_int("seq_len", seq_len)
        element_bytes = (
            self.dtype_bytes
            if dtype_bytes is None
            else _positive_int("dtype_bytes", dtype_bytes)
        )
        return sequence_length * self.num_moe_layers * self.topk * element_bytes


@dataclass(frozen=True)
class CachePlan:
    """A byte cap and the inputs used to derive it."""

    max_cache_bytes: int
    breakdown: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_cache_bytes",
            _positive_int("max_cache_bytes", self.max_cache_bytes),
        )

    @property
    def max_cache_gib(self) -> float | None:
        return _float_ratio_or_none(self.max_cache_bytes, GIB_BYTES)


@dataclass(frozen=True)
class VllmUtilizationPlan:
    """A vLLM utilization after reserving the sampling cache budget."""

    base_util: float
    new_util: float
    reserved_fraction: Fraction
    sampling_cache_bytes: int
    gpu_total_bytes: int

    @property
    def changed(self) -> bool:
        return self.new_util != self.base_util


def _config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _nested_config_items(config: Any, key: str) -> list[tuple[str, Any]]:
    if config is None:
        return []
    items = []
    value = _config_value(config, key)
    if value is not None:
        items.append((key, value))
    text_config = _config_value(config, "text_config")
    if text_config is not None:
        value = _config_value(text_config, key)
        if value is not None:
            items.append((f"text_config.{key}", value))
    return items


def _nested_config_value(config: Any, key: str) -> Any:
    items = _nested_config_items(config, key)
    if not items:
        return None
    first_value = items[0][1]
    if any(value != first_value for _, value in items[1:]):
        values = ", ".join(f"{name}={value!r}" for name, value in items)
        raise ValueError(f"Model config has conflicting {key} values: {values}")
    return first_value


def moe_geometry_from_config(
    config: Any,
    *,
    num_moe_layers_override: int | None = None,
    topk_override: int | None = None,
    num_experts_override: int | None = None,
) -> MoEGeometry:
    """Extract MoE routing geometry from an already-loaded model config."""
    num_layers = num_moe_layers_override
    topk = topk_override
    if num_layers is None:
        num_layers = _nested_config_value(config, "num_hidden_layers")
    if topk is None:
        topk = _nested_config_value(config, "num_experts_per_tok")
    if num_layers is None or topk is None:
        raise ValueError(
            "Model config is missing MoE routing geometry "
            f"(num_hidden_layers={num_layers!r}, num_experts_per_tok={topk!r}). "
            "Pass num_moe_layers_override and topk_override explicitly."
        )
    if config is None and num_experts_override is None:
        raise ValueError(
            "num_experts_override is required when explicit MoE layer and "
            "top-k overrides are used without a model config"
        )
    geometry = MoEGeometry(num_moe_layers=num_layers, topk=topk)
    expert_counts = {}
    for field_name in _EXPERT_COUNT_FIELDS:
        for qualified_name, expert_count in _nested_config_items(config, field_name):
            expert_counts[qualified_name] = expert_count
    if num_experts_override is not None:
        expert_counts["num_experts_override"] = num_experts_override
    if not expert_counts:
        raise ValueError(
            "Model config is missing a recognized expert count "
            f"({_EXPERT_COUNT_FIELDS!r}). Pass num_experts_override explicitly."
        )
    validated_expert_counts = {}
    for field_name, expert_count in expert_counts.items():
        validated_count = _positive_int(field_name, expert_count)
        validated_expert_counts[field_name] = validated_count
        if validated_count > MAX_ROUTER_REPLAY_EXPERTS:
            raise ValueError(
                f"{field_name} ({validated_count}) must not exceed "
                f"{MAX_ROUTER_REPLAY_EXPERTS}; router-replay expert IDs are "
                "stored as uint8 or uint16"
            )
        if geometry.topk > validated_count:
            raise ValueError(
                f"topk ({geometry.topk}) must not exceed {field_name} "
                f"({validated_count})"
            )
    if len(set(validated_expert_counts.values())) > 1:
        counts = ", ".join(
            f"{field_name}={expert_count}"
            for field_name, expert_count in validated_expert_counts.items()
        )
        raise ValueError(f"Model config has conflicting expert counts: {counts}")
    expert_count = next(iter(validated_expert_counts.values()))
    dtype_bytes = (
        ROUTER_REPLAY_DTYPE_BYTES
        if expert_count <= 1 << (8 * ROUTER_REPLAY_DTYPE_BYTES)
        else ROUTER_REPLAY_WIDE_DTYPE_BYTES
    )
    return MoEGeometry(
        num_moe_layers=geometry.num_moe_layers,
        topk=geometry.topk,
        dtype_bytes=dtype_bytes,
    )


def resolve_model_moe_geometry(
    model_name: str,
    *,
    num_moe_layers_override: int | None = None,
    topk_override: int | None = None,
    num_experts_override: int | None = None,
    trust_remote_code: bool = False,
) -> MoEGeometry:
    """Resolve MoE geometry from explicit overrides or a Hugging Face config."""
    if num_moe_layers_override is not None and topk_override is not None:
        return moe_geometry_from_config(
            None,
            num_moe_layers_override=num_moe_layers_override,
            topk_override=topk_override,
            num_experts_override=num_experts_override,
        )
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(trust_remote_code, bool):
        raise TypeError(
            "trust_remote_code must be a boolean, " f"got {trust_remote_code!r}"
        )

    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError(
            "Resolving router-replay geometry from a model requires transformers. "
            "Install transformers or set both geometry overrides."
        ) from exc

    try:
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        raise ValueError(
            f"Could not load model config for {model_name!r}: {exc}. "
            "Set num_moe_layers_override and topk_override explicitly."
        ) from exc

    return moe_geometry_from_config(
        config,
        num_moe_layers_override=num_moe_layers_override,
        topk_override=topk_override,
        num_experts_override=num_experts_override,
    )


def derive_backlog_rollouts(
    *,
    batch_size: int,
    async_depth: int,
    max_inflight_rollouts: int | None = None,
    over_generation_multiplier: float = 1.0,
) -> int:
    """Derive the outstanding sampling/TX rollout population.

    The peak is the sum of in-flight rollouts, one completed over-generated
    wave, and accepted batches retained across the asynchronous window.
    """
    batch = _positive_int("batch_size", batch_size)
    depth = _nonnegative_int("async_depth", async_depth)
    over_generation = _factor(
        "over_generation_multiplier",
        over_generation_multiplier,
    )
    inflight = (
        batch
        if max_inflight_rollouts is None
        else _positive_int("max_inflight_rollouts", max_inflight_rollouts)
    )
    return inflight + _ceil_fraction(batch * over_generation) + batch * max(depth, 1)


def derive_resident_rows(*, batch_size: int, dp_size: int) -> int:
    """Derive rows resident in one training/RX data-parallel shard."""
    batch = _positive_int("batch_size", batch_size)
    data_parallel_size = _positive_int("dp_size", dp_size)
    return (batch + data_parallel_size - 1) // data_parallel_size


def derive_dp_size(
    *,
    training_gpus: int | None,
    sp_size: int | None,
) -> int:
    """Derive the training row-sharding degree as ``training_gpus // sp_size``."""
    sequence_parallel_size = 1 if sp_size is None else _positive_int("sp_size", sp_size)
    if training_gpus is None:
        return 1
    gpu_count = _positive_int("training_gpus", training_gpus)
    if gpu_count % sequence_parallel_size:
        raise ValueError("training_gpus must be divisible by sp_size")
    return gpu_count // sequence_parallel_size


def derive_num_sampling_workers(
    *,
    sampling_gpus: int | None,
    sampling_tensor_parallel_size: int | None,
    sampling_pipeline_parallel_size: int | None = None,
    override: int | None = None,
) -> int:
    """Derive a safe planning divisor from sampling replica topology."""
    tensor_parallel_size = (
        1
        if sampling_tensor_parallel_size is None
        else _positive_int(
            "sampling_tensor_parallel_size",
            sampling_tensor_parallel_size,
        )
    )
    pipeline_parallel_size = (
        1
        if sampling_pipeline_parallel_size is None
        else _positive_int(
            "sampling_pipeline_parallel_size",
            sampling_pipeline_parallel_size,
        )
    )
    if sampling_gpus is None:
        return 1 if override is None else _positive_int("override", override)
    gpu_count = _positive_int("sampling_gpus", sampling_gpus)
    replica_width = tensor_parallel_size * pipeline_parallel_size
    if gpu_count % replica_width:
        raise ValueError(
            "sampling_gpus must be divisible by the product of "
            "sampling_tensor_parallel_size and sampling_pipeline_parallel_size"
        )
    replicas = gpu_count // replica_width
    if override is None:
        return replicas
    planning_workers = _positive_int("override", override)
    if planning_workers > replicas:
        raise ValueError(
            "override must not exceed the sampling replica count derived from "
            "sampling_gpus and tensor/pipeline parallelism "
            f"({planning_workers} > {replicas})"
        )
    return planning_workers


def plan_sampling_cache_bytes(
    *,
    seq_len: int,
    geometry: MoEGeometry,
    backlog_rollouts: int,
    num_sampling_workers: int,
    routing_skew: float = DEFAULT_ROUTING_SKEW,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    dtype_bytes: int | None = None,
) -> CachePlan:
    """Estimate TX capacity assuming occupancy stays within the configured skew."""
    backlog = _positive_int("backlog_rollouts", backlog_rollouts)
    workers = _positive_int("num_sampling_workers", num_sampling_workers)
    skew = _factor("routing_skew", routing_skew)
    safety = _factor("safety_factor", safety_factor)
    effective_dtype_bytes = (
        geometry.dtype_bytes
        if dtype_bytes is None
        else _positive_int("dtype_bytes", dtype_bytes)
    )
    entry_bytes = geometry.entry_bytes(seq_len, dtype_bytes=effective_dtype_bytes)
    per_worker_rollouts = _ceil_fraction(Fraction(backlog, workers) * skew)
    max_cache_bytes = _ceil_fraction(per_worker_rollouts * entry_bytes * safety)
    return CachePlan(
        max_cache_bytes=max_cache_bytes,
        breakdown={
            "kind": "sampling_tx",
            "seq_len": int(seq_len),
            "num_moe_layers": geometry.num_moe_layers,
            "topk": geometry.topk,
            "dtype_bytes": effective_dtype_bytes,
            "entry_bytes": entry_bytes,
            "backlog_rollouts": backlog,
            "num_sampling_workers": workers,
            "routing_skew": _factor_breakdown_value(skew),
            "effective_routing_skew": _factor_breakdown_value(skew),
            "safety_factor": _factor_breakdown_value(safety),
            "per_worker_rollouts": per_worker_rollouts,
            "max_cache_bytes": max_cache_bytes,
            "max_cache_gib": _float_ratio_or_none(max_cache_bytes, GIB_BYTES),
        },
    )


def plan_training_cache_bytes(
    *,
    seq_len: int,
    geometry: MoEGeometry,
    resident_rows: int,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    dtype_bytes: int | None = None,
) -> CachePlan:
    """Plan the per-training-worker RX cache cap."""
    rows = _positive_int("resident_rows", resident_rows)
    safety = _factor("safety_factor", safety_factor)
    effective_dtype_bytes = (
        geometry.dtype_bytes
        if dtype_bytes is None
        else _positive_int("dtype_bytes", dtype_bytes)
    )
    per_row_bytes = geometry.entry_bytes(seq_len, dtype_bytes=effective_dtype_bytes)
    max_cache_bytes = _ceil_fraction(rows * per_row_bytes * safety)
    return CachePlan(
        max_cache_bytes=max_cache_bytes,
        breakdown={
            "kind": "training_rx",
            "seq_len": int(seq_len),
            "num_moe_layers": geometry.num_moe_layers,
            "topk": geometry.topk,
            "dtype_bytes": effective_dtype_bytes,
            "per_row_bytes": per_row_bytes,
            "resident_rows": rows,
            "safety_factor": _factor_breakdown_value(safety),
            "max_cache_bytes": max_cache_bytes,
            "max_cache_gib": _float_ratio_or_none(max_cache_bytes, GIB_BYTES),
        },
    )


def plan_vllm_gpu_memory_utilization(
    *,
    sampling_cache_bytes: int,
    gpu_total_bytes: int,
    base_util: float = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    max_reserved_fraction: float = DEFAULT_MAX_RESERVED_UTILIZATION_FRACTION,
    min_util: float = MIN_VLLM_GPU_MEMORY_UTILIZATION,
) -> VllmUtilizationPlan:
    """Reserve the TX cache budget by lowering vLLM GPU utilization."""
    cache_bytes = _positive_int("sampling_cache_bytes", sampling_cache_bytes)
    total_bytes = _positive_int("gpu_total_bytes", gpu_total_bytes)
    base_fraction = _finite_fraction(
        "base_util",
        base_util,
        exact_float=True,
    )
    maximum_reservation = _finite_fraction(
        "max_reserved_fraction",
        max_reserved_fraction,
        exact_float=True,
    )
    minimum = _finite_fraction("min_util", min_util, exact_float=True)
    if not 0 < base_fraction <= 1:
        raise ValueError("base_util must be in (0, 1]")
    if not 0 < maximum_reservation <= 1:
        raise ValueError("max_reserved_fraction must be in (0, 1]")
    if not 0 < minimum <= base_fraction:
        raise ValueError("min_util must be in (0, base_util]")

    reserved = Fraction(cache_bytes, total_bytes)
    if reserved > maximum_reservation:
        raise ValueError(
            "sampling cache requires reserving "
            f"{_format_fraction(reserved)} of GPU memory, above "
            "max_reserved_fraction "
            f"{_format_fraction(maximum_reservation)}"
        )
    exact_new_util = base_fraction - reserved
    if exact_new_util < minimum:
        raise ValueError(
            "sampling cache reservation would lower vLLM GPU memory utilization "
            f"to {_format_fraction(exact_new_util)}, below min_util "
            f"{_format_fraction(minimum)}"
        )
    new_util = float(exact_new_util)
    if Fraction.from_float(new_util) > exact_new_util:
        new_util = _previous_float(new_util)
    return VllmUtilizationPlan(
        base_util=float(base_fraction),
        new_util=new_util,
        reserved_fraction=reserved,
        sampling_cache_bytes=cache_bytes,
        gpu_total_bytes=total_bytes,
    )
