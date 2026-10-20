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

"""Capacity planning for uint8 and uint16 router-replay caches."""

import math
from decimal import Decimal
from fractions import Fraction

import pytest

from cortex_training import router_replay_planner as planner

GIB = 1024**3
GEOMETRY = planner.MoEGeometry(num_moe_layers=40, topk=8)
SEQ_LEN = 8192
ENTRY_BYTES = SEQ_LEN * 40 * 8


def test_moe_geometry_from_nested_config_and_overrides():
    config = {
        "text_config": {
            "num_hidden_layers": 40,
            "num_experts_per_tok": 8,
            "num_experts": 16,
        }
    }

    assert planner.moe_geometry_from_config(config) == GEOMETRY
    assert planner.moe_geometry_from_config(
        config,
        num_moe_layers_override=12,
        topk_override=4,
    ) == planner.MoEGeometry(num_moe_layers=12, topk=4)


def test_moe_geometry_rejects_missing_or_invalid_values():
    with pytest.raises(ValueError, match="missing MoE routing geometry"):
        planner.moe_geometry_from_config({})
    with pytest.raises(ValueError, match="num_moe_layers must be positive"):
        planner.moe_geometry_from_config(
            {
                "num_hidden_layers": 0,
                "num_experts_per_tok": 8,
                "num_experts": 16,
            }
        )
    with pytest.raises(ValueError, match="missing a recognized expert count"):
        planner.moe_geometry_from_config(
            {"num_hidden_layers": 40, "num_experts_per_tok": 8}
        )
    with pytest.raises(
        ValueError,
        match=r"topk \(257\) must not exceed num_experts \(256\)",
    ):
        planner.moe_geometry_from_config(
            {
                "num_hidden_layers": 40,
                "num_experts_per_tok": 257,
                "num_experts": 256,
            }
        )


@pytest.mark.parametrize(
    ("field_name", "root_value", "nested_value"),
    [
        ("num_hidden_layers", 1, 40),
        ("num_experts_per_tok", 1, 8),
    ],
)
def test_moe_geometry_rejects_conflicting_root_and_text_geometry(
    field_name,
    root_value,
    nested_value,
):
    config = {
        "num_hidden_layers": 40,
        "num_experts_per_tok": 8,
        "num_experts": 16,
        "text_config": {
            "num_hidden_layers": 40,
            "num_experts_per_tok": 8,
            "num_experts": 16,
        },
    }
    config[field_name] = root_value
    config["text_config"][field_name] = nested_value

    with pytest.raises(ValueError, match=f"conflicting {field_name} values"):
        planner.moe_geometry_from_config(config)


@pytest.mark.parametrize(
    "field_name",
    ["num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts"],
)
def test_moe_geometry_validates_established_expert_count_fields(field_name):
    config = {
        "text_config": {
            "num_hidden_layers": 40,
            "num_experts_per_tok": 8,
            field_name: 16,
        }
    }

    assert planner.moe_geometry_from_config(config) == GEOMETRY

    config["text_config"][field_name] = 0
    with pytest.raises(ValueError, match=f"{field_name} must be positive"):
        planner.moe_geometry_from_config(config)


def test_moe_geometry_validates_all_expert_count_aliases():
    config = {
        "num_hidden_layers": 40,
        "num_experts_per_tok": 8,
        "num_local_experts": 8,
        "num_experts": 512,
    }
    with pytest.raises(ValueError, match="conflicting expert counts"):
        planner.moe_geometry_from_config(config)

    config["num_experts"] = 16
    with pytest.raises(ValueError, match="conflicting expert counts"):
        planner.moe_geometry_from_config(config)

    config = {
        "num_hidden_layers": 40,
        "num_experts_per_tok": 8,
        "num_experts": 512,
    }
    with pytest.raises(ValueError, match="conflicting expert counts"):
        planner.moe_geometry_from_config(config, num_experts_override=128)

    config = {
        "num_hidden_layers": 40,
        "num_experts_per_tok": 8,
        "num_experts": 8,
        "text_config": {"num_experts": 512},
    }
    with pytest.raises(ValueError, match="conflicting expert counts"):
        planner.moe_geometry_from_config(config)


def test_moe_geometry_rejects_topk_above_expert_count_with_override():
    with pytest.raises(
        ValueError, match=r"topk \(9\) must not exceed num_experts \(8\)"
    ):
        planner.moe_geometry_from_config(
            {
                "num_hidden_layers": 40,
                "num_experts_per_tok": 4,
                "num_experts": 8,
            },
            topk_override=9,
        )


def test_moe_geometry_selects_capture_width_from_expert_count():
    config = {
        "num_hidden_layers": 40,
        "num_experts_per_tok": 8,
        "num_experts": 256,
    }

    assert planner.moe_geometry_from_config(config) == GEOMETRY

    config["num_experts"] = 257
    assert planner.moe_geometry_from_config(config) == planner.MoEGeometry(
        num_moe_layers=40,
        topk=8,
        dtype_bytes=2,
    )

    config["num_experts"] = 65536
    assert planner.moe_geometry_from_config(config).dtype_bytes == 2

    config["num_experts"] = 65537
    with pytest.raises(
        ValueError, match=r"num_experts \(65537\) must not exceed 65536.*uint16"
    ):
        planner.moe_geometry_from_config(config)


def test_explicit_geometry_avoids_transformers_import(monkeypatch):
    real_import = __import__

    def reject_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise AssertionError("explicit geometry must not import transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_transformers)

    assert (
        planner.resolve_model_moe_geometry(
            "not/loaded",
            num_moe_layers_override=40,
            topk_override=8,
            num_experts_override=16,
        )
        == GEOMETRY
    )


def test_explicit_geometry_requires_and_validates_expert_count():
    with pytest.raises(ValueError, match="num_experts_override is required"):
        planner.moe_geometry_from_config(
            None,
            num_moe_layers_override=40,
            topk_override=999,
        )
    with pytest.raises(ValueError, match="num_experts_override is required"):
        planner.resolve_model_moe_geometry(
            "not/loaded",
            num_moe_layers_override=40,
            topk_override=8,
        )
    with pytest.raises(ValueError, match="num_experts_override must be positive"):
        planner.resolve_model_moe_geometry(
            "not/loaded",
            num_moe_layers_override=40,
            topk_override=8,
            num_experts_override=0,
        )
    with pytest.raises(
        ValueError,
        match=r"topk \(17\) must not exceed num_experts_override \(16\)",
    ):
        planner.resolve_model_moe_geometry(
            "not/loaded",
            num_moe_layers_override=40,
            topk_override=17,
            num_experts_override=16,
        )


def test_partial_geometry_override_validates_loaded_expert_count(monkeypatch):
    class AutoConfig:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return {
                "num_hidden_layers": 40,
                "num_experts_per_tok": 4,
                "num_experts": 8,
            }

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        type("Transformers", (), {"AutoConfig": AutoConfig}),
    )

    with pytest.raises(ValueError, match=r"topk \(9\).*num_experts \(8\)"):
        planner.resolve_model_moe_geometry(
            "loaded/model",
            topk_override=9,
        )


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_model_geometry_forwards_explicit_remote_code_policy(
    monkeypatch,
    trust_remote_code,
):
    calls = []

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls.append((model_name, kwargs))
            return {
                "num_hidden_layers": 40,
                "num_experts_per_tok": 8,
                "num_experts": 16,
            }

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        type("Transformers", (), {"AutoConfig": AutoConfig}),
    )

    assert (
        planner.resolve_model_moe_geometry(
            "org/custom-model",
            trust_remote_code=trust_remote_code,
        )
        == GEOMETRY
    )
    assert calls == [
        (
            "org/custom-model",
            {"trust_remote_code": trust_remote_code},
        )
    ]


def test_model_geometry_remote_code_policy_defaults_to_false(monkeypatch):
    calls = []

    class AutoConfig:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            calls.append(kwargs)
            return {
                "num_hidden_layers": 40,
                "num_experts_per_tok": 8,
                "num_experts": 16,
            }

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        type("Transformers", (), {"AutoConfig": AutoConfig}),
    )

    planner.resolve_model_moe_geometry("org/custom-model")

    assert calls == [{"trust_remote_code": False}]


def test_derive_workload_geometry():
    assert (
        planner.derive_backlog_rollouts(
            batch_size=16,
            async_depth=2,
            max_inflight_rollouts=32,
            over_generation_multiplier=2.5,
        )
        == 32 + 40 + 32
    )
    assert (
        planner.derive_backlog_rollouts(
            batch_size=16,
            async_depth=0,
            max_inflight_rollouts=32,
            over_generation_multiplier=2.5,
        )
        == 32 + 40 + 16
    )
    assert planner.derive_resident_rows(batch_size=30, dp_size=8) == 4
    assert planner.derive_dp_size(training_gpus=8, sp_size=2) == 4
    assert planner.derive_dp_size(training_gpus=None, sp_size=None) == 1
    assert planner.derive_dp_size(training_gpus=None, sp_size=2) == 1
    assert (
        planner.derive_num_sampling_workers(
            sampling_gpus=8,
            sampling_tensor_parallel_size=2,
            sampling_pipeline_parallel_size=2,
        )
        == 2
    )
    assert (
        planner.derive_num_sampling_workers(
            sampling_gpus=None,
            sampling_tensor_parallel_size=4,
        )
        == 1
    )


@pytest.mark.parametrize("sp_size", [0, -1, True, 1.5, "2"])
def test_derive_dp_size_validates_sp_size_without_training_gpu_geometry(sp_size):
    with pytest.raises((TypeError, ValueError), match="sp_size must be"):
        planner.derive_dp_size(training_gpus=None, sp_size=sp_size)


def test_sampling_worker_override_cannot_under_size_known_topology():
    with pytest.raises(ValueError, match="must not exceed.*replica count"):
        planner.derive_num_sampling_workers(
            sampling_gpus=4,
            sampling_tensor_parallel_size=2,
            sampling_pipeline_parallel_size=2,
            override=100,
        )


def test_sampling_worker_override_may_conservatively_understate_known_topology():
    assert (
        planner.derive_num_sampling_workers(
            sampling_gpus=8,
            sampling_tensor_parallel_size=2,
            sampling_pipeline_parallel_size=2,
            override=1,
        )
        == 1
    )


def test_sampling_worker_override_is_fallback_when_gpu_geometry_is_unavailable():
    assert (
        planner.derive_num_sampling_workers(
            sampling_gpus=None,
            sampling_tensor_parallel_size=2,
            sampling_pipeline_parallel_size=2,
            override=3,
        )
        == 3
    )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda: planner.derive_backlog_rollouts(
                batch_size=0,
                async_depth=1,
            ),
            "batch_size must be positive",
        ),
        (
            lambda: planner.derive_backlog_rollouts(
                batch_size=1,
                async_depth=-1,
            ),
            "async_depth must be non-negative",
        ),
        (
            lambda: planner.derive_backlog_rollouts(
                batch_size=1,
                async_depth=1,
                max_inflight_rollouts=0,
            ),
            "max_inflight_rollouts must be positive",
        ),
        (
            lambda: planner.derive_backlog_rollouts(
                batch_size=1,
                async_depth=1,
                over_generation_multiplier=0.9,
            ),
            "over_generation_multiplier must be at least 1.0",
        ),
        (
            lambda: planner.derive_dp_size(training_gpus=8, sp_size=3),
            "training_gpus must be divisible",
        ),
        (
            lambda: planner.derive_num_sampling_workers(
                sampling_gpus=6,
                sampling_tensor_parallel_size=2,
                sampling_pipeline_parallel_size=2,
            ),
            "sampling_gpus must be divisible",
        ),
        (
            lambda: planner.derive_num_sampling_workers(
                sampling_gpus=4,
                sampling_tensor_parallel_size=0,
                override=1,
            ),
            "sampling_tensor_parallel_size must be positive",
        ),
    ],
)
def test_derivations_reject_invalid_counts(call, match):
    with pytest.raises((TypeError, ValueError), match=match):
        call()


def test_plan_sampling_cache_bytes():
    plan = planner.plan_sampling_cache_bytes(
        seq_len=SEQ_LEN,
        geometry=GEOMETRY,
        backlog_rollouts=64,
        num_sampling_workers=8,
        routing_skew=1.5,
        safety_factor=1.2,
    )

    per_worker_rollouts = 12
    assert plan.max_cache_bytes == math.ceil(per_worker_rollouts * ENTRY_BYTES * 1.2)
    assert plan.breakdown["per_worker_rollouts"] == per_worker_rollouts
    assert plan.breakdown["effective_routing_skew"] == 1.5


@pytest.mark.parametrize(
    ("workers", "expected_rollouts"),
    [(1, 144), (2, 72), (4, 36), (8, 18), (96, 2)],
)
def test_sampling_plan_scales_by_replicas_with_configured_skew(
    workers, expected_rollouts
):
    plan = planner.plan_sampling_cache_bytes(
        seq_len=1,
        geometry=planner.MoEGeometry(num_moe_layers=1, topk=1),
        backlog_rollouts=96,
        num_sampling_workers=workers,
        routing_skew=1.5,
        safety_factor=1,
    )

    assert plan.max_cache_bytes == expected_rollouts
    assert plan.breakdown["per_worker_rollouts"] == expected_rollouts
    assert plan.breakdown["effective_routing_skew"] == 1.5


def test_sampling_plan_default_skew_for_96_replicas():
    plan = planner.plan_sampling_cache_bytes(
        seq_len=262144,
        geometry=planner.MoEGeometry(num_moe_layers=40, topk=8),
        backlog_rollouts=3072,
        num_sampling_workers=96,
    )

    assert plan.breakdown["entry_bytes"] == 80 * 1024**2
    assert plan.breakdown["routing_skew"] == 2.0
    assert plan.breakdown["effective_routing_skew"] == 2.0
    assert plan.breakdown["per_worker_rollouts"] == 64
    assert plan.max_cache_bytes == 6 * GIB


def test_plan_training_cache_bytes():
    plan = planner.plan_training_cache_bytes(
        seq_len=SEQ_LEN,
        geometry=GEOMETRY,
        resident_rows=4,
        safety_factor=1.2,
    )

    assert plan.max_cache_bytes == math.ceil(4 * ENTRY_BYTES * 1.2)
    assert plan.breakdown["resident_rows"] == 4


def test_wide_expert_ids_double_sampling_and_training_cache_bytes():
    geometry = planner.MoEGeometry(
        num_moe_layers=40,
        topk=8,
        dtype_bytes=2,
    )

    sampling = planner.plan_sampling_cache_bytes(
        seq_len=SEQ_LEN,
        geometry=geometry,
        backlog_rollouts=1,
        num_sampling_workers=1,
        routing_skew=1,
        safety_factor=1,
    )
    training = planner.plan_training_cache_bytes(
        seq_len=SEQ_LEN,
        geometry=geometry,
        resident_rows=1,
        safety_factor=1,
    )

    assert sampling.breakdown["dtype_bytes"] == 2
    assert sampling.max_cache_bytes == 2 * ENTRY_BYTES
    assert training.breakdown["dtype_bytes"] == 2
    assert training.max_cache_bytes == 2 * ENTRY_BYTES


def test_large_integer_ceil_paths_are_exact():
    large = 2**53 + 1

    assert (
        planner.derive_backlog_rollouts(
            batch_size=large,
            async_depth=0,
            max_inflight_rollouts=large,
            over_generation_multiplier=1.0,
        )
        == 3 * large
    )
    assert (
        planner.derive_resident_rows(
            batch_size=large,
            dp_size=2,
        )
        == (large + 1) // 2
    )

    sampling = planner.plan_sampling_cache_bytes(
        seq_len=1,
        geometry=planner.MoEGeometry(num_moe_layers=1, topk=1),
        backlog_rollouts=large,
        num_sampling_workers=2,
        routing_skew=1.0,
        safety_factor=1.0,
    )
    assert sampling.breakdown["per_worker_rollouts"] == (large + 1) // 2
    assert sampling.max_cache_bytes == (large + 1) // 2

    training = planner.plan_training_cache_bytes(
        seq_len=1,
        geometry=planner.MoEGeometry(num_moe_layers=1, topk=1),
        resident_rows=large,
        safety_factor=1.0,
    )
    assert training.max_cache_bytes == large


def test_decimal_factors_use_exact_rational_ceilings():
    large = 2**53 + 1
    six_fifths_ceiling = (large * 6 + 4) // 5

    assert (
        planner.derive_backlog_rollouts(
            batch_size=large,
            async_depth=0,
            max_inflight_rollouts=large,
            over_generation_multiplier=1.2,
        )
        == large + six_fifths_ceiling + large
    )

    sampling = planner.plan_sampling_cache_bytes(
        seq_len=1,
        geometry=planner.MoEGeometry(num_moe_layers=1, topk=1),
        backlog_rollouts=large,
        num_sampling_workers=1,
        routing_skew=1.2,
        safety_factor=1.2,
    )
    assert sampling.breakdown["per_worker_rollouts"] == six_fifths_ceiling
    assert sampling.max_cache_bytes == (six_fifths_ceiling * 6 + 4) // 5

    training = planner.plan_training_cache_bytes(
        seq_len=1,
        geometry=planner.MoEGeometry(num_moe_layers=1, topk=1),
        resident_rows=large,
        safety_factor=1.2,
    )
    assert training.max_cache_bytes == six_fifths_ceiling


def test_capacity_plans_accept_integers_beyond_float_range():
    huge = 10**1000
    sampling = planner.plan_sampling_cache_bytes(
        seq_len=1,
        geometry=planner.MoEGeometry(num_moe_layers=1, topk=1),
        backlog_rollouts=huge,
        num_sampling_workers=2,
        routing_skew=1,
        safety_factor=1,
    )
    training = planner.plan_training_cache_bytes(
        seq_len=1,
        geometry=planner.MoEGeometry(num_moe_layers=1, topk=1),
        resident_rows=huge,
        safety_factor=1,
    )

    assert sampling.max_cache_bytes == huge // 2
    assert sampling.max_cache_gib is None
    assert sampling.breakdown["max_cache_gib"] is None
    assert training.max_cache_bytes == huge
    assert training.max_cache_gib is None
    assert training.breakdown["max_cache_gib"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seq_len": 0},
        {"backlog_rollouts": 0},
        {"num_sampling_workers": 0},
        {"routing_skew": 0.5},
        {"safety_factor": 0.5},
        {"safety_factor": math.inf},
    ],
)
def test_sampling_plan_rejects_invalid_geometry_counts_and_factors(kwargs):
    values = {
        "seq_len": SEQ_LEN,
        "geometry": GEOMETRY,
        "backlog_rollouts": 64,
        "num_sampling_workers": 8,
        "routing_skew": 1.5,
        "safety_factor": 1.2,
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        planner.plan_sampling_cache_bytes(**values)


def test_plan_vllm_gpu_memory_utilization():
    plan = planner.plan_vllm_gpu_memory_utilization(
        sampling_cache_bytes=GIB,
        gpu_total_bytes=140 * GIB,
        base_util=0.9,
    )

    assert plan.reserved_fraction == pytest.approx(1 / 140)
    assert plan.new_util == pytest.approx(0.9 - 1 / 140)
    assert plan.changed


def test_vllm_plan_reserves_the_full_fraction_without_rounding():
    plan = planner.plan_vllm_gpu_memory_utilization(
        sampling_cache_bytes=1,
        gpu_total_bytes=80 * GIB,
    )

    exact_reserved = Fraction(1, 80 * GIB)
    exact_new_util = Fraction.from_float(0.9) - exact_reserved
    assert plan.reserved_fraction == exact_reserved
    assert Fraction.from_float(plan.new_util) <= exact_new_util
    assert plan.new_util < plan.base_util


def test_vllm_plan_conservatively_represents_sub_ulp_reservation_without_nextafter(
    monkeypatch,
):
    monkeypatch.delattr(math, "nextafter")

    plan = planner.plan_vllm_gpu_memory_utilization(
        sampling_cache_bytes=1,
        gpu_total_bytes=10**30,
    )
    exact_new_util = Fraction.from_float(0.9) - Fraction(1, 10**30)
    assert plan.reserved_fraction == Fraction(1, 10**30)
    assert plan.new_util < 0.9
    assert Fraction.from_float(plan.new_util) <= exact_new_util


def test_vllm_plan_rejects_cache_above_maximum_reservation():
    with pytest.raises(ValueError, match="above max_reserved_fraction"):
        planner.plan_vllm_gpu_memory_utilization(
            sampling_cache_bytes=60 * GIB,
            gpu_total_bytes=80 * GIB,
        )
    with pytest.raises(ValueError, match="above max_reserved_fraction"):
        planner.plan_vllm_gpu_memory_utilization(
            sampling_cache_bytes=10**1000,
            gpu_total_bytes=1,
        )


def test_vllm_plan_rejects_cache_that_would_cross_minimum_utilization():
    with pytest.raises(ValueError, match="below min_util"):
        planner.plan_vllm_gpu_memory_utilization(
            sampling_cache_bytes=20 * GIB,
            gpu_total_bytes=80 * GIB,
            base_util=0.2,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sampling_cache_bytes": 0},
        {"gpu_total_bytes": 0},
        {"base_util": 0},
        {"base_util": 1.1},
        {"max_reserved_fraction": 0},
        {"min_util": 0},
        {"min_util": 0.95},
    ],
)
def test_vllm_plan_rejects_invalid_inputs(kwargs):
    values = {
        "sampling_cache_bytes": GIB,
        "gpu_total_bytes": 140 * GIB,
        "base_util": 0.9,
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        planner.plan_vllm_gpu_memory_utilization(**values)


def test_gib_to_bytes_preserves_fractional_gib():
    assert planner.gib_to_bytes(3) == 3 * GIB
    assert planner.gib_to_bytes(0.5) == GIB // 2
    assert planner.gib_to_bytes(2**53 + 1) == (2**53 + 1) * GIB
    assert planner.gib_to_bytes(Decimal("0.1")) == 107374183
    assert planner.gib_to_bytes(Fraction(1, 10**30)) == 1

    with pytest.raises(ValueError, match="gib must be positive"):
        planner.gib_to_bytes(0)
