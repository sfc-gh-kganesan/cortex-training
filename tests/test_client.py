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

"""Unit tests for ``cortex_training.client.core``.

Two suites:

- Pure-logic tests for the typed config dataclasses (``JobType``,
  ``TrainingConfig``, ``InferenceConfig``, ``SubJobConfig``): construction,
  validation (mirrors the server-side validators), and ``to_wire``
  serialization (matches the documented REST shape).

- HTTP-surface tests for ``CortexTrainingClient`` that stub ``client._session`` with
  a :class:`unittest.mock.MagicMock`, so every test runs offline.
"""

from __future__ import annotations

import base64
import contextlib
import gzip
import importlib
import json
import logging
import os
import sys
import warnings
import tempfile
import threading
import time
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from pathlib import PurePosixPath
from types import SimpleNamespace
from urllib.parse import unquote
from urllib.parse import urlparse
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

# Import the module directly so we don't pull src/cortex_training/__init__.py
# (which imports torch via wire.py).
spec = importlib.util.spec_from_file_location(
    "cortex_training_client_under_test",
    str(
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "src"
        / "cortex_training"
        / "client"
        / "core.py"
    ),
)
nc = importlib.util.module_from_spec(spec)
sys.modules["cortex_training_client_under_test"] = nc
spec.loader.exec_module(nc)

JobType = nc.JobType
TrainingConfig = nc.TrainingConfig
InferenceConfig = nc.InferenceConfig
SubJobConfig = nc.SubJobConfig
CortexTrainingClient = nc.CortexTrainingClient
Hardware = nc.Hardware

# Spellings a caller could plausibly pass for the unsupported log-probability
# type: the enum, loose casing, the fully-qualified enum name, the short alias,
# and surrounding whitespace.
LOG_PROBABILITY_JOB_TYPES = [
    JobType.LOG_PROBABILITY,
    "LOG_PROBABILITY",
    "JOB_TYPE_LOG_PROBABILITY",
    "log_prob",
    " log_probability ",
]


def _wire_load(data: bytes):
    from cortex_training import wire

    return wire.loads(data)


# ─── Helpers ────────────────────────────────────────────────────────────


def _ok_training() -> TrainingConfig:
    return TrainingConfig(
        optimizer={"type": "adamw", "lr": 1e-5},
        max_seq_len=128,
        train_batch_size=1,
        n_gpus=2,
    )


def _ok_inference() -> InferenceConfig:
    return InferenceConfig(max_seq_len=2048, n_gpus=1)


def _make_response(json_body=None, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body if json_body is not None else {}
    resp.raise_for_status = MagicMock()
    return resp


def _make_error_response(json_body, status_code: int = 409):
    resp = nc.requests.Response()
    resp.status_code = status_code
    resp.url = "http://test.local/error"
    resp.headers["Content-Type"] = "application/json"
    resp._content = json.dumps(json_body).encode("utf-8")
    # As after requests reads a non-streamed body, so close() needs no connection.
    resp._content_consumed = True
    return resp


def _make_client(post_json=None, get_json=None) -> CortexTrainingClient:
    client = CortexTrainingClient(
        base_url="http://test.local",
        database="DB",
        schema="SCH",
    )
    client._session = MagicMock()
    client._session.post.return_value = _make_response(post_json)
    client._session.get.return_value = _make_response(get_json)
    return client


# ─── JobType ────────────────────────────────────────────────────────────


class TestJobType:
    def test_string_values(self):
        assert JobType.TRAINING.value == "training"
        assert JobType.SAMPLING.value == "sampling"
        assert JobType.LOG_PROBABILITY.value == "log_probability"

    def test_enum_is_str_subclass(self):
        # JSON serialization treats it as the literal string.
        assert JobType.TRAINING == "training"


# ─── TrainingConfig ─────────────────────────────────────────────────────


class TestTrainingConfig:
    def test_validate_ok(self):
        _ok_training().validate()

    def test_validate_rejects_zero_max_seq_len(self):
        with pytest.raises(ValidationError, match="max_seq_len"):
            TrainingConfig(optimizer={"type": "adamw"}, max_seq_len=0, train_batch_size=1, n_gpus=2)

    def test_validate_rejects_zero_train_batch_size(self):
        with pytest.raises(ValidationError, match="train_batch_size"):
            TrainingConfig(optimizer={"type": "adamw"}, max_seq_len=128, train_batch_size=0, n_gpus=2)

    def test_validate_rejects_empty_optimizer(self):
        with pytest.raises(ValidationError, match="optimizer"):
            TrainingConfig(optimizer={}, max_seq_len=128, train_batch_size=1, n_gpus=2)

    @pytest.mark.parametrize("stage", [3, "3"])
    def test_validate_rejects_zero_stage_3(self, stage):
        with pytest.raises(ValidationError, match=r"training\.extra\.ds_config\.zero_optimization\.stage=3 \(ZeRO-3\)"):
            TrainingConfig(
                optimizer={"type": "adamw"}, max_seq_len=128, train_batch_size=1, n_gpus=2,
                ds_config={"zero_optimization": {"stage": stage}},
            )

    @pytest.mark.parametrize("stage", [0, 1, 2])
    def test_validate_accepts_zero_stages_below_3(self, stage):
        TrainingConfig(
            optimizer={"type": "adamw"}, max_seq_len=128, train_batch_size=1, n_gpus=2,
            ds_config={"zero_optimization": {"stage": stage}},
        )

    def test_validate_accepts_nested_primerl_fused_ce_false(self):
        TrainingConfig(
            optimizer={"type": "adamw"}, max_seq_len=128, train_batch_size=1, n_gpus=2,
            fp32_lm_head=True, fused_lm_head_token_chunk_size=8192,
            prime_rl={"fused_cross_entropy": False, "fp32_lm_head": True, "fused_lm_head_token_chunk_size": 8192},
        )

    def test_validate_nested_primerl_overrides_top_level_fused_ce(self):
        TrainingConfig(
            optimizer={"type": "adamw"}, max_seq_len=128, train_batch_size=1, n_gpus=2,
            fused_cross_entropy="liger", fp32_lm_head=True,
            prime_rl={"fused_cross_entropy": False, "fp32_lm_head": True},
        )

    def test_validate_rejects_default_fused_ce_with_chunked_lm_head(self):
        with pytest.raises(ValidationError, match="cannot combine fused_cross_entropy"):
            TrainingConfig(
                optimizer={"type": "adamw"}, max_seq_len=128, train_batch_size=1, n_gpus=2,
                fused_lm_head_token_chunk_size=8192,
            )

    def test_validate_rejects_nested_primerl_fused_ce_with_fp32_lm_head(self):
        with pytest.raises(ValidationError, match="cannot combine fused_cross_entropy"):
            TrainingConfig(
                optimizer={"type": "adamw"}, max_seq_len=128, train_batch_size=1, n_gpus=2,
                fused_cross_entropy=False,
                prime_rl={"fused_cross_entropy": "liger", "fp32_lm_head": True},
            )

    def test_validate_rejects_mutation(self):
        tc = _ok_training()
        with pytest.raises(ValidationError, match="n_gpus"):
            tc.n_gpus = 0

    def test_to_wire_required_fields(self):
        wire = _ok_training().to_wire()
        assert wire == {
            "optimizer": {"name": "adamw", "lr": 1e-5},
            "max_seq_len": 128,
            "train_batch_size": 1,
            "n_gpus": 2,
        }

    def test_to_wire_includes_gradient_clipping_when_set(self):
        tc = _ok_training()
        tc.gradient_clipping = 1.5
        assert tc.to_wire()["gradient_clipping"] == 1.5

    def test_to_wire_omits_gradient_clipping_when_none(self):
        assert "gradient_clipping" not in _ok_training().to_wire()

    def test_to_wire_always_includes_n_gpus(self):
        assert _ok_training().to_wire()["n_gpus"] == 2

    def test_validate_rejects_zero_n_gpus(self):
        with pytest.raises(ValidationError, match="n_gpus"):
            TrainingConfig(optimizer={"type": "adamw"}, max_seq_len=128, train_batch_size=1, n_gpus=0)

    def test_to_wire_includes_multiplex_job_id_when_set(self):
        tc = TrainingConfig(optimizer={"type": "adamw", "lr": 1e-5}, max_seq_len=128, train_batch_size=1, n_gpus=2, multiplex_job_id="train-1")
        assert tc.to_wire()["multiplex_job_id"] == "train-1"

    def test_to_wire_omits_multiplex_job_id_when_none(self):
        assert "multiplex_job_id" not in _ok_training().to_wire()

    def test_to_wire_omits_load_optimizer_states_when_none(self):
        assert "load_optimizer_states" not in _ok_training().to_wire()

    def test_to_wire_includes_load_optimizer_states_when_false(self):
        tc = _ok_training()
        tc.load_optimizer_states = False
        assert tc.to_wire()["load_optimizer_states"] is False

    def test_to_wire_includes_load_optimizer_states_when_true(self):
        tc = _ok_training()
        tc.load_optimizer_states = True
        assert tc.to_wire()["load_optimizer_states"] is True

    def test_to_wire_merges_extra_passthrough(self):
        tc = TrainingConfig(
            optimizer={"type": "adamw", "lr": 1e-5}, max_seq_len=128, train_batch_size=1, n_gpus=2,
            extra={"fp16": {"enabled": True}, "zero_stage": 2},
        )
        wire = tc.to_wire()
        assert wire["fp16"] == {"enabled": True}
        assert wire["zero_stage"] == 2

    def test_to_wire_extra_does_not_override_required(self):
        tc = TrainingConfig(
            optimizer={"type": "adamw", "lr": 1e-5}, max_seq_len=128, train_batch_size=1, n_gpus=2,
            extra={"max_seq_len": 999, "optimizer": {"type": "sgd"}},
        )
        wire = tc.to_wire()
        assert wire["max_seq_len"] == 128
        assert wire["optimizer"] == {"name": "adamw", "lr": 1e-5}


# ─── TrainingConfig — optimizer canonicalization ────────────────────────


class TestTrainingConfigOptimizer:
    def test_aliases_are_canonicalized_on_construction(self):
        tc = TrainingConfig(
            optimizer={"type": "adamw", "beta1": 0.9, "beta2": 0.95},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
        )
        assert tc.optimizer == {"name": "adamw", "betas": [0.9, 0.95]}
        assert tc.to_wire()["optimizer"] == {"name": "adamw", "betas": [0.9, 0.95]}

    def test_nested_gradient_clipping_is_hoisted(self):
        tc = TrainingConfig(
            optimizer={"lr": 1e-5, "gradient_clipping": 0.5},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
        )
        assert tc.optimizer == {"lr": 1e-5}
        assert tc.gradient_clipping == 0.5
        assert tc.to_wire()["gradient_clipping"] == 0.5

    def test_typed_gradient_clipping_wins_over_the_nested_value(self):
        tc = TrainingConfig(
            optimizer={"lr": 1e-5, "gradient_clipping": 0.5},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
            gradient_clipping=2.0,
        )
        assert tc.gradient_clipping == 2.0
        assert "gradient_clipping" not in tc.optimizer

    def test_clip_only_optimizer_empties_the_block_and_revalidates(self):
        tc = TrainingConfig(
            optimizer={"gradient_clipping": 1.0},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
        )
        assert tc.optimizer == {}
        assert tc.gradient_clipping == 1.0
        # The canonical model must survive a second pass; validate() and
        # create_job both re-run model_validate.
        tc.validate()
        assert tc.to_wire()["optimizer"] == {}

    def test_empty_optimizer_with_gradient_clipping_is_accepted(self):
        tc = TrainingConfig(
            optimizer={},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
            gradient_clipping=1.0,
        )
        assert tc.to_wire()["optimizer"] == {}
        assert tc.to_wire()["gradient_clipping"] == 1.0

    def test_unknown_optimizer_key_is_rejected(self):
        with pytest.raises(ValidationError, match=r"training\.optimizer contains unsupported field"):
            TrainingConfig(
                optimizer={"lr": 1e-5, "momentum": 0.9},
                max_seq_len=128,
                train_batch_size=1,
                n_gpus=2,
            )

    def test_scheduler_optimizer_key_is_rejected(self):
        with pytest.raises(ValidationError, match="unsupported scheduler field"):
            TrainingConfig(
                optimizer={"lr": 1e-5, "warmup_steps": 10},
                max_seq_len=128,
                train_batch_size=1,
                n_gpus=2,
            )

    def test_peft_config_in_extra_is_normalized(self):
        tc = TrainingConfig(
            optimizer={"lr": 1e-5},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
            peft_config={"peft_type": "Lora", "r": 16, "target_modules": ["q_proj"]},
        )
        peft = tc.to_wire()["peft_config"]
        assert peft["task_type"] == "CAUSAL_LM"
        assert peft["lora_alpha"] == 8
        assert peft["target_parameters"] == []

    def test_invalid_peft_config_in_extra_is_rejected(self):
        with pytest.raises(ValidationError, match=r"training\.extra\.peft_config\.peft_type"):
            TrainingConfig(
                optimizer={"lr": 1e-5},
                max_seq_len=128,
                train_batch_size=1,
                n_gpus=2,
                peft_config={"peft_type": "LORA", "target_modules": ["q_proj"]},
            )


# ─── InferenceConfig ────────────────────────────────────────────────────


class TestInferenceConfig:
    def test_validate_ok(self):
        _ok_inference().validate()

    def test_validate_rejects_zero(self):
        with pytest.raises(ValidationError, match="max_seq_len"):
            InferenceConfig(max_seq_len=0, n_gpus=1)

    def test_validate_rejects_zero_n_gpus(self):
        with pytest.raises(ValidationError, match="n_gpus"):
            InferenceConfig(max_seq_len=2048, n_gpus=0)

    def test_to_wire_includes_extra(self):
        ic = InferenceConfig(max_seq_len=4096, n_gpus=1, extra={"gpu_memory_utilization": 0.9})
        assert ic.to_wire() == {"max_seq_len": 4096, "n_gpus": 1, "gpu_memory_utilization": 0.9}

    def test_to_wire_always_includes_n_gpus(self):
        assert _ok_inference().to_wire()["n_gpus"] == 1

    def test_to_wire_includes_multiplex_job_id_when_set(self):
        ic = InferenceConfig(max_seq_len=4096, n_gpus=1, multiplex_job_id="train-1")
        assert ic.to_wire()["multiplex_job_id"] == "train-1"

    def test_to_wire_omits_multiplex_job_id_when_none(self):
        assert "multiplex_job_id" not in _ok_inference().to_wire()

    def test_to_wire_extra_does_not_override_required(self):
        ic = InferenceConfig(max_seq_len=4096, n_gpus=1, extra={"max_seq_len": 1})
        assert ic.to_wire()["max_seq_len"] == 4096

    def test_peft_config_in_extra_is_normalized(self):
        sub = SubJobConfig.sampling_job(
            model_name="gpt2",
            max_seq_len=4096,
            n_gpus=1,
            extra_sampling={
                "peft_config": {"peft_type": "Lora", "r": 16, "target_modules": ["q_proj"]}
            },
        )
        peft = sub.to_wire()["inference_config"]["peft_config"]
        assert peft["task_type"] == "CAUSAL_LM"
        assert peft["target_parameters"] == []

    def test_invalid_peft_config_in_extra_is_rejected(self):
        with pytest.raises(ValidationError, match=r"sampling\.extra\.peft_config\.peft_type"):
            InferenceConfig(
                max_seq_len=4096,
                n_gpus=1,
                peft_config={"peft_type": "LORA", "target_modules": ["q_proj"]},
            )


# ─── SubJobConfig — factories ───────────────────────────────────────────


class TestSubJobConfigFactories:
    def test_training_job_factory_minimal(self):
        sub = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
        )
        assert sub.job_type == JobType.TRAINING
        assert sub.model_name == "gpt2"
        assert isinstance(sub.training, TrainingConfig)
        assert sub.sampling is None
        assert sub.training.model_extra == {}

    def test_training_job_factory_full(self):
        sub = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=4,
            gradient_clipping=1.0,
            multiplex_job_id="train-1",
            extra_training={"fp16": {"enabled": True}},
            global_batch_size=4,
            dtype="bf16",
            seed=42,
            model_post_init=["init_a", "init_b"],
            load_optimizer_states=False,
            source_checkpoint_info={"checkpoint_id": "cp_1", "source_job_id": "job-a"},
        )
        assert sub.training.gradient_clipping == 1.0
        assert sub.training.n_gpus == 4
        assert sub.training.multiplex_job_id == "train-1"
        assert sub.training.load_optimizer_states is False
        assert sub.training.to_wire()["load_optimizer_states"] is False
        assert sub.training.model_extra == {"fp16": {"enabled": True}}
        assert sub.global_batch_size == 4
        assert sub.dtype == "bf16"
        assert sub.seed == 42
        assert sub.model_post_init_ops == ["init_a", "init_b"]
        assert sub.source_checkpoint_info == {"checkpoint_id": "cp_1", "source_job_id": "job-a"}

    def test_sampling_job_factory_default_type(self):
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        assert sub.job_type == JobType.SAMPLING
        assert isinstance(sub.sampling, InferenceConfig)
        assert sub.training is None

    def test_sampling_job_factory_optional_sizing(self):
        sub = SubJobConfig.sampling_job(
            model_name="gpt2",
            max_seq_len=128,
            n_gpus=2,
            multiplex_job_id="train-1",
        )
        assert sub.sampling.n_gpus == 2
        assert sub.sampling.multiplex_job_id == "train-1"

    def test_sampling_job_factory_source_checkpoint(self):
        source = {"checkpoint_id": "cp_1", "source_job_id": "job-a"}
        sub = SubJobConfig.sampling_job(
            model_name="gpt2",
            max_seq_len=128,
            n_gpus=1,
            source_checkpoint_info=source,
        )
        assert sub.source_checkpoint_info == source

    @pytest.mark.parametrize("job_type", LOG_PROBABILITY_JOB_TYPES)
    def test_sampling_job_factory_rejects_log_probability(self, job_type):
        with pytest.raises(
            ValueError,
            match=r"sampling_job\(\)\.job_type: log_probability sub-jobs are not currently supported",
        ):
            SubJobConfig.sampling_job(
                model_name="gpt2",
                max_seq_len=128,
                n_gpus=1,
                job_type=job_type,
            )

    def test_sampling_job_factory_rejects_training_type(self):
        with pytest.raises(ValueError, match="only accepts SAMPLING, got"):
            SubJobConfig.sampling_job(
                model_name="gpt2",
                max_seq_len=128,
                n_gpus=1,
                job_type=JobType.TRAINING,
            )

    def test_extra_dicts_are_copied(self):
        # Mutating the caller's dict after construction must not leak into the
        # built sub-job (defensive copy).
        extra = {"fp16": {"enabled": True}}
        sub = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
            extra_training=extra,
        )
        extra["leaked"] = "yes"
        assert "leaked" not in (sub.training.model_extra or {})


# ─── SubJobConfig — validate ────────────────────────────────────────────


class TestSubJobConfigValidate:
    def test_ok_training(self):
        SubJobConfig(
            job_type=JobType.TRAINING,
            model_name="gpt2",
            training=_ok_training(),
        ).validate()

    def test_ok_sampling(self):
        SubJobConfig(
            job_type=JobType.SAMPLING,
            model_name="gpt2",
            sampling=_ok_inference(),
        ).validate()

    def test_ok_log_probability_uses_sampling_block(self):
        SubJobConfig(
            job_type=JobType.LOG_PROBABILITY,
            model_name="gpt2",
            sampling=_ok_inference(),
        ).validate()

    def test_rejects_empty_model_name(self):
        with pytest.raises(ValidationError, match="model_name"):
            SubJobConfig(
                job_type=JobType.TRAINING,
                model_name="",
                training=_ok_training(),
            )

    def test_training_without_training_block(self):
        with pytest.raises(ValidationError, match="training sub-job requires"):
            SubJobConfig(job_type=JobType.TRAINING, model_name="gpt2")

    def test_sampling_without_sampling_block(self):
        with pytest.raises(ValidationError, match="sampling sub-job requires"):
            SubJobConfig(job_type=JobType.SAMPLING, model_name="gpt2")

    def test_log_probability_without_sampling_block(self):
        with pytest.raises(ValidationError, match="log_probability sub-job requires"):
            SubJobConfig(
                job_type=JobType.LOG_PROBABILITY,
                model_name="gpt2",
            )

    def test_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            SubJobConfig(
                job_type=JobType.TRAINING,
                model_name="gpt2",
                training=_ok_training(),
                sampling=_ok_inference(),
            )

    def test_propagates_nested_validation_failure(self):
        with pytest.raises(ValidationError, match="train_batch_size"):
            SubJobConfig(
                job_type=JobType.TRAINING,
                model_name="gpt2",
                training=TrainingConfig(
                    optimizer={"type": "adamw"},
                    max_seq_len=128,
                    train_batch_size=0,
                    n_gpus=2,
                ),
            )


# ─── SubJobConfig — to_wire ─────────────────────────────────────────────


class TestSubJobConfigToWire:
    def test_training_minimal(self):
        sub = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
        )
        assert sub.to_wire() == {
            "job_type": "training",
            "model_name": "gpt2",
            "training_config": {
                "optimizer": {"name": "adamw"},
                "max_seq_len": 128,
                "train_batch_size": 1,
                "n_gpus": 2,
            },
        }

    def test_training_with_optionals(self):
        sub = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
            global_batch_size=4,
            dtype="bf16",
            seed=42,
            model_post_init=["init"],
            source_checkpoint_info={"checkpoint_id": "cp_1", "source_job_id": "job-a"},
        )
        wire = sub.to_wire()
        assert wire["global_batch_size"] == 4
        assert wire["model_post_init"] == ["init"]
        assert wire["dtype"] == "bf16"
        assert wire["seed"] == 42
        assert wire["source_checkpoint_info"] == {"checkpoint_id": "cp_1", "source_job_id": "job-a"}

    def test_sampling_block_used_for_log_probability(self):
        # Constructed directly: the factory rejects log-probability, but the
        # schema type still serializes.
        sub = SubJobConfig(
            job_type=JobType.LOG_PROBABILITY,
            model_name="gpt2",
            sampling=InferenceConfig(max_seq_len=128, n_gpus=1),
        )
        wire = sub.to_wire()
        assert wire["job_type"] == "log_probability"
        assert wire["inference_config"] == {"max_seq_len": 128, "n_gpus": 1}
        assert "training_config" not in wire

    def test_sampling_with_source_checkpoint(self):
        sub = SubJobConfig.sampling_job(
            model_name="gpt2",
            max_seq_len=128,
            n_gpus=1,
            source_checkpoint_info={
                "checkpoint_id": "cp_1",
                "source_job_id": "job-a",
            },
        )
        assert sub.to_wire()["source_checkpoint_info"] == {
            "checkpoint_id": "cp_1",
            "source_job_id": "job-a",
        }

    def test_omits_unset_optional_fields(self):
        sub = SubJobConfig(
            job_type=JobType.TRAINING,
            model_name="gpt2",
            training=_ok_training(),
        )
        wire = sub.to_wire()
        for absent in (
            "global_batch_size",
            "source_checkpoint_info",
            "dtype",
            "seed",
            "model_post_init",
            "inference_config",
        ):
            assert absent not in wire


# ─── CortexTrainingClient — auth & URL composition ────────────────────────────


class TestClientConstruction:
    def test_init_strips_trailing_slash(self):
        c = CortexTrainingClient(base_url="http://x.test/", database="DB", schema="SCH")
        assert c.base_url == "http://x.test"

    def test_default_polling_config(self):
        c = CortexTrainingClient(base_url="http://x.test", database="DB", schema="SCH")
        assert c.poll_interval == 0.5
        assert c.poll_timeout == 1800.0
        assert c.poll_backoff_multiplier == 1.25
        assert c.poll_max_interval == 6.0

    def test_default_request_timeout_is_a_connect_read_pair(self):
        c = CortexTrainingClient(
            base_url="http://x.test", database="DB", schema="SCH"
        )
        assert c.request_timeout == (30.0, 600.0)
        assert c._session.default_timeout == c.request_timeout

    def test_scalar_request_timeout_applies_to_connect_and_read(self):
        c = CortexTrainingClient(
            base_url="http://x.test",
            database="DB",
            schema="SCH",
            request_timeout=12,
        )
        assert c.request_timeout == (12.0, 12.0)

    @pytest.mark.parametrize(
        "bad",
        [0, -1, (30.0, 0), (30.0,), float("inf"), float("nan"), None, True, "30"],
    )
    def test_rejects_invalid_request_timeout(self, bad):
        with pytest.raises(ValueError, match="request_timeout"):
            CortexTrainingClient(
                base_url="http://x.test",
                database="DB",
                schema="SCH",
                request_timeout=bad,
            )

    def test_session_injects_default_timeout_unless_call_overrides_it(
        self, monkeypatch
    ):
        seen = []

        def fake_request(_session, *_args, **kwargs):
            seen.append(kwargs["timeout"])
            return _make_response()

        monkeypatch.setattr(nc.requests.Session, "request", fake_request)
        session = nc._DefaultTimeoutSession((2.0, 3.0))

        session.get("http://x.test/default")
        session.get("http://x.test/override", timeout=(0.5, 1.0))

        assert seen == [(2.0, 3.0), (0.5, 1.0)]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"poll_interval": 0}, "poll_interval"),
            ({"poll_timeout": 0}, "poll_timeout"),
            ({"poll_backoff_multiplier": 0.5}, "poll_backoff_multiplier"),
            ({"poll_interval": 2, "poll_max_interval": 1}, "poll_max_interval"),
        ],
    )
    def test_rejects_invalid_polling_config(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CortexTrainingClient(
                base_url="http://x.test",
                database="DB",
                schema="SCH",
                **kwargs,
            )

    def test_default_endpoint_is_cortex_training(self):
        c = CortexTrainingClient(base_url="http://x.test", database="DB", schema="SCH")
        assert c.endpoint == "cortex-training"

    def test_prefix_url(self):
        c = CortexTrainingClient(base_url="http://x.test", database="DB", schema="SCH")
        assert c._prefix == "http://x.test/api/v2/databases/DB/schemas/SCH/cortex-training"

    def test_prefix_url_leaves_dollar_in_plain_names_bare(self):
        c = CortexTrainingClient(base_url="http://x.test", database="MY$DB", schema="S$1")
        assert c._prefix == "http://x.test/api/v2/databases/MY$DB/schemas/S$1/cortex-training"

    def test_from_pat_sets_headers_and_verify(self):
        c = CortexTrainingClient._from_pat(
            host="x.test",
            pat="tok-xyz",
            database="DB",
            schema="SCH",
            verify_ssl=False,
        )
        assert c.base_url == "https://x.test"
        assert c._session.headers["Authorization"] == "Bearer tok-xyz"
        assert c._session.headers["X-Snowflake-Authorization-Token-Type"] == "PROGRAMMATIC_ACCESS_TOKEN"
        assert c._session.verify is False
        assert c._metric_emitter is not None
        assert c._metric_emitter.timeout == 3.0

    def test_from_pat_uses_dashes_in_the_rest_host(self):
        c = CortexTrainingClient._from_pat(
            host="org-my_acct.snowflakecomputing.com", pat="p", database="DB", schema="SCH"
        )
        assert c.base_url == "https://org-my-acct.snowflakecomputing.com"
        # The Connector accepts either form, so artifact connections keep the host as given.
        assert c._artifact_connection_config["host"] == "org-my_acct.snowflakecomputing.com"

    def test_from_pat_skips_telemetry_when_disabled(self, monkeypatch):
        monkeypatch.setenv(nc.DISABLE_TELEMETRY_ENV, "1")
        c = CortexTrainingClient._from_pat(
            host="x.test",
            pat="tok-xyz",
            database="DB",
            schema="SCH",
        )
        assert c._metric_emitter is None
        assert c.emit_metric("event") is None

    def test_from_pat_propagates_telemetry_timeout(self):
        c = CortexTrainingClient._from_pat(
            host="x.test",
            pat="tok-xyz",
            database="DB",
            schema="SCH",
            telemetry_timeout=1.5,
        )
        assert c._metric_emitter.timeout == 1.5
        assert c._metric_emitter.token_provider.timeout == 1.5

    def test_from_connection_name_uses_connector_context(self, monkeypatch):
        auth = MagicMock()
        auth.base_url = "https://profile.example:8443"
        auth.database = "PROFILE_DB"
        auth.schema = "PROFILE_SCHEMA"
        auth.open_artifact_connection = MagicMock()
        auth_type = MagicMock(return_value=auth)
        monkeypatch.setattr(nc, "SnowflakeProfileAuth", auth_type)

        c = CortexTrainingClient._from_profile("training-profile")

        auth_type.assert_called_once_with("training-profile")
        assert c.base_url == "https://profile.example:8443"
        assert c.database == "PROFILE_DB"
        assert c.schema == "PROFILE_SCHEMA"
        assert c._auth_provider is auth
        assert c._artifact_connection_factory is auth.open_artifact_connection
        assert isinstance(
            c._metric_emitter.token_provider,
            nc.SnowflakeTelemetryTokenProvider,
        )
        assert c._metric_emitter.token_provider._auth is auth

        c.close()
        auth.close.assert_called_once()

    def test_from_connection_name_allows_routing_overrides(self, monkeypatch):
        auth = MagicMock()
        auth.base_url = "https://profile.example"
        auth.database = "PROFILE_DB"
        auth.schema = "PROFILE_SCHEMA"
        monkeypatch.setattr(nc, "SnowflakeProfileAuth", MagicMock(return_value=auth))

        c = CortexTrainingClient._from_profile(
            "training-profile", database="OVERRIDE_DB", schema="OVERRIDE_SCHEMA"
        )

        assert c.database == "OVERRIDE_DB"
        assert c.schema == "OVERRIDE_SCHEMA"

    def test_from_connection_name_requires_database(self, monkeypatch):
        auth = MagicMock()
        auth.base_url = "https://profile.example"
        auth.database = None
        auth.schema = None
        monkeypatch.setattr(nc, "SnowflakeProfileAuth", MagicMock(return_value=auth))

        with pytest.raises(ValueError, match="must set database"):
            CortexTrainingClient._from_profile("training-profile")

        auth.close.assert_called_once()

    def test_profile_client_sends_live_session_token(self):
        c = _make_client(get_json={"job_id": "j1"})
        auth = MagicMock()
        auth.get_token.return_value = "session-token"
        c._auth_provider = auth

        assert c.get_job("j1") == {"job_id": "j1"}

        c._session.get.assert_called_once_with(
            f"{c._prefix}/j1",
            headers={"Authorization": 'Snowflake Token="session-token"'},
        )

    def test_profile_client_refreshes_and_replays_once_on_session_expiry(self):
        c = _make_client()
        expired = _make_error_response({"code": "390112"}, status_code=401)
        succeeded = _make_response({"job_id": "j1"})
        c._session.get.side_effect = [expired, succeeded]
        auth = MagicMock()
        auth.get_token.side_effect = ["old-token", "new-token"]
        c._auth_provider = auth

        assert c.get_job("j1") == {"job_id": "j1"}

        auth.refresh.assert_called_once_with("old-token")
        assert c._session.get.call_args_list[0].kwargs["headers"] == {
            "Authorization": 'Snowflake Token="old-token"'
        }
        assert c._session.get.call_args_list[1].kwargs["headers"] == {
            "Authorization": 'Snowflake Token="new-token"'
        }

    def test_profile_client_does_not_refresh_structured_non_expiry_401(self):
        c = _make_client()
        rejected = _make_error_response({"code": "390100"}, status_code=401)
        c._session.get.return_value = rejected
        auth = MagicMock()
        auth.get_token.return_value = "session-token"
        c._auth_provider = auth

        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.get_job("j1")

        auth.refresh.assert_not_called()

    def test_emit_metric_is_noop_without_pat(self):
        c = CortexTrainingClient(base_url="http://x.test", database="DB", schema="SCH")
        assert c.emit_metric("event") is None

    def test_emit_metric_swallows_unexpected_emitter_errors(self):
        c = CortexTrainingClient(base_url="http://x.test", database="DB", schema="SCH")
        c._metric_emitter = MagicMock()
        c._metric_emitter.emit.side_effect = RuntimeError("boom")

        assert c.emit_metric("event", 2, attributes={"operation": "test"}) is None

    def test_close_releases_metric_emitter_and_http_session(self):
        c = _make_client()
        c._metric_emitter = MagicMock()
        c.close()
        c._metric_emitter.close.assert_called_once()
        c._session.close.assert_called_once()

    def test_close_releases_profile_auth(self):
        c = _make_client()
        c._auth_provider = MagicMock()

        c.close()

        c._auth_provider.close.assert_called_once()

    def test_telemetry_close_error_does_not_skip_http_session_close(self):
        c = _make_client()
        c._metric_emitter = MagicMock()
        c._metric_emitter.close.side_effect = RuntimeError("telemetry close failed")
        c.close()
        c._session.close.assert_called_once()

    def test_no_legacy_auth_methods(self):
        # Production client only supports PAT.
        assert not hasattr(CortexTrainingClient, "from_snowflake_login")
        # And no legacy builder helpers.
        assert not hasattr(CortexTrainingClient, "build_training_sub_job")
        assert not hasattr(CortexTrainingClient, "build_sampling_sub_job")
        assert not hasattr(CortexTrainingClient, "create_training_engine")


class TestOperationMetrics:
    TRACKED_METHODS = {
        "create_job",
        "create_job_from_body",
        "wait_for_job",
        "get_job",
        "list_jobs",
        "list_models",
        "cancel_job",
        "get_capacity",
        "get_capacity_breakdown",
        "get_experiment_run",
        "fetch_execution_logs",
        "download_execution_logs",
        "download_stdout_logs",
        "download_metrics",
        "download_training_metrics",
        "forward_backward",
        "generate",
        "generate_stream",
        "step",
        "save",
        "load",
        "forward",
        "weight_sync",
        "bootstrap_router_replay",
        "router_replay_discard",
        "reset_prefix_cache",
        "poll_request",
        "get_request_status",
        "cancel_request",
        "list_checkpoints",
        "export_checkpoint",
        "delete_checkpoint",
        "tail_logs",
    }

    def test_essential_operations_are_tracked(self):
        for method_name in self.TRACKED_METHODS:
            assert hasattr(getattr(CortexTrainingClient, method_name), "__wrapped__")

    def test_success_metric_is_omitted_by_default(self, monkeypatch):
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        c = _make_client(post_json={"request_id": "r1"})
        c._metric_emitter = MagicMock()

        assert c.step("job-1") == "r1"
        c._metric_emitter.emit.assert_not_called()

    def test_success_metric_has_operation_job_and_duration(self, monkeypatch):
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "1")
        c = _make_client(post_json={"request_id": "r1"})
        c._metric_emitter = MagicMock()

        assert c.step("job-1", learning_rate=1e-4) == "r1"

        c._metric_emitter.emit.assert_called_once()
        operation, value = c._metric_emitter.emit.call_args.args
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert operation == "step"
        assert value["success"] is True
        assert value["duration_ms"] >= 0
        assert value["attempt_count"] == 1
        assert value["request_count"] == 1
        assert value["retry_count"] == 0
        assert attributes == {"client.surface": "sdk", "job_id": "job-1", "request_id": "r1"}

    def test_client_validation_error_metric_preserves_original_error(self):
        c = _make_client()
        c._metric_emitter = MagicMock()

        with pytest.raises(ValueError, match="checkpoint_type"):
            c.save("job-1", checkpoint_type="invalid")

        operation, value = c._metric_emitter.emit.call_args.args
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert operation == "save"
        assert value["success"] is False
        assert value["attempt_count"] == 0
        assert value["request_count"] == 0
        assert value["retry_count"] == 0
        assert "checkpoint_type" in value["error_message"]
        assert attributes == {
            "client.surface": "sdk",
            "job_id": "job-1",
            "error.type": "ValueError",
        }
        assert c._operation_metric_state.active == set()

    def test_http_error_metric_has_status_code_and_request_id(self):
        c = _make_client()
        c._metric_emitter = MagicMock()
        response = _make_error_response({"code": "CORTEX_TRAINING_BAD_REQUEST"}, 400)
        response.headers["x-snowflake-request-id"] = "sf-request-1"
        c._session.post.return_value = response

        with pytest.raises(
            nc.requests.exceptions.HTTPError,
            match=r"^\[snowflake request id: sf-request-1\] ",
        ):
            c.step("job-1")

        _, value = c._metric_emitter.emit.call_args.args
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert value["success"] is False
        assert value["attempt_count"] == 1
        assert value["request_count"] == 1
        assert value["retry_count"] == 0
        assert "snowflake request id: sf-request-1" in value["error_message"]
        assert attributes == {
            "client.surface": "sdk",
            "job_id": "job-1",
            "error.type": "HTTPError",
            "error.http_status": 400,
            "error.snowflake_request_id": "sf-request-1",
            "snowflake.request_id": "sf-request-1",
            "snowflake.request_ids": ["sf-request-1"],
            "error.code": "CORTEX_TRAINING_BAD_REQUEST",
        }

    def test_http_error_message_carries_request_id_without_telemetry(self, caplog):
        c = _make_client()
        c._metric_emitter = None
        response = _make_error_response({"code": "CORTEX_TRAINING_BAD_REQUEST"}, 400)
        response.headers["x-snowflake-request-id"] = "sf-request-1"
        c._session.post.return_value = response

        with caplog.at_level("WARNING", logger=nc.logger.name):
            with pytest.raises(
                nc.requests.exceptions.HTTPError,
                match=r"^\[snowflake request id: sf-request-1\] 400 Client Error",
            ):
                c.step("job-1")

        # The id is already in the raised error, so no duplicate warning.
        assert not [r for r in caplog.records if "last snowflake request id" in r.message]
        assert c._operation_metric_state.active == set()
        assert c._operation_metric_state.snowflake_request_ids == []

    def test_non_http_failure_logs_last_request_id_from_history(
        self, monkeypatch, caplog
    ):
        c = _make_client()
        c._metric_emitter = MagicMock()
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        responses = []
        for index, status in enumerate(("running", "failed"), start=1):
            response = _make_error_response({"status": status, "error": "oom"}, 200)
            response.headers["x-snowflake-request-id"] = f"sf-poll-{index}"
            responses.append(response)
        c._session.get.side_effect = responses

        with caplog.at_level("WARNING", logger=nc.logger.name):
            with pytest.raises(RuntimeError, match="oom"):
                c.poll_request("job-1", "r1")

        assert any(
            "[last snowflake request id: sf-poll-2] poll_request failed" in r.getMessage()
            for r in caplog.records
        )
        operation, value = c._metric_emitter.emit.call_args.args
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert operation == "poll_request"
        assert value["success"] is False
        assert attributes["error.type"] == "RuntimeError"
        assert "error.snowflake_request_id" not in attributes
        assert attributes["snowflake.request_ids"] == ["sf-poll-1", "sf-poll-2"]

    def test_non_http_failure_logs_request_id_without_telemetry(
        self, monkeypatch, caplog
    ):
        c = _make_client()
        c._metric_emitter = None
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        response = _make_error_response({"status": "failed", "error": "oom"}, 200)
        response.headers["x-snowflake-request-id"] = "sf-poll-1"
        c._session.get.return_value = response

        with caplog.at_level("WARNING", logger=nc.logger.name):
            with pytest.raises(RuntimeError, match="oom"):
                c.poll_request("job-1", "r1")

        assert any(
            "[last snowflake request id: sf-poll-1] poll_request failed" in r.getMessage()
            for r in caplog.records
        )

    def test_failure_before_any_request_logs_no_request_id(self, caplog):
        c = _make_client()

        with caplog.at_level("WARNING", logger=nc.logger.name):
            with pytest.raises(ValueError, match="checkpoint_type"):
                c.save("job-1", checkpoint_type="invalid")

        assert not [r for r in caplog.records if "snowflake request id" in r.message]

    def test_http_error_without_header_does_not_borrow_an_earlier_id(
        self, monkeypatch, caplog
    ):
        c = _make_client()
        c.max_retries = 0
        c._metric_emitter = MagicMock()
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        running = _make_error_response({"status": "running"}, 200)
        running.headers["x-snowflake-request-id"] = "sf-prev"
        c._session.get.side_effect = [running, _make_error_response({}, 502)]

        with caplog.at_level("WARNING", logger=nc.logger.name):
            with pytest.raises(nc.requests.exceptions.HTTPError) as excinfo:
                c.poll_request("job-1", "r1")

        assert "sf-prev" not in str(excinfo.value)
        assert not [r for r in caplog.records if "sf-prev" in r.getMessage()]
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert attributes["error.http_status"] == 502
        assert "error.snowflake_request_id" not in attributes
        assert attributes["snowflake.request_ids"] == ["sf-prev"]

    def test_validation_error_after_lookup_is_not_blamed_on_the_lookup(
        self, caplog
    ):
        c = _make_client(
            post_json={"request_id": "g1"},
            get_json={
                "sub_jobs": [
                    {"job_type": "sampling", "inference_config": {"max_seq_len": 4.0, "n_gpus": 1.0}}
                ]
            },
        )
        c._session.get.return_value.headers = {
            "x-snowflake-request-id": "sf-getjob"
        }
        c._metric_emitter = MagicMock()

        with caplog.at_level("WARNING", logger=nc.logger.name):
            with pytest.raises(ValueError, match="prompt at index 0 has 5 tokens"):
                c.generate("j1", prompts=[[1, 2, 3, 4, 5]])

        assert not [r for r in caplog.records if "sf-getjob" in r.getMessage()]
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert attributes["error.type"] == "ValueError"
        assert "error.snowflake_request_id" not in attributes
        # Recorded only as history, never as the failing request.
        assert attributes["snowflake.request_ids"] == ["sf-getjob"]

    def test_retried_attempts_record_every_request_id(self, monkeypatch):
        c = _make_client()
        c.max_retries = 2
        c._metric_emitter = MagicMock()
        monkeypatch.setattr(
            nc,
            "wait_exponential_jitter",
            lambda **kwargs: lambda retry_state: 0,
        )
        responses = []
        for index in range(1, 4):
            response = _make_error_response({"code": "TEMPORARY"}, 503)
            response.headers["x-snowflake-request-id"] = f"sf-{index}"
            responses.append(response)
        c._session.post.side_effect = responses

        with pytest.raises(nc.requests.exceptions.HTTPError) as excinfo:
            c.step("job-1")

        message = str(excinfo.value)
        assert message.startswith("[snowflake request id: sf-3] ")
        assert message.count("snowflake request id") == 1
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert attributes["error.snowflake_request_id"] == "sf-3"
        assert attributes["snowflake.request_ids"] == ["sf-1", "sf-2", "sf-3"]

    def test_nested_operation_accumulates_ids_with_one_warning(
        self, monkeypatch, caplog
    ):
        c = _make_client()
        c._metric_emitter = MagicMock()
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        pending = _make_error_response({"status": "pending"}, 200)
        pending.headers["x-snowflake-request-id"] = "sf-job-1"
        failed = _make_error_response({"status": "failed", "reason": "boom"}, 200)
        failed.headers["x-snowflake-request-id"] = "sf-job-2"
        c._session.get.side_effect = [pending, failed]

        with caplog.at_level("WARNING", logger=nc.logger.name):
            with pytest.raises(RuntimeError, match="boom"):
                c.wait_for_job("job-1")

        warnings = [r.getMessage() for r in caplog.records if "last snowflake request id" in r.getMessage()]
        assert warnings == [
            "[last snowflake request id: sf-job-2] wait_for_job failed: "
            "Job job-1 reached terminal state 'failed': boom"
        ]
        c._metric_emitter.emit.assert_called_once()
        operation, _ = c._metric_emitter.emit.call_args.args
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert operation == "wait_for_job"
        assert attributes["snowflake.request_ids"] == ["sf-job-1", "sf-job-2"]

    def test_chunk_group_error_message_carries_request_id(self, caplog):
        c = _make_client()
        response = _make_error_response(
            {
                "code": "517604",
                "message": (
                    "Cortex training forwardBackward failed: zone returned 409: "
                    '{"detail":{"message":"request chunk group is missing chunks",'
                    '"chunk_group_id":"group-a","missing_chunks":[0,1]}}'
                ),
            }
        )
        response.headers["x-snowflake-request-id"] = "sf-chunk"
        c._session.post.return_value = response

        with caplog.at_level("WARNING", logger=nc.logger.name):
            with pytest.raises(nc.ChunkGroupConflictError) as excinfo:
                c.forward_backward("j1", b"small")

        assert str(excinfo.value).startswith("[snowflake request id: sf-chunk] ")
        assert not [r for r in caplog.records if "last snowflake request id" in r.getMessage()]

    def test_redirect_error_message_carries_request_id(self):
        c = _make_client()
        response = _make_error_response({}, 302)
        response.headers["x-snowflake-request-id"] = "sf-redirect"
        c._session.post.return_value = response

        with pytest.raises(
            nc.requests.exceptions.HTTPError,
            match=r"^\[snowflake request id: sf-redirect\] Unexpected redirect 302 ",
        ):
            c.step("job-1")

    def test_untracked_send_still_tags_http_errors(self):
        c = _make_client()
        response = _make_error_response({}, 500)
        response.headers["x-snowflake-request-id"] = "sf-raw"
        c._session.get.return_value = response
        c.max_retries = 0

        with pytest.raises(
            nc.requests.exceptions.HTTPError,
            match=r"^\[snowflake request id: sf-raw\] ",
        ):
            c._send("GET", "http://test.local/raw")
        assert getattr(c._operation_metric_state, "active", set()) == set()

    def test_success_event_records_request_id_history(self, monkeypatch):
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "1")
        c = _make_client(post_json={"request_id": "r1"})
        c._session.post.return_value.headers = {"x-snowflake-request-id": "sf-ok"}
        c._metric_emitter = MagicMock()

        assert c.step("job-1") == "r1"

        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert "error.snowflake_request_id" not in attributes
        assert attributes["snowflake.request_ids"] == ["sf-ok"]

    def test_request_id_survives_error_message_truncation(self):
        c = _make_client()
        c._metric_emitter = MagicMock()
        response = _make_error_response({"code": "BAD"}, 400)
        response.url = "http://test.local/" + "x" * 1000
        response.headers["x-snowflake-request-id"] = "sf-long"
        c._session.post.return_value = response

        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.step("job-1")

        _, value = c._metric_emitter.emit.call_args.args
        assert len(value["error_message"]) == nc._METRIC_ERROR_MESSAGE_LIMIT
        assert value["error_message"].startswith("[snowflake request id: sf-long] ")

    def test_undecodable_server_payload_logs_last_request_id(
        self, monkeypatch, caplog
    ):
        c = _make_client()
        c._metric_emitter = MagicMock()
        response = _make_error_response({"status": "done"}, 200)
        response.headers["x-snowflake-request-id"] = "sf-bad-payload"
        c._session.get.return_value = response

        def bad_payload(_status):
            raise nc.wire.WireError("not a valid DSSST1 safetensors payload")

        monkeypatch.setattr(c, "_decode_stream_result_events", bad_payload)

        with caplog.at_level("WARNING", logger=nc.logger.name):
            with pytest.raises(nc.wire.WireError):
                c.poll_request("job-1", "r1")

        assert any(
            "[last snowflake request id: sf-bad-payload] poll_request failed"
            in r.getMessage()
            for r in caplog.records
        )
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert attributes["error.type"] == "WireError"
        assert attributes["snowflake.request_ids"] == ["sf-bad-payload"]

    @pytest.mark.parametrize("status", ["done", "cancelled"])
    def test_wait_for_job_non_failed_terminal_state_logs_nothing(
        self, monkeypatch, caplog, status
    ):
        c = _make_client()
        c._metric_emitter = MagicMock()
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        response = _make_error_response({"status": status, "reason": "r"}, 200)
        response.headers["x-snowflake-request-id"] = "sf-job"
        c._session.get.return_value = response

        with caplog.at_level("WARNING", logger=nc.logger.name):
            with pytest.raises(RuntimeError, match=f"terminal state '{status}'"):
                c.wait_for_job("job-1")

        assert not [
            r for r in caplog.records if "last snowflake request id" in r.getMessage()
        ]
        # Telemetry keeps the history even when nothing is logged.
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert attributes["snowflake.request_ids"] == ["sf-job"]

    def test_poll_request_cancelled_logs_nothing(self, monkeypatch, caplog):
        c = _make_client()
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        response = _make_error_response({"status": "cancelled"}, 200)
        response.headers["x-snowflake-request-id"] = "sf-poll"
        c._session.get.return_value = response

        with caplog.at_level("WARNING", logger=nc.logger.name):
            with pytest.raises(RuntimeError, match="cancelled"):
                c.poll_request("job-1", "r1")

        assert not [
            r for r in caplog.records if "last snowflake request id" in r.getMessage()
        ]

    def test_poll_request_timeout_logs_last_request_id(self, monkeypatch, caplog):
        c = _make_client()
        c.poll_interval = 0.01
        c.poll_max_interval = 0.01
        c.poll_timeout = 0.05
        response = _make_error_response({"status": "running"}, 200)
        response.headers["x-snowflake-request-id"] = "sf-running"
        c._session.get.return_value = response

        with caplog.at_level("WARNING", logger=nc.logger.name):
            with pytest.raises(TimeoutError):
                c.poll_request("job-1", "r1")

        assert any(
            "[last snowflake request id: sf-running] poll_request failed"
            in r.getMessage()
            for r in caplog.records
        )

    def test_session_expired_401_request_id_is_recorded(self, monkeypatch):
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "1")
        c = _make_client()
        c._metric_emitter = MagicMock()
        expired = _make_error_response({"code": "390112"}, status_code=401)
        expired.headers["x-snowflake-request-id"] = "sf-401"
        succeeded = _make_response({"job_id": "j1"})
        succeeded.headers = {"x-snowflake-request-id": "sf-ok"}
        c._session.get.side_effect = [expired, succeeded]
        auth = MagicMock()
        auth.get_token.side_effect = ["old-token", "new-token"]
        c._auth_provider = auth

        assert c.get_job("j1") == {"job_id": "j1"}

        auth.refresh.assert_called_once_with("old-token")
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert attributes["snowflake.request_ids"] == ["sf-401", "sf-ok"]

    def test_tracked_request_ids_are_bounded(self, monkeypatch):
        c = _make_client()
        c._metric_emitter = MagicMock()
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        total = nc._MAX_TRACKED_SNOWFLAKE_REQUEST_IDS + 5
        responses = []
        for index in range(1, total + 1):
            status = "failed" if index == total else "running"
            response = _make_error_response({"status": status, "error": "x"}, 200)
            response.headers["x-snowflake-request-id"] = f"sf-{index}"
            responses.append(response)
        c._session.get.side_effect = responses

        with pytest.raises(RuntimeError):
            c.poll_request("job-1", "r1")

        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert attributes["snowflake.request_ids"] == [
            f"sf-{index}" for index in range(6, total + 1)
        ]

    def test_retry_attempts_are_summarized_in_final_metric(self, monkeypatch):
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "1")
        c = _make_client()
        c.max_retries = 1
        c._metric_emitter = MagicMock()
        c._session.post.side_effect = [
            _make_error_response({"code": "TEMPORARY"}, 503),
            _make_response({"request_id": "r1"}),
        ]
        monkeypatch.setattr(
            nc,
            "wait_exponential_jitter",
            lambda **kwargs: lambda retry_state: 0,
        )

        assert c.step("job-1") == "r1"

        operation, value = c._metric_emitter.emit.call_args.args
        assert operation == "step"
        assert value["success"] is True
        assert value["request_count"] == 1
        assert value["attempt_count"] == 2
        assert value["retry_count"] == 1

    def test_create_job_delegation_emits_only_one_metric_with_returned_job_id(
        self, monkeypatch
    ):
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "1")
        c = _make_client(post_json={"job_id": "server-job"})
        c._metric_emitter = MagicMock()
        sub_job = SubJobConfig.sampling_job(
            model_name="gpt2",
            max_seq_len=128,
            n_gpus=1,
        )

        assert c.create_job([sub_job]) == "server-job"

        c._metric_emitter.emit.assert_called_once()
        operation, value = c._metric_emitter.emit.call_args.args
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert operation == "create_job"
        assert value["success"] is True
        assert attributes == {"client.surface": "sdk", "job_id": "server-job"}

    def test_sub_job_identity_is_included(self, monkeypatch):
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "1")
        c = _make_client(post_json={"request_id": "r1"})
        c._metric_emitter = MagicMock()

        c.weight_sync(
            "job-1",
            source_sub_job_id="job-1:training:0",
            target_sub_job_ids=["job-1:sampling:0"],
        )

        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert attributes == {
            "client.surface": "sdk",
            "job_id": "job-1",
            "source_sub_job_id": "job-1:training:0",
            "target_sub_job_ids": ["job-1:sampling:0"],
            "request_id": "r1",
        }

    def test_operation_payload_is_never_copied_into_metric_attributes(
        self, monkeypatch
    ):
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "1")
        c = _make_client(post_json={"request_id": "r1"})
        c._metric_emitter = MagicMock()

        c.forward_backward("job-1", b"customer-payload")

        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert attributes == {"client.surface": "sdk", "job_id": "job-1", "request_id": "r1"}
        assert "customer-payload" not in repr(c._metric_emitter.emit.call_args)

    def test_error_message_redacts_obvious_secrets(self):
        error = RuntimeError(
            'request failed Authorization: Bearer secret-pat password="secret"'
        )
        message = nc._safe_metric_error_message(error)
        assert "secret-pat" not in message
        assert '"secret"' not in message
        assert message.count("<redacted>") == 2

    @pytest.mark.parametrize(
        "message",
        [
            '{"token": "LEAKED-PAT"}',
            '{"password": "hunter2"}',
            "apikey=LEAKED",
            "api_key: LEAKED",
            "pat=LEAKED",
            "private_key='LEAKED'",
            "refresh_token=LEAKED",
            "Authorization: Snowflake Token=\"LEAKED\"",
        ],
    )
    def test_error_message_redacts_json_and_credential_aliases(self, message):
        redacted = nc._safe_metric_error_message(RuntimeError(message))
        assert "LEAKED" not in redacted
        assert "hunter2" not in redacted
        assert "<redacted>" in redacted

    def test_missing_metric_state_never_changes_operation_behavior(self):
        c = _make_client(post_json={"request_id": "r1"})
        del c._operation_metric_state
        assert c.step("job-1") == "r1"

    def test_disabled_telemetry_skips_error_metadata_construction(self, monkeypatch):
        c = _make_client()
        monkeypatch.setattr(
            nc,
            "_safe_metric_error_message",
            MagicMock(side_effect=AssertionError("must not run")),
        )
        with pytest.raises(ValueError, match="checkpoint_type"):
            c.save("job-1", checkpoint_type="invalid")

    def test_nested_tracked_operations_emit_only_outer_outcome(self, monkeypatch):
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "1")

        class NestedClient(CortexTrainingClient):
            @nc._track_operation("inner")
            def inner(self):
                return "inner-result"

            @nc._track_operation("outer")
            def outer(self):
                return self.inner()

        c = NestedClient(base_url="http://test.local", database="DB", schema="SCH")
        c._metric_emitter = MagicMock()
        assert c.outer() == "inner-result"
        c._metric_emitter.emit.assert_called_once()
        assert c._metric_emitter.emit.call_args.args[0] == "outer"

    def test_polling_suppresses_nested_status_read_metrics(self, monkeypatch):
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "1")
        c = _make_client(get_json={"status": "running"})
        c._metric_emitter = MagicMock()
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)

        assert c.wait_for_job("job-1")["status"] == "running"

        c._metric_emitter.emit.assert_called_once()
        assert c._metric_emitter.emit.call_args.args[0] == "wait_for_job"


class TestTelemetrySurface:
    """Every metric says which program made the call: sdk, cli or tui."""

    @pytest.fixture(autouse=True)
    def _default_success_policy(self, monkeypatch):
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)

    def _client(self, surface=None, **responses):
        c = _make_client(**responses)
        c._metric_emitter = MagicMock()
        if surface is not None:
            c._telemetry_surface = surface
        return c

    def _attributes(self, c):
        return [call.kwargs["attributes"] for call in c._metric_emitter.emit.call_args_list]

    def test_a_client_is_sdk_until_an_entry_point_says_otherwise(self):
        c = self._client()
        c._session.get.return_value = _make_error_response({"message": "bad"}, 400)

        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.get_job("job-1")

        assert self._attributes(c)[0][nc.SURFACE_ATTRIBUTE] == nc.SDK_SURFACE == "sdk"

    @pytest.mark.parametrize(
        ("operation", "call"),
        [
            ("get_job", lambda c: c.get_job("job-1")),
            ("list_jobs", lambda c: c.list_jobs()),
            ("list_jobs_page", lambda c: c.list_jobs_page()),
            ("list_checkpoints", lambda c: c.list_checkpoints("job-1")),
        ],
    )
    def test_a_tui_client_reports_each_success_with_its_timing(self, operation, call):
        c = self._client(nc.TUI_SURFACE, get_json={"job_id": "job-1", "jobs": [], "checkpoints": []})

        call(c)

        c._metric_emitter.emit.assert_called_once()
        name, value = c._metric_emitter.emit.call_args.args
        assert name == operation
        assert value["success"] is True and value["duration_ms"] >= 0
        assert self._attributes(c)[0][nc.SURFACE_ATTRIBUTE] == "tui"

    @pytest.mark.parametrize("surface", [nc.SDK_SURFACE, nc.CLI_SURFACE])
    def test_the_same_calls_from_sdk_or_cli_code_stay_failure_only(self, surface):
        c = self._client(surface, get_json={"job_id": "job-1"})

        c.get_job("job-1")

        c._metric_emitter.emit.assert_not_called()

    # The operations whose successes every program sends by default, as the Client
    # metrics section of the Python SDK reference names them.
    DEFAULT_SUCCESS_OPERATIONS = {
        "download_stdout_logs",
        "download_metrics",
        "download_training_metrics",
        "download_execution_logs",
        "fetch_execution_logs",
    }

    @pytest.mark.parametrize("surface", [nc.SDK_SURFACE, nc.CLI_SURFACE, nc.TUI_SURFACE])
    def test_by_default_only_the_downloads_and_the_tui_send_successes(self, surface):
        # Every operation a metric can name: create_job_from_body reports as create_job, the
        # job-list pages as iter_jobs and list_jobs_page, and the TUI's file download on its own.
        operations = (TestOperationMetrics.TRACKED_METHODS - {"create_job_from_body"}) | {
            "iter_jobs", "list_jobs_page", "download_checkpoint"}

        sent = {operation for operation in operations if nc._success_telemetry_enabled(operation, surface)}

        assert sent == (operations if surface == nc.TUI_SURFACE else self.DEFAULT_SUCCESS_OPERATIONS)

    def test_an_explicit_setting_still_wins_on_the_tui(self, monkeypatch):
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "0")
        c = self._client(nc.TUI_SURFACE, get_json={"job_id": "job-1"})

        c.get_job("job-1")
        c._metric_emitter.emit.assert_not_called()

        c._session.get.return_value = _make_error_response({"message": "bad"}, 400)
        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.get_job("job-1")
        # Failures are always sent, and still say where they came from.
        assert [attributes[nc.SURFACE_ATTRIBUTE] for attributes in self._attributes(c)] == ["tui"]

    def test_a_tui_client_s_per_job_clients_report_as_the_tui(self):
        c = self._client(nc.TUI_SURFACE, get_json={"job_id": "j1"})

        c.for_job({"job_id": "j1", "database_name": "ML", "schema_name": "EXP"}).get_job("j1")

        assert self._attributes(c) == [{"client.surface": "tui", "job_id": "j1"}]

    def test_custom_metrics_carry_the_surface_unless_they_name_one(self):
        c = self._client(nc.CLI_SURFACE)

        c.emit_metric("custom", {"n": 1}, attributes={"job_id": "job-1"})
        c.emit_metric("custom", {"n": 2}, attributes={nc.SURFACE_ATTRIBUTE: "notebook"})
        c.emit_metric("custom")

        assert self._attributes(c) == [
            {"client.surface": "cli", "job_id": "job-1"},
            {"client.surface": "notebook"},
            {"client.surface": "cli"},
        ]

    def test_with_telemetry_off_a_tui_client_has_nothing_to_send_with(self, monkeypatch):
        monkeypatch.setenv(nc.DISABLE_TELEMETRY_ENV, "1")
        monkeypatch.setattr(nc, "OtlpMetricEmitter", MagicMock(side_effect=AssertionError("built an emitter")))
        c = CortexTrainingClient._from_pat(host="account.test", pat="pat", database="DB", schema="SCH")
        c._telemetry_surface = nc.TUI_SURFACE

        c.emit_metric("custom", {"n": 1})

        assert c._metric_emitter is None


# ─── CortexTrainingClient — create_job ────────────────────────────────────────


class TestCreateJob:
    def test_empty_list_raises(self):
        c = _make_client()
        with pytest.raises(ValueError, match="non-empty"):
            c.create_job(sub_jobs=[])

    def test_validates_each_sub_job_before_post(self):
        # Pydantic validates on construction, so an invalid SubJobConfig
        # cannot even be created — no HTTP call is possible.
        with pytest.raises(ValidationError):
            SubJobConfig(job_type=JobType.TRAINING, model_name="gpt2")  # no training block

    def test_posts_in_flight_yaml_shape(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
        )
        job_id = c.create_job(sub_jobs=[sub])
        assert job_id == "srv-1"
        url, kwargs = c._session.post.call_args
        assert url[0] == c._prefix
        assert kwargs["json"] == {
            "sub_job_configs": [
                {
                    "job_type": "training",
                    "model_name": "gpt2",
                    "training_config": {
                        "optimizer": {"name": "adamw"},
                        "max_seq_len": 128,
                        "train_batch_size": 1,
                        "n_gpus": 2,
                    },
                }
            ],
        }

    def test_includes_job_id_when_given(self):
        c = _make_client(post_json={"job_id": "client-chosen"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub], job_id="client-chosen")
        body = c._session.post.call_args.kwargs["json"]
        assert body["job_id"] == "client-chosen"

    def test_omits_job_id_when_none(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub])
        body = c._session.post.call_args.kwargs["json"]
        assert "job_id" not in body

    def test_includes_experiment_name_when_given(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub], experiment_name="my-experiment")
        body = c._session.post.call_args.kwargs["json"]
        assert body["experiment_name"] == "my-experiment"

    def test_omits_experiment_name_when_none(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub])
        body = c._session.post.call_args.kwargs["json"]
        assert "experiment_name" not in body

    @pytest.mark.parametrize(
        "hardware, wire_value",
        [
            (Hardware.H200, "H200"),
            (Hardware.B200, "B200"),
            (Hardware.B300, "B300"),
            ("B200", "B200"),
        ],
    )
    def test_includes_hardware_when_given(self, hardware, wire_value):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub], hardware=hardware)
        body = c._session.post.call_args.kwargs["json"]
        assert body["hardware"] == wire_value

    # Omitted rather than sent as H200: the server owns the default.
    def test_omits_hardware_when_none(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub])
        body = c._session.post.call_args.kwargs["json"]
        assert "hardware" not in body

    @pytest.mark.parametrize("hardware", ["A10", "h200", "", 200])
    def test_rejects_unknown_hardware(self, hardware):
        c = _make_client()
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        with pytest.raises(ValueError, match="hardware must be one of"):
            c.create_job(sub_jobs=[sub], hardware=hardware)

    # 0 is accepted here; pending_timeout_seconds rejects it.
    @pytest.mark.parametrize(
        "idle_timeout_seconds", [0, 300, 1_800, 604_800]
    )
    def test_includes_valid_idle_timeout_seconds(self, idle_timeout_seconds):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub], idle_timeout_seconds=idle_timeout_seconds)
        body = c._session.post.call_args.kwargs["json"]
        assert body["idle_timeout_seconds"] == idle_timeout_seconds

    def test_omits_idle_timeout_when_none(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub])
        body = c._session.post.call_args.kwargs["json"]
        assert "idle_timeout_seconds" not in body

    @pytest.mark.parametrize(
        "idle_timeout_seconds",
        [-1, 1, 299, 604_801, True, 300.0, "300"],
    )
    def test_rejects_invalid_idle_timeout_seconds(self, idle_timeout_seconds):
        c = _make_client()
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        with pytest.raises(ValueError, match="idle_timeout_seconds"):
            c.create_job(sub_jobs=[sub], idle_timeout_seconds=idle_timeout_seconds)
        c._session.post.assert_not_called()

    @pytest.mark.parametrize(
        "pending_timeout_seconds", [300, 3_600, 86_400, 604_800]
    )
    def test_includes_valid_pending_timeout_seconds(self, pending_timeout_seconds):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub], pending_timeout_seconds=pending_timeout_seconds)
        body = c._session.post.call_args.kwargs["json"]
        assert body["pending_timeout_seconds"] == pending_timeout_seconds

    def test_omits_pending_timeout_when_none(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub])
        body = c._session.post.call_args.kwargs["json"]
        assert "pending_timeout_seconds" not in body

    # 0 is rejected here; idle_timeout_seconds accepts it.
    @pytest.mark.parametrize(
        "pending_timeout_seconds",
        [-1, 0, 1, 299, 604_801, True, 300.0, "300"],
    )
    def test_rejects_invalid_pending_timeout_seconds(self, pending_timeout_seconds):
        c = _make_client()
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        with pytest.raises(ValueError, match="pending_timeout_seconds"):
            c.create_job(
                sub_jobs=[sub], pending_timeout_seconds=pending_timeout_seconds
            )
        c._session.post.assert_not_called()

    def test_supports_multiple_sub_jobs(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        train = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"name": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
        )
        samp = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[train, samp])
        body = c._session.post.call_args.kwargs["json"]
        assert len(body["sub_job_configs"]) == 2
        assert body["sub_job_configs"][0]["job_type"] == "training"
        assert body["sub_job_configs"][1]["job_type"] == "sampling"

    def test_create_job_from_body_posts_raw_body(self):
        c = _make_client(post_json={"job_id": "srv-raw"})
        body = {
            "job_id": "client-raw",
            "sub_job_configs": [
                {
                    "job_type": "sampling",
                    "model_name": "gpt2",
                    "inference_config": {"max_seq_len": 128, "n_gpus": 1},
                }
            ],
        }
        assert c.create_job_from_body(body) == {"job_id": "srv-raw"}
        c._session.post.assert_called_once_with(c._prefix, json=body)

    def test_create_job_from_body_requires_object(self):
        c = _make_client()
        with pytest.raises(ValueError, match="JSON object"):
            c.create_job_from_body([])

    def test_create_job_from_body_requires_sub_job_configs(self):
        c = _make_client()
        with pytest.raises(ValueError, match="sub_job_configs"):
            c.create_job_from_body({"sub_job_configs": []})

    def test_rejects_two_training_sub_jobs_before_post(self):
        c = _make_client()
        training = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
        )
        with pytest.raises(ValueError, match="at most one training sub-job"):
            c.create_job(sub_jobs=[training, training])
        c._session.post.assert_not_called()

    def test_rejects_log_probability_sub_job_before_post(self):
        c = _make_client()
        training = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
        )
        log_prob = SubJobConfig(
            job_type=JobType.LOG_PROBABILITY,
            model_name="gpt2",
            sampling=_ok_inference(),
        )
        with pytest.raises(
            ValueError,
            match=r"sub_jobs\[1\]\.job_type: log_probability sub-jobs are not currently supported",
        ):
            c.create_job(sub_jobs=[training, log_prob])
        c._session.post.assert_not_called()

    @pytest.mark.parametrize("job_type", LOG_PROBABILITY_JOB_TYPES)
    def test_create_job_from_body_rejects_log_probability_before_post(self, job_type):
        c = _make_client()
        body = {
            "sub_job_configs": [
                {
                    "job_type": "sampling",
                    "model_name": "gpt2",
                    "inference_config": {"max_seq_len": 128, "n_gpus": 1},
                },
                {
                    "job_type": job_type,
                    "model_name": "gpt2",
                    "inference_config": {"max_seq_len": 128, "n_gpus": 1},
                },
            ],
        }
        with pytest.raises(
            ValueError,
            match=r"sub_job_configs\[1\]\.job_type: log_probability sub-jobs are not currently supported",
        ):
            c.create_job_from_body(body)
        c._session.post.assert_not_called()

    # Only the two documented aliases are rejected; anything else is left to the
    # server so a newly added schema value is not blocked client-side.
    @pytest.mark.parametrize("job_type", ["log-probability", "logprob", "sampling"])
    def test_create_job_from_body_allows_other_job_types(self, job_type):
        c = _make_client(post_json={"job_id": "srv-1"})
        body = {
            "sub_job_configs": [
                {
                    "job_type": job_type,
                    "model_name": "gpt2",
                    "inference_config": {"max_seq_len": 128, "n_gpus": 1},
                }
            ],
        }
        assert c.create_job_from_body(body) == {"job_id": "srv-1"}

    def test_create_job_from_body_skips_non_dict_sub_job_configs(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        body = {
            "sub_job_configs": [
                "log_probability",
                {
                    "job_type": "sampling",
                    "model_name": "gpt2",
                    "inference_config": {"max_seq_len": 128, "n_gpus": 1},
                },
            ],
        }
        assert c.create_job_from_body(body) == {"job_id": "srv-1"}

    def test_create_job_from_body_rejects_two_training_sub_jobs_before_post(self):
        c = _make_client()
        body = {
            "sub_job_configs": [
                {
                    "job_type": "training",
                    "model_name": "gpt2",
                    "training_config": {"max_seq_len": 128, "train_batch_size": 1, "n_gpus": 2},
                },
                {
                    "job_type": " TRAINING ",
                    "model_name": "gpt2",
                    "training_config": {"max_seq_len": 128, "train_batch_size": 1, "n_gpus": 2},
                },
            ],
        }
        with pytest.raises(ValueError, match="at most one training sub-job"):
            c.create_job_from_body(body)
        c._session.post.assert_not_called()

    @pytest.mark.parametrize(
        "job_type",
        ["training", "TRAINING", "JOB_TYPE_TRAINING", " training "],
    )
    def test_create_job_from_body_normalizes_training_peft_config(self, job_type, monkeypatch):
        normalized_peft = {"peft_type": "Lora"}
        normalize = MagicMock(return_value=normalized_peft)
        monkeypatch.setattr(nc, "normalize_lora_peft_config", normalize)
        c = _make_client(post_json={"job_id": "srv-raw"})
        body = {
            "sub_job_configs": [
                {
                    "job_type": job_type,
                    "model_name": "gpt2",
                    "training_config": {"peft_config": {"r": 8}},
                },
            ],
        }

        assert c.create_job_from_body(body) == {"job_id": "srv-raw"}
        assert body["sub_job_configs"][0]["training_config"]["peft_config"] == normalized_peft
        normalize.assert_called_once()

    def test_create_job_from_body_normalizes_sampling_peft_config(self, monkeypatch):
        normalized_peft = {"peft_type": "Lora"}
        normalize = MagicMock(return_value=normalized_peft)
        monkeypatch.setattr(nc, "normalize_lora_peft_config", normalize)
        c = _make_client(post_json={"job_id": "srv-raw"})
        body = {
            "sub_job_configs": [
                {
                    "job_type": "sampling",
                    "model_name": "gpt2",
                    "inference_config": {"max_seq_len": 128, "n_gpus": 1, "peft_config": {"r": 8}},
                },
            ],
        }

        assert c.create_job_from_body(body) == {"job_id": "srv-raw"}
        assert body["sub_job_configs"][0]["inference_config"]["peft_config"] == normalized_peft
        normalize.assert_called_once()

    def test_create_job_from_body_rewrites_peft_config_in_place(self):
        c = _make_client(post_json={"job_id": "srv-raw"})
        body = {
            "sub_job_configs": [
                {
                    "job_type": "training",
                    "model_name": "gpt2",
                    "training_config": {
                        "optimizer": {"type": "adamw"},
                        "peft_config": {
                            "peft_type": "Lora",
                            "r": 16,
                            "target_modules": ["q_proj"],
                        },
                    },
                },
            ],
        }

        assert c.create_job_from_body(body) == {"job_id": "srv-raw"}
        training_config = body["sub_job_configs"][0]["training_config"]
        assert training_config["peft_config"] == {
            "peft_type": "Lora",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": ["q_proj"],
            "target_parameters": [],
        }
        assert isinstance(training_config["peft_config"]["lora_dropout"], float)
        # The raw path forwards the optimizer untouched.
        assert training_config["optimizer"] == {"type": "adamw"}

    def test_the_api_readme_says_create_job_from_body_rewrites_peft_config_in_place(self):
        readme = (Path(__file__).resolve().parents[1] / "recipes" / "api_quickstart" / "README.md").read_text()
        text = " ".join(readme.split())
        claim = text[text.index("`create_job_from_body` checks the body") :]
        claim = claim[: claim.index(" See ")]
        assert "rewrites each `peft_config` in place into its canonical form" in claim
        assert "unchanged" not in claim

        c = _make_client(post_json={"job_id": "srv-raw"})
        peft = {"peft_type": "Lora", "r": 32, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]}
        body = {
            "hardware": "H200",
            "sub_job_configs": [
                {
                    "job_type": "training",
                    "model_name": "Qwen/Qwen3-8B",
                    "training_config": {"n_gpus": 4, "optimizer": {"lr": 2e-5}, "peft_config": dict(peft)},
                }
            ],
        }
        unchanged = json.loads(json.dumps(body))
        del unchanged["sub_job_configs"][0]["training_config"]["peft_config"]

        c.create_job_from_body(body)

        assert c._session.post.call_args.kwargs["json"] is body
        canonical = dict(peft, task_type="CAUSAL_LM", lora_dropout=0.0, bias="none", target_parameters=[])
        assert body["sub_job_configs"][0]["training_config"].pop("peft_config") == canonical
        assert body == unchanged

    def test_create_job_from_body_rejects_invalid_peft_config_before_post(self):
        c = _make_client()
        body = {
            "sub_job_configs": [
                {
                    "job_type": "training",
                    "model_name": "gpt2",
                    "training_config": {"peft_config": {"peft_type": "LORA"}},
                },
            ],
        }

        with pytest.raises(
            ValueError,
            match=r"sub_job_configs\[0\]\.training_config\.peft_config\.peft_type",
        ):
            c.create_job_from_body(body)
        c._session.post.assert_not_called()

    def test_create_job_from_body_rejects_zero_stage_3_before_post(self):
        c = _make_client()
        body = {
            "sub_job_configs": [
                {
                    "job_type": "training",
                    "model_name": "gpt2",
                    "training_config": {"ds_config": {"zero_optimization": {"stage": 3}}},
                },
            ],
        }

        with pytest.raises(
            ValueError,
            match=r"sub_job_configs\[0\]\.training_config\.ds_config\.zero_optimization\.stage=3 \(ZeRO-3\)",
        ):
            c.create_job_from_body(body)
        c._session.post.assert_not_called()

    @pytest.mark.parametrize(
        ("provider", "sent"),
        [("arctic", "prime_rl"), ("prime_rl", "prime_rl"), ("huggingface", "huggingface")],
    )
    def test_create_job_from_body_sends_arctic_provider_as_prime_rl(self, provider, sent):
        c = _make_client(post_json={"job_id": "srv-raw"})
        body = {
            "sub_job_configs": [
                {
                    "job_type": "training",
                    "model_name": "gpt2",
                    "training_config": {"model_provider": provider},
                },
            ],
        }

        c.create_job_from_body(body)

        posted = c._session.post.call_args.kwargs["json"]
        assert posted["sub_job_configs"][0]["training_config"]["model_provider"] == sent

    def test_create_job_from_body_ignores_absent_peft_config(self):
        c = _make_client(post_json={"job_id": "srv-raw"})
        body = {
            "sub_job_configs": [
                "not-an-object",
                {
                    "job_type": "sampling",
                    "model_name": "gpt2",
                    "inference_config": {"max_seq_len": 128, "n_gpus": 1},
                },
            ],
        }

        assert c.create_job_from_body(body) == {"job_id": "srv-raw"}

    def test_allows_one_training_with_multiple_sampling(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        training = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
        )
        sampling = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        assert c.create_job(sub_jobs=[training, sampling, sampling]) == "srv-1"
        body = c._session.post.call_args.kwargs["json"]
        assert [sj["job_type"] for sj in body["sub_job_configs"]] == [
            "training",
            "sampling",
            "sampling",
        ]

    def test_allows_multiple_sampling_without_training(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        sampling = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        assert c.create_job(sub_jobs=[sampling, sampling]) == "srv-1"
        body = c._session.post.call_args.kwargs["json"]
        assert [sj["job_type"] for sj in body["sub_job_configs"]] == ["sampling", "sampling"]

    def test_create_job_from_body_rejects_debug_without_env(self, monkeypatch):
        monkeypatch.delenv(nc.DEBUG_OPTIONS_ENV, raising=False)
        c = _make_client(post_json={"job_id": "srv"})
        body = {
            "sub_job_configs": [
                {
                    "job_type": "training",
                    "model_name": "gpt2",
                    "training_config": {"n_gpus": 1},
                }
            ],
            "debug": {"job": {"image_tag": "release_internal"}},
        }
        with pytest.raises(ValueError, match=nc.DEBUG_OPTIONS_ENV):
            c.create_job_from_body(body)
        # The gate must short-circuit before any request is sent.
        c._session.post.assert_not_called()

    def test_create_job_from_body_allows_debug_when_env_set(self, monkeypatch):
        monkeypatch.setenv(nc.DEBUG_OPTIONS_ENV, "1")
        c = _make_client(post_json={"job_id": "srv-dbg"})
        body = {
            "sub_job_configs": [
                {
                    "job_type": "training",
                    "model_name": "gpt2",
                    "training_config": {"n_gpus": 1},
                }
            ],
            "debug": {"job": {"image_tag": "release_internal"}},
        }
        assert c.create_job_from_body(body) == {"job_id": "srv-dbg"}
        c._session.post.assert_called_once_with(c._prefix, json=body)

    def test_create_job_from_body_allows_empty_debug_without_env(self, monkeypatch):
        # A falsy/empty debug block carries no directives, so it is not gated.
        monkeypatch.delenv(nc.DEBUG_OPTIONS_ENV, raising=False)
        c = _make_client(post_json={"job_id": "srv"})
        body = {
            "sub_job_configs": [
                {
                    "job_type": "training",
                    "model_name": "gpt2",
                    "training_config": {"n_gpus": 1},
                }
            ],
            "debug": {},
        }
        assert c.create_job_from_body(body) == {"job_id": "srv"}


# ─── CortexTrainingClient — transient HTTP retries ───────────────────────────


class TestTransientHttpRetries:
    def test_307_and_308_are_transient(self):
        for status in (307, 308):
            resp = _make_response(status_code=status)
            exc = nc.requests.exceptions.HTTPError(response=resp)
            assert nc._is_transient(exc)

    def test_302_is_not_transient(self):
        resp = _make_response(status_code=302)
        exc = nc.requests.exceptions.HTTPError(response=resp)
        assert not nc._is_transient(exc)

    @pytest.mark.parametrize("caller_kwargs", [{}, {"allow_redirects": True}])
    def test_session_does_not_follow_redirects(self, monkeypatch, caller_kwargs):
        seen_kwargs = {}

        def fake_request(_session, *_args, **kwargs):
            seen_kwargs.update(kwargs)
            return _make_response()

        monkeypatch.setattr(nc.requests.Session, "request", fake_request)
        session = nc._DefaultTimeoutSession((1.0, 1.0))

        session.get("http://test.local/redirect", **caller_kwargs)

        assert seen_kwargs["allow_redirects"] is False

    def test_send_retries_308_against_original_url(self, monkeypatch):
        from tenacity import wait_none

        monkeypatch.setattr(
            nc, "wait_exponential_jitter", lambda **_kwargs: wait_none()
        )
        c = _make_client()
        c.max_retries = 1
        leaked = nc.requests.Response()
        leaked.status_code = 308
        leaked.url = f"{c._prefix}/job-1/forward-backward"
        leaked.headers["Location"] = "http://unroutable.invalid/forward-backward"
        ok = _make_response({"job_id": "job-1", "status": "running"})
        c._session.post.side_effect = [leaked, ok]

        out = c._send("POST", leaked.url, data=b"payload")

        assert out is ok
        assert c._session.post.call_count == 2
        urls = [call.args[0] for call in c._session.post.call_args_list]
        assert urls == [leaked.url, leaked.url]

    def test_send_rejects_non_transient_redirect_without_retry(self):
        c = _make_client()
        c.max_retries = 1
        redirect = nc.requests.Response()
        redirect.status_code = 302
        redirect.url = f"{c._prefix}/job-1"
        redirect.headers["Location"] = "http://example.test/elsewhere"
        c._session.get.return_value = redirect

        with pytest.raises(
            nc.requests.exceptions.HTTPError, match="Unexpected redirect 302"
        ):
            c._send("GET", redirect.url)

        c._session.get.assert_called_once_with(redirect.url)


# ─── CortexTrainingClient — read & control endpoints ─────────────────────────


class TestReadAndControl:
    def test_normalize_job_status(self):
        assert CortexTrainingClient._normalize_job_status("JOB_STATE_RUNNING") == "running"
        assert CortexTrainingClient._normalize_job_status("running") == "running"
        assert CortexTrainingClient._normalize_job_status("JOB_STATE_FAILED") == "failed"

    def test_normalize_job_status_tolerates_unknown_and_non_string(self):
        # A CP ahead of GS can surface a JobState enum GS doesn't know, which GS
        # renders as a raw integer. Must not crash on int.lower(); an unknown
        # value normalizes to a harmless non-terminal string.
        assert CortexTrainingClient._normalize_job_status(12) == "12"
        assert CortexTrainingClient._normalize_job_status(None) == ""
        assert CortexTrainingClient._normalize_job_status("JOB_STATE_INITIALIZING") == "initializing"

    def test_get_job(self):
        c = _make_client(get_json={"job_id": "j1", "status": "running"})
        out = c.get_job("j1")
        assert out["job_id"] == "j1"
        c._session.get.assert_called_once_with(f"{c._prefix}/j1")

    def test_list_jobs_no_filter(self):
        c = _make_client(get_json={"jobs": [{"job_id": "a"}, {"job_id": "b"}]})
        jobs = c.list_jobs()
        assert [j["job_id"] for j in jobs] == ["a", "b"]
        c._session.get.assert_called_once_with(c._prefix, params={})

    def test_list_jobs_with_status(self):
        c = _make_client(get_json={"jobs": []})
        c.list_jobs(status="running")
        c._session.get.assert_called_once_with(c._prefix, params={"status": "running"})

    def test_list_models_returns_models_from_models_route(self):
        models = [
            {"name": "Qwen/Qwen3-0.6B"},
            {"name": "Qwen/Qwen3-8B"},
        ]
        c = _make_client(get_json={"models": models})

        assert c.list_models() == models
        c._session.get.assert_called_once_with(
            f"{c.base_url}/api/v2/{c.endpoint}/models"
        )

    def test_cancel_job_uses_colon_action(self):
        c = _make_client(post_json={})
        c.cancel_job("j1")
        c._session.post.assert_called_once_with(f"{c._prefix}/j1:cancel")

    def test_get_capacity_reserved_account(self):
        c = _make_client(
            get_json={
                "has_reservation": True,
                "max_total_gpus": 64,
                "reserved_gpus": 64,
                "in_use_gpus": 8,
                "pending_gpus": 16,
                "available_gpus": 40,
            }
        )
        cap = c.get_capacity()
        assert cap == {
            "has_reservation": True,
            "max_total_gpus": 64,
            "reserved_gpus": 64,
            "in_use_gpus": 8,
            "pending_gpus": 16,
            "available_gpus": 40,
        }
        c._session.get.assert_called_once_with(
            f"{c._prefix}/capacity", timeout=nc._CAPACITY_REQUEST_TIMEOUT
        )

    def test_get_capacity_reports_uncapped_ceiling(self):
        # -1 is the uncapped sentinel and must survive as-is: 0 is a real quota
        # of zero, so collapsing the two would report the opposite ceiling.
        c = _make_client(
            get_json={"max_total_gpus": -1, "in_use_gpus": 16, "available_gpus": 96}
        )
        cap = c.get_capacity()
        assert cap["max_total_gpus"] == -1
        assert cap["has_reservation"] is False

    def test_get_capacity_sends_hardware_query(self):
        c = _make_client(get_json={"has_reservation": True, "reserved_gpus": 32})
        c.get_capacity(hardware=Hardware.B200)
        c._session.get.assert_called_once_with(
            f"{c._prefix}/capacity",
            timeout=nc._CAPACITY_REQUEST_TIMEOUT,
            params={"hardware": "B200"},
        )

    def test_get_capacity_uses_shorter_caller_timeout_and_does_not_retry(self):
        c = _make_client()
        c.request_timeout = (2.0, 0.5)
        c._session.get.side_effect = nc.requests.exceptions.ReadTimeout("silent peer")

        with pytest.raises(nc.requests.exceptions.ReadTimeout):
            c.get_capacity()

        c._session.get.assert_called_once_with(
            f"{c._prefix}/capacity", timeout=(2.0, 0.5)
        )

    def test_get_capacity_still_replays_once_for_expired_profile_token(self):
        c = _make_client()
        c._session.get.side_effect = [
            _make_error_response({"code": "390112"}, status_code=401),
            _make_response({"available_gpus": 8}),
        ]
        auth = MagicMock()
        auth.get_token.side_effect = ["old-token", "new-token"]
        c._auth_provider = auth

        assert c.get_capacity()["available_gpus"] == 8

        auth.refresh.assert_called_once_with("old-token")
        assert c._session.get.call_count == 2
        assert c._session.get.call_args_list[0].kwargs == {
            "headers": {"Authorization": 'Snowflake Token="old-token"'},
            "timeout": nc._CAPACITY_REQUEST_TIMEOUT,
        }
        assert c._session.get.call_args_list[1].kwargs == {
            "headers": {"Authorization": 'Snowflake Token="new-token"'},
            "timeout": nc._CAPACITY_REQUEST_TIMEOUT,
        }

    def test_get_capacity_rejects_unknown_hardware(self):
        c = _make_client(get_json={})
        with pytest.raises(ValueError, match="hardware must be one of"):
            c.get_capacity(hardware="A10")
        c._session.get.assert_not_called()

    def test_get_capacity_fills_proto3_omitted_defaults(self):
        # proto3 JSON omits zero/false fields; an account holding nothing under a
        # zero ceiling is `{}`.
        c = _make_client(get_json={})
        cap = c.get_capacity()
        assert cap == {
            "has_reservation": False,
            "max_total_gpus": 0,
            "reserved_gpus": 0,
            "in_use_gpus": 0,
            "pending_gpus": 0,
            "available_gpus": 0,
        }

    def test_get_capacity_fills_partial_omitted_fields(self):
        # Ceiling fully claimed by queued work: in_use and available are omitted.
        c = _make_client(
            get_json={
                "has_reservation": True,
                "max_total_gpus": 8,
                "reserved_gpus": 8,
                "pending_gpus": 8,
            }
        )
        cap = c.get_capacity()
        assert cap == {
            "has_reservation": True,
            "max_total_gpus": 8,
            "reserved_gpus": 8,
            "in_use_gpus": 0,
            "pending_gpus": 8,
            "available_gpus": 0,
        }


# ─── CortexTrainingClient — account-level job listing ────────────────────────

ACCOUNT_ROUTE = "http://test.local/api/v2/cortex-training"


def _resolve_identifier(segment: str) -> str:
    """Resolve a URL path segment the way the server resolves an identifier."""
    identifier = unquote(segment)
    if identifier.startswith('"'):
        return identifier[1:-1].replace('""', '"')
    return identifier.upper()


class TestIterJobs:
    def test_gets_the_account_route_not_the_schema(self):
        c = _make_client(get_json={"jobs": [{"job_id": "a"}]})

        assert list(c.iter_jobs()) == [{"job_id": "a"}]

        c._session.get.assert_called_once_with(ACCOUNT_ROUTE, params={"limit": 100})

    @pytest.mark.parametrize("page_size", [0, 1001])
    def test_refuses_a_page_size_the_server_would_reject(self, page_size):
        c = _make_client(get_json={"jobs": []})

        with pytest.raises(ValueError, match="page_size must be between 1 and 1000"):
            next(c.iter_jobs(page_size=page_size))

        c._session.get.assert_not_called()

    @pytest.mark.parametrize("page_size", [1, 1000])
    def test_accepts_the_page_size_bounds(self, page_size):
        c = _make_client(get_json={"jobs": []})

        assert list(c.iter_jobs(page_size=page_size)) == []

        c._session.get.assert_called_once_with(
            ACCOUNT_ROUTE, params={"limit": page_size}
        )

    def test_sends_every_filter(self):
        c = _make_client(get_json={"jobs": []})

        list(
            c.iter_jobs(
                database="ML",
                schema='"Exp 1"',
                status=["running", "FAILED"],
                submitted_by="jdoe",
                created_after=date(2026, 10, 1),
                created_before=datetime(
                    2026, 10, 2, 9, 30, tzinfo=timezone(timedelta(hours=2))
                ),
                page_size=50,
            )
        )

        c._session.get.assert_called_once_with(
            ACCOUNT_ROUTE,
            params={
                "database": "ML",
                "schema": '"Exp 1"',
                "status": "running,FAILED",
                "submitted_by": "jdoe",
                "created_after": "2026-10-01",
                "created_before": "2026-10-02T07:30:00Z",
                "limit": 50,
            },
        )

    def test_refuses_a_schema_without_a_database(self):
        c = _make_client(get_json={"jobs": []})

        with pytest.raises(ValueError, match="needs database=... with it"):
            c.iter_jobs(schema="EXP")

        c._session.get.assert_not_called()

    def test_checks_arguments_before_the_first_page_is_iterated(self):
        c = _make_client(get_json={"jobs": []})

        # Not next(...): the call itself refuses, so nothing is left to iterate.
        with pytest.raises(ValueError, match="page_size must be between 1 and 1000"):
            c.iter_jobs(page_size=0)
        with pytest.raises(ValueError, match="needs database"):
            c.iter_jobs(schema="EXP")

        c._session.get.assert_not_called()

    def test_ignores_a_count_the_server_no_longer_returns(self):
        c = _make_client(
            get_json={"jobs": [{"job_id": "a"}], "unknown_status_count": 3}
        )

        assert list(c.iter_jobs(status="running")) == [{"job_id": "a"}]

    @pytest.mark.parametrize(
        ("value", "wire"),
        [
            (date(2026, 10, 1), "2026-10-01"),
            (
                datetime(2026, 10, 1, 1, 0, tzinfo=timezone(timedelta(hours=-7))),
                "2026-10-01T08:00:00Z",
            ),
            (datetime(2026, 10, 1, 8, 0, 0, 250000), "2026-10-01T08:00:00.250000Z"),
            ("2026-10-01T00:00:00+02:00", "2026-10-01T00:00:00+02:00"),
        ],
    )
    def test_created_after_is_sent_in_utc_or_as_given(self, value, wire):
        c = _make_client(get_json={"jobs": []})

        list(c.iter_jobs(status="active", created_after=value))

        assert c._session.get.call_args.kwargs["params"] == {
            "status": "active",
            "created_after": wire,
            "limit": 100,
        }

    def test_follows_next_page_token_with_the_same_filters(self):
        c = _make_client()
        c._session.get.side_effect = [
            _make_response(
                {"jobs": [{"job_id": "c"}, {"job_id": "b"}], "next_page_token": "t1"}
            ),
            # A page may be empty and still carry a token.
            _make_response({"jobs": [], "next_page_token": "t2"}),
            _make_response({"jobs": [{"job_id": "a"}], "next_page_token": None}),
        ]

        jobs = list(c.iter_jobs(status="failed", page_size=2))

        assert [job["job_id"] for job in jobs] == ["c", "b", "a"]
        calls = c._session.get.call_args_list
        assert [call.args for call in calls] == [(ACCOUNT_ROUTE,)] * 3
        assert [call.kwargs["params"] for call in calls] == [
            {"status": "failed", "limit": 2},
            {"status": "failed", "limit": 2, "page_token": "t1"},
            {"status": "failed", "limit": 2, "page_token": "t2"},
        ]

    def test_reads_the_next_page_only_when_iterated(self):
        c = _make_client()
        c._session.get.side_effect = [
            _make_response({"jobs": [{"job_id": "b"}], "next_page_token": "t1"}),
            _make_response({"jobs": [{"job_id": "a"}]}),
        ]

        jobs = c.iter_jobs()

        assert next(jobs) == {"job_id": "b"}
        assert c._session.get.call_count == 1
        assert list(jobs) == [{"job_id": "a"}]
        assert c._session.get.call_count == 2

    def test_404_is_not_retried_and_says_listing_is_not_enabled(self):
        c = _make_client()
        c._session.get.return_value = _make_error_response(
            {"message": "Account-level job listing is not enabled for this account"},
            404,
        )

        with pytest.raises(
            nc.requests.exceptions.HTTPError,
            match="account-level job listing is not enabled for this account",
        ) as exc:
            list(c.iter_jobs())

        assert exc.value.response.status_code == 404
        assert c.max_retries == 10
        c._session.get.assert_called_once()
        # No auto-create, or any other request, follows a 404.
        c._session.post.assert_not_called()

    def test_a_not_found_error_does_not_create_the_configured_database(self):
        c = _make_client()
        c._session.get.return_value = _make_error_response(
            {"message": "Database 'NOPE' does not exist or not authorized."}, 400
        )

        with pytest.raises(nc.requests.exceptions.HTTPError, match="400 Client Error"):
            list(c.iter_jobs(database="NOPE"))

        c._session.get.assert_called_once()
        c._session.post.assert_not_called()

    def test_other_transient_errors_are_still_retried(self, monkeypatch):
        monkeypatch.setattr(
            nc,
            "wait_exponential_jitter",
            lambda **kwargs: lambda retry_state: 0,
        )
        c = _make_client()
        c._session.get.side_effect = [
            _make_error_response({"message": "unavailable"}, 503),
            _make_response({"jobs": [{"job_id": "a"}]}),
        ]

        assert list(c.iter_jobs()) == [{"job_id": "a"}]
        assert c._session.get.call_count == 2

    def test_bad_request_is_raised_unchanged(self):
        c = _make_client()
        c._session.get.return_value = _make_error_response(
            {"message": "Invalid database or schema scope"}, 400
        )

        with pytest.raises(nc.requests.exceptions.HTTPError, match="400 Client Error"):
            list(c.iter_jobs(database="NOPE"))

        c._session.get.assert_called_once()

    def test_emits_one_metric_per_page_without_filter_values(self, monkeypatch):
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "1")
        c = _make_client()
        c._metric_emitter = MagicMock()
        c._session.get.side_effect = [
            _make_response({"jobs": [{"job_id": "b"}], "next_page_token": "t1"}),
            _make_error_response({"message": "not enabled"}, 404),
        ]

        with pytest.raises(nc.requests.exceptions.HTTPError):
            list(c.iter_jobs(submitted_by="jdoe"))

        calls = c._metric_emitter.emit.call_args_list
        assert [call.args[0] for call in calls] == ["iter_jobs", "iter_jobs"]
        assert [call.args[1]["success"] for call in calls] == [True, False]
        assert [call.args[1]["request_count"] for call in calls] == [1, 1]
        assert [call.kwargs["attributes"] for call in calls] == [
            {"client.surface": "sdk"},
            {"client.surface": "sdk", "error.type": "HTTPError", "error.http_status": 404},
        ]
        assert "jdoe" not in repr(calls)


class TestForJob:
    @pytest.mark.parametrize(
        ("database_name", "schema_name", "path"),
        [
            ("ML", "EXP_1$", "ML/schemas/EXP_1$"),
            ("_DB", "PUBLIC", "_DB/schemas/PUBLIC"),
            ("ml", "Exp1", "%22ml%22/schemas/%22Exp1%22"),
            ("My DB", 'a"b', "%22My%20DB%22/schemas/%22a%22%22b%22"),
            ("1DB", "a/b?c#d%e", "%221DB%22/schemas/%22a%2Fb%3Fc%23d%25e%22"),
            ("DB\n", "$X", "%22DB%0A%22/schemas/%22$X%22"),
        ],
    )
    def test_binds_quoted_percent_encoded_names(
        self, database_name, schema_name, path
    ):
        c = _make_client()

        job_client = c.for_job(
            {"job_id": "j1", "database_name": database_name, "schema_name": schema_name}
        )

        assert job_client._prefix == (
            f"http://test.local/api/v2/databases/{path}/cortex-training"
        )
        database_segment, schema_segment = path.split("/schemas/")
        assert _resolve_identifier(database_segment) == database_name
        assert _resolve_identifier(schema_segment) == schema_name

    def test_shares_session_auth_and_telemetry_and_copies_settings(self):
        c = CortexTrainingClient(
            base_url="http://test.local",
            database="DB",
            schema="SCH",
            endpoint="training-v2",
            poll_interval=1.0,
            poll_timeout=60.0,
            poll_backoff_multiplier=2.0,
            poll_max_interval=8.0,
            max_retries=3,
            request_timeout=(5.0, 7.0),
        )
        c._session = MagicMock()
        c._session.get.return_value = _make_response({"job_id": "j1"})
        c._auth_provider = MagicMock()
        c._auth_provider.get_token.return_value = "tok"
        c._metric_emitter = MagicMock()
        c._telemetry_surface = nc.TUI_SURFACE
        c._artifact_connection_config = {"host": "x.test", "pat": "pat"}
        c._artifact_connection_factory = MagicMock()

        job_client = c.for_job(
            {"job_id": "j1", "database_name": "ML", "schema_name": "EXP"}
        )

        assert type(job_client) is CortexTrainingClient and job_client is not c
        assert job_client._session is c._session
        assert job_client._auth_provider is c._auth_provider
        assert job_client._metric_emitter is c._metric_emitter
        assert job_client._telemetry_surface == nc.TUI_SURFACE
        assert job_client._artifact_connection_config is c._artifact_connection_config
        assert job_client._artifact_connection_factory is c._artifact_connection_factory
        assert (
            job_client.endpoint,
            job_client.poll_interval,
            job_client.poll_timeout,
            job_client.poll_backoff_multiplier,
            job_client.poll_max_interval,
            job_client.max_retries,
            job_client.request_timeout,
        ) == ("training-v2", 1.0, 60.0, 2.0, 8.0, 3, (5.0, 7.0))
        assert (c.database, c.schema) == ("DB", "SCH")

        job_client.get_job("j1")

        c._session.get.assert_called_once_with(
            "http://test.local/api/v2/databases/ML/schemas/EXP/training-v2/j1",
            headers={"Authorization": 'Snowflake Token="tok"'},
        )

    def test_closes_the_session_it_opened_before_borrowing(self):
        c = _make_client()

        with patch.object(
            nc._DefaultTimeoutSession, "close", autospec=True
        ) as closed:
            job_client = c.for_job({"database_name": "ML", "schema_name": "EXP"})

        # The constructor opens a session and a connection pool; for_job
        # releases them instead of leaking one per call.
        closed.assert_called_once()
        leaked = closed.call_args.args[0]
        assert leaked is not c._session
        assert job_client._session is c._session

    def test_closing_it_leaves_the_parent_session_auth_and_telemetry_open(self):
        c = _make_client()
        c._auth_provider = MagicMock()
        c._metric_emitter = MagicMock()

        job_client = c.for_job({"database_name": "ML", "schema_name": "EXP"})
        with job_client:
            pass
        job_client.close()

        assert c._session.close.called is False
        assert c._auth_provider.close.called is False
        assert c._metric_emitter.close.called is False
        # The parent still owns all three and closes them.
        c.close()
        assert c._session.close.called
        assert c._auth_provider.close.called
        assert c._metric_emitter.close.called

    def test_does_not_create_the_jobs_database(self):
        c = _make_client()
        c._session.get.return_value = _make_error_response(
            {"message": "Schema OTHER_DB.PUBLIC is not found or not authorized."}, 400
        )

        job_client = c.for_job({"database_name": "OTHER_DB", "schema_name": "PUBLIC"})
        with pytest.raises(nc.requests.exceptions.HTTPError, match="400 Client Error"):
            job_client.get_job("j1")

        # The listed database already exists and may belong to another role, so
        # the not-found answer surfaces instead of a CREATE DATABASE for it.
        c._session.post.assert_not_called()

    def test_returns_a_client_of_the_same_class(self):
        class SubClient(CortexTrainingClient):
            pass

        c = SubClient(base_url="http://test.local", database="DB", schema="SCH")

        job_client = c.for_job({"database_name": "ML", "schema_name": "EXP"})

        assert type(job_client) is SubClient


# ─── CortexTrainingClient — get_capacity_breakdown ───────────────────────────

# One H200 snapshot (a reservation of 64, caller ALICE) in which every split
# sums to the totals.
_CAPACITY_TOTALS = {
    "has_reservation": True,
    "max_total_gpus": 64,
    "reserved_gpus": 64,
    "in_use_gpus": 44,
    "pending_gpus": 16,
    "available_gpus": 4,
}
_CAPACITY_BREAKDOWN_BODY = {
    **_CAPACITY_TOTALS,
    "breakdown": {
        "unsupported_dimensions": [],
        "job_count": 6,
        "truncated": False,
        "by_job_type": {
            "status": "OK",
            "groups": [
                {"job_type": "TRAINING", "in_use_gpus": 28, "pending_gpus": 16, "sub_job_count": 5},
                {"job_type": "SAMPLING", "in_use_gpus": 16, "pending_gpus": 0, "sub_job_count": 2},
                {"job_type": "LOG_PROBABILITY", "in_use_gpus": 0, "pending_gpus": 0, "sub_job_count": 1},
            ],
        },
        "by_submitted_by": {
            "status": "OK",
            "groups": [
                {"kind": "USER", "submitted_by": "ALICE", "is_caller": True,
                 "in_use_gpus": 24, "pending_gpus": 0, "job_count": 2},
                {"kind": "OTHER", "in_use_gpus": 16, "pending_gpus": 16, "job_count": 3},
                {"kind": "UNATTRIBUTED", "in_use_gpus": 4, "pending_gpus": 0, "job_count": 1},
            ],
        },
    },
}


class TestCapacityBreakdown:
    def test_sends_breakdown_with_capacity_timeout_and_no_retries(self):
        c = _make_client(get_json=_CAPACITY_BREAKDOWN_BODY)
        c._send = MagicMock(wraps=c._send)

        c.get_capacity_breakdown(hardware=Hardware.B200)

        c._session.get.assert_called_once_with(
            f"{c._prefix}/capacity",
            timeout=nc._CAPACITY_REQUEST_TIMEOUT,
            params={"hardware": "B200", "breakdown": "job_type,submitted_by"},
        )
        assert c._send.call_args.kwargs["max_retries"] == 0

    def test_omits_hardware_when_not_given(self):
        c = _make_client(get_json=_CAPACITY_BREAKDOWN_BODY)

        c.get_capacity_breakdown()

        c._session.get.assert_called_once_with(
            f"{c._prefix}/capacity",
            timeout=nc._CAPACITY_REQUEST_TIMEOUT,
            params={"breakdown": "job_type,submitted_by"},
        )

    def test_uses_shorter_caller_timeout_and_does_not_retry(self):
        c = _make_client()
        c.request_timeout = (2.0, 0.5)
        c._session.get.side_effect = nc.requests.exceptions.ReadTimeout("silent peer")

        with pytest.raises(nc.requests.exceptions.ReadTimeout):
            c.get_capacity_breakdown(dimensions=("job_type",))

        c._session.get.assert_called_once_with(
            f"{c._prefix}/capacity",
            timeout=(2.0, 0.5),
            params={"breakdown": "job_type"},
        )

    def test_joins_any_iterable_of_dimensions_in_order(self):
        c = _make_client(get_json={})
        c.get_capacity_breakdown(dimensions=iter(["submitted_by", "job_type"]))
        assert c._session.get.call_args.kwargs["params"] == {
            "breakdown": "submitted_by,job_type",
        }

    def test_str_enum_dimension_reads_its_own_section(self):
        from enum import Enum

        # On Python 3.11+ a member like this formats as "Dimension.JOB_TYPE".
        class Dimension(str, Enum):
            JOB_TYPE = "job_type"

        c = _make_client(get_json={
            "in_use_gpus": 8,
            "breakdown": {
                "job_count": 1,
                "by_job_type": {
                    "status": "OK",
                    "groups": [{"job_type": "TRAINING", "in_use_gpus": 8, "sub_job_count": 1}],
                },
            },
        })

        by_job_type = c.get_capacity_breakdown(dimensions=(Dimension.JOB_TYPE,))[
            "breakdown"
        ]["by_job_type"]

        assert c._session.get.call_args.kwargs["params"] == {"breakdown": "job_type"}
        assert by_job_type["status"] == "OK"
        assert by_job_type["groups"][0]["in_use_gpus"] == 8

    @pytest.mark.parametrize(
        "dimensions",
        [
            (), [], "job_type", "job_type,submitted_by", None, 7,
            ("owner_role",), ("job-type",), ("JOB_TYPE",), ("job_type", None),
        ],
    )
    def test_rejects_bad_dimensions_before_any_request(self, dimensions):
        c = _make_client(get_json={})
        with pytest.raises(
            ValueError,
            match=(
                r"^dimensions must be a non-empty iterable of job_type, "
                r"submitted_by \(not a string\), got "
            ),
        ):
            c.get_capacity_breakdown(dimensions=dimensions)
        c._session.get.assert_not_called()

    def test_rejects_unknown_hardware_before_any_request(self):
        c = _make_client(get_json={})
        with pytest.raises(ValueError, match="hardware must be one of"):
            c.get_capacity_breakdown(hardware="A10")
        c._session.get.assert_not_called()

    def test_ok_breakdown_is_normalized(self):
        c = _make_client(get_json=_CAPACITY_BREAKDOWN_BODY)
        cap = c.get_capacity_breakdown()
        assert cap == {
            **_CAPACITY_TOTALS,
            "breakdown": {
                "job_count": 6,
                "truncated": False,
                "unsupported_dimensions": [],
                "by_job_type": {
                    "status": "OK",
                    "groups": [
                        {"job_type": "TRAINING", "in_use_gpus": 28, "pending_gpus": 16, "sub_job_count": 5},
                        {"job_type": "SAMPLING", "in_use_gpus": 16, "pending_gpus": 0, "sub_job_count": 2},
                        {"job_type": "LOG_PROBABILITY", "in_use_gpus": 0, "pending_gpus": 0, "sub_job_count": 1},
                    ],
                },
                "by_submitted_by": {
                    "status": "OK",
                    "groups": [
                        {"kind": "USER", "submitted_by": "ALICE", "is_caller": True,
                         "in_use_gpus": 24, "pending_gpus": 0, "job_count": 2},
                        {"kind": "OTHER", "submitted_by": None, "is_caller": False,
                         "in_use_gpus": 16, "pending_gpus": 16, "job_count": 3},
                        {"kind": "UNATTRIBUTED", "submitted_by": None, "is_caller": False,
                         "in_use_gpus": 4, "pending_gpus": 0, "job_count": 1},
                    ],
                },
            },
        }
        # The CLI prints with sort_keys=True, which needs every key to be a string.
        json.dumps(cap, sort_keys=True)

    def test_job_types_are_zero_filled_in_order_with_prefix_stripped(self):
        c = _make_client(get_json={
            "in_use_gpus": 12,
            "pending_gpus": 16,
            "breakdown": {
                "job_count": 3,
                "by_job_type": {
                    "status": "OK",
                    "groups": [
                        {"job_type": "JOB_TYPE_SAMPLING", "in_use_gpus": 8, "sub_job_count": 1},
                        {"job_type": "TRAINING", "pending_gpus": 16, "sub_job_count": 2},
                        {"job_type": "UNKNOWN", "in_use_gpus": 4, "sub_job_count": 1},
                    ],
                },
            },
        })

        cap = c.get_capacity_breakdown(dimensions=("job_type",))

        assert cap["breakdown"] == {
            "job_count": 3,
            "truncated": False,
            "unsupported_dimensions": [],
            "by_job_type": {
                "status": "OK",
                "groups": [
                    {"job_type": "TRAINING", "in_use_gpus": 0, "pending_gpus": 16, "sub_job_count": 2},
                    {"job_type": "SAMPLING", "in_use_gpus": 8, "pending_gpus": 0, "sub_job_count": 1},
                    {"job_type": "LOG_PROBABILITY", "in_use_gpus": 0, "pending_gpus": 0, "sub_job_count": 0},
                    {"job_type": "UNKNOWN", "in_use_gpus": 4, "pending_gpus": 0, "sub_job_count": 1},
                ],
            },
        }

    def test_repeated_known_job_type_is_summed_into_its_row(self):
        # The server sends each known type once. If it ever repeated one,
        # summing keeps the groups adding up to the totals.
        c = _make_client(get_json={
            "in_use_gpus": 10,
            "breakdown": {
                "job_count": 2,
                "by_job_type": {
                    "status": "OK",
                    "groups": [
                        {"job_type": "TRAINING", "in_use_gpus": 8, "sub_job_count": 1},
                        {"job_type": "JOB_TYPE_TRAINING", "in_use_gpus": 2, "sub_job_count": 1},
                    ],
                },
            },
        })

        groups = c.get_capacity_breakdown(dimensions=("job_type",))["breakdown"][
            "by_job_type"
        ]["groups"]

        assert [group["job_type"] for group in groups] == list(nc._CAPACITY_BREAKDOWN_JOB_TYPES)
        assert groups[0] == {
            "job_type": "TRAINING", "in_use_gpus": 10, "pending_gpus": 0, "sub_job_count": 2,
        }

    def test_only_user_rows_carry_a_name(self):
        c = _make_client(get_json={
            "in_use_gpus": 20,
            "pending_gpus": 12,
            "breakdown": {
                "job_count": 5,
                "by_submitted_by": {
                    "status": "OK",
                    "groups": [
                        {"kind": "USER", "submitted_by": "ALICE", "is_caller": True,
                         "in_use_gpus": 8, "job_count": 1},
                        # A USER row's is_caller is passed through, and is False if omitted.
                        {"kind": "USER", "submitted_by": "CAROL", "is_caller": False,
                         "in_use_gpus": 4, "job_count": 1},
                        {"kind": "USER", "submitted_by": "DAVE", "pending_gpus": 4, "job_count": 1},
                        {"kind": "OTHER", "submitted_by": "BOB", "is_caller": True,
                         "in_use_gpus": 8, "job_count": 1},
                        {"kind": "UNATTRIBUTED", "submitted_by": "", "pending_gpus": 8, "job_count": 1},
                    ],
                },
            },
        })

        groups = c.get_capacity_breakdown(dimensions=("submitted_by",))["breakdown"][
            "by_submitted_by"
        ]["groups"]

        assert groups == [
            {"kind": "USER", "submitted_by": "ALICE", "is_caller": True,
             "in_use_gpus": 8, "pending_gpus": 0, "job_count": 1},
            {"kind": "USER", "submitted_by": "CAROL", "is_caller": False,
             "in_use_gpus": 4, "pending_gpus": 0, "job_count": 1},
            {"kind": "USER", "submitted_by": "DAVE", "is_caller": False,
             "in_use_gpus": 0, "pending_gpus": 4, "job_count": 1},
            {"kind": "OTHER", "submitted_by": None, "is_caller": False,
             "in_use_gpus": 8, "pending_gpus": 0, "job_count": 1},
            {"kind": "UNATTRIBUTED", "submitted_by": None, "is_caller": False,
             "in_use_gpus": 0, "pending_gpus": 8, "job_count": 1},
        ]

    def test_user_rows_keep_the_server_order(self):
        # The server orders USER rows by in_use_gpus + pending_gpus descending
        # and then by name, so the caller, ALICE, comes third.
        c = _make_client(get_json={
            "in_use_gpus": 52,
            "pending_gpus": 16,
            "breakdown": {
                "unsupported_dimensions": [],
                "job_count": 7,
                "truncated": False,
                "by_submitted_by": {
                    "status": "OK",
                    "groups": [
                        {"kind": "USER", "submitted_by": "DAVE", "is_caller": False,
                         "in_use_gpus": 16, "pending_gpus": 8, "job_count": 2},
                        {"kind": "USER", "submitted_by": "BOB", "is_caller": False,
                         "in_use_gpus": 16, "pending_gpus": 0, "job_count": 1},
                        {"kind": "USER", "submitted_by": "ALICE", "is_caller": True,
                         "in_use_gpus": 8, "pending_gpus": 0, "job_count": 1},
                        {"kind": "USER", "submitted_by": "CAROL", "is_caller": False,
                         "in_use_gpus": 0, "pending_gpus": 8, "job_count": 1},
                        {"kind": "OTHER", "in_use_gpus": 8, "pending_gpus": 0, "job_count": 1},
                        {"kind": "UNATTRIBUTED", "in_use_gpus": 4, "pending_gpus": 0, "job_count": 1},
                    ],
                },
            },
        })

        by_submitted_by = c.get_capacity_breakdown(dimensions=("submitted_by",))[
            "breakdown"
        ]["by_submitted_by"]

        assert by_submitted_by == {
            "status": "OK",
            "groups": [
                {"kind": "USER", "submitted_by": "DAVE", "is_caller": False,
                 "in_use_gpus": 16, "pending_gpus": 8, "job_count": 2},
                {"kind": "USER", "submitted_by": "BOB", "is_caller": False,
                 "in_use_gpus": 16, "pending_gpus": 0, "job_count": 1},
                {"kind": "USER", "submitted_by": "ALICE", "is_caller": True,
                 "in_use_gpus": 8, "pending_gpus": 0, "job_count": 1},
                {"kind": "USER", "submitted_by": "CAROL", "is_caller": False,
                 "in_use_gpus": 0, "pending_gpus": 8, "job_count": 1},
                {"kind": "OTHER", "submitted_by": None, "is_caller": False,
                 "in_use_gpus": 8, "pending_gpus": 0, "job_count": 1},
                {"kind": "UNATTRIBUTED", "submitted_by": None, "is_caller": False,
                 "in_use_gpus": 4, "pending_gpus": 0, "job_count": 1},
            ],
        }

    def test_old_server_reports_every_dimension_unavailable(self):
        c = _make_client(get_json=_CAPACITY_TOTALS)
        assert c.get_capacity_breakdown() == {
            **_CAPACITY_TOTALS,
            "breakdown": {
                "job_count": 0,
                "truncated": False,
                "unsupported_dimensions": [],
                "by_job_type": {"status": "UNAVAILABLE", "groups": []},
                "by_submitted_by": {"status": "UNAVAILABLE", "groups": []},
            },
        }

    def test_dimension_missing_from_server_breakdown_is_unavailable(self):
        # A server that knows job_type but not submitted_by lists it as unsupported.
        c = _make_client(get_json={
            "in_use_gpus": 8,
            "breakdown": {
                "unsupported_dimensions": ["submitted_by"],
                "job_count": 1,
                "truncated": False,
                "by_job_type": {
                    "status": "OK",
                    "groups": [{"job_type": "TRAINING", "in_use_gpus": 8, "sub_job_count": 1}],
                },
            },
        })

        breakdown = c.get_capacity_breakdown()["breakdown"]

        assert breakdown["unsupported_dimensions"] == ["submitted_by"]
        assert breakdown["by_job_type"]["status"] == "OK"
        assert breakdown["by_submitted_by"] == {"status": "UNAVAILABLE", "groups": []}

    @pytest.mark.parametrize("status", ["DISABLED", "UNAVAILABLE", "UNSUPPORTED", "SOMETHING_NEW", 1])
    def test_status_is_passed_through_as_sent(self, status):
        c = _make_client(get_json={
            "in_use_gpus": 8,
            "breakdown": {
                "unsupported_dimensions": [],
                "by_job_type": {"status": status, "groups": []},
                "by_submitted_by": {"status": status, "groups": []},
            },
        })

        breakdown = c.get_capacity_breakdown()["breakdown"]

        assert breakdown["by_job_type"] == {"status": status, "groups": []}
        assert breakdown["by_submitted_by"] == {"status": status, "groups": []}

    def test_groups_are_kept_whatever_the_status(self):
        # The data decides, so a status this client does not know cannot hide a split.
        c = _make_client(get_json={
            "in_use_gpus": 8,
            "breakdown": {
                "by_job_type": {
                    "status": "SOMETHING_NEW",
                    "groups": [{"job_type": "TRAINING", "in_use_gpus": 8, "sub_job_count": 1}],
                },
            },
        })

        by_job_type = c.get_capacity_breakdown(dimensions=("job_type",))["breakdown"][
            "by_job_type"
        ]

        assert by_job_type["status"] == "SOMETHING_NEW"
        assert by_job_type["groups"][0] == {
            "job_type": "TRAINING", "in_use_gpus": 8, "pending_gpus": 0, "sub_job_count": 1,
        }

    @pytest.mark.parametrize("status", [None, ""], ids=["null", "empty"])
    def test_missing_status_is_ok_when_groups_are_sent(self, status):
        section = {"groups": [{"job_type": "TRAINING", "in_use_gpus": 8, "sub_job_count": 1}]}
        if status is not None:
            section["status"] = status
        c = _make_client(get_json={
            "in_use_gpus": 8,
            "breakdown": {"job_count": 1, "by_job_type": section},
        })

        by_job_type = c.get_capacity_breakdown(dimensions=("job_type",))["breakdown"][
            "by_job_type"
        ]

        assert by_job_type["status"] == "OK"
        assert by_job_type["groups"][0]["in_use_gpus"] == 8

    def test_section_without_status_or_groups_fails_closed_as_unavailable(self):
        c = _make_client(get_json={"in_use_gpus": 8, "breakdown": {"by_job_type": {}}})

        by_job_type = c.get_capacity_breakdown(dimensions=("job_type",))["breakdown"][
            "by_job_type"
        ]

        assert by_job_type == {"status": "UNAVAILABLE", "groups": []}

    def test_truncated_is_passed_through(self):
        # The jobs the server did not attribute one by one hold the fewest
        # GPUs, and are counted as UNATTRIBUTED.
        c = _make_client(get_json={
            "in_use_gpus": 48,
            "pending_gpus": 16,
            "breakdown": {
                "unsupported_dimensions": [],
                "job_count": 7,
                "truncated": True,
                "by_submitted_by": {
                    "status": "OK",
                    "groups": [
                        {"kind": "OTHER", "in_use_gpus": 48, "job_count": 5},
                        {"kind": "UNATTRIBUTED", "pending_gpus": 16, "job_count": 2},
                    ],
                },
            },
        })

        breakdown = c.get_capacity_breakdown(dimensions=("submitted_by",))["breakdown"]

        assert breakdown == {
            "job_count": 7,
            "truncated": True,
            "unsupported_dimensions": [],
            "by_submitted_by": {
                "status": "OK",
                "groups": [
                    {"kind": "OTHER", "submitted_by": None, "is_caller": False,
                     "in_use_gpus": 48, "pending_gpus": 0, "job_count": 5},
                    {"kind": "UNATTRIBUTED", "submitted_by": None, "is_caller": False,
                     "in_use_gpus": 0, "pending_gpus": 16, "job_count": 2},
                ],
            },
        }

    def test_records_its_own_operation_metric(self, monkeypatch):
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "1")
        c = _make_client(get_json=_CAPACITY_BREAKDOWN_BODY)
        c._metric_emitter = MagicMock()

        c.get_capacity_breakdown()

        c._metric_emitter.emit.assert_called_once()
        operation, value = c._metric_emitter.emit.call_args.args
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert operation == "get_capacity_breakdown"
        assert value["success"] is True
        assert value["request_count"] == 1
        assert value["attempt_count"] == 1
        # The result names the caller, so none of it becomes part of the metric.
        assert attributes == {"client.surface": "sdk"}
        assert "ALICE" not in repr(c._metric_emitter.emit.call_args)

    def test_bad_dimensions_record_a_failed_operation_without_a_request(self):
        c = _make_client(get_json=_CAPACITY_BREAKDOWN_BODY)
        c._metric_emitter = MagicMock()

        with pytest.raises(ValueError, match="dimensions must be"):
            c.get_capacity_breakdown(dimensions=("owner_role",))

        c._session.get.assert_not_called()
        c._metric_emitter.emit.assert_called_once()
        operation, value = c._metric_emitter.emit.call_args.args
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert operation == "get_capacity_breakdown"
        assert value["success"] is False
        assert value["request_count"] == 0
        assert attributes == {"client.surface": "sdk", "error.type": "ValueError"}

    def test_get_capacity_still_returns_only_the_six_totals(self):
        c = _make_client(get_json=_CAPACITY_BREAKDOWN_BODY)
        assert c.get_capacity() == _CAPACITY_TOTALS
        c._session.get.assert_called_once_with(
            f"{c._prefix}/capacity", timeout=nc._CAPACITY_REQUEST_TIMEOUT
        )


# ─── CortexTrainingClient — wait_for_job ─────────────────────────────────────


class TestWaitForJob:
    def test_returns_when_running(self, monkeypatch):
        c = _make_client(get_json={"status": "JOB_STATE_RUNNING"})
        sleep = MagicMock()
        monkeypatch.setattr(nc.time, "sleep", sleep)
        assert c.wait_for_job("j1")["status"] == "JOB_STATE_RUNNING"
        sleep.assert_not_called()

    def test_uses_exponential_backoff_until_running(self, monkeypatch):
        c = _make_client()
        c.poll_interval = 1.0
        c.poll_backoff_multiplier = 2.0
        c.poll_max_interval = 3.0
        c._session.get.side_effect = [
            _make_response({"status": "pending"}),
            _make_response({"status": "pending"}),
            _make_response({"status": "pending"}),
            _make_response({"status": "running"}),
        ]
        sleeps = []
        monkeypatch.setattr(nc.time, "sleep", sleeps.append)

        assert c.wait_for_job("j1")["status"] == "running"
        assert sleeps == [1.0, 2.0, 3.0]

    def test_backoff_sleep_does_not_exceed_remaining_timeout(self, monkeypatch):
        c = _make_client()
        c.poll_backoff_multiplier = 2.0
        c.poll_max_interval = 30.0
        sleeps = []
        monkeypatch.setattr(nc.time, "monotonic", lambda: 95.0)
        monkeypatch.setattr(nc.time, "sleep", sleeps.append)

        assert c._sleep_with_backoff(10.0, deadline=100.0) == 20.0
        assert sleeps == [5.0]

    def test_raises_on_terminal(self, monkeypatch):
        c = _make_client(get_json={"status": "failed", "reason": "boom"})
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        with pytest.raises(RuntimeError, match="boom"):
            c.wait_for_job("j1")

    def test_times_out(self, monkeypatch):
        c = _make_client(get_json={"status": "pending"})
        c.poll_timeout = 0.01
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        with pytest.raises(TimeoutError):
            c.wait_for_job("j1")

    def test_polls_through_unknown_numeric_status(self, monkeypatch):
        # GS running behind a CP that emits a new JobState renders it as a raw
        # integer (e.g. 12 for a state GS doesn't know). wait_for_job must treat
        # it as still-in-progress and keep polling until RUNNING — not crash on
        # int.lower() and not mistake it for a terminal state.
        c = _make_client()
        c._session.get.side_effect = [
            _make_response({"status": 12}),  # unknown enum, rendered as int
            _make_response({"status": "running"}),
        ]
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        assert c.wait_for_job("j1")["status"] == "running"


# ─── CortexTrainingClient — data-plane ───────────────────────────────────────


class TestForwardBackwardPayloadHelpers:
    def test_build_forward_backward_payload_from_tensor_kwargs(self):
        body = nc.build_forward_backward_payload(
            {
                "payload": {
                    "kwargs": {
                        "input_ids": {"data": [[1, 2, 3]], "dtype": "long"},
                        "labels": {"data": [[2, 3, -100]], "dtype": "long"},
                    },
                },
            }
        )

        loaded = _wire_load(body)
        assert loaded["args"] == ()
        assert loaded["kwargs"]["input_ids"].tolist() == [[1, 2, 3]]
        assert loaded["kwargs"]["labels"].tolist() == [[2, 3, -100]]

    def test_tokenized_payload_builds_prime_rl_style_kwargs(self, monkeypatch):
        import torch

        class FakeTokenizer:
            pad_token_id = None
            eos_token = "<eos>"

            def __call__(self, texts, **kwargs):
                assert texts == ["first", "second"]
                assert kwargs["padding"] == "max_length"
                assert kwargs["max_length"] == 4
                return {
                    "input_ids": torch.tensor(
                        [
                            [10, 11, 0, 0],
                            [20, 21, 22, 0],
                        ]
                    ),
                    "attention_mask": torch.tensor(
                        [
                            [1, 1, 0, 0],
                            [1, 1, 1, 0],
                        ]
                    ),
                }

        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda model_name, **kwargs: FakeTokenizer())
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        kwargs = nc.build_forward_backward_kwargs(
            {
                "tokenizer": {"model_name": "fake-model"},
                "texts": ["first", "second"],
                "batch_size": 2,
                "max_length": 4,
                "position_ids": "arange",
                "labels": {"strategy": "next_token", "ignore_index": -100},
            }
        )

        assert kwargs["input_ids"].tolist() == [
            [10, 11, 0, 0],
            [20, 21, 22, 0],
        ]
        assert kwargs["position_ids"].tolist() == [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]
        assert kwargs["labels"].tolist() == [
            [11, -100, -100, -100],
            [21, 22, -100, -100],
        ]
        assert "attention_mask" not in kwargs


class TestDataPlane:
    def test_forward_backward(self):
        c = _make_client(post_json={"request_id": "r1"})
        rid = c.forward_backward("j1", b"\x01\x02")
        assert rid == "r1"
        url, kwargs = c._session.post.call_args
        assert url[0] == f"{c._prefix}/j1/forward-backward"
        assert kwargs["headers"]["Content-Type"] == "application/octet-stream"
        from cortex_training import wire

        metadata = wire.read_byte_chunk_metadata(kwargs["data"])
        assert metadata["operation"] == "fwd-bwd"
        assert metadata["chunk_idx"] == 0
        assert metadata["total_chunks"] == 1
        assert wire.decode_byte_chunks([kwargs["data"]], kind="request") == b"\x01\x02"

    def test_forward_backward_chunks_oversized_payload(self):
        c = _make_client()
        c._MAX_FWD_BWD_BYTES = 64 * 1024
        c._session.post.side_effect = [
            _make_response({"chunk_cached": True}),
            _make_response({"request_id": "r1"}),
        ]
        oversized = b"x" * (70 * 1024)

        assert c.forward_backward("j1", oversized) == "r1"
        assert c._session.post.call_count == 2
        first = c._session.post.call_args_list[0].kwargs["data"]
        from cortex_training import wire

        assert wire.read_byte_chunk_metadata(first)["operation"] == "fwd-bwd"
        group_ids = {
            wire.read_byte_chunk_metadata(call.kwargs["data"])["chunk_group_id"]
            for call in c._session.post.call_args_list
        }
        assert len(group_ids) == 1

    def test_forward_backward_replays_once_from_zero_on_restart_required(self):
        c = _make_client()
        c._MAX_FWD_BWD_BYTES = 64 * 1024
        restart = _make_error_response(
            {
                "code": "517604",
                "message": (
                    "Cortex training forwardBackward failed: zone returned 409: "
                    '{"detail":{"code":"chunk_group_restart_required",'
                    '"message":"request chunk group state is missing; restart from chunk 0",'
                    '"chunk_group_id":"group-a","received_chunk":1,'
                    '"expected_start_chunk":0}}'
                ),
            }
        )
        c._session.post.side_effect = [
            _make_response({"chunk_cached": True}),
            restart,
            _make_response({"chunk_cached": True}),
            _make_response({"request_id": "r1"}),
        ]

        assert c.forward_backward("j1", b"x" * (70 * 1024)) == "r1"
        assert c._session.post.call_count == 4

        from cortex_training import wire

        sent = [call.kwargs["data"] for call in c._session.post.call_args_list]
        metadata = [wire.read_byte_chunk_metadata(chunk) for chunk in sent]
        assert [item["chunk_idx"] for item in metadata] == [0, 1, 0, 1]
        assert len({item["chunk_group_id"] for item in metadata}) == 1
        assert sent[0] == sent[2]
        assert sent[1] == sent[3]

    def test_forward_backward_limits_whole_group_replay_to_once(self):
        c = _make_client()
        c._MAX_FWD_BWD_BYTES = 64 * 1024

        def restart_response():
            return _make_error_response(
                {
                    "detail": {
                        "code": "chunk_group_restart_required",
                        "message": "request chunk group state is missing; restart from chunk 0",
                        "chunk_group_id": "group-a",
                        "received_chunk": 1,
                        "expected_start_chunk": 0,
                    }
                }
            )

        c._session.post.side_effect = [
            _make_response({"chunk_cached": True}),
            restart_response(),
            _make_response({"chunk_cached": True}),
            restart_response(),
        ]

        with pytest.raises(nc.ChunkGroupRestartError):
            c.forward_backward("j1", b"x" * (70 * 1024))
        assert c._session.post.call_count == 4

    def test_forward_backward_legacy_missing_chunks_409_is_not_blind_retried(self):
        c = _make_client()
        c.max_retries = 20
        c._session.post.return_value = _make_error_response(
            {
                "code": "517604",
                "message": (
                    "Cortex training forwardBackward failed: zone returned 409: "
                    '{"detail":{"message":"request chunk group is missing chunks",'
                    '"chunk_group_id":"group-a","missing_chunks":[0,1]}}'
                ),
            }
        )

        with pytest.raises(nc.ChunkGroupConflictError) as exc_info:
            c.forward_backward("j1", b"small")
        assert exc_info.value.detail["code"] == "chunk_group_missing_chunks"
        assert c._session.post.call_count == 1

    def test_generate_minimal_omits_optionals(self):
        c = _make_client(post_json={"request_id": "g1"})
        rid = c.generate("j1", prompts=["hello"])
        assert rid == "g1"
        url, kwargs = c._session.post.call_args
        assert url[0] == f"{c._prefix}/j1/generate"
        assert kwargs["headers"]["Content-Type"] == "application/octet-stream"
        body = _wire_load(kwargs["data"])
        assert body == {"prompts": ["hello"]}

    def test_generate_full_payload(self):
        c = _make_client(post_json={"request_id": "g2"})
        rid = c.generate(
            "j1",
            prompts=["a", "b"],
            sampling_params={"max_tokens": 4, "temperature": 0.7},
            routing_key="rk-1",
            strict=True,
        )
        assert rid == "g2"
        kwargs = c._session.post.call_args.kwargs
        body = _wire_load(kwargs["data"])
        assert body == {
            "prompts": ["a", "b"],
            "sampling_params": {"max_tokens": 4, "temperature": 0.7},
            "routing_key": "rk-1",
            "strict": True,
        }
        assert kwargs["headers"]["Content-Type"] == "application/octet-stream"

    def test_generate_accepts_per_prompt_sampling_params(self):
        c = _make_client(post_json={"request_id": "g-list"})
        rid = c.generate(
            "j1",
            prompts=["a", "b"],
            sampling_params=[
                {"max_tokens": 4, "temperature": 0.7},
                None,
            ],
            routing_key=["rk-1", None],
        )
        assert rid == "g-list"
        body = _wire_load(c._session.post.call_args.kwargs["data"])
        assert body == {
            "prompts": ["a", "b"],
            "sampling_params": [
                {"max_tokens": 4, "temperature": 0.7},
                None,
            ],
            "routing_key": ["rk-1", None],
        }

    def test_generate_strict_false_is_sent(self):
        c = _make_client(post_json={"request_id": "g3"})
        c.generate("j1", prompts=["x"], strict=False)
        body = _wire_load(c._session.post.call_args.kwargs["data"])
        assert body == {"prompts": ["x"], "strict": False}

    def test_generate_stream_full_payload(self):
        c = _make_client(post_json={"request_id": "s1", "count": 2})
        out = c.generate_stream(
            "j1",
            prompts=[[1, 2], [3, 4]],
            sampling_params=[
                {"max_tokens": 4, "temperature": 0.7},
                {"max_tokens": 2, "temperature": 0.3},
            ],
            routing_key=["rk-1", None],
            strict=True,
        )
        assert out == {"request_id": "s1", "count": 2}
        # The stream path sends one unchunked frame, never a chunk group.
        c._session.post.assert_called_once()
        url, kwargs = c._session.post.call_args
        assert url[0] == f"{c._prefix}/j1/generate-stream"
        assert kwargs["headers"]["Content-Type"] == "application/octet-stream"
        body = _wire_load(kwargs["data"])
        assert body["sampling_params"] == [
            {"max_tokens": 4, "temperature": 0.7},
            {"max_tokens": 2, "temperature": 0.3},
        ]
        assert body["routing_key"] == ["rk-1", None]
        assert body["strict"] is True
        prompts = body["prompts"]
        assert [p.dtype for p in prompts] == [torch.int32, torch.int32]
        assert [p.tolist() for p in prompts] == [[1, 2], [3, 4]]

    def test_generate_stream_frame_matches_generate_frame(self):
        prompts = ["hello", [1, 2, 3]]
        unary = _make_client(post_json={"request_id": "g-same"})
        unary.generate("j1", prompts=prompts)
        stream = _make_client(post_json={"request_id": "s-same"})
        stream.generate_stream("j1", prompts=prompts)

        from_unary = _wire_load(unary._session.post.call_args.kwargs["data"])["prompts"]
        from_stream = _wire_load(stream._session.post.call_args.kwargs["data"])["prompts"]
        assert from_unary[0] == from_stream[0] == "hello"
        assert from_unary[1].dtype == from_stream[1].dtype == torch.int32
        assert torch.equal(from_unary[1], from_stream[1])

    def test_generate_chunks_oversized_payload(self):
        c = _make_client()
        c._MAX_GENERATE_BYTES = 64 * 1024
        c._session.post.side_effect = [
            _make_response({"chunk_cached": True}),
            _make_response({"request_id": "g4"}),
        ]
        oversized_prompt = "x" * (70 * 1024)

        assert c.generate("j1", prompts=[oversized_prompt]) == "g4"
        assert c._session.post.call_count == 2
        first = c._session.post.call_args_list[0].kwargs["data"]
        from cortex_training import wire

        assert wire.read_byte_chunk_metadata(first)["operation"] == "generate"

    def test_generate_stream_rejects_oversized_payload(self):
        c = _make_client(post_json={"request_id": "s2", "count": 1})
        oversized_prompt = "x" * (CortexTrainingClient._MAX_GENERATE_BYTES + 1)
        with pytest.raises(ValueError, match="exceeds the maximum allowed size"):
            c.generate_stream("j1", prompts=[oversized_prompt])
        c._session.post.assert_not_called()

    def test_generate_rejects_overlong_tokenized_prompt(self):
        # Sampling sub-job exposes max_seq_len=4 (proto numbers arrive as floats).
        c = _make_client(
            post_json={"request_id": "g5"},
            get_json={"sub_jobs": [{"job_type": "sampling", "inference_config": {"max_seq_len": 4.0, "n_gpus": 1.0}}]},
        )
        # 4 tokens == max_seq_len leaves no room for output -> rejected.
        with pytest.raises(ValueError, match="does not fit the sampling job's max_seq_len of 4"):
            c.generate("j1", prompts=[[1, 2, 3, 4]])
        c._session.post.assert_not_called()

    def test_generate_allows_tokenized_prompt_under_limit(self):
        c = _make_client(
            post_json={"request_id": "g6"},
            get_json={"sub_jobs": [{"job_type": "sampling", "inference_config": {"max_seq_len": 4.0, "n_gpus": 1.0}}]},
        )
        # 3 tokens < max_seq_len=4 -> accepted.
        assert c.generate("j1", prompts=[[1, 2, 3]]) == "g6"
        c._session.post.assert_called_once()

    def test_generate_caches_max_seq_len_across_calls(self):
        c = _make_client(
            post_json={"request_id": "g7"},
            get_json={"sub_jobs": [{"job_type": "sampling", "inference_config": {"max_seq_len": 8.0, "n_gpus": 1.0}}]},
        )
        c.generate("j1", prompts=[[1, 2]])
        c.generate("j1", prompts=[[3, 4]])
        # Only the first call should have hit get_job; the value is cached.
        assert c._session.get.call_count == 1

    # ─── sub_job_id addressing ───────────────────────────────────────────
    #
    # The id travels as a query param because the body is an opaque DSSST1 frame.
    # generate chunks that frame; generate_stream sends it in one POST — two
    # transports, so both need covering.

    def test_generate_sends_sub_job_id_query_param(self):
        c = _make_client(post_json={"request_id": "sj1"})
        assert c.generate("j1", prompts=["hi"], sub_job_id="j1:sampling:1") == "sj1"
        _, kwargs = c._session.post.call_args
        assert kwargs["params"] == {"sub_job_id": "j1:sampling:1"}

    def test_generate_omits_params_when_no_sub_job_id(self):
        # No param key at all, so the request stays byte-identical for old callers.
        c = _make_client(post_json={"request_id": "sj2"})
        c.generate("j1", prompts=["hi"])
        _, kwargs = c._session.post.call_args
        assert "params" not in kwargs

    def test_generate_sends_sub_job_id_on_every_chunk(self):
        # The chunk loop reuses one URL, so the param has to be attached per send —
        # a server that only saw it on the final chunk could route the rest wrongly.
        c = _make_client()
        c._MAX_GENERATE_BYTES = 64 * 1024
        c._session.post.side_effect = [
            _make_response({"chunk_cached": True}),
            _make_response({"request_id": "sj3"}),
        ]
        oversized_prompt = "x" * (70 * 1024)

        assert c.generate("j1", prompts=[oversized_prompt], sub_job_id="j1:sampling:0") == "sj3"
        assert c._session.post.call_count == 2
        for call in c._session.post.call_args_list:
            assert call.kwargs["params"] == {"sub_job_id": "j1:sampling:0"}

    def test_generate_stream_sends_sub_job_id_query_param(self):
        c = _make_client(post_json={"request_id": "sj4", "count": 1})
        c.generate_stream("j1", prompts=["hi"], sub_job_id="j1:sampling:1")
        url, kwargs = c._session.post.call_args
        assert url[0] == f"{c._prefix}/j1/generate-stream"
        assert kwargs["params"] == {"sub_job_id": "j1:sampling:1"}

    def test_generate_stream_omits_params_when_no_sub_job_id(self):
        c = _make_client(post_json={"request_id": "sj5", "count": 1})
        c.generate_stream("j1", prompts=["hi"])
        _, kwargs = c._session.post.call_args
        assert "params" not in kwargs

    # ─── per-sub-job max_seq_len ─────────────────────────────────────────

    def test_max_seq_len_ignores_log_prob_sub_job(self):
        # A log_prob sub-job also carries an inference_config, but its window says
        # nothing about where a generate runs. With only log_prob present there is
        # no sampling window to check, so validation is skipped rather than applied
        # to the wrong sub-job — an 8-token prompt must not be rejected against 4.
        c = _make_client(
            post_json={"request_id": "m1"},
            get_json={
                "sub_jobs": [
                    {"job_type": "log_probability", "inference_config": {"max_seq_len": 4.0}},
                ]
            },
        )
        assert c.generate("j1", prompts=[[1, 2, 3, 4, 5, 6, 7, 8]]) == "m1"

    def test_max_seq_len_ignores_log_prob_even_when_named(self):
        # Naming a log_prob sub-job must not resolve its window either.
        c = _make_client(
            post_json={"request_id": "m1b"},
            get_json={
                "sub_jobs": [
                    {"sub_job_id": "j1:log_prob:0", "job_type": "log_probability",
                     "inference_config": {"max_seq_len": 4.0}},
                ]
            },
        )
        assert c.generate(
            "j1", prompts=[[1, 2, 3, 4, 5, 6, 7, 8]], sub_job_id="j1:log_prob:0"
        ) == "m1b"

    def test_max_seq_len_uses_the_named_sub_job(self):
        c = _make_client(
            post_json={"request_id": "m2"},
            get_json={
                "sub_jobs": [
                    {"sub_job_id": "j1:sampling:0", "job_type": "sampling",
                     "inference_config": {"max_seq_len": 4.0}},
                    {"sub_job_id": "j1:sampling:1", "job_type": "sampling",
                     "inference_config": {"max_seq_len": 64.0}},
                ]
            },
        )
        # Same prompt: rejected against :0, accepted against :1.
        with pytest.raises(ValueError, match="max_seq_len of 4"):
            c.generate("j1", prompts=[[1, 2, 3, 4, 5]], sub_job_id="j1:sampling:0")
        assert c.generate("j1", prompts=[[1, 2, 3, 4, 5]], sub_job_id="j1:sampling:1") == "m2"

    def test_max_seq_len_skips_when_ambiguous(self):
        # Two sampling sub-jobs and no sub_job_id: guessing risks a wrong
        # client-side rejection, so validation is skipped and the server decides.
        c = _make_client(
            post_json={"request_id": "m3"},
            get_json={
                "sub_jobs": [
                    {"sub_job_id": "j1:sampling:0", "job_type": "sampling",
                     "inference_config": {"max_seq_len": 4.0}},
                    {"sub_job_id": "j1:sampling:1", "job_type": "sampling",
                     "inference_config": {"max_seq_len": 64.0}},
                ]
            },
        )
        assert c.generate("j1", prompts=[[1, 2, 3, 4, 5]]) == "m3"

    def test_max_seq_len_cache_is_keyed_per_sub_job(self):
        c = _make_client(
            post_json={"request_id": "m4"},
            get_json={
                "sub_jobs": [
                    {"sub_job_id": "j1:sampling:0", "job_type": "sampling",
                     "inference_config": {"max_seq_len": 4.0}},
                    {"sub_job_id": "j1:sampling:1", "job_type": "sampling",
                     "inference_config": {"max_seq_len": 64.0}},
                ]
            },
        )
        c.generate("j1", prompts=[[1, 2]], sub_job_id="j1:sampling:1")
        # A different sub-job must not read :1's cached value.
        with pytest.raises(ValueError, match="max_seq_len of 4"):
            c.generate("j1", prompts=[[1, 2, 3, 4, 5]], sub_job_id="j1:sampling:0")

    def test_generate_skips_validation_for_string_prompts(self):
        # String prompts are not validated client-side, so get_job is not called.
        c = _make_client(post_json={"request_id": "g8"})
        assert c.generate("j1", prompts=["a very long prompt"]) == "g8"
        c._session.get.assert_not_called()

    def test_generate_skips_when_no_sampling_config(self):
        # A job without an inference_config (e.g. training-only) disables the
        # check; the request is sent through unvalidated.
        c = _make_client(
            post_json={"request_id": "g9"},
            get_json={"sub_jobs": [{"training_config": {"max_seq_len": 2.0}}]},
        )
        assert c.generate("j1", prompts=[[1, 2, 3, 4, 5]]) == "g9"
        c._session.post.assert_called_once()

    def test_generate_skips_when_get_job_fails(self):
        # A transient get_job failure must not break generate, and must not be
        # cached as a permanent miss.
        c = _make_client(post_json={"request_id": "g10"})
        c._session.get.side_effect = RuntimeError("boom")
        assert c.generate("j1", prompts=[[1, 2, 3]]) == "g10"
        assert "j1" not in c._sampling_max_seq_len

    def test_generate_stream_rejects_overlong_tokenized_prompt(self):
        c = _make_client(
            post_json={"request_id": "s3", "count": 1},
            get_json={"sub_jobs": [{"job_type": "sampling", "inference_config": {"max_seq_len": 4.0, "n_gpus": 1.0}}]},
        )
        with pytest.raises(ValueError, match="does not fit the sampling job's max_seq_len of 4"):
            c.generate_stream("j1", prompts=[[1, 2, 3, 4, 5]])
        c._session.post.assert_not_called()

    def test_generate_rejects_overlong_flat_tokenized_prompt(self):
        # A flat list of ids is one prompt, so the reported index is 0 even
        # though the list has four elements.
        c = _make_client(
            post_json={"request_id": "g-flat"},
            get_json={"sub_jobs": [{"job_type": "sampling", "inference_config": {"max_seq_len": 4.0, "n_gpus": 1.0}}]},
        )
        with pytest.raises(ValueError, match="prompt at index 0 has 4 tokens"):
            c.generate("j1", prompts=[1, 2, 3, 4])
        c._session.post.assert_not_called()

    def test_generate_rejects_mixed_flat_token_list(self):
        c = _make_client(
            post_json={"request_id": "g-mixed"},
            get_json={"sub_jobs": [{"job_type": "sampling", "inference_config": {"max_seq_len": 8.0, "n_gpus": 1.0}}]},
        )
        with pytest.raises(ValueError, match=r"prompts\[1\] must be an integer token id"):
            c.generate("j1", prompts=[1, 2.5])
        c._session.post.assert_not_called()

    def test_generate_allows_bare_tensor_prompt_under_limit(self):
        c = _make_client(
            post_json={"request_id": "g-tensor"},
            get_json={"sub_jobs": [{"job_type": "sampling", "inference_config": {"max_seq_len": 4.0, "n_gpus": 1.0}}]},
        )
        assert c.generate("j1", prompts=torch.tensor([1, 2, 3], dtype=torch.int32)) == "g-tensor"
        c._session.post.assert_called_once()

    def test_generate_rejects_overlong_bare_tensor_prompt(self):
        c = _make_client(
            post_json={"request_id": "g-tensor-long"},
            get_json={"sub_jobs": [{"job_type": "sampling", "inference_config": {"max_seq_len": 4.0, "n_gpus": 1.0}}]},
        )
        with pytest.raises(ValueError, match="prompt at index 0 has 4 tokens"):
            c.generate("j1", prompts=torch.tensor([1, 2, 3, 4], dtype=torch.int32))
        c._session.post.assert_not_called()

    def test_generate_stream_allows_bare_tensor_prompt_under_limit(self):
        c = _make_client(
            post_json={"request_id": "s-tensor", "count": 1},
            get_json={"sub_jobs": [{"job_type": "sampling", "inference_config": {"max_seq_len": 4.0, "n_gpus": 1.0}}]},
        )
        assert c.generate_stream("j1", prompts=torch.tensor([1, 2, 3], dtype=torch.int32)) == {
            "request_id": "s-tensor",
            "count": 1,
        }
        c._session.post.assert_called_once()

    def test_step_no_lr(self):
        c = _make_client(post_json={"request_id": "r2"})
        c.step("j1")
        kwargs = c._session.post.call_args.kwargs
        assert kwargs["json"] == {}

    def test_step_with_lr(self):
        c = _make_client(post_json={"request_id": "r2"})
        c.step("j1", learning_rate=2e-5)
        assert c._session.post.call_args.kwargs["json"] == {"learning_rate": 2e-5}

    def test_save_omits_optionals(self):
        c = _make_client(post_json={"request_id": "r3"})
        c.save("j1")
        assert c._session.post.call_args.kwargs["json"] == {}

    def test_save_with_optionals(self):
        c = _make_client(post_json={"request_id": "r3"})
        c.save("j1", checkpoint_id="cp", checkpoint_type="WEIGHTS-ONLY")
        assert c._session.post.call_args.kwargs["json"] == {
            "checkpoint_id": "cp",
            "checkpoint_type": "weights-only",
        }

    def test_save_rejects_unknown_checkpoint_type(self):
        c = _make_client(post_json={"request_id": "r3"})
        with pytest.raises(ValueError, match="resumable.*weights-only"):
            c.save("j1", checkpoint_type="full")
        c._session.post.assert_not_called()

    def test_load_minimal(self):
        c = _make_client(post_json={"request_id": "r-load"})
        rid = c.load("j1", checkpoint_id="cp")

        assert rid == "r-load"
        c._session.post.assert_called_once_with(f"{c._prefix}/j1/load", json={"checkpoint_id": "cp"})

    def test_load_with_source_job(self):
        c = _make_client(post_json={"request_id": "r-load"})
        c.load(
            "j1",
            checkpoint_id="cp",
            source_job_id="source-job",
        )

        assert c._session.post.call_args.kwargs["json"] == {
            "checkpoint_id": "cp",
            "source_job_id": "source-job",
        }

    def test_load_with_target_sub_job(self):
        c = _make_client(post_json={"request_id": "r-load"})
        rid = c.load("j1", checkpoint_id="cp", target_sub_job_id="j1:training:0")

        assert rid == "r-load"
        c._session.post.assert_called_once_with(
            f"{c._prefix}/j1/load",
            json={"checkpoint_id": "cp", "target_sub_job_id": "j1:training:0"},
        )

    def test_load_omits_target_sub_job_when_none(self):
        c = _make_client(post_json={"request_id": "r-load"})
        c.load("j1", checkpoint_id="cp", target_sub_job_id=None)

        assert c._session.post.call_args.kwargs["json"] == {"checkpoint_id": "cp"}

    def test_load_with_source_job_and_target_sub_job(self):
        c = _make_client(post_json={"request_id": "r-load"})
        c.load(
            "j1",
            checkpoint_id="cp",
            source_job_id="source-job",
            target_sub_job_id="j1:training:1",
        )

        assert c._session.post.call_args.kwargs["json"] == {
            "checkpoint_id": "cp",
            "source_job_id": "source-job",
            "target_sub_job_id": "j1:training:1",
        }

    def test_load_target_sub_job_wrong_session_returns_400(self):
        """Server returns 400 when target_sub_job_id belongs to different session."""
        c = _make_client()
        c.max_retries = 0  # Skip retry backoff for error tests
        resp = _make_response(
            json_body={"error": "target_sub_job_id 'other-job:training:0' does not belong to session 'j1'"},
            status_code=400,
        )
        resp.raise_for_status.side_effect = nc.requests.exceptions.HTTPError(response=resp)
        c._session.post.return_value = resp

        with pytest.raises(nc.requests.exceptions.HTTPError) as exc_info:
            c.load("j1", checkpoint_id="cp", target_sub_job_id="other-job:training:0")

        assert exc_info.value.response.status_code == 400

    def test_load_target_sub_job_sampling_returns_501(self):
        """Server returns 501 when target_sub_job_id is a sampling sub-job."""
        c = _make_client()
        c.max_retries = 0
        resp = _make_response(
            json_body={
                "error": (
                    "target_sub_job_id 'j1:sampling:0' is a sampling sub-job; loading into non-training zone not"
                    " supported"
                )
            },
            status_code=501,
        )
        resp.raise_for_status.side_effect = nc.requests.exceptions.HTTPError(response=resp)
        c._session.post.return_value = resp

        with pytest.raises(nc.requests.exceptions.HTTPError) as exc_info:
            c.load("j1", checkpoint_id="cp", target_sub_job_id="j1:sampling:0")

        assert exc_info.value.response.status_code == 501

    def test_load_target_sub_job_not_found_returns_400(self):
        """Server returns 400 when target_sub_job_id doesn't exist in session."""
        c = _make_client()
        c.max_retries = 0
        resp = _make_response(
            json_body={"error": "target_sub_job_id 'j1:training:99' does not belong to session 'j1'"}, status_code=400
        )
        resp.raise_for_status.side_effect = nc.requests.exceptions.HTTPError(response=resp)
        c._session.post.return_value = resp

        with pytest.raises(nc.requests.exceptions.HTTPError) as exc_info:
            c.load("j1", checkpoint_id="cp", target_sub_job_id="j1:training:99")

        assert exc_info.value.response.status_code == 400

    def test_load_empty_string_target_sub_job_id_sent_to_server(self):
        """Documents that empty string target_sub_job_id is sent to server (not filtered).

        Server will reject it. This test documents current behavior rather than
        adding client-side validation (thin client design).
        """
        c = _make_client(post_json={"request_id": "r-load"})
        c.load("j1", checkpoint_id="cp", target_sub_job_id="")

        # Empty string is sent in the request body
        assert c._session.post.call_args.kwargs["json"] == {
            "checkpoint_id": "cp",
            "target_sub_job_id": "",
        }

    def test_load_no_training_sub_job_in_session(self):
        """Server returns 400 when session has no training sub-job and target is omitted."""
        c = _make_client()
        c.max_retries = 0
        resp = _make_response(json_body={"error": "session 'j1' has no training sub-job"}, status_code=400)
        resp.raise_for_status.side_effect = nc.requests.exceptions.HTTPError(response=resp)
        c._session.post.return_value = resp

        with pytest.raises(nc.requests.exceptions.HTTPError) as exc_info:
            c.load("j1", checkpoint_id="cp")  # No target, sampling-only session

        assert exc_info.value.response.status_code == 400

    def test_weight_sync(self):
        c = _make_client(post_json={"request_id": "r4"})
        rid = c.weight_sync(
            "j1",
            source_sub_job_id="j1:training:0",
            target_sub_job_ids=["j1:sampling:0", "j1:sampling:1"],
        )
        assert rid == "r4"
        url, kwargs = c._session.post.call_args
        assert url[0] == f"{c._prefix}/j1/operation"
        assert kwargs["json"] == {
            "operation_type": "weight-sync",
            "sub_job_id": "j1:training:0",
            "sub_job_type": "training",
            "payload": {
                "source_sub_job_id": "j1:training:0",
                "target_sub_job_ids": ["j1:sampling:0", "j1:sampling:1"],
            },
        }

    def test_weight_sync_includes_lora_weight_format(self):
        c = _make_client(post_json={"request_id": "r4-lora"})
        rid = c.weight_sync(
            "j1",
            source_sub_job_id="j1:training:0",
            target_sub_job_ids=["j1:sampling:0"],
            weight_format="lora",
        )
        assert rid == "r4-lora"
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "weight-sync",
            "sub_job_id": "j1:training:0",
            "sub_job_type": "training",
            "payload": {
                "source_sub_job_id": "j1:training:0",
                "target_sub_job_ids": ["j1:sampling:0"],
                "weight_format": "lora",
            },
        }

    def test_weight_sync_accepts_explicit_operation_route(self):
        c = _make_client(post_json={"request_id": "r4"})
        c.weight_sync(
            "j1",
            source_sub_job_id="j1:training:0",
            target_sub_job_ids=["j1:sampling:0"],
            sub_job_id="j1:training:1",
            sub_job_type="training",
        )
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "weight-sync",
            "sub_job_id": "j1:training:1",
            "sub_job_type": "training",
            "payload": {
                "source_sub_job_id": "j1:training:0",
                "target_sub_job_ids": ["j1:sampling:0"],
            },
        }

    def test_forward_operation(self):
        c = _make_client(post_json={"request_id": "r-forward"})
        out = c.forward(
            "j1",
            {"tokens": [1, 2, 3]},
            sub_job_id="j1:training:0",
            sub_job_type="training",
        )
        assert out == {"request_id": "r-forward"}
        assert "r-forward" in c._forward_request_ids
        c._session.post.assert_called_once_with(
            f"{c._prefix}/j1/operation",
            json={
                "operation_type": "forward",
                "sub_job_id": "j1:training:0",
                "sub_job_type": "training",
                "payload": {"tokens": [1, 2, 3]},
            },
        )

    def test_forward_operation_omits_none_fields(self):
        c = _make_client(post_json={"ok": True})
        c.forward("j1")
        assert not c._forward_request_ids
        c._session.post.assert_called_once_with(
            f"{c._prefix}/j1/operation",
            json={"operation_type": "forward"},
        )

    def test_forward_wraps_bytes_payload(self):
        c = _make_client(post_json={"request_id": "r-forward"})
        out = c.forward("j1", b"\x03\x04", sub_job_id="j1:training:0")
        assert out == {"request_id": "r-forward"}
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "forward",
            "sub_job_id": "j1:training:0",
            "payload": {
                "payload_b64": base64.b64encode(b"\x03\x04").decode("ascii"),
                "content_type": "application/octet-stream",
            },
        }

    def test_fwd_alias_uses_forward_operation(self):
        c = _make_client(post_json={"request_id": "r-forward"})
        out = c.fwd("j1", b"\x05\x06", sub_job_id="j1:training:0")
        assert out == {"request_id": "r-forward"}
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "forward",
            "sub_job_id": "j1:training:0",
            "payload": {
                "payload_b64": base64.b64encode(b"\x05\x06").decode("ascii"),
                "content_type": "application/octet-stream",
            },
        }

    def test_fwd_no_grad_alias_uses_forward_operation(self):
        c = _make_client(post_json={"request_id": "r-forward"})
        out = c.fwd_no_grad("j1", b"\x07\x08", sub_job_type="training")
        assert out == {"request_id": "r-forward"}
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "forward",
            "sub_job_type": "training",
            "payload": {
                "payload_b64": base64.b64encode(b"\x07\x08").decode("ascii"),
                "content_type": "application/octet-stream",
            },
        }

    def test_forward_chunk_envelope_keys_match_the_backend_reader(self, monkeypatch):
        """Pin the envelope shape the backend parses: payload_b64 + content_type."""
        from cortex_training import wire

        monkeypatch.setenv(nc._OPERATION_MAX_JSON_BYTES_ENV, str(64 * 1024))
        c = _make_client(post_json={"chunk_cached": True})
        frame = wire.dumps({"input_ids": torch.zeros(1, 60_000, dtype=torch.long)})
        with contextlib.suppress(RuntimeError):
            c.forward("j1", frame, sub_job_id="j1:training:0")

        assert c._session.post.call_count > 1
        for call in c._session.post.call_args_list:
            payload = call.kwargs["json"]["payload"]
            assert set(payload) == {"payload_b64", "content_type"}
            assert payload["content_type"] == "application/octet-stream"
            assert isinstance(payload["payload_b64"], str)

    def test_forward_sends_a_fitting_frame_as_one_unchunked_post(self):
        # The server assembler rejects total_chunks <= 1, so a fitting frame
        # is posted raw.
        from cortex_training import wire

        c = _make_client(post_json={"request_id": "r-forward"})
        frame = wire.dumps({"input_ids": torch.zeros(2, 4, dtype=torch.long)})
        c.forward("j1", frame, sub_job_id="j1:training:0")
        assert c._session.post.call_count == 1
        payload = c._session.post.call_args.kwargs["json"]["payload"]
        sent = base64.b64decode(payload["payload_b64"])
        assert sent == frame
        assert wire.read_byte_chunk_metadata(sent) is None
        assert "r-forward" in c._forward_request_ids

    def test_forward_chunks_an_oversized_frame_across_operation_posts(self, monkeypatch):
        from cortex_training import wire

        max_json = 64 * 1024
        monkeypatch.setenv(nc._OPERATION_MAX_JSON_BYTES_ENV, str(max_json))
        frame = wire.dumps({"input_ids": torch.zeros(1, 60_000, dtype=torch.long)})
        expected_chunks = len(
            wire.encode_byte_chunks(
                frame,
                kind="request",
                operation="forward",
                max_bytes=nc._operation_chunk_max_bytes(max_json),
            )
        )
        assert expected_chunks > 1

        c = _make_client()
        bodies = [{"chunk_cached": True}] * (expected_chunks - 1) + [{"request_id": "r-forward"}]
        c._session.post.side_effect = [_make_response(body) for body in bodies]

        out = c.forward("j1", frame, sub_job_id="j1:training:0")
        assert out == {"request_id": "r-forward"}
        assert "r-forward" in c._forward_request_ids
        assert c._session.post.call_count == expected_chunks

        chunks = []
        for call in c._session.post.call_args_list:
            body = call.kwargs["json"]
            assert body["operation_type"] == "forward"
            assert body["sub_job_id"] == "j1:training:0"
            assert len(json.dumps(body).encode("utf-8")) <= 64 * 1024
            chunks.append(base64.b64decode(body["payload"]["payload_b64"]))

        descriptors = [wire.read_byte_chunk_metadata(chunk) for chunk in chunks]
        assert all(desc is not None for desc in descriptors)
        assert {desc["operation"] for desc in descriptors} == {"forward"}
        assert len({desc["chunk_group_id"] for desc in descriptors}) == 1
        assert wire.decode_byte_chunks(chunks, kind="request") == frame

    def test_forward_raises_when_a_non_final_chunk_returns_a_request_id(self, monkeypatch):
        from cortex_training import wire

        monkeypatch.setenv(nc._OPERATION_MAX_JSON_BYTES_ENV, str(64 * 1024))
        c = _make_client(post_json={"request_id": "r-forward"})
        frame = wire.dumps({"input_ids": torch.zeros(1, 60_000, dtype=torch.long)})
        with pytest.raises(RuntimeError, match="chunk 0 unexpectedly returned request_id"):
            c.forward("j1", frame)
        assert "r-forward" not in c._forward_request_ids

    def test_operation_rejects_a_body_over_the_declared_ceiling(self):
        c = _make_client(post_json={})
        with pytest.raises(ValueError, match="exceeds the .* limit for one /operation"):
            c._operation("j1", "forward", payload=b"\x00" * 4096, max_json_bytes=1024)
        c._session.post.assert_not_called()

    def test_forward_does_not_retry_a_chunk_group_conflict(self, monkeypatch):
        from cortex_training import wire

        monkeypatch.setenv(nc._OPERATION_MAX_JSON_BYTES_ENV, str(64 * 1024))
        c = _make_client()
        conflict = _make_error_response(
            {
                "code": "chunk_group_missing_chunks",
                "message": "request chunk group is missing chunks",
                "chunk_group_id": "grp-1",
                "missing_chunks": [0, 1],
            }
        )
        c._session.post.return_value = conflict
        frame = wire.dumps({"input_ids": torch.zeros(1, 60_000, dtype=torch.long)})

        with pytest.raises(nc.ChunkGroupConflictError) as excinfo:
            c.forward("j1", frame)
        assert excinfo.value.detail["missing_chunks"] == [0, 1]
        assert c._session.post.call_count == 1

    def test_operation_body_size_is_measured_without_copying_the_payload(self):
        body = {
            "operation_type": "forward",
            "sub_job_id": "j1:training:0",
            "payload": {"payload_b64": "QUJD" * 1000, "content_type": "application/octet-stream"},
        }
        assert nc._operation_body_json_bytes(body) == len(json.dumps(body).encode("utf-8"))
        no_payload = {"operation_type": "step"}
        assert nc._operation_body_json_bytes(no_payload) == len(
            json.dumps(no_payload).encode("utf-8")
        )

    def test_operation_without_a_declared_ceiling_is_unbounded(self):
        c = _make_client(post_json={"ok": True})
        assert c._operation("j1", "weight-sync", payload=b"\x00" * 4096) == {"ok": True}
        c._session.post.assert_called_once()

    def test_forward_operation_json_ceiling_env_must_be_sane(self, monkeypatch):
        monkeypatch.setenv(nc._OPERATION_MAX_JSON_BYTES_ENV, "not-a-number")
        with pytest.raises(ValueError, match="must be an integer"):
            nc._operation_max_json_bytes()
        monkeypatch.setenv(
            nc._OPERATION_MAX_JSON_BYTES_ENV,
            str(nc._MIN_OPERATION_MAX_JSON_BYTES - 1),
        )
        with pytest.raises(ValueError, match="must be at least 16384 bytes"):
            nc._operation_max_json_bytes()
        monkeypatch.setenv(
            nc._OPERATION_MAX_JSON_BYTES_ENV,
            str(nc._MIN_OPERATION_MAX_JSON_BYTES),
        )
        assert nc._operation_max_json_bytes() == 16 * 1024
        assert nc._operation_chunk_max_bytes(nc._operation_max_json_bytes()) == 9 * 1024

    def test_forward_dict_payload_still_takes_the_plain_operation_path(self):
        c = _make_client(post_json={"request_id": "r-forward"})
        c.forward("j1", {"probe": True}, sub_job_type="training")
        assert c._session.post.call_count == 1
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "forward",
            "sub_job_type": "training",
            "payload": {"probe": True},
        }

    def test_operation_debug_log_omits_the_base64_payload(self, caplog, capsys):
        c = _make_client(post_json={"request_id": "r-forward"})
        with caplog.at_level(logging.DEBUG, logger=nc.logger.name):
            c.forward("j1", b"\xab" * 4096)
        assert "base64 chars>" in caplog.text
        assert base64.b64encode(b"\xab" * 4096).decode("ascii") not in caplog.text
        assert capsys.readouterr().out == ""

    def test_bootstrap_router_replay(self):
        c = _make_client(post_json={"request_id": "r-bootstrap"})
        out = c.bootstrap_router_replay(
            "j1",
            "j1:training:0",
            "j1:sampling:0",
            max_cache_bytes=4096,
            sub_job_id="j1:sampling:0",
            sub_job_type="sampling",
        )
        assert out == {"request_id": "r-bootstrap"}
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "bootstrap-router-replay",
            "sub_job_id": "j1:sampling:0",
            "sub_job_type": "sampling",
            "payload": {
                "source_sub_job_id": "j1:training:0",
                "target_sub_job_id": "j1:sampling:0",
                "max_cache_bytes": 4096,
            },
        }

    def test_router_replay_discard_sends_operation(self):
        c = _make_client(post_json={"status": "discarded"})
        out = c.router_replay_discard(
            "j1",
            ["sample-1", "sample-2"],
            sub_job_id="j1:sampling:0",
            sub_job_type="sampling",
        )
        assert out == {"status": "discarded"}
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "router-replay-discard",
            "sub_job_id": "j1:sampling:0",
            "sub_job_type": "sampling",
            "payload": {"sample_ids": ["sample-1", "sample-2"]},
        }

    def test_router_replay_discard_accepts_extra_payload(self):
        c = _make_client(post_json={"status": "discarded"})
        out = c.router_replay_discard(
            "j1",
            sub_job_type="sampling",
            extra={"sample_ids": ["sample-extra"], "reason": "test"},
        )
        assert out == {"status": "discarded"}
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "router-replay-discard",
            "sub_job_type": "sampling",
            "payload": {"sample_ids": ["sample-extra"], "reason": "test"},
        }

    def test_reset_prefix_cache(self):
        c = _make_client(post_json={"reset_ok": True})
        out = c.reset_prefix_cache(
            "j1",
            sub_job_id="j1:sampling:0",
            sub_job_type="sampling",
            drain=False,
            timeout_s=12.5,
            retry_interval_s=0.25,
        )
        assert out == {"reset_ok": True}
        url, kwargs = c._session.post.call_args
        assert url[0] == f"{c._prefix}/j1/operation"
        assert kwargs["json"] == {
            "operation_type": "reset-prefix-cache",
            "sub_job_id": "j1:sampling:0",
            "sub_job_type": "sampling",
            "payload": {
                "drain": False,
                "timeout_s": 12.5,
                "retry_interval_s": 0.25,
            },
        }

    def test_reset_prefix_cache_preserves_extra_payload(self):
        c = _make_client(post_json={"request_id": "r-reset"})
        c.reset_prefix_cache(
            "j1",
            sub_job_id="j1:sampling:0",
            extra={"target_sub_job_ids": ["j1:sampling"], "drain": False},
        )
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "reset-prefix-cache",
            "sub_job_id": "j1:sampling:0",
            "sub_job_type": "sampling",
            "payload": {
                "target_sub_job_ids": ["j1:sampling"],
                "drain": False,
                "timeout_s": 60.0,
                "retry_interval_s": 0.1,
            },
        }


# ─── Pre-tokenized prompt packing ────────────────────────────────────────


class TestPromptTensorPacking:
    def test_batch_of_token_lists_becomes_int32_tensors(self):
        packed = nc._pack_token_prompts([[1, 2], [3, 4, 5]])
        assert [p.dtype for p in packed] == [torch.int32, torch.int32]
        assert [p.tolist() for p in packed] == [[1, 2], [3, 4, 5]]

    def test_string_prompts_are_left_alone(self):
        assert nc._pack_token_prompts(["a", "b"]) == ["a", "b"]

    def test_mixed_batch_packs_only_token_lists(self):
        packed = nc._pack_token_prompts(["a", [1, 2]])
        assert packed[0] == "a"
        assert packed[1].tolist() == [1, 2]

    def test_empty_inner_list_stays_a_list(self):
        # The server's own non-empty check should be the one that rejects it.
        assert nc._pack_token_prompts([[]]) == [[]]

    def test_mixed_flat_list_is_rejected(self):
        with pytest.raises(ValueError, match=r"prompts\[1\] must be an integer token id"):
            nc._pack_token_prompts([1, 2.5])

    def test_bool_list_is_not_packed_as_token_ids(self):
        assert nc._pack_token_prompts([True, False]) == [True, False]

    def test_flat_token_list_is_one_prompt(self):
        packed = nc._pack_token_prompts([7, 8, 9])
        assert torch.is_tensor(packed)
        assert packed.dtype == torch.int32
        assert packed.tolist() == [7, 8, 9]

    def test_empty_and_non_list_prompts_are_unchanged(self):
        assert nc._pack_token_prompts([]) == []
        assert nc._pack_token_prompts("hello") == "hello"

    def test_existing_tensor_is_not_recast(self):
        tensor = torch.tensor([1, 2], dtype=torch.int64)
        assert nc._pack_token_prompts(tensor) is tensor
        assert nc._pack_token_prompts([tensor])[0] is tensor

    def test_token_id_beyond_int32_falls_back_to_int64(self, caplog):
        caplog.set_level(logging.WARNING, logger=nc.logger.name)
        packed = nc._pack_token_prompts([[2**31]])
        assert packed[0].dtype == torch.int64
        assert packed[0].tolist() == [2**31]
        assert "int32" in caplog.text

    def test_env_var_keeps_list_prompts_inside_the_frame(self, monkeypatch):
        # The hatch changes the encoding of the prompts, not the body: both
        # calls still post a DSSST1 frame.
        monkeypatch.setenv(nc.DISABLE_TENSOR_PROMPTS_ENV, "1")

        unary = _make_client(post_json={"request_id": "g-raw"})
        unary.generate("j1", prompts=[[1, 2], [3, 4]])
        stream = _make_client(post_json={"request_id": "s-raw", "count": 2})
        stream.generate_stream("j1", prompts=[[1, 2], [3, 4]])

        for client in (unary, stream):
            kwargs = client._session.post.call_args.kwargs
            assert kwargs["headers"]["Content-Type"] == "application/octet-stream"
            assert _wire_load(kwargs["data"])["prompts"] == [[1, 2], [3, 4]]


# ─── CortexTrainingClient — request polling ──────────────────────────────────


class TestRequestPolling:
    def test_get_request_status(self):
        c = _make_client(get_json={"request_id": "r1", "status": "running"})
        out = c.get_request_status("j1", "r1")
        assert out["status"] == "running"
        c._session.get.assert_called_once_with(
            f"{c._prefix}/j1/requests/r1",
            params=None,
        )

    def test_get_request_status_with_stream_max_events(self):
        c = _make_client(get_json={"request_id": "r1", "status": "streaming"})
        c.get_request_status("j1", "r1", max_events=64)
        c._session.get.assert_called_once_with(
            f"{c._prefix}/j1/requests/r1",
            params={"max_events": 64},
        )

    def test_get_request_status_with_cursor(self):
        c = _make_client(get_json={"request_id": "r1", "status": "streaming"})
        c.get_request_status("j1", "r1", max_events=64, cursor="2")
        c._session.get.assert_called_once_with(
            f"{c._prefix}/j1/requests/r1",
            params={"max_events": 64, "cursor": "2"},
        )

    def test_get_request_status_decodes_stream_dssst1_result_events(self):
        from cortex_training import wire

        payload = wire.dumps(
            {
                "text": "ok",
                "token_ids": torch.tensor([1, 2, 3], dtype=torch.int64),
                "logprobs": torch.tensor([0.125, -0.5], dtype=torch.float32),
                "metrics": {"model/tokens": 3},
            }
        )
        c = _make_client(
            get_json={
                "request_id": "r1",
                "status": "streaming",
                "events": [
                    {
                        "type": "result",
                        "index": 0,
                        "result": {
                            "content_type": "application/octet-stream",
                            "encoding": "base64",
                            "wire_format": "DSSST1",
                            "payload_b64": base64.b64encode(payload).decode("ascii"),
                            "metrics": {"router_replay/tx_bytes_avg": 12.0},
                        },
                    },
                    {"type": "done", "completed": 1, "failed": 0},
                ],
            }
        )

        assert c.get_request_status("j1", "r1")["events"] == [
            {
                "type": "result",
                "index": 0,
                "result": {
                    "text": "ok",
                    "token_ids": [1, 2, 3],
                    "logprobs": [0.125, -0.5],
                    "metrics": {
                        "model/tokens": 3,
                        "router_replay/tx_bytes_avg": 12.0,
                    },
                },
            },
            {"type": "done", "completed": 1, "failed": 0},
        ]

    def test_cancel_request(self):
        c = _make_client(post_json={"request_id": "r1"})
        out = c.cancel_request("j1", "r1", sub_job_id="j1:sampling:0")
        assert out["request_id"] == "r1"
        c._session.post.assert_called_once_with(
            f"{c._prefix}/j1/operation",
            json={
                "operation_type": "cancel-request",
                "sub_job_id": "j1:sampling:0",
                "payload": {"request_id": "r1"},
            },
        )

    def test_cancel_request_fans_out_without_sub_job_hint(self):
        c = _make_client(post_json={"request_id": "r1"})
        c.cancel_request("j1", "r1")
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "cancel-request",
            "payload": {"request_id": "r1"},
        }

    def test_poll_request_done(self, monkeypatch):
        c = _make_client(get_json={"status": "done", "result": {"avg_loss": 0.5}})
        sleep = MagicMock()
        monkeypatch.setattr(nc.time, "sleep", sleep)
        assert c.poll_request("j1", "r1") == {"avg_loss": 0.5}
        sleep.assert_not_called()

    def test_poll_request_done_no_result_returns_empty(self, monkeypatch):
        c = _make_client(get_json={"status": "done"})
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        assert c.poll_request("j1", "r1") == {}

    def test_poll_request_decodes_dssst1_result_payload(self, monkeypatch):
        from cortex_training import wire

        payload = wire.dumps({"avg_loss": 0.5})
        c = _make_client(
            get_json={
                "status": "done",
                "result": {
                    "content_type": "application/octet-stream",
                    "encoding": "base64",
                    "wire_format": "DSSST1",
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                    "metrics": {"router_replay/tx_bytes_avg": 12.0},
                },
            }
        )
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        assert c.poll_request("j1", "r1") == {
            "avg_loss": 0.5,
            "metrics": {"router_replay/tx_bytes_avg": 12.0},
        }

    def test_poll_request_keeps_payload_metrics_when_envelope_has_none(self, monkeypatch):
        from cortex_training import wire

        payload = wire.dumps({"avg_loss": 0.5, "metrics": {"model/tokens": 8}})
        c = _make_client(
            get_json={
                "status": "done",
                "result": {
                    "content_type": "application/octet-stream",
                    "encoding": "base64",
                    "wire_format": "DSSST1",
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                },
            }
        )
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        assert c.poll_request("j1", "r1") == {
            "avg_loss": 0.5,
            "metrics": {"model/tokens": 8},
        }

    def test_poll_request_envelope_metrics_win_on_key_collision(self, monkeypatch):
        from cortex_training import wire

        payload = wire.dumps(
            {
                "avg_loss": 0.5,
                "metrics": {"model/tokens": 8, "shared/bytes": 1.0},
            }
        )
        c = _make_client(
            get_json={
                "status": "done",
                "result": {
                    "content_type": "application/octet-stream",
                    "encoding": "base64",
                    "wire_format": "DSSST1",
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                    "metrics": {"router_replay/tx_bytes_avg": 12.0, "shared/bytes": 2.0},
                },
            }
        )
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        assert c.poll_request("j1", "r1") == {
            "avg_loss": 0.5,
            "metrics": {
                "model/tokens": 8,
                "shared/bytes": 2.0,
                "router_replay/tx_bytes_avg": 12.0,
            },
        }

    def test_poll_request_keeps_non_generate_dssst1_tensors(self, monkeypatch):
        from cortex_training import wire

        payload = wire.dumps({"logprobs": torch.tensor([0.125, -0.5], dtype=torch.float32)})
        c = _make_client(
            get_json={
                "status": "done",
                "result": {
                    "content_type": "application/octet-stream",
                    "encoding": "base64",
                    "wire_format": "DSSST1",
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                },
            }
        )
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)

        result = c.poll_request("j1", "r1")
        assert torch.equal(result["logprobs"], torch.tensor([0.125, -0.5]))

    def test_poll_request_restores_generate_dssst1_tensors_to_lists(self, monkeypatch):
        from cortex_training import wire

        payload = wire.dumps(
            {
                "job_id": "j1",
                "results": [
                    {
                        "text": "ok",
                        "token_ids": torch.tensor([1, 2, 3], dtype=torch.int64),
                        "logprobs": torch.tensor([0.125, -0.5], dtype=torch.float32),
                        "action_masks": torch.tensor([True, False], dtype=torch.bool),
                    }
                ],
            }
        )
        c = _make_client(
            get_json={
                "status": "done",
                "result": {
                    "content_type": "application/octet-stream",
                    "encoding": "base64",
                    "wire_format": "DSSST1",
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                },
            }
        )
        c._generate_request_ids.add("r-generate")
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)

        assert c.poll_request("j1", "r-generate") == {
            "job_id": "j1",
            "results": [
                {
                    "text": "ok",
                    "token_ids": [1, 2, 3],
                    "logprobs": [0.125, -0.5],
                    "action_masks": [True, False],
                }
            ],
        }
        assert "r-generate" not in c._generate_request_ids

    def test_poll_request_decodes_a_legacy_forward_result_envelope(self, monkeypatch):
        from cortex_training import wire

        payload = wire.dumps(
            {"job_id": "j1", "logprobs": torch.tensor([0.25, -1.5], dtype=torch.float32)}
        )
        c = _make_client(
            get_json={
                "status": "done",
                "result": {
                    "job_id": "j1",
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                    "metrics": {"router_replay/tx_bytes_peak": 24.0},
                },
            }
        )
        c._forward_request_ids.add("r-forward")
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)

        result = c.poll_request("j1", "r-forward")
        assert torch.equal(result["logprobs"], torch.tensor([0.25, -1.5]))
        assert result["metrics"] == {"router_replay/tx_bytes_peak": 24.0}
        assert "r-forward" not in c._forward_request_ids

    def test_poll_request_leaves_a_self_describing_forward_result_alone(self, monkeypatch):
        from cortex_training import wire

        payload = wire.dumps(
            {"job_id": "j1", "logprobs": torch.tensor([0.5], dtype=torch.float32)}
        )
        c = _make_client(
            get_json={
                "status": "done",
                "result": {
                    "content_type": "application/octet-stream",
                    "encoding": "base64",
                    "wire_format": "DSSST1",
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                },
            }
        )
        c._forward_request_ids.add("r-forward")
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)

        result = c.poll_request("j1", "r-forward")
        assert torch.equal(result["logprobs"], torch.tensor([0.5]))
        assert "r-forward" not in c._forward_request_ids

    def test_poll_request_forgets_a_forward_id_on_failure(self, monkeypatch):
        c = _make_client(get_json={"status": "failed", "error": "boom"})
        c._forward_request_ids.add("r-forward")
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)

        with pytest.raises(RuntimeError, match="boom"):
            c.poll_request("j1", "r-forward")
        assert "r-forward" not in c._forward_request_ids

    def test_poll_request_decodes_chunked_dssst1_result(self, monkeypatch):
        from cortex_training import wire

        chunks = wire.encode_result_chunks({"text": "x" * 20_000}, max_bytes=12_000)
        assert len(chunks) > 1
        responses = []
        for idx, chunk in enumerate(chunks):
            event = {
                "type": "result_chunk",
                "payload_b64": base64.b64encode(chunk).decode("ascii"),
                "payload_sha256": __import__("hashlib").sha256(chunk).hexdigest(),
            }
            body = {"status": "running", "events": [event], "next_cursor": str(idx + 1)}
            if idx == len(chunks) - 1:
                body = {
                    "status": "done",
                    "events": [event],
                    "result": {
                        "metrics": {"router_replay/tx_max_bytes": 4096.0},
                    },
                }
            responses.append(_make_response(body))

        c = _make_client()
        c._session.get.side_effect = responses
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)

        assert c.poll_request("j1", "r1") == {
            "text": "x" * 20_000,
            "metrics": {"router_replay/tx_max_bytes": 4096.0},
        }
        assert c._session.get.call_args_list[1].kwargs["params"] == {"cursor": "1"}

    def test_poll_request_failed(self, monkeypatch):
        c = _make_client(get_json={"status": "failed", "error": "oom"})
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        with pytest.raises(RuntimeError, match="oom"):
            c.poll_request("j1", "r1")

    def test_poll_request_uses_exponential_backoff_until_done(self, monkeypatch):
        c = _make_client()
        c.poll_interval = 1.0
        c.poll_backoff_multiplier = 2.0
        c.poll_max_interval = 3.0
        c._session.get.side_effect = [
            _make_response({"status": "running"}),
            _make_response({"status": "running"}),
            _make_response({"status": "running"}),
            _make_response({"status": "done", "result": {"ok": True}}),
        ]
        sleeps = []
        monkeypatch.setattr(nc.time, "sleep", sleeps.append)

        assert c.poll_request("j1", "r1") == {"ok": True}
        assert sleeps == [1.0, 2.0, 3.0]

    def test_poll_request_times_out(self, monkeypatch):
        c = _make_client(get_json={"status": "running"})
        c.poll_timeout = 0.01
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        with pytest.raises(TimeoutError):
            c.poll_request("j1", "r1")


# ─── CortexTrainingClient — checkpoints ──────────────────────────────────────


class TestCheckpoints:
    def test_list_checkpoints(self):
        c = _make_client(get_json={"checkpoints": [{"checkpoint_id": "cp1"}]})
        out = c.list_checkpoints("j1")
        assert out == [{"checkpoint_id": "cp1"}]
        c._session.get.assert_called_once_with(f"{c._prefix}/j1/checkpoints")

    def test_export_checkpoint(self):
        c = _make_client(post_json={"checkpoint_id": "cp1", "files": []})
        out = c.export_checkpoint("j1", "cp1")
        assert out["checkpoint_id"] == "cp1"
        c._session.post.assert_called_once_with(f"{c._prefix}/j1/checkpoints/cp1:export")

    def test_delete_checkpoint(self):
        c = _make_client()
        # GS returns 204 No Content; delete_checkpoint returns None.
        c._session.delete.return_value = _make_response(status_code=204)
        assert c.delete_checkpoint("j1", "cp1") is None
        c._session.delete.assert_called_once_with(f"{c._prefix}/j1/checkpoints/cp1")

    def test_delete_checkpoint_raises_on_error(self):
        # delete_checkpoint discards the response, so pin the inherited
        # raise_for_status() contract: a non-2xx status propagates rather than
        # returning None. Models re-deleting an already-absent checkpoint (404).
        c = _make_client()
        c.max_retries = 0  # 404 is transient/retried; skip the backoff here
        resp = _make_response(status_code=404)
        resp.raise_for_status.side_effect = nc.requests.exceptions.HTTPError(response=resp)
        c._session.delete.return_value = resp
        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.delete_checkpoint("j1", "missing")


# ─── CortexTrainingClient — logs (read-only) ─────────────────────────────────


class TestLogs:
    def test_tail_logs_builds_payload(self):
        c = _make_client(post_json={"entries": [], "next_cursor": "c1", "eof": True})
        c.tail_logs("j1", cursor="c0", max_lines=50, sub_job_id="sj-0")
        body = c._session.post.call_args.kwargs["json"]
        assert body["operation_type"] == "tail-logs"
        assert body["sub_job_id"] == "sj-0"
        assert body["payload"] == {"cursor": "c0", "max_lines": 50}

    def test_tail_logs_omits_unset_payload_fields(self):
        c = _make_client(post_json={"entries": [], "next_cursor": "c1", "eof": True})
        c.tail_logs("j1", sub_job_id="sj-0")
        body = c._session.post.call_args.kwargs["json"]
        assert body["payload"] == {}  # nothing set → empty payload

    def test_stream_logs_drains_then_stops_on_eof(self, monkeypatch):
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        c = _make_client()
        c._session.post.side_effect = [
            _make_response({"entries": [{"msg": "a"}, {"msg": "b"}], "next_cursor": "c1", "eof": False}),
            _make_response({"entries": [{"msg": "c"}], "next_cursor": "c2", "eof": False}),
            _make_response({"entries": [], "next_cursor": "c2", "eof": True}),
        ]
        got = [e["msg"] for e in c.stream_logs("j1", sub_job_id="sj-0", follow=False)]
        assert got == ["a", "b", "c"]
        # Cursor advanced across polls: the last call carried the prior next_cursor.
        assert c._session.post.call_args.kwargs["json"]["payload"]["cursor"] == "c2"

    def test_stream_logs_follow_keeps_yielding(self, monkeypatch):
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        c = _make_client(post_json={"entries": [{"msg": "a"}, {"msg": "b"}], "next_cursor": "c1", "eof": False})
        got = []
        for entry in c.stream_logs("j1", sub_job_id="sj-0", follow=True):
            got.append(entry["msg"])
            if len(got) == 2:
                break  # an infinite tail; caller stops by breaking
        assert got == ["a", "b"]


# ─── CortexTrainingClient — zone scheduling events (read-only) ───────────────


class TestEvents:
    def test_tail_events_builds_payload(self):
        c = _make_client(post_json={"events": [], "next_cursor": "3", "eof": True})
        c.tail_events("j1", cursor="2", max_events=10, sub_job_id="sj-0")
        url, kwargs = c._session.post.call_args
        assert url[0] == f"{c._prefix}/j1/operation"
        body = kwargs["json"]
        assert body["operation_type"] == "zmd-events"
        assert body["sub_job_id"] == "sj-0"
        assert body["payload"] == {"cursor": "2", "max_events": 10}

    def test_tail_events_omits_empty_payload(self):
        c = _make_client(post_json={"events": [], "next_cursor": "0", "eof": True})
        c.tail_events("j1")
        body = c._session.post.call_args.kwargs["json"]
        assert body["operation_type"] == "zmd-events"
        assert "payload" not in body  # nothing to send

    def test_stream_events_drains_to_eof(self, monkeypatch):
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        c = _make_client()
        c._session.post.side_effect = [
            _make_response({"events": [{"seq": 1}, {"seq": 2}], "next_cursor": "2", "eof": False}),
            _make_response({"events": [{"seq": 3}], "next_cursor": "3", "eof": False}),
            _make_response({"events": [], "next_cursor": "3", "eof": True}),
        ]
        seqs = [e["seq"] for e in c.stream_events("j1", follow=False)]
        assert seqs == [1, 2, 3]
        assert c._session.post.call_args.kwargs["json"]["payload"]["cursor"] == "3"


class TestExecutionLogDownload:
    class FakeArtifactConnection:
        def __init__(self, list_rows=None, content_by_uri=None, on_get=None):
            self.list_rows = list_rows or {}
            self.content_by_uri = content_by_uri or {}
            self.commands = []
            self.closed = False
            self.close_count = 0
            self.raise_on_close = False
            # Called with the source URI before a GET writes its file; may block or raise.
            self.on_get = on_get

        class Cursor:
            def __init__(self, connection):
                self.connection = connection
                self.rows = []

            @staticmethod
            def _unquote(value):
                assert value.startswith("'") and value.endswith("'")
                return value[1:-1].replace("''", "'")

            def execute(self, statement):
                self.connection.commands.append(statement)
                if statement.startswith("LIST "):
                    uri = self._unquote(statement.removeprefix("LIST "))
                    self.rows = self.connection.listed(uri)
                elif statement.startswith("GET "):
                    source_literal, target_literal = statement.removeprefix("GET ").split(" ", 1)
                    source = self._unquote(source_literal)
                    target_uri = self._unquote(target_literal)
                    if self.connection.on_get is not None:
                        self.connection.on_get(source)
                    target = Path(unquote(urlparse(target_uri).path))
                    target.mkdir(parents=True, exist_ok=True)
                    assert not list(target.iterdir())
                    (target / PurePosixPath(source).name).write_bytes(
                        self.connection.content_by_uri[source]
                    )
                    self.rows = [(source, "DOWNLOADED")]
                else:
                    raise AssertionError(f"unexpected SQL: {statement}")
                return self

            def fetchall(self):
                return self.rows

            def close(self):
                pass

        def listed(self, uri):
            """What LIST ``uri`` returns. A stage LIST matches by path prefix: the rows
            seeded for ``uri`` itself, else those of a seeded location above it that
            lie under ``uri`` (a row's name is its path from ``/versions/``)."""
            if uri in self.list_rows:
                return self.list_rows[uri]
            above = [location for location in self.list_rows if uri.startswith(location) and "/versions/" in location]
            if not above:
                return []
            location = max(above, key=len)
            stage = location[: location.index("/versions/")]
            return [row for row in self.list_rows[location] if (stage + row[0]).startswith(uri)]

        def cursor(self):
            return self.Cursor(self)

        def close(self):
            self.close_count += 1
            self.closed = True
            if self.raise_on_close:
                raise RuntimeError("close failed")

    def test_sql_literal_and_artifact_path_validation(self):
        assert nc._sql_string_literal("snow://exp/O'Brien") == "'snow://exp/O''Brien'"
        assert nc._validate_artifact_relative_path("_stdout/job/file.gz") == "_stdout/job/file.gz"
        for path in ("", "/absolute", "../escape", "a/../escape", r"a\b"):
            with pytest.raises(ValueError, match="unsafe"):
                nc._validate_artifact_relative_path(path)

    def test_artifact_connection_requires_snowflake_client(self):
        with pytest.raises(RuntimeError, match="Snowflake-authenticated"):
            _make_client()._open_experiment_artifact_connection()

    def test_artifact_connection_uses_profile_factory(self):
        c = _make_client()
        connected = object()
        c._artifact_connection_factory = MagicMock(return_value=connected)

        assert c._open_experiment_artifact_connection() is connected
        c._artifact_connection_factory.assert_called_once_with()

    @pytest.mark.parametrize(
        ("host", "account"),
        [
            ("locatorhost.snowflakecomputing.com", "locatorhost"),
            ("myorg-myaccount.snowflakecomputing.com", "myorg-myaccount"),
            ("account.qa6.us-west-2.aws.snowflakecomputing.com", "account"),
            ("testaccount-12345.global.snowflakecomputing.com", "testaccount"),
        ],
    )
    def test_connection_kwargs_use_pat_host_as_account(self, monkeypatch, host, account):
        monkeypatch.setenv(nc.DISABLE_TELEMETRY_ENV, "1")
        c = CortexTrainingClient._from_pat(
            host=host,
            pat="tok-xyz",
            database="DB",
            schema="SCH",
        )
        identity_queries = []

        def identity(statement):
            identity_queries.append(statement)
            return ["USER1", "ROLE1"]

        monkeypatch.setattr(c, "_query_sql_row", identity)
        kwargs = c._snowflake_connection_kwargs()
        assert identity_queries == ["SELECT CURRENT_USER(), CURRENT_ROLE()"]
        assert importlib.import_module("snowflake.connector.util_text").parse_account(host) == account
        assert kwargs["host"] == host
        assert kwargs["account"] == host
        assert kwargs["user"] == "USER1"
        assert kwargs["role"] == "ROLE1"
        assert kwargs["authenticator"] == "PROGRAMMATIC_ACCESS_TOKEN"
        assert kwargs["token"] == "tok-xyz"

        connector = importlib.import_module("snowflake.connector")
        connected = object()
        connect = MagicMock(return_value=connected)
        monkeypatch.setattr(connector, "connect", connect)
        assert c._open_experiment_artifact_connection() is connected
        # Without the Connector's cloud-metadata probes, which add up to seconds to a login.
        assert "platform_detection_timeout_seconds" in importlib.import_module(
            "snowflake.connector.connection"
        ).DEFAULT_CONFIGURATION
        connect.assert_called_once_with(**kwargs, platform_detection_timeout_seconds=0.0)

    def test_get_experiment_run_calls_endpoint(self):
        c = _make_client(
            get_json={
                "experiment_name": "DB.SCH.EXP",
                "experiment_run_name": "RUN_ABC",
            }
        )
        out = c.get_experiment_run("job-1")
        assert out == {"experiment_name": "DB.SCH.EXP", "experiment_run_name": "RUN_ABC"}
        c._session.get.assert_called_once_with(f"{c._prefix}/job-1/experiment-run")

    def test_fetch_execution_logs_lists_and_downloads_every_file_under_logs(self, monkeypatch):
        c = _make_client(
            get_json={
                "experiment_name": "DB.SCH.EXP",
                "experiment_run_name": "RUN_ABC",
            }
        )
        run_uri = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
        paths = [
            "_logs/job-1:eval:0/execution.jsonl",
            "checkpoints/_logs/job-1:training:0/execution.jsonl",
            "checkpoints/_logs/job-1:training:0/server.log",
            "_logs/other-job:eval:0/execution.jsonl",
        ]
        connection = self.FakeArtifactConnection(
            {
                run_uri + "_logs/": [
                    (f"/versions/RUN_ABC/{paths[0]}",),
                    (f"/versions/RUN_ABC/{paths[3]}",),
                ],
                run_uri + "checkpoints/_logs/": [
                    (f"/versions/RUN_ABC/{path}",) for path in paths[1:3]
                ],
            },
            {
                run_uri + paths[0]: b'{"c":3}\n',
                run_uri + paths[1]: b'{"a":1}\n',
                run_uri + paths[2]: b"server line\n",
            },
        )
        monkeypatch.setattr(
            c, "_open_experiment_artifact_connection", lambda: connection
        )

        assert c.fetch_execution_logs("job-1") == [
            {
                "sub_job_id": "job-1:eval:0",
                "filename": "execution.jsonl",
                "artifact_uri": run_uri + paths[0],
                "content": '{"c":3}\n',
            },
            {
                "sub_job_id": "job-1:training:0",
                "filename": "execution.jsonl",
                "artifact_uri": run_uri + paths[1],
                "content": '{"a":1}\n',
            },
            {
                "sub_job_id": "job-1:training:0",
                "filename": "server.log",
                "artifact_uri": run_uri + paths[2],
                "content": "server line\n",
            },
        ]
        assert connection.closed

    @pytest.mark.parametrize(
        ("artifact_name", "chunk_name", "destination_name", "method_name"),
        [
            ("stdout", "console", "stdout.log", "download_stdout_logs"),
            ("metrics", "gpu", "gpu.jsonl", "download_metrics"),
        ],
    )
    def test_download_reconstructs_sorted_chunks_atomically(
        self,
        artifact_name,
        chunk_name,
        destination_name,
        method_name,
        tmp_path,
        monkeypatch,
    ):
        c = _make_client()
        run_uri = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
        root = f"_{artifact_name}"
        paths = [
            f"{root}/job-1:training:0/{chunk_name}.20260904-120002.zulu.gz",
            f"{root}/job-1:training:0/{chunk_name}.20260904-120001.first.gz",
            f"{root}/job-1:training:0/{chunk_name}.20260904-120002.alpha.gz",
            f"{root}/other-job:training:0/{chunk_name}.20260904-120003.other.gz",
            f"{root}/job-1:training:0/{chunk_name}.bad.gz",
        ]
        connection = self.FakeArtifactConnection(
            {
                run_uri + root + "/": [
                    (f"/versions/RUN_ABC/{path}",) for path in paths
                ],
                run_uri + "checkpoints/" + root + "/": [],
            },
            {
                run_uri + paths[0]: gzip.compress(b"third\n"),
                run_uri + paths[1]: gzip.compress(b"first\n"),
                run_uri + paths[2]: gzip.compress(b"second\n"),
            },
        )
        monkeypatch.setattr(c, "_experiment_run_uri", lambda _: (run_uri, "RUN_ABC"))
        monkeypatch.setattr(
            c, "_open_experiment_artifact_connection", lambda: connection
        )

        result = getattr(c, method_name)("job-1", tmp_path)

        destination = tmp_path / "job-1:training:0" / destination_name
        assert destination.read_bytes() == b"first\nsecond\nthird\n"
        assert result == [
            {
                "sub_job_id": "job-1:training:0",
                "filename": destination_name,
                "saved_path": str(destination),
                "chunk_count": 3,
                "first_artifact_uri": run_uri + paths[1],
                "last_artifact_uri": run_uri + paths[0],
            }
        ]
        assert connection.closed

    @pytest.mark.parametrize(
        ("artifact_name", "chunk_name", "destination_name", "method_name"),
        [
            ("stdout", "console", "stdout.log", "download_stdout_logs"),
            ("metrics", "gpu", "gpu.jsonl", "download_metrics"),
        ],
    )
    def test_download_corruption_preserves_existing_file(
        self,
        artifact_name,
        chunk_name,
        destination_name,
        method_name,
        tmp_path,
        monkeypatch,
    ):
        c = _make_client()
        run_uri = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
        root = f"_{artifact_name}"
        good = f"{root}/job-1:training:0/{chunk_name}.20260904-120001.good.gz"
        bad = f"{root}/job-1:training:0/{chunk_name}.20260904-120002.bad.gz"
        destination = tmp_path / "job-1:training:0" / destination_name
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"previous\n")
        connection = self.FakeArtifactConnection(
            {
                run_uri + root + "/": [
                    (f"/versions/RUN_ABC/{good}",),
                    (f"/versions/RUN_ABC/{bad}",),
                ],
                run_uri + "checkpoints/" + root + "/": [],
            },
            {
                run_uri + good: gzip.compress(b"new\n"),
                run_uri + bad: b"not gzip",
            },
        )
        monkeypatch.setattr(c, "_experiment_run_uri", lambda _: (run_uri, "RUN_ABC"))
        monkeypatch.setattr(
            c, "_open_experiment_artifact_connection", lambda: connection
        )

        with pytest.raises(gzip.BadGzipFile):
            getattr(c, method_name)("job-1", tmp_path, strict=True)

        assert destination.read_bytes() == b"previous\n"
        assert not list(destination.parent.glob(".*.tmp"))
        assert connection.closed

    def test_the_two_metric_streams_are_reconstructed_into_their_own_files(
        self, tmp_path, monkeypatch
    ):
        """The GPU chunks and the zone's training chunks share ``_metrics/`` but are
        separate streams: each download takes only its own chunks and writes its own
        file, so training rows are never concatenated into ``gpu.jsonl``."""
        c = _make_client()
        run_uri = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
        sub_job_id = "job-1:training:0"
        gpu = f"_metrics/{sub_job_id}/gpu.20260904-120001.a.gz"
        training_first = f"_metrics/{sub_job_id}/training.20260904-120002.a.gz"
        training_second = f"_metrics/{sub_job_id}/training.20260904-120003.b.gz"
        connection = self.FakeArtifactConnection(
            {
                run_uri + "_metrics/": [
                    (f"/versions/RUN_ABC/{gpu}",),
                    (f"/versions/RUN_ABC/{training_first}",),
                    (f"/versions/RUN_ABC/{training_second}",),
                ],
                run_uri + "checkpoints/_metrics/": [],
            },
            {
                run_uri + gpu: gzip.compress(b'{"gpu_util_pct": 90}\n'),
                run_uri + training_first: gzip.compress(b'{"kind": "training", "step": 1}\n'),
                run_uri + training_second: gzip.compress(b'{"kind": "training", "step": 2}\n'),
            },
        )
        monkeypatch.setattr(c, "_experiment_run_uri", lambda _: (run_uri, "RUN_ABC"))
        monkeypatch.setattr(
            c, "_open_experiment_artifact_connection", lambda: connection
        )

        metrics = c.download_metrics("job-1", tmp_path)
        training = c.download_training_metrics("job-1", tmp_path)

        assert (tmp_path / sub_job_id / "gpu.jsonl").read_bytes() == b'{"gpu_util_pct": 90}\n'
        assert (tmp_path / sub_job_id / "training.jsonl").read_bytes() == (
            b'{"kind": "training", "step": 1}\n{"kind": "training", "step": 2}\n'
        )
        assert metrics[0]["chunk_count"] == 1
        assert training[0]["chunk_count"] == 2
        # Each stream keeps its own resume manifest beside its own file.
        assert (tmp_path / sub_job_id / ".gpu.jsonl.manifest.json").exists()
        assert (tmp_path / sub_job_id / ".training.jsonl.manifest.json").exists()

    def test_a_sub_job_without_training_chunks_is_not_warned_about(
        self, tmp_path, monkeypatch
    ):
        """Unlike the GPU stream, a sub-job with saved stdout and no training chunks
        gets no warning: only training sub-jobs write them, and only where the
        deployment saves them."""
        c = _make_client()
        run_uri = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
        sub_job_id = "job-1:training:0"
        training = f"_metrics/{sub_job_id}/training.20260904-120001.a.gz"
        connection = self.FakeArtifactConnection(
            {
                run_uri + "_metrics/": [(f"/versions/RUN_ABC/{training}",)],
                run_uri + "checkpoints/_metrics/": [],
                run_uri + "_stdout/": [
                    (f"/versions/RUN_ABC/_stdout/job-1:sampling:0/console.20260904-120001.a.gz",),
                ],
            },
            {run_uri + training: gzip.compress(b'{"kind": "training", "step": 1}\n')},
        )
        monkeypatch.setattr(c, "_experiment_run_uri", lambda _: (run_uri, "RUN_ABC"))
        monkeypatch.setattr(
            c, "_open_experiment_artifact_connection", lambda: connection
        )
        monkeypatch.setattr(
            c, "_job_for_upload_warnings", lambda _: ({"status": "SUCCEEDED"}, None)
        )

        result = c.download_training_metrics("job-1", tmp_path)

        assert [entry["sub_job_id"] for entry in result] == [sub_job_id]
        assert not result[0].get("warnings")

    def test_download_metrics_merges_chunks_from_both_roots_in_time_order(
        self, tmp_path, monkeypatch
    ):
        c = _make_client()
        run_uri = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
        sub_job_id = "job-1:training:0"
        first = f"_metrics/{sub_job_id}/gpu.20260904-120001.first.gz"
        second = f"checkpoints/_metrics/{sub_job_id}/gpu.20260904-120002.second.gz"
        third = f"_metrics/{sub_job_id}/gpu.20260904-120003.third.gz"
        both = "gpu.20260904-120004.both.gz"
        connection = self.FakeArtifactConnection(
            {
                run_uri + "_metrics/": [
                    (f"/versions/RUN_ABC/{first}",),
                    (f"/versions/RUN_ABC/{third}",),
                    (f"/versions/RUN_ABC/_metrics/{sub_job_id}/{both}",),
                ],
                run_uri + "checkpoints/_metrics/": [
                    (f"/versions/RUN_ABC/{second}",),
                    (f"/versions/RUN_ABC/checkpoints/_metrics/{sub_job_id}/{both}",),
                ],
            },
            {
                run_uri + first: gzip.compress(b"1\n"),
                run_uri + second: gzip.compress(b"2\n"),
                run_uri + third: gzip.compress(b"3\n"),
                run_uri + f"_metrics/{sub_job_id}/{both}": gzip.compress(b"4\n"),
                run_uri + f"checkpoints/_metrics/{sub_job_id}/{both}": gzip.compress(b"dup\n"),
            },
        )
        monkeypatch.setattr(c, "_experiment_run_uri", lambda _: (run_uri, "RUN_ABC"))
        monkeypatch.setattr(
            c, "_open_experiment_artifact_connection", lambda: connection
        )

        result = c.download_metrics("job-1", tmp_path)

        assert (tmp_path / sub_job_id / "gpu.jsonl").read_bytes() == b"1\n2\n3\n4\n"
        assert result[0]["chunk_count"] == 4
        assert result[0]["last_artifact_uri"] == run_uri + f"_metrics/{sub_job_id}/{both}"
        assert "merged in chunk time order" in result[0]["warnings"][0]
        assert connection.closed

        # The merged order is stable, so a resume finds nothing new.
        gets = [cmd for cmd in connection.commands if cmd.startswith("GET ")]
        again = c.download_metrics("job-1", tmp_path, resume=True)
        assert [cmd for cmd in connection.commands if cmd.startswith("GET ")] == gets
        assert again[0]["chunk_count"] == 4

    def _gzip_download(self, monkeypatch, tmp_path, artifact_name, listed, bodies):
        c = _make_client()
        run_uri = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
        root = f"_{artifact_name}"
        connection = self.FakeArtifactConnection(
            {
                run_uri + root + "/": [(f"/versions/RUN_ABC/{path}",) for path in listed],
                run_uri + "checkpoints/" + root + "/": [],
            },
            {run_uri + path: gzip.compress(body) for path, body in bodies.items()},
        )
        monkeypatch.setattr(c, "_experiment_run_uri", lambda _: (run_uri, "RUN_ABC"))
        monkeypatch.setattr(c, "_open_experiment_artifact_connection", lambda: connection)
        return c, connection, run_uri

    @pytest.mark.parametrize(
        ("artifact_name", "chunk_name", "destination_name", "method_name"),
        [
            ("stdout", "console", "stdout.log", "download_stdout_logs"),
            ("metrics", "gpu", "gpu.jsonl", "download_metrics"),
        ],
    )
    def test_download_resume_appends_suffix_and_writes_manifest(
        self, artifact_name, chunk_name, destination_name, method_name, tmp_path, monkeypatch
    ):
        root = f"_{artifact_name}"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/{chunk_name}.20260904-120001.first.gz"
        later = f"{root}/{sub}/{chunk_name}.20260904-120002.zulu.gz"
        c, connection, run_uri = self._gzip_download(
            monkeypatch, tmp_path, artifact_name, [first], {first: b"first\n"}
        )
        download = getattr(c, method_name)
        download("job-1", tmp_path)
        destination = tmp_path / sub / destination_name
        manifest = destination.with_name("." + destination.name + ".manifest.json")
        assert json.loads(manifest.read_text())["names"] == [first]
        gets_before = [cmd for cmd in connection.commands if cmd.startswith("GET ")]
        connection.list_rows[run_uri + root + "/"] = [
            (f"/versions/RUN_ABC/{first}",),
            (f"/versions/RUN_ABC/{later}",),
        ]
        connection.content_by_uri[run_uri + later] = gzip.compress(b"second\n")
        result = download("job-1", tmp_path, resume=True)
        new_gets = [cmd for cmd in connection.commands if cmd.startswith("GET ")][len(gets_before) :]
        assert len(new_gets) == 1
        assert later in new_gets[0]
        assert destination.read_bytes() == b"first\nsecond\n"
        assert result[0]["chunk_count"] == 2
        assert result[0]["first_artifact_uri"] == run_uri + first
        assert result[0]["last_artifact_uri"] == run_uri + later

    @pytest.mark.parametrize(
        ("artifact_name", "chunk_name", "destination_name", "method_name"),
        [
            ("stdout", "console", "stdout.log", "download_stdout_logs"),
            ("metrics", "gpu", "gpu.jsonl", "download_metrics"),
        ],
    )
    def test_download_resume_rebuilds_when_new_name_sorts_earlier(
        self, artifact_name, chunk_name, destination_name, method_name, tmp_path, monkeypatch
    ):
        root = f"_{artifact_name}"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/{chunk_name}.20260904-120001.first.gz"
        zulu = f"{root}/{sub}/{chunk_name}.20260904-120002.zulu.gz"
        alpha = f"{root}/{sub}/{chunk_name}.20260904-120002.alpha.gz"
        c, connection, run_uri = self._gzip_download(
            monkeypatch,
            tmp_path,
            artifact_name,
            [first, zulu],
            {first: b"first\n", zulu: b"third\n"},
        )
        download = getattr(c, method_name)
        download("job-1", tmp_path)
        connection.list_rows[run_uri + root + "/"].append((f"/versions/RUN_ABC/{alpha}",))
        connection.content_by_uri[run_uri + alpha] = gzip.compress(b"second\n")
        download("job-1", tmp_path, resume=True)
        destination = tmp_path / sub / destination_name
        assert destination.read_bytes() == b"first\nsecond\nthird\n"

    @pytest.mark.parametrize(
        ("artifact_name", "chunk_name", "destination_name", "method_name"),
        [
            ("stdout", "console", "stdout.log", "download_stdout_logs"),
            ("metrics", "gpu", "gpu.jsonl", "download_metrics"),
        ],
    )
    def test_download_resume_truncates_torn_tail(
        self, artifact_name, chunk_name, destination_name, method_name, tmp_path, monkeypatch
    ):
        root = f"_{artifact_name}"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/{chunk_name}.20260904-120001.first.gz"
        c, connection, _run_uri = self._gzip_download(
            monkeypatch, tmp_path, artifact_name, [first], {first: b"first\n"}
        )
        download = getattr(c, method_name)
        download("job-1", tmp_path)
        destination = tmp_path / sub / destination_name
        committed = destination.read_bytes()
        destination.write_bytes(committed + b"TORN")
        gets_before = [cmd for cmd in connection.commands if cmd.startswith("GET ")]
        download("job-1", tmp_path, resume=True)
        assert destination.read_bytes() == committed
        assert [cmd for cmd in connection.commands if cmd.startswith("GET ")] == gets_before

    @pytest.mark.parametrize("manifest_setup", ["missing", "malformed", "version", "short"])
    def test_download_resume_unusable_manifest_rebuilds(self, manifest_setup, tmp_path, monkeypatch):
        root = "_stdout"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/console.20260904-120001.first.gz"
        c, _connection, _run_uri = self._gzip_download(
            monkeypatch, tmp_path, "stdout", [first], {first: b"first\n"}
        )
        c.download_stdout_logs("job-1", tmp_path)
        destination = tmp_path / sub / "stdout.log"
        manifest = destination.with_name(".stdout.log.manifest.json")
        if manifest_setup == "missing":
            manifest.unlink()
        elif manifest_setup == "malformed":
            manifest.write_text("not-json", encoding="utf-8")
        elif manifest_setup == "version":
            doc = json.loads(manifest.read_text())
            doc["version"] = 2
            manifest.write_text(json.dumps(doc), encoding="utf-8")
        else:
            destination.write_bytes(b"x")
        c.download_stdout_logs("job-1", tmp_path, resume=True)
        assert destination.read_bytes() == b"first\n"
        assert json.loads(manifest.read_text())["version"] == 1

    def test_download_resume_rebuilds_when_manifest_was_removed(self, tmp_path, monkeypatch):
        root = "_stdout"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/console.20260904-120001.first.gz"
        c, _connection, _run_uri = self._gzip_download(
            monkeypatch, tmp_path, "stdout", [first], {first: b"first\n"}
        )
        c.download_stdout_logs("job-1", tmp_path)
        destination = tmp_path / sub / "stdout.log"
        destination.with_name(".stdout.log.manifest.json").unlink()
        destination.write_bytes(b"replaced\n")
        c.download_stdout_logs("job-1", tmp_path, resume=True)
        assert destination.read_bytes() == b"first\n"

    def test_download_resume_failed_chunk_keeps_prefix(self, tmp_path, monkeypatch):
        root = "_stdout"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/console.20260904-120001.first.gz"
        bad = f"{root}/{sub}/console.20260904-120002.bad.gz"
        c, connection, run_uri = self._gzip_download(
            monkeypatch, tmp_path, "stdout", [first], {first: b"first\n"}
        )
        c.download_stdout_logs("job-1", tmp_path)
        destination = tmp_path / sub / "stdout.log"
        connection.list_rows[run_uri + root + "/"].append((f"/versions/RUN_ABC/{bad}",))
        connection.content_by_uri[run_uri + bad] = b"not gzip"
        with pytest.raises(gzip.BadGzipFile):
            c.download_stdout_logs("job-1", tmp_path, resume=True, strict=True)
        assert destination.read_bytes() == b"first\n"
        manifest = json.loads(destination.with_name(".stdout.log.manifest.json").read_text())
        assert manifest["names"] == [first]

    def test_download_resume_keeps_committed_sub_job_when_later_one_fails(
        self, tmp_path, monkeypatch
    ):
        c = _make_client()
        run_uri = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
        first_sub = "job-1:training:0"
        second_sub = "job-1:training:1"
        good = f"_stdout/{first_sub}/console.20260904-120001.first.gz"
        bad = f"_stdout/{second_sub}/console.20260904-120001.bad.gz"
        connection = self.FakeArtifactConnection(
            {
                run_uri + "_stdout/": [
                    (f"/versions/RUN_ABC/{good}",),
                    (f"/versions/RUN_ABC/{bad}",),
                ],
                run_uri + "checkpoints/_stdout/": [],
            },
            {run_uri + good: gzip.compress(b"ok\n"), run_uri + bad: b"not gzip"},
        )
        monkeypatch.setattr(c, "_experiment_run_uri", lambda _: (run_uri, "RUN_ABC"))
        monkeypatch.setattr(c, "_open_experiment_artifact_connection", lambda: connection)
        with pytest.raises(gzip.BadGzipFile):
            c.download_stdout_logs("job-1", tmp_path, strict=True)
        committed = tmp_path / first_sub / "stdout.log"
        assert committed.read_bytes() == b"ok\n"
        connection.content_by_uri[run_uri + bad] = gzip.compress(b"later\n")
        gets_before = [cmd for cmd in connection.commands if cmd.startswith("GET ") and good in cmd]
        c.download_stdout_logs("job-1", tmp_path, resume=True)
        gets_after = [cmd for cmd in connection.commands if cmd.startswith("GET ") and good in cmd]
        assert gets_after == gets_before
        assert (tmp_path / second_sub / "stdout.log").read_bytes() == b"later\n"

    def _parallel_stdout(self, monkeypatch, count, *, bodies=None, on_get=None):
        """One sub-job of ``count`` stdout chunks; every connection opened is recorded."""
        c = _make_client()
        run_uri = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
        sub = "job-1:training:0"
        paths = [
            f"_stdout/{sub}/console.20260904-{index:06d}.part.gz"
            for index in range(count)
        ]
        if bodies is None:
            bodies = {
                run_uri + path: gzip.compress(f"{index}\n".encode())
                for index, path in enumerate(paths)
            }
        list_rows = {
            run_uri + "_stdout/": [(f"/versions/RUN_ABC/{path}",) for path in paths],
            run_uri + "checkpoints/_stdout/": [],
        }
        opened = []

        def opener():
            conn = self.FakeArtifactConnection(list_rows, bodies, on_get=on_get)
            opened.append(conn)
            return conn

        monkeypatch.setattr(c, "_experiment_run_uri", lambda _: (run_uri, "RUN_ABC"))
        monkeypatch.setattr(c, "_open_experiment_artifact_connection", opener)
        return c, opened, run_uri, paths

    def test_parallel_download_overlaps_on_one_session_and_keeps_sorted_order(
        self, tmp_path, monkeypatch
    ):
        workers = 8
        barrier = threading.Barrier(workers, timeout=5)
        started = []
        lock = threading.Lock()

        def on_get(_source):
            with lock:
                started.append(1)
                current = len(started)
            if current <= workers:
                barrier.wait()  # only passes if `workers` GETs run at once

        c, opened, _run_uri, _paths = self._parallel_stdout(monkeypatch, 20, on_get=on_get)
        c.download_stdout_logs("job-1", tmp_path, max_workers=workers)
        expected = b"".join(f"{index}\n".encode() for index in range(20))
        assert (tmp_path / "job-1:training:0" / "stdout.log").read_bytes() == expected
        # The GETs share one session (one cursor each); nothing else is opened.
        assert len(opened) == 1
        assert opened[0].close_count == 1

    def test_parallel_get_failure_keeps_previous_file(self, tmp_path, monkeypatch):
        def on_get(source):
            if source.endswith("000001.part.gz"):
                raise RuntimeError("get failed")

        c, opened, _run_uri, _paths = self._parallel_stdout(monkeypatch, 4, on_get=on_get)
        destination = tmp_path / "job-1:training:0" / "stdout.log"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"previous\n")
        with pytest.raises(RuntimeError, match="get failed"):
            c.download_stdout_logs("job-1", tmp_path)
        assert destination.read_bytes() == b"previous\n"
        assert [conn.close_count for conn in opened] == [1]

    def test_parallel_bad_gzip_on_resume_keeps_committed_chunks(self, tmp_path, monkeypatch):
        c, opened, run_uri, paths = self._parallel_stdout(monkeypatch, 1)
        c.download_stdout_logs("job-1", tmp_path)
        destination = tmp_path / "job-1:training:0" / "stdout.log"
        committed = destination.read_bytes()
        manifest = destination.with_name(".stdout.log.manifest.json")
        names = json.loads(manifest.read_text())["names"]
        later = [
            f"_stdout/job-1:training:0/console.20260904-{index:06d}.part.gz" for index in (1, 2)
        ]
        opened[0].content_by_uri[run_uri + later[0]] = gzip.compress(b"new\n")
        opened[0].content_by_uri[run_uri + later[1]] = b"not gzip"
        opened[0].list_rows[run_uri + "_stdout/"].extend(
            (f"/versions/RUN_ABC/{path}",) for path in later
        )
        with pytest.raises(gzip.BadGzipFile):
            c.download_stdout_logs("job-1", tmp_path, resume=True, strict=True)
        # The chunk appended before the damaged one is kept and recorded.
        assert destination.read_bytes() == committed + b"new\n"
        assert json.loads(manifest.read_text())["names"] == names + later[:1]
        assert [conn.close_count for conn in opened] == [1, 1]

    def test_close_failure_does_not_fail_the_download(self, tmp_path, monkeypatch, caplog):
        c, opened, _run_uri, _paths = self._parallel_stdout(monkeypatch, 3)
        real_opener = c._open_experiment_artifact_connection

        def opener():
            conn = real_opener()
            conn.raise_on_close = True
            return conn

        monkeypatch.setattr(c, "_open_experiment_artifact_connection", opener)
        with caplog.at_level(logging.WARNING):
            c.download_stdout_logs("job-1", tmp_path)
        assert (tmp_path / "job-1:training:0" / "stdout.log").read_bytes() == b"0\n1\n2\n"
        assert [conn.close_count for conn in opened] == [1]
        assert "could not close the artifact download session" in caplog.text

    def test_get_error_survives_a_close_error(self, tmp_path, monkeypatch):
        def on_get(source):
            if source.endswith("000001.part.gz"):
                raise RuntimeError("get failed")

        c, opened, _run_uri, _paths = self._parallel_stdout(monkeypatch, 3, on_get=on_get)
        real_opener = c._open_experiment_artifact_connection

        def opener():
            conn = real_opener()
            conn.raise_on_close = True
            return conn

        monkeypatch.setattr(c, "_open_experiment_artifact_connection", opener)
        with pytest.raises(RuntimeError, match="get failed"):
            c.download_stdout_logs("job-1", tmp_path)
        assert [conn.close_count for conn in opened] == [1]

    def test_parallel_progress_reports_on_caller_after_each_copy(
        self, tmp_path, monkeypatch
    ):
        c, opened, _run_uri, _paths = self._parallel_stdout(monkeypatch, 20)
        sub = "job-1:training:0"
        caller = threading.get_ident()
        copied = []
        real_copy = nc._artifact_download.copy_gzip_member

        def copy(downloaded, output, buffer_bytes):
            real_copy(downloaded, output, buffer_bytes)
            copied.append(output.tell())

        monkeypatch.setattr(nc._artifact_download, "copy_gzip_member", copy)
        calls = []

        def progress(message):
            assert threading.get_ident() == caller
            calls.append(message)
            if message == "\n":
                return
            done = int(message.split("] ", 1)[1].split("/", 1)[0])
            assert len(copied) == done

        c.download_stdout_logs("job-1", tmp_path, progress=progress)
        assert len(opened) == 1  # one session shared by the concurrent GETs
        status = c._artifact_progress_status
        assert calls == [status("stdout", index, 20, sub) for index in range(21)] + ["\n"]
        expected = b"".join(f"{index}\n".encode() for index in range(20))
        assert (tmp_path / sub / "stdout.log").read_bytes() == expected

    def test_parallel_progress_callback_error_cleans_up(self, tmp_path, monkeypatch):
        c, opened, run_uri, paths = self._parallel_stdout(monkeypatch, 4)
        sub = "job-1:training:0"
        destination = tmp_path / sub / "stdout.log"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"previous\n")
        calls = []

        def stop(message):
            calls.append(message)
            if " 2/" in message:
                raise RuntimeError("stop")

        with pytest.raises(RuntimeError, match="stop"):
            c.download_stdout_logs("job-1", tmp_path, progress=stop)
        assert calls[-1] == "\n"
        assert destination.read_bytes() == b"previous\n"
        assert not list(destination.parent.glob(".*.tmp"))
        assert [conn.close_count for conn in opened] == [1] * len(opened)

        c.download_stdout_logs("job-1", tmp_path)
        committed = destination.read_bytes()
        manifest = destination.with_name(".stdout.log.manifest.json")
        manifest_before = manifest.read_text(encoding="utf-8")
        listed = opened[0].list_rows[run_uri + "_stdout/"]
        for index in range(4):
            path = f"_stdout/{sub}/console.20260904-{index + 4:06d}.part.gz"
            listed.append((f"/versions/RUN_ABC/{path}",))
            opened[0].content_by_uri[run_uri + path] = gzip.compress(
                f"{index + 4}\n".encode()
            )
        before = len(opened)
        calls.clear()
        with pytest.raises(RuntimeError, match="stop"):
            c.download_stdout_logs("job-1", tmp_path, resume=True, progress=stop)
        assert calls[0] == c._artifact_progress_status("stdout", 0, 4, sub)
        assert calls[-1] == "\n"
        # Stopping at "2/4" keeps the two chunks already appended for a resume.
        assert destination.read_bytes() == committed + b"4\n5\n"
        names = json.loads(manifest_before)["names"]
        assert json.loads(manifest.read_text(encoding="utf-8"))["names"] == names + [
            f"_stdout/{sub}/console.20260904-{index:06d}.part.gz" for index in (4, 5)
        ]
        resume_connections = opened[before:]
        assert resume_connections
        assert [conn.close_count for conn in resume_connections] == [1] * len(resume_connections)

    @pytest.mark.parametrize(
        ("artifact_name", "chunk_name", "destination_name", "method_name"),
        [
            ("stdout", "console", "stdout.log", "download_stdout_logs"),
            ("metrics", "gpu", "gpu.jsonl", "download_metrics"),
        ],
    )
    def test_download_progress_reports_each_chunk(
        self, artifact_name, chunk_name, destination_name, method_name, tmp_path, monkeypatch
    ):
        root = f"_{artifact_name}"
        sub = "job-1:training:0"
        paths = [
            f"{root}/{sub}/{chunk_name}.20260904-120001.a.gz",
            f"{root}/{sub}/{chunk_name}.20260904-120002.b.gz",
            f"{root}/{sub}/{chunk_name}.20260904-120003.c.gz",
        ]
        c, _connection, _run_uri = self._gzip_download(
            monkeypatch,
            tmp_path,
            artifact_name,
            paths,
            {paths[0]: b"a\n", paths[1]: b"b\n", paths[2]: b"c\n"},
        )
        calls = []
        getattr(c, method_name)("job-1", tmp_path, progress=calls.append)
        quiet = getattr(c, method_name)("job-1", tmp_path / "quiet")
        assert quiet[0]["chunk_count"] == 3
        status = c._artifact_progress_status
        assert calls == [
            status(artifact_name, 0, 3, sub),
            status(artifact_name, 1, 3, sub),
            status(artifact_name, 2, 3, sub),
            status(artifact_name, 3, 3, sub),
            "\n",
        ]
        assert (tmp_path / sub / destination_name).read_bytes() == (
            tmp_path / "quiet" / sub / destination_name
        ).read_bytes()

    def test_download_progress_two_sub_jobs(self, tmp_path, monkeypatch):
        c = _make_client()
        run_uri = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
        first = "_stdout/job-1:training:0/console.20260904-120001.a.gz"
        second = "_stdout/job-1:training:1/console.20260904-120001.a.gz"
        connection = self.FakeArtifactConnection(
            {
                run_uri + "_stdout/": [
                    (f"/versions/RUN_ABC/{first}",),
                    (f"/versions/RUN_ABC/{second}",),
                ],
                run_uri + "checkpoints/_stdout/": [],
            },
            {
                run_uri + first: gzip.compress(b"one\n"),
                run_uri + second: gzip.compress(b"two\n"),
            },
        )
        monkeypatch.setattr(c, "_experiment_run_uri", lambda _: (run_uri, "RUN_ABC"))
        monkeypatch.setattr(c, "_open_experiment_artifact_connection", lambda: connection)
        calls = []
        c.download_stdout_logs("job-1", tmp_path, progress=calls.append)
        status = c._artifact_progress_status
        assert calls == [
            status("stdout", 0, 1, "job-1:training:0"),
            status("stdout", 1, 1, "job-1:training:0"),
            "\n",
            status("stdout", 0, 1, "job-1:training:1"),
            status("stdout", 1, 1, "job-1:training:1"),
            "\n",
        ]

    def test_download_progress_status_format(self):
        status = CortexTrainingClient._artifact_progress_status
        assert status("metrics", 0, 3, "job:s:0") == (
            "metrics: [--------------------] 0/3 job:s:0"
        )
        assert status("metrics", 1, 3, "job:s:0") == (
            "metrics: [######--------------] 1/3 job:s:0"
        )
        assert status("metrics", 3, 3, "job:s:0") == (
            "metrics: [####################] 3/3 job:s:0"
        )

    def test_download_progress_resume_counts_new_chunks_only(self, tmp_path, monkeypatch):
        root = "_metrics"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/gpu.20260904-120001.a.gz"
        second = f"{root}/{sub}/gpu.20260904-120002.b.gz"
        third = f"{root}/{sub}/gpu.20260904-120003.c.gz"
        c, connection, run_uri = self._gzip_download(
            monkeypatch, tmp_path, "metrics", [first], {first: b"a\n"}
        )
        c.download_metrics("job-1", tmp_path)
        connection.list_rows[run_uri + root + "/"] = [
            (f"/versions/RUN_ABC/{path}",) for path in (first, second, third)
        ]
        connection.content_by_uri[run_uri + second] = gzip.compress(b"b\n")
        connection.content_by_uri[run_uri + third] = gzip.compress(b"c\n")
        calls = []
        c.download_metrics("job-1", tmp_path, resume=True, progress=calls.append)
        status = c._artifact_progress_status
        assert calls == [
            status("metrics", 0, 2, sub),
            status("metrics", 1, 2, sub),
            status("metrics", 2, 2, sub),
            "\n",
        ]

    def test_download_progress_resume_nothing_new(self, tmp_path, monkeypatch):
        root = "_stdout"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/console.20260904-120001.a.gz"
        c, _connection, _run_uri = self._gzip_download(
            monkeypatch, tmp_path, "stdout", [first], {first: b"first\n"}
        )
        c.download_stdout_logs("job-1", tmp_path)
        destination = tmp_path / sub / "stdout.log"
        committed = destination.read_bytes()
        calls = []
        c.download_stdout_logs("job-1", tmp_path, resume=True, progress=calls.append)
        assert calls == [f"stdout: no new chunks for {sub}\n"]
        assert destination.read_bytes() == committed
        destination.write_bytes(committed + b"TORN")
        calls.clear()
        c.download_stdout_logs("job-1", tmp_path, resume=True, progress=calls.append)
        assert calls == [f"stdout: no new chunks for {sub}\n"]
        assert destination.read_bytes() == committed

    def test_download_progress_callback_error_skips_side_effects(self, tmp_path, monkeypatch):
        root = "_stdout"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/console.20260904-120001.a.gz"
        second = f"{root}/{sub}/console.20260904-120002.b.gz"
        c, connection, run_uri = self._gzip_download(
            monkeypatch, tmp_path, "stdout", [first], {first: b"first\n"}
        )
        c.download_stdout_logs("job-1", tmp_path)
        destination = tmp_path / sub / "stdout.log"
        committed = destination.read_bytes()
        destination.write_bytes(committed + b"TORN")

        calls = []

        def stop(message):
            calls.append(message)
            raise RuntimeError("stop")

        with pytest.raises(RuntimeError, match="stop"):
            c.download_stdout_logs("job-1", tmp_path / "fresh", progress=stop)
        assert calls == [c._artifact_progress_status("stdout", 0, 1, sub), "\n"]
        assert not (tmp_path / "fresh" / sub).exists()

        connection.list_rows[run_uri + root + "/"].append((f"/versions/RUN_ABC/{second}",))
        connection.content_by_uri[run_uri + second] = gzip.compress(b"second\n")
        with pytest.raises(RuntimeError, match="stop"):
            c.download_stdout_logs("job-1", tmp_path, resume=True, progress=stop)
        assert destination.read_bytes() == committed + b"TORN"

        connection.list_rows[run_uri + root + "/"] = [(f"/versions/RUN_ABC/{first}",)]
        with pytest.raises(RuntimeError, match="stop"):
            c.download_stdout_logs("job-1", tmp_path, resume=True, progress=stop)
        assert destination.read_bytes() == committed + b"TORN"

    def test_download_progress_chunk_failure_and_callback_failure(self, tmp_path, monkeypatch):
        root = "_metrics"
        sub = "job-1:training:0"
        names = [
            f"{root}/{sub}/gpu.20260904-12000{index}.c.gz" for index in (1, 2, 3)
        ]
        c, connection, run_uri = self._gzip_download(
            monkeypatch,
            tmp_path,
            "metrics",
            names[:1],
            {names[0]: b"a\n"},
        )
        c.download_metrics("job-1", tmp_path)
        destination = tmp_path / sub / "gpu.jsonl"
        committed = destination.read_bytes()
        connection.list_rows[run_uri + root + "/"] = [
            (f"/versions/RUN_ABC/{path}",) for path in names
        ]
        connection.content_by_uri[run_uri + names[1]] = gzip.compress(b"b\n")
        connection.content_by_uri[run_uri + names[2]] = b"not gzip"
        calls = []
        with pytest.raises(gzip.BadGzipFile):
            c.download_metrics(
                "job-1", tmp_path, resume=True, progress=calls.append, strict=True
            )
        assert c._artifact_progress_status("metrics", 1, 2, sub) in calls
        assert c._artifact_progress_status("metrics", 2, 2, sub) not in calls
        assert calls[-1] == "\n"
        # The chunk appended before the damaged one is kept for a resume.
        assert destination.read_bytes() == committed + b"b\n"

        connection.content_by_uri[run_uri + names[2]] = gzip.compress(b"c\n")

        def stop_on_second(message):
            calls.append(message)
            if " 2/" in message:
                raise RuntimeError("stop")

        def stop_on_first(message):
            calls.append(message)
            if " 1/" in message:
                raise RuntimeError("stop")

        calls.clear()
        with pytest.raises(RuntimeError, match="stop"):
            c.download_metrics("job-1", tmp_path, resume=True, progress=stop_on_first)
        assert calls[-1] == "\n"
        # The callback stops the download after the chunk it reports; that chunk
        # stays committed and a resume has nothing left to fetch.
        assert destination.read_bytes() == committed + b"b\nc\n"
        calls.clear()
        c.download_metrics("job-1", tmp_path, resume=True, progress=calls.append)
        assert calls == [f"metrics: no new chunks for {sub}\n"]

        fresh = tmp_path / "fresh"
        calls.clear()
        with pytest.raises(RuntimeError, match="stop"):
            c.download_metrics("job-1", fresh, progress=stop_on_second)
        assert calls[-1] == "\n"
        assert not list((fresh / sub).glob(".*.tmp"))
        assert not (fresh / sub / "gpu.jsonl").exists()

    def test_download_progress_setup_failures_close_line(self, tmp_path, monkeypatch):
        root = "_stdout"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/console.20260904-120001.a.gz"
        second = f"{root}/{sub}/console.20260904-120002.b.gz"
        c, connection, run_uri = self._gzip_download(
            monkeypatch, tmp_path, "stdout", [first], {first: b"first\n"}
        )
        not_a_dir = tmp_path / "file"
        not_a_dir.write_text("x")
        calls = []
        with pytest.raises(OSError):
            c.download_stdout_logs("job-1", not_a_dir, progress=calls.append)
        assert calls == [c._artifact_progress_status("stdout", 0, 1, sub), "\n"]

        if os.geteuid() == 0:
            pytest.skip("root ignores the read-only destination")
        c.download_stdout_logs("job-1", tmp_path)
        destination = tmp_path / sub / "stdout.log"
        manifest = destination.with_name(".stdout.log.manifest.json")
        before = manifest.read_text(encoding="utf-8")
        connection.list_rows[run_uri + root + "/"].append((f"/versions/RUN_ABC/{second}",))
        connection.content_by_uri[run_uri + second] = gzip.compress(b"second\n")
        destination.chmod(0o444)
        calls.clear()
        try:
            with pytest.raises(PermissionError):
                c.download_stdout_logs("job-1", tmp_path, resume=True, progress=calls.append)
        finally:
            destination.chmod(0o644)
        assert calls == [c._artifact_progress_status("stdout", 0, 1, sub), "\n"]
        assert manifest.read_text(encoding="utf-8") == before
        assert destination.read_bytes() == b"first\n"

    def test_download_progress_success_newline_error_keeps_commit(self, tmp_path, monkeypatch):
        root = "_metrics"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/gpu.20260904-120001.a.gz"
        second = f"{root}/{sub}/gpu.20260904-120002.b.gz"
        c, connection, run_uri = self._gzip_download(
            monkeypatch, tmp_path, "metrics", [first], {first: b"a\n"}
        )

        destination = tmp_path / sub / "gpu.jsonl"
        manifest = destination.with_name(".gpu.jsonl.manifest.json")
        seen = []

        def fail_on_newline(message):
            if message == "\n":
                seen.append(json.loads(manifest.read_text(encoding="utf-8"))["names"])
                raise RuntimeError("newline")

        with pytest.raises(RuntimeError, match="newline"):
            c.download_metrics("job-1", tmp_path, progress=fail_on_newline)
        assert seen == [[first]]
        assert destination.read_bytes() == b"a\n"
        assert json.loads(manifest.read_text(encoding="utf-8"))["names"] == [first]

        connection.list_rows[run_uri + root + "/"].append((f"/versions/RUN_ABC/{second}",))
        connection.content_by_uri[run_uri + second] = gzip.compress(b"b\n")
        with pytest.raises(RuntimeError, match="newline"):
            c.download_metrics("job-1", tmp_path, resume=True, progress=fail_on_newline)
        assert seen == [[first], [first, second]]
        assert destination.read_bytes() == b"a\nb\n"
        assert json.loads(manifest.read_text(encoding="utf-8"))["names"] == [first, second]
        calls = []
        c.download_metrics("job-1", tmp_path, resume=True, progress=calls.append)
        assert calls == [f"metrics: no new chunks for {sub}\n"]

    @pytest.mark.parametrize("newline_error", [RuntimeError, KeyboardInterrupt])
    def test_download_progress_failure_newline_error_keeps_download_error(
        self, newline_error, tmp_path, monkeypatch
    ):
        root = "_stdout"
        sub = "job-1:training:0"
        good = f"{root}/{sub}/console.20260904-120001.a.gz"
        later = f"{root}/{sub}/console.20260904-120002.b.gz"
        bad = f"{root}/{sub}/console.20260904-120003.c.gz"
        c, connection, run_uri = self._gzip_download(
            monkeypatch, tmp_path, "stdout", [good], {good: b"good\n"}
        )
        destination = tmp_path / sub / "stdout.log"
        fresh = tmp_path / "fresh"
        seen = []

        def fail_on_newline(message):
            if message == "\n":
                fresh_temps = list((fresh / sub).glob(".*.tmp"))
                seen.append((destination.read_bytes(), fresh_temps))
                raise newline_error("newline")

        c.download_stdout_logs("job-1", tmp_path)
        connection.list_rows[run_uri + root + "/"] += [
            (f"/versions/RUN_ABC/{later}",),
            (f"/versions/RUN_ABC/{bad}",),
        ]
        connection.content_by_uri[run_uri + later] = gzip.compress(b"later\n")
        connection.content_by_uri[run_uri + bad] = b"not gzip"
        with pytest.raises(gzip.BadGzipFile):
            c.download_stdout_logs(
                "job-1", tmp_path, resume=True, progress=fail_on_newline, strict=True
            )
        # The line closes after the chunks before the damaged one are saved.
        assert seen == [(b"good\nlater\n", [])]
        assert destination.read_bytes() == b"good\nlater\n"

        seen.clear()
        with pytest.raises(gzip.BadGzipFile):
            c.download_stdout_logs("job-1", fresh, progress=fail_on_newline, strict=True)
        assert [fresh_temps for _bytes, fresh_temps in seen] == [[]]
        assert not list((fresh / sub).glob(".*.tmp"))
        assert not (fresh / sub / "stdout.log").exists()

    def test_download_progress_interrupt_closes_line(self, tmp_path, monkeypatch):
        sub = "job-1:training:0"
        first = f"_stdout/{sub}/console.20260904-120001.a.gz"
        c, _connection, _run_uri = self._gzip_download(
            monkeypatch, tmp_path, "stdout", [first], {first: b"first\n"}
        )

        def interrupt(*_args):
            raise KeyboardInterrupt

        monkeypatch.setattr(c, "_fetch_artifact_chunk", interrupt)
        calls = []
        with pytest.raises(KeyboardInterrupt):
            c.download_stdout_logs("job-1", tmp_path, progress=calls.append)
        assert calls == [c._artifact_progress_status("stdout", 0, 1, sub), "\n"]
        assert not list((tmp_path / sub).glob(".*.tmp"))

    def test_an_empty_training_stream_is_named_as_training_metrics(self, tmp_path, monkeypatch):
        """The training stream shares ``_metrics/`` with the GPU one, so its empty line
        must not read as if the GPU download found nothing."""
        c, _connection, _run_uri = self._gzip_download(monkeypatch, tmp_path, "metrics", [], {})
        monkeypatch.setattr(c, "_job_for_upload_warnings", lambda _: ({"status": "SUCCEEDED"}, None))
        calls = []
        assert c.download_training_metrics("job-1", tmp_path, progress=calls.append) == []
        assert calls == ["training metrics: found 0 chunks\n"]

    def test_a_named_sub_job_without_training_rows_is_left_out_not_an_error(self, tmp_path, monkeypatch):
        """Only training sub-jobs write training rows, so naming a sampling sub-job beside a
        training one returns the training one's rows instead of raising NoChunksError."""
        sub = "job-1:training:0"
        chunk = f"_metrics/{sub}/training.20260904-120001.a.gz"
        c, _connection, _run_uri = self._gzip_download(
            monkeypatch, tmp_path, "metrics", [chunk], {chunk: b'{"kind": "training"}\n'}
        )
        monkeypatch.setattr(c, "_job_for_upload_warnings", lambda _: ({"status": "SUCCEEDED"}, None))

        result = c.download_training_metrics("job-1", tmp_path, sub_job_ids=[sub, "job-1:sampling:0"])
        assert [entry["sub_job_id"] for entry in result] == [sub]
        assert (tmp_path / sub / "training.jsonl").read_bytes() == b'{"kind": "training"}\n'

        # Naming only sub-jobs without rows is an empty result, not an error.
        assert c.download_training_metrics("job-1", tmp_path / "none", sub_job_ids=["job-1:sampling:0"]) == []

    def test_a_named_sub_job_without_gpu_metrics_still_raises(self, tmp_path, monkeypatch):
        """The GPU stream keeps raising: every sub-job with GPUs uploads it."""
        sub = "job-1:training:0"
        chunk = f"_metrics/{sub}/gpu.20260904-120001.a.gz"
        c, _connection, _run_uri = self._gzip_download(
            monkeypatch, tmp_path, "metrics", [chunk], {chunk: b'{"gpu_util_pct": 1}\n'}
        )
        monkeypatch.setattr(c, "_job_for_upload_warnings", lambda _: ({"status": "SUCCEEDED"}, None))
        with pytest.raises(nc.NoChunksError) as excinfo:
            c.download_metrics("job-1", tmp_path, sub_job_ids=["job-1:sampling:0"])
        assert str(excinfo.value) == (
            "no metrics chunks for sub-job(s) job-1:sampling:0; sub-jobs with metrics: job-1:training:0"
        )

    def test_download_progress_empty_and_directory_destination(self, tmp_path, monkeypatch):
        c, _connection, _run_uri = self._gzip_download(monkeypatch, tmp_path, "stdout", [], {})
        calls = []
        assert c.download_stdout_logs("job-1", tmp_path, progress=calls.append) == []
        assert calls == ["stdout: found 0 chunks\n"]
        calls.clear()
        assert c.download_stdout_logs("job-1", tmp_path, resume=True, progress=calls.append) == []
        assert calls == ["stdout: found 0 chunks\n"]

        root = "_stdout"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/console.20260904-120001.a.gz"
        c, _connection, _run_uri = self._gzip_download(
            monkeypatch, tmp_path, "stdout", [first], {first: b"first\n"}
        )
        destination = tmp_path / "dir-case" / sub / "stdout.log"
        destination.mkdir(parents=True)
        manifest = destination.with_name(".stdout.log.manifest.json")
        manifest.write_text(
            json.dumps({"version": 1, "names": [first], "bytes": 0}),
            encoding="utf-8",
        )
        calls.clear()
        with pytest.raises(OSError):
            c.download_stdout_logs(
                "job-1", tmp_path / "dir-case", resume=True, progress=calls.append
            )
        assert calls == [
            c._artifact_progress_status("stdout", 0, 1, sub),
            c._artifact_progress_status("stdout", 1, 1, sub),
            "\n",
        ]
        assert destination.is_dir()
        assert not list(destination.parent.glob(".*.tmp"))

    def test_download_progress_non_prefix_reports_full_list(self, tmp_path, monkeypatch):
        root = "_stdout"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/console.20260904-120002.b.gz"
        earlier = f"{root}/{sub}/console.20260904-120001.a.gz"
        c, connection, run_uri = self._gzip_download(
            monkeypatch, tmp_path, "stdout", [first], {first: b"b\n"}
        )
        c.download_stdout_logs("job-1", tmp_path)
        connection.list_rows[run_uri + root + "/"].insert(0, (f"/versions/RUN_ABC/{earlier}",))
        connection.content_by_uri[run_uri + earlier] = gzip.compress(b"a\n")
        calls = []
        c.download_stdout_logs("job-1", tmp_path, resume=True, progress=calls.append)
        assert calls[0] == c._artifact_progress_status("stdout", 0, 2, sub)
        assert calls.count(calls[0]) == 1

    def test_download_progress_rebuild_chunk_failure_keeps_previous(self, tmp_path, monkeypatch):
        root = "_stdout"
        sub = "job-1:training:0"
        good = f"{root}/{sub}/console.20260904-120001.a.gz"
        bad = f"{root}/{sub}/console.20260904-120002.b.gz"
        c, connection, run_uri = self._gzip_download(
            monkeypatch,
            tmp_path,
            "stdout",
            [good, bad],
            {good: b"new\n", bad: b"ignored\n"},
        )
        connection.content_by_uri[run_uri + bad] = b"not gzip"
        destination = tmp_path / sub / "stdout.log"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"previous\n")
        calls = []
        with pytest.raises(gzip.BadGzipFile):
            c.download_stdout_logs("job-1", tmp_path, progress=calls.append, strict=True)
        assert c._artifact_progress_status("stdout", 1, 2, sub) in calls
        assert c._artifact_progress_status("stdout", 2, 2, sub) not in calls
        assert calls[-1] == "\n"
        assert destination.read_bytes() == b"previous\n"
        assert not list(destination.parent.glob(".*.tmp"))

    def test_download_progress_shorter_file_rebuilds_full_list(self, tmp_path, monkeypatch):
        root = "_metrics"
        sub = "job-1:training:0"
        first = f"{root}/{sub}/gpu.20260904-120001.a.gz"
        second = f"{root}/{sub}/gpu.20260904-120002.b.gz"
        c, _connection, _run_uri = self._gzip_download(
            monkeypatch,
            tmp_path,
            "metrics",
            [first, second],
            {first: b"a\n", second: b"b\n"},
        )
        c.download_metrics("job-1", tmp_path)
        destination = tmp_path / sub / "gpu.jsonl"
        destination.write_bytes(b"x")
        calls = []
        c.download_metrics("job-1", tmp_path, resume=True, progress=calls.append)
        assert calls[0] == c._artifact_progress_status("metrics", 0, 2, sub)
        assert calls.count(calls[0]) == 1
        assert destination.read_bytes() == b"a\nb\n"

    def test_fetch_execution_logs_errors_when_experiment_run_missing_fields(self):
        c = _make_client(get_json={"experiment_name": "DB.SCH.EXP"})
        with pytest.raises(ValueError, match="experiment_run_name"):
            c.fetch_execution_logs("job-1")


class TestChunkedArtifactDownload:
    """Parallel fetch, ordered commit, retries, and durable progress."""

    @pytest.fixture(autouse=True)
    def _default_telemetry_env(self, monkeypatch):
        # Telemetry defaults are part of what these tests check.
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        monkeypatch.delenv(nc.DISABLE_TELEMETRY_ENV, raising=False)

    RUN_URI = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"
    SUB = "job-1:training:0"

    def _setup(self, monkeypatch, *, subs=None, count=6, artifact_name="stdout", chunk_name=None):
        subs = subs or [self.SUB]
        root = f"_{artifact_name}"
        chunk_name = chunk_name or ("console" if artifact_name == "stdout" else "gpu")
        paths = {
            sub: [
                f"{root}/{sub}/{chunk_name}.20260904-"
                f"{12 + index // 3600:02d}{index // 60 % 60:02d}{index % 60:02d}.c{index:04d}.gz"
                for index in range(count)
            ]
            for sub in subs
        }
        connection = TestExecutionLogDownload.FakeArtifactConnection(
            {
                self.RUN_URI + root + "/": [
                    (f"/versions/RUN_ABC/{path}",) for sub in subs for path in reversed(paths[sub])
                ],
                self.RUN_URI + "checkpoints/" + root + "/": [],
            },
            {
                self.RUN_URI + path: gzip.compress(f"{sub} line {index}\n".encode())
                for sub in subs
                for index, path in enumerate(paths[sub])
            },
        )
        c = _make_client()
        sleeps = []
        c._artifact_retry_policy = nc._artifact_download.RetryPolicy(sleep=sleeps.append)
        opened = []

        def open_connection():
            opened.append(connection)
            return connection

        monkeypatch.setattr(c, "_experiment_run_uri", lambda _: (self.RUN_URI, "RUN_ABC"))
        monkeypatch.setattr(c, "_open_experiment_artifact_connection", open_connection)
        return c, connection, paths, sleeps, opened

    @staticmethod
    def _expected(sub, indexes):
        return "".join(f"{sub} line {index}\n" for index in indexes).encode()

    @staticmethod
    def _settle(timeout=5.0):
        """Wait for workers a cancelled download abandoned (it does not join them)."""
        deadline = time.monotonic() + timeout
        while any(t.name.startswith("cortex-training-download") for t in threading.enumerate()):
            assert time.monotonic() < deadline, "download workers kept running"
            time.sleep(0.01)

    @staticmethod
    def _gets(connection, needle=""):
        return [cmd for cmd in connection.commands if cmd.startswith("GET ") and needle in cmd]

    def test_out_of_order_completion_commits_in_sorted_order(self, tmp_path, monkeypatch):
        subs = [self.SUB, "job-1:training:1"]
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, subs=subs, count=3)
        first = self.RUN_URI + paths[self.SUB][0]
        last = self.RUN_URI + paths[subs[1]][-1]
        last_done = threading.Event()
        finished = []
        lock = threading.Lock()

        def on_get(source):
            if source == first:
                # The first chunk completes only after the very last one has.
                assert last_done.wait(5)
            with lock:
                finished.append(source)
            if source == last:
                last_done.set()

        connection.on_get = on_get
        result = c.download_stdout_logs("job-1", tmp_path, max_workers=6)

        assert finished.index(last) < finished.index(first)
        for sub in subs:
            assert (tmp_path / sub / "stdout.log").read_bytes() == self._expected(sub, range(3))
            manifest = json.loads((tmp_path / sub / ".stdout.log.manifest.json").read_text())
            assert manifest["names"] == paths[sub]
            assert manifest["bytes"] == (tmp_path / sub / "stdout.log").stat().st_size
        assert [entry["sub_job_id"] for entry in result] == subs
        assert not list((tmp_path / self.SUB).glob(".*.partial*"))
        assert connection.closed

    def test_get_targets_are_raw_paths_without_percent_encoding(self, tmp_path, monkeypatch):
        c, connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=2)
        c.download_stdout_logs("job-1", tmp_path, max_workers=2)
        for command in self._gets(connection):
            target = command.rsplit(" ", 1)[1]
            assert target.startswith("'file:///") and "%" not in target and ":" not in target[8:]

    def test_fetch_window_bounds_chunks_held_ahead_of_the_writer(self, tmp_path, monkeypatch):
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=8)
        monkeypatch.setattr(c, "_DOWNLOAD_WINDOW_PER_WORKER", 1)
        first = self.RUN_URI + paths[self.SUB][0]
        started_while_first_blocked = []

        def on_get(source):
            if source == first:
                time.sleep(0.3)
                started_while_first_blocked.append(len(self._gets(connection)))

        connection.on_get = on_get
        c.download_stdout_logs("job-1", tmp_path, max_workers=2)

        # Two workers with a window of two: only the blocked chunk and one more.
        assert started_while_first_blocked == [2]
        assert (tmp_path / self.SUB / "stdout.log").read_bytes() == self._expected(self.SUB, range(8))

    def test_transient_get_failure_is_retried(self, tmp_path, monkeypatch):
        errors = importlib.import_module("snowflake.connector.errors")
        c, connection, paths, sleeps, _opened = self._setup(monkeypatch, count=4)
        flaky = self.RUN_URI + paths[self.SUB][2]
        failures = [errors.OperationalError(msg="connection reset"), ConnectionResetError("reset")]

        def on_get(source):
            if source == flaky and failures:
                raise failures.pop(0)

        connection.on_get = on_get
        c.download_stdout_logs("job-1", tmp_path)

        assert len(sleeps) == 2
        assert len(self._gets(connection, flaky)) == 3
        assert len(self._gets(connection)) == 6
        assert (tmp_path / self.SUB / "stdout.log").read_bytes() == self._expected(self.SUB, range(4))

    def test_transient_list_failure_is_retried(self, tmp_path, monkeypatch):
        errors = importlib.import_module("snowflake.connector.errors")
        c, connection, _paths, sleeps, _opened = self._setup(monkeypatch, count=2)
        cursor_cls = connection.Cursor
        failures = [errors.ServiceUnavailableError(msg="503")]

        class FlakyCursor(cursor_cls):
            def execute(self, statement):
                if statement.startswith("LIST ") and failures:
                    raise failures.pop(0)
                return super().execute(statement)

        monkeypatch.setattr(connection, "cursor", lambda: FlakyCursor(connection))
        c.download_stdout_logs("job-1", tmp_path)

        assert len(sleeps) == 1
        assert (tmp_path / self.SUB / "stdout.log").read_bytes() == self._expected(self.SUB, range(2))

    def test_expired_session_reopens_the_connection(self, tmp_path, monkeypatch):
        errors = importlib.import_module("snowflake.connector.errors")
        c, connection, paths, sleeps, opened = self._setup(monkeypatch, count=3)
        target = self.RUN_URI + paths[self.SUB][1]
        failures = [errors.ProgrammingError(msg="token expired", errno=390114)]

        def on_get(source):
            if source == target and failures:
                raise failures.pop(0)

        connection.on_get = on_get
        c.download_stdout_logs("job-1", tmp_path)

        assert len(opened) == 2
        assert len(sleeps) == 1
        assert (tmp_path / self.SUB / "stdout.log").read_bytes() == self._expected(self.SUB, range(3))

    def test_permanent_error_fails_fast_and_keeps_committed_chunks(self, tmp_path, monkeypatch):
        errors = importlib.import_module("snowflake.connector.errors")
        c, connection, paths, sleeps, _opened = self._setup(monkeypatch, count=5)
        denied = self.RUN_URI + paths[self.SUB][3]

        def on_get(source):
            if source == denied:
                raise errors.ProgrammingError(msg="not authorized", errno=2003)

        connection.on_get = on_get
        with pytest.raises(errors.ProgrammingError, match="not authorized"):
            c.download_stdout_logs("job-1", tmp_path, max_workers=1)
        self._settle()

        assert sleeps == []
        assert len(self._gets(connection, denied)) == 1
        destination = tmp_path / self.SUB / "stdout.log"
        assert not destination.exists()
        partial = tmp_path / self.SUB / ".stdout.log.partial"
        assert partial.read_bytes() == self._expected(self.SUB, range(3))
        manifest = json.loads((tmp_path / self.SUB / ".stdout.log.partial.manifest.json").read_text())
        assert manifest == {"bytes": partial.stat().st_size, "names": paths[self.SUB][:3], "version": 1}

    def test_retries_are_bounded(self, tmp_path, monkeypatch):
        errors = importlib.import_module("snowflake.connector.errors")
        c, connection, paths, sleeps, _opened = self._setup(monkeypatch, count=2)
        broken = self.RUN_URI + paths[self.SUB][0]

        def on_get(source):
            if source == broken:
                raise errors.OperationalError(msg="network down")

        connection.on_get = on_get
        with pytest.raises(errors.OperationalError):
            c.download_stdout_logs("job-1", tmp_path)
        self._settle()

        attempts = c._artifact_retry_policy.attempts
        assert len(self._gets(connection, broken)) == attempts
        assert len(sleeps) == attempts - 1

    @pytest.mark.parametrize(
        ("artifact_name", "method_name", "destination_name"),
        [("stdout", "download_stdout_logs", "stdout.log"), ("metrics", "download_metrics", "gpu.jsonl")],
    )
    def test_interrupted_first_download_resumes_from_saved_chunks(
        self, artifact_name, method_name, destination_name, tmp_path, monkeypatch
    ):
        c, connection, paths, _sleeps, _opened = self._setup(
            monkeypatch, count=8, artifact_name=artifact_name
        )
        chunks = paths[self.SUB]
        interrupt_at = self.RUN_URI + chunks[5]
        interrupt = [True]

        def on_get(source):
            if source == interrupt_at and interrupt:
                interrupt.clear()
                raise KeyboardInterrupt

        connection.on_get = on_get
        download = getattr(c, method_name)
        with pytest.raises(KeyboardInterrupt):
            download("job-1", tmp_path, max_workers=3)
        self._settle()

        destination = tmp_path / self.SUB / destination_name
        assert not destination.exists()
        partial_manifest = json.loads(
            (tmp_path / self.SUB / f".{destination_name}.partial.manifest.json").read_text()
        )
        assert partial_manifest["names"] == chunks[:5]

        gets_before = len(self._gets(connection))
        download("job-1", tmp_path, resume=True, max_workers=3)
        resumed = self._gets(connection)[gets_before:]

        assert sorted(cmd.split(" ")[1].strip("'") for cmd in resumed) == [
            self.RUN_URI + path for path in chunks[5:]
        ]
        assert destination.read_bytes() == self._expected(self.SUB, range(8))
        manifest = json.loads((tmp_path / self.SUB / f".{destination_name}.manifest.json").read_text())
        assert manifest == {"bytes": destination.stat().st_size, "names": chunks, "version": 1}
        assert not list((tmp_path / self.SUB).glob(f".{destination_name}.partial*"))

    def test_periodic_checkpoint_survives_a_hard_kill(self, tmp_path, monkeypatch):
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=7)
        chunks = paths[self.SUB]
        monkeypatch.setattr(c, "_DOWNLOAD_CHECKPOINT_CHUNKS", 2)
        monkeypatch.setattr(c, "_DOWNLOAD_CHECKPOINT_SECONDS", 3600.0)
        monkeypatch.setattr(c, "_DOWNLOAD_WINDOW_PER_WORKER", 1)
        partial = tmp_path / self.SUB / ".stdout.log.partial"
        partial_manifest = tmp_path / self.SUB / ".stdout.log.partial.manifest.json"
        seen = {}
        killed = [True]

        def on_get(source):
            if source == self.RUN_URI + chunks[4]:
                seen["manifest"] = json.loads(partial_manifest.read_text())
                seen["size"] = partial.stat().st_size
            if source == self.RUN_URI + chunks[5] and killed:
                killed.clear()
                raise SystemExit("killed")

        # A hard kill runs no cleanup: the file keeps whatever was written and
        # the manifest stays at the last checkpoint.
        def no_cleanup(writer):
            writer._output.close()

        monkeypatch.setattr(nc._artifact_download.ChunkWriter, "abort", no_cleanup)
        connection.on_get = on_get
        with pytest.raises(SystemExit):
            c.download_stdout_logs("job-1", tmp_path, max_workers=1)
        self._settle()

        assert seen["manifest"]["names"] == chunks[:4]
        assert seen["manifest"]["bytes"] <= seen["size"]
        assert json.loads(partial_manifest.read_text())["names"] == chunks[:4]
        assert partial.stat().st_size > seen["manifest"]["bytes"]

        gets_before = len(self._gets(connection))
        c.download_stdout_logs("job-1", tmp_path, resume=True, max_workers=1)
        assert len(self._gets(connection)) - gets_before == 3
        destination = tmp_path / self.SUB / "stdout.log"
        assert destination.read_bytes() == self._expected(self.SUB, range(7))

    def test_non_resume_run_restarts_an_interrupted_download(self, tmp_path, monkeypatch):
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=4)
        target = self.RUN_URI + paths[self.SUB][2]
        interrupt = [True]

        def on_get(source):
            if source == target and interrupt:
                interrupt.clear()
                raise KeyboardInterrupt

        connection.on_get = on_get
        with pytest.raises(KeyboardInterrupt):
            c.download_stdout_logs("job-1", tmp_path, max_workers=1)
        self._settle()
        gets_before = len(self._gets(connection))
        c.download_stdout_logs("job-1", tmp_path, max_workers=1)
        assert len(self._gets(connection)) - gets_before == 4
        assert (tmp_path / self.SUB / "stdout.log").read_bytes() == self._expected(self.SUB, range(4))

    def test_resume_appends_to_completed_file_and_drops_stale_partial(self, tmp_path, monkeypatch):
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=5)
        chunks = paths[self.SUB]
        listing = connection.list_rows[self.RUN_URI + "_stdout/"]
        connection.list_rows[self.RUN_URI + "_stdout/"] = [
            row for row in listing if row[0].removeprefix("/versions/RUN_ABC/") in chunks[:3]
        ]
        c.download_stdout_logs("job-1", tmp_path)
        directory = tmp_path / self.SUB
        (directory / ".stdout.log.partial").write_bytes(b"stale")
        (directory / ".stdout.log.partial.manifest.json").write_text(
            json.dumps({"bytes": 5, "names": chunks[:1], "version": 1})
        )
        connection.list_rows[self.RUN_URI + "_stdout/"] = listing
        gets_before = len(self._gets(connection))

        c.download_stdout_logs("job-1", tmp_path, resume=True)

        assert len(self._gets(connection)) - gets_before == 2
        assert (directory / "stdout.log").read_bytes() == self._expected(self.SUB, range(5))
        assert not list(directory.glob(".stdout.log.partial*"))

    def test_leftover_temp_files_are_removed(self, tmp_path, monkeypatch, caplog):
        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=2)
        directory = tmp_path / self.SUB
        directory.mkdir(parents=True)
        removed = [
            directory / ".stdout-abc123.tmp",
            directory / ".stdout.log.manifest-xyz.tmp",
            directory / ".stdout.log.partial.manifest-xyz.tmp",
            directory / ".manifest-old.tmp",
        ]
        kept = [
            # Recently written: may belong to a download still running here.
            directory / ".manifest-recent.tmp",
            directory / ".stdout-recent.tmp",
            directory / ".stdout.log.manifest-recent.tmp",
            # Not ours to remove.
            directory / ".gpu-metrics-abc.tmp",
            directory / ".gpu.jsonl.manifest-abc.tmp",
            directory / "notes.tmp",
        ]
        for path in removed + kept:
            path.write_bytes(b"x")
        old = time.time() - 2 * c._TEMP_ORPHAN_MIN_AGE_SECONDS
        for path in removed + kept[3:]:
            os.utime(path, (old, old))

        with caplog.at_level(logging.DEBUG):
            c.download_stdout_logs("job-1", tmp_path)

        assert [path for path in removed if path.exists()] == []
        assert caplog.text.count("removed leftover temporary file") == len(removed)
        assert "could not remove leftover temporary file" not in caplog.text
        assert all(path.exists() for path in kept)
        assert (directory / "stdout.log").read_bytes() == self._expected(self.SUB, range(2))

    def test_on_chunk_reports_each_committed_chunk(self, tmp_path, monkeypatch):
        subs = [self.SUB, "job-1:training:1"]
        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, subs=subs, count=3)
        calls = []
        c.download_stdout_logs(
            "job-1", tmp_path, on_chunk=lambda *args: calls.append(args), max_workers=4
        )
        for sub in subs:
            mine = [call for call in calls if call[0] == sub]
            assert [call[1] for call in mine] == [0, 1, 2, 3]
            assert {call[2] for call in mine} == {3}
            assert mine[0][3] == 0
            assert mine[-1][3] == (tmp_path / sub / "stdout.log").stat().st_size
        assert calls[0][0] == self.SUB and calls[-1][0] == subs[1]

    def test_sub_job_filter_downloads_only_selected(self, tmp_path, monkeypatch):
        subs = [self.SUB, "job-1:training:1"]
        c, connection, _paths, _sleeps, _opened = self._setup(monkeypatch, subs=subs, count=2)
        result = c.download_stdout_logs("job-1", tmp_path, sub_job_ids=[subs[1]])
        assert [entry["sub_job_id"] for entry in result] == [subs[1]]
        assert not (tmp_path / self.SUB).exists()
        assert all(subs[1] in cmd for cmd in self._gets(connection))

    def test_sub_job_filter_rejects_unknown_sub_job_before_get(self, tmp_path, monkeypatch):
        c, connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=2)
        with pytest.raises(ValueError, match="job-1:sampling:0.*job-1:training:0"):
            c.download_stdout_logs("job-1", tmp_path, sub_job_ids=["job-1:sampling:0"])
        self._settle()
        assert self._gets(connection) == []
        assert connection.closed

    @pytest.mark.parametrize("workers", [0, -1, True, 1.5])
    def test_max_workers_must_be_positive_int(self, tmp_path, monkeypatch, workers):
        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=1)
        with pytest.raises(ValueError, match="max_workers"):
            c.download_stdout_logs("job-1", tmp_path, max_workers=workers)
        self._settle()

    def test_damaged_chunk_is_fetched_again_then_kept(self, tmp_path, monkeypatch):
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=4)
        chunks = paths[self.SUB]
        target = self.RUN_URI + chunks[1]
        good = connection.content_by_uri[target]
        # A truncated member writes part of its output before failing.
        connection.content_by_uri[target] = good[:-6]

        def on_get(source):
            if source == target and len(self._gets(connection, target)) > 1:
                connection.content_by_uri[target] = good

        connection.on_get = on_get
        result = c.download_stdout_logs("job-1", tmp_path, max_workers=2)

        assert len(self._gets(connection, target)) == 2
        assert (tmp_path / self.SUB / "stdout.log").read_bytes() == self._expected(self.SUB, range(4))
        assert "skipped_chunks" not in result[0] and "warnings" not in result[0]

    @pytest.mark.parametrize("damage", [b"not gzip", "truncated"])
    def test_damaged_chunk_is_skipped_and_recorded(self, tmp_path, monkeypatch, damage):
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=4)
        chunks = paths[self.SUB]
        target = self.RUN_URI + chunks[2]
        good = connection.content_by_uri[target]
        connection.content_by_uri[target] = good[:-6] if damage == "truncated" else damage

        result = c.download_stdout_logs("job-1", tmp_path)

        destination = tmp_path / self.SUB / "stdout.log"
        assert destination.read_bytes() == self._expected(self.SUB, [0, 1, 3])
        assert len(self._gets(connection, target)) == 2
        assert result[0]["chunk_count"] == 3
        assert result[0]["skipped_chunks"] == [target]
        [warning] = result[0]["warnings"]
        assert warning.startswith(f"skipped chunk {target}: it is damaged")
        assert warning.endswith("the saved file is missing that chunk's output")
        manifest = json.loads((tmp_path / self.SUB / ".stdout.log.manifest.json").read_text())
        assert manifest == {
            "bytes": destination.stat().st_size,
            "names": chunks,
            "skipped": [chunks[2]],
            "version": 1,
        }

        # Resume keeps the gap without fetching the damaged chunk again.
        gets_before = len(self._gets(connection))
        resumed = c.download_stdout_logs("job-1", tmp_path, resume=True)
        assert len(self._gets(connection)) == gets_before
        assert resumed[0]["skipped_chunks"] == [target]
        assert "earlier download" in resumed[0]["warnings"][0]

        # A full download retries it.
        connection.content_by_uri[target] = good
        full = c.download_stdout_logs("job-1", tmp_path)
        assert destination.read_bytes() == self._expected(self.SUB, range(4))
        assert "skipped_chunks" not in full[0]
        assert "skipped" not in json.loads(
            (tmp_path / self.SUB / ".stdout.log.manifest.json").read_text()
        )

    def test_strict_keeps_fail_fast_on_damaged_chunk(self, tmp_path, monkeypatch):
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=3)
        target = self.RUN_URI + paths[self.SUB][1]
        connection.content_by_uri[target] = b"not gzip"
        with pytest.raises(gzip.BadGzipFile):
            c.download_stdout_logs("job-1", tmp_path, strict=True, max_workers=1)
        self._settle()
        assert len(self._gets(connection, target)) == 1
        partial = tmp_path / self.SUB / ".stdout.log.partial"
        assert partial.read_bytes() == self._expected(self.SUB, [0])

    def test_strict_damaged_chunk_keeps_prefix_and_resume_refetches_rest(
        self, tmp_path, monkeypatch
    ):
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=5)
        chunks = paths[self.SUB]
        target = self.RUN_URI + chunks[2]
        good = connection.content_by_uri[target]
        connection.content_by_uri[target] = good[:-6]  # truncated: writes, then fails
        with pytest.raises(EOFError):
            c.download_stdout_logs("job-1", tmp_path, strict=True, max_workers=3)
        self._settle()
        partial = tmp_path / self.SUB / ".stdout.log.partial"
        assert partial.read_bytes() == self._expected(self.SUB, [0, 1])
        connection.content_by_uri[target] = good
        gets_before = len(self._gets(connection))
        c.download_stdout_logs("job-1", tmp_path, resume=True, strict=True, max_workers=3)
        resumed = self._gets(connection)[gets_before:]
        assert sorted(cmd.split(" ")[1].strip("'") for cmd in resumed) == [
            self.RUN_URI + path for path in chunks[2:]
        ]
        assert (tmp_path / self.SUB / "stdout.log").read_bytes() == self._expected(
            self.SUB, range(5)
        )

    def test_chunk_removed_after_listing_is_skipped_without_retries(
        self, tmp_path, monkeypatch
    ):
        errors = importlib.import_module("snowflake.connector.errors")
        c, connection, paths, sleeps, _opened = self._setup(monkeypatch, count=4)
        gone = self.RUN_URI + paths[self.SUB][1]

        def on_get(source):
            if source == gone:
                raise errors.OperationalError(msg="the file does not exist.", errno=253006)

        connection.on_get = on_get
        result = c.download_stdout_logs("job-1", tmp_path)

        assert sleeps == []
        assert len(self._gets(connection, gone)) == 1
        assert (tmp_path / self.SUB / "stdout.log").read_bytes() == self._expected(
            self.SUB, [0, 2, 3]
        )
        assert result[0]["skipped_chunks"] == [gone]
        assert "no longer exists" in result[0]["warnings"][0]

        with pytest.raises(errors.OperationalError):
            c.download_stdout_logs("job-1", tmp_path / "strict", strict=True)
        self._settle()

    def test_cancel_during_retry_backoff_returns_promptly_and_saves(
        self, tmp_path, monkeypatch
    ):
        errors = importlib.import_module("snowflake.connector.errors")
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=6)
        # Real, long backoff waits: only the cancel event can end them early.
        c._artifact_retry_policy = nc._artifact_download.RetryPolicy(initial_delay=30.0)
        chunks = paths[self.SUB]
        failing = self.RUN_URI + chunks[3]
        in_backoff = threading.Event()

        def on_get(source):
            if source == self.RUN_URI + chunks[1]:
                assert in_backoff.wait(5)
            if source == failing:
                in_backoff.set()
                raise errors.OperationalError(msg="network down")

        def progress(sub_job_id, done, total, nbytes):
            if done == 3:
                raise nc.DownloadCancelled("leaving")

        connection.on_get = on_get
        started = time.monotonic()
        with pytest.raises(nc.DownloadCancelled):
            c.download_stdout_logs("job-1", tmp_path, max_workers=4, on_chunk=progress)
        self._settle()
        assert time.monotonic() - started < 5

        deadline = time.monotonic() + 5
        while any(t.name.startswith("cortex-training-download") for t in threading.enumerate()):
            assert time.monotonic() < deadline, "download workers kept running"
            time.sleep(0.05)
        assert len(self._gets(connection, failing)) == 1
        manifest = json.loads((tmp_path / self.SUB / ".stdout.log.partial.manifest.json").read_text())
        assert manifest["names"] == chunks[:3]

    def test_cancellation_is_not_reported_as_a_failure(self, tmp_path, monkeypatch):
        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=3)
        c._metric_emitter = MagicMock()

        def progress(*_args):
            raise nc.DownloadCancelled("stop")

        # Explicitly turning success telemetry off also silences cancellations.
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "0")
        with pytest.raises(nc.DownloadCancelled):
            c.download_stdout_logs("job-1", tmp_path, on_chunk=progress)
        self._settle()
        c._metric_emitter.emit.assert_not_called()

        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        with pytest.raises(nc.DownloadCancelled):
            c.download_stdout_logs("job-1", tmp_path, on_chunk=progress)
        self._settle()
        operation, value = c._metric_emitter.emit.call_args.args
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert operation == "download_stdout_logs"
        assert value["cancelled"] is True and value["success"] is False
        assert "error_message" not in value and "error.type" not in attributes

    def _emitted(self, c, operation="download_stdout_logs"):
        calls = [call for call in c._metric_emitter.emit.call_args_list if call.args[0] == operation]
        assert len(calls) == 1, calls
        return calls[0].args[1], calls[0].kwargs["attributes"]

    def test_training_metrics_download_successes_are_on_by_default_too(self, tmp_path, monkeypatch):
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        c, _connection, _paths, _sleeps, _opened = self._setup(
            monkeypatch, count=3, artifact_name="metrics", chunk_name="training"
        )
        c._metric_emitter = MagicMock()

        c.download_training_metrics("job-1", tmp_path)

        value, attributes = self._emitted(c, "download_training_metrics")
        assert (value["success"], value["chunk_count"], value["cancelled"]) == (True, 3, False)
        assert attributes == {"client.surface": "sdk", "job_id": "job-1"}

    def test_download_success_telemetry_is_on_by_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=3)
        c._metric_emitter = MagicMock()
        c.download_stdout_logs("job-1", tmp_path)
        value, attributes = self._emitted(c)
        size = (tmp_path / self.SUB / "stdout.log").stat().st_size
        varying = ("duration_ms", "list_ms", "fetch_ms", "first_commit_ms", "attempt_count", "request_count")
        assert {key: value[key] for key in value if key not in varying} == {
            "artifact": "stdout",
            "max_workers": 16,
            "resume": False,
            "resume_mode": "fresh",
            "sub_job_count": 1,
            "chunk_count": 3,
            "chunks_skipped": 0,
            "bytes_written": size,
            "artifact_retry_count": 0,
            "transfer_timeout_count": 0,
            "cancelled": False,
            "retry_count": 0,
            "success": True,
        }
        assert value["list_ms"] >= 0 and value["fetch_ms"] >= 0
        assert value["duration_ms"] >= value["fetch_ms"]
        # Its one file was whole before the call ended; with no on_preview there is no preview_ms.
        assert 0 <= value["first_commit_ms"] <= value["duration_ms"] and "preview_ms" not in value
        # Surface and identity only: per-download detail stays in the value.
        assert attributes == {"client.surface": "sdk", "job_id": "job-1"}

        # Other operations keep successes opt-in.
        c._metric_emitter.reset_mock()
        c._session.post.return_value = _make_response({"request_id": "r1"})
        c.step("job-1")
        c._metric_emitter.emit.assert_not_called()

        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "0")
        c.download_stdout_logs("job-1", tmp_path / "quiet")
        c._metric_emitter.emit.assert_not_called()

    def test_download_telemetry_for_resume_append_and_skipped_chunks(self, tmp_path, monkeypatch):
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=4)
        chunks = paths[self.SUB]
        listing = connection.list_rows[self.RUN_URI + "_stdout/"]
        connection.list_rows[self.RUN_URI + "_stdout/"] = [
            row for row in listing if row[0].removeprefix("/versions/RUN_ABC/") in chunks[:2]
        ]
        c.download_stdout_logs("job-1", tmp_path)
        connection.list_rows[self.RUN_URI + "_stdout/"] = listing
        connection.content_by_uri[self.RUN_URI + chunks[3]] = b"not gzip"
        c._metric_emitter = MagicMock()
        before = (tmp_path / self.SUB / "stdout.log").stat().st_size
        result = c.download_stdout_logs("job-1", tmp_path, resume=True, max_workers=2)
        value, _attributes = self._emitted(c)
        after = (tmp_path / self.SUB / "stdout.log").stat().st_size
        assert result[0]["skipped_chunks"] == [self.RUN_URI + chunks[3]]
        assert value["resume"] is True and value["resume_mode"] == "append"
        assert value["chunk_count"] == 2 and value["chunks_skipped"] == 1
        assert value["bytes_written"] == after - before > 0
        assert value["max_workers"] == 2 and value["success"] is True

    def test_download_failure_telemetry_counts_retries_and_progress(self, tmp_path, monkeypatch):
        errors = importlib.import_module("snowflake.connector.errors")
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=4)
        chunks = paths[self.SUB]
        flaky = [errors.OperationalError(msg="reset")]

        def on_get(source):
            if source == self.RUN_URI + chunks[1] and flaky:
                raise flaky.pop()
            if source == self.RUN_URI + chunks[2]:
                raise errors.ProgrammingError(msg="denied", errno=2003)

        connection.on_get = on_get
        c._metric_emitter = MagicMock()
        with pytest.raises(errors.ProgrammingError):
            c.download_stdout_logs("job-1", tmp_path, max_workers=1)
        self._settle()
        value, attributes = self._emitted(c)
        assert value["success"] is False and value["cancelled"] is False
        assert "denied" in value["error_message"]
        assert value["chunk_count"] == 2  # committed before the failure
        assert value["artifact_retry_count"] == 1
        assert value["transfer_timeout_count"] == 0
        assert attributes["error.type"] == "ProgrammingError"

    def test_download_telemetry_counts_transfer_timeouts(self, tmp_path, monkeypatch):
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=2)
        hung = self.RUN_URI + paths[self.SUB][0]
        release = threading.Event()
        first = [True]

        def on_get(source):
            if source == hung and first:
                first.clear()
                release.wait(10)

        connection.on_get = on_get
        c._metric_emitter = MagicMock()
        try:
            c.download_stdout_logs("job-1", tmp_path, transfer_timeout_seconds=0.3)
        finally:
            release.set()
        value, _attributes = self._emitted(c)
        assert value["transfer_timeout_count"] == 1
        assert value["artifact_retry_count"] == 1
        assert value["success"] is True

    def test_interrupted_download_reports_cancelled(self, tmp_path, monkeypatch):
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=4)
        target = self.RUN_URI + paths[self.SUB][2]

        def on_get(source):
            if source == target:
                raise KeyboardInterrupt

        connection.on_get = on_get
        c._metric_emitter = MagicMock()
        with pytest.raises(KeyboardInterrupt):
            c.download_stdout_logs("job-1", tmp_path, max_workers=1)
        self._settle()
        value, _attributes = self._emitted(c)
        assert value["cancelled"] is True and value["success"] is False
        assert value["chunk_count"] == 2
        assert "error_message" not in value

    def test_fetch_execution_logs_success_telemetry(self, monkeypatch):
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        c = _make_client()
        run_uri = self.RUN_URI
        path = "_logs/job-1:training:0/execution.jsonl"
        connection = TestExecutionLogDownload.FakeArtifactConnection(
            {run_uri + "_logs/": [(f"/versions/RUN_ABC/{path}",)], run_uri + "checkpoints/_logs/": []},
            {run_uri + path: b'{"a":1}\n'},
        )
        monkeypatch.setattr(c, "_experiment_run_uri", lambda _: (run_uri, "RUN_ABC"))
        monkeypatch.setattr(c, "_open_experiment_artifact_connection", lambda: connection)
        c._metric_emitter = MagicMock()
        c.fetch_execution_logs("job-1")
        value, _attributes = self._emitted(c, "fetch_execution_logs")
        assert value["artifact"] == "execution_logs"
        assert value["file_count"] == 1 and value["bytes_read"] == 8
        assert value["list_ms"] >= 0 and value["fetch_ms"] >= 0
        assert value["success"] is True and value["cancelled"] is False

        # A file that does not decode is not counted as read.
        connection.content_by_uri[run_uri + path] = b"\xff\xfe not utf-8"
        c._metric_emitter.reset_mock()
        with pytest.raises(UnicodeDecodeError):
            c.fetch_execution_logs("job-1")
        value, _attributes = self._emitted(c, "fetch_execution_logs")
        assert value["success"] is False
        assert value["file_count"] == 0 and value["bytes_read"] == 0
        assert value["cancelled"] is False

    def test_download_telemetry_sends_resume_as_a_bool(self, tmp_path, monkeypatch):
        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=1)
        c._metric_emitter = MagicMock()
        c.download_stdout_logs("job-1", tmp_path, resume=1)
        value, _attributes = self._emitted(c)
        assert value["resume"] is True

    @pytest.mark.parametrize("setting", ["no", "off", "false", "2"])
    def test_any_explicit_non_truthy_setting_turns_download_successes_off(
        self, tmp_path, monkeypatch, setting
    ):
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, setting)
        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=1)
        c._metric_emitter = MagicMock()
        c.download_stdout_logs("job-1", tmp_path)
        c._metric_emitter.emit.assert_not_called()

    def test_download_failures_are_still_sent_with_success_telemetry_off(
        self, tmp_path, monkeypatch
    ):
        errors = importlib.import_module("snowflake.connector.errors")
        monkeypatch.setenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, "0")
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=2)

        def on_get(source):
            raise errors.ProgrammingError(msg="denied", errno=2003)

        connection.on_get = on_get
        c._metric_emitter = MagicMock()
        with pytest.raises(errors.ProgrammingError):
            c.download_stdout_logs("job-1", tmp_path)
        self._settle()
        value, _attributes = self._emitted(c)
        assert value["success"] is False and value["artifact"] == "stdout"

    def test_interrupting_another_operation_sends_nothing_by_default(self):
        c = _make_client()
        c._metric_emitter = MagicMock()
        c._session.post.side_effect = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            c.step("job-1")
        c._metric_emitter.emit.assert_not_called()

    def test_rebuilding_an_existing_file_reports_rebuild(self, tmp_path, monkeypatch):
        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=1)
        c.download_stdout_logs("job-1", tmp_path)
        c._metric_emitter = MagicMock()
        c.download_stdout_logs("job-1", tmp_path)
        value, _attributes = self._emitted(c)
        assert value["resume_mode"] == "rebuild"

    def test_download_cancelled_is_public(self):
        import cortex_training

        assert cortex_training.DownloadCancelled is nc.DownloadCancelled
        assert issubclass(nc.DownloadCancelled, Exception)

    def test_failure_marks_whether_progress_was_saved(self, tmp_path, monkeypatch):
        errors = importlib.import_module("snowflake.connector.errors")
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=3)
        denied = {self.RUN_URI + paths[self.SUB][0]}

        def on_get(source):
            if source in denied:
                raise errors.ProgrammingError(msg="denied", errno=2003)

        connection.on_get = on_get
        with pytest.raises(errors.ProgrammingError) as first:
            c.download_stdout_logs("job-1", tmp_path, max_workers=1)
        assert first.value.download_progress_saved is False

        denied.clear()
        denied.add(self.RUN_URI + paths[self.SUB][2])
        with pytest.raises(errors.ProgrammingError) as second:
            c.download_stdout_logs("job-1", tmp_path, max_workers=1)
        assert second.value.download_progress_saved is True

    def test_on_chunk_done_equals_total_only_after_commit(self, tmp_path, monkeypatch):
        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=3)
        destination = tmp_path / self.SUB / "stdout.log"
        seen = []

        def progress(sub_job_id, done, total, nbytes):
            seen.append((done, destination.exists()))
            if done == total:
                assert destination.read_bytes() == self._expected(self.SUB, range(3))
                assert destination.stat().st_size == nbytes

        c.download_stdout_logs("job-1", tmp_path, on_chunk=progress)
        assert seen == [(0, False), (1, False), (2, False), (3, True)]
        seen.clear()
        c.download_stdout_logs("job-1", tmp_path, on_chunk=progress, resume=True)
        assert seen == [(3, True)]

    @pytest.mark.parametrize("crash", ["before_data_replace", "between_replaces"])
    def test_promote_crash_windows(self, tmp_path, monkeypatch, crash):
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=3)
        directory = tmp_path / self.SUB
        destination = directory / "stdout.log"
        manifest = directory / ".stdout.log.manifest.json"
        real_replace = os.replace
        crashed = []

        def replace(src, dst):
            target = destination if crash == "before_data_replace" else manifest
            if Path(dst) == target and not crashed:
                crashed.append(dst)
                raise SystemExit("crash")
            return real_replace(src, dst)

        monkeypatch.setattr(nc.os, "replace", replace)
        with pytest.raises(SystemExit):
            c.download_stdout_logs("job-1", tmp_path)
        self._settle()
        monkeypatch.setattr(nc.os, "replace", real_replace)

        gets_before = len(self._gets(connection))
        c.download_stdout_logs("job-1", tmp_path, resume=True)
        refetched = len(self._gets(connection)) - gets_before
        assert destination.read_bytes() == self._expected(self.SUB, range(3))
        assert json.loads(manifest.read_text())["names"] == paths[self.SUB]
        assert not list(directory.glob(".stdout.log.partial*"))
        if crash == "before_data_replace":
            # The complete partial file is promoted without any GET.
            assert refetched == 0
        else:
            # The new file has no manifest yet, so it is rebuilt (wasteful, safe).
            assert refetched == 3

    def test_reconnect_while_another_get_runs_on_the_old_connection(
        self, tmp_path, monkeypatch
    ):
        errors = importlib.import_module("snowflake.connector.errors")
        c, first, paths, _sleeps, _opened = self._setup(monkeypatch, count=4)
        second = TestExecutionLogDownload.FakeArtifactConnection(
            first.list_rows, first.content_by_uri
        )
        connections = iter([first, second])
        monkeypatch.setattr(c, "_open_experiment_artifact_connection", lambda: next(connections))
        chunks = paths[self.SUB]
        slow = self.RUN_URI + chunks[0]
        expiring = self.RUN_URI + chunks[1]
        reopened = threading.Event()
        old_still_open = []

        def on_first_get(source):
            if source == slow:
                assert reopened.wait(5)
                old_still_open.append(not first.closed)
            if source == expiring:
                raise errors.ProgrammingError(msg="session expired", errno=390112)

        def on_second_get(source):
            if source == expiring:
                reopened.set()

        first.on_get = on_first_get
        second.on_get = on_second_get
        c.download_stdout_logs("job-1", tmp_path, max_workers=2)

        assert old_still_open == [True]
        assert any(expiring in cmd for cmd in self._gets(second))
        assert (tmp_path / self.SUB / "stdout.log").read_bytes() == self._expected(
            self.SUB, range(4)
        )
        assert first.closed and second.closed

    def test_stress_many_chunks_with_random_latency(self, tmp_path, monkeypatch):
        import random as random_module

        subs = [self.SUB, "job-1:training:1", "job-1:sampling:0"]
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, subs=subs, count=170)
        rng = random_module.Random(1234)
        delays = {
            self.RUN_URI + path: rng.random() * 0.004 for sub in subs for path in paths[sub]
        }
        connection.on_get = lambda source: time.sleep(delays[source])

        result = c.download_stdout_logs("job-1", tmp_path, max_workers=32)

        assert len(self._gets(connection)) == 510
        for sub in subs:
            assert (tmp_path / sub / "stdout.log").read_bytes() == self._expected(sub, range(170))
        assert [entry["chunk_count"] for entry in result] == [170, 170, 170]

    @pytest.mark.parametrize(
        "skipped", [["not-listed.gz"], [{"x": 1}], "names"]
    )
    def test_manifest_with_invalid_skipped_list_is_rejected(self, tmp_path, skipped):
        path = tmp_path / ".x.manifest.json"
        path.write_text(json.dumps({"bytes": 1, "names": ["a.gz"], "skipped": skipped, "version": 1}))
        assert nc._artifact_download.read_manifest(path) is None

    # Newest chunk time in _setup(count=3): 2026-09-04 12:00:02 UTC.
    NEWEST = datetime(2026, 9, 4, 12, 0, 2, tzinfo=timezone.utc).timestamp()

    def _stale_setup(
        self,
        monkeypatch,
        *,
        status="RUNNING",
        age_minutes=11,
        server_age_minutes=None,
        job_fields=None,
        **kwargs,
    ):
        """``server_age_minutes`` sets the server's clock; None means no Date header."""
        c, connection, paths, sleeps, opened = self._setup(monkeypatch, **kwargs)
        status_calls = []

        def job_for_upload_warnings(job_id):
            status_calls.append(job_id)
            if isinstance(status, Exception):
                raise status
            server_now = (
                None if server_age_minutes is None else self.NEWEST + server_age_minutes * 60
            )
            return {"job_id": job_id, "status": status, **(job_fields or {})}, server_now

        monkeypatch.setattr(c, "_job_for_upload_warnings", job_for_upload_warnings)
        c._wall_clock = lambda: self.NEWEST + age_minutes * 60
        return c, connection, paths, status_calls

    def test_stale_stdout_warns_while_running(self, tmp_path, monkeypatch, caplog):
        c, _connection, _paths, status_calls = self._stale_setup(monkeypatch, count=3)
        with caplog.at_level(logging.WARNING):
            result = c.download_stdout_logs("job-1", tmp_path)
        [warning] = result[0]["warnings"]
        assert warning.startswith(
            f"newest saved stdout output for {self.SUB} is from 2026-09-04 12:00:02 UTC "
            "(11 min ago) while the job is still running; uploads may be stalled. "
            "Contact support if this persists."
        )
        assert warning in caplog.text
        assert status_calls == ["job-1"]
        assert (tmp_path / self.SUB / "stdout.log").read_bytes() == self._expected(self.SUB, range(3))

    def test_old_training_metrics_of_a_running_job_are_not_called_stalled(
        self, tmp_path, monkeypatch, caplog
    ):
        """Training rows are written only on /fwd-bwd and /step, so a running job whose
        newest training chunk is old is idle, not stalled: no warning, unlike GPU metrics."""
        c, _connection, _paths, _status_calls = self._stale_setup(
            monkeypatch, count=3, artifact_name="metrics", chunk_name="training"
        )
        with caplog.at_level(logging.WARNING):
            result = c.download_training_metrics("job-1", tmp_path)
        assert [entry["sub_job_id"] for entry in result] == [self.SUB]
        assert not result[0].get("warnings")
        assert "stalled" not in caplog.text
        assert (tmp_path / self.SUB / "training.jsonl").read_bytes() == self._expected(self.SUB, range(3))

    def test_old_gpu_metrics_of_a_running_job_are_still_called_stalled(self, tmp_path, monkeypatch):
        c, _connection, _paths, _status_calls = self._stale_setup(
            monkeypatch, count=3, artifact_name="metrics"
        )
        result = c.download_metrics("job-1", tmp_path)
        [warning] = result[0]["warnings"]
        assert "uploads may be stalled" in warning

    @pytest.mark.parametrize(
        ("status", "age_minutes", "stale_after"),
        [
            ("DONE", 60, 600),
            ("failed", 60, 600),
            ("JOB_STATE_CANCELLED", 60, 600),
            ("RUNNING", 9, 600),
            ("RUNNING", 60, None),
            ("RUNNING", 60, 7200),
            (RuntimeError("status unavailable"), 60, 600),
        ],
    )
    def test_no_stale_warning(self, tmp_path, monkeypatch, status, age_minutes, stale_after):
        c, _connection, _paths, _calls = self._stale_setup(
            monkeypatch, status=status, age_minutes=age_minutes, count=3
        )
        result = c.download_stdout_logs("job-1", tmp_path, stale_after_seconds=stale_after)
        assert "warnings" not in result[0]
        assert result[0]["chunk_count"] == 3

    def test_fresh_stdout_of_a_running_job_looks_up_status_once_and_stays_quiet(self, tmp_path, monkeypatch):
        # Recent chunks mean the job may have only just ended, so its status is
        # read once (in the background); a running job gets no warning.
        c, _connection, _paths, status_calls = self._stale_setup(
            monkeypatch, age_minutes=1, count=3
        )
        assert "warnings" not in c.download_stdout_logs("job-1", tmp_path)[0]
        assert status_calls == ["job-1"]

    @pytest.mark.parametrize("age_minutes", [1, 30])
    def test_turning_the_checks_off_skips_the_status_lookup(self, tmp_path, monkeypatch, age_minutes):
        c, _connection, _paths, status_calls = self._stale_setup(monkeypatch, age_minutes=age_minutes, count=3)
        c.download_stdout_logs("job-1", tmp_path, stale_after_seconds=None)
        assert status_calls == []

    @pytest.mark.parametrize("artifact_name, label", [("stdout", "stdout output"), ("metrics", "GPU metrics")])
    def test_a_just_ended_job_warns_that_its_last_chunks_may_still_be_uploading(
        self, tmp_path, monkeypatch, artifact_name, label
    ):
        ended = datetime.fromtimestamp(self.NEWEST + 30, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        c, _connection, _paths, status_calls = self._stale_setup(
            monkeypatch,
            status="CANCELLED",
            age_minutes=1,
            count=3,
            artifact_name=artifact_name,
            job_fields={"state_changed_at": ended},
        )
        method = c.download_stdout_logs if artifact_name == "stdout" else c.download_metrics
        [warning] = method("job-1", tmp_path)[0]["warnings"]
        assert warning == (
            f"the job ended 30s ago, so the last {label} for {self.SUB} may still be uploading; "
            "download again with resume in a minute or two to pick it up."
        )
        assert status_calls == ["job-1"]

    def test_a_job_that_ended_a_while_ago_has_no_upload_warning(self, tmp_path, monkeypatch):
        ended = datetime.fromtimestamp(self.NEWEST - 3600, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        c, _connection, _paths, _calls = self._stale_setup(
            monkeypatch, status="DONE", age_minutes=1, count=3, job_fields={"state_changed_at": ended}
        )
        assert "warnings" not in c.download_stdout_logs("job-1", tmp_path)[0]

    def test_stale_metrics_warning_has_no_stdout_hint(self, tmp_path, monkeypatch):
        c, _connection, _paths, _calls = self._stale_setup(
            monkeypatch, count=3, artifact_name="metrics"
        )
        [warning] = c.download_metrics("job-1", tmp_path)[0]["warnings"]
        assert warning.startswith(f"newest saved GPU metrics for {self.SUB} is from")
        assert "prints nothing" not in warning

    def _metrics_missing_setup(self, monkeypatch, status="RUNNING", **kwargs):
        kwargs.setdefault("age_minutes", 3)
        c, connection, paths, status_calls = self._stale_setup(
            monkeypatch,
            status=status,
            count=3,
            artifact_name="metrics",
            **kwargs,
        )
        # A second sub-job has saved stdout but no metrics.
        stdout_only = "job-1:sampling:0"
        connection.list_rows[self.RUN_URI + "_stdout/"] = [
            (f"/versions/RUN_ABC/_stdout/{stdout_only}/console.20260904-120001.a.gz",)
        ]
        connection.list_rows[self.RUN_URI + "checkpoints/_stdout/"] = []
        return c, connection, stdout_only

    def test_running_job_with_stdout_but_no_metrics_warns(self, tmp_path, monkeypatch):
        c, connection, stdout_only = self._metrics_missing_setup(monkeypatch)
        result = c.download_metrics("job-1", tmp_path)
        assert [entry["sub_job_id"] for entry in result] == [stdout_only, self.SUB]
        assert result[0] == {
            "sub_job_id": stdout_only,
            "filename": "gpu.jsonl",
            "saved_path": None,
            "chunk_count": 0,
            "first_artifact_uri": None,
            "last_artifact_uri": None,
            "warnings": [
                f"no GPU metrics have been saved for {stdout_only} while the job is still "
                "running, although its stdout output has been; metric uploads may be "
                "stalled. Contact support if this persists."
            ],
        }
        assert "warnings" not in result[1]
        assert not (tmp_path / stdout_only).exists()
        assert not any(stdout_only in cmd for cmd in self._gets(connection))

        # Selecting that sub-job is not an error while the warning applies.
        only = c.download_metrics("job-1", tmp_path, sub_job_ids=[stdout_only])
        assert [entry["sub_job_id"] for entry in only] == [stdout_only]

    @pytest.mark.parametrize("status", ["DONE", "TERMINATED", "CANCELLED", "FAILED"])
    def test_ended_job_with_stdout_but_no_metrics_explains_why(self, tmp_path, monkeypatch, status):
        c, connection, stdout_only = self._metrics_missing_setup(monkeypatch, status=status)
        result = c.download_metrics("job-1", tmp_path)
        assert [entry["sub_job_id"] for entry in result] == [stdout_only, self.SUB]
        assert result[0]["chunk_count"] == 0 and result[0]["saved_path"] is None
        (message,) = result[0]["warnings"]
        assert message.startswith(f"no GPU metrics were saved for {stdout_only}, although its stdout")
        assert "images from before GPU metrics were uploaded" in message
        assert "warnings" not in result[1]
        assert any(cmd == f"LIST '{self.RUN_URI}_stdout/'" for cmd in connection.commands)
        # Naming that sub-job returns the explanation instead of NoChunksError.
        (only,) = c.download_metrics("job-1", tmp_path, sub_job_ids=[stdout_only])
        assert only["sub_job_id"] == stdout_only and only["warnings"] == [message]
        self._settle()

    def test_ended_job_without_stdout_either_stays_an_unknown_sub_job(self, tmp_path, monkeypatch):
        c, _connection, _stdout_only = self._metrics_missing_setup(monkeypatch, status="DONE")
        with pytest.raises(nc.NoChunksError):
            c.download_metrics("job-1", tmp_path, sub_job_ids=["job-1:other:0"])
        self._settle()

    def test_pending_job_with_stdout_but_no_metrics_is_quiet(self, tmp_path, monkeypatch):
        c, _connection, _stdout_only = self._metrics_missing_setup(monkeypatch, status="PENDING")
        result = c.download_metrics("job-1", tmp_path)
        assert [entry["sub_job_id"] for entry in result] == [self.SUB]
        self._settle()

    def _iso(self, minutes_after_newest):
        stamp = datetime.fromtimestamp(self.NEWEST + minutes_after_newest * 60, tz=timezone.utc)
        return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    @pytest.mark.parametrize("sub_status", ["DONE", "failed", "JOB_STATE_PENDING"])
    def test_no_stall_warning_for_a_sub_job_that_is_not_running(
        self, tmp_path, monkeypatch, sub_status
    ):
        fields = {"sub_jobs": [{"sub_job_id": self.SUB, "status": sub_status}]}
        c, _connection, _paths, _calls = self._stale_setup(
            monkeypatch, count=3, job_fields=fields
        )
        assert "warnings" not in c.download_stdout_logs("job-1", tmp_path)[0]

    def test_stall_warning_for_a_running_sub_job(self, tmp_path, monkeypatch):
        fields = {
            "state_changed_at": self._iso(-60),
            "sub_jobs": [
                {"sub_job_id": self.SUB, "status": "RUNNING"},
                {"sub_job_id": "job-1:sampling:0", "status": "DONE"},
            ],
        }
        c, _connection, _paths, _calls = self._stale_setup(
            monkeypatch, count=3, job_fields=fields
        )
        assert "11 min ago" in c.download_stdout_logs("job-1", tmp_path)[0]["warnings"][0]

    def test_no_stall_warning_soon_after_the_job_started_running(self, tmp_path, monkeypatch):
        # The newest chunk predates a restart five minutes ago.
        c, _connection, _paths, _calls = self._stale_setup(
            monkeypatch, count=3, job_fields={"state_changed_at": self._iso(6)}
        )
        assert "warnings" not in c.download_stdout_logs("job-1", tmp_path)[0]

    @pytest.mark.parametrize(
        ("local", "server", "warned"),
        [(11, 2, None), (6, 12, "12 min ago"), (30, 11, "11 min ago"), (11, None, "11 min ago")],
    )
    def test_stall_age_uses_the_server_clock(
        self, tmp_path, monkeypatch, local, server, warned
    ):
        c, _connection, _paths, _calls = self._stale_setup(
            monkeypatch, count=3, age_minutes=local, server_age_minutes=server
        )
        warnings = c.download_stdout_logs("job-1", tmp_path)[0].get("warnings")
        if warned is None:
            assert warnings is None
        else:
            assert warned in warnings[0]

    def test_no_missing_metrics_warning_right_after_the_sub_job_starts(
        self, tmp_path, monkeypatch
    ):
        # Its first stdout chunk is from 12:00:01; one minute later is too soon.
        c, _connection, stdout_only = self._metrics_missing_setup(
            monkeypatch, age_minutes=1
        )
        result = c.download_metrics("job-1", tmp_path)
        assert [entry["sub_job_id"] for entry in result] == [self.SUB]

    def test_no_missing_metrics_warning_for_a_finished_sub_job(self, tmp_path, monkeypatch):
        c, _connection, stdout_only = self._metrics_missing_setup(
            monkeypatch,
            job_fields={"sub_jobs": [{"sub_job_id": "job-1:sampling:0", "status": "DONE"}]},
        )
        result = c.download_metrics("job-1", tmp_path)
        assert [entry["sub_job_id"] for entry in result] == [self.SUB]

    def test_job_for_upload_warnings_reads_server_date(self):
        c = _make_client()
        response = _make_response({"status": "RUNNING"})
        response.headers = {"Date": "Thu, 15 Oct 2026 01:52:53 GMT"}
        c._session.get.return_value = response
        job, server_now = c._job_for_upload_warnings("job-1")
        assert job == {"status": "RUNNING"}
        assert server_now == datetime(2026, 10, 15, 1, 52, 53, tzinfo=timezone.utc).timestamp()
        response.headers = {}
        assert c._job_for_upload_warnings("job-1")[1] is None

    def test_status_lookup_failure_skips_warnings_but_finishes(self, tmp_path, monkeypatch):
        c, _connection, _paths, calls = self._stale_setup(
            monkeypatch, status=RuntimeError("boom"), count=3, artifact_name="metrics"
        )
        result = c.download_metrics("job-1", tmp_path)
        assert calls == ["job-1"]
        assert "warnings" not in result[0]
        assert result[0]["chunk_count"] == 3

    def test_named_sub_job_without_chunks_raises_no_chunks_error(self, tmp_path, monkeypatch):
        import cortex_training

        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=1)
        with pytest.raises(cortex_training.NoChunksError) as raised:
            c.download_stdout_logs("job-1", tmp_path, sub_job_ids=["job-1:sampling:0", self.SUB])
        assert isinstance(raised.value, ValueError)
        assert raised.value.sub_job_ids == ["job-1:sampling:0"]
        assert raised.value.available == [self.SUB]

    def test_spool_is_removed_after_cancel_once_abandoned_gets_finish(
        self, tmp_path, monkeypatch
    ):
        spool_parent = tmp_path / "tmp"
        spool_parent.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(spool_parent))
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=6)
        chunks = paths[self.SUB]
        release = threading.Event()
        cancel_at = self.RUN_URI + chunks[1]

        def on_get(source):
            if source == cancel_at:
                raise KeyboardInterrupt
            if source == self.RUN_URI + chunks[3]:
                # Still on the wire when the download gives up on it.
                assert release.wait(5)

        connection.on_get = on_get
        with pytest.raises(KeyboardInterrupt):
            c.download_stdout_logs("job-1", tmp_path / "out", max_workers=4)
        assert list(spool_parent.iterdir()), "spool removed while a GET was still running"
        release.set()
        self._settle()
        deadline = time.monotonic() + 5
        while list(spool_parent.iterdir()):
            assert time.monotonic() < deadline, list(spool_parent.rglob("*"))
            time.sleep(0.01)

    def test_spool_waits_for_the_fetch_the_writer_was_blocked_on(self, tmp_path, monkeypatch):
        import signal

        # A process started in the background by a non-interactive shell
        # inherits SIGINT as ignored, and Python then never raises
        # KeyboardInterrupt for it. Install the default handler for this test.
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            self._spool_waits_for_the_blocked_fetch(tmp_path, monkeypatch, signal)
        finally:
            signal.signal(signal.SIGINT, previous)

    def _spool_waits_for_the_blocked_fetch(self, tmp_path, monkeypatch, signal):
        spool_parent = tmp_path / "tmp"
        spool_parent.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(spool_parent))
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=4)
        head = self.RUN_URI + paths[self.SUB][0]
        release = threading.Event()
        get_finished = threading.Event()
        timed_out = []

        def on_get(source):
            if source == head:
                try:
                    # Ctrl-C arrives while the writer waits on this very fetch.
                    signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
                    # Bounded generously: under load the main thread can take a
                    # while to unwind the interrupt before it releases us.
                    if not release.wait(30):
                        timed_out.append(source)
                finally:
                    get_finished.set()

        connection.on_get = on_get
        with pytest.raises(KeyboardInterrupt):
            c.download_stdout_logs("job-1", tmp_path / "out", max_workers=1)
        # The interrupted download has returned, but the head GET is still
        # inside on_get: give a premature cleanup time to happen, then check.
        assert not get_finished.wait(0.3)
        assert list(spool_parent.iterdir()), "spool removed while the head GET was running"
        release.set()
        assert get_finished.wait(30)
        self._settle()
        deadline = time.monotonic() + 30
        while list(spool_parent.iterdir()):
            assert time.monotonic() < deadline, list(spool_parent.rglob("*"))
            time.sleep(0.01)
        assert timed_out == []

    def test_sweep_reports_only_directories_it_removed(self, tmp_path, monkeypatch, caplog):
        spool_parent = tmp_path / "tmp"
        spool_parent.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(spool_parent))
        stuck = spool_parent / "cortex-training-stdout-artifacts-stuck"
        stuck.mkdir()
        stamp = time.time() - 7200
        os.utime(stuck, (stamp, stamp))
        monkeypatch.setattr(nc._artifact_download.shutil, "rmtree", lambda *a, **k: None)
        with caplog.at_level(logging.DEBUG):
            nc._artifact_download.Spool.sweep("cortex-training-stdout-artifacts-", 3600)
        assert "could not remove leftover download directory" in caplog.text
        assert "removed leftover download directory" not in caplog.text

    def test_old_spool_directories_are_swept(self, tmp_path, monkeypatch):
        spool_parent = tmp_path / "tmp"
        spool_parent.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(spool_parent))
        old = spool_parent / "cortex-training-stdout-artifacts-old"
        recent = spool_parent / "cortex-training-stdout-artifacts-recent"
        other = spool_parent / "cortex-training-metrics-artifacts-old"
        for directory in (old, recent, other):
            (directory / "chunk-x").mkdir(parents=True)
        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=1)
        stamp = time.time() - 2 * c._TEMP_ORPHAN_MIN_AGE_SECONDS
        for directory in (old, other):
            os.utime(directory, (stamp, stamp))
        c.download_stdout_logs("job-1", tmp_path / "out")
        assert sorted(path.name for path in spool_parent.iterdir()) == [other.name, recent.name]

    def test_hung_get_is_abandoned_and_retried(self, tmp_path, monkeypatch):
        c, connection, paths, sleeps, _opened = self._setup(monkeypatch, count=4)
        hung = self.RUN_URI + paths[self.SUB][1]
        release = threading.Event()
        first = [True]

        def on_get(source):
            if source == hung and first:
                first.clear()
                assert release.wait(10)

        connection.on_get = on_get
        started = time.monotonic()
        result = c.download_stdout_logs("job-1", tmp_path, transfer_timeout_seconds=0.5)
        assert time.monotonic() - started < 5
        destination = tmp_path / self.SUB / "stdout.log"
        expected = self._expected(self.SUB, range(4))
        assert destination.read_bytes() == expected
        assert len(self._gets(connection, hung)) == 2
        assert len(sleeps) == 1
        assert result[0]["chunk_count"] == 4

        # The abandoned attempt finishing later cannot touch the output.
        release.set()
        deadline = time.monotonic() + 5
        while any(t.name == "cortex-training-transfer" for t in threading.enumerate()):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert destination.read_bytes() == expected
        assert sorted(p.name for p in (tmp_path / self.SUB).iterdir()) == [
            ".stdout.log.manifest.json",
            "stdout.log",
        ]

    def test_hung_list_is_abandoned_and_retried(self, tmp_path, monkeypatch):
        c, connection, _paths, sleeps, _opened = self._setup(monkeypatch, count=2)
        cursor_cls = connection.Cursor
        release = threading.Event()
        first = [True]

        class HangingCursor(cursor_cls):
            def execute(self, statement):
                if statement.startswith("LIST ") and first:
                    first.clear()
                    release.wait(10)
                return super().execute(statement)

        monkeypatch.setattr(connection, "cursor", lambda: HangingCursor(connection))
        try:
            c.download_stdout_logs("job-1", tmp_path, transfer_timeout_seconds=0.5)
        finally:
            release.set()
        assert len(sleeps) == 1
        assert (tmp_path / self.SUB / "stdout.log").read_bytes() == self._expected(
            self.SUB, range(2)
        )

    def test_run_bounded_gives_a_cancelled_attempt_a_short_grace(self, monkeypatch):
        monkeypatch.setattr(nc._artifact_download, "CANCEL_GRACE_SECONDS", 0.3)
        cancel = threading.Event()
        cancel.set()
        release = threading.Event()
        started = time.monotonic()
        with pytest.raises(nc.DownloadCancelled):
            nc._artifact_download.run_bounded(
                lambda: release.wait(10), timeout=None, cancel=cancel, description="GET x"
            )
        assert 0.2 < time.monotonic() - started < 3
        release.set()
        # A quick attempt still completes normally after cancel.
        assert nc._artifact_download.run_bounded(
            lambda: 7, timeout=1.0, cancel=cancel, description="GET y"
        ) == 7

    def test_run_bounded_reraises_the_attempt_error(self):
        def fail():
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            nc._artifact_download.run_bounded(fail, timeout=1.0, cancel=None, description="x")
        assert nc._artifact_download.classify_artifact_error(
            nc._artifact_download.TransferTimeout("slow")
        ) == "retry"

    def test_transfer_timeout_message_suggests_the_knobs(self):
        with pytest.raises(nc._artifact_download.TransferTimeout) as raised:
            nc._artifact_download.run_bounded(
                lambda: time.sleep(1), timeout=0.05, cancel=None, description="GET x"
            )
        message = str(raised.value)
        assert message.startswith("GET x did not finish within 0.05s.")
        assert "--transfer-timeout" in message and "--parallel" in message

    @pytest.mark.parametrize("value", [0, -1, True, "300"])
    def test_transfer_timeout_must_be_positive(self, tmp_path, monkeypatch, value):
        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=1)
        with pytest.raises(ValueError, match="transfer_timeout_seconds"):
            c.download_stdout_logs("job-1", tmp_path, transfer_timeout_seconds=value)

    def test_resume_with_nothing_new_leaves_a_read_only_file_alone(self, tmp_path, monkeypatch):
        if os.geteuid() == 0:
            pytest.skip("root ignores the read-only destination")
        c, connection, paths, _sleeps, _opened = self._setup(monkeypatch, count=3)
        c.download_stdout_logs("job-1", tmp_path)
        destination = tmp_path / self.SUB / "stdout.log"
        destination.chmod(0o444)
        gets_before = len(self._gets(connection))
        calls, counts = [], []
        try:
            result = c.download_stdout_logs(
                "job-1",
                tmp_path,
                resume=True,
                progress=calls.append,
                on_chunk=lambda *args: counts.append(args),
            )
        finally:
            destination.chmod(0o644)
        assert len(self._gets(connection)) == gets_before
        assert destination.read_bytes() == self._expected(self.SUB, range(3))
        assert result[0]["chunk_count"] == 3
        assert result[0]["last_artifact_uri"] == self.RUN_URI + paths[self.SUB][-1]
        assert calls == [f"stdout: no new chunks for {self.SUB}\n"]
        assert counts == [(self.SUB, 3, 3, destination.stat().st_size)]

    @pytest.mark.parametrize("value", ["job-1:training:0", b"job-1:training:0"])
    def test_sub_job_ids_rejects_a_bare_string(self, tmp_path, monkeypatch, value):
        c, connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=1)
        with pytest.raises(TypeError, match="not a single string"):
            c.download_stdout_logs("job-1", tmp_path, sub_job_ids=value)
        assert connection.commands == []

    def test_stall_check_runs_while_chunks_are_fetched(self, tmp_path, monkeypatch):
        c, connection, paths, _calls = self._stale_setup(monkeypatch, count=3)
        fetching = threading.Event()
        threads = []
        real_lookup = c._job_for_upload_warnings

        def lookup(job_id):
            threads.append(threading.current_thread().name)
            # Only returns quickly if the GETs run at the same time.
            assert fetching.wait(5)
            return real_lookup(job_id)

        monkeypatch.setattr(c, "_job_for_upload_warnings", lookup)
        connection.on_get = lambda _source: fetching.set()
        result = c.download_stdout_logs("job-1", tmp_path)
        assert threads == ["cortex-training-stall-check"]
        assert "11 min ago" in result[0]["warnings"][0]

    def test_slow_stall_check_does_not_hold_the_download(self, tmp_path, monkeypatch):
        c, _connection, _paths, _calls = self._stale_setup(monkeypatch, count=2)
        monkeypatch.setattr(c, "_STALL_CHECK_WAIT_SECONDS", 0.2)
        release = threading.Event()
        monkeypatch.setattr(c, "_job_for_upload_warnings", lambda _job: release.wait(10))
        started = time.monotonic()
        try:
            result = c.download_stdout_logs("job-1", tmp_path)
        finally:
            release.set()
        assert time.monotonic() - started < 3
        assert "warnings" not in result[0]
        assert result[0]["chunk_count"] == 2

    def test_interrupt_while_waiting_on_stall_check_keeps_saved_progress(
        self, tmp_path, monkeypatch
    ):
        c, _connection, _paths, _calls = self._stale_setup(monkeypatch, count=2)
        release = threading.Event()
        monkeypatch.setattr(c, "_job_for_upload_warnings", lambda _job: release.wait(10))
        destination = tmp_path / self.SUB / "stdout.log"
        committed = []

        def interrupted_wait(_self, _timeout, _default):
            # Ctrl-C lands while the stall check is still running, after the
            # files were written.
            committed.append(destination.read_bytes())
            raise KeyboardInterrupt

        monkeypatch.setattr(nc._artifact_download.Background, "result", interrupted_wait)
        try:
            with pytest.raises(KeyboardInterrupt) as excinfo:
                c.download_stdout_logs("job-1", tmp_path)
        finally:
            release.set()
        assert excinfo.value.download_progress_saved is True
        assert committed == [self._expected(self.SUB, range(2))]
        assert destination.read_bytes() == self._expected(self.SUB, range(2))

    @pytest.mark.parametrize("interrupt_wait_too", [False, True])
    def test_interrupt_while_closing_the_session_keeps_saved_progress(
        self, tmp_path, monkeypatch, interrupt_wait_too
    ):
        c, _connection, _paths, _calls = self._stale_setup(monkeypatch, count=2)
        if interrupt_wait_too:
            # A second Ctrl-C in the close replaces the first one's exception.
            def interrupted_wait(_self, _timeout, _default):
                raise KeyboardInterrupt

            monkeypatch.setattr(nc._artifact_download.Background, "result", interrupted_wait)
        real_close = nc._artifact_download.SharedConnection.close
        escaped = []

        def interrupted_close(shared, **options):
            real_close(shared, **options)
            escaped.append(KeyboardInterrupt())
            raise escaped[-1]

        monkeypatch.setattr(nc._artifact_download.SharedConnection, "close", interrupted_close)
        with pytest.raises(KeyboardInterrupt) as excinfo:
            c.download_stdout_logs("job-1", tmp_path)
        assert excinfo.value is escaped[0]
        assert excinfo.value.download_progress_saved is True
        destination = tmp_path / self.SUB / "stdout.log"
        assert destination.read_bytes() == self._expected(self.SUB, range(2))

    def test_interrupt_in_pool_shutdown_after_the_last_commit_keeps_saved_progress(
        self, tmp_path, monkeypatch
    ):
        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=2)
        real_shutdown = nc.ThreadPoolExecutor.shutdown
        spools = []

        def live_spools():
            return set(Path(tempfile.gettempdir()).glob("*-stdout-artifacts-*"))

        earlier = live_spools()

        def interrupted_shutdown(pool, wait=True, **kwargs):
            real_shutdown(pool, wait=wait, **kwargs)
            if wait:
                spools.extend(live_spools() - earlier)
                raise KeyboardInterrupt

        monkeypatch.setattr(nc.ThreadPoolExecutor, "shutdown", interrupted_shutdown)
        with pytest.raises(KeyboardInterrupt) as excinfo:
            c.download_stdout_logs("job-1", tmp_path)
        assert excinfo.value.download_progress_saved is True
        destination = tmp_path / self.SUB / "stdout.log"
        assert destination.read_bytes() == self._expected(self.SUB, range(2))
        assert spools and not any(path.exists() for path in spools)

    def test_interrupt_while_closing_after_nothing_was_saved_is_not_marked(
        self, tmp_path, monkeypatch
    ):
        c, _connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=0)
        real_close = nc._artifact_download.SharedConnection.close

        def interrupted_close(shared, **options):
            real_close(shared, **options)
            raise KeyboardInterrupt

        monkeypatch.setattr(nc._artifact_download.SharedConnection, "close", interrupted_close)
        with pytest.raises(KeyboardInterrupt) as excinfo:
            c.download_stdout_logs("job-1", tmp_path)
        assert not getattr(excinfo.value, "download_progress_saved", False)

    @pytest.mark.parametrize("new_chunk", [False, True])
    def test_resume_interrupted_before_writing_keeps_saved_progress(
        self, tmp_path, monkeypatch, new_chunk
    ):
        c, connection, _paths, _sleeps, _opened = self._setup(monkeypatch, count=3)
        listing = self.RUN_URI + "_stdout/"
        rows = connection.list_rows[listing]
        if new_chunk:
            # The newest chunk is uploaded after the first download.
            connection.list_rows[listing] = rows[1:]
        c.download_stdout_logs("job-1", tmp_path)
        connection.list_rows[listing] = rows
        destination = tmp_path / self.SUB / "stdout.log"
        saved = destination.read_bytes()

        def interrupted_progress(_line):
            # Ctrl-C at the first status line, before any new chunk is written.
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt) as excinfo:
            c.download_stdout_logs("job-1", tmp_path, resume=True, progress=interrupted_progress)
        assert excinfo.value.download_progress_saved is True
        assert destination.read_bytes() == saved
        assert saved == self._expected(self.SUB, range(2 if new_chunk else 3))

    @pytest.mark.parametrize("download_fails", [True, False])
    def test_late_stall_check_never_reopens_the_session(
        self, tmp_path, monkeypatch, caplog, download_fails
    ):
        errors = importlib.import_module("snowflake.connector.errors")
        base = TestExecutionLogDownload.FakeArtifactConnection

        class ClosingConnection(base):
            class Cursor(base.Cursor):
                def execute(self, statement):
                    if self.connection.closed:
                        raise errors.DatabaseError(msg="Connection is closed", errno=250002)
                    if statement.startswith("GET ") and download_fails:
                        raise errors.ProgrammingError(msg="invalid stage location", errno=253004)
                    return super().execute(statement)

            def cursor(self):
                return self.Cursor(self)

        c, connection, _paths, _calls = self._stale_setup(
            monkeypatch, count=1, artifact_name="metrics"
        )
        opened = []

        def opener():
            conn = ClosingConnection(connection.list_rows, connection.content_by_uri)
            opened.append(conn)
            return conn

        monkeypatch.setattr(c, "_open_experiment_artifact_connection", opener)
        monkeypatch.setattr(c, "_STALL_CHECK_WAIT_SECONDS", 0.1)
        real_lookup = c._job_for_upload_warnings
        finished = threading.Event()

        def slow_lookup(job_id):
            # Outlives both a failed download and the wait cap.
            time.sleep(0.4)
            try:
                return real_lookup(job_id)
            finally:
                finished.set()

        monkeypatch.setattr(c, "_job_for_upload_warnings", slow_lookup)
        with caplog.at_level(logging.DEBUG):
            if download_fails:
                with pytest.raises(errors.ProgrammingError):
                    c.download_metrics("job-1", tmp_path)
            else:
                c.download_metrics("job-1", tmp_path)
            returned = len(caplog.records)
            assert finished.wait(5)
            deadline = time.monotonic() + 5
            while any(t.name == "cortex-training-stall-check" for t in threading.enumerate()):
                assert time.monotonic() < deadline
                time.sleep(0.01)
        assert len(opened) == 1
        assert all(conn.closed for conn in opened)
        late = [r for r in caplog.records[returned:] if r.levelno >= logging.WARNING]
        assert late == []

    def test_closed_shared_connection_refuses_to_reopen(self):
        opened = []

        def opener():
            opened.append(MagicMock())
            return opened[-1]

        shared = nc._artifact_download.SharedConnection(opener)
        _connection, generation = shared.current()
        shared.close()
        with pytest.raises(nc._artifact_download.SessionClosed):
            shared.current()
        with pytest.raises(nc._artifact_download.SessionClosed):
            shared.reopen(generation)
        assert len(opened) == 1
        assert nc._artifact_download.classify_artifact_error(
            nc._artifact_download.SessionClosed("closed")
        ) is None

    @pytest.mark.parametrize("value", [0, -5, True, "600"])
    def test_stale_after_must_be_positive(self, tmp_path, monkeypatch, value):
        c, _connection, _paths, _calls = self._stale_setup(monkeypatch, count=1)
        with pytest.raises(ValueError, match="stale_after_seconds"):
            c.download_stdout_logs("job-1", tmp_path, stale_after_seconds=value)
        self._settle()

    @pytest.mark.parametrize("missing", ["NonRetryableTlsError", "RevocationCheckError", "TooManyRequests"])
    def test_classification_tolerates_connector_classes_missing_in_older_releases(
        self, monkeypatch, missing
    ):
        errors = importlib.import_module("snowflake.connector.errors")
        operational = errors.OperationalError(msg="network down")
        permission = errors.ProgrammingError(msg="denied", errno=2003)
        monkeypatch.delattr(errors, missing, raising=False)
        classify = nc._artifact_download.classify_artifact_error
        assert classify(operational) == "retry"
        assert classify(permission) is None
        assert classify(errors.ProgrammingError(msg="expired", errno=390112)) == "reconnect"

    def test_classification_never_raises(self, monkeypatch):
        class Weird(Exception):
            errno = 253002

            def __str__(self):
                raise RuntimeError("unprintable")

        assert nc._artifact_download.classify_artifact_error(Weird()) is None
        errors = importlib.import_module("snowflake.connector.errors")
        monkeypatch.setattr(errors, "OperationalError", "not a class")
        assert nc._artifact_download.classify_artifact_error(ConnectionResetError()) == "retry"

    @staticmethod
    def _checkpoint_positions(tmp_path, monkeypatch, n, *, seconds_per_chunk=0.0):
        """Chunk counts at which ChunkWriter wrote its manifest while appending n chunks."""
        chunk = tmp_path / "chunk.gz"
        chunk.write_bytes(gzip.compress(b"x\n"))
        now = [0.0]
        writes = []
        real_write = nc._artifact_download.write_manifest

        def record(path, names, nbytes, skipped=None):
            writes.append(len(names))
            real_write(path, names, nbytes, skipped)

        monkeypatch.setattr(nc._artifact_download, "write_manifest", record)
        writer = nc._artifact_download.ChunkWriter(
            tmp_path / "out",
            tmp_path / "out.manifest.json",
            [],
            0,
            buffer_bytes=1024,
            checkpoint_chunks=64,
            checkpoint_seconds=5.0,
            clock=lambda: now[0],
        )
        for index in range(n):
            now[0] += seconds_per_chunk
            writer.append(f"c{index:05d}.gz", chunk)
        positions = list(writes)
        writer.finish()
        assert writes[-1] == n  # finish always records everything
        assert (tmp_path / "out").read_bytes() == b"x\n" * n
        return positions

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (100, [64]),
            (1000, [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 711, 789, 876, 973]),
            (
                5000,
                [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 711, 789, 876, 973, 1081,
                 1201, 1334, 1482, 1646, 1828, 2031, 2256, 2506, 2784, 3093, 3436, 3817,
                 4241, 4712],
            ),
        ],
    )
    def test_checkpoint_cadence_by_chunk_count(self, tmp_path, monkeypatch, n, expected):
        # Rule: checkpoint once pending >= max(64, n // 10) chunks, or after 5 s
        # once pending >= n // 50. With a frozen clock only the first applies.
        assert self._checkpoint_positions(tmp_path, monkeypatch, n) == expected

    def test_checkpoint_cadence_by_time(self, tmp_path, monkeypatch):
        positions = self._checkpoint_positions(
            tmp_path, monkeypatch, 5000, seconds_per_chunk=1.0
        )
        # One chunk per second: every 5 s while n // 50 <= 5, then spacing
        # follows n // 50 (for example 4787 -> 4884 -> 4983).
        assert positions[:4] == [5, 10, 15, 20]
        assert positions[-3:] == [4787, 4884, 4983]
        assert len(positions) == 203
        assert all(b - a >= max(5, b // 50) for a, b in zip(positions, positions[1:]))

    def test_storage_refusal_logs_the_matched_text(self, caplog):
        errors = importlib.import_module("snowflake.connector.errors")
        exc = errors.OperationalError(msg="error: '403 Forbidden' from storage", errno=253002)
        with caplog.at_level(logging.DEBUG):
            assert nc._artifact_download.classify_artifact_error(exc) is None
        assert "message matched '403'" in caplog.text

    def test_error_classification(self):
        errors = importlib.import_module("snowflake.connector.errors")
        classify = nc._artifact_download.classify_artifact_error
        assert classify(errors.OperationalError(msg="x")) == "retry"
        assert classify(errors.ServiceUnavailableError(msg="x")) == "retry"
        assert classify(errors.TooManyRequests(msg="x")) == "retry"
        assert classify(errors.DatabaseError(msg="x", errno=603)) == "retry"
        assert classify(ConnectionResetError()) == "retry"
        assert classify(nc.requests.exceptions.ConnectionError()) == "retry"
        assert classify(errors.ProgrammingError(msg="x", errno=390112)) == "reconnect"
        assert classify(errors.OperationalError(msg="x", errno=390114)) == "reconnect"
        assert classify(errors.ProgrammingError(msg="x", errno=2003)) is None
        if hasattr(errors, "NonRetryableTlsError"):  # newer connector releases only
            assert classify(errors.NonRetryableTlsError(msg="x")) is None
        assert classify(RuntimeError("x")) is None
        assert classify(gzip.BadGzipFile("x")) is None
        assert classify(PermissionError("x")) is None
        # File transfer errors that another attempt cannot fix.
        assert classify(errors.OperationalError(msg="does not exist", errno=253006)) is None
        assert classify(errors.ProgrammingError(msg="not a dir", errno=253005)) is None
        assert classify(errors.OperationalError(msg="bad stage", errno=253004)) is None
        denied = "While getting file(s) there was an error: '{}', this might be caused by your access"
        for detail in ("403 Forbidden", "AccessDenied: Access Denied", "AuthorizationFailure"):
            exc = errors.OperationalError(msg=denied.format(detail), errno=253002)
            assert classify(exc) is None, detail
        network = errors.OperationalError(msg=denied.format("ConnectionResetError(54)"), errno=253002)
        assert classify(network) == "retry"


# ─── A database or schema the API can't find ──────────────────────────────

def _failed_request(client):
    """What the API returns, with HTTP 400, when the client's URL DATABASE.SCHEMA can't be resolved."""
    return {"code": "517602", "message": f"Schema {client.database}.{client.schema} is not found or not authorized"}


class TestMissingDatabase:
    @staticmethod
    def _client(database="MY_DB", schema="PUBLIC", *, profile=None, role=None, pat=False):
        c = CortexTrainingClient(
            base_url="https://test.snowflakecomputing.com", database=database, schema=schema
        )
        c._session = MagicMock()
        if pat:
            c._session.headers = {"X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN"}
        if profile is not False and (profile or role):
            c._auth_provider = MagicMock()
            c._auth_provider.get_token.return_value = "session-token"
            c._auth_provider.connection_name = profile
            c._auth_provider.role = role
        return c

    def test_a_pat_client_gets_a_clear_error_and_nothing_runs(self):
        c = self._client(pat=True)
        c.max_retries = 5
        c._session.get.return_value = _make_error_response(_failed_request(c), 400)
        with pytest.raises(nc.DatabaseAccessError) as exc:
            c._send("GET", f"{c.base_url}/test")
        assert str(exc.value) == (
            "Your PAT's role (its role restriction, or your user's DEFAULT_ROLE if it has none) "
            "can't find database MY_DB or its schema PUBLIC.\n"
            "- If that's the wrong role: restrict the PAT to a role that has access, or run "
            "`ALTER USER <USER> SET DEFAULT_ROLE = <ROLE>`. Snowsight shows the token's role under "
            "Settings > Authentication.\n"
            "- If they don't exist: run `CREATE DATABASE IF NOT EXISTS MY_DB` and "
            "`CREATE SCHEMA IF NOT EXISTS MY_DB.PUBLIC` (or ask an admin to).\n"
            "- If they exist: ask an admin to run `GRANT USAGE ON DATABASE MY_DB TO ROLE <ROLE>` and "
            "`GRANT USAGE ON SCHEMA MY_DB.PUBLIC TO ROLE <ROLE>`.\n"
            "- Or use another database: --database NAME / CORTEX_TRAINING_DATABASE."
        )
        c._session.post.assert_not_called()  # no SQL, nothing created
        c._session.get.assert_called_once()  # not retried
        error = exc.value
        assert (error.kind, error.name, error.role, error.statement) == (
            "schema", "MY_DB.PUBLIC", None,
            "CREATE DATABASE IF NOT EXISTS MY_DB; CREATE SCHEMA IF NOT EXISTS MY_DB.PUBLIC",
        )
        assert "cortex-training login" not in str(error)  # not on main until the new login ships
        assert error.response.status_code == 400
        assert isinstance(error, nc.requests.exceptions.HTTPError) and isinstance(error, RuntimeError)

    def test_a_profile_client_names_its_role_and_profile(self):
        c = self._client(profile="prod", role="TRAINER")
        c._session.get.return_value = _make_error_response(_failed_request(c), 400)
        with pytest.raises(nc.DatabaseAccessError) as exc:
            c._send("GET", f"{c.base_url}/test")
        message = str(exc.value)
        assert message.startswith(
            "Role TRAINER can't find database MY_DB or its schema PUBLIC (from profile prod).\n"
        )
        assert "GRANT USAGE ON SCHEMA MY_DB.PUBLIC TO ROLE TRAINER" in message
        assert "PAT" not in message
        assert exc.value.role == "TRAINER"
        c._session.post.assert_not_called()

    def test_a_client_without_a_pat_or_profile_does_not_claim_a_pat(self):
        c = self._client()
        c._session.get.return_value = _make_error_response(_failed_request(c), 400)
        with pytest.raises(nc.DatabaseAccessError) as exc:
            c._send("GET", f"{c.base_url}/test")
        message = str(exc.value)
        assert message.splitlines()[0] == (
            "Role <ROLE> (couldn't determine your role) can't find database MY_DB or its schema PUBLIC."
        )
        assert "PAT" not in message

    @pytest.mark.parametrize(
        "message",
        [
            pytest.param('Schema "my db".PUBLIC is not found or not authorized', id="quoted-name"),
            pytest.param("Schema MY_DB.PUBLIC is not found or not authorized.", id="trailing-period"),
            pytest.param("Schema MY_DB.PUBLIC is not found or not authorized for role R", id="appended-detail"),
        ],
    )
    def test_the_message_is_explained_with_small_wording_changes(self, message):
        c = self._client(profile="prod", role="TRAINER")
        body = {"code": "517602", "message": message}
        c._session.get.return_value = _make_error_response(body, 400)
        with pytest.raises(nc.DatabaseAccessError) as exc:
            c._send("GET", f"{c.base_url}/test")
        assert exc.value.name == "MY_DB.PUBLIC"

    def test_the_default_profile_is_named_as_such(self):
        c = self._client(profile=None, role="TRAINER")
        c._session.get.return_value = _make_error_response(_failed_request(c), 400)
        with pytest.raises(nc.DatabaseAccessError, match=r"\(from your default Snowflake profile\)"):
            c._send("GET", f"{c.base_url}/test")

    def test_a_non_public_schema_covers_both_database_and_schema(self):
        c = self._client(schema="CUSTOM", profile="prod", role="TRAINER")
        c._session.get.return_value = _make_error_response(_failed_request(c), 400)
        with pytest.raises(nc.DatabaseAccessError) as exc:
            c._send("GET", f"{c.base_url}/test")
        assert str(exc.value).splitlines()[:3] == [
            "Role TRAINER can't find database MY_DB or its schema CUSTOM (from profile prod).",
            "- If they don't exist: run `CREATE DATABASE IF NOT EXISTS MY_DB` and "
            "`CREATE SCHEMA IF NOT EXISTS MY_DB.CUSTOM` (or ask an admin to).",
            "- If they exist: ask an admin to run `GRANT USAGE ON DATABASE MY_DB TO ROLE TRAINER` and "
            "`GRANT USAGE ON SCHEMA MY_DB.CUSTOM TO ROLE TRAINER`.",
        ]
        assert (exc.value.kind, exc.value.name, exc.value.statement) == (
            "schema", "MY_DB.CUSTOM", "CREATE DATABASE IF NOT EXISTS MY_DB; CREATE SCHEMA IF NOT EXISTS MY_DB.CUSTOM"
        )

    @pytest.mark.parametrize(
        ("body", "status"),
        [
            pytest.param({"code": "517604", "message": 'invalid hardware "NOT_A_GPU"'}, 400, id="other-code"),
            pytest.param({"message": "Schema MY_DB.PUBLIC is not found or not authorized"}, 400, id="no-code"),
            pytest.param({"code": "517602", "message": "Schema MY_DB.PUBLIC is not found or not authorized"}, 422, id="not-400"),
            pytest.param({"code": "517602", "message": "Unsupported model: x/y"}, 400, id="invalid-request"),
            pytest.param({"code": "517602"}, 400, id="invalid-request-no-message"),
            pytest.param(
                {"code": "517602", "message": "Stage MY_DB.PUBLIC.EXPORTS is not found or not authorized"}, 400,
                id="stage-not-found",
            ),
            pytest.param(
                {"code": "517602", "message": "Experiment MY_DB.PUBLIC.EXP not found or not authorized"}, 400,
                id="experiment-not-found",
            ),
            pytest.param(
                {"code": "517602", "message": "CREATE EXPERIMENT privilege required on schema MY_DB.PUBLIC"}, 400,
                id="missing-create-experiment",
            ),
            pytest.param({}, 404, id="missing-job"),
            pytest.param({"message": "boom"}, 500, id="server-error"),
        ],
    )
    def test_other_errors_are_left_as_they_are(self, body, status):
        c = self._client()
        c.max_retries = 0
        c._session.get.return_value = _make_error_response(body, status)
        with pytest.raises(nc.requests.exceptions.HTTPError, match=str(status)) as exc:
            c._send("GET", f"{c.base_url}/test")
        assert not isinstance(exc.value, nc.DatabaseAccessError)

    def test_routes_without_a_database_leave_the_error_as_it_is(self):
        c = self._client()
        c._session.get.return_value = _make_error_response(_failed_request(c), 400)
        with pytest.raises(nc.requests.exceptions.HTTPError) as exc:
            c._send("GET", f"{c.base_url}/test", explain_database=False)
        assert not isinstance(exc.value, nc.DatabaseAccessError)

    def test_a_job_client_leaves_the_error_as_it_is(self):
        c = self._client()
        job_client = c.for_job({"database_name": "OTHER_DB", "schema_name": "PUBLIC"})
        c._session.get.return_value = _make_error_response(_failed_request(job_client), 400)
        with pytest.raises(nc.requests.exceptions.HTTPError) as exc:
            job_client._send("GET", f"{c.base_url}/test")
        assert not isinstance(exc.value, nc.DatabaseAccessError)

    def test_auto_create_database_is_deprecated_and_does_nothing(self):
        c = self._client()
        assert c.auto_create_database is False
        with pytest.warns(FutureWarning, match="auto_create_database is deprecated and does nothing"):
            c.auto_create_database = True
        assert c.auto_create_database is False

    def test_setting_auto_create_database_to_false_is_silent(self):
        c = self._client()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            c.auto_create_database = False
        assert c.auto_create_database is False


# ─── Not-found handling and log-tail stopping ────────────────────────────────


_SAMPLING_BODY = {
    "sub_job_configs": [{"job_type": "sampling", "model_name": "gpt2", "inference_config": {"n_gpus": 1}}]
}


def _not_found(code, message, *, url="http://test.local/api/v2/x/jobs/missing", status=404):
    response = _make_error_response({"code": code, "message": message, "error_code": code}, status)
    response.url = url
    return response


class TestNotFoundFailsFast:
    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch):
        monkeypatch.setattr(nc, "wait_exponential_jitter", lambda **kwargs: lambda retry_state: 0)

    def test_unknown_job_is_not_retried(self):
        c = _make_client()
        c.max_retries = 5
        c._session.get.return_value = _not_found("517601", "Training job missing not found in PUBLIC")
        with pytest.raises(nc.requests.exceptions.HTTPError) as excinfo:
            c.get_job("missing")
        assert excinfo.value.response.status_code == 404
        assert c._session.get.call_count == 1

    @pytest.mark.parametrize(
        "response",
        [
            # A gateway or proxy 404 has no service error code.
            pytest.param(lambda: _make_error_response({"message": "no route"}, 404), id="no-code"),
            pytest.param(lambda: _make_response(status_code=404), id="no-json"),
            pytest.param(lambda: _not_found("517604", "session s not found"), id="other-code"),
        ],
    )
    def test_other_404s_are_still_retried(self, response):
        c = _make_client()
        c.max_retries = 2
        first = response()
        if isinstance(first, MagicMock):
            first.raise_for_status.side_effect = nc.requests.exceptions.HTTPError(response=first)
        c._session.get.side_effect = [first, _make_response({"job_id": "j1", "status": "RUNNING"})]
        assert c.get_job("j1")["status"] == "RUNNING"
        assert c._session.get.call_count == 2

    def test_a_job_this_client_just_created_is_retried_while_not_yet_visible(self):
        c = _make_client(post_json={"job_id": "new-job"})
        c.max_retries = 3
        c.create_job_from_body(_SAMPLING_BODY)
        c._session.get.side_effect = [
            _not_found("517601", "Training job new-job not found", url=f"{c._prefix}/new-job"),
            _not_found("517601", "Training job new-job not found", url=f"{c._prefix}/new-job"),
            _make_response({"job_id": "new-job", "status": "PENDING"}),
        ]
        assert c.get_job("new-job")["status"] == "PENDING"
        assert c._session.get.call_count == 3

    def test_the_create_grace_period_expires(self, monkeypatch):
        c = _make_client(post_json={"job_id": "new-job"})
        c.max_retries = 3
        c.create_job_from_body(_SAMPLING_BODY)
        created_at = c._recently_created_jobs["new-job"]
        monkeypatch.setattr(nc.time, "monotonic", lambda: created_at + nc._RECENTLY_CREATED_SECONDS + 1)
        c._session.get.return_value = _not_found("517601", "not found", url=f"{c._prefix}/new-job")
        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.get_job("new-job")
        assert c._session.get.call_count == 1


class TestTailOnEndedJobs:
    @pytest.fixture(autouse=True)
    def _no_backoff(self, monkeypatch):
        monkeypatch.setattr(nc, "wait_exponential_jitter", lambda **kwargs: lambda retry_state: 0)

    def _job(self, status):
        return _make_response(
            {"job_id": "j1", "status": status, "sub_jobs": [{"sub_job_id": "j1:sampling:0", "job_type": "SAMPLING"}]}
        )

    @pytest.mark.parametrize("status", ["TERMINATED", "CANCELLED", "JOB_STATE_FAILED"])
    def test_finished_job_fails_at_once_and_points_at_download_log(self, status):
        c = _make_client()
        c.max_retries = 10
        c._session.post.return_value = _not_found("517604", "Cortex training operation failed: session s not found")
        c._session.get.return_value = self._job(status)
        with pytest.raises(nc.requests.exceptions.HTTPError) as excinfo:
            c.tail_logs("j1", sub_job_id="j1:sampling:0")
        message = str(excinfo.value)
        assert "cortex-training download-log j1 --log-type stdout" in message
        assert "download_stdout_logs" in message
        assert excinfo.value.response.status_code == 404
        assert c._session.post.call_count == 1
        assert c._session.get.call_count == 1  # one job lookup per call

    def test_failed_session_410_gets_the_same_hint(self):
        c = _make_client()
        c._session.post.return_value = _not_found("517604", "session s has failed: runtime_sub_job_failed", status=410)
        c._session.get.return_value = self._job("FAILED")
        with pytest.raises(nc.requests.exceptions.HTTPError, match="download-log j1 --log-type stdout"):
            c.tail_logs("j1", sub_job_type="sampling")
        assert c._session.post.call_count == 1

    def test_unknown_sub_job_of_a_running_job_fails_at_once(self):
        c = _make_client()
        c.max_retries = 10
        c._session.post.return_value = _not_found("517604", "session s not found")
        c._session.get.return_value = self._job("RUNNING")
        match = "Sub-job j1:training:0 is not part of job j1.*j1:sampling:0"
        with pytest.raises(nc.requests.exceptions.HTTPError, match=match):
            c.tail_logs("j1", sub_job_id="j1:training:0")
        assert c._session.post.call_count == 1

    def test_a_starting_job_is_still_retried(self):
        c = _make_client()
        c.max_retries = 3
        c._session.post.side_effect = [
            _not_found("517604", "session s not found"),
            _make_response({"entries": [{"_raw": "x"}], "next_cursor": "1", "eof": False}),
        ]
        c._session.get.return_value = self._job("PLACING")
        assert c.tail_logs("j1", sub_job_id="j1:sampling:0")["entries"] == [{"_raw": "x"}]
        assert c._session.post.call_count == 2
        assert c._session.get.call_count == 1

    def test_lookup_failure_keeps_the_old_retry_behavior(self):
        c = _make_client()
        c.max_retries = 2
        c._session.post.return_value = _not_found("517604", "session s not found")
        c._session.get.side_effect = RuntimeError("lookup down")
        with pytest.raises(nc.requests.exceptions.HTTPError) as excinfo:
            c.tail_logs("j1", sub_job_id="j1:sampling:0")
        assert "download-log" not in str(excinfo.value)
        assert c._session.post.call_count == 3

    def test_the_job_lookup_is_tried_once(self):
        c = _make_client()
        c.max_retries = 2
        c._session.post.return_value = _not_found("517604", "session s not found")
        c._session.get.return_value = _make_error_response({"message": "unavailable"}, 503)
        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.tail_logs("j1", sub_job_id="j1:sampling:0")
        assert c._session.get.call_count == 1
        assert c._session.post.call_count == 3

    def test_each_tui_tail_page_is_one_metric_without_its_cursor(self, monkeypatch):
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        c = _make_client(post_json={"entries": [{"_raw": "x"}], "next_cursor": "c-secret", "eof": False})
        c._metric_emitter = MagicMock()
        c._telemetry_surface = nc.TUI_SURFACE

        c.tail_logs("j1", sub_job_id="j1:sampling:0")
        c.tail_logs("j1", cursor="c-secret", max_lines=50, sub_job_id="j1:sampling:0")

        calls = c._metric_emitter.emit.call_args_list
        assert [call.args[0] for call in calls] == ["tail_logs", "tail_logs"]
        assert [call.args[1]["success"] for call in calls] == [True, True]
        assert [call.kwargs["attributes"] for call in calls] == [
            {"client.surface": "tui", "job_id": "j1", "sub_job_id": "j1:sampling:0"}
        ] * 2
        assert "c-secret" not in repr(calls)

    @pytest.mark.parametrize(
        ("status_code", "message", "status", "job_status"),
        [
            (404, "session s not found", "TERMINATED", "terminated"),
            (409, "session s not found", "TERMINATED", "terminated"),
            # Not retried, so the job is looked up only once the call has failed.
            (410, "session s has failed: runtime_sub_job_failed", "JOB_STATE_FAILED", "failed"),
        ],
        ids=["404", "409", "410"],
    )
    def test_an_ended_job_s_tail_is_one_failure_marked_job_ended(
        self, monkeypatch, status_code, message, status, job_status
    ):
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        c = _make_client()
        c._metric_emitter = MagicMock()
        c._session.post.return_value = _not_found("517604", message, status=status_code)
        c._session.get.return_value = self._job(status)

        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.tail_logs("j1", sub_job_id="j1:sampling:0")

        assert (c._session.post.call_count, c._session.get.call_count) == (1, 1)
        c._metric_emitter.emit.assert_called_once()
        operation, value = c._metric_emitter.emit.call_args.args
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert operation == "tail_logs"
        assert value["success"] is False and value["job_ended"] is True
        assert value["job_status"] == job_status
        assert value["request_count"] == 2  # the tail, and the job lookup that explains it
        assert (attributes["job_id"], attributes["sub_job_id"]) == ("j1", "j1:sampling:0")
        assert (attributes["error.type"], attributes["error.http_status"]) == ("HTTPError", status_code)

    def test_other_tail_failures_are_not_marked_job_ended(self):
        c = _make_client()
        c.max_retries = 0
        c._metric_emitter = MagicMock()
        c._session.post.return_value = _make_error_response({"message": "bad"}, 400)

        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.tail_logs("j1", sub_job_id="j1:sampling:0")

        value = c._metric_emitter.emit.call_args.args[1]
        assert value["success"] is False and "job_ended" not in value and "job_status" not in value

    def test_a_tail_failing_while_the_job_is_placed_says_so(self):
        # The service answers 409 until the job's session exists: expected while placing.
        c = _make_client()
        c.max_retries = 0
        c._metric_emitter = MagicMock()
        c._session.post.return_value = _not_found("517604", "job j1 is pending placement", status=409)
        c._session.get.return_value = self._job("JOB_STATE_PLACING")

        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.tail_logs("j1", sub_job_id="j1:sampling:0")

        value = c._metric_emitter.emit.call_args.args[1]
        assert (value["success"], value["job_status"]) == (False, "placing")
        assert "job_ended" not in value

    def test_a_status_the_server_makes_up_is_cut_short_in_the_metric(self):
        c = _make_client()
        c.max_retries = 0
        c._metric_emitter = MagicMock()
        c._session.post.return_value = _not_found("517604", "session s not found", status=409)
        c._session.get.return_value = _make_response(
            {"job_id": "j1", "status": "JOB_STATE_" + "X" * 100, "sub_jobs": [{"sub_job_id": "j1:sampling:0"}]}
        )

        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.tail_logs("j1", sub_job_id="j1:sampling:0")

        assert c._metric_emitter.emit.call_args.args[1]["job_status"] == "x" * 32

    def test_a_tail_of_a_sub_job_the_job_lacks_is_not_marked_job_ended(self):
        # The job runs; the sub-job id is what is wrong, so the failure is a real one.
        c = _make_client()
        c._metric_emitter = MagicMock()
        c._session.post.return_value = _not_found("517604", "session s not found")
        c._session.get.return_value = self._job("RUNNING")

        with pytest.raises(nc.requests.exceptions.HTTPError, match="is not part of job j1") as raised:
            c.tail_logs("j1", sub_job_id="j1:training:0")

        assert raised.value.job_ended is False
        c._metric_emitter.emit.assert_called_once()
        value = c._metric_emitter.emit.call_args.args[1]
        assert (value["success"], value["job_status"]) == (False, "running")
        assert "job_ended" not in value

    def test_a_410_while_the_job_still_runs_has_its_status_but_is_not_marked_job_ended(self):
        # A 410 is not retried, so only the failure's own lookup reads the status.
        c = _make_client()
        c._metric_emitter = MagicMock()
        c._session.post.return_value = _not_found(
            "517604", "session s has failed: runtime_sub_job_failed", status=410)
        c._session.get.return_value = self._job("RUNNING")

        with pytest.raises(nc.requests.exceptions.HTTPError) as raised:
            c.tail_logs("j1", sub_job_id="j1:sampling:0")

        assert raised.value.response.status_code == 410
        assert (c._session.post.call_count, c._session.get.call_count) == (1, 1)
        c._metric_emitter.emit.assert_called_once()
        value = c._metric_emitter.emit.call_args.args[1]
        assert (value["success"], value["job_status"]) == (False, "running")
        assert "job_ended" not in value

    @pytest.mark.parametrize("status", [{}, {"status": None}, {"status": ""}], ids=["missing", "null", "empty"])
    def test_a_job_lookup_without_a_status_sends_no_job_status(self, status):
        c = _make_client()
        c.max_retries = 0
        c._metric_emitter = MagicMock()
        c._session.post.return_value = _not_found("517604", "session s not found", status=409)
        c._session.get.return_value = _make_response(
            {"job_id": "j1", **status, "sub_jobs": [{"sub_job_id": "j1:sampling:0"}]}
        )

        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.tail_logs("j1", sub_job_id="j1:sampling:0")

        assert c._session.get.call_count == 1  # the lookup was made, and read no status
        value = c._metric_emitter.emit.call_args.args[1]
        assert value["success"] is False
        assert "job_status" not in value and "job_ended" not in value

    @pytest.mark.parametrize(
        ("first", "status"),
        [
            (lambda: _not_found("517604", "session s not found"), "PLACING"),
            (lambda: _not_found("517604", "job j1 is pending placement", status=409), "JOB_STATE_PLACING"),
            (lambda: _make_error_response({"message": "the target zone is restarting"}, 409), "RUNNING"),
        ],
        ids=["404-placing", "409-placing", "409-running"],
    )
    def test_a_tail_a_retry_gets_through_sends_no_job_status(self, monkeypatch, first, status):
        # The lookup explained an attempt the retry got past, not the page that came back.
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        c = _make_client()
        c.max_retries = 2  # as the TUI reads
        c._metric_emitter = MagicMock()
        c._telemetry_surface = nc.TUI_SURFACE
        c._session.post.side_effect = [
            first(),
            _make_response({"entries": [{"_raw": "x"}], "next_cursor": "1", "eof": False}),
        ]
        c._session.get.return_value = self._job(status)

        assert c.tail_logs("j1", sub_job_id="j1:sampling:0")["entries"] == [{"_raw": "x"}]

        assert c._session.get.call_count == 1  # the job was looked up
        c._metric_emitter.emit.assert_called_once()
        value = c._metric_emitter.emit.call_args.args[1]
        assert (value["success"], value["retry_count"]) == (True, 1)
        assert "job_status" not in value and "job_ended" not in value

    def test_a_retried_tail_that_still_fails_has_the_job_status_its_lookup_read(self):
        c = _make_client()
        c.max_retries = 2
        c._metric_emitter = MagicMock()
        c._session.post.return_value = _not_found("517604", "job j1 is pending placement", status=409)
        c._session.get.return_value = self._job("JOB_STATE_PLACING")

        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.tail_logs("j1", sub_job_id="j1:sampling:0")

        assert (c._session.post.call_count, c._session.get.call_count) == (3, 1)
        c._metric_emitter.emit.assert_called_once()
        value = c._metric_emitter.emit.call_args.args[1]
        assert (value["success"], value["retry_count"], value["job_status"]) == (False, 2, "placing")

    @pytest.mark.parametrize(
        ("job", "job_status"),
        [
            ({"status": "JOB_STATE_PLACING"}, "placing"),
            ({"status": "JOB_STATE_" + "X" * 100}, "x" * 32),  # made up by the server: cut short
            ({}, None),
            ({"status": None}, None),
            ({"status": ""}, None),
        ],
        ids=["placing", "made-up", "missing", "null", "empty"],
    )
    @pytest.mark.parametrize(
        "error",
        [
            nc.requests.exceptions.ConnectionError("connection reset by peer"),
            nc.requests.exceptions.ReadTimeout("read timed out"),
        ],
        ids=["reset", "timeout"],
    )
    def test_a_tail_whose_retries_then_fail_off_the_wire_has_the_job_status_too(self, error, job, job_status):
        # The first attempt's 409 had the job looked up; the retries got no answer at all.
        # Its status is sent as for a tail whose own answer failed it: cut short, and only when read.
        c = _make_client()
        c.max_retries = 2  # as the TUI reads
        c._metric_emitter = MagicMock()
        c._session.post.side_effect = [
            _not_found("517604", "job j1 is pending placement", status=409), error, error]
        c._session.get.return_value = _make_response(
            {"job_id": "j1", **job, "sub_jobs": [{"sub_job_id": "j1:sampling:0"}]}
        )

        with pytest.raises(type(error)):
            c.tail_logs("j1", sub_job_id="j1:sampling:0")

        assert (c._session.post.call_count, c._session.get.call_count) == (3, 1)
        c._metric_emitter.emit.assert_called_once()
        value = c._metric_emitter.emit.call_args.args[1]
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert (value["success"], value.get("job_status")) == (False, job_status)
        assert attributes["error.type"] == type(error).__name__
        assert "job_ended" not in value

    def test_a_tail_whose_retry_cannot_get_a_token_has_the_job_status_too(self):
        # Not every later attempt fails off the wire: one that gets no token fails the call too.
        from cortex_training.client.auth import SnowflakeProfileError

        c = _make_client()
        c.max_retries = 2  # as the TUI reads
        c._metric_emitter = MagicMock()
        auth = MagicMock()
        # The first attempt and its job lookup get a token; the retry's request for one fails.
        no_token = SnowflakeProfileError("Snowflake Connector did not expose a session token")
        auth.get_token.side_effect = ["session-token", "session-token", no_token]
        c._auth_provider = auth
        c._session.post.return_value = _not_found("517604", "job j1 is pending placement", status=409)
        c._session.get.return_value = self._job("JOB_STATE_PLACING")

        with pytest.raises(SnowflakeProfileError):
            c.tail_logs("j1", sub_job_id="j1:sampling:0")

        assert (c._session.post.call_count, c._session.get.call_count, auth.get_token.call_count) == (1, 1, 3)
        c._metric_emitter.emit.assert_called_once()
        value = c._metric_emitter.emit.call_args.args[1]
        attributes = c._metric_emitter.emit.call_args.kwargs["attributes"]
        assert (value["success"], value["job_status"]) == (False, "placing")
        assert attributes["error.type"] == "SnowflakeProfileError"
        assert "job_ended" not in value

    def test_a_tail_that_never_got_an_answer_sends_no_job_status(self):
        c = _make_client()
        c.max_retries = 2
        c._metric_emitter = MagicMock()
        c._session.post.side_effect = nc.requests.exceptions.ConnectionError("connection reset by peer")

        with pytest.raises(nc.requests.exceptions.ConnectionError):
            c.tail_logs("j1", sub_job_id="j1:sampling:0")

        assert (c._session.post.call_count, c._session.get.call_count) == (3, 0)  # nothing to look up for
        value = c._metric_emitter.emit.call_args.args[1]
        assert value["success"] is False and "job_status" not in value

    def test_a_tail_stopped_after_a_lookup_is_cancelled_without_a_job_status(self, monkeypatch):
        # Only a failed call's event has the status; Ctrl-C is a stop, not a failure.
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        c = _make_client()
        c.max_retries = 2
        c._metric_emitter = MagicMock()
        c._telemetry_surface = nc.TUI_SURFACE
        c._session.post.side_effect = [
            _not_found("517604", "job j1 is pending placement", status=409), KeyboardInterrupt()]
        c._session.get.return_value = self._job("JOB_STATE_PLACING")

        with pytest.raises(KeyboardInterrupt):
            c.tail_logs("j1", sub_job_id="j1:sampling:0")

        assert c._session.get.call_count == 1  # the job was looked up
        c._metric_emitter.emit.assert_called_once()
        value = c._metric_emitter.emit.call_args.args[1]
        assert (value["success"], value["cancelled"]) == (False, True)
        assert "job_status" not in value


class TestStreamStopsWithoutEof:
    def test_follow_false_stops_at_the_first_empty_page_without_eof(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(nc.time, "sleep", sleeps.append)
        c = _make_client()
        c._session.post.side_effect = [
            _make_response({"entries": [{"msg": "a"}], "next_cursor": "c1", "eof": False}),
            _make_response({"entries": [{"msg": "b"}], "next_cursor": "c2", "eof": False}),
            _make_response({"entries": [], "next_cursor": "c2", "eof": False}),
            AssertionError("polled past the tail"),
        ]
        assert [e["msg"] for e in c.stream_logs("j1", sub_job_id="sj-0", follow=False)] == ["a", "b"]
        assert c._session.post.call_count == 3
        assert sleeps == []

    def test_follow_false_on_an_empty_log_stops_after_one_page(self):
        c = _make_client(post_json={"entries": [], "next_cursor": "", "eof": False})
        assert list(c.stream_logs("j1", sub_job_id="sj-0", follow=False)) == []
        assert c._session.post.call_count == 1

    def test_follow_true_keeps_polling_through_empty_pages(self, monkeypatch):
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        c = _make_client()
        c._session.post.side_effect = [
            _make_response({"entries": [], "next_cursor": "c0", "eof": False}),
            _make_response({"entries": [{"msg": "late"}], "next_cursor": "c1", "eof": False}),
        ]
        stream = c.stream_logs("j1", sub_job_id="sj-0", follow=True)
        assert next(stream)["msg"] == "late"

    def test_follow_true_ends_at_an_empty_eof_page(self, monkeypatch):
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        c = _make_client()
        c._session.post.side_effect = [
            _make_response({"entries": [{"msg": "a"}], "next_cursor": "c1", "eof": True}),
            _make_response({"entries": [], "next_cursor": "c1", "eof": True}),
            AssertionError("polled past eof"),
        ]
        assert [e["msg"] for e in c.stream_logs("j1", sub_job_id="sj-0", follow=True)] == ["a"]

class TestStreamEndsWhenTheJobEnds:
    @pytest.fixture(autouse=True)
    def _no_waits(self, monkeypatch):
        monkeypatch.setattr(nc, "wait_exponential_jitter", lambda **kwargs: lambda retry_state: 0)
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)

    def _job(self, status):
        return _make_response({"job_id": "j1", "status": status, "sub_jobs": [{"sub_job_id": "j1:s:0"}]})

    @pytest.mark.parametrize("status_code", [404, 409, 410])
    def test_a_following_stream_ends_cleanly_when_the_job_ends(self, status_code):
        c = _make_client()
        c.max_retries = 10
        c._session.post.side_effect = [
            _make_response({"entries": [{"msg": "a"}], "next_cursor": "c1", "eof": False}),
            _make_response({"entries": [], "next_cursor": "c1", "eof": False}),
            _not_found("517604", "session s not found", status=status_code),
            AssertionError("polled after the job ended"),
        ]
        c._session.get.return_value = self._job("CANCELLED")
        assert [e["msg"] for e in c.stream_logs("j1", sub_job_id="j1:s:0", follow=True)] == ["a"]
        assert c._session.get.call_count == 1

    def test_by_default_a_follow_sends_only_the_end_of_the_job(self, monkeypatch):
        # A CLI or SDK follow polls for as long as the job runs, so its pages stay
        # failure-only; the one event is the tail the job's end stopped.
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        c = _make_client()
        c._metric_emitter = MagicMock()
        c._telemetry_surface = nc.CLI_SURFACE
        c._session.post.side_effect = [
            _make_response({"entries": [{"msg": "a"}], "next_cursor": "c1", "eof": False}),
            _make_response({"entries": [], "next_cursor": "c1", "eof": False}),
            _not_found("517604", "session s not found", status=409),
        ]
        c._session.get.return_value = self._job("CANCELLED")

        assert [e["msg"] for e in c.stream_logs("j1", sub_job_id="j1:s:0", follow=True)] == ["a"]

        c._metric_emitter.emit.assert_called_once()
        operation, value = c._metric_emitter.emit.call_args.args
        assert (operation, value["success"], value["job_ended"]) == ("tail_logs", False, True)
        assert c._metric_emitter.emit.call_args.kwargs["attributes"][nc.SURFACE_ATTRIBUTE] == "cli"

    def test_a_job_that_ended_before_the_first_read_still_raises(self):
        c = _make_client()
        c._session.post.return_value = _not_found("517604", "session s not found")
        c._session.get.return_value = self._job("TERMINATED")
        with pytest.raises(nc.requests.exceptions.HTTPError, match="download-log j1 --log-type stdout"):
            list(c.stream_logs("j1", sub_job_id="j1:s:0", follow=True))

    def test_other_errors_mid_stream_still_raise(self):
        c = _make_client()
        c.max_retries = 0
        c._session.post.side_effect = [
            _make_response({"entries": [{"msg": "a"}], "next_cursor": "c1", "eof": False}),
            _make_error_response({"message": "bad"}, 400),
        ]
        stream = c.stream_logs("j1", sub_job_id="j1:s:0", follow=True)
        assert next(stream)["msg"] == "a"
        with pytest.raises(nc.requests.exceptions.HTTPError):
            next(stream)

    def test_tail_logs_marks_an_ended_job_error(self):
        c = _make_client()
        c._session.post.return_value = _not_found("517604", "session s not found", status=409)
        c._session.get.return_value = self._job("FAILED")
        with pytest.raises(nc.requests.exceptions.HTTPError) as excinfo:
            c.tail_logs("j1", sub_job_id="j1:s:0")
        assert excinfo.value.job_ended is True
        assert excinfo.value.response.status_code == 409
        assert c._session.post.call_count == 1  # a 409 on an ended job is not retried


# ─── Streamed execution-log download ─────────────────────────────────────────


class TestDownloadExecutionLogs:
    RUN_URI = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"

    def _setup(self, monkeypatch, files, *, checkpoint_files=None):
        """``files`` maps a ``_logs/``-relative path to ``(bytes, md5)``."""
        c = _make_client(get_json={"experiment_name": "DB.SCH.EXP", "experiment_run_name": "RUN_ABC"})
        c._artifact_retry_policy = nc._artifact_download.RetryPolicy(sleep=lambda _delay: None)

        def rows(root, entries):
            return [
                (f"/versions/RUN_ABC/{root}{path}", len(body), md5, "Thu, 15 Oct 2026 00:00:00 GMT")
                for path, (body, md5) in entries.items()
            ]

        checkpoint_files = checkpoint_files or {}
        connection = TestExecutionLogDownload.FakeArtifactConnection(
            {
                self.RUN_URI + "_logs/": rows("_logs/", files),
                self.RUN_URI + "checkpoints/_logs/": rows("checkpoints/_logs/", checkpoint_files),
            },
            {
                **{self.RUN_URI + "_logs/" + path: body for path, (body, _md5) in files.items()},
                **{self.RUN_URI + "checkpoints/_logs/" + path: body for path, (body, _md5) in checkpoint_files.items()},
            },
        )
        monkeypatch.setattr(c, "_open_experiment_artifact_connection", lambda: connection)
        return c, connection

    @staticmethod
    def _gets(connection):
        return [cmd for cmd in connection.commands if cmd.startswith("GET ")]

    def test_saves_each_file_privately_without_returning_its_body(self, tmp_path, monkeypatch):
        files = {
            "job-1:sampling:0/execution.jsonl": (b'{"a": 1}\n' * 1000, "m1"),
            "job-1:sampling:0/server.log": (b"server line\n", "m2"),
            "other-job:sampling:0/execution.jsonl": (b"not ours\n", "m3"),
        }
        c, connection = self._setup(monkeypatch, files)
        messages = []
        result = c.download_execution_logs("job-1", tmp_path, progress=messages.append)
        assert [(r["sub_job_id"], r["filename"]) for r in result] == [
            ("job-1:sampling:0", "execution.jsonl"),
            ("job-1:sampling:0", "server.log"),
        ]
        for entry, (path, (body, _md5)) in zip(result, list(files.items())[:2]):
            saved = Path(entry["saved_path"])
            assert saved == tmp_path / path
            assert saved.read_bytes() == body
            assert entry["bytes"] == len(body) and entry["unchanged"] is False
            assert entry["artifact_uri"] == self.RUN_URI + "_logs/" + path
            assert "content" not in entry
            assert saved.stat().st_mode & 0o777 == 0o600
        assert not (tmp_path / "other-job:sampling:0").exists()
        # No download directories are left behind.
        assert not list((tmp_path / "job-1:sampling:0").glob(".execution-log-*"))
        assert "".join(messages).count("\n") == 2
        assert "execution logs: job-1:sampling:0/execution.jsonl: saved 8.8 KiB" in "".join(messages)
        assert connection.closed

    def test_execution_log_download_successes_are_on_by_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        c, _connection = self._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"one\n", "m1")})
        c._metric_emitter = MagicMock()

        c.download_execution_logs("job-1", tmp_path)
        c.download_execution_logs("job-1", tmp_path, resume=True)  # unchanged: kept, not fetched

        calls = [call for call in c._metric_emitter.emit.call_args_list
                 if call.args[0] == "download_execution_logs"]
        fields = ("success", "artifact", "file_count", "files_skipped", "bytes_written", "cancelled")
        assert [{key: call.args[1][key] for key in fields} for call in calls] == [
            {"success": True, "artifact": "execution_logs", "file_count": 1, "files_skipped": 0,
             "bytes_written": 4, "cancelled": False},
            {"success": True, "artifact": "execution_logs", "file_count": 1, "files_skipped": 1,
             "bytes_written": 0, "cancelled": False},
        ]
        assert calls[0].kwargs["attributes"] == {"client.surface": "sdk", "job_id": "job-1"}

    @staticmethod
    def _events(c):
        return [call.args[1] for call in c._metric_emitter.emit.call_args_list
                if call.args[0] == "download_execution_logs"]

    def test_an_execution_log_download_times_its_list_and_fetch_and_counts_retried_gets(self, tmp_path, monkeypatch):
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        c, connection = self._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"one\n", "m1")})
        c._metric_emitter = MagicMock()
        now = [1000.0]
        monkeypatch.setattr(nc.time, "monotonic", lambda: now[0])
        run = c._session.get.return_value

        def slow_run_lookup(*args, **kwargs):  # the experiment-run lookup takes 3 s
            now[0] += 3.0
            return run

        def slow_open():  # and opening the artifact connection 1.5 s
            now[0] += 1.5
            return connection

        listing = c._list_experiment_artifact_files

        def slow_list(*args, **kwargs):  # each of its two LISTs takes 0.5 s
            now[0] += 0.5
            return listing(*args, **kwargs)

        gets = []

        def slow_get(source):  # each GET attempt takes 2 s; the first runs past its time limit
            gets.append(source)
            now[0] += 2.0
            if len(gets) == 1:
                raise nc._artifact_download.TransferTimeout(f"GET {source} did not finish in time")

        c._session.get.side_effect = slow_run_lookup
        monkeypatch.setattr(c, "_open_experiment_artifact_connection", slow_open)
        monkeypatch.setattr(c, "_list_experiment_artifact_files", slow_list)
        connection.on_get = slow_get
        c.download_execution_logs("job-1", tmp_path)

        assert len(gets) == 2
        (value,) = self._events(c)
        assert (value["success"], value["artifact_retry_count"], value["transfer_timeout_count"]) == (True, 1, 1)
        assert value["retry_count"] == 0  # REST retries only: the GET's retry is counted above
        # The run lookup and the connection come before the listing: only the whole call's time has them.
        assert (value["list_ms"], value["fetch_ms"], value["duration_ms"]) == (1000.0, 4000.0, 9500.0)

    def test_an_execution_log_download_that_keeps_timing_out_counts_its_retries_in_the_failure(
        self, tmp_path, monkeypatch
    ):
        c, connection = self._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"one\n", "m1")})
        c._metric_emitter = MagicMock()

        def get(source):  # every attempt runs past its time limit
            raise nc._artifact_download.TransferTimeout(f"GET {source} did not finish in time")

        connection.on_get = get
        with pytest.raises(nc._artifact_download.TransferTimeout):
            c.download_execution_logs("job-1", tmp_path)

        attempts = c._artifact_retry_policy.attempts
        assert len(self._gets(connection)) == attempts
        (value,) = self._events(c)
        # Every attempt timed out; all but the last were retried.
        assert (value["success"], value["artifact_retry_count"], value["transfer_timeout_count"]) == (
            False, attempts - 1, attempts)

    def test_an_execution_log_download_tells_retried_gets_from_timed_out_ones(self, tmp_path, monkeypatch):
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        c, connection = self._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"one\n", "m1")})
        c._metric_emitter = MagicMock()
        # The first GET's connection is reset, the second runs past its time limit, the third works.
        failures = [ConnectionResetError("connection reset by peer"),
                    nc._artifact_download.TransferTimeout("GET did not finish in time")]

        def get(source):
            if failures:
                raise failures.pop(0)

        connection.on_get = get
        c.download_execution_logs("job-1", tmp_path)

        assert len(self._gets(connection)) == 3
        (value,) = self._events(c)
        assert (value["success"], value["artifact_retry_count"], value["transfer_timeout_count"]) == (True, 2, 1)

    def test_an_execution_log_download_counts_every_file_its_sub_job_filter_keeps(self, tmp_path, monkeypatch):
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        files = {
            "job-1:sampling:0/execution.jsonl": (b"one\n", "m1"),  # 4 bytes
            "job-1:sampling:0/server.log": (b"server\n", "s1"),  # 7 bytes
            "job-1:training:0/execution.jsonl": (b"training\n", "t1"),  # not asked for
        }
        c, _connection = self._setup(monkeypatch, files)
        c._metric_emitter = MagicMock()

        c.download_execution_logs("job-1", tmp_path, sub_job_ids=["job-1:sampling:0"])
        c.download_execution_logs("job-1", tmp_path, sub_job_ids=["job-1:sampling:0"], resume=True)  # both kept

        fields = ("success", "file_count", "files_skipped", "bytes_written")
        assert [tuple(value[key] for key in fields) for value in self._events(c)] == [
            (True, 2, 0, 11), (True, 2, 2, 0)]

    @pytest.mark.parametrize(
        ("error", "cancelled"),
        [(nc._artifact_download.DownloadCancelled("stopped"), True), (RuntimeError("GET refused"), False)],
        ids=["stopped", "failed"],
    )
    def test_a_stopped_or_failed_execution_log_download_reports_the_work_before_it(
        self, tmp_path, monkeypatch, error, cancelled
    ):
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        files = {
            "job-1:sampling:0/execution.jsonl": (b"one\n", "m1"),
            "job-1:sampling:0/server.log": (b"server\n", "s1"),
        }
        c, connection = self._setup(monkeypatch, files)
        c._metric_emitter = MagicMock()
        now = [1000.0]
        monkeypatch.setattr(nc.time, "monotonic", lambda: now[0])

        def get(source):  # each GET takes 2 s; server.log's, the second, ends the download
            now[0] += 2.0
            if source.endswith("/server.log"):
                raise error

        connection.on_get = get
        with pytest.raises(type(error)):
            c.download_execution_logs("job-1", tmp_path)

        (value,) = self._events(c)
        fields = ("success", "artifact", "file_count", "files_skipped", "bytes_written", "cancelled", "fetch_ms")
        assert {key: value[key] for key in fields} == {
            "success": False, "artifact": "execution_logs", "file_count": 2, "files_skipped": 0,
            "bytes_written": 4, "cancelled": cancelled, "fetch_ms": 4000.0}

    # Every field a download_execution_logs event has, for a failure before any work.
    _NOTHING_DONE = {
        "success": False, "artifact": "execution_logs", "file_count": 0, "files_skipped": 0,
        "bytes_written": 0, "list_ms": 0.0, "fetch_ms": 0.0, "artifact_retry_count": 0,
        "transfer_timeout_count": 0, "cancelled": False}

    def test_a_failed_run_lookup_is_still_an_execution_log_download_event(self, tmp_path, monkeypatch):
        c, connection = self._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"one\n", "m1")})
        c.max_retries = 0
        c._metric_emitter = MagicMock()
        c._session.get.return_value = _make_error_response({"message": "experiment run unavailable"}, 404)

        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.download_execution_logs("job-1", tmp_path)

        assert connection.commands == []  # it failed before listing anything
        (value,) = self._events(c)
        assert {key: value[key] for key in self._NOTHING_DONE} == self._NOTHING_DONE

    @pytest.mark.parametrize(
        ("arguments", "error"),
        [
            ({"sub_job_ids": "job-1:sampling:0"}, TypeError),
            ({"sub_job_ids": []}, ValueError),
            ({"transfer_timeout_seconds": 0}, ValueError),
        ],
        ids=["one-string", "no-sub-jobs", "no-time"],
    )
    def test_a_download_refused_for_its_arguments_is_still_an_execution_log_download_event(
        self, tmp_path, monkeypatch, arguments, error
    ):
        c, connection = self._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"one\n", "m1")})
        c._metric_emitter = MagicMock()

        with pytest.raises(error):
            c.download_execution_logs("job-1", tmp_path, **arguments)

        assert (c._session.get.call_count, connection.commands) == (0, [])  # refused before any work
        (value,) = self._events(c)
        assert {key: value[key] for key in self._NOTHING_DONE} == self._NOTHING_DONE
        assert c._metric_emitter.emit.call_args.kwargs["attributes"]["error.type"] == error.__name__

    def test_a_download_for_a_sub_job_without_execution_logs_still_times_its_listing(self, tmp_path, monkeypatch):
        # The listing finds which sub-jobs have logs, so a missing one fails after it.
        c, connection = self._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"one\n", "m1")})
        c._metric_emitter = MagicMock()
        now = [1000.0]
        monkeypatch.setattr(nc.time, "monotonic", lambda: now[0])
        listing = c._list_experiment_artifact_files

        def slow_list(*args, **kwargs):  # each of its two LISTs takes 0.5 s
            now[0] += 0.5
            return listing(*args, **kwargs)

        monkeypatch.setattr(c, "_list_experiment_artifact_files", slow_list)
        with pytest.raises(nc.NoChunksError, match="job-1:training:0"):
            c.download_execution_logs("job-1", tmp_path, sub_job_ids=["job-1:training:0"])

        assert self._gets(connection) == []  # nothing fetched
        (value,) = self._events(c)
        assert (value["success"], value["list_ms"], value["fetch_ms"], value["file_count"]) == (
            False, 1000.0, 0.0, 0)
        assert c._metric_emitter.emit.call_args.kwargs["attributes"]["error.type"] == "NoChunksError"

    def test_resume_skips_unchanged_files_and_refetches_changed_ones(self, tmp_path, monkeypatch):
        monkeypatch.delenv(nc.ENABLE_SUCCESS_TELEMETRY_ENV, raising=False)
        files = {
            "job-1:sampling:0/execution.jsonl": (b"one\n", "m1"),  # 4 bytes
            "job-1:sampling:0/server.log": (b"server\n", "s1"),  # 7 bytes
        }
        c, connection = self._setup(monkeypatch, files)
        c._metric_emitter = MagicMock()
        c.download_execution_logs("job-1", tmp_path)
        assert len(self._gets(connection)) == 2

        connection.commands.clear()
        again = c.download_execution_logs("job-1", tmp_path, resume=True)
        assert self._gets(connection) == []
        assert [r["unchanged"] for r in again] == [True, True]

        # The running job rewrote its execution log.
        grown = b"one\ntwo\n"
        uri = self.RUN_URI + "_logs/job-1:sampling:0/execution.jsonl"
        connection.content_by_uri[uri] = grown
        connection.list_rows[self.RUN_URI + "_logs/"][0] = (
            "/versions/RUN_ABC/_logs/job-1:sampling:0/execution.jsonl", len(grown), "m2", "later"
        )
        connection.commands.clear()
        third = c.download_execution_logs("job-1", tmp_path, resume=True)
        (get,) = self._gets(connection)
        assert get.startswith(f"GET '{uri}' ")
        assert (tmp_path / "job-1:sampling:0" / "execution.jsonl").read_bytes() == grown
        assert [r["unchanged"] for r in third] == [False, True]
        # Each call's event says the same: the third kept server.log and wrote the 8 new bytes.
        fields = ("success", "file_count", "files_skipped", "bytes_written")
        assert [tuple(value[key] for key in fields) for value in self._events(c)] == [
            (True, 2, 0, 11), (True, 2, 2, 0), (True, 2, 1, 8)]

    def test_without_resume_every_file_is_fetched_again(self, tmp_path, monkeypatch):
        c, connection = self._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"x\n", "m1")})
        c.download_execution_logs("job-1", tmp_path)
        c.download_execution_logs("job-1", tmp_path)
        assert len(self._gets(connection)) == 2

    def test_resume_refetches_a_locally_edited_file(self, tmp_path, monkeypatch):
        c, connection = self._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"abc\n", "m1")})
        c.download_execution_logs("job-1", tmp_path)
        (tmp_path / "job-1:sampling:0" / "execution.jsonl").write_bytes(b"truncated")
        c.download_execution_logs("job-1", tmp_path, resume=True)
        assert len(self._gets(connection)) == 2
        assert (tmp_path / "job-1:sampling:0" / "execution.jsonl").read_bytes() == b"abc\n"

    def test_a_failed_download_keeps_the_previous_copy(self, tmp_path, monkeypatch):
        c, connection = self._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"old\n", "m1")})
        c.download_execution_logs("job-1", tmp_path)
        saved = tmp_path / "job-1:sampling:0" / "execution.jsonl"

        def fail(_source):
            raise nc._artifact_download.DownloadCancelled("stopped")

        connection.on_get = fail
        with pytest.raises(nc.DownloadCancelled):
            c.download_execution_logs("job-1", tmp_path)
        assert saved.read_bytes() == b"old\n"
        assert not list(saved.parent.glob(".execution-log-*"))

    def test_logs_copy_wins_over_the_checkpoints_copy(self, tmp_path, monkeypatch):
        c, _connection = self._setup(
            monkeypatch,
            {"job-1:training:0/execution.jsonl": (b"logs copy\n", "a")},
            checkpoint_files={
                "job-1:training:0/execution.jsonl": (b"checkpoints copy\n", "b"),
                "job-1:training:0/only-here.log": (b"only\n", "c"),
            },
        )
        result = c.download_execution_logs("job-1", tmp_path)
        assert {r["filename"]: r["artifact_uri"] for r in result} == {
            "execution.jsonl": self.RUN_URI + "_logs/job-1:training:0/execution.jsonl",
            "only-here.log": self.RUN_URI + "checkpoints/_logs/job-1:training:0/only-here.log",
        }
        assert (tmp_path / "job-1:training:0" / "execution.jsonl").read_bytes() == b"logs copy\n"

    def test_sub_job_filter(self, tmp_path, monkeypatch):
        files = {
            "job-1:sampling:0/execution.jsonl": (b"s\n", "a"),
            "job-1:training:0/execution.jsonl": (b"t\n", "b"),
        }
        c, _connection = self._setup(monkeypatch, files)
        (only,) = c.download_execution_logs("job-1", tmp_path, sub_job_ids=["job-1:training:0"])
        assert only["sub_job_id"] == "job-1:training:0"
        assert not (tmp_path / "job-1:sampling:0").exists()
        with pytest.raises(nc.NoChunksError, match="job-1:eval:0.*job-1:sampling:0, job-1:training:0") as raised:
            c.download_execution_logs("job-1", tmp_path, sub_job_ids=["job-1:eval:0"])
        assert raised.value.sub_job_ids == ["job-1:eval:0"]
        with pytest.raises(TypeError):
            c.download_execution_logs("job-1", tmp_path, sub_job_ids="job-1:training:0")

    def test_no_files_reports_it(self, tmp_path, monkeypatch):
        c, _connection = self._setup(monkeypatch, {})
        messages = []
        assert c.download_execution_logs("job-1", tmp_path, progress=messages.append) == []
        assert messages == ["execution logs: found 0 files\n"]

    def test_a_listing_without_size_columns_still_downloads_but_never_skips(self, tmp_path, monkeypatch):
        c, connection = self._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"x\n", "a")})
        connection.list_rows[self.RUN_URI + "_logs/"] = [("/versions/RUN_ABC/_logs/job-1:sampling:0/execution.jsonl",)]
        c.download_execution_logs("job-1", tmp_path, resume=True)
        c.download_execution_logs("job-1", tmp_path, resume=True)
        assert len(self._gets(connection)) == 2

    def test_old_leftover_download_directories_are_removed(self, tmp_path, monkeypatch):
        c, _connection = self._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"x\n", "a")})
        sub = tmp_path / "job-1:sampling:0"
        sub.mkdir()
        old = sub / ".execution-log-old.tmp"
        old.mkdir()
        (old / "partial").write_bytes(b"junk")
        fresh = sub / ".execution-log-fresh.tmp"
        fresh.mkdir()
        hour_ago = time.time() - 2 * nc.CortexTrainingClient._TEMP_ORPHAN_MIN_AGE_SECONDS
        os.utime(old, (hour_ago, hour_ago))
        c.download_execution_logs("job-1", tmp_path)
        assert not old.exists()
        assert fresh.exists()  # possibly another download's, still in use

    def test_the_body_is_never_read_into_memory(self, tmp_path, monkeypatch):
        c, _connection = self._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"x" * 4096, "a")})

        def refuse(*_args, **_kwargs):
            raise AssertionError("an execution log body was read into memory")

        monkeypatch.setattr(Path, "read_text", refuse)
        monkeypatch.setattr(Path, "read_bytes", refuse)
        (entry,) = c.download_execution_logs("job-1", tmp_path)
        assert entry["bytes"] == 4096


def test_execution_log_get_time_limit_scales_with_file_size(tmp_path, monkeypatch):
    setup = TestDownloadExecutionLogs()
    big = 10 * nc.CortexTrainingClient._EXECUTION_LOG_MIN_BYTES_PER_SECOND
    c, connection = setup._setup(monkeypatch, {"job-1:sampling:0/execution.jsonl": (b"x" * big, "m")})
    seen = []
    real = nc._artifact_download.run_bounded

    def spy(operation, *, timeout, cancel, description):
        seen.append((description.split(" ")[0], timeout))
        return real(operation, timeout=timeout, cancel=cancel, description=description)

    monkeypatch.setattr(nc._artifact_download, "run_bounded", spy)
    c.download_execution_logs("job-1", tmp_path, transfer_timeout_seconds=2)
    assert ("GET", 10.0) in seen
    assert all(timeout == 2 for kind, timeout in seen if kind == "LIST")



def test_stall_check_job_lookup_is_tried_once_and_stays_quiet_after_the_download(tmp_path, monkeypatch, caplog):
    c, _connection, _paths, _sleeps, _opened = TestChunkedArtifactDownload()._setup(
        monkeypatch, count=2, artifact_name="metrics"
    )
    calls = []

    def unavailable(*_args, **_kwargs):
        calls.append(time.monotonic())
        return _make_error_response({"message": "control plane unavailable"}, 503)

    c._session.get.side_effect = unavailable
    with caplog.at_level(logging.WARNING):
        result = c.download_metrics("job-1", tmp_path)
        returned = time.time()
        time.sleep(1.0)  # long enough for a retrying lookup to have logged again
    assert [entry["sub_job_id"] for entry in result] == [TestChunkedArtifactDownload.SUB]
    assert len(calls) == 1
    assert [r.getMessage() for r in caplog.records if r.created > returned] == []
    assert not any("retrying" in r.getMessage() for r in caplog.records)



@pytest.mark.parametrize(
    "status",
    ["TERMINATED", "JOB_STATE_TERMINATED", "terminated", "CANCELLED", "canceled", "FAILED", "DONE", "succeeded"],
)
def test_wait_for_job_stops_at_every_ended_state(monkeypatch, status):
    c = _make_client(get_json={"status": status, "reason": "r"})
    monkeypatch.setattr(nc.time, "sleep", lambda *_: (_ for _ in ()).throw(AssertionError("kept polling")))
    with pytest.raises(RuntimeError, match="reached terminal state"):
        c.wait_for_job("job-1")
    assert c._session.get.call_count == 1


def test_wait_for_job_keeps_polling_through_starting_states(monkeypatch):
    monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
    c = _make_client()
    c._session.get.side_effect = [
        _make_response({"status": status}) for status in ("PENDING", "PLACING", "INITIALIZING", "RUNNING")
    ]
    assert c.wait_for_job("job-1")["status"] == "RUNNING"
    assert c._session.get.call_count == 4


class TestCreatedJobBookkeeping:
    def test_concurrent_creates_on_one_client_never_raise(self, monkeypatch):
        import sys
        import threading

        c = _make_client()
        counter = iter(range(10**9))
        lock = threading.Lock()

        def post(*_args, **_kwargs):
            with lock:
                n = next(counter)
            return _make_response({"job_id": f"job-{n}"})

        c._session.post.side_effect = post
        # A large map makes each pruning pass long enough to be interrupted.
        c._recently_created_jobs.update({f"old-{i}": time.monotonic() for i in range(20000)})
        errors, created = [], []
        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            def worker():
                try:
                    for _ in range(200):
                        created.append(c.create_job_from_body(_SAMPLING_BODY)["job_id"])
                except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(60)
        finally:
            sys.setswitchinterval(old_interval)
        assert errors == []
        assert len(set(created)) == 800
        assert all(job_id in c._recently_created_jobs for job_id in created)

    def test_recording_a_create_never_hides_its_result(self):
        c = _make_client(post_json={"job_id": "srv-1"})

        class Broken(dict):
            def items(self):
                raise RuntimeError("boom")

        c._recently_created_jobs = Broken()
        assert c.create_job_from_body(_SAMPLING_BODY) == {"job_id": "srv-1"}

    def test_a_response_without_job_id_falls_back_to_the_requested_one(self):
        c = _make_client(post_json={"status": "PENDING"})
        c.create_job_from_body({**_SAMPLING_BODY, "job_id": "client-chosen"})
        assert "client-chosen" in c._recently_created_jobs



def test_execution_log_filenames_keep_their_subdirectories(tmp_path, monkeypatch):
    setup = TestDownloadExecutionLogs()
    files = {
        "job-1:sampling:0/ray/worker.log": (b"worker\n", "a"),
        "job-1:sampling:0/execution.jsonl": (b"e\n", "b"),
    }
    c, _connection = setup._setup(monkeypatch, files)
    messages = []
    result = c.download_execution_logs("job-1", tmp_path, progress=messages.append)
    assert [(r["filename"], r["saved_path"]) for r in result] == [
        ("execution.jsonl", str(tmp_path / "job-1:sampling:0" / "execution.jsonl")),
        ("ray/worker.log", str(tmp_path / "job-1:sampling:0" / "ray" / "worker.log")),
    ]
    assert (tmp_path / "job-1:sampling:0" / "ray" / "worker.log").read_bytes() == b"worker\n"
    # The same names fetch_execution_logs reports.
    assert [r["filename"] for r in result] == sorted(
        f["filename"] for f in c.fetch_execution_logs("job-1")
    )
    assert any("execution logs: job-1:sampling:0/ray/worker.log: saved" in m for m in messages)
    # Resume finds the nested file's record.
    again = c.download_execution_logs("job-1", tmp_path, resume=True)
    assert [r["unchanged"] for r in again] == [True, True]
