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

"""Offline tests for the Cortex Training CLI."""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import cortex_training.cli.commands as cli


class FakeClient:
    def __init__(self):
        self.download_interrupt = None
        self.download_warning = None
        # The training stream's own failure, so a test can fail one download and not the other.
        self.training_download_interrupt = None
        self.download_options = None
        self.submitted_body = None
        self.status_filter = None
        self.cancelled_job_id = None
        self.waited_job_id = None
        self.forward_backward_job_id = None
        self.forward_backward_payload = None
        self.polled_request = None
        self.stepped_job_id = None
        self.step_learning_rate = None
        self.loaded_job_id = None
        self.load_checkpoint_id = None
        self.load_source_job_id = None
        self.generated_job_id = None
        self.generate_prompts = None
        self.generate_sampling_params = None
        self.generate_routing_key = None
        self.generate_strict = None
        self.weight_sync_job_id = None
        self.weight_sync_source_sub_job_id = None
        self.weight_sync_target_sub_job_ids = None
        self.weight_sync_sub_job_id = None
        self.weight_sync_sub_job_type = None
        self.capacity_requested = False
        self.capacity_hardware = None
        self.capacity_hardware_requests = []
        self.capacity_breakdown_requests = []
        self.capacity_breakdown_errors = {}
        self.checkpoints_job_id = None
        self.jobs = None
        self.models_requested = False
        self.account_jobs = []
        self.account_jobs_pulled = 0
        self.iter_jobs_filters = None
        self.stdout_log_job_id = None
        self.stdout_log_output_dir = None
        self.metrics_job_id = None
        self.metrics_output_dir = None

    def create_job_from_body(self, body):
        self.submitted_body = body
        return {"job_id": body.get("job_id", "server-job")}

    def wait_for_job(self, job_id):
        self.waited_job_id = job_id
        return {"job_id": job_id, "status": "running"}

    def get_job(self, job_id):
        return {"job_id": job_id, "status": "running"}

    def list_jobs(self, status=None):
        self.status_filter = status
        if self.jobs is not None:
            return self.jobs
        return [{"job_id": "j1", "status": status or "running"}]

    def list_models(self):
        self.models_requested = True
        return [
            {"name": "Qwen/Qwen3-0.6B"},
            {"name": "Qwen/Qwen3-8B"},
        ]

    def iter_jobs(self, **filters):
        self.iter_jobs_filters = filters
        for job in self.account_jobs:
            self.account_jobs_pulled += 1
            yield job

    def list_checkpoints(self, job_id):
        self.checkpoints_job_id = job_id
        return [
            {"checkpoint_id": "cp-1"},
            {"checkpoint_id": "cp-2"},
        ]

    def cancel_job(self, job_id):
        self.cancelled_job_id = job_id

    def get_capacity(self, hardware=None):
        self.capacity_requested = True
        self.capacity_hardware = hardware
        self.capacity_hardware_requests.append(hardware)
        return {
            "has_reservation": True,
            "max_total_gpus": 64,
            "reserved_gpus": 64,
            "in_use_gpus": 8,
            "pending_gpus": 16,
            "available_gpus": {
                "H200": 40,
                "B200": 24,
                "B300": 0,
            }.get(hardware, 40),
        }

    def get_capacity_breakdown(
        self, hardware=None, *, dimensions=("job_type", "submitted_by")
    ):
        self.capacity_breakdown_requests.append((hardware, dimensions))
        if hardware in self.capacity_breakdown_errors:
            raise self.capacity_breakdown_errors[hardware]
        return _capacity_with_breakdown(hardware, dimensions)

    def forward_backward(self, job_id, payload):
        self.forward_backward_job_id = job_id
        self.forward_backward_payload = payload
        return "req-fb-1"

    def poll_request(self, job_id, request_id):
        self.polled_request = (job_id, request_id)
        if request_id == "req-generate-1":
            return {"responses": [{"text": "MoE routing sends tokens to experts."}]}
        if request_id == "req-weight-sync-1":
            return {"synced": True}
        if request_id == "req-load-1":
            return {"checkpoint_id": "cp-7"}
        return {"avg_loss": 0.125}

    def step(self, job_id, learning_rate=None):
        self.stepped_job_id = job_id
        self.step_learning_rate = learning_rate
        return "req-step-1"

    def load(
        self,
        job_id,
        checkpoint_id,
        source_job_id=None,
        target_sub_job_id=None,
    ):
        self.loaded_job_id = job_id
        self.load_checkpoint_id = checkpoint_id
        self.load_source_job_id = source_job_id
        self.load_target_sub_job_id = target_sub_job_id
        return "req-load-1"

    def generate(
        self,
        job_id,
        prompts,
        sampling_params=None,
        routing_key=None,
        strict=None,
    ):
        self.generated_job_id = job_id
        self.generate_prompts = prompts
        self.generate_sampling_params = sampling_params
        self.generate_routing_key = routing_key
        self.generate_strict = strict
        return "req-generate-1"

    def weight_sync(
        self,
        job_id,
        source_sub_job_id,
        target_sub_job_ids,
        weight_format=None,
        sub_job_id=None,
        sub_job_type=None,
    ):
        self.weight_sync_job_id = job_id
        self.weight_sync_source_sub_job_id = source_sub_job_id
        self.weight_sync_target_sub_job_ids = target_sub_job_ids
        self.weight_sync_weight_format = weight_format
        self.weight_sync_sub_job_id = sub_job_id
        self.weight_sync_sub_job_type = sub_job_type
        return "req-weight-sync-1"

    def fetch_execution_logs(self, job_id):
        return [
            {
                "sub_job_id": f"{job_id}:training:0",
                "filename": "execution.jsonl",
                "artifact_uri": f"snow://experiment/DB.SCH.EXP/versions/RUN_ABC/checkpoints/_logs/{job_id}:training:0/execution.jsonl",
                "content": '{"a":1}\n',
            },
            {
                "sub_job_id": f"{job_id}:training:0",
                "filename": "server.log",
                "artifact_uri": f"snow://experiment/DB.SCH.EXP/versions/RUN_ABC/checkpoints/_logs/{job_id}:training:0/server.log",
                "content": "server line\n",
            },
            {
                "sub_job_id": f"{job_id}:sampling:0",
                "filename": "execution.jsonl",
                "artifact_uri": f"snow://experiment/DB.SCH.EXP/versions/RUN_ABC/checkpoints/_logs/{job_id}:sampling:0/execution.jsonl",
                "content": '{"b":2}\n',
            },
        ]

    def download_execution_logs(
        self,
        job_id,
        output_dir,
        *,
        resume=False,
        sub_job_ids=None,
        progress=None,
        transfer_timeout_seconds=300.0,
    ):
        self.execution_log_options = {
            "output_dir": output_dir,
            "resume": resume,
            "sub_job_ids": sub_job_ids,
            "transfer_timeout_seconds": transfer_timeout_seconds,
        }
        saved = []
        for log in self.fetch_execution_logs(job_id):
            if sub_job_ids is not None and log["sub_job_id"] not in sub_job_ids:
                continue
            path = output_dir / log["sub_job_id"] / log["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(log["content"], encoding="utf-8")
            if progress is not None:
                progress(f"execution logs: {log['sub_job_id']}/{log['filename']}: saved")
                progress("\n")
            saved.append(
                {
                    "sub_job_id": log["sub_job_id"],
                    "filename": log["filename"],
                    "artifact_uri": log["artifact_uri"],
                    "saved_path": str(path),
                    "bytes": path.stat().st_size,
                    "unchanged": False,
                }
            )
        return saved

    def download_stdout_logs(
        self,
        job_id,
        output_dir,
        *,
        resume=False,
        max_workers=None,
        progress=None,
        sub_job_ids=None,
        strict=False,
        stale_after_seconds=600.0,
        transfer_timeout_seconds=300.0,
    ):
        self.stdout_log_job_id = job_id
        self.stdout_log_output_dir = output_dir
        self.stdout_resume = resume
        self.download_options = {
            "max_workers": max_workers,
            "sub_job_ids": sub_job_ids,
            "strict": strict,
            "stale_after_seconds": stale_after_seconds,
            "transfer_timeout_seconds": transfer_timeout_seconds,
        }
        if self.download_warning:
            logging.getLogger("cortex_training.client.core").warning(self.download_warning)
        if self.download_interrupt:
            raise self.download_interrupt
        if progress is not None:
            progress(f"stdout: [####################] 1/1 {job_id}:training:0")
            progress("\n")
        path = output_dir / f"{job_id}:training:0" / "stdout.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("console\n", encoding="utf-8")
        return [
            {
                "sub_job_id": f"{job_id}:training:0",
                "filename": "stdout.log",
                "saved_path": str(path),
                "chunk_count": 1,
                "first_artifact_uri": "snow://experiment/first.gz",
                "last_artifact_uri": "snow://experiment/first.gz",
            }
        ]

    def download_metrics(
        self,
        job_id,
        output_dir,
        *,
        resume=False,
        max_workers=None,
        progress=None,
        sub_job_ids=None,
        strict=False,
        stale_after_seconds=600.0,
        transfer_timeout_seconds=300.0,
    ):
        self.metrics_job_id = job_id
        self.metrics_output_dir = output_dir
        self.metrics_resume = resume
        self.download_options = {
            "max_workers": max_workers,
            "sub_job_ids": sub_job_ids,
            "strict": strict,
            "stale_after_seconds": stale_after_seconds,
            "transfer_timeout_seconds": transfer_timeout_seconds,
        }
        if self.download_warning:
            logging.getLogger("cortex_training.client.core").warning(self.download_warning)
        if self.download_interrupt:
            raise self.download_interrupt
        if progress is not None:
            progress(f"metrics: [####################] 1/1 {job_id}:training:0")
            progress("\n")
        path = output_dir / f"{job_id}:training:0" / "gpu.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"gpu_utilization":0.5}\n', encoding="utf-8")
        return [
            {
                "sub_job_id": f"{job_id}:training:0",
                "filename": "gpu.jsonl",
                "saved_path": str(path),
                "chunk_count": 1,
                "first_artifact_uri": "snow://experiment/first.gz",
                "last_artifact_uri": "snow://experiment/first.gz",
            }
        ]

    def download_training_metrics(
        self,
        job_id,
        output_dir,
        *,
        resume=False,
        max_workers=None,
        progress=None,
        sub_job_ids=None,
        strict=False,
        stale_after_seconds=600.0,
        transfer_timeout_seconds=300.0,
    ):
        """The training stream ``download-metrics`` fetches beside the GPU one.

        It records its own arguments, so a test can tell the two calls apart,
        and raises or warns as ``download_metrics`` does: the command runs both,
        so a failure of either has to surface.
        """
        self.training_metrics_job_id = job_id
        self.training_metrics_output_dir = output_dir
        self.training_metrics_resume = resume
        self.training_download_options = {
            "max_workers": max_workers,
            "sub_job_ids": sub_job_ids,
            "strict": strict,
            "stale_after_seconds": stale_after_seconds,
            "transfer_timeout_seconds": transfer_timeout_seconds,
        }
        if self.training_download_interrupt:
            raise self.training_download_interrupt
        if progress is not None:
            progress(f"training metrics: [####################] 1/1 {job_id}:training:0")
            progress("\n")
        path = output_dir / f"{job_id}:training:0" / "training.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"kind":"training","step":1}\n', encoding="utf-8")
        return [
            {
                "sub_job_id": f"{job_id}:training:0",
                "filename": "training.jsonl",
                "saved_path": str(path),
                "chunk_count": 1,
                "first_artifact_uri": "snow://experiment/training-first.gz",
                "last_artifact_uri": "snow://experiment/training-first.gz",
            }
        ]


class FakeHTTPError(OSError):
    def __init__(self):
        super().__init__("500 Server Error: Internal Server Error for url: http://test")
        self.response = type(
            "Response",
            (),
            {
                "headers": {"x-snowflake-request-id": "sf-req-1"},
                "text": '{"message":"backend detail"}',
            },
        )()


def _capacity_with_breakdown(hardware, dimensions):
    """A normalized get_capacity_breakdown() result for ``dimensions``."""
    breakdown = {"job_count": 1, "truncated": False, "unsupported_dimensions": []}
    if "job_type" in dimensions:
        breakdown["by_job_type"] = {
            "groups": [
                {"job_type": "TRAINING", "in_use_gpus": 8, "pending_gpus": 0, "sub_job_count": 1},
                {"job_type": "SAMPLING", "in_use_gpus": 0, "pending_gpus": 0, "sub_job_count": 0},
                {"job_type": "LOG_PROBABILITY", "in_use_gpus": 0, "pending_gpus": 0, "sub_job_count": 0},
            ],
        }
    if "submitted_by" in dimensions:
        breakdown["by_submitted_by"] = {
            "groups": [
                {"kind": "USER", "submitted_by": "ALICE", "is_caller": True,
                 "in_use_gpus": 8, "pending_gpus": 0, "job_count": 1},
                {"kind": "OTHER", "submitted_by": None, "is_caller": False,
                 "in_use_gpus": 0, "pending_gpus": 0, "job_count": 0},
                {"kind": "UNATTRIBUTED", "submitted_by": None, "is_caller": False,
                 "in_use_gpus": 0, "pending_gpus": 0, "job_count": 0},
            ],
        }
    return {
        "has_reservation": True,
        "max_total_gpus": 8,
        "reserved_gpus": 8,
        "in_use_gpus": 8,
        "pending_gpus": 0,
        "available_gpus": {"H200": 0, "B200": 4, "B300": 2}.get(hardware, 0),
        "breakdown": breakdown,
    }


def _factory(instances):
    def make_client(_args):
        client = FakeClient()
        instances.append(client)
        return client

    return make_client


def _write_job(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(
        json.dumps(
            {
                "sub_job_configs": [
                    {
                        "job_type": "sampling",
                        "model_name": "gpt2",
                        "inference_config": {"max_seq_len": 128, "n_gpus": 1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_config(tmp_path, data):
    path = tmp_path / "cortex-training.config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _wire_load(data: bytes):
    from cortex_training import wire

    return wire.loads(data)


def _base_args():
    return ["--base-url", "http://test.local", "--database", "DB"]


def test_submit_reads_create_job_json_and_overrides_job_id(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = _write_job(tmp_path)

    rc = cli.main(
        _base_args() + ["submit", str(path), "--job-id", "client-job"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].submitted_body["job_id"] == "client-job"
    assert json.loads(stdout.getvalue()) == {"job_id": "client-job"}


def test_submit_wait_prints_waited_job(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = _write_job(tmp_path)

    rc = cli.main(
        _base_args() + ["submit", str(path), "--wait"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].waited_job_id == "server-job"
    assert json.loads(stdout.getvalue()) == {"job_id": "server-job", "status": "running"}


def test_submit_dry_run_does_not_require_connection(tmp_path, monkeypatch):
    stdout = io.StringIO()
    path = _write_job(tmp_path)
    monkeypatch.setattr(
        cli,
        "_load_cortex_training_client_class",
        lambda: (_ for _ in ()).throw(AssertionError("client loaded")),
    )

    rc = cli.main(["submit", str(path), "--dry-run", "--job-id", "dry"], stdout=stdout)

    assert rc == 0
    assert json.loads(stdout.getvalue())["job_id"] == "dry"


def _write_two_training_job(tmp_path):
    path = tmp_path / "two-training.json"
    training = {
        "job_type": "training",
        "model_name": "gpt2",
        "training_config": {"max_seq_len": 128, "train_batch_size": 1, "n_gpus": 2},
    }
    path.write_text(json.dumps({"sub_job_configs": [training, training]}), encoding="utf-8")
    return path


def test_submit_rejects_two_training_sub_jobs(tmp_path):
    instances = []
    stderr = io.StringIO()
    path = _write_two_training_job(tmp_path)

    rc = cli.main(
        _base_args() + ["submit", str(path)],
        client_factory=_factory(instances),
        stderr=stderr,
    )

    assert rc == 1
    assert "at most one training sub-job is supported per job" in stderr.getvalue()
    assert instances[0].submitted_body is None


def test_submit_dry_run_rejects_two_training_sub_jobs(tmp_path):
    stderr = io.StringIO()
    path = _write_two_training_job(tmp_path)

    rc = cli.main(["submit", str(path), "--dry-run"], stderr=stderr)

    assert rc == 1
    assert "at most one training sub-job is supported per job" in stderr.getvalue()


def _write_lora_job(tmp_path, peft_config):
    path = tmp_path / "lora.json"
    sub_job = {
        "job_type": "training",
        "model_name": "gpt2",
        "training_config": {
            "max_seq_len": 128,
            "train_batch_size": 1,
            "n_gpus": 2,
            "peft_config": peft_config,
        },
    }
    path.write_text(json.dumps({"sub_job_configs": [sub_job]}), encoding="utf-8")
    return path


def test_submit_dry_run_rejects_invalid_peft_config(tmp_path):
    stderr = io.StringIO()
    path = _write_lora_job(tmp_path, {"peft_type": "LORA", "target_modules": ["q_proj"]})

    rc = cli.main(["submit", str(path), "--dry-run"], stderr=stderr)

    assert rc == 1
    assert "peft_config.peft_type" in stderr.getvalue()


def test_submit_dry_run_prints_the_normalized_peft_config(tmp_path):
    stdout = io.StringIO()
    path = _write_lora_job(tmp_path, {"peft_type": "Lora", "r": 16, "target_modules": ["q_proj"]})

    rc = cli.main(["submit", str(path), "--dry-run"], stdout=stdout)

    assert rc == 0
    peft = json.loads(stdout.getvalue())["sub_job_configs"][0]["training_config"]["peft_config"]
    assert peft["task_type"] == "CAUSAL_LM"
    assert peft["target_parameters"] == []


def test_submit_dry_run_prints_arctic_provider_as_prime_rl():
    stdout = io.StringIO()
    path = Path(__file__).resolve().parents[1] / "recipes" / "api_quickstart" / "training.json"

    rc = cli.main(["submit", str(path), "--dry-run"], stdout=stdout)

    assert rc == 0
    config = json.loads(stdout.getvalue())["sub_job_configs"][0]["training_config"]
    assert config["model_provider"] == "prime_rl"


# Spellings a caller could plausibly put in job JSON for the unsupported
# log-probability type.
LOG_PROBABILITY_JOB_TYPES = [
    "log_probability",
    "LOG_PROBABILITY",
    "JOB_TYPE_LOG_PROBABILITY",
    "log_prob",
    " log_probability ",
]


def _write_log_probability_job(tmp_path, job_type):
    path = tmp_path / "log-probability.json"
    path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("job_type", LOG_PROBABILITY_JOB_TYPES)
def test_submit_dry_run_rejects_log_probability_sub_jobs(tmp_path, job_type):
    stderr = io.StringIO()
    path = _write_log_probability_job(tmp_path, job_type)

    rc = cli.main(["submit", str(path), "--dry-run"], stderr=stderr)

    assert rc == 1
    assert (
        "sub_job_configs[1].job_type: log_probability sub-jobs are not currently supported"
        in stderr.getvalue()
    )


def test_submit_rejects_log_probability_sub_jobs(tmp_path):
    instances = []
    stderr = io.StringIO()
    path = _write_log_probability_job(tmp_path, "log_probability")

    rc = cli.main(
        _base_args() + ["submit", str(path)],
        client_factory=_factory(instances),
        stderr=stderr,
    )

    assert rc == 1
    assert (
        "sub_job_configs[1].job_type: log_probability sub-jobs are not currently supported"
        in stderr.getvalue()
    )
    assert instances[0].submitted_body is None


# The CLI duplicates these so `submit --dry-run` never imports the client; keep
# the two copies from drifting apart.
def test_cli_log_probability_rejection_matches_client():
    from cortex_training.client import core as client

    assert (
        cli._LOG_PROBABILITY_JOB_TYPE_ALIASES
        == client._LOG_PROBABILITY_JOB_TYPE_ALIASES
    )
    assert (
        cli._UNSUPPORTED_LOG_PROBABILITY_MESSAGE
        == client._UNSUPPORTED_LOG_PROBABILITY_MESSAGE
    )


def test_list_prints_jobs_with_status_filter():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["list", "--status", "running"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    # The schema route mishandles comma lists and groups, so the CLI asks it for
    # every job and filters the rows itself.
    assert instances[0].status_filter is None
    assert json.loads(stdout.getvalue()) == {
        "jobs": [{"job_id": "j1", "status": "running"}],
    }


_STATUS_ROWS = [
    {"job_id": status.lower(), "status": status}
    for status in (
        "PENDING",
        "PLACING",
        "INITIALIZING",
        "RUNNING",
        "FAILED",
        "CANCELLED",
        "TERMINATED",
        "UNKNOWN",
    )
]

_SUBMITTED_ROWS = [
    {"job_id": "upper", "submitted_by": "JDOE"},
    {"job_id": "lower", "submitted_by": "jdoe"},
    {"job_id": "quoted", "submitted_by": 'J"DOE'},
    {"job_id": "other", "submitted_by": "ASMITH"},
    {"job_id": "anonymous"},
]

_DATED_ROWS = [
    {"job_id": "before", "created_at": "2026-09-30T23:59:59Z"},
    {"job_id": "midnight", "created_at": "2026-10-01T00:00:00Z"},
    {"job_id": "noon", "created_at": "2026-10-01T12:00:00.250Z"},
    {"job_id": "next-day", "created_at": "2026-10-02T00:00:00+00:00"},
]


def _list_ids(argv, *, jobs=None, account_jobs=()):
    """Run `list ARGV` on one FakeClient; return it and the printed job ids."""
    client = FakeClient()
    client.jobs = jobs
    client.account_jobs = list(account_jobs)
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["list", *argv],
        client_factory=lambda _args: client,
        stdout=stdout,
    )

    assert rc == 0
    return client, [job["job_id"] for job in json.loads(stdout.getvalue())["jobs"]]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("pending", {"pending"}),
        ("placing", {"placing"}),
        ("active", {"pending", "placing", "initializing", "running"}),
        ("done", {"cancelled", "terminated"}),
        (" Running , FAILED ", {"running", "failed"}),
        ("done,terminated", {"cancelled", "terminated"}),
        (
            "active,failed,done",
            {
                "pending",
                "placing",
                "initializing",
                "running",
                "failed",
                "cancelled",
                "terminated",
            },
        ),
    ],
)
def test_list_status_accepts_words_groups_and_comma_lists(status, expected):
    client, ids = _list_ids(["--status", status], jobs=_STATUS_ROWS)

    assert client.status_filter is None
    assert set(ids) == expected


@pytest.mark.parametrize("scope", [[], ["--all"]])
@pytest.mark.parametrize(
    "status", ["unknown", "completed", "running,", "job_state_running"]
)
def test_list_rejects_unknown_status_before_building_a_client(capsys, scope, status):
    instances = []

    with pytest.raises(SystemExit) as exc:
        cli.main(
            _base_args() + ["list", *scope, "--status", status],
            client_factory=_factory(instances),
        )

    assert exc.value.code == 2
    assert "unknown status" in capsys.readouterr().err
    # Building a client can open a Snowflake login.
    assert instances == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        # Enum names and aliases a server can answer with, not only the words
        # --status takes.
        ("running", {"enum", "plain"}),
        ("active", {"enum", "plain", "queued", "creating"}),
        ("done", {"canceled", "completed", "succeeded"}),
        ("failed", {"error"}),
        # UNKNOWN is deliberately matched by no word.
        ("active,done,failed", {"enum", "plain", "queued", "creating",
                                "canceled", "completed", "succeeded", "error"}),
    ],
)
def test_list_status_matches_enum_names_and_aliases(status, expected):
    rows = [
        {"job_id": "enum", "status": "JOB_STATE_RUNNING"},
        {"job_id": "plain", "status": "running"},
        {"job_id": "queued", "status": "queued"},
        {"job_id": "creating", "status": "creating"},
        {"job_id": "canceled", "status": "canceled"},
        {"job_id": "completed", "status": "completed"},
        {"job_id": "succeeded", "status": "succeeded"},
        {"job_id": "error", "status": "error"},
        {"job_id": "unknown", "status": "UNKNOWN"},
    ]

    _, ids = _list_ids(["--status", status], jobs=rows)

    assert set(ids) == expected


def test_list_status_done_includes_legacy_done_rows():
    _, ids = _list_ids(
        ["--status", "done"],
        jobs=[
            {"job_id": "legacy", "status": "DONE"},
            {"job_id": "cancelled", "status": "CANCELLED"},
            {"job_id": "failed", "status": "FAILED"},
        ],
    )

    assert set(ids) == {"legacy", "cancelled"}


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        # As on the server: unquoted means uppercase, quoted means exact.
        ("jdoe", {"upper"}),
        ("JDoe", {"upper"}),
        ('"jdoe"', {"lower"}),
        ('"J""DOE"', {"quoted"}),
        ("asmith", {"other"}),
    ],
)
def test_list_user_follows_sql_identifier_rules(user, expected):
    _, ids = _list_ids(["--user", user], jobs=_SUBMITTED_ROWS)

    assert set(ids) == expected


@pytest.mark.parametrize("scope", [[], ["--all"]])
@pytest.mark.parametrize("user", ["", "  "])
def test_list_rejects_an_empty_user(capsys, scope, user):
    instances = []

    with pytest.raises(SystemExit) as exc:
        cli.main(
            _base_args() + ["list", *scope, "--user", user],
            client_factory=_factory(instances),
        )

    assert exc.value.code == 2
    assert "--user must not be empty" in capsys.readouterr().err
    assert instances == []


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--since", "2026-10-01"], ["midnight", "noon", "next-day"]),
        (["--until", "2026-09-30"], ["before"]),
        # --until keeps the whole day it names.
        (["--until", "2026-10-01"], ["before", "midnight", "noon"]),
        (["--since", "2026-10-01", "--until", "2026-10-01"], ["midnight", "noon"]),
    ],
)
def test_list_since_and_until_keep_whole_utc_days(argv, expected):
    _, ids = _list_ids(argv, jobs=_DATED_ROWS + [{"job_id": "undated"}])

    assert ids == expected


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset")
@pytest.mark.parametrize("zone", ["America/Los_Angeles", "Asia/Tokyo"])
def test_list_dates_are_utc_days_in_any_local_zone(monkeypatch, zone):
    monkeypatch.setenv("TZ", zone)
    time.tzset()
    try:
        _, ids = _list_ids(
            ["--since", "2026-10-01", "--until", "2026-10-01"], jobs=_DATED_ROWS
        )
    finally:
        monkeypatch.undo()
        time.tzset()

    assert ids == ["midnight", "noon"]


def test_list_dates_read_created_at_with_any_fraction_precision():
    jobs = [
        {"job_id": "five", "created_at": "2026-10-01T01:00:00.12345Z"},
        {"job_id": "nine", "created_at": "2026-10-01T02:00:00.123456789Z"},
    ]

    _, ids = _list_ids(["--since", "2026-10-01", "--until", "2026-10-01"], jobs=jobs)

    assert ids == ["five", "nine"]


@pytest.mark.parametrize("scope", [[], ["--all"]])
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--since", "yesterday"], "expected a YYYY-MM-DD date, got 'yesterday'"),
        (["--until", "10/01/2026"], "expected a YYYY-MM-DD date"),
        # Dates only: a time of day, or a compact date, is refused.
        (["--since", "2026-10-01T12:00:00Z"], "expected a YYYY-MM-DD date"),
        (["--since", "20261001"], "expected a YYYY-MM-DD date"),
        (["--until", "2026-02-30"], "expected a YYYY-MM-DD date"),
        (
            ["--since", "2026-10-02", "--until", "2026-10-01"],
            "--since must not be later than --until",
        ),
        (["--until", "9999-12-31"], "--until must be earlier than 9999-12-31"),
    ],
)
def test_list_rejects_bad_dates_before_building_a_client(capsys, scope, argv, message):
    instances = []

    with pytest.raises(SystemExit) as exc:
        cli.main(
            _base_args() + ["list", *scope, *argv],
            client_factory=_factory(instances),
        )

    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert instances == []


def test_list_limit_keeps_the_newest_matching_jobs_printed_last():
    jobs = [
        {**row, "status": status}
        for row, status in zip(_DATED_ROWS, ("RUNNING", "RUNNING", "FAILED", "RUNNING"))
    ]

    _, ids = _list_ids(["--limit", "2"], jobs=jobs)
    assert ids == ["noon", "next-day"]

    _, ids = _list_ids(["--status", "running", "--limit", "2"], jobs=jobs)
    assert ids == ["midnight", "next-day"]


def test_list_limit_drops_undated_jobs_before_newer_dated_ones():
    # A row whose live status could not be read carries no created_at.
    jobs = [{"job_id": "undated-a"}, {"job_id": "undated-b"}] + _DATED_ROWS

    _, ids = _list_ids(["--limit", "2"], jobs=jobs)
    assert ids == ["noon", "next-day"]

    # Every dated job outranks both undated rows, however old it is.
    _, ids = _list_ids(["--limit", "4"], jobs=jobs)
    assert ids == ["before", "midnight", "noon", "next-day"]

    # Undated rows fill what the dated ones leave, newest first, and still
    # print after them.
    _, ids = _list_ids(["--limit", "5"], jobs=jobs)
    assert ids == ["before", "midnight", "noon", "next-day", "undated-a"]


def test_list_limit_keeps_the_first_undated_jobs_when_none_are_dated():
    # Nothing is dated, so the server's newest-first order is all there is.
    jobs = [{"job_id": name} for name in ("newest", "middle", "oldest")]

    _, ids = _list_ids(["--limit", "2"], jobs=jobs)

    assert ids == ["middle", "newest"]


@pytest.mark.parametrize("limit", ["0", "-1", "two"])
def test_list_rejects_a_limit_below_one(limit):
    with pytest.raises(SystemExit):
        cli.parse_args(_base_args() + ["list", "--limit", limit])


def test_list_all_reads_the_account_and_prints_it_oldest_first():
    newest_first = list(reversed(_DATED_ROWS))

    client, ids = _list_ids(
        ["--all"], jobs=[{"job_id": "schema-row"}], account_jobs=newest_first
    )

    assert ids == ["before", "midnight", "noon", "next-day"]
    assert client.iter_jobs_filters == {
        "status": None,
        "submitted_by": None,
        "created_after": None,
        "created_before": None,
        "page_size": 100,
    }


def test_list_all_sends_every_filter_to_the_server():
    rows = [
        {"job_id": "b", "status": "FAILED", "created_at": "2026-10-01T02:00:00Z"},
        {"job_id": "a", "status": "RUNNING", "created_at": "2026-10-01T01:00:00Z"},
    ]

    client, ids = _list_ids(
        [
            "--all",
            "--status",
            "active, FAILED",
            "--user",
            "jdoe",
            "--since",
            "2026-10-01",
            "--until",
            "2026-10-02",
        ],
        account_jobs=rows,
    )

    assert client.iter_jobs_filters == {
        "status": "active,failed",
        "submitted_by": "jdoe",
        "created_after": date(2026, 10, 1),
        # The server's bound is exclusive, so --until sends the day after.
        "created_before": date(2026, 10, 3),
        "page_size": 100,
    }
    assert ids == ["a", "b"]


def test_list_all_limit_stops_reading_after_the_newest_jobs():
    client, ids = _list_ids(
        ["--all", "--limit", "2"], account_jobs=list(reversed(_DATED_ROWS))
    )

    assert ids == ["noon", "next-day"]
    assert client.account_jobs_pulled == 2
    assert client.iter_jobs_filters["page_size"] == 2


def test_list_all_caps_the_page_size_at_the_server_maximum():
    client, _ = _list_ids(["--all", "--limit", "5000"])

    assert client.iter_jobs_filters["page_size"] == 1000


def test_list_all_reports_listing_that_is_not_enabled():
    import requests

    class NotEnabledClient(FakeClient):
        def iter_jobs(self, **filters):
            response = requests.Response()
            response.status_code = 404
            raise requests.HTTPError(
                "account-level job listing is not enabled for this account",
                response=response,
            )
            yield

    stdout = io.StringIO()
    stderr = io.StringIO()

    rc = cli.main(
        _base_args() + ["list", "--all"],
        client_factory=lambda _args: NotEnabledClient(),
        stdout=stdout,
        stderr=stderr,
    )

    assert rc == 1
    assert (
        "error: account-level job listing is not enabled for this account"
        in stderr.getvalue()
    )
    assert stdout.getvalue() == ""


def test_list_prints_latest_jobs_last():
    instances = []
    stdout = io.StringIO()
    jobs = [
        {
            "job_id": "latest",
            "status": "running",
            "created_at": "2026-06-13T00:00:00Z",
        },
        {
            "job_id": "mid",
            "status": "done",
            "created_at": "2026-06-12T00:00:00Z",
        },
        {
            "job_id": "oldest",
            "status": "done",
            "created_at": "2026-06-10T00:00:00Z",
        },
    ]

    def make_client(_args):
        client = FakeClient()
        client.jobs = jobs
        instances.append(client)
        return client

    rc = cli.main(_base_args() + ["list"], client_factory=make_client, stdout=stdout)

    assert rc == 0
    assert json.loads(stdout.getvalue()) == {
        "jobs": [
            {
                "job_id": "oldest",
                "status": "done",
                "created_at": "2026-06-10T00:00:00Z",
            },
            {
                "job_id": "mid",
                "status": "done",
                "created_at": "2026-06-12T00:00:00Z",
            },
            {
                "job_id": "latest",
                "status": "running",
                "created_at": "2026-06-13T00:00:00Z",
            },
        ],
    }


def test_list_sorts_missing_created_at_after_dated_jobs():
    stdout = io.StringIO()

    def make_client(_args):
        client = FakeClient()
        client.jobs = [
            {"job_id": "missing", "status": "running"},
            {
                "job_id": "latest",
                "status": "running",
                "created_at": "2026-06-13T00:00:00Z",
            },
            {
                "job_id": "oldest",
                "status": "done",
                "created_at": "2026-06-10T00:00:00Z",
            },
        ]
        return client

    rc = cli.main(_base_args() + ["list"], client_factory=make_client, stdout=stdout)

    assert rc == 0
    assert [job["job_id"] for job in json.loads(stdout.getvalue())["jobs"]] == [
        "oldest",
        "latest",
        "missing",
    ]


def test_models_dispatches_without_job_id_and_prints_models():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["models"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].models_requested is True
    assert json.loads(stdout.getvalue()) == {
        "models": [
            {"name": "Qwen/Qwen3-0.6B"},
            {"name": "Qwen/Qwen3-8B"},
        ],
    }


def test_models_help_explains_model_availability(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_args(["models", "--help"])

    assert exc_info.value.code == 0
    assert "List active models configured for new jobs." in capsys.readouterr().out


def test_get_prints_job():
    stdout = io.StringIO()

    rc = cli.main(_base_args() + ["get", "j1"], client_factory=_factory([]), stdout=stdout)

    assert rc == 0
    assert json.loads(stdout.getvalue()) == {"job_id": "j1", "status": "running"}


def test_checkpoints_lists_checkpoints_for_job():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["checkpoints", "j1"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].checkpoints_job_id == "j1"
    assert json.loads(stdout.getvalue()) == {
        "checkpoints": [
            {"checkpoint_id": "cp-1"},
            {"checkpoint_id": "cp-2"},
        ],
    }


def test_capacity_prints_account_gpu_usage():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["capacity"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].capacity_requested is True
    assert instances[0].capacity_hardware_requests == ["H200", "B200", "B300"]
    assert json.loads(stdout.getvalue()) == {
        "capacity_by_hardware": {
            "H200": {
                "has_reservation": True,
                "max_total_gpus": 64,
                "reserved_gpus": 64,
                "in_use_gpus": 8,
                "pending_gpus": 16,
                "available_gpus": 40,
            },
            "B200": {
                "has_reservation": True,
                "max_total_gpus": 64,
                "reserved_gpus": 64,
                "in_use_gpus": 8,
                "pending_gpus": 16,
                "available_gpus": 24,
            },
            "B300": {
                "has_reservation": True,
                "max_total_gpus": 64,
                "reserved_gpus": 64,
                "in_use_gpus": 8,
                "pending_gpus": 16,
                "available_gpus": 0,
            },
        },
        "unsupported_hardware": [],
    }


def _unsupported_hardware_error(hardware: str, *, code: str = "517604", message: str | None = None):
    import requests

    response = requests.Response()
    response.status_code = 409
    response._content = json.dumps(
        {
            "code": code,
            "message": message
            or f'Cortex training getCapacity failed: unsupported hardware "{hardware}": '
            "supported hardware is H200, B300",
        }
    ).encode()
    return requests.HTTPError("409 Client Error: Conflict", response=response)


class _PartialHardwareClient(FakeClient):
    def __init__(self, failures):
        super().__init__()
        self._failures = failures

    def get_capacity(self, hardware=None):
        if hardware in self._failures:
            self.capacity_hardware_requests.append(hardware)
            raise self._failures[hardware]
        return super().get_capacity(hardware)


def test_capacity_lists_unsupported_hardware_separately():
    client = _PartialHardwareClient({"B200": _unsupported_hardware_error("B200")})
    stdout = io.StringIO()

    rc = cli.main(_base_args() + ["capacity"], client_factory=lambda _args: client, stdout=stdout)

    assert rc == 0
    assert client.capacity_hardware_requests == ["H200", "B200", "B300"]
    result = json.loads(stdout.getvalue())
    assert set(result["capacity_by_hardware"]) == {"H200", "B300"}
    assert result["capacity_by_hardware"]["H200"]["available_gpus"] == 40
    assert result["unsupported_hardware"] == ["B200"]


def test_capacity_still_fails_on_other_errors():
    other = _unsupported_hardware_error("B200", code="517604", message="Cortex training getCapacity failed: timeout")
    client = _PartialHardwareClient({"B200": other})
    stderr = io.StringIO()

    rc = cli.main(
        _base_args() + ["capacity"],
        client_factory=lambda _args: client,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert rc == 1
    assert "timeout" in stderr.getvalue()


def test_capacity_errors_when_no_hardware_is_supported():
    client = _PartialHardwareClient(
        {hardware: _unsupported_hardware_error(hardware) for hardware in ("H200", "B200", "B300")}
    )
    stderr = io.StringIO()

    rc = cli.main(
        _base_args() + ["capacity"],
        client_factory=lambda _args: client,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert rc == 1
    assert "unsupported hardware" in stderr.getvalue()


def test_capacity_with_unsupported_hardware_flag_reports_service_error():
    client = _PartialHardwareClient({"B200": _unsupported_hardware_error("B200")})
    stderr = io.StringIO()

    rc = cli.main(
        _base_args() + ["capacity", "--hardware", "B200"],
        client_factory=lambda _args: client,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert rc == 1
    assert "supported hardware is H200, B300" in stderr.getvalue()


def test_capacity_passes_hardware():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["capacity", "--hardware", "B300"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].capacity_hardware == "B300"
    assert instances[0].capacity_hardware_requests == ["B300"]


def test_capacity_rejects_unknown_hardware():
    with pytest.raises(SystemExit):
        cli.main(
            _base_args() + ["capacity", "--hardware", "A10"],
            client_factory=_factory([]),
            stdout=io.StringIO(),
        )


_CAPACITY_HARDWARE = ("H200", "B200", "B300")


def _run_capacity(argv, *, client=None):
    client = client or FakeClient()
    stdout = io.StringIO()
    stderr = io.StringIO()

    rc = cli.main(
        _base_args() + argv,
        client_factory=lambda _args: client,
        stdout=stdout,
        stderr=stderr,
    )

    return rc, client, stdout.getvalue(), stderr.getvalue()


@pytest.mark.parametrize(
    "hardware_args", [[], ["--hardware", "B300"]], ids=["fan-out", "single"]
)
def test_capacity_without_breakdown_prints_get_capacity_unchanged(hardware_args):
    rc, client, out, err = _run_capacity(["capacity", *hardware_args])

    if hardware_args:
        expected = FakeClient().get_capacity("B300")
        lookups = ["B300"]
    else:
        expected = {
            "capacity_by_hardware": {
                hardware: FakeClient().get_capacity(hardware)
                for hardware in _CAPACITY_HARDWARE
            },
            "unsupported_hardware": [],
        }
        lookups = list(_CAPACITY_HARDWARE)
    assert rc == 0
    assert err == ""
    assert client.capacity_breakdown_requests == []
    assert client.capacity_hardware_requests == lookups
    assert out == json.dumps(expected, indent=2, sort_keys=True) + "\n"


def test_capacity_help_lists_breakdown(capsys, monkeypatch):
    # Wide enough that argparse does not wrap the help text mid-phrase.
    monkeypatch.setenv("COLUMNS", "200")

    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["capacity", "--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--breakdown {job-type,submitted-by}" in help_text
    assert (
        "by sub-job type (job-type) or by submitting user (submitted-by)" in help_text
    )


@pytest.mark.parametrize("choice", ["owner-role", "job_type", "JOB-TYPE"])
def test_capacity_breakdown_rejects_unknown_choice(choice):
    with pytest.raises(SystemExit):
        cli.parse_args(_base_args() + ["capacity", "--breakdown", choice])


# The CLI spells each dimension with a hyphen; it must send the client's name.
def test_cli_breakdown_choices_match_client_dimensions():
    from cortex_training.client import core as client

    assert (
        tuple(cli._CAPACITY_BREAKDOWN_CHOICES.values())
        == client._CAPACITY_BREAKDOWN_DIMENSIONS
    )


@pytest.mark.parametrize(
    "breakdown_args, dimensions",
    [
        (["--breakdown", "job-type"], ("job_type",)),
        (["--breakdown", "submitted-by"], ("submitted_by",)),
        (
            ["--breakdown", "job-type", "--breakdown", "submitted-by"],
            ("job_type", "submitted_by"),
        ),
        (
            ["--breakdown", "submitted-by", "--breakdown", "job-type"],
            ("submitted_by", "job_type"),
        ),
    ],
    ids=["job-type", "submitted-by", "both", "both-reversed"],
)
def test_capacity_breakdown_fans_out_with_wire_dimensions(breakdown_args, dimensions):
    rc, client, out, err = _run_capacity(["capacity", *breakdown_args])

    assert rc == 0
    assert err == ""
    assert client.capacity_hardware_requests == []
    assert client.capacity_breakdown_requests == [
        (hardware, dimensions) for hardware in _CAPACITY_HARDWARE
    ]
    assert json.loads(out) == {
        "capacity_by_hardware": {
            hardware: _capacity_with_breakdown(hardware, dimensions)
            for hardware in _CAPACITY_HARDWARE
        },
        "unsupported_hardware": [],
    }


def test_capacity_breakdown_lists_unsupported_hardware_separately():
    client = FakeClient()
    client.capacity_breakdown_errors["B200"] = _unsupported_hardware_error("B200")

    rc, client, out, err = _run_capacity(
        ["capacity", "--breakdown", "job-type"], client=client
    )

    assert rc == 0
    assert err == ""
    result = json.loads(out)
    assert set(result["capacity_by_hardware"]) == {"H200", "B300"}
    assert result["unsupported_hardware"] == ["B200"]


def test_capacity_breakdown_single_hardware_prints_compact_sorted_json():
    rc, client, out, err = _run_capacity(
        ["--compact", "capacity", "--hardware", "B300", "--breakdown", "submitted-by"]
    )

    assert rc == 0
    assert err == ""
    assert client.capacity_hardware_requests == []
    assert client.capacity_breakdown_requests == [("B300", ("submitted_by",))]
    expected = cli._breakdown_last(_capacity_with_breakdown("B300", ("submitted_by",)))
    assert out == json.dumps(expected, separators=(",", ":")) + "\n"
    assert list(json.loads(out))[-1] == "breakdown"


@pytest.mark.parametrize("hardware", _CAPACITY_HARDWARE)
def test_capacity_breakdown_fails_on_the_first_lookup_error(hardware):
    client = FakeClient()
    client.capacity_breakdown_errors = {hardware: _http_error(503)}

    rc, _, out, err = _run_capacity(
        ["capacity", "--breakdown", "job-type"], client=client
    )

    assert rc == 1
    assert out == ""
    assert err.startswith("error: 503 Client Error")
    # As without --breakdown, the first failed lookup fails the command.
    looked_up = [requested for requested, _ in client.capacity_breakdown_requests]
    assert looked_up == list(
        _CAPACITY_HARDWARE[: _CAPACITY_HARDWARE.index(hardware) + 1]
    )
    assert client.capacity_hardware_requests == []


def test_capacity_breakdown_prints_the_real_client_result_with_sorted_keys():
    from cortex_training.client import core as client_module

    client = client_module.CortexTrainingClient(
        base_url="http://test.local", database="DB", schema="SCH"
    )
    client._session = MagicMock()
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "has_reservation": True,
        "max_total_gpus": 16,
        "reserved_gpus": 16,
        "in_use_gpus": 12,
        "available_gpus": 4,
        "breakdown": {
            "unsupported_dimensions": [],
            "job_count": 2,
            "truncated": False,
            "by_job_type": {"status": "DISABLED", "groups": []},
            "by_submitted_by": {
                "status": "OK",
                "groups": [
                    {"kind": "USER", "submitted_by": "ALICE", "is_caller": True,
                     "in_use_gpus": 8, "pending_gpus": 0, "job_count": 1},
                    {"kind": "OTHER", "in_use_gpus": 4, "pending_gpus": 0, "job_count": 1},
                    {"kind": "UNATTRIBUTED", "in_use_gpus": 0, "pending_gpus": 0, "job_count": 0},
                ],
            },
        },
    }
    client._session.get.return_value = response

    rc, _, out, err = _run_capacity(
        [
            "capacity",
            "--hardware",
            "H200",
            "--breakdown",
            "job-type",
            "--breakdown",
            "submitted-by",
        ],
        client=client,
    )

    assert rc == 0, err
    client._session.get.assert_called_once_with(
        f"{client._prefix}/capacity",
        timeout=client_module._CAPACITY_REQUEST_TIMEOUT,
        params={"hardware": "H200", "breakdown": "job_type,submitted_by"},
    )
    expected = {
        "has_reservation": True,
        "max_total_gpus": 16,
        "reserved_gpus": 16,
        "in_use_gpus": 12,
        "pending_gpus": 0,
        "available_gpus": 4,
        "breakdown": {
            "job_count": 2,
            "truncated": False,
            "unsupported_dimensions": [],
            "by_job_type": {"status": "DISABLED", "groups": []},
            "by_submitted_by": {
                "status": "OK",
                "groups": [
                    {"kind": "USER", "submitted_by": "ALICE", "is_caller": True,
                     "in_use_gpus": 8, "pending_gpus": 0, "job_count": 1},
                    {"kind": "OTHER", "submitted_by": None, "is_caller": False,
                     "in_use_gpus": 4, "pending_gpus": 0, "job_count": 1},
                    {"kind": "UNATTRIBUTED", "submitted_by": None, "is_caller": False,
                     "in_use_gpus": 0, "pending_gpus": 0, "job_count": 0},
                ],
            },
        },
    }
    assert out == json.dumps(cli._breakdown_last(expected), indent=2) + "\n"
    assert list(json.loads(out))[-1] == "breakdown"


def test_capacity_breakdown_prints_visible_user_rows_in_server_order():
    from cortex_training.client import core as client_module

    client = client_module.CortexTrainingClient(
        base_url="http://test.local", database="DB", schema="SCH"
    )
    client._session = MagicMock()
    response = MagicMock(status_code=200)
    # Scope VISIBLE names other users as well as the caller, ALICE, by
    # in_use_gpus + pending_gpus descending and then by name. Printing with
    # sorted keys must not reorder the rows.
    response.json.return_value = {
        "in_use_gpus": 28,
        "pending_gpus": 16,
        "breakdown": {
            "unsupported_dimensions": [],
            "job_count": 5,
            "truncated": False,
            "by_submitted_by": {
                "status": "OK",
                "groups": [
                    {"kind": "USER", "submitted_by": "CAROL", "is_caller": False,
                     "in_use_gpus": 16, "pending_gpus": 8, "job_count": 2},
                    {"kind": "USER", "submitted_by": "ALICE", "is_caller": True,
                     "in_use_gpus": 8, "pending_gpus": 0, "job_count": 1},
                    {"kind": "USER", "submitted_by": "BOB", "is_caller": False,
                     "in_use_gpus": 0, "pending_gpus": 8, "job_count": 1},
                    {"kind": "OTHER", "in_use_gpus": 4, "pending_gpus": 0, "job_count": 1},
                    {"kind": "UNATTRIBUTED", "in_use_gpus": 0, "pending_gpus": 0, "job_count": 0},
                ],
            },
        },
    }
    client._session.get.return_value = response

    rc, _, out, err = _run_capacity(
        ["--compact", "capacity", "--hardware", "H200", "--breakdown", "submitted-by"],
        client=client,
    )

    assert rc == 0, err
    assert err == ""
    assert client._session.get.call_args.kwargs["params"] == {
        "hardware": "H200",
        "breakdown": "submitted_by",
    }
    assert json.loads(out)["breakdown"]["by_submitted_by"] == {
        "status": "OK",
        "groups": [
            {"kind": "USER", "submitted_by": "CAROL", "is_caller": False,
             "in_use_gpus": 16, "pending_gpus": 8, "job_count": 2},
            {"kind": "USER", "submitted_by": "ALICE", "is_caller": True,
             "in_use_gpus": 8, "pending_gpus": 0, "job_count": 1},
            {"kind": "USER", "submitted_by": "BOB", "is_caller": False,
             "in_use_gpus": 0, "pending_gpus": 8, "job_count": 1},
            {"kind": "OTHER", "submitted_by": None, "is_caller": False,
             "in_use_gpus": 4, "pending_gpus": 0, "job_count": 1},
            {"kind": "UNATTRIBUTED", "submitted_by": None, "is_caller": False,
             "in_use_gpus": 0, "pending_gpus": 0, "job_count": 0},
        ],
    }


def test_cancel_prints_confirmation():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["cancel", "j1"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].cancelled_job_id == "j1"
    assert json.loads(stdout.getvalue()) == {"cancelled": True, "job_id": "j1"}


def test_wait_prints_running_job():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["wait", "j1"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].waited_job_id == "j1"
    assert json.loads(stdout.getvalue()) == {"job_id": "j1", "status": "running"}


def test_fwd_bwd_serializes_payload_and_polls(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = tmp_path / "fwd-bwd.json"
    path.write_text(
        json.dumps(
            {
                "payload": {
                    "kwargs": {
                        "input_ids": {"data": [[1, 2, 3]], "dtype": "long"},
                        "labels": {"data": [[2, 3, -100]], "dtype": "long"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        _base_args() + ["--job", "job-1", "fwd-bwd", str(path)],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    client = instances[0]
    assert client.forward_backward_job_id == "job-1"
    assert client.polled_request == ("job-1", "req-fb-1")
    loaded = _wire_load(client.forward_backward_payload)
    assert loaded["args"] == ()
    assert loaded["kwargs"]["input_ids"].tolist() == [[1, 2, 3]]
    assert loaded["kwargs"]["labels"].tolist() == [[2, 3, -100]]
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "payload_size_bytes": len(client.forward_backward_payload),
        "request_id": "req-fb-1",
        "result": {"avg_loss": 0.125},
    }


def test_fwd_bwd_can_skip_poll(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = tmp_path / "fwd-bwd.json"
    path.write_text(
        json.dumps(
            {
                "poll": False,
                "payload": {"kwargs": {"input_ids": [[1]]}},
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        _base_args() + ["--job", "job-1", "fwd-bwd", str(path)],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].polled_request is None
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "payload_size_bytes": len(instances[0].forward_backward_payload),
        "request_id": "req-fb-1",
    }


def test_fwd_bwd_accepts_job_id_alias(tmp_path):
    instances = []
    path = tmp_path / "fwd-bwd.json"
    path.write_text(
        json.dumps(
            {
                "poll": False,
                "payload": {"kwargs": {"input_ids": [[1]]}},
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        _base_args() + ["--job-id", "job-alias", "fwd-bwd", str(path)],
        client_factory=_factory(instances),
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert instances[0].forward_backward_job_id == "job-alias"


def test_step_uses_default_lr_and_polls():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "step"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].stepped_job_id == "job-1"
    assert instances[0].step_learning_rate == 1e-4
    assert instances[0].polled_request == ("job-1", "req-step-1")
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "learning_rate": 1e-4,
        "request_id": "req-step-1",
        "result": {"avg_loss": 0.125},
    }


def test_step_accepts_lr_override():
    instances = []

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "step", "--lr", "2e-5"],
        client_factory=_factory(instances),
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert instances[0].step_learning_rate == 2e-5


def test_load_uses_checkpoint_id_and_polls():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "load", "cp-7"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    client = instances[0]
    assert client.loaded_job_id == "job-1"
    assert client.load_checkpoint_id == "cp-7"
    assert client.load_source_job_id is None
    assert client.polled_request == ("job-1", "req-load-1")
    assert json.loads(stdout.getvalue()) == {
        "checkpoint_id": "cp-7",
        "job_id": "job-1",
        "request_id": "req-load-1",
        "result": {"checkpoint_id": "cp-7"},
    }


def test_load_accepts_source_job_and_can_skip_poll():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args()
        + [
            "--job-id",
            "job-1",
            "load",
            "cp-7",
            "--source-job-id",
            "source-job",
            "--no-poll",
        ],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    client = instances[0]
    assert client.load_checkpoint_id == "cp-7"
    assert client.load_source_job_id == "source-job"
    assert client.polled_request is None
    assert json.loads(stdout.getvalue()) == {
        "checkpoint_id": "cp-7",
        "job_id": "job-1",
        "request_id": "req-load-1",
        "source_job_id": "source-job",
    }


def test_load_passes_target_sub_job_id():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args()
        + [
            "--job-id",
            "job-1",
            "load",
            "cp-7",
            "--target-sub-job-id",
            "job-1:training:1",
        ],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    client = instances[0]
    assert client.load_target_sub_job_id == "job-1:training:1"
    assert client.polled_request == ("job-1", "req-load-1")
    assert json.loads(stdout.getvalue()) == {
        "checkpoint_id": "cp-7",
        "job_id": "job-1",
        "request_id": "req-load-1",
        "target_sub_job_id": "job-1:training:1",
        "result": {"checkpoint_id": "cp-7"},
    }


def test_load_without_target_sub_job_id_omits_it():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "load", "cp-7", "--no-poll"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].load_target_sub_job_id is None
    assert "target_sub_job_id" not in json.loads(stdout.getvalue())


def test_generate_reads_json_and_polls(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = tmp_path / "generate.json"
    path.write_text(
        json.dumps(
            {
                "prompts": [
                    "Explain mixture-of-experts routing in one sentence.",
                    "Name one benefit of sparse expert models.",
                ],
                "sampling_params": {
                    "max_tokens": 32,
                    "temperature": 0.2,
                    "top_p": 0.9,
                },
                "routing_key": "sample-route",
                "strict": False,
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "generate", str(path)],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    client = instances[0]
    assert client.generated_job_id == "job-1"
    assert client.generate_prompts == [
        "Explain mixture-of-experts routing in one sentence.",
        "Name one benefit of sparse expert models.",
    ]
    assert client.generate_sampling_params == {
        "max_tokens": 32,
        "temperature": 0.2,
        "top_p": 0.9,
    }
    assert client.generate_routing_key == "sample-route"
    assert client.generate_strict is False
    assert client.polled_request == ("job-1", "req-generate-1")
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "prompt_count": 2,
        "request_id": "req-generate-1",
        "result": {"responses": [{"text": "MoE routing sends tokens to experts."}]},
    }


def test_generate_accepts_per_prompt_sampling_params(tmp_path):
    instances = []
    path = tmp_path / "generate.json"
    path.write_text(
        json.dumps(
            {
                "prompts": ["a", "b"],
                "sampling_params": [
                    {"max_tokens": 32, "temperature": 0.2},
                    None,
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "generate", str(path)],
        client_factory=_factory(instances),
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert instances[0].generate_sampling_params == [
        {"max_tokens": 32, "temperature": 0.2},
        None,
    ]


def test_generate_can_skip_poll(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = tmp_path / "generate.json"
    path.write_text(
        json.dumps(
            {
                "poll": False,
                "payload": {
                    "prompts": ["hello"],
                    "sampling_params": {"max_tokens": 4},
                },
            }
        ),
        encoding="utf-8",
    )

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "generate", str(path)],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].polled_request is None
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "prompt_count": 1,
        "request_id": "req-generate-1",
    }


def test_generate_counts_flat_token_list_as_one_prompt(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = tmp_path / "generate.json"
    path.write_text(json.dumps({"poll": False, "prompts": [1, 2, 3]}), encoding="utf-8")

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "generate", str(path)],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].generate_prompts == [1, 2, 3]
    assert json.loads(stdout.getvalue())["prompt_count"] == 1


def test_generate_rejects_sampling_params_list_longer_than_one_flat_prompt(tmp_path):
    path = tmp_path / "generate.json"
    path.write_text(
        json.dumps({"prompts": [1, 2, 3], "sampling_params": [{"max_tokens": 4}, None, None]}),
        encoding="utf-8",
    )
    stderr = io.StringIO()

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "generate", str(path)],
        client_factory=_factory([]),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert rc == 1
    assert "sampling_params list length must match prompts length" in stderr.getvalue()


def test_generate_counts_nested_token_lists_as_a_batch(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = tmp_path / "generate.json"
    path.write_text(json.dumps({"poll": False, "prompts": [[1, 2], [3, 4]]}), encoding="utf-8")

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "generate", str(path)],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert json.loads(stdout.getvalue())["prompt_count"] == 2


def test_weight_sync_defaults_to_training_and_sampling_subjobs():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "weight-sync"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    client = instances[0]
    assert client.weight_sync_job_id == "job-1"
    assert client.weight_sync_source_sub_job_id == "job-1:training:0"
    assert client.weight_sync_target_sub_job_ids == ["job-1:sampling:0"]
    assert client.weight_sync_sub_job_id is None
    assert client.weight_sync_sub_job_type is None
    assert client.polled_request == ("job-1", "req-weight-sync-1")
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "request_id": "req-weight-sync-1",
        "result": {"synced": True},
        "source_sub_job_id": "job-1:training:0",
        "target_sub_job_ids": ["job-1:sampling:0"],
    }


def test_weight_sync_accepts_subjob_overrides_and_can_skip_poll():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args()
        + [
            "--job-id",
            "job-1",
            "weight-sync",
            "--source-sub-job-id",
            "job-1:training:1",
            "--target-sub-job-id",
            "job-1:sampling:0",
            "--target-sub-job-id",
            "job-1:sampling:1",
            "--operation-sub-job-id",
            "job-1:training:route",
            "--operation-sub-job-type",
            "training",
            "--no-poll",
        ],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    client = instances[0]
    assert client.weight_sync_source_sub_job_id == "job-1:training:1"
    assert client.weight_sync_target_sub_job_ids == [
        "job-1:sampling:0",
        "job-1:sampling:1",
    ]
    assert client.weight_sync_sub_job_id == "job-1:training:route"
    assert client.weight_sync_sub_job_type == "training"
    assert client.polled_request is None
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "request_id": "req-weight-sync-1",
        "source_sub_job_id": "job-1:training:1",
        "target_sub_job_ids": ["job-1:sampling:0", "job-1:sampling:1"],
    }


def test_download_log_defaults_to_cwd_when_output_dir_omitted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["download-log", "job-1"],
        client_factory=lambda _: FakeClient(),
        stdout=stdout,
    )

    assert rc == 0
    training_dir = tmp_path / "job-1:training:0"
    sampling_dir = tmp_path / "job-1:sampling:0"
    assert (training_dir / "execution.jsonl").read_text(encoding="utf-8") == '{"a":1}\n'
    assert (training_dir / "server.log").read_text(encoding="utf-8") == "server line\n"
    assert (sampling_dir / "execution.jsonl").read_text(encoding="utf-8") == '{"b":2}\n'

    payload = json.loads(stdout.getvalue())
    assert payload["job_id"] == "job-1"
    assert [(log["sub_job_id"], log["filename"]) for log in payload["logs"]] == [
        ("job-1:training:0", "execution.jsonl"),
        ("job-1:training:0", "server.log"),
        ("job-1:sampling:0", "execution.jsonl"),
    ]
    assert payload["logs"][0]["saved_path"] == str(training_dir / "execution.jsonl")


def test_download_log_writes_each_log_under_sub_job_dir(tmp_path):
    stdout = io.StringIO()

    rc = cli.main(
        _base_args()
        + [
            "download-log",
            "job-1",
            "--output-dir",
            str(tmp_path),
        ],
        client_factory=lambda _: FakeClient(),
        stdout=stdout,
    )

    assert rc == 0
    training_dir = tmp_path / "job-1:training:0"
    sampling_dir = tmp_path / "job-1:sampling:0"
    assert (training_dir / "execution.jsonl").read_text(encoding="utf-8") == '{"a":1}\n'
    assert (training_dir / "server.log").read_text(encoding="utf-8") == "server line\n"
    assert (sampling_dir / "execution.jsonl").read_text(encoding="utf-8") == '{"b":2}\n'

    payload = json.loads(stdout.getvalue())
    assert payload["job_id"] == "job-1"
    assert payload["logs"] == [
        {
            "sub_job_id": "job-1:training:0",
            "filename": "execution.jsonl",
            "artifact_uri": "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/checkpoints/_logs/job-1:training:0/execution.jsonl",
            "saved_path": str(training_dir / "execution.jsonl"),
            "bytes": (training_dir / "execution.jsonl").stat().st_size,
            "unchanged": False,
        },
        {
            "sub_job_id": "job-1:training:0",
            "filename": "server.log",
            "artifact_uri": "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/checkpoints/_logs/job-1:training:0/server.log",
            "saved_path": str(training_dir / "server.log"),
            "bytes": (training_dir / "server.log").stat().st_size,
            "unchanged": False,
        },
        {
            "sub_job_id": "job-1:sampling:0",
            "filename": "execution.jsonl",
            "artifact_uri": "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/checkpoints/_logs/job-1:sampling:0/execution.jsonl",
            "saved_path": str(sampling_dir / "execution.jsonl"),
            "bytes": (sampling_dir / "execution.jsonl").stat().st_size,
            "unchanged": False,
        },
    ]


def test_download_log_stdout_routes_to_reconstruction(tmp_path):
    stdout = io.StringIO()
    client = FakeClient()

    rc = cli.main(
        _base_args()
        + [
            "download-log",
            "job-1",
            "--log-type",
            "stdout",
            "--output-dir",
            str(tmp_path),
        ],
        client_factory=lambda _: client,
        stdout=stdout,
    )

    assert rc == 0
    assert client.stdout_log_job_id == "job-1"
    assert client.stdout_log_output_dir == tmp_path
    assert (tmp_path / "job-1:training:0" / "stdout.log").read_text() == "console\n"
    assert json.loads(stdout.getvalue())["logs"][0]["filename"] == "stdout.log"


def test_download_log_resume_reaches_stdout_download(tmp_path):
    stdout = io.StringIO()
    client = FakeClient()

    rc = cli.main(
        _base_args()
        + ["download-log", "job-1", "--log-type", "stdout", "--resume", "--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=stdout,
    )

    assert rc == 0
    assert client.stdout_resume is True


def test_download_log_execution_options_reach_the_streaming_download(tmp_path):
    client = FakeClient()
    rc = cli.main(
        _base_args()
        + [
            "download-log", "job-1", "--resume", "--sub-job", "job-1:sampling:0",
            "--transfer-timeout", "60", "--output-dir", str(tmp_path),
        ],
        client_factory=lambda _: client,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    assert client.execution_log_options == {
        "output_dir": tmp_path,
        "resume": True,
        "sub_job_ids": ["job-1:sampling:0"],
        "transfer_timeout_seconds": 60.0,
    }
    assert not (tmp_path / "job-1:training:0").exists()


def test_download_log_execution_transfer_timeout_zero_means_no_limit(tmp_path):
    client = FakeClient()
    rc = cli.main(
        _base_args() + ["download-log", "job-1", "--transfer-timeout", "0", "--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    assert client.execution_log_options["transfer_timeout_seconds"] is None


def test_download_metrics_resume_reaches_client(tmp_path):
    client = FakeClient()

    rc = cli.main(
        _base_args() + ["download-metrics", "job-1", "--resume", "--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert client.metrics_resume is True


def test_download_metrics_passes_parallel_and_sub_jobs_and_reports_progress(tmp_path):
    client = FakeClient()
    stdout = io.StringIO()
    stderr = io.StringIO()

    rc = cli.main(
        _base_args()
        + [
            "download-metrics",
            "job-1",
            "--parallel",
            "4",
            "--sub-job",
            "job-1:training:0",
            "--sub-job",
            "job-1:sampling:0",
            "--output-dir",
            str(tmp_path),
        ],
        client_factory=lambda _: client,
        stdout=stdout,
        stderr=stderr,
    )

    assert rc == 0
    assert client.download_options == {
        "max_workers": 4,
        "sub_job_ids": ["job-1:training:0", "job-1:sampling:0"],
        "strict": False,
        "stale_after_seconds": 600.0,
        "transfer_timeout_seconds": 300.0,
    }
    # The command fetches both metric streams, with the same options, and each reports.
    assert client.training_download_options == client.download_options
    assert stderr.getvalue() == (
        "metrics: [####################] 1/1 job-1:training:0\n"
        "training metrics: [####################] 1/1 job-1:training:0\n"
    )
    assert json.loads(stdout.getvalue())["metrics"][0]["filename"] == "gpu.jsonl"


def test_download_metrics_interrupted_in_the_training_stream_offers_resume(tmp_path):
    """The GPU stream has committed its file before the training stream starts, so an
    interrupt there must say output was saved and offer --resume, not claim nothing was."""
    client = FakeClient()
    client.training_download_interrupt = KeyboardInterrupt()
    stderr = io.StringIO()

    rc = cli.main(
        _base_args() + ["download-metrics", "job-1", "--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert rc == 130
    assert (tmp_path / "job-1:training:0" / "gpu.jsonl").exists()
    assert stderr.getvalue().endswith(f"interrupted; {cli._RESUME_HINT}\n")
    assert "before any output was saved" not in stderr.getvalue()


def test_download_metrics_error_in_the_training_stream_still_reports_the_gpu_stream(tmp_path):
    """An error from the training stream fails the command, but the finished GPU stream is
    still printed (so a pipe to jq gets it) and --resume is offered."""
    client = FakeClient()
    client.training_download_interrupt = OSError("transient stage listing failure")
    stdout = io.StringIO()
    stderr = io.StringIO()

    rc = cli.main(
        _base_args() + ["download-metrics", "job-1", "--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=stdout,
        stderr=stderr,
    )

    assert rc == 1
    body = json.loads(stdout.getvalue())
    assert body["metrics"][0]["filename"] == "gpu.jsonl"
    assert body["training_metrics"] is None
    assert "error: transient stage listing failure" in stderr.getvalue()
    assert cli._FAILED_RESUME_HINT in stderr.getvalue()


def test_download_metrics_interrupted_before_any_stream_saved_says_so(tmp_path):
    client = FakeClient()
    client.download_interrupt = KeyboardInterrupt()
    stderr = io.StringIO()

    rc = cli.main(
        _base_args() + ["download-metrics", "job-1", "--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert rc == 130
    assert stderr.getvalue().endswith("interrupted before any output was saved\n")
    assert not hasattr(client, "training_download_options")  # the training stream never started


def test_download_log_stdout_defaults_leave_parallel_to_client(tmp_path):
    client = FakeClient()
    stderr = io.StringIO()

    rc = cli.main(
        _base_args()
        + ["download-log", "job-1", "--log-type", "stdout", "--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert rc == 0
    assert client.download_options == {
        "max_workers": None,
        "sub_job_ids": None,
        "strict": False,
        "stale_after_seconds": 600.0,
        "transfer_timeout_seconds": 300.0,
    }
    assert stderr.getvalue() == "stdout: [####################] 1/1 job-1:training:0\n"


@pytest.mark.parametrize(
    "flag",
    [
        ["--parallel", "2"],
        ["--strict"],
        ["--stale-after-minutes", "5"],
    ],
)
def test_download_log_chunk_flags_reject_execution_logs(tmp_path, flag):
    stderr = io.StringIO()

    def client_factory(_args):
        raise AssertionError("client constructed before the flag check")

    rc = cli.main(
        _base_args() + ["download-log", "job-1", *flag, "--output-dir", str(tmp_path)],
        client_factory=client_factory,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert rc == 1
    assert f"{flag[0]} applies to --log-type stdout" in stderr.getvalue()


@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_download_metrics_rejects_bad_parallel(tmp_path, value, capsys):
    with pytest.raises(SystemExit):
        cli.main(
            _base_args() + ["download-metrics", "job-1", "--parallel", value],
            client_factory=lambda _: FakeClient(),
            stdout=io.StringIO(),
        )
    assert "positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("command", [["download-metrics", "job-1"], ["download-log", "job-1", "--log-type", "stdout"]])
def test_parallel_is_capped(tmp_path, command, capsys):
    with pytest.raises(SystemExit):
        cli.main(_base_args() + command + ["--parallel", "129"], client_factory=lambda _: FakeClient(), stdout=io.StringIO())
    assert "--parallel: must be at most 128, got 129" in capsys.readouterr().err
    client = FakeClient()
    rc = cli.main(
        _base_args() + command + ["--parallel", "128", "--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    assert client.download_options["max_workers"] == 128


def _interrupted(saved):
    exc = KeyboardInterrupt()
    exc.download_progress_saved = saved
    return exc


@pytest.mark.parametrize(
    "command",
    [["download-metrics", "job-1"], ["download-log", "job-1", "--log-type", "stdout"]],
)
@pytest.mark.parametrize(
    ("raised", "message"),
    [
        (_interrupted(True), "interrupted; downloaded chunks were saved. Run the same command with --resume to continue.\n"),
        (_interrupted(False), "interrupted before any output was saved\n"),
        (KeyboardInterrupt(), "interrupted before any output was saved\n"),
    ],
)
def test_download_interrupt_exits_130_with_accurate_message(tmp_path, command, raised, message):
    client = FakeClient()
    client.download_interrupt = raised
    stdout = io.StringIO()
    stderr = io.StringIO()

    rc = cli.main(
        _base_args() + command + ["--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=stdout,
        stderr=stderr,
    )

    assert rc == 130
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == message


@pytest.mark.parametrize("where", ["stall wait", "session close"])
def test_download_interrupted_after_the_files_are_written_prints_the_resume_hint(
    tmp_path, monkeypatch, where
):
    from cortex_training import _artifact_download
    from cortex_training.client import core as client_module

    client = client_module.CortexTrainingClient(
        base_url="http://test.local", database="DB", schema="SCH"
    )
    sub = "job-1:training:0"
    saved = tmp_path / sub / "metrics.jsonl"
    release = threading.Event()

    def write_plans(*_args, **_kwargs):
        saved.parent.mkdir(parents=True)
        saved.write_text("{}\n")
        return [{"sub_job_id": sub, "saved_path": str(saved)}]

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(client, "_experiment_run_uri", lambda _job: ("snow://run/", "RUN"))
    monkeypatch.setattr(client, "_open_experiment_artifact_connection", MagicMock)
    monkeypatch.setattr(
        client, "_list_gzip_chunks", lambda *_args, **_kwargs: ({sub: ["_metrics/c0.gz"]}, {})
    )
    # The stall check outlives the wait, as when the job lookup is slow.
    monkeypatch.setattr(client, "_STALL_CHECK_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(client, "_stale_upload_warnings", lambda *_args: release.wait(10))
    monkeypatch.setattr(client, "_write_gzip_plans", write_plans)
    if where == "stall wait":
        monkeypatch.setattr(_artifact_download.Background, "result", interrupted)
    else:
        monkeypatch.setattr(_artifact_download.SharedConnection, "close", interrupted)
    stderr = io.StringIO()
    try:
        rc = cli.main(
            _base_args() + ["download-metrics", "job-1", "--output-dir", str(tmp_path)],
            client_factory=lambda _: client,
            stdout=io.StringIO(),
            stderr=stderr,
        )
    finally:
        release.set()

    assert rc == 130
    assert stderr.getvalue() == (
        "interrupted; downloaded chunks were saved. "
        "Run the same command with --resume to continue.\n"
    )
    assert saved.read_text() == "{}\n"


@pytest.mark.parametrize("saved", [True, False])
def test_download_connector_error_is_short_with_query_id_and_hint(tmp_path, saved):
    errors = pytest.importorskip("snowflake.connector.errors")
    exc = errors.ProgrammingError(msg="Insufficient privileges", errno=3001, sfqid="query-123")
    exc.download_progress_saved = saved
    client = FakeClient()
    client.download_interrupt = exc
    stderr = io.StringIO()

    rc = cli.main(
        _base_args() + ["download-metrics", "job-1", "--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert rc == 1
    text = stderr.getvalue()
    assert text.startswith("error: ") and "Insufficient privileges" in text
    assert "query-123" in text
    assert "Traceback" not in text
    assert ("--resume" in text) is saved


def test_download_os_error_gets_resume_hint_when_saved(tmp_path):
    exc = OSError("disk full")
    exc.download_progress_saved = True
    client = FakeClient()
    client.download_interrupt = exc
    stderr = io.StringIO()
    rc = cli.main(
        _base_args() + ["download-log", "job-1", "--log-type", "stdout", "--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=io.StringIO(),
        stderr=stderr,
    )
    assert rc == 1
    assert stderr.getvalue() == (
        "error: disk full\n"
        "Downloaded chunks were saved. Run the same command with --resume to continue.\n"
    )


@pytest.mark.parametrize(
    "command",
    [["download-metrics", "job-1"], ["download-log", "job-1", "--log-type", "stdout"]],
)
def test_download_strict_and_library_warnings_reach_stderr(tmp_path, command):
    client = FakeClient()
    client.download_warning = "skipped chunk snow://x: damaged"
    stdout = io.StringIO()
    stderr = io.StringIO()

    rc = cli.main(
        _base_args() + command + ["--strict", "--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=stdout,
        stderr=stderr,
    )

    assert rc == 0
    assert client.download_options["strict"] is True
    assert stderr.getvalue().count("warning: skipped chunk snow://x: damaged") == 1
    assert "warning" not in stdout.getvalue()
    assert logging.getLogger("cortex_training").handlers == []


def test_download_warnings_still_reach_the_callers_log_handlers(tmp_path):
    client = FakeClient()
    client.download_warning = "skipped chunk snow://x: damaged"
    seen = []

    class Collect(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    root = logging.getLogger()
    collect = Collect(logging.WARNING)
    root.addHandler(collect)
    stderr = io.StringIO()
    try:
        rc = cli.main(
            _base_args() + ["download-metrics", "job-1", "--output-dir", str(tmp_path)],
            client_factory=lambda _: client,
            stdout=io.StringIO(),
            stderr=stderr,
        )
    finally:
        root.removeHandler(collect)
    assert rc == 0
    assert seen == ["skipped chunk snow://x: damaged"]
    assert stderr.getvalue().count("warning: skipped chunk snow://x: damaged") == 1


@pytest.mark.parametrize(("value", "expected"), [("0", None), ("2.5", 150.0), ("30", 1800.0)])
def test_download_metrics_stale_after_minutes(tmp_path, value, expected):
    client = FakeClient()
    rc = cli.main(
        _base_args()
        + ["download-metrics", "job-1", "--stale-after-minutes", value, "--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    assert client.download_options["stale_after_seconds"] == expected


@pytest.mark.parametrize(("value", "expected"), [("0", None), ("45", 45.0)])
def test_download_log_transfer_timeout(tmp_path, value, expected):
    client = FakeClient()
    rc = cli.main(
        _base_args()
        + ["download-log", "job-1", "--log-type", "stdout", "--transfer-timeout", value]
        + ["--output-dir", str(tmp_path)],
        client_factory=lambda _: client,
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    assert client.download_options["transfer_timeout_seconds"] == expected


@pytest.mark.parametrize("value", ["-1", "x", "nan", "inf"])
def test_download_metrics_rejects_bad_stale_after(value, capsys):
    with pytest.raises(SystemExit):
        cli.main(
            _base_args() + ["download-metrics", "job-1", "--stale-after-minutes", value],
            client_factory=lambda _: FakeClient(),
            stdout=io.StringIO(),
        )
    assert "non-negative number" in capsys.readouterr().err


def test_download_commands_write_progress_to_stderr(tmp_path):
    class ProgressClient(FakeClient):
        def download_stdout_logs(self, job_id, output_dir, *, resume=False, progress=None, **options):
            if progress is not None:
                progress("stdout: found 1 chunks for job\n")
            return super().download_stdout_logs(
                job_id, output_dir, resume=resume, progress=None, **options
            )

        def download_metrics(self, job_id, output_dir, *, resume=False, progress=None, **options):
            if progress is not None:
                progress("metrics: found 1 chunks for job\n")
            return super().download_metrics(
                job_id, output_dir, resume=resume, progress=None, **options
            )

        def download_training_metrics(self, job_id, output_dir, *, resume=False, progress=None, **options):
            if progress is not None:
                progress("training metrics: found 1 chunks for job\n")
            return super().download_training_metrics(
                job_id, output_dir, resume=resume, progress=None, **options
            )

    stdout = io.StringIO()
    stderr = io.StringIO()
    rc = cli.main(
        _base_args()
        + ["download-log", "job-1", "--log-type", "stdout", "--output-dir", str(tmp_path)],
        client_factory=lambda _: ProgressClient(),
        stdout=stdout,
        stderr=stderr,
    )
    assert rc == 0
    assert json.loads(stdout.getvalue())["logs"][0]["filename"] == "stdout.log"
    assert stderr.getvalue() == "stdout: found 1 chunks for job\n"

    stdout = io.StringIO()
    stderr = io.StringIO()
    rc = cli.main(
        _base_args() + ["download-metrics", "job-1", "--output-dir", str(tmp_path / "metrics")],
        client_factory=lambda _: ProgressClient(),
        stdout=stdout,
        stderr=stderr,
    )
    assert rc == 0
    body = json.loads(stdout.getvalue())
    assert body["metrics"][0]["filename"] == "gpu.jsonl"
    assert body["training_metrics"][0]["filename"] == "training.jsonl"
    assert stderr.getvalue() == (
        "metrics: found 1 chunks for job\n"
        "training metrics: found 1 chunks for job\n"
    )


class _TtyStderr(io.StringIO):
    def isatty(self):
        return True


class _WriteFails(_TtyStderr):
    def write(self, message):
        raise BrokenPipeError(message)


class _FlushFails(_TtyStderr):
    def flush(self):
        raise OSError("flush")


@pytest.mark.parametrize("stream_cls", [_WriteFails, _FlushFails])
@pytest.mark.parametrize(
    "messages",
    [["metrics: found 1 chunks for job\n"], ["metrics: [#] 1/1 job", "\n"]],
)
def test_download_metrics_survives_stderr_oserror(stream_cls, messages, tmp_path):
    class ProgressClient(FakeClient):
        def download_metrics(self, job_id, output_dir, *, resume=False, progress=None, **options):
            for message in messages:
                progress(message)
            return super().download_metrics(job_id, output_dir, resume=resume, progress=None, **options)

    stdout = io.StringIO()
    rc = cli.main(
        _base_args() + ["download-metrics", "job-1", "--output-dir", str(tmp_path)],
        client_factory=lambda _: ProgressClient(),
        stdout=stdout,
        stderr=stream_cls(),
    )
    assert rc == 0
    assert json.loads(stdout.getvalue())["job_id"] == "job-1"


def test_artifact_progress_writer_edges():
    for stream in (_TtyStderr(), io.StringIO()):
        write = cli._artifact_progress_writer(stream)
        write("\n")
        write("stdout: no new chunks for job\n")
        assert stream.getvalue() == "stdout: no new chunks for job\n"

    for error in (ValueError, OSError):

        class IsattyFails(io.StringIO):
            isatty_calls = 0

            def isatty(self):
                type(self).isatty_calls += 1
                raise error("closed")

        stream = IsattyFails()
        write = cli._artifact_progress_writer(stream)
        write("metrics: [#] 1/2 job")
        write("metrics: [##] 2/2 job")
        write("\n")
        assert stream.getvalue() == "metrics: [##] 2/2 job\n"
        assert IsattyFails.isatty_calls == 1

    class FailsOnce(io.StringIO):
        failed = False

        def write(self, message):
            if not self.failed:
                self.failed = True
                raise BrokenPipeError(message)
            return super().write(message)

    stream = FailsOnce()
    write = cli._artifact_progress_writer(stream)
    write("metrics: [#] 1/1 job")
    write("\n")
    write("\n")
    assert stream.getvalue() == ""


def test_download_progress_is_one_terminal_line(tmp_path):
    class ProgressClient(FakeClient):
        def download_metrics(self, job_id, output_dir, *, resume=False, progress=None, **options):
            if progress is not None:
                progress("metrics: [##------------------] 1/2 job")
                progress("metrics: [####################] 2/2 job")
                progress("\n")
            return super().download_metrics(job_id, output_dir, resume=resume, progress=None, **options)

        def download_training_metrics(self, job_id, output_dir, *, progress=None, **options):
            # Silent: this test is about one stream's line, not how many streams run.
            return super().download_training_metrics(job_id, output_dir, progress=None, **options)

    tty = _TtyStderr()
    rc = cli.main(
        _base_args() + ["download-metrics", "job-1", "--output-dir", str(tmp_path)],
        client_factory=lambda _: ProgressClient(),
        stdout=io.StringIO(),
        stderr=tty,
    )
    assert rc == 0
    assert tty.getvalue() == (
        "\rmetrics: [##------------------] 1/2 job"
        "\rmetrics: [####################] 2/2 job\n"
    )

    plain = io.StringIO()
    rc = cli.main(
        _base_args() + ["download-metrics", "job-1", "--output-dir", str(tmp_path / "plain")],
        client_factory=lambda _: ProgressClient(),
        stdout=io.StringIO(),
        stderr=plain,
    )
    assert rc == 0
    assert plain.getvalue() == "metrics: [####################] 2/2 job\n"


def test_execution_log_download_reports_each_file(tmp_path):
    stderr = io.StringIO()
    rc = cli.main(
        _base_args() + ["download-log", "job-1", "--output-dir", str(tmp_path)],
        client_factory=lambda _: FakeClient(),
        stdout=io.StringIO(),
        stderr=stderr,
    )
    assert rc == 0
    assert stderr.getvalue().splitlines() == [
        "execution logs: job-1:training:0/execution.jsonl: saved",
        "execution logs: job-1:training:0/server.log: saved",
        "execution logs: job-1:sampling:0/execution.jsonl: saved",
    ]


def test_download_warning_does_not_break_the_progress_line():
    tty = _TtyStderr()
    write = cli._artifact_progress_writer(tty)
    write("metrics: [#-] 1/2 job")
    write.warn("warning: GET x failed; retrying")
    write("metrics: [##] 2/2 job")
    write("\n")
    assert tty.getvalue() == (
        "\rmetrics: [#-] 1/2 job"
        "\nwarning: GET x failed; retrying\n"
        "\rmetrics: [#-] 1/2 job"
        "\rmetrics: [##] 2/2 job\n"
    )

    plain = io.StringIO()
    write = cli._artifact_progress_writer(plain)
    write("metrics: [#-] 1/2 job")
    write.warn("warning: GET x failed; retrying")
    write("\n")
    assert plain.getvalue() == "warning: GET x failed; retrying\nmetrics: [#-] 1/2 job\n"


class _SlowTty(_TtyStderr):
    """A terminal whose writes yield between characters, so unserialized writes interleave."""

    def write(self, message):
        for char in message:
            super().write(char)
            time.sleep(0)
        return len(message)


def test_progress_writer_keeps_lines_whole_under_concurrent_warnings():
    tty = _SlowTty()
    write = cli._artifact_progress_writer(tty)
    statuses = [f"metrics: [{'#' * (i % 20):-<20}] {i}/200 job" for i in range(200)]
    warnings = [f"warning: GET chunk-{t}-{i} failed; retrying" for t in range(4) for i in range(25)]

    def warn_all(thread):
        for i in range(25):
            write.warn(f"warning: GET chunk-{thread}-{i} failed; retrying")

    threads = [threading.Thread(target=warn_all, args=(t,)) for t in range(4)]
    for thread in threads:
        thread.start()
    for status in statuses:
        write(status)
    for thread in threads:
        thread.join()
    write("\n")

    text = tty.getvalue()
    lines = text.split("\n")
    assert sorted(line for line in lines if line.startswith("warning:")) == sorted(warnings)
    for line in lines:
        if line.startswith("warning:"):
            continue
        # Every other line is a run of whole status redraws.
        for segment in line.split("\r")[1:]:
            assert segment in statuses, repr(segment)
    assert text.endswith(statuses[-1] + "\n")


def test_session_close_failure_is_one_warning_line(monkeypatch):
    from cortex_training import _artifact_download

    class Broken:
        def close(self):
            raise RuntimeError("close failed")

    stderr = io.StringIO()
    writer = cli._artifact_progress_writer(stderr)
    with cli._library_warnings_to(writer):
        _artifact_download.SharedConnection(Broken).close()
    assert stderr.getvalue() == (
        "warning: could not close the artifact download session: close failed\n"
    )


def test_download_metrics_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stdout = io.StringIO()
    client = FakeClient()

    rc = cli.main(
        _base_args() + ["download-metrics", "job-1"],
        client_factory=lambda _: client,
        stdout=stdout,
    )

    assert rc == 0
    assert client.metrics_job_id == "job-1"
    assert client.metrics_output_dir == tmp_path
    assert (tmp_path / "job-1:training:0" / "gpu.jsonl").read_text() == (
        '{"gpu_utilization":0.5}\n'
    )
    payload = json.loads(stdout.getvalue())
    assert payload["job_id"] == "job-1"
    assert payload["metrics"][0]["filename"] == "gpu.jsonl"


def test_http_errors_include_response_details():
    stderr = io.StringIO()

    def make_client(_args):
        raise FakeHTTPError()

    rc = cli.main(_base_args() + ["list"], client_factory=make_client, stderr=stderr)

    assert rc == 1
    err = stderr.getvalue()
    assert "500 Server Error" in err
    assert "snowflake request id: sf-req-1" in err
    assert 'response body: {"message":"backend detail"}' in err


def test_a_schema_not_found_error_shows_the_server_message_once():
    import requests

    from cortex_training import CortexTrainingClient

    client = CortexTrainingClient(
        base_url="https://test.snowflakecomputing.com", database="MY_DB", schema="PUBLIC", max_retries=0
    )
    message = "Schema MY_DB.PUBLIC is not found or not authorized"
    response = requests.Response()
    response.status_code = 400
    response.url = "https://test.snowflakecomputing.com/error"
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps({"code": "517602", "message": message}).encode()
    response._content_consumed = True
    client._session = MagicMock()
    client._session.get.return_value = response
    stderr = io.StringIO()

    rc = cli.main(["get", "job-1"], client_factory=lambda _args: client, stderr=stderr)

    err = stderr.getvalue()
    assert rc == 1
    assert err.startswith("error: Role <ROLE> (couldn't determine your role) can't find database MY_DB")
    assert err.count(message) == 1


def test_invalid_job_json_returns_error(tmp_path):
    path = tmp_path / "job.json"
    path.write_text("[]", encoding="utf-8")
    stderr = io.StringIO()

    rc = cli.main(["submit", str(path), "--dry-run"], stderr=stderr)

    assert rc == 1
    assert "job JSON must be an object" in stderr.getvalue()


def test_config_file_supplies_connection_values(tmp_path):
    config = _write_config(
        tmp_path,
        {
            "base_url": "http://config.local",
            "database": "CONFIG_DB",
            "schema": "CONFIG_SCHEMA",
            "endpoint": "config-endpoint",
            "poll_interval": 2.0,
            "poll_timeout": 30.0,
        },
    )
    seen = {}
    stdout = io.StringIO()

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(["--config", str(config), "list"], client_factory=make_client, stdout=stdout)

    assert rc == 0
    assert seen["base_url"] == "http://config.local"
    assert seen["database"] == "CONFIG_DB"
    assert seen["schema"] == "CONFIG_SCHEMA"
    assert seen["endpoint"] == "config-endpoint"
    assert seen["poll_interval"] == 2.0
    assert seen["poll_timeout"] == 30.0


def test_cli_flags_override_config_file(tmp_path):
    config = _write_config(
        tmp_path,
        {
            "base_url": "http://config.local",
            "database": "CONFIG_DB",
            "schema": "CONFIG_SCHEMA",
        },
    )
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(
        [
            "--config",
            str(config),
            "--base-url",
            "http://cli.local",
            "--database",
            "CLI_DB",
            "list",
        ],
        client_factory=make_client,
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert seen["base_url"] == "http://cli.local"
    assert seen["database"] == "CLI_DB"
    assert seen["schema"] == "CONFIG_SCHEMA"


def test_config_can_come_from_env(tmp_path, monkeypatch):
    config = _write_config(
        tmp_path,
        {
            "base_url": "http://env-config.local",
            "database": "ENV_CONFIG_DB",
        },
    )
    monkeypatch.setenv("CORTEX_TRAINING_CONFIG", str(config))
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(["list"], client_factory=make_client, stdout=io.StringIO())

    assert rc == 0
    assert seen["base_url"] == "http://env-config.local"
    assert seen["database"] == "ENV_CONFIG_DB"


def test_config_accepts_common_aliases(tmp_path):
    config = _write_config(tmp_path, {"url": "http://alias.local", "db": "ALIAS_DB"})
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(["--config", str(config), "list"], client_factory=make_client, stdout=io.StringIO())

    assert rc == 0
    assert seen["base_url"] == "http://alias.local"
    assert seen["database"] == "ALIAS_DB"


def test_bare_base_url_with_pat_is_treated_as_host(tmp_path):
    config = _write_config(
        tmp_path,
        {
            "base_url": "ACCOUNT.snowflakecomputing.com",
            "pat": "test-pat",
            "database": "CORTEX_TRAINING_DB",
        },
    )
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(
        ["--config", str(config), "list"],
        client_factory=make_client,
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert seen["base_url"] is None
    assert seen["host"] == "ACCOUNT.snowflakecomputing.com"
    assert seen["pat"] == "test-pat"


def test_bare_base_url_without_pat_returns_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("CORTEX_TRAINING_PAT", raising=False)
    monkeypatch.delenv("SNOWFLAKE_PAT", raising=False)
    config = _write_config(
        tmp_path,
        {
            "base_url": "ACCOUNT.snowflakecomputing.com",
            "database": "CORTEX_TRAINING_DB",
        },
    )
    stderr = io.StringIO()

    rc = cli.main(["--config", str(config), "list"], stderr=stderr)

    assert rc == 1
    assert "base_url must start with http:// or https://" in stderr.getvalue()


def test_invalid_config_returns_error(tmp_path):
    config = _write_config(tmp_path, {"base_url": "http://test.local", "typo": "value"})
    stderr = io.StringIO()

    rc = cli.main(["--config", str(config), "submit", "missing.json", "--dry-run"], stderr=stderr)

    assert rc == 1
    assert "unknown config key" in stderr.getvalue()


def test_invalid_config_value_type_returns_error(tmp_path):
    config = _write_config(tmp_path, {"base_url": 123, "database": "DB"})
    stderr = io.StringIO()

    rc = cli.main(["--config", str(config), "list"], stderr=stderr)

    assert rc == 1
    assert "config base_url must be a string" in stderr.getvalue()


@pytest.mark.parametrize("config_flags", [[], ["--config"]])
def test_login_persists_config_path(tmp_path, monkeypatch, config_flags):
    login_state = tmp_path / "login.json"
    config = _write_config(
        tmp_path,
        {"base_url": "http://login.local", "database": "LOGIN_DB"},
    )
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login_state))
    stdout = io.StringIO()

    rc = cli.main(["login"] + config_flags + [str(config)], stdout=stdout)

    assert rc == 0
    saved = json.loads(login_state.read_text(encoding="utf-8"))
    assert saved == {"config_path": str(config.resolve())}
    assert json.loads(stdout.getvalue()) == {
        "config_path": str(config.resolve()),
        "logged_in": True,
    }


def test_logged_in_config_is_used_without_config_arg(tmp_path, monkeypatch):
    login_state = tmp_path / "login.json"
    config = _write_config(
        tmp_path,
        {"base_url": "http://login.local", "database": "LOGIN_DB"},
    )
    login_state.write_text(
        json.dumps({"config_path": str(config)}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login_state))
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(["list"], client_factory=make_client, stdout=io.StringIO())

    assert rc == 0
    assert seen["base_url"] == "http://login.local"
    assert seen["database"] == "LOGIN_DB"


def test_cli_flags_override_logged_in_config(tmp_path, monkeypatch):
    login_state = tmp_path / "login.json"
    config = _write_config(
        tmp_path,
        {
            "base_url": "http://login.local",
            "database": "LOGIN_DB",
            "schema": "LOGIN_SCHEMA",
        },
    )
    login_state.write_text(
        json.dumps({"config_path": str(config)}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login_state))
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(
        ["--database", "CLI_DB", "list"],
        client_factory=make_client,
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert seen["base_url"] == "http://login.local"
    assert seen["database"] == "CLI_DB"
    assert seen["schema"] == "LOGIN_SCHEMA"


def test_direct_connection_flags_do_not_read_login_state(tmp_path, monkeypatch):
    login_state = tmp_path / "login.json"
    login_state.write_text("not json", encoding="utf-8")
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login_state))

    rc = cli.main(
        _base_args() + ["list"],
        client_factory=_factory([]),
        stdout=io.StringIO(),
    )

    assert rc == 0


@pytest.mark.parametrize("config_flags", [[], ["--config"]])
def test_login_rejects_invalid_config(tmp_path, monkeypatch, config_flags):
    login_state = tmp_path / "login.json"
    config = _write_config(tmp_path, {"typo": "value"})
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login_state))
    stderr = io.StringIO()

    rc = cli.main(["login"] + config_flags + [str(config)], stderr=stderr)

    assert rc == 1
    assert not login_state.exists()
    assert "unknown config key" in stderr.getvalue()


def test_no_legacy_config_falls_back_to_default_snowflake_profile(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(
        "CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "missing-login.json")
    )
    for name in (
        "CORTEX_TRAINING_CONFIG",
        "CORTEX_TRAINING_BASE_URL",
        "CORTEX_TRAINING_HOST",
        "SNOWFLAKE_HOST",
        "CORTEX_TRAINING_PAT",
        "SNOWFLAKE_PAT",
    ):
        monkeypatch.delenv(name, raising=False)
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(["list"], client_factory=make_client, stdout=io.StringIO())

    assert rc == 0
    assert seen["use_connection_profile"] is True
    assert seen["connection"] is None
    assert seen["database"] is None
    assert seen["schema"] is None


@pytest.mark.parametrize(
    ("environment_name", "environment_value"),
    [
        ("SNOWFLAKE_HOST", "leftover.example"),
        ("SNOWFLAKE_PAT", "leftover-pat"),
    ],
)
def test_incomplete_direct_environment_falls_back_to_default_profile(
    tmp_path, monkeypatch, environment_name, environment_value
):
    monkeypatch.setenv(
        "CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "missing-login.json")
    )
    for name in (
        "CORTEX_TRAINING_CONFIG",
        "CORTEX_TRAINING_BASE_URL",
        "CORTEX_TRAINING_HOST",
        "SNOWFLAKE_HOST",
        "CORTEX_TRAINING_PAT",
        "SNOWFLAKE_PAT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(environment_name, environment_value)
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(["list"], client_factory=make_client, stdout=io.StringIO())

    assert rc == 0
    assert seen["use_connection_profile"] is True
    assert seen["connection"] is None


def test_named_connection_bypasses_invalid_login_state(tmp_path, monkeypatch):
    login_state = tmp_path / "login.json"
    login_state.write_text("not json", encoding="utf-8")
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login_state))
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(
        ["--connection", "training-profile", "list"],
        client_factory=make_client,
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert seen["use_connection_profile"] is True
    assert seen["connection"] == "training-profile"


def test_existing_legacy_config_wins_over_profile_fallback(tmp_path, monkeypatch):
    config = _write_config(
        tmp_path,
        {"base_url": "http://legacy.local", "database": "LEGACY_DB"},
    )
    monkeypatch.setenv("CORTEX_TRAINING_CONFIG", str(config))
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(["list"], client_factory=make_client, stdout=io.StringIO())

    assert rc == 0
    assert seen["use_connection_profile"] is False
    assert seen["base_url"] == "http://legacy.local"


def test_missing_remembered_config_falls_back_to_profile(tmp_path, monkeypatch):
    login_state = tmp_path / "login.json"
    login_state.write_text(
        json.dumps({"config_path": str(tmp_path / "removed-config.json")}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CORTEX_TRAINING_LOGIN_FILE", str(login_state))
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(["list"], client_factory=make_client, stdout=io.StringIO())

    assert rc == 0
    assert seen["use_connection_profile"] is True


def test_named_connection_conflicts_with_direct_flags():
    stderr = io.StringIO()

    rc = cli.main(
        ["--connection", "training-profile", "--host", "x.test", "list"],
        stderr=stderr,
    )

    assert rc == 1
    assert "--connection cannot be combined" in stderr.getvalue()


def test_build_client_uses_named_snowflake_connection(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "missing.json")
    )
    args = cli.parse_args(["--connection", "training-profile", "list"])
    client_cls = MagicMock()
    connected = object()
    client_cls._from_profile.return_value = connected

    assert cli.build_client(args, client_cls) is connected
    client_cls._from_profile.assert_called_once_with(
        connection_name="training-profile",
        database=None,
        schema=None,
        endpoint="cortex-training",
        poll_interval=0.5,
        poll_timeout=1800.0,
        verify_ssl=True,
    )


def test_build_client_forwards_read_bounds_only_when_set(tmp_path, monkeypatch):
    # `cortex-training tui` sets these on its args; the CLI commands never do,
    # so they keep the client's defaults (the test above).
    monkeypatch.setenv(
        "CORTEX_TRAINING_LOGIN_FILE", str(tmp_path / "missing.json")
    )
    args = cli.parse_args(["--connection", "training-profile", "list"])
    args.max_retries, args.request_timeout = 2, (10.0, 45.0)
    client_cls = MagicMock()

    cli.build_client(args, client_cls)
    kwargs = client_cls._from_profile.call_args.kwargs
    assert (kwargs["max_retries"], kwargs["request_timeout"]) == (2, (10.0, 45.0))


def test_the_commands_tag_the_client_they_build_as_the_cli(monkeypatch):
    # Its metrics then say a command made the call (client.surface).
    built = []
    monkeypatch.setattr(cli, "build_client", lambda args, cls: built.append(FakeClient()) or built[-1])

    assert cli.main(_base_args() + ["list"], stdout=io.StringIO()) == 0

    assert [client._telemetry_surface for client in built] == ["cli"]


def _http_error(status_code: int, headers: dict[str, str] | None = None):
    import requests

    request = requests.Request(
        "GET", "https://account.snowflakecomputing.com/api", headers=headers or {}
    ).prepare()
    response = requests.Response()
    response.status_code = status_code
    response.request = request
    response._content = b'{"code": "390100", "message": "rejected"}'
    return requests.HTTPError(f"{status_code} Client Error", response=response)


_PAT_HEADERS = {
    "Authorization": "Bearer REDACTED",
    "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
}


def test_format_error_explains_rejected_pat_on_401() -> None:
    message = cli._format_error(_http_error(401, _PAT_HEADERS))

    assert message.startswith("401 Client Error")
    assert "Snowflake rejected your credentials (HTTP 401)." in message
    assert "Your user has no network policy." in message
    assert "authentication.md#network-policy-requirement" in message


def test_format_error_skips_pat_hint_for_session_token_401() -> None:
    message = cli._format_error(
        _http_error(401, {"Authorization": 'Snowflake Token="REDACTED"'})
    )

    assert "401 Client Error" in message
    assert "network policy" not in message


def test_format_error_leaves_other_http_errors_unchanged() -> None:
    message = cli._format_error(_http_error(409, _PAT_HEADERS))

    assert "409 Client Error" in message
    assert "network policy" not in message


def test_format_error_does_not_repeat_request_id_already_in_message() -> None:
    exc = FakeHTTPError()
    exc.args = (f"[snowflake request id: sf-req-1] {exc.args[0]}",)

    message = cli._format_error(exc)

    assert message.count("sf-req-1") == 1
    assert "response body:" in message


@pytest.mark.parametrize(
    "command",
    [
        ["download-log", "job-1", "--log-type", "stdout"],
        ["download-metrics", "job-1"],
    ],
)
def test_chunk_download_refuses_a_directory_the_tui_is_downloading(tmp_path, command):
    fcntl = pytest.importorskip("fcntl")
    out_dir = tmp_path / "stdout"
    lock_path = tmp_path / ".stdout.lock"
    stderr = io.StringIO()
    client = FakeClient()
    # A separate open file description stands in for the TUI process.
    with open(lock_path, "a+b") as tui:
        fcntl.flock(tui.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc = cli.main(
            _base_args() + command + ["--output-dir", str(out_dir)],
            client_factory=lambda _: client,
            stdout=io.StringIO(),
            stderr=stderr,
        )
    assert rc == 1
    assert f"another download is using {out_dir}" in stderr.getvalue()
    assert not out_dir.exists()


def test_chunk_download_takes_a_free_tui_lock_and_creates_none(tmp_path):
    client = FakeClient()
    free = tmp_path / "free"
    (tmp_path / ".free.lock").touch()
    rc = cli.main(
        _base_args() + ["download-metrics", "job-1", "--output-dir", str(free)],
        client_factory=lambda _: client,
        stdout=io.StringIO(),
    )
    assert rc == 0
    plain = tmp_path / "plain"
    rc = cli.main(
        _base_args() + ["download-log", "job-1", "--log-type", "stdout", "--output-dir", str(plain)],
        client_factory=lambda _: client,
        stdout=io.StringIO(),
    )
    assert rc == 0
    assert not (tmp_path / ".plain.lock").exists(), "a plain CLI download created a lock file"


_CHUNK_COMMANDS = [
    pytest.param(["download-log", "job-1", "--log-type", "stdout"], "stdout.log", id="download-log"),
    pytest.param(["download-metrics", "job-1"], "gpu.jsonl", id="download-metrics"),
]


@pytest.mark.parametrize("command, filename", _CHUNK_COMMANDS)
@pytest.mark.parametrize("output_dir", [None, ".", "./", "sub", "sub/../sub", "other/.."])
def test_chunk_download_accepts_relative_and_default_output_dirs(tmp_path, monkeypatch, command, filename, output_dir):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "other").mkdir()
    stderr = io.StringIO()
    args = command + ([] if output_dir is None else ["--output-dir", output_dir])
    rc = cli.main(_base_args() + args, client_factory=lambda _: FakeClient(), stdout=io.StringIO(), stderr=stderr)
    assert rc == 0, stderr.getvalue()
    expected = (tmp_path / (output_dir or ".")).resolve()
    assert (expected / "job-1:training:0" / filename).is_file()
    # No TUI lock existed, so none was created anywhere.
    assert not list(tmp_path.glob(".*.lock")) and not list(tmp_path.parent.glob(f".{tmp_path.name}.lock"))


@pytest.mark.parametrize("command, filename", _CHUNK_COMMANDS)
@pytest.mark.parametrize("cwd, output_dir", [("cache", None), ("cache", "."), ("cache", "./"), (".", "cache"),
                                             (".", "x/../cache"), ("x", "../cache")])
def test_chunk_download_finds_the_tui_lock_from_any_spelling(tmp_path, monkeypatch, command, filename, cwd, output_dir):
    from cortex_training._download_lock import DownloadLock

    fcntl = pytest.importorskip("fcntl")
    cache = tmp_path / "cache"
    cache.mkdir()
    (tmp_path / "x").mkdir()
    monkeypatch.chdir(tmp_path / cwd)
    tui_lock = DownloadLock(cache)
    assert tui_lock.lock_path == tmp_path / ".cache.lock"
    stderr = io.StringIO()
    args = command + ([] if output_dir is None else ["--output-dir", output_dir])
    # A separate open file description stands in for the TUI process.
    with open(tui_lock.lock_path, "a+b") as tui:
        fcntl.flock(tui.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        rc = cli.main(_base_args() + args, client_factory=lambda _: FakeClient(), stdout=io.StringIO(), stderr=stderr)
    assert rc == 1
    assert "another download is using" in stderr.getvalue()
    assert not (cache / "job-1:training:0").exists()


def test_download_lock_paths_are_resolved(tmp_path, monkeypatch):
    from cortex_training._download_lock import DownloadLock

    monkeypatch.chdir(tmp_path)
    (tmp_path / "foo").mkdir()
    absolute = DownloadLock(tmp_path / "cache").lock_path
    assert absolute == tmp_path / ".cache.lock"
    for spelling in ["cache", "./cache", "foo/../cache", str(tmp_path / "foo" / ".." / "cache")]:
        assert DownloadLock(Path(spelling)).lock_path == absolute
    assert DownloadLock(Path(".")).lock_path == tmp_path.parent / f".{tmp_path.name}.lock"
    root = DownloadLock(Path("/"))
    assert root.lock_path is None
    assert root.try_acquire()
    root.release()
    with cli._output_dir_lock(Path("/")):
        pass
