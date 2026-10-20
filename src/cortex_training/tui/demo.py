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

"""Synthetic, offline client for trying and debugging the TUI (``cortex-training tui --debug``).

Implements the read-only client methods the TUI calls, returning records in
the shapes the service returns (job / sub-job metadata,
``gpu.jsonl`` telemetry, tail-logs pages). Every value is generated from a
fixed seed relative to "now"; nothing touches the network.
"""

from __future__ import annotations

import bisect
import copy
import json
import math
import random
import uuid
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Optional

import requests

from cortex_training import AccountJobListUnavailable
from cortex_training.tui import views
from cortex_training.tui.format import format_log_entry

_GIB = 1024**3
_H200_FB = 141 * _GIB
# Each GPU type's memory, about as NVML reports it.
_GPU_MEMORY = {"H200": _H200_FB, "B200": 179 * _GIB, "B300": 268 * _GIB}
# Reasons a job ended without ever being placed: no GPU telemetry and no console.
_NEVER_PLACED = frozenset({"pending_timeout", "placement_too_large", "placement_timeout"})


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _training(model, gpus, *, lr=2e-5, bs=8, seq=4096, lora=None, opt="AdamW"):
    cfg: dict = {"n_gpus": float(gpus), "max_seq_len": float(seq), "train_batch_size": float(bs),
                 "optimizer": {"name": opt, "lr": lr}}
    if lora:
        cfg["peft_config"] = {"peft_type": "Lora", "r": float(lora), "lora_alpha": float(lora * 2),
                              "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
    if model.endswith("-A3B"):  # MoE: trained expert-parallel, as the recipes' configs do
        cfg["ep_size"] = 4.0
    return {"job_type": "TRAINING", "model_name": model, "dtype": "bfloat16", "seed": 42, "training_config": cfg}


def _sampling(model, gpus, *, seq=4096, mem=0.85):
    return {"job_type": "SAMPLING", "model_name": model, "dtype": "bfloat16", "seed": 42,
            "inference_config": {"n_gpus": float(gpus), "max_seq_len": float(seq), "gpu_memory_utilization": mem}}


def _log_prob(model, *, seq=4096):
    """A reference-log-prob sub-job. It requests no GPUs of its own (the control plane's
    rule), whatever its config says, so it reports no GPU telemetry."""
    return {"job_type": "LOG_PROBABILITY", "model_name": model, "dtype": "bfloat16", "seed": 42,
            "inference_config": {"n_gpus": 2.0, "max_seq_len": float(seq)}}


# Where the demo's jobs live: the TUI's configured location, and two more that
# only the account-wide list shows (one a lowercase name, which SQL quotes).
DEMO_LOCATION = ("CORTEX_TRAINING_DB", "PUBLIC")
_TEAM = ("TEAM_DB", "EXPERIMENTS")
_LOWERCASE = ("sandbox", "dev")

# (status, reason, age, duration, hardware, sub-jobs, log kind[, overrides]).
# Overrides exercise missing metrics, multiplexing, UNKNOWN/not-found states,
# Experiment access, other locations, sparse logs, and a stuck oversized line.
_SPECS = [
    # Its nodes report GPU rows only, as on a deployment that doesn't collect CPU and memory.
    ("RUNNING", "", timedelta(hours=2, minutes=14), None, "H200",
     [_training("Qwen/Qwen3-8B", 8, lr=1e-4, bs=16, seq=8192, lora=16)], "sft", {"system_metrics": False}),
    ("RUNNING", "", timedelta(hours=5, minutes=40), None, "H200",
     [_training("Qwen/Qwen3.6-35B-A3B", 8, lr=1e-6, bs=64, seq=4096), _sampling("Qwen/Qwen3.6-35B-A3B", 8)], "rl"),
    ("PLACING", "", timedelta(minutes=3), None, "B300",
     [_training("Qwen/Qwen3.8-27B", 16, lr=1e-5, bs=32, seq=16384), _sampling("Qwen/Qwen3.8-27B", 8, seq=16384)],
     None),
    ("FAILED", "sub_job_readiness", timedelta(days=1, hours=3), timedelta(minutes=31), "H200",
     [_training("Qwen/Qwen3-8B", 4, lr=2e-5, bs=4, seq=32768)], "sft-fail"),
    ("CANCELLED", "cancelled_by_client", timedelta(days=2, hours=6), timedelta(hours=3, minutes=12), "H200",
     [_training("Qwen/Qwen3-8B", 8, lr=1e-4, bs=16, seq=8192, lora=8)], "sft"),
    ("CANCELLED", "cancelled_by_client", timedelta(days=4), timedelta(minutes=20), "H200",
     [_training("Qwen/Qwen3-0.6B", 1, lr=1e-5, bs=1, seq=128), _sampling("Qwen/Qwen3-0.6B", 1, seq=128, mem=0.8)], "sft"),
    ("TERMINATED", "idle_timeout", timedelta(days=9), timedelta(hours=1, minutes=2), "H200",
     [_sampling("Qwen/Qwen3-8B", 2, seq=32768)], None),
    ("CANCELLED", "cancelled_by_client", timedelta(days=16), timedelta(hours=6), "B200",
     [_training("Qwen/Qwen3.5-9B", 8, lr=5e-5, bs=32, seq=8192, lora=32)], "sft"),
    ("CANCELLED", "cancelled_by_client", timedelta(days=16, hours=1), timedelta(hours=5, minutes=30), "B200",
     [_training("Qwen/Qwen3.5-9B", 8, lr=5e-5, bs=32, seq=8192, lora=8)], "sft"),
    ("FAILED", "pending_timeout", timedelta(days=27), timedelta(hours=24), "B300",
     [_training("Qwen/Qwen3.6-35B-A3B", 32, lr=1e-6, bs=128, seq=4096)], None),
    ("CANCELLED", "cancelled_by_client", timedelta(days=41), timedelta(hours=9), "H200",
     [_training("Qwen/Qwen3-8B", 4, lr=1e-6, bs=32, seq=4096), _sampling("Qwen/Qwen3-8B", 4)], "rl"),
    ("CANCELLED", "cancelled_by_client", timedelta(days=58), timedelta(minutes=48), "H200",
     [_sampling("Qwen/Qwen3-8B", 2)], None),
    ("TERMINATED", "idle_timeout", timedelta(days=77), timedelta(hours=4), "H200",
     [_training("Qwen/Qwen3-8B", 8, lr=2e-5, bs=16, seq=16384)], "sft"),
    ("FAILED", "out_of_memory", timedelta(days=96), timedelta(minutes=7), "H200",
     [_training("Qwen/Qwen3.8-27B", 8, lr=1e-5, bs=64, seq=32768)], None),
    ("CANCELLED", "cancelled_by_client", timedelta(days=120), timedelta(hours=2), "H200",
     [_training("Qwen/Qwen3-8B", 4, lr=3e-5, bs=8, seq=4096, lora=16)], None, {"experiment": "missing"}),
    ("CANCELLED", "cancelled_by_client", timedelta(days=171), timedelta(minutes=25), "H200",
     [_training("Qwen/Qwen3-8B", 4, lr=2e-5, bs=4, seq=32768)], None),
    ("CANCELLED", "cancelled_by_client", timedelta(days=201), timedelta(hours=1), "H200",
     [_training("Qwen/Qwen3-8B", 4)], None),  # older than six months: hidden
    ("FAILED", "sub_job_readiness", timedelta(days=230), timedelta(minutes=12), "H200",
     [_sampling("Qwen/Qwen3-8B", 2)], None),  # older than six months: hidden
    ("RUNNING", "", timedelta(minutes=25), None, "H200",
     [_training("Qwen/Qwen3-8B", 8, lr=1e-4, bs=8, seq=4096, lora=8)], None,
     {"multiplex_onto": 0}),
    ("PENDING", "quota_filled", timedelta(minutes=12), None, "B200",
     [_training("Qwen/Qwen3.8-27B", 16, lr=1e-5, bs=32, seq=16384)], None,
     {"reason_details": "needs 16 GPUs; 8 of 16 reserved GPUs are held by your other jobs"}),
    ("RUNNING", "", timedelta(minutes=50), None, "H200",
     [_training("Qwen/Qwen3-8B", 8, lr=2e-5, bs=8, seq=8192), _sampling("Qwen/Qwen3-8B", 2)], None,
     {"unreachable": True}),
    ("FAILED", "node_evacuated", timedelta(days=6), timedelta(hours=2), "H200",
     [_training("Qwen/Qwen3-8B", 8, lr=1e-4, bs=16, seq=8192, lora=16)], "sft",
     {"reason_details": "The node running this job was taken out of service; resume from its last checkpoint"}),
    ("CANCELLED", "cancelled_by_client", timedelta(days=11), timedelta(minutes=40), "H200",
     [_sampling("Qwen/Qwen3-8B", 1)], None),
    ("RUNNING", "", timedelta(hours=1, minutes=5), None, "B200",
     [_training("Qwen/Qwen3-8B", 8, lr=3e-5, bs=16, seq=8192)], "sft",
     {"location": _TEAM}),
    ("FAILED", "placement_too_large", timedelta(days=3, hours=2), timedelta(minutes=1), "B200",
     [_training("Qwen/Qwen3.8-27B", 64, lr=1e-5, bs=64, seq=32768)], None,
     {"location": _TEAM, "reason_details": "asks for 64 GPUs; the largest placement here is 32"}),
    ("CANCELLED", "cancelled_by_client", timedelta(days=12), timedelta(hours=2, minutes=30), "H200",
     [_training("Qwen/Qwen3-8B", 4, lr=1e-6, bs=32, seq=4096), _sampling("Qwen/Qwen3-8B", 4)], "rl",
     {"location": _TEAM, "experiment": "sql"}),
    ("RUNNING", "", timedelta(minutes=40), None, "H200",
     [_sampling("Qwen/Qwen3-8B", 2)], None,
     {"location": _LOWERCASE}),
    ("TERMINATED", "idle_timeout", timedelta(days=20), timedelta(hours=1), "H200",
     [_training("Qwen/Qwen3-0.6B", 1, lr=1e-5, bs=1, seq=128)], "sft",
     {"location": _LOWERCASE}),
    ("CANCELLED", "cancelled_by_client", timedelta(days=5, hours=3), timedelta(hours=7), "H200",
     [_sampling("Qwen/Qwen3-8B", 4)], None,
     {"log_every": 1.0}),
    ("RUNNING", "", timedelta(minutes=35), None, "H200",
     [_training("Qwen/Qwen3-8B", 4, lr=2e-5, bs=8, seq=8192)], None,
     {"stuck": True}),
    ("TERMINATED", "idle_timeout", timedelta(days=8), timedelta(hours=1, minutes=10), "H200",
     [_training("Qwen/Qwen3-8B", 4, lr=1e-6, bs=16, seq=4096), _log_prob("Qwen/Qwen3-8B")], "sft"),
    ("SUCCEEDED", "", timedelta(days=13), timedelta(hours=2), "H200",
     [_training("Qwen/Qwen3-0.6B", 2, lr=1e-5, bs=8, seq=2048)], "sft"),
    ("DONE", "", timedelta(days=60), timedelta(minutes=50), "H200",
     [_training("Qwen/Qwen3-8B", 4, lr=2e-5, bs=8, seq=4096)], None),
    ("JOB_STATE_CANCELLED", "cancelled_by_client", timedelta(days=70), timedelta(minutes=30), "H200",
     [_training("Qwen/Qwen3-8B", 4, lr=2e-5, bs=8, seq=4096)], None),
    ("RUNNING", "", timedelta(hours=3), None, "H200",
     [_training("Qwen/Qwen3-8B", 8, lr=2e-5, bs=16, seq=8192)], None,
     {"not_found": True}),
]
_NOT_FOUND = "Job status unavailable: job not found on Control Plane"
# What the service answers when the role can't read a job's Experiment, by how
# (views.experiment_access_hint): the Experiment is gone or hidden ("missing": its
# saved logs and GPU metrics fail), or Snowflake refuses SQL on it ("sql", a
# MONITOR-only role: saved logs and GPU metrics fail).
_EXPERIMENT_GONE = "Linked experiment is not found for job {job}"
_EXPERIMENT_SQL = ("002003 (02000): SQL compilation error: Experiment '{database}.{schema}.{name}' "
                   "does not exist or not authorized.")
# The stuck job's oversized lines: the first this long after the demo starts, then one per period.
_STUCK_AFTER = timedelta(minutes=1)
_STUCK_EVERY = timedelta(minutes=10)

# What the client raises for each reason the account-wide list is unavailable.
_UNAVAILABLE = {
    AccountJobListUnavailable.NOT_DEPLOYED: (
        "account-level job listing is not available on this service yet: the account-level route "
        "answered 404 with no error body, so it is not deployed"),
    AccountJobListUnavailable.DISABLED: (
        "account-level job listing is not enabled for this account (HTTP 404); Snowflake enables it per account"),
    AccountJobListUnavailable.FORBIDDEN: (
        "account-level job listing was refused (HTTP 403): Cortex Training is not enabled for this account"),
    AccountJobListUnavailable.REST_DISABLED: (
        "account-level job listing is unavailable: the Cortex Training REST API is turned off for this account "
        "(HTTP 405)"),
}
# account_list=UNAVAILABLE: the account-wide list fails as when the control plane is down (503).
UNAVAILABLE = "unavailable"
_ACTIVE = ("pending", "queued", "placing", "initializing", "creating", "running")


def _http_error(status: int, message: str) -> requests.HTTPError:
    """The error the client raises for a service answer of ``status`` with ``message``."""
    response = requests.Response()
    response.status_code = status
    response.reason = {400: "Bad Request", 409: "Conflict", 500: "Internal Server Error",
                       503: "Service Unavailable"}.get(status, "Error")
    response._content = json.dumps({"code": str(status), "message": message}).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    return requests.HTTPError(f"{status} {response.reason}", response=response)


class DemoClient:
    """Deterministic, read-only ``CortexTrainingClient`` stand-in.

    Jobs advance on wall time. ``account_list`` selects an unavailable-account
    response; None lists every location.
    """

    _STEP_SECONDS = 6.0
    _GENERATE_SECONDS = 20.0

    _ACCOUNT_PAGE_ROWS = 10

    def __init__(self, *, account_list: Optional[str] = None) -> None:
        if account_list is not None and account_list not in (*_UNAVAILABLE, UNAVAILABLE):
            raise ValueError(f"account_list must be None or one of {', '.join(sorted((*_UNAVAILABLE, UNAVAILABLE)))}")
        self.now = datetime.now(timezone.utc).replace(microsecond=0)
        # job id -> [(seconds after start, status, reason)]
        self._timeline: dict = {}
        self._run_steps: dict = {}  # job id -> the epoch of each step of its training run, computed once
        self._rng = random.Random(7)
        self._account_list = account_list
        self._location = DEMO_LOCATION
        self.database, self.schema = DEMO_LOCATION
        self._jobs: list = []
        self._kinds: dict = {}
        self._extra: dict = {}  # job id -> its spec's overrides
        for status, reason, age, duration, hw, subs, kind, *extra in _SPECS:
            jid = str(uuid.UUID(int=self._rng.getrandbits(128), version=4))
            created = self.now - age
            ended = created + duration if duration else None
            updated = ended or self.now - timedelta(seconds=self._rng.randint(4, 40))
            counts: dict = {}
            sub_jobs = []
            for sj in subs:
                t = views.sub_job_kind(sj)
                i = counts.get(t, 0)
                counts[t] = i + 1
                sub_jobs.append({"sub_job_id": f"{jid}:{t}:{i}", "status": status, **json.loads(json.dumps(sj))})
            self._jobs.append({
                "job_id": jid, "status": status, "reason": reason, "created_at": _iso(created),
                "updated_at": _iso(updated), "state_changed_at": _iso(ended or updated), "sub_jobs": sub_jobs,
                "image_tag": "release_demo", "hardware": hw, "owner_role": "ML_RESEARCH",
                "submitted_by": self._rng.choice(["USER_A", "USER_B", "USER_A", "USER_C"]),
                "idle_timeout_seconds": 1800, "pending_timeout_seconds": 86400,
            })
            self._kinds[jid] = kind
            self._extra[jid] = extra[0] if extra else {}
        for jid, more in self._extra.items():
            job = self._find(jid)
            job.update({key: more[key] for key in ("reason_details",) if key in more})
            if "multiplex_onto" in more:
                host = self._jobs[more["multiplex_onto"]]["job_id"]
                for sj in job["sub_jobs"]:  # its training sub-job runs on the host's
                    if isinstance(sj.get("training_config"), dict):
                        sj["training_config"]["multiplex_job_id"] = f"{host}:training:0"
        # The placing job starts up and runs; the team's running job is cancelled.
        placing = next(j for j in self._jobs if j["status"] == "PLACING")
        self._timeline[placing["job_id"]] = [(45, "INITIALIZING", ""), (100, "RUNNING", "")]
        team = next(j for j in self._jobs if self._extra[j["job_id"]].get("location") == _TEAM and j["status"] == "RUNNING")
        self._timeline[team["job_id"]] = [(300, "CANCELLED", "cancelled_by_client")]

    def _clock(self) -> datetime:
        return datetime.now(timezone.utc)

    def _steps_reached(self, job_id: str) -> list:
        elapsed = (self._clock() - self.now).total_seconds()
        return [step for step in self._timeline.get(job_id, []) if step[0] <= elapsed]

    def _as_of_now(self, job: dict) -> dict:
        """``job`` with the last state change its timeline has reached (see the class docstring)."""
        reached = self._steps_reached(job["job_id"])
        if not reached:
            return job
        offset, status, reason = reached[-1]
        at = _iso(self.now + timedelta(seconds=offset))
        return {**job, "status": status, "reason": reason, "state_changed_at": at, "updated_at": at,
                "sub_jobs": [{**sj, "status": status} for sj in job["sub_jobs"]]}

    def _started(self, job: dict) -> datetime:
        """When the job's sub-jobs started (their logs and GPU samples begin): its creation, or
        the moment its timeline set it running."""
        running = [offset for offset, status, _ in self._steps_reached(job["job_id"]) if status == "RUNNING"]
        if running:
            return self.now + timedelta(seconds=running[0])
        return datetime.fromisoformat(job["created_at"].replace("Z", "+00:00"))

    def _find(self, job_id: str) -> dict:
        for j in self._jobs:
            if j["job_id"] == job_id:
                return j
        raise KeyError(job_id)

    def _here(self, job_id: str) -> bool:
        """Whether the job is in the location this client is bound to."""
        return self._extra[job_id].get("location", DEMO_LOCATION) == self._location

    def _record(self, job_id: str) -> dict:
        """A job in this client's location."""
        job = self._find(job_id)
        if not self._here(job_id):
            raise KeyError(job_id)
        return job

    def _on_stage(self, job_id: str) -> dict:
        """Stage-visible state, excluding multiplexed sub-jobs' borrowed GPUs."""
        job = self._as_of_now(self._record(job_id))
        return {**job, "sub_jobs": [sj for sj in job["sub_jobs"] if not views.multiplex_target(sj)]}

    def _served(self, job: dict) -> dict:
        """A copy of ``job`` as the service returns it.

        For a job whose control plane it can't reach, the service answers its
        UNKNOWN fallback: the stored configs with no sub-job ids, no times, and
        a reason sentence.
        """
        out = json.loads(json.dumps(self._as_of_now(job)))
        more = self._extra[job["job_id"]]
        if not (more.get("unreachable") or more.get("not_found")):
            return out
        fallback = {key: out[key] for key in ("job_id", "owner_role", "submitted_by") if key in out}
        if more.get("not_found"):  # its control plane doesn't know it: no sub-jobs either
            fallback.update(status="UNKNOWN", reason=_NOT_FOUND, sub_jobs=[])
            return fallback
        fallback.update(status="UNKNOWN", reason="Job status unavailable: Control Plane is unavailable")
        fallback["sub_jobs"] = [{**{k: val for k, val in sj.items() if k != "sub_job_id"}, "status": "UNKNOWN"}
                                for sj in out["sub_jobs"]]
        return fallback

    def _check_experiment(self, job: dict) -> None:
        """Raise what reading ``job``'s Experiment raises for this role, if it may not."""
        how = self._extra[job["job_id"]].get("experiment")
        if how == "missing":
            raise _http_error(500, _EXPERIMENT_GONE.format(job=job["job_id"]))
        if how == "sql":
            database, schema = self._extra[job["job_id"]].get("location", DEMO_LOCATION)
            raise RuntimeError(_EXPERIMENT_SQL.format(database=database, schema=schema, name="TRAINING_RUNS"))

    # ── client surface used by the TUI ──
    def list_jobs(self, status: Optional[str] = None) -> list:
        """This client's location's jobs, as the per-schema list returns them (no location fields).

        ``status`` is accepted, as the real client takes it, and unused (the job list sends none).
        """
        return [self._served(j) for j in self._jobs if self._here(j["job_id"])]

    def get_job(self, job_id: str) -> dict:
        return self._served(self._record(job_id))

    def _account_row(self, job: dict) -> dict:
        """``job`` as the account-wide list returns it: with its location, and its
        ``created_at`` even while its status is UNKNOWN."""
        row = self._served(job)
        if views.is_unknown(row):
            row["created_at"] = job["created_at"]
        database, schema = self._extra[job["job_id"]].get("location", DEMO_LOCATION)
        return {**row, "database_name": database, "schema_name": schema}

    def list_jobs_page(self, *, database: Optional[str] = None, schema: Optional[str] = None,
                       status: Any = None, submitted_by: Optional[str] = None, created_after: Any = None,
                       created_before: Any = None, page_size: int = 100, page_token: Optional[str] = None,
                       max_retries: Optional[int] = None, request_timeout: Any = None) -> dict:
        """One page of the account-wide list: every location, newest first, a few rows per page.

        Only what the job list sends applies: ``created_after``, and ``status`` as the active group (a
        job whose status can't be read never matches a status filter).
        """
        if self._account_list == UNAVAILABLE:
            raise _http_error(503, "Control Plane unavailable")
        if self._account_list is not None:
            raise AccountJobListUnavailable(_UNAVAILABLE[self._account_list], reason=self._account_list)
        newest = sorted(self._jobs, key=lambda j: j["created_at"], reverse=True)
        rows = [row for row in map(self._account_row, newest)
                if (created_after is None or views.parse_ts(row["created_at"]) >= created_after)
                and (not status or views.status_word(row.get("status")) in _ACTIVE)]
        start = int(page_token or 0)
        page = rows[start:start + min(int(page_size), self._ACCOUNT_PAGE_ROWS)]
        end = start + len(page)
        return {"jobs": page, "next_page_token": str(end) if end < len(rows) else None}

    def for_job(self, job: dict) -> "DemoClient":
        """A view of this client bound to ``job``'s database and schema, as ``CortexTrainingClient.for_job``."""
        view = copy.copy(self)  # shares the jobs, stores and caches
        view._location = (job["database_name"], job["schema_name"])
        view.database = views.sql_identifier(job["database_name"])
        view.schema = views.sql_identifier(job["schema_name"])
        return view

    # The sampler reads every GPU every 5 s, as the service's does. A long job's file
    # keeps every sample of its last hour (the windows a job opens on, and the
    # interval the tab reports, come from those) and every stride-th one before,
    # so the demo's files stay small; each row still says 5 s, the sampler's interval.
    _SAMPLE_SECONDS = 5.0
    _DENSE_SECONDS = 3600.0
    _OLDER_SAMPLES = 1680
    # A node holds this many of a sub-job's GPUs, so an 8-GPU sub-job runs on two
    # nodes. Each node also reports a system row (CPU and memory) every scrape,
    # with this many CPUs and this much memory per GPU, and misses about one
    # scrape in this many (never two in a row), as real nodes do.
    _GPUS_PER_NODE = 4
    _CPUS_PER_GPU = 22
    _MEMORY_PER_GPU = 215 * _GIB
    _MISSED_SCRAPE = 40

    def _sample_times(self, total: float) -> list:
        """Seconds after the first sample of each sample a sub-job's file holds."""
        interval = self._SAMPLE_SECONDS
        dense_from = max(0.0, total - self._DENSE_SECONDS)
        stride = interval * max(1, math.ceil(dense_from / self._OLDER_SAMPLES / interval))
        older = [k * stride for k in range(int(dense_from // stride) + 1) if k * stride < dense_from]
        first = math.ceil(dense_from / interval) * interval
        return older + [first + k * interval for k in range(int((total - first) // interval) + 1)]

    def download_metrics(self, job_id: str, output_dir, *, resume: bool = False, priority=(),
                         on_commit=None) -> list:
        """The client's download, generated: ``priority`` sub-jobs first, ``on_commit`` after each file."""
        job = self._on_stage(job_id)
        self._check_experiment(job)  # the samples live on the job's Experiment
        created = self._started(job) + timedelta(minutes=2)
        end = self._clock() if job["status"] in ("RUNNING",) else datetime.fromisoformat(job["state_changed_at"].replace("Z", "+00:00"))
        if job["status"] in ("PLACING", "PENDING") or job.get("reason") in _NEVER_PLACED:
            return []  # never placed: no GPUs ever reported
        fb_total = _GPU_MEMORY.get(job.get("hardware"), _H200_FB)
        loading = job.get("reason") == "out_of_memory"  # it ran out of memory loading the weights
        written = []
        first = list(priority)
        ordered = sorted(job["sub_jobs"], key=lambda sj: first.index(sj["sub_job_id"]) if sj["sub_job_id"] in first else len(first))
        for sj in ordered:
            sid = sj["sub_job_id"]
            if not views.requested_gpus(sj):
                continue  # a log_prob sub-job runs on no GPUs of its own
            n_gpus = int((sj.get("training_config") or sj.get("inference_config") or {}).get("n_gpus", 1))
            sampling = views.sub_job_kind(sj) == "sampling"
            rng = random.Random("gpu" + sid)
            # The system rows draw from their own stream, so the GPU rows are the same with or without them.
            host = random.Random("system" + sid) if self._extra[job["job_id"]].get("system_metrics", True) else None
            missed: set = set()  # nodes whose previous scrape was missed
            path = Path(output_dir) / sid / "gpu.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            interval = self._SAMPLE_SECONDS
            total = (end - created).total_seconds()
            lines = []
            for t in self._sample_times(total):
                seq = int(t // interval)  # the sampler's sequence number: a skipped sample skips one
                ts = created + timedelta(seconds=t)
                stamp = ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                phase = t / max(total, 1)
                for g in range(n_gpus):
                    node, local = divmod(g, self._GPUS_PER_NODE)
                    if host is not None and local == 0:  # the node's system row comes before its GPU rows
                        on_node = min(self._GPUS_PER_NODE, n_gpus - g)
                        row = self._system_row(host, sid, node, on_node, t, phase, stamp, seq,
                                               sampling=sampling, loading=loading)
                        if host.randrange(self._MISSED_SCRAPE) or node in missed:
                            missed.discard(node)
                            lines.append(json.dumps(row))
                        else:
                            missed.add(node)
                    if loading:
                        util = max(0.0, min(100.0, 6 + rng.gauss(0, 3)))  # copying weights, not computing
                        used = (0.2 + 0.79 * phase) * fb_total + rng.gauss(0, 0.2) * _GIB
                    elif sampling:
                        busy = 0.5 + 0.5 * math.sin(t / 90.0 + g) ** 2
                        util = max(0.0, min(100.0, 35 + 50 * busy + rng.gauss(0, 6)))
                        used = 0.88 * fb_total + rng.gauss(0, 0.3) * _GIB
                    else:
                        dip = 35 if int(t / 600) % 7 == 3 else 0  # periodic checkpoint/eval stalls
                        util = max(0.0, min(100.0, 91 - dip + rng.gauss(0, 4) - 25 * math.exp(-phase * 40)))
                        used = (0.56 + 0.12 * min(1.0, phase * 30)) * fb_total + rng.gauss(0, 0.4) * _GIB
                    used = max(0.0, min(fb_total, used))
                    lines.append(json.dumps({
                        "record_version": 2, "sub_job_id": sid, "sample_ts": stamp,
                        "worker_num": node, "local_gpu_index": local, "global_gpu_index": g,
                        "gpu_uuid": f"gpu-{sid[-12:]}-{g}", "gpu_util_pct": round(util, 1),
                        "fb_used_bytes": float(used), "fb_free_bytes": float(fb_total - used),
                        "power_watts": round(180 + 5 * util, 1), "gpu_temp_c": round(38 + util / 3, 1),
                        "sm_active_pct": None, "sm_occupancy_pct": None, "tensor_active_pct": round(util * 0.6, 1),
                        "dram_active_pct": round(util * 0.4, 1), "seq": seq, "seq_epoch": "demo",
                        "sample_interval_s": interval,
                    }))
            data = ("\n".join(lines) + "\n").encode("utf-8")
            path.write_bytes(data)
            # The resume manifest the real client writes; the Resources tab reads
            # only the bytes it commits.
            manifest = path.with_name("." + path.name + ".manifest.json")
            manifest.write_text(json.dumps({"version": 1, "names": [f"_metrics/{sid}/gpu.demo.jsonl.gz"], "bytes": len(data)}),
                                encoding="utf-8")
            written.append({"sub_job_id": sid, "path": str(path)})
            if on_commit is not None:
                on_commit(dict(written[-1]))
        return written

    def _system_row(self, rng: random.Random, sid: str, node: int, gpus: int, t: float, phase: float, stamp: str,
                    seq: int, *, sampling: bool, loading: bool) -> dict:
        """One node's system row at one scrape, in the service's shape (record version 3)."""
        cpus = self._CPUS_PER_GPU * gpus
        limit = self._MEMORY_PER_GPU * gpus
        if loading:  # copying the weights in
            cpu = 28 + 5 * node + rng.gauss(0, 3)
            used = (0.1 + 0.85 * phase) * limit
        elif sampling:
            cpu = 7 + 2.5 * node + 3 * math.sin(t / 150 + 1.7 * node) + rng.gauss(0, 0.6)
            used = (0.33 + 0.03 * math.sin(t / 420 + node)) * limit + rng.gauss(0, 0.4) * _GIB
        else:
            checkpointing = int(t / 600) % 7 == 3  # the GPU rows' periodic stalls
            cpu = (3.5 + 1.2 * node + 1.5 * math.sin(t / 240 + 1.3 * node) + (6 if checkpointing else 0)
                   + rng.gauss(0, 0.4))
            used = ((0.22 + 0.12 * min(1.0, phase * 30) + 0.02 * math.sin(t / 600 + node)) * limit
                    + rng.gauss(0, 0.5) * _GIB)
        return {
            "kind": "system", "record_version": 3, "sub_job_id": sid, "sample_ts": stamp, "worker_num": node,
            "seq": seq, "seq_epoch": "demo", "sample_interval_s": self._SAMPLE_SECONDS,
            "cpu_util_pct": round(max(0.1, min(100.0, cpu)), 1), "cpu_count": float(cpus),
            "mem_used_bytes": float(int(max(0.0, min(limit, used)))), "mem_total_bytes": float(limit),
        }

    # ── logs: line i of a sub-job is always the same line, so a cursor is an index ──
    def _log_end(self, job: dict) -> datetime:
        """Where a sub-job's log ends for now: the clock while the job runs, else when it ended."""
        if views.status_word(job.get("status")) in ("running", "unknown"):
            return self._clock()
        return datetime.fromisoformat(job["state_changed_at"].replace("Z", "+00:00"))

    def _startup(self, job: dict, sj: dict) -> list:
        """(seconds after the sub-job started, level, logger, message) of its startup lines."""
        cfg = sj.get("training_config") or sj.get("inference_config") or {}
        if views.sub_job_kind(sj) == "log_prob":  # no GPUs of its own: it scores on the training sub-job's
            return [(2.0, "INFO", "zone", "log_prob worker scheduled beside training:0 (no GPUs of its own)"),
                    (60.0, "INFO", "loader", f"loading reference {sj['model_name']} weights (bfloat16)"),
                    (130.0, "INFO", "log_prob", "reference model ready")]
        n = int(cfg.get("n_gpus", 1))
        lines = [(2.0, "INFO", "zone", f"placed on {n} × {job['hardware']} (zone ZONE_A)"),
                 (40.0, "INFO", "ray", f"ray head ready; {max(1, n // 8)} worker node(s) joined"),
                 (48.0, "INFO", "loader", f"loading {sj['model_name']} weights (bfloat16)"),
                 (122.0, "INFO", "loader", "weights loaded in 74.2s")]
        if views.sub_job_kind(sj) == "sampling":
            lines.append((140.0, "INFO", "vllm", f"engine ready: max_model_len={int(cfg.get('max_seq_len', 4096))} "
                                                 f"gpu_memory_utilization={cfg.get('gpu_memory_utilization', 0.85)}"))
        else:
            lines.append((128.0, "INFO", "deepspeed", "ZeRO stage 2 · activation checkpointing on · flash_attention_3"))
        return lines

    def _period(self, job: dict, sj: dict) -> float:
        """Seconds between periodic lines: a sampling engine's batches, or a training step,
        spaced as the steps of the job's run are (an RL step takes about 45 s)."""
        if self._extra[job["job_id"]].get("log_every"):
            return float(self._extra[job["job_id"]]["log_every"])
        if views.sub_job_kind(sj) in ("sampling", "log_prob"):
            return self._GENERATE_SECONDS
        times = self._step_times(job["job_id"])
        if len(times) > 1:
            return max(self._STEP_SECONDS, (times[-1] - times[0]) / (len(times) - 1))
        return 45.0 if self._kinds.get(job["job_id"]) == "rl" else self._STEP_SECONDS

    def _closing(self, job: dict) -> list:
        if views.status_word(job.get("status")) != "failed":
            return []
        return [("ERROR", "worker", "rank 3: CUDA error: an illegal memory access was encountered"),
                ("ERROR", "zone", "sub-job readiness probe failed 3 times; tearing down")]

    def _periodic(self, job: dict, sj: dict) -> tuple:
        """(epochs of the lines that are the steps of the job's run, epoch of the first line
        after them, seconds between lines). A training sub-job logs one step line per step
        of its run; then, and without a run, one line every period."""
        period = self._period(job, sj)
        times = self._step_times(job["job_id"]) if views.sub_job_kind(sj) == "training" else []
        if times:
            return times, times[-1] + period, period
        startup = self._startup(job, sj)
        return [], (self._started(job) + timedelta(seconds=startup[-1][0] + period)).timestamp(), period

    def _log_count(self, job: dict, sj: dict) -> int:
        times, after, period = self._periodic(job, sj)
        end = self._log_end(job).timestamp()
        periodic = bisect.bisect_right(times, end)
        if end >= after:
            periodic += int((end - after) // period) + 1
        return len(self._startup(job, sj)) + periodic + len(self._closing(job))

    def _step_times(self, job_id: str) -> list:
        """The epoch of each step of the job's training run, in order (none for a job without a kind)."""
        if job_id not in self._run_steps:
            kind = self._kinds.get(job_id)
            times: list = []
            if kind in ("sft", "rl", "sft-fail"):
                job = self._find(job_id)
                start = datetime.fromisoformat(job["created_at"].replace("Z", "+00:00")) + timedelta(minutes=6)
                end = datetime.fromisoformat(job["state_changed_at"].replace("Z", "+00:00"))
                if job["status"] == "RUNNING":
                    end = self.now - timedelta(seconds=20)
                span = max(60.0, (end - start).total_seconds())
                steps = int(min(400, span / 45)) if kind == "rl" else int(min(1500, span / 6))
                if kind == "sft-fail":
                    steps = min(steps, 140)
                times = [(start + timedelta(seconds=span * (s / max(1, steps - 1)))).timestamp() for s in range(steps)]
            self._run_steps[job_id] = times
        return self._run_steps[job_id]

    def _stuck_lines(self, job: dict, sj: dict) -> list:
        """Indices of the lines of ``sj``'s log that are 300 KiB or more (see _SPECS):
        one line every _STUCK_EVERY from _STUCK_AFTER after the demo started, on the clock."""
        if not self._extra[job["job_id"]].get("stuck"):
            return []
        times, after, period = self._periodic(job, sj)
        skip = len(self._startup(job, sj)) + len(times)
        out = []
        at = self.now + _STUCK_AFTER
        while at <= self._clock() + _STUCK_EVERY:
            out.append(skip + max(0, math.ceil((at.timestamp() - after) / period)))
            at += _STUCK_EVERY
        return out

    def _step_line(self, job: dict, sj: dict, k: int, at: datetime) -> tuple:
        """The k-th periodic line: a generate batch, a reference log-prob batch, or a
        training step that follows the job's run and config."""
        rng = random.Random(f"log{sj['sub_job_id']}:{k}")
        kind = views.sub_job_kind(sj)
        if kind == "log_prob":
            return "INFO", "log_prob", (f"scored batch={rng.choice([16, 32])} sequences "
                                        f"tokens={rng.randint(40000, 130000)} in {rng.uniform(3.5, 9.0):.1f}s")
        if kind == "sampling":
            if self._extra[job["job_id"]].get("log_every"):  # a batch eval that logs each request
                return "INFO", "vllm", (f"request {k:>6}: prompt_tokens={rng.randint(200, 2400)} "
                                        f"completion_tokens={rng.randint(20, 512)} "
                                        f"latency={rng.uniform(0.4, 3.9):.2f}s")
            trains = any(views.sub_job_kind(other) == "training" for other in job["sub_jobs"])
            if trains and k % 12 == 11:  # only an RL job's engine gets weights from its trainer
                return "INFO", "weight-sync", "received weights from training:0 (broadcast, 4.1s); prefix cache reset"
            return "INFO", "vllm", (f"generate batch={rng.choice([32, 64, 128])} prompt_tokens={rng.randint(9000, 30000)} "
                                    f"throughput={rng.randint(5200, 7400)} tok/s")
        if k % 37 == 20:
            return "WARNING", "train", "dataloader stalled 11.8s waiting for next batch (consider num_workers>4)"
        # The k-th line is the run's step k, and past its last step (the run stops when the
        # demo starts) the steps go on; a job without a run logs every tenth step.
        step = k if self._step_times(job["job_id"]) else k * 10
        lr = float(sj["training_config"]["optimizer"]["lr"])
        if self._kinds.get(job["job_id"]) == "rl":
            # A reward that climbs, a KL that grows, and a policy loss that eases off, with some noise.
            reward = 0.18 + 0.55 * (1 - math.exp(-k / 150)) + rng.gauss(0, 0.025)
            kl = 0.002 + 0.03 * (1 - math.exp(-k / 300)) + abs(rng.gauss(0, 0.003))
            loss = 0.04 * math.exp(-k / 150) + rng.gauss(0, 0.006)
            return "INFO", "train", (f"step {step:>4}  reward={reward:.3f}  kl={kl:.4f}  policy_loss={loss:.4f}  "
                                     f"lr={lr:.2e}")
        # A loss that eases off with some noise.
        loss = 0.38 + 1.9 * math.exp(-k / 150) + abs(rng.gauss(0, 0.03))
        grad = rng.uniform(0.3, 1.2)
        lr_now = lr * min(1.0, step / 50)
        return "INFO", "train", (f"step {step:>4}  loss={loss:.4f}  grad_norm={grad:.3f}  lr={lr_now:.2e}  "
                                 f"tok/s={rng.randint(20500, 22800)}")

    def _log_entry(self, job: dict, sj: dict, i: int) -> dict:
        """Line ``i`` of the sub-job's log as the service sends it: ``{"_raw": "<RFC 3339 ns> <JSON record>"}``."""
        startup = self._startup(job, sj)
        start = self._started(job)
        closing = self._closing(job)
        if i < len(startup):
            offset, level, logger, msg = startup[i]
            at = start + timedelta(seconds=offset)
        elif closing and i >= self._log_count(job, sj) - len(closing):
            level, logger, msg = closing[i - (self._log_count(job, sj) - len(closing))]
            at = self._log_end(job)
        else:
            k = i - len(startup)
            times, after, period = self._periodic(job, sj)
            epoch = times[k] if k < len(times) else after + (k - len(times)) * period
            at = datetime.fromtimestamp(epoch, tz=timezone.utc)
            level, logger, msg = self._step_line(job, sj, k, at)
            if i in self._stuck_lines(job, sj):  # (a saved console keeps it, cut short here)
                level, logger, msg = "INFO", "debug", "activations of layer 31: [0.0132, -0.0871, ...] (300 KiB)"
        record = json.dumps({"level": level, "logger": logger, "msg": msg}, ensure_ascii=False)
        return {"_raw": f"{at.astimezone(timezone.utc):%Y-%m-%dT%H:%M:%S.%f}000Z {record}"}  # RFC 3339, ns

    # Jobs that failed before their head pod wrote a console: no saved console,
    # and (once placed) an execution log that records the error.
    _NO_CONSOLE_REASONS = {"out_of_memory", *_NEVER_PLACED}
    # How the service says a job is no longer running, by its status word.
    _ENDED = {"failed": "has failed", "cancelled": "has been cancelled", "terminated": "has been terminated",
              "succeeded": "has completed", "done": "has completed"}

    def tail_logs(self, job_id: str, *, cursor: Optional[str] = None, max_lines: Optional[int] = None,
                  sub_job_id: Optional[str] = None) -> dict:
        if self._extra[job_id].get("not_found"):
            raise _http_error(503, _NOT_FOUND)  # its control plane doesn't know it: no session to tail
        job = self._as_of_now(self._record(job_id))  # (served UNKNOWN rows carry no sub-job ids)
        word = views.status_word(job["status"])
        if word in self._ENDED:
            # As the service does once a job is no longer running.
            reason = f": {job['reason']}" if job.get("reason") else ""
            raise _http_error(409, f"Cortex training operation failed: session {job_id} {self._ENDED[word]}{reason}")
        sj = next((s for s in job["sub_jobs"] if s["sub_job_id"] == sub_job_id), job["sub_jobs"][0])
        if str(sj.get("status") or job["status"]).upper() in ("PLACING", "INITIALIZING"):
            if sj is not job["sub_jobs"][0]:
                # A sibling the service isn't routing to until the job runs.
                raise _http_error(400, f"Cortex training operation failed: sub_job_id {sj['sub_job_id']} "
                                       f"does not belong to session {job_id}")
            # As the service does until the head pod is up: an empty page, same cursor.
            return {"entries": [], "next_cursor": cursor or "", "eof": False}
        total = self._log_count(job, sj)
        if cursor:  # from the cursor; max_lines caps the page
            first = min(int(cursor), total)
            stop = total if max_lines is None else min(total, first + max_lines)
        else:  # no cursor: the newest max_lines lines (or the whole log)
            stop = total
            first = max(0, total - max_lines) if max_lines else 0
        # A line of 300 KiB or more is never delivered: the page stops before it, and
        # so does the cursor (a read of the end whose window starts after it gets past).
        stop = next((n for n in self._stuck_lines(job, sj) if first <= n < stop), stop)
        return {"entries": [self._log_entry(job, sj, i) for i in range(first, stop)], "next_cursor": str(stop),
                "eof": False}

    def download_stdout_logs(self, job_id: str, output_dir, *, resume: bool = False) -> list:
        job = self.get_job(job_id)
        self._check_experiment(self._find(job_id))  # the saved consoles live on the job's Experiment
        if job["reason"] in self._NO_CONSOLE_REASONS:
            return []
        for sj in job["sub_jobs"]:
            path = Path(output_dir) / sj["sub_job_id"] / "stdout.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = (format_log_entry(self._log_entry(job, sj, i)) for i in range(self._log_count(job, sj)))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return []

    def fetch_execution_logs(self, job_id: str, *, sub_job_id: Optional[str] = None,
                             filename: Optional[str] = None) -> list:
        job = self.get_job(job_id)
        self._check_experiment(self._find(job_id))  # the execution logs live on the job's Experiment
        if job["reason"] != "out_of_memory" or filename not in (None, "execution.jsonl"):
            return []  # a job that never left pending has no execution log
        created = datetime.fromisoformat(job["created_at"].replace("Z", "+00:00"))
        out = []
        for sj in job["sub_jobs"]:
            sid = sj["sub_job_id"]
            if sub_job_id is not None and sid != sub_job_id:
                continue

            def at(minutes: float) -> str:
                return _iso(created + timedelta(minutes=minutes))

            records = [
                {"ts": at(1), "level": "INFO", "logger": "runtime", "msg": "init phase: placement", "job_id": sid},
                {"ts": at(3), "level": "INFO", "logger": "runtime", "msg": f"init phase: load_model {sj['model_name']}",
                 "job_id": sid},
                {"ts": at(6.5), "level": "ERROR", "logger": "runtime", "msg": f"failed to create job {sid}", "job_id": sid,
                 "op": "create-job", "exc": (
                     "Traceback (most recent call last):\n"
                     '  File "loader.py", line 212, in load_weights\n'
                     "    shard = read_shard(path, device)\n"
                     "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.50 GiB (GPU 3; 79.6 GiB total)\n"
                     "\n"
                     "The above exception was the direct cause of the following exception:\n"
                     "\n"
                     "Traceback (most recent call last):\n"
                     '  File "manager.py", line 88, in create_job\n'
                     "    self._load(model)\n"
                     "RuntimeError: loading the model failed: CUDA out of memory")},
                {"ts": at(6.5), "level": "INFO", "logger": "runtime", "msg": "job_metrics", "job_id": sid, "op": "create-job",
                 "duration_s": 331.4, "error": "RuntimeError: loading the model failed: CUDA out of memory"},
            ]
            out.append({"sub_job_id": sid, "filename": "execution.jsonl",
                        "artifact_uri": f"snow://experiment/DEMO/versions/RUN/_logs/{sid}/execution.jsonl",
                        "content": "".join(json.dumps(r) + "\n" for r in records)})
        return out

