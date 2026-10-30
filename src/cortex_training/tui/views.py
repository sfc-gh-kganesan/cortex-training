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

"""Pure view-model helpers for the TUI's job list, summary, and detail tabs.

Search, sorting, and summary fields. Like ``format.py`` this never imports
``textual``, so every rule here is unit-testable without the TUI installed.
"""

from __future__ import annotations

import functools
import json
import math
import re
import shlex
import unicodedata
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from typing import Iterable
from typing import Optional

import requests
from rich.cells import cell_len

from cortex_training.client.core import _deployment_id_from_job_id
from cortex_training.client.core import _model_deployment_status
from cortex_training.client.core import _model_deployment_sub_job
from cortex_training.client.core import create_job_body_from_job
from cortex_training.model_deployment import DEPLOYMENT_STATUSES_WITHOUT_ENDPOINT
from cortex_training.model_deployment import WIRE_DEPLOYMENT_CLASS
from cortex_training.model_deployment import WIRE_INFERENCE_CONFIG
from cortex_training.model_deployment import WIRE_NUM_REPLICAS
from cortex_training.tui.format import _ACTIVE_STATUSES as ACTIVE_STATUSES
from cortex_training.tui.format import _num as compact_number
from cortex_training.tui.format import cut_cells
from cortex_training.tui.format import _subjob_config as subjob_config
from cortex_training.tui.format import finite_number as _as_float
from cortex_training.tui.format import format_created_at
from cortex_training.tui.format import local_time
from cortex_training.tui.format import parse_ts  # noqa: F401 - views.parse_ts for the screens
from cortex_training.tui.format import printable
from cortex_training.tui.format import source_label

RETENTION_DAYS = 183  # six months of job history


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "—"
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# A time at most this far ahead of this computer's clock reads "just now": the
# service's clock and this one rarely agree to the second.
CLOCK_SKEW_SECONDS = 300


def fmt_ago(raw: Any, now: datetime) -> str:
    """How long before ``now`` the time ``raw`` was, e.g. ``4m 05s ago``; ``—`` without one."""
    dt = raw if isinstance(raw, datetime) else parse_ts(raw)
    if not dt:
        return "—"
    seconds = (now - dt).total_seconds()
    if seconds < 0:
        if -seconds <= CLOCK_SKEW_SECONDS:
            return "just now"
        return f"{fmt_duration(-seconds)} ahead of this computer's clock"
    return f"{fmt_duration(seconds)} ago"


# The job list's times are read at a glance, so they are coarse: one unit, in a
# column 8 cells wide. Each helper also says how many seconds its text stays
# the same, so the list rebuilds a row only once its text would change.


def list_age(seconds: Optional[float]) -> tuple[str, float]:
    """How long ago, as the job list's Created and Updated show it (``just now``,
    ``12m ago``, ``3h ago``, ``45d ago``), and the seconds until that reads
    differently; ``—`` (never changing) without a time.

    A time slightly ahead of this computer's clock (see :data:`CLOCK_SKEW_SECONDS`)
    is ``just now``; one further ahead reads ``in 2h``.
    """
    if seconds is None:
        return "—", math.inf
    if seconds < 0:
        if -seconds <= CLOCK_SKEW_SECONDS:
            return "just now", 60 - seconds  # until a minute after it
        return f"in {list_age(-seconds)[0].removesuffix(' ago')}", 60.0
    s = int(seconds)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= size:
            return f"{s // size}{unit} ago", size - seconds % size
    return "just now", 60 - seconds


def list_span(seconds: Optional[float]) -> tuple[str, float]:
    """A job's run so far or its whole run, as the job list's Duration shows it
    (``40s``, ``12m``, ``3h 05m``, ``2d 4h``), and the seconds until a span that
    goes on growing reads differently; ``—`` without one."""
    if seconds is None or seconds < 0:
        return "—", math.inf
    s = int(seconds)
    d, rest = divmod(s, 86400)
    h, rest = divmod(rest, 3600)
    m, sec = divmod(rest, 60)
    if d:
        return f"{d}d {h}h", 3600 - seconds % 3600
    if h:
        return f"{h}h {m:02d}m", 60 - seconds % 60
    if m:
        return f"{m}m", 60 - seconds % 60
    return f"{sec}s", 1 - seconds % 1


def status_word(status: Any) -> str:
    return (str(status or "").strip() or "?").removeprefix("JOB_STATE_").lower()


def describe_error(exc: BaseException) -> str:
    """A failed call for display, on one printable line: the HTTP status and the
    service's message, not the URL.

    requests' own text ("500 Server Error: ... for url: <full URL>") repeats the
    host, database and schema and never includes the service's explanation.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(exc, requests.exceptions.Timeout):
        return _timeout_text(exc)
    if not isinstance(status, int):
        return collapse(f"{type(exc).__name__}: {exc}")
    try:
        body = response.json()
        message = str(body.get("message") or body.get("error") or "") if isinstance(body, dict) else ""
    except Exception:  # noqa: BLE001 - not JSON
        message = ""
    reason = str(getattr(response, "reason", "") or "")
    return collapse(f"HTTP {status}" + (f" {reason}" if reason else "") + (f": {message}" if message else ""))


# How long a timed-out request waited, as urllib3 says it: "(read timeout=45.0)".
_TIMEOUT_SECONDS = re.compile(r"(?:read|connect) timeout=([0-9]+(?:\.[0-9]+)?)")


def _timeout_text(exc: BaseException) -> str:
    """A request that timed out, in a few words: how long it waited, not the connection pool's text."""
    found = _TIMEOUT_SECONDS.search(str(exc))
    within = f" within {float(found.group(1)):g} s" if found else " in time"
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return f"couldn't connect to the service{within} (ConnectTimeout)"
    return f"no answer from the service{within} ({type(exc).__name__})"


# The service's 500 when it can't resolve the job's Experiment for the caller: the
# checkpoint routes say "... or not authorized for the training job", and the
# experiment-run route that saved logs and GPU metrics start with says "... for
# job <id>", which a deleted Experiment gets too.
EXPERIMENT_ACCESS_MARKER = "Linked experiment is not found or not authorized"
_EXPERIMENT_NOT_FOUND = "Linked experiment is not found"
_ASK = "ask for access to the Experiment, or use the role that created the job"


def experiment_access_hint(exc: BaseException) -> str | None:
    """The one-line explanation when ``exc`` means this role can't read the job's Experiment, else None.

    The service's message is in the HTTP reply, not in ``str(exc)``, so both are
    read. A refusal about another object (a warehouse, a schema) is None even
    when its message names the Experiment.
    """
    text = f"{exc}\n{_reply_text(exc)}"
    if EXPERIMENT_ACCESS_MARKER in text:
        return f"this role can't read the job's Experiment (saved logs, GPU metrics and checkpoints live there); {_ASK}"
    if _EXPERIMENT_NOT_FOUND in text:  # the experiment-run route's: hidden from this role, or gone
        return ("this role can't read the job's Experiment, or it no longer exists (saved logs, GPU metrics and "
                f"checkpoints live there); {_ASK}")
    if _EXPERIMENT_SQL_DENIED.search(text):  # a MONITOR-only role: the service still lists the checkpoints
        return f"this role can't read the job's Experiment, where saved logs and GPU metrics live (the checkpoints command works); {_ASK}"
    return None


def is_live(job: Any) -> bool:
    """Whether a job is active, including UNKNOWN unless it is not found."""
    word = status_word(job.get("status")) if isinstance(job, dict) else ""
    return word in ACTIVE_STATUSES or (word == "unknown" and not job_not_found(job))


def job_not_found(job: Any) -> bool:
    """An UNKNOWN job whose reason says the control plane has no record of it."""
    return is_unknown(job) and "not found" in " ".join(collapse(job.get(key) or "") for key in ("reason", "reason_details")).lower()


def job_id_of(job: dict) -> str:
    return str(job.get("job_id") or job.get("id") or "")


def job_key(job: dict, home: Optional[tuple] = None) -> tuple:
    """A job's location-aware identity, including its configured location when omitted."""
    location = location_names(job) or home or (None, None)
    return (*location, job_id_of(job))


def is_unknown(job: Any) -> bool:
    """The service's UNKNOWN: it could not read the job's live state from the control plane."""
    return isinstance(job, dict) and status_word(job.get("status")) == "unknown"


def collapse(text: Any) -> str:
    """``text`` printable on one line: each run of whitespace, newlines too, as one space."""
    return " ".join(printable(text if text is not None else "").split())


def printable_data(value: Any) -> Any:
    """``value`` (decoded JSON) with every string in it, dict keys too, made :func:`printable`."""
    if isinstance(value, str):
        return printable(value)
    if isinstance(value, dict):
        return {printable_data(key): printable_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [printable_data(item) for item in value]
    return value


def cell_width(text: str) -> int:
    """The terminal cells ``text`` takes: wide (CJK, emoji) characters take two."""
    return cell_len(text)


def fit_cells(text: str, width: int) -> str:
    """``text`` cut to at most ``width`` terminal cells, ending in ``…`` when cut
    (see :func:`cortex_training.tui.format.cut_cells`)."""
    if cell_len(text) <= width:
        return text
    return cut_cells(text, width - 1).rstrip() + "…"


def wrap_cells(line: str, width: int) -> list:
    """Cell-aware word wrapping that preserves first-line indentation."""
    width = max(2, width)
    body = line.lstrip(" ")
    indent = line[: len(line) - len(body)]
    row = indent if cell_len(indent) < width else ""
    rows: list = []
    for token in re.findall(r"\S+|\s+", body):
        if token.isspace():
            if row.strip():
                row += token  # between words; dropped where the row breaks
            continue
        while token:
            if cell_len(row) + cell_len(token) <= width:
                row += token
                break
            if row.strip():  # the word starts the next row
                rows.append(row.rstrip())
                row = ""
                continue
            head = cut_cells(token, width - cell_len(row))
            if not head:  # not even one character fits after the indent
                row = ""
                head = cut_cells(token, width) or token[0]
            rows.append(row + head)
            row, token = "", token[len(head):]
    rows.append(row.rstrip())
    return rows


# Sub-job type as the id's middle segment: the create-time type
# ``log_probability`` is ``log_prob`` there.
_ID_TYPES = {"training": "training", "sampling": "sampling", "log_probability": "log_prob", "log_prob": "log_prob"}


def sub_job_kind(sj: dict) -> str:
    """``training``, ``sampling`` or ``log_prob``; ``""`` when the entry names no known type."""
    raw = str(sj.get("job_type") or sj.get("type") or "").strip().lower().removeprefix("job_type_")
    return _ID_TYPES.get(raw, "")


def sub_jobs(job: dict) -> list:
    """Sub-jobs, deriving ``JOB_ID:TYPE:INDEX`` for complete UNKNOWN configs."""
    subs = [s for s in job.get("sub_jobs") or [] if isinstance(s, dict)]
    for sj in subs:
        if not (sj.get("sub_job_id") or sj.get("id")):
            break
    else:
        if not is_unknown(job):
            return subs  # the usual row (every entry has its id): nothing to derive or label
    kinds = [sub_job_kind(s) for s in subs]
    jid = job_id_of(job)
    if subs and is_unknown(job) and jid and all(kinds) and not any(s.get("sub_job_id") for s in subs):
        counts: dict = {}
        derived = []
        for sj, kind in zip(subs, kinds):
            derived.append({**sj, "sub_job_id": f"{jid}:{kind}:{counts.get(kind, 0)}"})
            counts[kind] = counts.get(kind, 0) + 1
        return derived
    return [sj if kind or sj.get("sub_job_id") or sj.get("id") else {**sj, "label": f"config #{i}"}
            for i, (sj, kind) in enumerate(zip(subs, kinds))]


def is_deployment(job: Any) -> bool:
    """Whether a job is a model deployment, by the CLI's own rule (the client's
    ``_model_deployment_sub_job``), so the TUI and ``ct`` never disagree."""
    return _model_deployment_sub_job(job) is not None


def deployment_class(job: Any) -> str:
    """``EVALUATION``, ...; ``""`` for a job that isn't a deployment."""
    sj = _model_deployment_sub_job(job)
    return str(sj[WIRE_INFERENCE_CONFIG].get(WIRE_DEPLOYMENT_CLASS) or "") if sj else ""


def deployment_sub_job_id(job: Any) -> Optional[str]:
    """The sub-job that serves the deployment. An UNKNOWN job's is found the way :func:`sub_jobs` finds it."""
    if _model_deployment_sub_job(job) is None:
        return None
    sid = sub_jobs(job)[0].get("sub_job_id")
    return str(sid) if sid else None


def job_tabs(job: Any) -> tuple:
    """The job screen's tabs for ``job``, in order: a deployment has Serving and no Training
    metrics or Checkpoints."""
    if is_deployment(job):
        return ("serving", "logs", "resource")
    return ("logs", "resource", "training", "checkpoints")


def status_word_for(job: Any, status: Any) -> str:
    """``status`` (the job's, or one of its sub-jobs') as the screens word it: a
    deployment's in the CLI's deployment words (``RUNNING`` -> ``ready``)."""
    if is_deployment(job) and status_word(status) != "?":  # the CLI reads no status as its UNKNOWN
        return _model_deployment_status({"status": status}).lower()
    return status_word(status)


def _call_as(job: dict) -> str:
    try:
        return _deployment_id_from_job_id(job_id_of(job))
    except ValueError:  # the client won't name this job; neither does the TUI
        return ""


def deployment_facts(job: dict) -> list:
    """The preview's deployment rows: class, replicas, idle timeout and the model name to call."""
    cfg = _model_deployment_sub_job(job)[WIRE_INFERENCE_CONFIG]
    replicas = _as_float(cfg.get(WIRE_NUM_REPLICAS))
    idle = _as_float(job.get("idle_timeout_seconds"))
    call_as = _call_as(job)
    return [
        ("Class", deployment_class(job)),
        ("Replicas", str(int(replicas)) if replicas is not None else "—"),
        # 0 disables reclamation; none set means the server's own setting.
        ("Idle timeout", "off" if idle == 0 else "server default" if idle is None else fmt_duration(idle)),
        *([("Call as", call_as)] if call_as else []),
    ]


def sub_job_type(sj: dict) -> str:
    """The normalized type shown in lists and log tags."""
    return sub_job_kind(sj) or str(sj.get("job_type") or sj.get("type") or "").strip().lower().removeprefix("job_type_")


def sub_job_label(sj: Any) -> str:
    """A printable one-line label such as ``training #0``."""
    if isinstance(sj, dict):
        sj = {**sj, "job_type": sub_job_type(sj) or None}
    return collapse(source_label(sj))


def job_types(job: dict) -> list:
    """Each sub-job's type, as :func:`sub_job_type` spells it."""
    return [t for t in (sub_job_type(s) for s in sub_jobs(job)) if t]


def _raw_job_types(job: dict) -> list:
    """Each sub-job's type as the API spells it (``log_probability``), lowercase."""
    types = [str(s.get("job_type") or s.get("type") or "").strip().lower().removeprefix("job_type_")
             for s in sub_jobs(job)]
    return [t for t in types if t]


# A job of more than one kind of sub-job: the kinds, shortened, in this order.
_SHORT_KINDS = {"training": "train", "sampling": "sample", "log_prob": "logprob"}


def job_type_label(job: dict) -> str:
    """One kind, or a compact ``train+sample``-style combination."""
    if is_deployment(job):
        return "deployment"
    kinds = list(dict.fromkeys(job_types(job)))
    if len(kinds) > 1 and all(kind in _SHORT_KINDS for kind in kinds):
        return "+".join(_SHORT_KINDS[kind] for kind in _SHORT_KINDS if kind in kinds)
    return ", ".join(kinds) or "—"


def short_model(name: Any) -> str:
    """A model name without its organization."""
    return str(name or "").split("/")[-1]


def _gpu_count(cfg: dict) -> int:
    """A finite whole ``n_gpus``, else zero."""
    count = _as_float(cfg.get("n_gpus"))
    return int(count) if count is not None else 0


_CONFIG_KEYS = {"training": "training_config", "sampling": "inference_config", "log_prob": "inference_config"}


def _type_config(sj: dict, kind: Optional[str] = None) -> dict:
    """The training or inference config for a sub-job."""
    key = _CONFIG_KEYS.get(sub_job_kind(sj) if kind is None else kind)
    cfg = sj.get(key) if key else None
    return cfg if isinstance(cfg, dict) else subjob_config(sj)


def _multiplex_id(cfg: dict) -> str:
    value = cfg.get("multiplex_job_id")
    return collapse(value) if value else ""


def multiplex_target(sj: dict) -> str:
    """The printable sub-job whose GPUs this one borrows, else empty."""
    return _multiplex_id(_type_config(sj))


def requested_gpus(sj: dict) -> int:
    """GPUs this sub-job adds to its job's request, by the control plane's rule:
    its ``n_gpus``, except that a ``log_prob`` sub-job uses none and a sub-job
    multiplexed onto another sub-job adds none (it shares that one's GPUs)."""
    kind = sub_job_kind(sj)
    if kind == "log_prob":
        return 0
    cfg = _type_config(sj, kind)
    if _multiplex_id(cfg):
        return 0
    return _gpu_count(cfg)


def total_gpus(job: dict) -> int:
    """The GPUs a job requested: :func:`requested_gpus` summed over its sub-jobs.

    This is the request, not the allocation, which the job record does not
    carry (placement can round up to whole nodes).
    """
    return sum(requested_gpus(s) for s in sub_jobs(job))


def sub_job_gpu_text(sj: dict) -> str:
    """A sub-job's GPU cell: what it requested, or ``shared`` when it runs on
    another sub-job's GPUs. A multiplexed sub-job requests none of its own,
    whatever its config's ``n_gpus`` says, which is how the job's GPU line and
    the Resources tab count it too."""
    if multiplex_target(sj):
        return "shared"
    return str(requested_gpus(sj) or ("0" if sub_job_kind(sj) == "log_prob" else "—"))


def job_gpu_cell(job: dict, unit: str = "") -> str:
    """A job's GPU cell: what it requested (``8 H200``, ``8 requested``), else
    ``shared`` when it runs only on another job's GPUs, ``0`` when its only
    sub-jobs are log_prob ones (which use none), else ``—``."""
    total = total_gpus(job)
    if total:
        return f"{total} {unit}".rstrip()
    subs = sub_jobs(job)
    if any(multiplex_target(s) for s in subs):
        return "shared"
    return "0" if any(sub_job_kind(s) == "log_prob" for s in subs) else "—"


def gpu_request_text(job: dict) -> str:
    """The preview's GPU line, e.g. ``10 × H200 requested  (8 training #0 + 2 sampling #0)``."""
    subs = sub_jobs(job)
    total = total_gpus(job)
    hw = str(job.get("hardware") or "").strip()
    shared = [s for s in subs if multiplex_target(s)]
    log_prob = [s for s in subs if sub_job_kind(s) == "log_prob"]
    if total:
        head = f"{total} × {hw} requested" if hw else f"{total} requested"
    elif shared:
        head = "no GPUs of its own requested"
    elif log_prob:
        head = "0 requested"
    else:
        return "—"
    own = [f"{requested_gpus(s)} {sub_job_label(s)}" for s in subs if requested_gpus(s)]
    notes = [" + ".join(own)] if len(own) > 1 or (own and shared) else []
    notes += [f"{sub_job_label(s)} runs on the GPUs of {multiplex_target(s)}" for s in shared]
    notes += [f"{sub_job_label(s)} uses none" for s in log_prob]
    return head + (f"  ({'; '.join(notes)})" if notes else "")


def job_models(job: dict) -> list:
    return list(dict.fromkeys(str(s["model_name"]) for s in sub_jobs(job) if s.get("model_name")))


def owner(job: dict) -> str:
    """``USER (ROLE)``; ``role ROLE`` when the job records no submitting user, so a
    role is never taken for a user name; ``—`` when neither."""
    user = collapse(job.get("submitted_by") or "")
    role = collapse(job.get("owner_role") or "")
    if user and role:
        return f"{user} ({role})"
    if role:
        return f"role {role}"
    return user or "—"


def training_method(sj: dict) -> str:
    cfg = subjob_config(sj)
    return "lora" if isinstance(cfg.get("peft_config"), dict) else "full"


def job_elapsed(job: dict, now: datetime) -> Optional[float]:
    start = parse_ts(job.get("created_at"))
    if not start:
        return None
    end = now if is_live(job) else parse_ts(job.get("state_changed_at") or job.get("updated_at")) or now
    return max(0.0, (end - start).total_seconds())


def within_retention(job: dict, now: datetime, days: int = RETENTION_DAYS) -> bool:
    created = parse_ts(job.get("created_at"))
    return created is None or (now - created).total_seconds() <= days * 86400


def job_config_json(job: dict, sub_job: Optional[dict] = None, *, experiment_name: Optional[str] = None) -> list:
    """The job's config as the lines ``cortex-training get`` prints: a CreateJob
    body that ``submit`` takes as it is, with ``experiment_name`` the job's
    experiment (see the client's ``create_job_body_from_job``). With ``sub_job``,
    one of the job's entries, the body has that sub-job alone, so a copy of it
    launches just that one. Raises ``ValueError`` for a job that cannot be copied."""
    if sub_job is not None:
        # Without the label sub_jobs gives an entry that has no id.
        job = {**job, "sub_jobs": [{key: value for key, value in sub_job.items() if key != "label"}]}
    return json.dumps(create_job_body_from_job(job, experiment_name=experiment_name),
                      indent=2, sort_keys=True).splitlines()


def preview_facts(job: dict, now: datetime, *, base_url: str = "") -> list:
    """(label, value) facts for the job preview's grid, in display order.

    The id, status and elapsed time head the preview, and its lifecycle has a
    row of its own, so they are not here. A time the job record lacks is ``—``,
    with no "ago" after it. A deployment adds its own rows (class, replicas,
    idle timeout, the model name to call) right after Type.
    """
    location = location_label(job)
    # The age only beside a time that is shown: a dash has none.
    updated, ago = format_created_at(job.get("updated_at")) or "—", fmt_ago(job.get("updated_at"), now)
    facts = [
        ("Model", ", ".join(job_models(job)) or "—"),
        ("Type", job_type_label(job)),
        ("GPUs", gpu_request_text(job)),
        ("Owner", owner(job)),
        ("Created", format_created_at(job.get("created_at")) or "—"),
        ("Last update", f"{updated}  ({ago})" if "—" not in (updated, ago) else updated),
        *([("Schema", location)] if location else []),
    ]
    if is_deployment(job):
        # The CLI's rule: a deployment that failed or is being dropped no longer serves.
        serves = _model_deployment_status(job) not in DEPLOYMENT_STATUSES_WITHOUT_ENDPOINT
        facts[2:2] = [*deployment_facts(job), *([("Base URL", base_url)] if base_url and serves else [])]
    return facts


def elapsed_text(job: dict, now: datetime) -> str:
    """The time beside the preview's status: ``12m 04s so far`` while the job is
    live, ``lasted 1h 20m`` once it ended, ``""`` without a creation time.

    It runs from creation (:func:`job_elapsed`), so it includes any time spent
    queued or placing: "lasted", not "ran", since a job may never have run.
    """
    seconds = job_elapsed(job, now)
    if seconds is None:
        return ""
    return f"{fmt_duration(seconds)} so far" if is_live(job) else f"lasted {fmt_duration(seconds)}"


def short_time(raw: Any, now: datetime) -> str:
    """A local ``YYYY-MM-DD HH:MM``, without the date when it is the same local
    day as ``now`` (``15:14``, or ``2026-10-12 16:02``); ``—`` without a time."""
    dt = raw if isinstance(raw, datetime) else parse_ts(raw)
    text = local_time(dt) if dt else ""
    if not text:
        return "—"
    return text[11:] if text[:10] == local_time(now)[:10] else text


@dataclass(frozen=True)
class LifeStep:
    """One step of a job's lifecycle on the preview."""

    status: str  # a status word (drawn with its glyph and color), or "created" / "now"
    label: str
    at: str = ""  # when the step began, "" when the job record has no time for it
    then: str = ""  # how long until the next step, "" when the record can't tell


def lifecycle(job: dict, now: datetime) -> list:
    """The :class:`LifeStep` s from the job's creation to its current state, and on
    to now while it is live.

    The job record has two times: ``created_at`` and ``state_changed_at``, when
    it entered its current status. So there are no steps for the states between
    (a running job's time pending and placing is in its first gap), and none for
    a time the record lacks; an ended job's current state began when it ended.
    An ``unknown`` job's state has no time: the service could not read it.
    """
    created = parse_ts(job.get("created_at"))
    word = status_word(job.get("status"))
    changed = parse_ts(job.get("state_changed_at")) if word != "unknown" else None
    steps = []
    if created:
        gap = (changed - created).total_seconds() if changed else None
        steps.append(LifeStep("created", "Created", short_time(created, now),
                              fmt_duration(gap) if gap is not None and gap >= 1 else ""))
    if word == "unknown":
        return [*steps, LifeStep("unknown", "status unavailable")]
    name = word.capitalize()
    if not changed:
        return [*steps, LifeStep(word, name)]
    if not is_live(job):
        return [*steps, LifeStep(word, name, f"{short_time(changed, now)} ({fmt_ago(changed, now)})")]
    so_far = (now - changed).total_seconds()
    if created and abs((changed - created).total_seconds()) < 1:
        # In this state since it was created (a job still pending): one step says both.
        steps, name = [], f"Created, {word}"
    return [*steps, LifeStep(word, f"{name} since", short_time(changed, now),
                             f"{fmt_duration(so_far)} so far" if so_far >= 0 else ""),
            LifeStep("now", "now")]


# What a waiting job is doing, by state, when the service sends no reason_details.
WAITING_TEXT = {
    "pending": "waiting to be placed on GPUs",
    "queued": "waiting to be placed on GPUs",
    "creating": "waiting to be placed on GPUs",
    "placing": "waiting for its GPUs to come up",
    "initializing": "placed on GPUs; loading the model and starting the engines",
}
_NO_LOGS = "It never got GPUs, so no sub-job started or wrote logs."


@dataclass(frozen=True)
class Callout:
    """Why a job is where it is, for the preview's callout box."""

    tone: str  # "failed", "waiting", "starting", "running", "ended" or "unknown": its color
    status: str  # the status word whose glyph leads the title
    title: str
    reason: str = ""  # the service's reason code
    details: str = ""  # the service's reason_details, or what the state means
    hint: str = ""  # where to look next


def health_callout(job: dict) -> Optional[Callout]:
    """The preview's callout for ``job``, or ``None`` for a running job the
    service says nothing more about (its status pill says it all).

    The service's ``reason`` and ``reason_details`` show the same way for every
    state; a failure always gets a reason (``not reported``).
    """
    word = status_word(job.get("status"))
    reason, details = collapse(job.get("reason") or ""), collapse(job.get("reason_details") or "")
    no_logs = _NO_LOGS if never_started(job) else ""
    if word == "failed":
        if no_logs:
            return Callout("failed", word, "Failed before it got GPUs", reason or "not reported", details, no_logs)
        return Callout("failed", word, "Why it failed", reason or "not reported", details,
                       "l opens Logs: pick the failing sub-job on the left.")
    if word in ("cancelled", "canceled"):
        return Callout("ended", word, "Cancelled", reason, details, no_logs)
    if word == "terminated":
        return Callout("ended", word, "Terminated by the platform", reason, details, no_logs)
    if word in ("succeeded", "completed", "done"):
        return Callout("ended", word, "Completed", reason, details)
    if word in WAITING_TEXT:
        # The service says why a job waits (e.g. quota_filled) and, often, how far
        # off it is; without that, what the state means.
        title = "Starting up" if word == "initializing" else "Placing on GPUs" if word == "placing" \
            else "Why it's waiting"
        return Callout("starting" if word == "initializing" else "waiting", word, title, reason,
                       details or WAITING_TEXT[word])
    if word == "unknown":
        # The service could not read the job's live state: it may still be running,
        # unless the service says the control plane has no record of the job.
        hint = "" if job_not_found(job) else "The job may still be running."
        return Callout("unknown", word, "Status unavailable", details=status_unavailable_text(job), hint=hint)
    if word == "running":
        return Callout("running", word, "Running", reason, details) if reason or details else None
    # A status this client doesn't know: the service's reason, if it sent one.
    return Callout("unknown", word, word.capitalize(), reason, details) if reason or details else None


def reason_text(job: dict) -> str:
    """``reason`` and ``reason_details`` on one printable line, ``""`` when neither.

    For example ``quota_filled — needs 8 GPUs; 64 currently active``.
    """
    parts = [collapse(job.get(key) or "") for key in ("reason", "reason_details")]
    return " — ".join(p for p in parts if p)


# Failure reasons that mean the job never got GPUs, so no sub-job started or
# wrote logs: every ``placement_*`` reason (``placement_timeout``,
# ``placement_too_large``, ``placement_failed``) and these.
PRE_START_REASONS = frozenset({"hardware_unavailable", "pending_timeout"})
# Sub-job states before placement: a sub-job still in one never got GPUs.
_UNPLACED = frozenset({"pending", "queued", "creating", "placing"})


def never_started(job: Any) -> bool:
    """An ended job that never got GPUs, so it has no logs to show (see :data:`PRE_START_REASONS`)."""
    if not isinstance(job, dict) or is_live(job):
        return False
    reason = str(job.get("reason") or "").strip().lower()
    if status_word(job.get("status")) == "failed" and (reason.startswith("placement_") or reason in PRE_START_REASONS):
        return True
    words = [status_word(sj.get("status")) for sj in sub_jobs(job)]
    return bool(words) and all(word in _UNPLACED for word in words)


def never_started_text(job: dict) -> str:
    """Why a job that :func:`never_started` has no logs, as a clause, for example
    ``the job never got GPUs (reason: placement_timeout), so no sub-job started or wrote logs``."""
    ended = {"cancelled": "was cancelled and ", "canceled": "was cancelled and ",
             "terminated": "was terminated and "}.get(status_word(job.get("status")), "")
    reason = collapse(job.get("reason") or "")
    why = f" (reason: {reason})" if reason else ""
    return f"the job {ended}never got GPUs{why}, so no sub-job started or wrote logs"


def status_unavailable_text(job: dict) -> str:
    """What to say for an UNKNOWN job, using the sentence the service sends in
    ``reason`` (``Job status unavailable: Control Plane is unavailable``)."""
    reason = reason_text(job)
    if reason.lower().startswith("job status unavailable"):
        return reason
    return f"status unavailable: {reason or 'the service could not read the live state'}"


# Fields the service recomputes on every get: a fresh answer replaces them as a
# whole, so one it stopped sending (a cleared reason) disappears. created_at is
# not one: it never changes, and an UNKNOWN answer leaves it out.
SERVICE_FIELDS = ("status", "reason", "reason_details", "updated_at", "state_changed_at", "sub_jobs")


def refresh_job(current: Optional[dict], fresh: dict) -> dict:
    """``current`` (say, a list row) updated with a fresh ``get_job`` answer (see :data:`SERVICE_FIELDS`)."""
    kept = {k: val for k, val in (current or {}).items() if k not in SERVICE_FIELDS}
    return {**kept, **keep_deployment(current, fresh)}


def keep_deployment(old: Any, fresh: dict) -> dict:
    """``fresh``, kept a deployment. When ``old`` was one and ``fresh`` has no sub-jobs (an UNKNOWN
    answer, say), ``fresh`` gets ``old``'s sub-jobs without their statuses, which are no longer known."""
    if fresh.get("sub_jobs") or not is_deployment(old):
        return fresh
    return {**fresh, "sub_jobs": [{k: val for k, val in sj.items() if k != "status"} for sj in old["sub_jobs"]]}


# field -> help text; the field names users type in the search box.
SEARCH_FIELDS = {
    "id": "job id",
    "status": "job status",
    "owner": "submitting user or owning role",
    "type": "training / sampling / deployment",
    "class": "deployment class (EVALUATION)",
    "model": "model name",
    "method": "full / lora",
    "gpu": "GPU type (H200, B200, B300)",
    "gpus": "requested GPU count",
    "optimizer": "optimizer name",
    "lr": "learning rate",
    "batch": "train batch size",
    "seq": "max sequence length",
    "lora_r": "LoRA rank",
    "lora_alpha": "LoRA alpha",
    "dtype": "dtype",
    "db": "database name, or DB.SCHEMA (db=NAME reads that database from the server)",
    "schema": "schema name, or DB.SCHEMA as the Schema column shows it",
}
# Fields that name where a job is. A row without a location (the configured
# schema's own list) is in the configured database and schema, which the
# caller passes as ``home``.
_LOCATION_FIELDS = ("db", "schema")
_FIELD_ALIASES = {
    "hardware": "gpu",
    "hw": "gpu",
    "gpu_type": "gpu",
    "n_gpus": "gpus",
    "learning_rate": "lr",
    "opt": "optimizer",
    "bs": "batch",
    "batch_size": "batch",
    "train_batch_size": "batch",
    "seq_len": "seq",
    "max_seq_len": "seq",
    "rank": "lora_r",
    "r": "lora_r",
    "alpha": "lora_alpha",
    "user": "owner",
    "job_type": "type",
    "deployment_class": "class",
    "model_name": "model",
    "database": "db",
    "database_name": "db",
    "schema_name": "schema",
}
_TOKEN = re.compile(r"^([A-Za-z_][\w.]*)(>=|<=|!=|:|=|>|<)(.+)$")
# One search token: runs of anything but blanks and quotes, and quoted pieces.
_RAW_TOKEN = re.compile(r"""(?:[^\s"']+|"[^"]*"|'[^']*')+""")
_WHOLLY_QUOTED = re.compile(r"""^(?:"[^"]*"|'[^']*')$""")


def fold(text: Any) -> str:
    """NFKC-normalized, case-folded search text."""
    return _fold(text if isinstance(text, str) else str(text))


@functools.lru_cache(maxsize=8192)
def _fold(s: str) -> str:
    return unicodedata.normalize("NFKC", s).casefold()


@dataclass
class Query:
    terms: list = field(default_factory=list)
    filters: list = field(default_factory=list)  # (field, op, value)
    errors: list = field(default_factory=list)
    # Unknown filters remain errors until resolved against loaded job members.
    unknown: list = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.terms and not self.filters


def _tokens(text: str) -> list:
    """``(token, quoted)`` pairs: shell-like words, ``quoted`` when the whole token was in quotes."""
    text = text or ""
    raw = _RAW_TOKEN.findall(text)
    if "".join("".join(raw).split()) != "".join(text.split()):  # an unbalanced quote: plain words, as typed
        return [(tok, False) for tok in text.split()]
    out = []
    for tok in raw:
        try:
            words = shlex.split(tok)
        except ValueError:
            words = [tok]
        out.append((" ".join(words), bool(_WHOLLY_QUOTED.match(tok))))
    return out


def parse_query(text: str) -> Query:
    """Parse free words and field filters; quoted tokens remain plain words."""
    q = Query()
    for tok, quoted in _tokens(text):
        m = None if quoted else _TOKEN.match(tok)
        if not m:
            if tok:
                q.terms.append(fold(tok))
            continue
        key, op, value = m.group(1).lower(), m.group(2), m.group(3)
        key = _FIELD_ALIASES.get(key, key)
        if key not in SEARCH_FIELDS and "." not in key:
            q.errors.append(f"unknown field '{m.group(1)}'")
            q.unknown.append((key, op, value, m.group(1)))
            continue
        q.filters.append((key, op, value))
    return q


def _member_keys(job: dict) -> set:
    """The lowercase member names a job has at the top level of its record, its sub-jobs and their configs."""
    keys = {str(k).lower() for k in job}
    for sj in sub_jobs(job):
        keys.update(str(k).lower() for k in sj)
        keys.update(str(k).lower() for k in subjob_config(sj))
    return keys


def _config_paths(job: dict, key: str, prefix: str = "", obj: Any = None, depth: int = 0) -> set:
    """Dotted paths, from a sub-job, of every config member named ``key`` below the top level."""
    found: set = set()
    if obj is None:
        for sj in sub_jobs(job):
            for top, cfg in sj.items():
                if isinstance(cfg, dict):
                    found |= _config_paths(job, key, str(top), cfg, 1)
        return found
    if depth > 6 or not isinstance(obj, dict):
        return found
    for k, v in obj.items():
        path = f"{prefix}.{k}"
        if str(k).lower() == key:
            found.add(path)
        if isinstance(v, dict):
            found |= _config_paths(job, key, path, v, depth + 1)
    return found


def resolve_query(query: Query, jobs: Iterable[dict]) -> Query:
    """Resolve unknown filters against loaded members, suggesting dotted paths."""
    if not query.unknown:
        return query
    jobs = [j for j in jobs if isinstance(j, dict)]
    present: set = set()
    for job in jobs:
        present |= _member_keys(job)
    filters, errors, unknown = list(query.filters), [], []
    for key, op, value, typed in query.unknown:
        if key in present:
            filters.append((key, op, value))
            continue
        paths = sorted({p for job in jobs[:2000] for p in _config_paths(job, key)})[:3]
        hint = f" (did you mean {' or '.join(paths)}?)" if paths else ""
        errors.append(f"unknown field '{typed}'{hint}")
        unknown.append((key, op, value, typed))
    return Query(terms=list(query.terms), filters=filters, errors=errors, unknown=unknown)


def _dig(obj: Any, path: list) -> list:
    if not path:
        return [obj]
    if isinstance(obj, list):
        return [v for item in obj for v in _dig(item, path)]
    if isinstance(obj, dict):
        key = path[0]
        for k, v in obj.items():
            if str(k).lower() == key:
                return _dig(v, path[1:])
    return []


def _scalars(values: Iterable) -> list:
    """Recursively flatten list scalars and drop dictionaries."""
    out: list = []
    for value in values:
        if isinstance(value, list):
            out.extend(_scalars(value))
        elif not isinstance(value, dict):
            out.append(value)
    return out


def _location_name(job: dict, key: str, home: Optional[tuple]) -> Optional[str]:
    """The stored database (``db``) or schema name of ``job``, else of ``home``; None when neither says."""
    names = location_names(job) or home
    return None if names is None else names[0 if key == "db" else 1]


def field_values(job: dict, key: str, *, home: Optional[tuple] = None) -> list:
    """All values a job carries for one search field."""
    if "." in key or key not in SEARCH_FIELDS:
        path = key.split(".")
        found = _dig(job, path)
        for sj in sub_jobs(job):
            found.extend(_dig(sj, path))
            found.extend(_dig(subjob_config(sj), path))
        return _scalars(found)
    subs = sub_jobs(job)
    cfgs = [subjob_config(s) for s in subs]
    if key == "id":
        return [job_id_of(job)] + [str(s.get("sub_job_id") or "") for s in subs]
    if key == "status":
        return list(dict.fromkeys([status_word(job.get("status")), status_word_for(job, job.get("status"))]))
    if key == "owner":
        return [str(v) for v in (job.get("submitted_by"), job.get("owner_role")) if v]
    if key == "type":
        # The short kind the list shows (log_prob) and the type as the API spells it (log_probability).
        kinds = [*(["deployment"] if is_deployment(job) else []), *job_types(job), *_raw_job_types(job)]
        return list(dict.fromkeys(kinds))
    if key == "class":
        return [deployment_class(job)] if is_deployment(job) else []
    if key == "model":
        # The whole name, and the part after the organization that the Model column shows.
        return list(dict.fromkeys(name for model in job_models(job) for name in (model, short_model(model))))
    if key == "method":
        return [training_method(s) for s in subs if str(s.get("job_type") or "").lower() == "training"]
    if key == "gpu":
        return [job["hardware"]] if job.get("hardware") else []
    if key == "gpus":
        return [total_gpus(job)]
    if key == "dtype":
        return [s["dtype"] for s in subs if s.get("dtype")]
    if key in _LOCATION_FIELDS:
        stored = _location_name(job, key, home)
        if stored is None:
            return []
        return list(dict.fromkeys((stored, sql_identifier(stored))))  # as stored, and as SQL quotes it
    if key in ("optimizer", "lr"):
        opts = [c.get("optimizer") for c in cfgs if isinstance(c.get("optimizer"), dict)]
        return [o.get("name" if key == "optimizer" else "lr") for o in opts if o.get("name" if key == "optimizer" else "lr") is not None]
    if key == "batch":
        return [c["train_batch_size"] for c in cfgs if c.get("train_batch_size") is not None]
    if key == "seq":
        return [c["max_seq_len"] for c in cfgs if c.get("max_seq_len") is not None]
    if key in ("lora_r", "lora_alpha"):
        peft_key = "r" if key == "lora_r" else "lora_alpha"
        return [c["peft_config"][peft_key] for c in cfgs if isinstance(c.get("peft_config"), dict) and c["peft_config"].get(peft_key) is not None]
    return []


def _compare(actual: Any, op: str, wanted: str) -> bool:
    a, w = _as_float(actual), _as_float(wanted)
    if a is not None and w is not None:
        close = math.isclose(a, w, rel_tol=1e-9, abs_tol=1e-12)
        return {
            ":": close, "=": close, "!=": not close,
            ">": a > w and not close, ">=": a >= w or close,
            "<": a < w and not close, "<=": a <= w or close,
        }[op]
    sa, sw = fold(actual), fold(wanted)
    if op == ":":
        return sw in sa
    if op == "=":
        return sa == sw
    if op == "!=":
        return sa != sw
    return False  # ordering needs numbers


def search_haystack(job: dict) -> str:
    """Folded id, status, owner, GPU, job types, and models; a deployment's status word and class too."""
    bits = [job_id_of(job), status_word(job.get("status")), owner(job),
            job.get("hardware") or "", job_type_label(job)]
    if is_deployment(job):
        bits += [status_word_for(job, job.get("status")), deployment_class(job)]
    bits += job_types(job) + _raw_job_types(job) + job_models(job)
    bits += [s.get("sub_job_id") or "" for s in sub_jobs(job)]  # ids copied from logs or --sub-job-id
    return fold(" ".join(str(b) for b in bits))


def location_words(job: dict, *, home: Optional[tuple] = None) -> list:
    """A location folded both as displayed and bare.

    A row that names no location (the configured schema's own list) is in ``home``, the same
    fallback the ``db`` and ``schema`` filters use, so a word matches that schema's own name
    there too.
    """
    names = location_names(job) or home
    if names is None:
        return []
    shown = printable(f"{sql_identifier(names[0])}.{sql_identifier(names[1])}")
    return list(dict.fromkeys((fold(shown), fold(printable(f"{names[0]}.{names[1]}")))))


def split_location(value: str) -> Optional[tuple]:
    """``(database, schema)`` of a whole location as the Schema column shows it
    (``TEAM_DB.EXPERIMENTS``, ``"ml"."dev"``); None when ``value`` names one part only.

    Only a dot outside quotes divides the two, so a single quoted name holding one
    (``"my.db"``) stays one name.
    """
    text = value.strip()
    parts, held, quoted, at = [], [], False, 0
    while at < len(text):
        char = text[at]
        if char == '"':
            if quoted and text[at:at + 2] == '""':  # an escaped quote inside a quoted name
                held.append('""')
                at += 2
                continue
            quoted = not quoted
        elif char == "." and not quoted:
            parts.append("".join(held))
            held = []
            at += 1
            continue
        held.append(char)
        at += 1
    parts.append("".join(held))
    if quoted or len(parts) != 2 or not all(parts):
        return None  # unbalanced quotes, one name, or more dots than a location has
    return parts[0], parts[1]


def _starts_a_name(term: str, words: list) -> bool:
    """Whether a term starts a database or schema component."""
    for text in words:
        at = text.find(term)
        while at != -1:
            if at == 0 or text[at - 1] in '."':
                return True
            at = text.find(term, at + 1)
    return False


def _same_name(stored: str, wanted: str) -> bool:
    """Whether typed SQL identifier rules resolve ``wanted`` to ``stored``."""
    return wanted == stored or identifier_name(wanted) == stored


def _location_matches(job: dict, side: str, op: str, name: str, home: Optional[tuple]) -> bool:
    """Whether ``job``'s database (``side`` ``db``) or schema satisfies one side of a location filter.

    ``=`` and ``!=`` read ``name`` by SQL identifier rules (see :func:`_same_name`); every other
    operator keeps its usual meaning over the name as stored and as SQL quotes it, so ``:``
    matches part of it.
    """
    stored = _location_name(job, side, home)
    if stored is None:
        return False
    if op in ("=", "!="):
        return _same_name(stored, name)
    return any(_compare(held, op, name) for held in (stored, sql_identifier(stored)))


def job_matches_query(job: dict, query: Query, *, home: Optional[tuple] = None,
                      hay: Optional[str] = None) -> bool:
    """Whether ``job`` matches every term and filter, using ``home`` if unlocated."""
    if query.errors:
        return False  # a typo'd field should surface, not silently match everything
    if query.empty:
        return True
    if hay is None:
        hay = search_haystack(job)
    places = location_words(job, home=home)
    if any(t not in hay and not _starts_a_name(t, places) for t in query.terms):
        return False
    for key, op, value in query.filters:
        if key in _LOCATION_FIELDS and (op in ("=", "!=") or split_location(value)):
            # A dotted value is a whole location, as the Schema column shows it, so what that
            # column holds can be searched for exactly as it reads, with any operator:
            # ``schema=TEAM_DB.EXPERIMENTS`` asks for both names whole and
            # ``schema:team_db.exper`` for part of each, while ``schema=EXPERIMENTS`` and
            # ``schema:exper`` go on asking about the schema alone.
            pair = split_location(value)
            wanted = (("db", pair[0]), ("schema", pair[1])) if pair else ((key, value),)
            matched = all(_location_matches(job, side, op, name, home) for side, name in wanted)
            if matched == (op == "!="):
                return False
            continue
        values = field_values(job, key, home=home)
        if op == "!=":
            if any(not _compare(v, op, value) for v in values):
                return False
            continue
        if not any(_compare(v, op, value) for v in values):
            return False
    return True


def pushdown_location(query: Query) -> Optional[tuple]:
    """``(database, schema)`` the account-wide list can ask the server for; None to read it whole.

    Only a whole ``db=NAME``, and with it a whole ``schema=NAME``, narrows at the server: the
    account-level route takes those two filters and no others, and it refuses a schema without
    a database. ``schema`` is None when the database alone narrows. A dotted value names a whole
    location (``schema=TEAM_DB.EXPERIMENTS``, as the Schema column shows it) and pins both.

    Everything else stays here, matched by :func:`job_matches_query` over what the walk read: a
    partial match (``schema:exp``), a negation (``schema!=X``), an ordering op, a bare word, and
    a ``schema=`` with no ``db=`` of its own (the configured database is not implied, and
    assuming it would hide the same schema name in another database).

    A name narrows only in the form the service resolves it to (see :func:`narrowable_name`), so
    narrowing never drops a row the search would have kept. The client-side filter runs over the
    answer either way.
    """
    if query.errors:
        return None  # a typo'd field matches nothing; narrowing on it would ask for the wrong rows
    pinned: dict = {"db": set(), "schema": set()}
    for key, op, value in query.filters:
        if key not in _LOCATION_FIELDS or op != "=":
            continue
        pair = split_location(value)  # a whole location pins both sides at once
        if pair:
            pinned["db"].add(pair[0].strip())
            pinned["schema"].add(pair[1].strip())
        else:
            pinned[key].add(value.strip())

    def whole(side: str) -> Optional[str]:
        """The one name ``side`` is pinned to by an ``=`` and can be narrowed on; else None."""
        wanted = pinned[side]
        if len(wanted) != 1:
            return None  # nothing to narrow on, or two names at once (which matches nothing)
        name = next(iter(wanted))
        return name if name and narrowable_name(name) else None

    database = whole("db")
    if database is None:
        return None  # a schema needs its database with it
    return database, whole("schema")


def narrowable_name(name: str) -> bool:
    """Whether the service, asked for ``name``, returns every row ``name=`` keeps here.

    ``=`` keeps a row whose name is ``name`` as stored *or* the name SQL resolves ``name`` to
    (see :func:`_same_name`), while a walk narrowed to ``name`` only reads the second. The two
    agree when ``name`` is already in its resolved form (``TEAM_DB``), and when it is quoted
    (``"ml"``, which no stored name is spelled as, so only the resolved ``ml`` can match).

    An unquoted ``ml`` or ``team_db`` resolves to ``ML`` / ``TEAM_DB`` but would also match a
    database stored under that exact lowercase spelling, which a narrowed walk would never read.
    Such a name is matched here instead, as it was before narrowing existed -- never worse than
    reading every schema, and the form the Schema column shows is always narrowable.
    """
    return name == identifier_name(name) or _QUOTED_IDENTIFIER.fullmatch(name) is not None


def no_match_message(text: str, query: Query, searched: int, *, cut: str = "", older: Iterable[str] = (),
                     location: str = "", widen: bool = False, narrow: bool = False, everything: int = 0,
                     elsewhere: str = "", noun: str = "job") -> str:
    """Explain an empty search, including truncated, old, or wider matches; ``elsewhere`` says
    where the other view's matches are, and ``noun`` what the view on screen lists."""
    older = list(older)
    if cut:
        lines = [f"No {noun}s match “{text}” among the {searched:,} {noun}(s) loaded: {cut}.",
                 "  Search covers only the loaded jobs."]
    else:
        where = f" in {location}" if location else ""
        lines = [f"No {noun}s match “{text}” among {searched:,} {noun}(s){where} in the last 6 months."]
    if everything:
        lines.append(f"  f searches all {everything:,} jobs of the schema.")
    if older:
        many = len(older) > 1
        lines.append(f"  {len(older):,} older job{'s' if many else ''} match{'' if many else 'es'} (the list hides "
                     f"jobs older than 6 months): cortex-training tui {older[0]} opens {'the newest' if many else 'it'}.")
    if elsewhere:
        lines.append(f"  {elsewhere}")
    if widen:
        lines.append("  a searches every database and schema.")
    if narrow:
        # The list is cut short and nothing narrowed it, so the jobs looked for may simply not be
        # among the ones read. A whole name is asked of the service instead (see pushdown_location).
        lines.append("  db=NAME, or db=NAME schema=NAME, reads that whole database from the service "
                     "rather than only the jobs loaded here.")
    if query.errors:
        lines.append("  " + "; ".join(query.errors) + ".")
        lines.append("  Fields: " + ", ".join(SEARCH_FIELDS) + "; any other key reads that member of a job, its "
                     "sub-jobs or their configs (seed=42), and a dotted one a nested member "
                     "(training_config.optimizer.lr<1e-4).")
        if any(op == ":" for _key, op, _value, _typed in query.unknown):
            lines.append("  To search for text with a colon in it, quote it: \"training:0\".")
    if query.terms:
        lines.append("  Words match id, sub-job id, status, owner, GPU type, job type, model, and "
                     "the start of a location's database or schema name (case-insensitive): "
                     "db:TEXT matches anywhere in the name, db=NAME the whole name.")
    if query.filters:
        lines.append("  Filters must all match; try fewer, or ':' for a partial match (model:qwen).")
    lines.append("  Esc clears the search.")
    return "\n".join(lines)


SORT_KEYS = ("created", "updated", "status")
_STATUS_ORDER = ["running", "initializing", "placing", "pending", "queued", "creating", "unknown", "failed", "error", "done",
                 "cancelled", "terminated"]
# Other spellings of a status, sorted with it.
_STATUS_SPELLINGS = {"succeeded": "done", "completed": "done", "canceled": "cancelled"}
_STATUS_RANK = {**{s: i for i, s in enumerate(_STATUS_ORDER)},
                **{s: _STATUS_ORDER.index(same) for s, same in _STATUS_SPELLINGS.items()}}


_NO_TIME = float("-inf")  # what _epoch makes of a missing or placeholder time


def _epoch(raw: Any) -> float:
    dt = parse_ts(raw)
    return dt.timestamp() if dt else _NO_TIME


def sort_job_list(jobs: Iterable[dict], key: str = "created", descending: bool = True) -> list:
    """``jobs`` by ``key``: "created" or "updated" time (newest first, or oldest), or "status".

    Under a time sort, the jobs without that time (shown as ``—``) come after
    the others in both directions, newest created first. Under the status sort,
    jobs of one status are newest created first in both directions.
    """
    jobs = [j for j in jobs if isinstance(j, dict)]
    if key == "status":
        rank = _STATUS_RANK
        by_created = sorted(jobs, key=lambda j: _epoch(j.get("created_at")), reverse=True)
        return sorted(
            by_created,
            key=lambda j: rank.get(status_word(j.get("status")), len(_STATUS_ORDER)),
            reverse=not descending,
        )
    field_name = "updated_at" if key == "updated" else "created_at"
    keyed = [(_epoch(j.get(field_name)), j) for j in jobs]
    timed = sorted((pair for pair in keyed if pair[0] != _NO_TIME), key=lambda pair: pair[0], reverse=descending)
    untimed = [j for at, j in keyed if at == _NO_TIME]
    if field_name != "created_at":
        untimed.sort(key=lambda j: _epoch(j.get("created_at")), reverse=True)
    return [j for _at, j in timed] + untimed


_PLAIN_IDENTIFIER = re.compile(r"[A-Z_][A-Z0-9_$]*")
_QUOTED_IDENTIFIER = re.compile(r'"(?:[^"]|"")*"')


def sql_identifier(name: str) -> str:
    """``name`` as a SQL identifier: bare when it is a plain uppercase name, else double-quoted.

    ``CortexTrainingClient.for_job``'s rule: ``TEAM_DB`` stays bare, ``ml`` becomes ``"ml"``.
    """
    return name if _PLAIN_IDENTIFIER.fullmatch(name) else '"' + name.replace('"', '""') + '"'


def identifier_name(text: str) -> str:
    """The name a SQL identifier resolves to: a ``"quoted"`` one exactly, any other uppercase.

    The inverse of :func:`sql_identifier`: ``"ml"`` is ``ml``, ``team_db`` is ``TEAM_DB``.
    """
    text = text.strip()
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        return text[1:-1].replace('""', '"')
    return text.upper()


def location_names(job: Any) -> Optional[tuple]:
    """``(database_name, schema_name)`` of a row from the account-wide list, else None.

    Only that list names a job's location (both names, as stored, unquoted); the
    configured schema's own list and ``get_job`` don't, and a job without one is
    in the configured database and schema.
    """
    if not isinstance(job, dict):
        return None
    db, sch = job.get("database_name"), job.get("schema_name")
    if isinstance(db, str) and db and isinstance(sch, str) and sch:
        return db, sch
    return None


def client_location(client: Any) -> Optional[tuple]:
    """``(database_name, schema_name)`` a client is bound to, resolved from its identifiers;
    None when it doesn't say (a test client's attributes may be anything)."""
    db, sch = getattr(client, "database", None), getattr(client, "schema", None)
    if isinstance(db, str) and db.strip() and isinstance(sch, str) and sch.strip():
        return identifier_name(db), identifier_name(sch)
    return None


def location_parts(job: Any) -> Optional[tuple]:
    """``(DB, SCHEMA)`` of a row that names its location, as :func:`location_label` draws
    them on either side of its dot; None without one."""
    names = location_names(job)
    if names is None:
        return None
    return printable(sql_identifier(names[0])), printable(sql_identifier(names[1]))


def location_label(job: Any) -> str:
    """``DB.SCHEMA`` of a row that names its location, each name quoted when SQL
    needs it (``TEAM_DB.EXPERIMENTS``, ``"ml"."dev"``); ``""`` without one."""
    parts = location_parts(job)
    return "" if parts is None else ".".join(parts)


def location_filter_text(location: Optional[tuple]) -> str:
    """``DB.SCHEMA`` (or ``DB`` alone) of a narrowing, each name as the server resolves it and
    SQL quotes it (``db=team_db`` reads ``TEAM_DB``; ``db="ml"`` reads ``"ml"``); "" without one."""
    if not location:
        return ""
    return printable(".".join(sql_identifier(identifier_name(name)) for name in location if name))


def cli_options(*, database: Optional[str] = None, schema: Optional[str] = None,
                connection: Optional[str] = None, config: Optional[str] = None, host: Optional[str] = None,
                base_url: Optional[str] = None, endpoint: Optional[str] = None) -> list:
    """The CLI's global options that reach one job: how it connects (``--connection
    NAME``, a profile; ``--config FILE``; ``--host HOST``, with a PAT; or
    ``--base-url URL``), then its database, schema and non-default endpoint,
    each value quoted for the shell.
    """
    found = (("--connection", connection), ("--config", config), ("--host", host), ("--base-url", base_url),
             ("--database", database), ("--schema", schema), ("--endpoint", endpoint))
    return [f"{flag} {shlex.quote(value)}" for flag, value in found if isinstance(value, str) and value]


def _reply_text(exc: BaseException) -> str:
    """The body of the reply an HTTP error carries; empty for anything else."""
    try:
        text = getattr(getattr(exc, "response", None), "text", "")
    except Exception:  # noqa: BLE001 - an unreadable body
        return ""
    return text if isinstance(text, str) else ""


# How Snowflake refuses SQL on the job's Experiment for this role (the LIST and
# GET that read saved logs and GPU metrics): "003001 (42501): SQL access control
# error: Insufficient privileges to operate on experiment 'NAME'", "Experiment
# 'NAME' does not exist or not authorized", or the same said of a
# snow://experiment/ path. The refusal itself must be about the Experiment, so a
# denial about some other object (a warehouse, a schema) keeps its own text even
# when the message names the Experiment elsewhere.
_EXPERIMENT_SQL_DENIED = re.compile(
    r"insufficient privileges to operate on experiment\b"
    r"|\bexperiment(?:\s+run)?\s+(?:'[^']*'|\"[^\"]*\"|[^\s'\"]+)\s+does not exist or not authorized"
    r"|snow://experiment/[^\s'\"]*['\"]?:?\s+does not exist or not authorized",
    re.IGNORECASE)
# A 403 refuses this role when its reply says so in Snowflake's words. An empty or
# unrelated 403 (a proxy or a network gate can send one) keeps the standard text.
_PRIVILEGE_REFUSAL = re.compile(r"privilege|not authorized|access control error", re.IGNORECASE)


def access_hint(exc: BaseException, subject: str) -> Optional[str]:
    """One line when ``exc`` means this role may not read the job's ``subject`` ("saved logs", "GPU
    metrics"), else None: the caller then shows the standard error text."""
    hint = experiment_access_hint(exc)
    if hint:
        return hint
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 403 and _PRIVILEGE_REFUSAL.search(_reply_text(exc)):
        return f"this role isn't allowed to read this job's {subject} ({describe_error(exc)})"
    return None


def _name_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def sign_in_options(sign_in: Any) -> dict:
    """How a copied CLI command connects as the TUI does: ``config``, ``host`` and
    ``base_url`` for :func:`cli_options`, from ``sign_in``
    (``CortexTrainingLogTUI.sign_in``).

    The TUI's config file, when it read one, as ``--config FILE``; and its PAT
    host (``--host``) or base URL (``--base-url``), unless that file gives the
    same one by itself (``host_in_config`` / ``base_url_in_config``): a flag or
    the environment that set or overrode it is not part of the file. Empty for
    a connection profile, or when it isn't known.
    """
    sign = sign_in if isinstance(sign_in, dict) else {}
    kind, config = _name_or_none(sign.get("kind")), _name_or_none(sign.get("config"))
    if kind == "pat":
        in_config = bool(config) and sign.get("host_in_config") is True
        return {"config": config, "host": None if in_config else _name_or_none(sign.get("host"))}
    if kind == "url":
        in_config = bool(config) and sign.get("base_url_in_config") is True
        return {"config": config, "base_url": None if in_config else _name_or_none(sign.get("base_url"))}
    return {}
