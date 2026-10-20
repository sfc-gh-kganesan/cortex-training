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

"""Command-line implementation for Cortex Training job management."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import logging
import os
import re
import sys
import threading
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterator
from typing import TextIO

_CONFIG_KEYS = {
    "base_url",
    "host",
    "pat",
    "database",
    "schema",
    "endpoint",
    "poll_interval",
    "poll_timeout",
    "no_verify_ssl",
    "verify_ssl",
}

_CONFIG_ALIASES = {
    "url": "base_url",
    "db": "database",
}

_LOGIN_STATE_ENV = "CORTEX_TRAINING_LOGIN_FILE"

# The statuses each `list --status` word matches, as on the account-level
# route: pending excludes placing, done takes the DONE older jobs show, and
# UNKNOWN rows match nothing.
_LIST_STATUS_FILTERS = {
    "pending": {"pending"},
    "placing": {"placing"},
    "initializing": {"initializing"},
    "running": {"running"},
    "failed": {"failed"},
    "cancelled": {"cancelled"},
    "terminated": {"terminated"},
    "active": {"pending", "placing", "initializing", "running"},
    "done": {"cancelled", "terminated", "done"},
}

# Status words the server can answer with that are not the ones it filters on:
# older rows, and states a backend ahead of the API layer reports.
_STATUS_ALIASES = {
    "queued": "pending",
    "creating": "initializing",
    "canceled": "cancelled",
    "completed": "done",
    "succeeded": "done",
    "error": "failed",
}

# `capacity --breakdown` choice -> get_capacity_breakdown() dimension.
_CAPACITY_BREAKDOWN_CHOICES = {
    "job-type": "job_type",
    "submitted-by": "submitted_by",
}


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _normalize_host(host: str) -> str:
    for prefix in ("https://", "http://"):
        if host.startswith(prefix):
            host = host[len(prefix) :]
    return host.rstrip("/")


def _has_url_scheme(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


# The service rejects a GPU type the account does not support with this code.
_UNSUPPORTED_HARDWARE_CODE = "517604"


def _load_cortex_training_client_class():
    from .client import CortexTrainingClient

    return CortexTrainingClient


def _load_forward_backward_payload_builder():
    from .client import build_forward_backward_payload

    return build_forward_backward_payload


def _hardware_choices() -> list[str]:
    from .client import Hardware

    return [member.value for member in Hardware]


_FRACTION_RE = re.compile(r"\.([0-9]+)")


def _created_epoch(raw: Any) -> float | None:
    if raw is None:
        return None
    created_at = str(raw).strip()
    if not created_at:
        return None
    if created_at.endswith("Z"):
        created_at = created_at[:-1] + "+00:00"
    # Before Python 3.11, fromisoformat takes only 3 or 6 fractional digits.
    created_at = _FRACTION_RE.sub(
        lambda match: "." + match.group(1)[:6].ljust(6, "0"), created_at, count=1
    )
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.timestamp()


def _jobs_latest_last(jobs: list[Any]) -> list[Any]:
    """Render jobs oldest-first so the newest entries land at the terminal."""
    dated_jobs = [
        (
            _created_epoch(job.get("created_at")) if isinstance(job, dict) else None,
            index,
            job,
        )
        for index, job in enumerate(jobs)
    ]
    if any(created is not None for created, _, _ in dated_jobs):
        return [
            job
            for _, _, job in sorted(
                dated_jobs,
                key=lambda item: (
                    item[0] is None,
                    item[0] if item[0] is not None else float("inf"),
                    item[1],
                ),
            )
        ]
    return list(reversed(jobs))


def _newest_jobs(jobs: list[Any], limit: int) -> list[Any]:
    """The ``limit`` newest jobs by ``created_at``, in the order given.

    A missing timestamp ranks oldest, so ``--limit`` drops those rows first;
    ties break toward the earlier row, the server answering newest first.
    """
    if limit >= len(jobs):
        return list(jobs)
    last = len(jobs) - 1

    def rank(item: tuple[int, Any]) -> tuple[float, int]:
        index, job = item
        created = _created_epoch(job.get("created_at")) if isinstance(job, dict) else None
        return (created if created is not None else float("-inf"), last - index)

    newest = sorted(enumerate(jobs), key=rank)[-limit:]
    return [job for _, job in sorted(newest)]


def _status_list(raw: str) -> str:
    """argparse type for --status: the comma list, lowercased, every word known."""
    words = [word.strip().lower() for word in raw.split(",")]
    for word in words:
        if word not in _LIST_STATUS_FILTERS:
            raise argparse.ArgumentTypeError(
                f"unknown status {word!r}; use a comma-separated list of "
                + ", ".join(_LIST_STATUS_FILTERS)
            )
    return ",".join(words)


def _list_statuses(words: str) -> set[str]:
    """The job statuses a --status list from _status_list keeps."""
    statuses: set[str] = set()
    for word in words.split(","):
        statuses |= _LIST_STATUS_FILTERS[word]
    return statuses


_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _utc_date(value: str) -> date:
    """argparse type for --since/--until: a YYYY-MM-DD date, read as a UTC day."""
    # Matched first because date.fromisoformat also takes forms such as 20261001.
    if _DATE_RE.fullmatch(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"expected a YYYY-MM-DD date, got {value!r}")


def _utc_midnight(day: date) -> float:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()


def _same_name(wanted: str, actual: Any) -> bool:
    # Rows carry names as stored. SQL identifier rules, as the server applies to
    # submitted_by: an unquoted name means its uppercase form, a "quoted" one
    # exactly what is between the quotes.
    if len(wanted) > 1 and wanted[0] == wanted[-1] == '"':
        wanted = wanted[1:-1].replace('""', '"')
    else:
        wanted = wanted.upper()
    return isinstance(actual, str) and actual == wanted


def _list_filter(args: argparse.Namespace) -> Callable[[dict], bool]:
    """Build the list filters for the schema's jobs; --all has the server apply them."""
    # Rows can carry an enum name (JOB_STATE_RUNNING) or an alias, so compare
    # on the normalized word rather than the raw field.
    from .client import CortexTrainingClient

    def status_of(job: dict) -> str:
        word = CortexTrainingClient._normalize_job_status(job.get("status"))
        return _STATUS_ALIASES.get(word, word)

    statuses = None if args.status is None else _list_statuses(args.status)
    since = None if args.since is None else _utc_midnight(args.since)
    # --until names the last day kept, so the bound is the next midnight.
    until = None if args.until is None else _utc_midnight(args.until + timedelta(days=1))

    def keep(job: dict) -> bool:
        if statuses is not None and status_of(job) not in statuses:
            return False
        if args.user is not None and not _same_name(args.user, job.get("submitted_by")):
            return False
        if since is None and until is None:
            return True
        created = _created_epoch(job.get("created_at"))
        return (
            created is not None
            and (since is None or created >= since)
            and (until is None or created < until)
        )

    return keep


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def build_parser(
    *,
    prog: str = "cortex-training",
    include_tui: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Manage Cortex Training jobs through the Cortex Training REST endpoint.",
    )
    parser.add_argument(
        "--config",
        default=_env("CORTEX_TRAINING_CONFIG"),
        help="Path to a reusable Cortex Training CLI config JSON file.",
    )
    parser.add_argument(
        "--connection",
        "-c",
        default=_env("CORTEX_TRAINING_CONNECTION"),
        help=(
            "Snowflake connection profile name. When no legacy config or direct "
            "credentials are configured, the Snowflake default profile is used."
        ),
    )
    parser.add_argument(
        "--base-url",
        help="Base URL for a local or otherwise compatible server. Skips PAT auth.",
    )
    parser.add_argument(
        "--host",
        help="Snowflake account host for PAT auth.",
    )
    parser.add_argument(
        "--pat",
        help="Programmatic access token.",
    )
    parser.add_argument(
        "--database",
        help="Database containing the cortex-training endpoint.",
    )
    parser.add_argument(
        "--schema",
        help="Schema containing the cortex-training endpoint. Defaults to PUBLIC.",
    )
    parser.add_argument(
        "--endpoint",
        help="REST endpoint name. Defaults to cortex-training.",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification for authenticated requests.",
    )
    parser.add_argument("--poll-interval", type=float)
    parser.add_argument("--poll-timeout", type=float)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact JSON instead of pretty JSON.",
    )
    parser.add_argument(
        "--job",
        "--job-id",
        dest="job",
        help="Job id for data-plane commands such as fwd-bwd.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser(
        "submit",
        help="Submit a Cortex Training job from a CreateJob JSON file.",
    )
    submit.add_argument(
        "json_file",
        help="Path to CreateJob JSON, or '-' for stdin.",
    )
    submit.add_argument("--job-id", help="Set or override the top-level job_id.")
    submit.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the CreateJob body without sending it.",
    )
    submit.add_argument(
        "--wait",
        action="store_true",
        help="Wait until the submitted job reaches running.",
    )

    get = subparsers.add_parser("get", help="Fetch one Cortex Training job.")
    get.add_argument("job_id")

    checkpoints = subparsers.add_parser(
        "checkpoints",
        help="List checkpoints for one Cortex Training job.",
    )
    checkpoints.add_argument("job_id")

    list_jobs = subparsers.add_parser("list", help="List Cortex Training jobs.")
    list_jobs.add_argument(
        "--status",
        type=_status_list,
        help=(
            "Comma-separated statuses to keep: "
            + ", ".join(_LIST_STATUS_FILTERS)
            + ". active is pending through running; done is cancelled, terminated, "
            "and the DONE that older jobs show."
        ),
    )
    list_jobs.add_argument(
        "--all",
        action="store_true",
        help=(
            "List jobs across every schema in the account that the current user "
            "can see, not just the configured one. Fails if account-level listing "
            "is not enabled for the account."
        ),
    )
    list_jobs.add_argument("--user", help="Keep jobs submitted by this user.")
    list_jobs.add_argument(
        "--since",
        type=_utc_date,
        metavar="YYYY-MM-DD",
        help="Keep jobs created on or after this date (a UTC day).",
    )
    list_jobs.add_argument(
        "--until",
        type=_utc_date,
        metavar="YYYY-MM-DD",
        help="Keep jobs created on or before this date (a UTC day).",
    )
    list_jobs.add_argument(
        "--limit",
        type=_positive_int,
        help="Print at most this many of the newest matching jobs.",
    )

    capacity = subparsers.add_parser(
        "capacity",
        help="Show reserved GPU capacity and current usage for the caller account.",
    )
    capacity.add_argument(
        "--hardware",
        choices=_hardware_choices(),
        help=(
            "Show only this GPU hardware. Omit to show capacity for every type "
            "your account supports; unsupported types are listed separately."
        ),
    )
    capacity.add_argument(
        "--breakdown",
        action="append",
        choices=tuple(_CAPACITY_BREAKDOWN_CHOICES),
        help=(
            "Also split in_use_gpus and pending_gpus by sub-job type "
            "(job-type) or by submitting user (submitted-by). Repeat to get "
            "both."
        ),
    )

    cancel = subparsers.add_parser("cancel", help="Cancel one Cortex Training job.")
    cancel.add_argument("job_id")

    wait = subparsers.add_parser("wait", help="Wait until one job reaches running.")
    wait.add_argument("job_id")

    fwd_bwd = subparsers.add_parser(
        "fwd-bwd",
        help="Run one forward-backward request from a readable payload JSON file.",
    )
    fwd_bwd.add_argument(
        "json_file",
        help="Path to fwd-bwd JSON, or '-' for stdin.",
    )

    step = subparsers.add_parser(
        "step",
        help="Run one optimizer step for a training job.",
    )
    step.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer step. Defaults to 1e-4.",
    )

    load = subparsers.add_parser(
        "load",
        help="Load a checkpoint into an already-created Cortex Training job.",
    )
    load.add_argument("checkpoint_id", help="Checkpoint id/tag to load.")
    load.add_argument(
        "--source-job-id",
        help="Load the checkpoint from another job's checkpoint store.",
    )
    load.add_argument(
        "--target-sub-job-id",
        dest="target_sub_job_id",
        help=(
            "Training sub-job to load the checkpoint into, e.g. JOB_ID:training:0. "
            "Use this when you need explicit routing control; a job has at most one "
            "training sub-job, so omitting it uses that sub-job. "
            "Use 'cortex-training get JOB_ID' to discover available sub-job IDs."
        ),
    )
    load.add_argument(
        "--no-poll",
        action="store_true",
        help="Print the request id without polling for completion.",
    )

    generate = subparsers.add_parser(
        "generate",
        help="Run one generate request from a readable payload JSON file.",
    )
    generate.add_argument(
        "json_file",
        help="Path to generate JSON, or '-' for stdin.",
    )

    weight_sync = subparsers.add_parser(
        "weight-sync",
        help="Sync weights from a training sub-job to one or more sampling sub-jobs.",
    )
    weight_sync.add_argument(
        "--source-sub-job-id",
        help="Training sub-job id. Defaults to JOB_ID:training:0.",
    )
    weight_sync.add_argument(
        "--target-sub-job-id",
        action="append",
        dest="target_sub_job_ids",
        help="Sampling sub-job id. Can be repeated. Defaults to JOB_ID:sampling:0.",
    )
    weight_sync.add_argument(
        "--operation-sub-job-id",
        help="Sub-job id used to route the operation envelope. Defaults to the source training sub-job id.",
    )
    weight_sync.add_argument(
        "--operation-sub-job-type",
        help="Sub-job type used to route the operation envelope.",
    )
    weight_sync.add_argument(
        "--weight-format",
        choices=("vllm", "hf", "lora"),
        help="Weight sync payload format. Use 'lora' for adapter-only sync.",
    )
    weight_sync.add_argument(
        "--no-poll",
        action="store_true",
        help="Print the request id without polling for completion.",
    )

    for command, job_parser in (
        ("fwd-bwd", fwd_bwd),
        ("step", step),
        ("load", load),
        ("generate", generate),
        ("weight-sync", weight_sync),
    ):
        job_parser.prog = f"{prog} --job JOB_ID {command}"
        job_parser.epilog = (
            "Required global option: --job JOB_ID (alias: --job-id JOB_ID). "
            "Place it before the subcommand."
        )

    download_log = subparsers.add_parser(
        "download-log",
        help="Download all log files for a Cortex Training job's experiment run.",
    )
    download_log.add_argument("job_id")
    download_log.add_argument(
        "--log-type",
        choices=("execution", "stdout"),
        default="execution",
        help=(
            "Log source to download: execution artifacts (default), or the "
            "reconstructed head-pod stdout/stderr console."
        ),
    )
    download_log.add_argument(
        "--output-dir",
        dest="output_dir",
        help=(
            "Directory to write log files into, grouped as "
            "<output_dir>/<sub_job_id>/<filename>. Created if missing. "
            "Defaults to the current working directory."
        ),
    )
    download_log.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an earlier download in the output directory and fetch "
            "only what is not already saved. For stdout that includes output an "
            "interrupted download already saved; for execution logs, files "
            "unchanged since the last download are skipped. Two downloads of "
            "the same directory at once are unsupported."
        ),
    )
    _add_chunk_download_arguments(download_log, "--log-type stdout")

    download_metrics = subparsers.add_parser(
        "download-metrics",
        help="Download and reconstruct GPU metrics for a Cortex Training job.",
    )
    download_metrics.add_argument("job_id")
    download_metrics.add_argument(
        "--output-dir",
        dest="output_dir",
        help=(
            "Directory to write <sub_job_id>/gpu.jsonl files. Created if "
            "missing. Defaults to the current working directory."
        ),
    )
    download_metrics.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an earlier metrics download in the output directory and "
            "fetch only metrics that are not already saved, including metrics "
            "an interrupted download already saved. Two downloads of the same "
            "directory at once are unsupported."
        ),
    )
    _add_chunk_download_arguments(download_metrics, None)

    login = subparsers.add_parser(
        "login",
        usage="%(prog)s [-h] (config | --config config)",
        help="Remember a Cortex Training config file for future commands.",
    )
    login_config = login.add_mutually_exclusive_group(required=True)
    login_config.add_argument(
        "login_config",
        nargs="?",
        metavar="config",
        help="Path to the Cortex Training CLI config JSON file to remember.",
    )
    login_config.add_argument(
        "--config",
        dest="login_config_option",
        metavar="config",
        help="Alternative to the positional config path.",
    )

    if include_tui:
        subparsers.add_parser(
            "tui",
            help="Open the read-only Cortex Training logs and resources TUI.",
        )

    return parser


def _login_state_path() -> Path:
    override = _env(_LOGIN_STATE_ENV)
    if override:
        return Path(override).expanduser()
    config_home = _env("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "cortex-training" / "login.json"


def _read_login_config_path() -> str | None:
    path = _login_state_path()
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid cortex-training login state {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"invalid cortex-training login state {path}: expected object")
    config_path = parsed.get("config_path")
    if not isinstance(config_path, str) or not config_path:
        raise ValueError(f"invalid cortex-training login state {path}: missing config_path")
    return config_path


def _write_login_config_path(config_path: str) -> str:
    path = Path(config_path).expanduser()
    _load_config(str(path))
    saved_path = str(path.resolve())
    state_path = _login_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config_path": saved_path}
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return saved_path


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    path_obj = Path(path).expanduser()
    try:
        parsed = json.loads(path_obj.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in config {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("config JSON must be an object")

    if "connection" in parsed:
        connection = parsed["connection"]
        if not isinstance(connection, dict):
            raise ValueError("config connection must be an object")
        parsed = connection

    config: dict[str, Any] = {}
    unknown = []
    for key, value in parsed.items():
        normalized = _CONFIG_ALIASES.get(key, key)
        if normalized not in _CONFIG_KEYS:
            unknown.append(key)
            continue
        config[normalized] = value
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown config key(s): {names}")
    return config


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _config_str(config: dict[str, Any], key: str) -> str | None:
    if key not in config or config[key] is None:
        return None
    if not isinstance(config[key], str):
        raise ValueError(f"config {key} must be a string")
    return config[key]


def _config_float(config: dict[str, Any], key: str) -> float | None:
    if key not in config or config[key] is None:
        return None
    if isinstance(config[key], bool) or not isinstance(config[key], (int, float)):
        raise ValueError(f"config {key} must be a number")
    return float(config[key])


def _config_bool(config: dict[str, Any], key: str) -> bool | None:
    if key not in config or config[key] is None:
        return None
    if not isinstance(config[key], bool):
        raise ValueError(f"config {key} must be a boolean")
    return config[key]


def _has_connection(
    base_url: str | None,
    host: str | None,
    pat: str | None,
    database: str | None,
) -> bool:
    return bool(database and (base_url or (host and pat)))


def _select_config(args: argparse.Namespace, *, load_login: bool) -> dict[str, Any]:
    args._legacy_config_selected = False
    if getattr(args, "connection", None):
        return {}

    if args.config:
        args._legacy_config_selected = True
        return _load_config(args.config)

    if not load_login:
        return {}

    base_url = _coalesce(args.base_url, _env("CORTEX_TRAINING_BASE_URL"))
    host = _coalesce(args.host, _env("CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST"))
    pat = _coalesce(args.pat, _env("CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT"))
    database = _coalesce(
        args.database,
        _env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE"),
    )
    if _has_connection(base_url, host, pat, database):
        return {}

    login_config = _read_login_config_path()
    if login_config is None:
        return {}
    if not Path(login_config).expanduser().is_file():
        return {}
    args.config = login_config
    args._legacy_config_selected = True
    return _load_config(login_config)


def _resolve_args(
    args: argparse.Namespace,
    *,
    load_login: bool = True,
) -> argparse.Namespace:
    config = _select_config(args, load_login=load_login)
    profile_requested = bool(getattr(args, "connection", None))
    args.base_url = _coalesce(
        args.base_url,
        _config_str(config, "base_url"),
        None if profile_requested else _env("CORTEX_TRAINING_BASE_URL"),
    )
    args.host = _coalesce(
        args.host,
        _config_str(config, "host"),
        None
        if profile_requested
        else _env("CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST"),
    )
    args.pat = _coalesce(
        args.pat,
        _config_str(config, "pat"),
        None
        if profile_requested
        else _env("CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT"),
    )
    args.database = _coalesce(
        args.database,
        _config_str(config, "database"),
        _env("CORTEX_TRAINING_DATABASE", "SNOWFLAKE_DATABASE"),
    )
    args.schema = _coalesce(
        args.schema,
        _config_str(config, "schema"),
        _env("CORTEX_TRAINING_SCHEMA", "SNOWFLAKE_SCHEMA"),
    )
    args.endpoint = _coalesce(
        args.endpoint,
        _config_str(config, "endpoint"),
        _env("CORTEX_TRAINING_ENDPOINT"),
        "cortex-training",
    )
    args.poll_interval = _coalesce(
        args.poll_interval,
        _config_float(config, "poll_interval"),
        0.5,
    )
    args.poll_timeout = _coalesce(
        args.poll_timeout,
        _config_float(config, "poll_timeout"),
        1800.0,
    )
    if not args.no_verify_ssl:
        no_verify_ssl = _config_bool(config, "no_verify_ssl")
        verify_ssl = _config_bool(config, "verify_ssl")
        if no_verify_ssl is not None:
            args.no_verify_ssl = no_verify_ssl
        elif verify_ssl is not None:
            args.no_verify_ssl = not verify_ssl
    direct_credentials_present = any((args.base_url, args.host, args.pat))
    if profile_requested and direct_credentials_present:
        raise ValueError(
            "--connection cannot be combined with --base-url, --host, or --pat"
        )
    direct_connection_complete = bool(args.base_url or (args.host and args.pat))
    args.use_connection_profile = profile_requested or (
        not args._legacy_config_selected and not direct_connection_complete
    )
    if not args.use_connection_profile and args.schema is None:
        args.schema = "PUBLIC"
    return args


def _normalize_connection_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.base_url is None or _has_url_scheme(args.base_url):
        return args
    if args.pat and args.host is None:
        args.host = args.base_url
        args.base_url = None
        return args
    raise ValueError(
        "base_url must start with http:// or https://. For Snowflake PAT auth, use host instead of base_url."
    )


def parse_args(
    argv: list[str] | None = None,
    *,
    prog: str = "cortex-training",
    include_tui: bool = False,
) -> argparse.Namespace:
    parser = build_parser(prog=prog, include_tui=include_tui)
    args = parser.parse_args(argv)
    if args.command == "list":
        if args.since is not None and args.until is not None and args.since > args.until:
            parser.error("--since must not be later than --until")
        # --until is sent as the day after, which does not exist for the last date.
        if args.until == date.max:
            parser.error(f"--until must be earlier than {date.max}")
        if args.user is not None and not args.user.strip():
            parser.error("--user must not be empty")
    if args.command == "login":
        args.login_config = args.login_config or args.login_config_option
        return args

    dry_run = args.command == "submit" and args.dry_run
    args = _resolve_args(args, load_login=not dry_run)
    if dry_run:
        return args
    args = _normalize_connection_args(args)
    if args.use_connection_profile:
        return args
    if not args.database:
        parser.error("provide --database or set CORTEX_TRAINING_DATABASE/SNOWFLAKE_DATABASE")
    if args.base_url is None and (args.host is None or args.pat is None):
        parser.error("provide --base-url for local/mock use, or both --host and --pat")
    return args


def build_client(args: argparse.Namespace, cortex_training_client_cls):
    kwargs = {
        "database": args.database,
        "schema": args.schema,
        "endpoint": args.endpoint,
        "poll_interval": args.poll_interval,
        "poll_timeout": args.poll_timeout,
    }
    # The TUI bounds every read (`cortex-training tui` sets these); the CLI
    # commands keep the client's defaults.
    for option in ("max_retries", "request_timeout"):
        value = getattr(args, option, None)
        if value is not None:
            kwargs[option] = value
    if args.base_url:
        return cortex_training_client_cls(base_url=args.base_url, **kwargs)
    if args.use_connection_profile:
        return cortex_training_client_cls.from_connection_name(
            connection_name=args.connection,
            verify_ssl=not args.no_verify_ssl,
            **kwargs,
        )
    return cortex_training_client_cls.from_pat(
        host=_normalize_host(args.host),
        pat=args.pat,
        verify_ssl=not args.no_verify_ssl,
        **kwargs,
    )


def _read_json_object(path: str, stdin: TextIO) -> dict[str, Any]:
    if path == "-":
        raw = stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("job JSON must be an object")
    return parsed


# Duplicated from client.py rather than imported: submit --dry-run must stay
# importable without pulling in the client (and therefore torch).
_LOG_PROBABILITY_JOB_TYPE_ALIASES = frozenset({"log_probability", "log_prob"})
_UNSUPPORTED_LOG_PROBABILITY_MESSAGE = (
    "log_probability sub-jobs are not currently supported"
)


def _normalized_job_type(job_type: Any) -> str:
    if not isinstance(job_type, str):
        return ""
    return job_type.strip().lower().removeprefix("job_type_")


def _is_log_probability_job_type(job_type: Any) -> bool:
    return _normalized_job_type(job_type) in _LOG_PROBABILITY_JOB_TYPE_ALIASES


def _reject_log_probability_sub_job(job_type: Any, *, location: str) -> None:
    if _is_log_probability_job_type(job_type):
        raise ValueError(f"{location}: {_UNSUPPORTED_LOG_PROBABILITY_MESSAGE}")


def _validate_create_job_body(body: dict[str, Any]) -> None:
    from .client import _validate_raw_sub_job_configs

    sub_job_configs = body.get("sub_job_configs")
    if not isinstance(sub_job_configs, list) or not sub_job_configs:
        raise ValueError("job JSON must contain a non-empty sub_job_configs list")
    # Mirrors client.create_job_from_body; submit --dry-run never builds a client.
    training_sub_jobs = 0
    for index, cfg in enumerate(sub_job_configs):
        if not isinstance(cfg, dict):
            continue
        job_type = cfg.get("job_type")
        _reject_log_probability_sub_job(
            job_type, location=f"sub_job_configs[{index}].job_type"
        )
        if str(job_type or "").strip().lower() == "training":
            training_sub_jobs += 1
            if training_sub_jobs > 1:
                raise ValueError("at most one training sub-job is supported per job")
    _validate_raw_sub_job_configs(sub_job_configs)


def _print_json(value: Any, stdout: TextIO, *, compact: bool, sort_keys: bool = True) -> None:
    if compact:
        json.dump(value, stdout, separators=(",", ":"), sort_keys=sort_keys)
    else:
        json.dump(value, stdout, indent=2, sort_keys=sort_keys)
    stdout.write("\n")


def _request_sent_pat(response: Any) -> bool:
    # Connection-profile requests carry a session token from a login that
    # already accepted the PAT, so a 401 there is not a PAT problem.
    headers = getattr(getattr(response, "request", None), "headers", None) or {}
    return headers.get("X-Snowflake-Authorization-Token-Type") == "PROGRAMMATIC_ACCESS_TOKEN"


def _format_error(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)

    parts = [str(exc)]
    request_id = getattr(response, "headers", {}).get("x-snowflake-request-id")
    if request_id and request_id not in parts[0]:
        parts.append(f"snowflake request id: {request_id}")

    body = (getattr(response, "text", "") or "").strip()
    if body:
        if len(body) > 4000:
            body = body[:4000] + "...<truncated>"
        parts.append(f"response body: {body}")
    if getattr(response, "status_code", None) == 401 and _request_sent_pat(response):
        from cortex_training.snowflake_auth import credentials_rejected_hint

        parts.append(
            credentials_rejected_hint("Snowflake rejected your credentials (HTTP 401).")
        )
    return "\n".join(parts)


def _cmd_submit(
    args: argparse.Namespace,
    client,
    stdout: TextIO,
    stdin: TextIO,
) -> int:
    body = _read_json_object(args.json_file, stdin)
    _validate_create_job_body(body)
    if args.job_id is not None:
        body = dict(body)
        body["job_id"] = args.job_id

    if args.dry_run:
        _print_json(body, stdout, compact=args.compact)
        return 0

    response = client.create_job_from_body(body)
    if args.wait:
        job_id = response.get("job_id") or body.get("job_id")
        if not job_id:
            raise ValueError("submit --wait requires the create response to include job_id")
        response = client.wait_for_job(str(job_id))
    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_list(args: argparse.Namespace, client, stdout: TextIO) -> int:
    if args.all:
        jobs = list(
            itertools.islice(
                client.iter_jobs(
                    status=args.status,
                    submitted_by=args.user,
                    created_after=args.since,
                    # The server's bound is exclusive; --until names the last day kept.
                    created_before=(
                        None if args.until is None else args.until + timedelta(days=1)
                    ),
                    # No bigger than what gets printed; the server takes 1 to 1000.
                    page_size=min(args.limit or 100, 1000),
                ),
                args.limit,
            )
        )
    else:
        keep = _list_filter(args)
        jobs = [job for job in client.list_jobs() if keep(job)]
    if args.limit is not None:
        jobs = _newest_jobs(jobs, args.limit)
    _print_json({"jobs": _jobs_latest_last(jobs)}, stdout, compact=args.compact)
    return 0


def _cmd_login(args: argparse.Namespace, stdout: TextIO) -> int:
    config_path = _write_login_config_path(args.login_config)
    _print_json(
        {"config_path": config_path, "logged_in": True},
        stdout,
        compact=args.compact,
    )
    return 0


def _cmd_capacity(args: argparse.Namespace, client, stdout: TextIO) -> int:
    dimensions = tuple(
        _CAPACITY_BREAKDOWN_CHOICES[choice] for choice in args.breakdown or ()
    )

    def lookup(hardware: str) -> dict:
        if dimensions:
            return client.get_capacity_breakdown(
                hardware=hardware, dimensions=dimensions
            )
        return client.get_capacity(hardware=hardware)

    if args.hardware is not None:
        capacity = lookup(args.hardware)
    else:
        capacity = _capacity_by_hardware(lookup)
    _print_json(_breakdown_last(capacity), stdout, compact=args.compact, sort_keys=False)
    return 0


def _breakdown_last(value: Any) -> Any:
    """``value`` with keys sorted, except that ``breakdown`` follows the totals."""
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: (key == "breakdown", key))
        return {key: _breakdown_last(value[key]) for key in keys}
    if isinstance(value, list):
        return [_breakdown_last(item) for item in value]
    return value


def _json_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _cmd_fwd_bwd(
    args: argparse.Namespace,
    client,
    stdout: TextIO,
    stdin: TextIO,
) -> int:
    if not args.job:
        raise ValueError("provide --job JOB_ID for fwd-bwd")

    spec = _read_json_object(args.json_file, stdin)
    poll = _json_bool(spec.get("poll", True), "fwd-bwd poll")
    build_payload = _load_forward_backward_payload_builder()
    payload = build_payload(spec)

    request_id = client.forward_backward(args.job, payload)
    response = {
        "job_id": args.job,
        "payload_size_bytes": len(payload),
        "request_id": request_id,
    }
    if poll:
        response["result"] = client.poll_request(args.job, request_id)

    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_step(args: argparse.Namespace, client, stdout: TextIO) -> int:
    if not args.job:
        raise ValueError("provide --job-id JOB_ID for step")

    request_id = client.step(args.job, learning_rate=args.lr)
    response = {
        "job_id": args.job,
        "learning_rate": args.lr,
        "request_id": request_id,
        "result": client.poll_request(args.job, request_id),
    }
    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_load(args: argparse.Namespace, client, stdout: TextIO) -> int:
    if not args.job:
        raise ValueError("provide --job-id JOB_ID for load")

    request_id = client.load(
        args.job,
        checkpoint_id=args.checkpoint_id,
        source_job_id=args.source_job_id,
        target_sub_job_id=args.target_sub_job_id,
    )
    response = {
        "checkpoint_id": args.checkpoint_id,
        "job_id": args.job,
        "request_id": request_id,
    }
    if args.source_job_id is not None:
        response["source_job_id"] = args.source_job_id
    if args.target_sub_job_id is not None:
        response["target_sub_job_id"] = args.target_sub_job_id
    if not args.no_poll:
        response["result"] = client.poll_request(args.job, request_id)

    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_generate(
    args: argparse.Namespace,
    client,
    stdout: TextIO,
    stdin: TextIO,
) -> int:
    if not args.job:
        raise ValueError("provide --job-id JOB_ID for generate")

    spec = _read_json_object(args.json_file, stdin)
    payload = spec.get("payload", spec)
    if not isinstance(payload, dict):
        raise ValueError("generate payload must be an object")

    poll = _json_bool(spec.get("poll", True), "generate poll")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("generate JSON must contain a non-empty prompts list")

    # A flat list of integers is one pre-tokenized prompt, not a batch of
    # single-token prompts, and the client encodes it that way. ``bool`` is an
    # ``int`` subclass, so exclude it rather than read [true, false] as tokens.
    single_tokenized = isinstance(prompts[0], int) and not isinstance(prompts[0], bool)
    prompt_count = 1 if single_tokenized else len(prompts)

    sampling_params = payload.get("sampling_params")
    if sampling_params is not None:
        if isinstance(sampling_params, list):
            if len(sampling_params) != prompt_count:
                raise ValueError("generate sampling_params list length must match prompts length")
            if any(item is not None and not isinstance(item, dict) for item in sampling_params):
                raise ValueError("generate sampling_params list items must be objects or null")
        elif not isinstance(sampling_params, dict):
            raise ValueError("generate sampling_params must be an object or list")

    strict = payload.get("strict")
    if strict is not None:
        strict = _json_bool(strict, "generate strict")

    request_id = client.generate(
        args.job,
        prompts=prompts,
        sampling_params=sampling_params,
        routing_key=payload.get("routing_key"),
        strict=strict,
    )
    response = {
        "job_id": args.job,
        "prompt_count": prompt_count,
        "request_id": request_id,
    }
    if poll:
        response["result"] = client.poll_request(args.job, request_id)

    _print_json(response, stdout, compact=args.compact)
    return 0


def _cmd_weight_sync(args: argparse.Namespace, client, stdout: TextIO) -> int:
    if not args.job:
        raise ValueError("provide --job-id JOB_ID for weight-sync")

    source_sub_job_id = args.source_sub_job_id or f"{args.job}:training:0"
    target_sub_job_ids = args.target_sub_job_ids or [f"{args.job}:sampling:0"]
    request_id = client.weight_sync(
        args.job,
        source_sub_job_id=source_sub_job_id,
        target_sub_job_ids=target_sub_job_ids,
        weight_format=args.weight_format,
        sub_job_id=args.operation_sub_job_id,
        sub_job_type=args.operation_sub_job_type,
    )
    response = {
        "job_id": args.job,
        "request_id": request_id,
        "source_sub_job_id": source_sub_job_id,
        "target_sub_job_ids": target_sub_job_ids,
    }
    if args.weight_format is not None:
        response["weight_format"] = args.weight_format
    if not args.no_poll:
        response["result"] = client.poll_request(args.job, request_id)

    _print_json(response, stdout, compact=args.compact)
    return 0


_CHUNK_DOWNLOAD_COMMANDS = ("download-log", "download-metrics")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


# More concurrent fetches than this only add load; the default is 16.
_MAX_PARALLEL = 128


def _parallel_count(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > _MAX_PARALLEL:
        raise argparse.ArgumentTypeError(f"must be at most {_MAX_PARALLEL}, got {parsed}")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {value!r}")
    if not parsed >= 0 or parsed == float("inf"):
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {value!r}")
    return parsed


def _add_chunk_download_arguments(parser: argparse.ArgumentParser, applies_to: str | None) -> None:
    suffix = f" Applies to {applies_to}." if applies_to else ""
    parser.add_argument(
        "--parallel",
        type=_parallel_count,
        default=None,
        metavar="N",
        help=f"Number of chunks to fetch at once, 1 to {_MAX_PARALLEL} (default: 16).{suffix}",
    )
    parser.add_argument(
        "--sub-job",
        dest="sub_job_ids",
        action="append",
        default=None,
        metavar="SUB_JOB_ID",
        help="Download only this sub-job. Repeat for more than one.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Stop on a chunk that cannot be decompressed instead of skipping it "
            f"and reporting it in skipped_chunks.{suffix}"
        ),
    )
    parser.add_argument(
        "--stale-after-minutes",
        type=_non_negative_float,
        default=10.0,
        metavar="MINUTES",
        help=(
            "Warn when a running job's newest saved chunk is older than this "
            "(default: 10). 0 turns off this and the other job-status notices "
            f"(a just-ended job, missing GPU metrics).{suffix}"
        ),
    )
    parser.add_argument(
        "--transfer-timeout",
        type=_non_negative_float,
        default=300.0,
        metavar="SECONDS",
        help=(
            "Give up on and retry a single file or chunk fetch, or listing, that "
            "takes longer than this (default: 300; 0 means no limit)."
        ),
    )


class _ArtifactProgressWriter:
    """Draw one updating terminal line. A redirected stderr gets the final line.

    Messages follow the download ``progress`` contract: a status without a
    newline replaces the current line, ``"\\n"`` ends it, and a message that
    already ends in a newline is printed as is. :meth:`warn` prints a full
    line without breaking the status line. Warnings come from download worker
    threads, so each update is written in one call under a lock.
    """

    def __init__(self, stderr: TextIO):
        self._lock = threading.Lock()
        self._stderr = stderr
        try:
            self._tty = stderr.isatty()
        except (OSError, ValueError):
            self._tty = False
        self._latest: str | None = None

    def __call__(self, message: str) -> None:
        with self._lock:
            if message == "\n":
                status, self._latest = self._latest, None
                if status is None:
                    return
                self._write("\n" if self._tty else status + "\n")
            elif message.endswith("\n"):
                self._write(message)
            else:
                self._latest = message
                if self._tty:
                    self._write("\r" + message)

    def warn(self, line: str) -> None:
        with self._lock:
            if self._tty and self._latest is not None:
                # Move off the status line, then redraw it below the warning.
                self._write("\n" + line + "\n\r" + self._latest)
            else:
                self._write(line + "\n")

    def _write(self, text: str) -> None:
        try:
            self._stderr.write(text)
            self._stderr.flush()
        except OSError:
            return


def _artifact_progress_writer(stderr: TextIO) -> _ArtifactProgressWriter:
    return _ArtifactProgressWriter(stderr)


def _chunk_download_options(
    args: argparse.Namespace, progress: Callable[[str], None]
) -> dict[str, Any]:
    return {
        "resume": args.resume,
        "max_workers": args.parallel,
        "progress": progress,
        "sub_job_ids": args.sub_job_ids,
        "strict": args.strict,
        "stale_after_seconds": (
            args.stale_after_minutes * 60 if args.stale_after_minutes else None
        ),
        "transfer_timeout_seconds": args.transfer_timeout or None,
    }


class _WarningLines(logging.Handler):
    def __init__(self, writer: _ArtifactProgressWriter):
        super().__init__(logging.WARNING)
        self._writer = writer
        self.setFormatter(logging.Formatter("warning: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._writer.warn(self.format(record))
        except Exception:
            self.handleError(record)


@contextlib.contextmanager
def _library_warnings_to(writer: _ArtifactProgressWriter) -> Iterator[None]:
    """Show the client's warnings (retries, skipped chunks) as ``warning:`` lines.

    Records still propagate, so a caller's own logging handlers see them too.
    """
    library = logging.getLogger("cortex_training")
    handler = _WarningLines(writer)
    library.addHandler(handler)
    try:
        yield
    finally:
        library.removeHandler(handler)


@contextlib.contextmanager
def _output_dir_lock(out_dir: Path) -> Iterator[None]:
    """Refuse to share a directory with a running TUI download of it.

    The TUI marks each cache directory it downloads into with a sibling lock
    file. Only an existing lock file is honored, so a plain CLI download never
    creates files outside its output directory.
    """
    from cortex_training._download_lock import DownloadLock

    lock = DownloadLock(out_dir)
    if lock.lock_path is None or not lock.lock_path.exists():
        yield
        return
    if not lock.try_acquire():
        raise RuntimeError(
            f"another download is using {out_dir}; close that job's view in the "
            "TUI or wait for its download to finish, then retry"
        )
    try:
        yield
    finally:
        lock.release()


def _cmd_download_log(
    args: argparse.Namespace, client, stdout: TextIO, stderr: TextIO
) -> int:
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else Path.cwd()
    if args.log_type == "stdout":
        progress = _artifact_progress_writer(stderr)
        with _output_dir_lock(out_dir), _library_warnings_to(progress):
            logs = client.download_stdout_logs(
                args.job_id,
                out_dir,
                **_chunk_download_options(args, progress),
            )
        _print_json(
            {"job_id": args.job_id, "logs": logs},
            stdout,
            compact=args.compact,
        )
        return 0

    progress = _artifact_progress_writer(stderr)
    with _library_warnings_to(progress):
        saved = client.download_execution_logs(
            args.job_id,
            out_dir,
            resume=args.resume,
            sub_job_ids=args.sub_job_ids,
            progress=progress,
            transfer_timeout_seconds=args.transfer_timeout or None,
        )
    _print_json({"job_id": args.job_id, "logs": saved}, stdout, compact=args.compact)
    return 0


def _cmd_download_metrics(
    args: argparse.Namespace, client, stdout: TextIO, stderr: TextIO
) -> int:
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else Path.cwd()
    progress = _artifact_progress_writer(stderr)
    with _output_dir_lock(out_dir), _library_warnings_to(progress):
        metrics = client.download_metrics(
            args.job_id,
            out_dir,
            **_chunk_download_options(args, progress),
        )
    _print_json(
        {"job_id": args.job_id, "metrics": metrics},
        stdout,
        compact=args.compact,
    )
    return 0


def _is_unsupported_hardware(exc: BaseException) -> bool:
    """Whether a capacity lookup failed only because the account lacks that GPU type."""
    response = getattr(exc, "response", None)
    if response is None:
        return False
    try:
        body = response.json()
    except Exception:
        return False
    if not isinstance(body, dict):
        return False
    code = str(body.get("code") or body.get("error_code") or "")
    message = str(body.get("message") or "")
    return code == _UNSUPPORTED_HARDWARE_CODE and "unsupported hardware" in message


def _capacity_by_hardware(lookup: Callable[[str], dict]) -> dict[str, Any]:
    """Capacity for every GPU type the account supports, plus the ones it does not."""
    supported: dict[str, Any] = {}
    unsupported: list[str] = []
    first_unsupported: BaseException | None = None
    for hardware in _hardware_choices():
        try:
            supported[hardware] = lookup(hardware)
        except Exception as exc:
            if not _is_unsupported_hardware(exc):
                raise
            unsupported.append(hardware)
            first_unsupported = first_unsupported or exc
    if not supported and first_unsupported is not None:
        raise first_unsupported
    return {"capacity_by_hardware": supported, "unsupported_hardware": unsupported}


def _run(
    args: argparse.Namespace,
    client_factory: Callable[[argparse.Namespace], Any],
    stdout: TextIO,
    stdin: TextIO,
    stderr: TextIO | None = None,
) -> int:
    stderr = stderr or sys.stderr
    if args.command == "login":
        return _cmd_login(args, stdout)

    if args.command == "submit" and args.dry_run:
        return _cmd_submit(args, None, stdout, stdin)

    if args.command == "download-log" and args.log_type != "stdout":
        # Execution logs are whole files, not chunks.
        for flag, value in (
            ("--parallel", args.parallel),
            ("--strict", args.strict),
            ("--stale-after-minutes", args.stale_after_minutes != 10.0),
        ):
            if value:
                raise ValueError(f"{flag} applies to --log-type stdout, not execution logs")

    client = client_factory(args)
    if args.command == "submit":
        return _cmd_submit(args, client, stdout, stdin)
    if args.command == "get":
        _print_json(client.get_job(args.job_id), stdout, compact=args.compact)
        return 0
    if args.command == "checkpoints":
        _print_json(
            {"checkpoints": client.list_checkpoints(args.job_id)},
            stdout,
            compact=args.compact,
        )
        return 0
    if args.command == "list":
        return _cmd_list(args, client, stdout)
    if args.command == "capacity":
        return _cmd_capacity(args, client, stdout)
    if args.command == "cancel":
        client.cancel_job(args.job_id)
        _print_json(
            {"cancelled": True, "job_id": args.job_id},
            stdout,
            compact=args.compact,
        )
        return 0
    if args.command == "wait":
        _print_json(client.wait_for_job(args.job_id), stdout, compact=args.compact)
        return 0
    if args.command == "fwd-bwd":
        return _cmd_fwd_bwd(args, client, stdout, stdin)
    if args.command == "step":
        return _cmd_step(args, client, stdout)
    if args.command == "load":
        return _cmd_load(args, client, stdout)
    if args.command == "generate":
        return _cmd_generate(args, client, stdout, stdin)
    if args.command == "weight-sync":
        return _cmd_weight_sync(args, client, stdout)
    if args.command == "download-log":
        return _cmd_download_log(args, client, stdout, stderr)
    if args.command == "download-metrics":
        return _cmd_download_metrics(args, client, stdout, stderr)
    raise ValueError(f"unknown command: {args.command}")


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "cortex-training",
    include_tui: bool = False,
    client_factory: Callable[[argparse.Namespace], Any] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdin: TextIO | None = None,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    stdin = stdin or sys.stdin

    args = None
    try:
        args = parse_args(argv, prog=prog, include_tui=include_tui)
        dry_run = args.command == "submit" and args.dry_run
        no_client = dry_run or args.command == "login"
        if client_factory is None and not no_client:
            cortex_training_client_cls = _load_cortex_training_client_class()

            def default_client_factory(parsed_args: argparse.Namespace) -> Any:
                return build_client(parsed_args, cortex_training_client_cls)

            client_factory = default_client_factory
        elif client_factory is None:

            def empty_client_factory(_parsed_args: argparse.Namespace) -> None:
                return None

            client_factory = empty_client_factory
        return _run(args, client_factory, stdout, stdin, stderr)
    except KeyboardInterrupt as exc:
        command = getattr(args, "command", None)
        if command not in _CHUNK_DOWNLOAD_COMMANDS:
            raise
        if not _resumable_download(args):
            print("interrupted", file=stderr)
        elif getattr(exc, "download_progress_saved", False):
            print(f"interrupted; {_RESUME_HINT}", file=stderr)
        else:
            print("interrupted before any output was saved", file=stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {_format_error(exc)}", file=stderr)
        _print_resume_hint(args, exc, stderr)
        return 1
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        print(
            f"error: missing Python dependency '{missing}'. "
            "Install the missing package, or install this repo first, "
            "for example: python3 -m pip install -e .",
            file=stderr,
        )
        return 1
    except Exception as exc:
        if not _is_connector_error(exc):
            raise
        print(f"error: {_format_connector_error(exc)}", file=stderr)
        _print_resume_hint(args, exc, stderr)
        return 1


_RESUME_HINT = (
    "downloaded chunks were saved. Run the same command with --resume to continue."
)
_FAILED_RESUME_HINT = (
    "Downloaded chunks were saved. Run the same command with --resume to continue."
)


def _resumable_download(args: argparse.Namespace | None) -> bool:
    command = getattr(args, "command", None)
    if command == "download-metrics":
        return True
    return command == "download-log" and getattr(args, "log_type", None) == "stdout"


def _print_resume_hint(args: argparse.Namespace | None, exc: BaseException, stderr: TextIO) -> None:
    if _resumable_download(args) and getattr(exc, "download_progress_saved", False):
        print(_FAILED_RESUME_HINT, file=stderr)


def _is_connector_error(exc: BaseException) -> bool:
    try:
        from snowflake.connector import errors as sf
    except ImportError:  # pragma: no cover - the connector is a dependency
        return False
    return isinstance(exc, sf.Error)


def _format_connector_error(exc: BaseException) -> str:
    message = str(exc).strip() or type(exc).__name__
    query_id = getattr(exc, "sfqid", None)
    if query_id and str(query_id) not in message:
        message += f"\nsnowflake query id: {query_id}"
    return message


if __name__ == "__main__":
    raise SystemExit(main())
