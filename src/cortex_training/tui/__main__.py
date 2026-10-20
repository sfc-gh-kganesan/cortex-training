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

"""Resolve shared CLI connection arguments and launch the read-only TUI."""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

_CONNECT_HINT = ("to connect, run 'cortex-training login config.json', set CORTEX_TRAINING_CONFIG, pass "
                 "--config config.json, --connection NAME, --base-url (local/mock), or --host + --pat; "
                 "or try the TUI without a connection: --debug")
# Bounds for every TUI read.
READ_MAX_RETRIES = 2
READ_TIMEOUT = (10.0, 45.0)
# The oldest Textual the TUI is tested on (pyproject.toml requires it, below 9).
TEXTUAL_FLOOR = (8, 2, 8)
_TEXTUAL_REQUIREMENT = "textual>=8.2.8,<9"


def _version_tuple(version: str) -> tuple:
    """The leading release numbers of a version string: ``8.2.8.dev1`` is (8, 2, 8)."""
    numbers = []
    for part in version.split(".")[:3]:
        match = re.match(r"\d+", part)
        if match is None:
            break
        numbers.append(int(match.group()))
    return tuple(numbers)


def textual_problem() -> str | None:
    """Why the installed Textual cannot run the TUI, else None."""
    from importlib import metadata

    try:
        installed = metadata.version("textual")
    except metadata.PackageNotFoundError:
        return f"the TUI needs the textual package: pip install '{_TEXTUAL_REQUIREMENT}'"
    if _version_tuple(installed) < TEXTUAL_FLOOR:
        floor = ".".join(map(str, TEXTUAL_FLOOR))
        return (f"the TUI needs textual {floor} or later, and {installed} is installed: "
                f"pip install --upgrade '{_TEXTUAL_REQUIREMENT}'")
    return None


def _escaped(text: str) -> str:
    """``text`` for one error line on stderr: each character a terminal would act on
    (control, format and other unprintable ones) becomes its escape, such as ``\\x1b``."""
    return "".join(ch if ch.isprintable() else ascii(ch)[1:-1] for ch in text)


def _positive_finite(value: str) -> float:
    number = float(value)
    if number <= 0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def _build_arg_parser(*, prog: str = "cortex-training tui") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Read-only TUI for Cortex Training jobs: logs and GPU usage.",
    )
    p.add_argument(
        "job_id", nargs="?",
        help="Job (session) id whose details to open directly, on the Logs tab. If omitted, the TUI shows the job list.",
    )
    p.add_argument(
        "--sub-job-id",
        help=("Open the job details scoped to this sub-job (JOB_ID:TYPE:INDEX, for example JOB_ID:training:0); "
              "JOB_ID may then be omitted."),
    )
    p.add_argument(
        "--config",
        dest="config",
        default=os.environ.get("CORTEX_TRAINING_CONFIG"),
        help=(
            "Path to a reusable Cortex Training config or credential JSON file "
            "(same format as the cortex-training CLI config)."
        ),
    )
    p.add_argument(
        "--connection",
        "-c",
        default=os.environ.get("CORTEX_TRAINING_CONNECTION"),
        help="Snowflake connection profile name.",
    )
    p.add_argument("--base-url", help="Direct base URL for a local or otherwise compatible server.")
    p.add_argument("--host", help="Snowflake account host for PAT auth.")
    p.add_argument("--pat", help="Programmatic access token.")
    p.add_argument("--database", help="Database containing the endpoint.")
    # Left as None so config-file / env values can fill them; the CLI resolver
    # applies the PUBLIC / cortex-training defaults when nothing is set.
    p.add_argument("--schema", default=None, help="Schema containing the endpoint. Defaults to PUBLIC.")
    p.add_argument("--endpoint", default=None, help="REST endpoint name. Defaults to cortex-training.")
    p.add_argument(
        "--poll-interval",
        type=_positive_finite,
        default=1.0,
        help="Minimum seconds between log polls per source (request-rate floor).",
    )
    p.add_argument(
        "--poll-timeout", type=float, default=1800.0,
        help=("Seconds the client waits for an asynchronous request's result (default 1800), as in the CLI; "
              "the TUI sends none, so it has no effect here."),
    )
    p.add_argument("--no-verify-ssl", action="store_true",
                   help="Disable SSL certificate verification for authenticated requests.")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Run the TUI on synthetic jobs and telemetry, with no connection (to try it or debug it).",
    )
    return p


def run(argv=None, *, prog: str = "cortex-training tui") -> int:
    parser = _build_arg_parser(prog=prog)
    args = parser.parse_args(argv)
    if args.sub_job_id and not args.job_id:
        # A sub-job id is JOB_ID:TYPE:INDEX, so it also names its job.
        parts = args.sub_job_id.rsplit(":", 2)
        if len(parts) != 3 or not all(parts):
            parser.error("--sub-job-id expects JOB_ID:TYPE:INDEX, for example JOB_ID:training:0")
        args.job_id = parts[0]

    problem = textual_problem()
    if problem:
        parser.exit(2, f"{parser.prog}: error: {problem}\n")
    try:
        from cortex_training.tui.app import CortexTrainingLogTUI
    except ImportError as exc:
        parser.exit(2, f"{parser.prog}: error: the TUI couldn't load ({exc}); reinstall cortex-training, "
                       f"or pip install --upgrade '{_TEXTUAL_REQUIREMENT}'\n")

    if args.debug:
        return _run_demo(args, CortexTrainingLogTUI)

    import cortex_training._cli as cli

    # Reuse the CLI's resolution: --config / login state / env / defaults.
    try:
        args = cli._resolve_args(args)
        # What the resolved arguments don't say, from the TUI's one read of the
        # config file they came from (the CLI's resolution reads it for itself):
        # the host or base URL it gives by itself (for the commands the TUI shows to copy).
        selected = getattr(args, "_legacy_config_selected", False) and args.config
        config = cli._load_config(args.config) if selected else {}
        from_config = _config_connection(cli, config)
    except (OSError, ValueError, RuntimeError) as exc:
        # A config the TUI can't use (a value of the wrong type, such as a quoted
        # poll_interval, JSON that doesn't parse, an unknown key, a file that isn't
        # there), or flags that don't go together: one error line, as the CLI gives,
        # not a traceback. Escaped: it can quote the file's keys.
        parser.exit(2, f"{parser.prog}: error: {_escaped(str(exc))}\n")
    try:
        args = cli._normalize_connection_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.use_connection_profile and not args.database:
        parser.error("provide --database or set CORTEX_TRAINING_DATABASE/SNOWFLAKE_DATABASE")
    if (
        not args.use_connection_profile
        and args.base_url is None
        and (args.host is None or args.pat is None)
    ):
        parser.error(f"no connection configured: {_CONNECT_HINT}")

    # Every read the TUI makes, the job list's too, is bounded here: the client's own
    # defaults (ten retries, a ten-minute read timeout) left a view silent for
    # about a minute on a 404 or a failing service.
    args.max_retries, args.request_timeout = READ_MAX_RETRIES, READ_TIMEOUT
    from cortex_training.snowflake_auth import SnowflakeProfileError

    # --poll-interval paces the TUI's log polls only. The client's own poll interval
    # is for waiting on an asynchronous result, which the TUI never does, and it
    # refuses more than 6 s: it keeps the default, so a slow --poll-interval works.
    log_poll_interval, args.poll_interval = args.poll_interval, parser.get_default("poll_interval")
    try:
        client = cli.build_client(args, cli._load_cortex_training_client_class())
    except (SnowflakeProfileError, ValueError) as exc:
        # With no config, login state or flags, the TUI tries the configured-default
        # Snowflake connection, which a first run doesn't have: say how to connect.
        parser.exit(2, f"{parser.prog}: error: {exc}\n{_CONNECT_HINT}\n")
    # The TUI only reads: "database or schema not found" is shown, as a for_job client
    # shows it, not answered with CREATE DATABASE / SCHEMA and a retry.
    client.auto_create_database = False

    # job_id is optional: given, open that job's details; omitted, open the job
    # list so the user can choose one.
    app = CortexTrainingLogTUI(client, args.job_id, sub_job_id=args.sub_job_id, poll_interval=log_poll_interval)
    app.sign_in = _sign_in(cli, args, from_config)  # for the commands the TUI shows to copy
    app.run()
    return app.return_code or 0  # non-zero when the app stopped on an unhandled error


def _sign_in(cli, args, from_config: dict) -> dict:
    """How the TUI's client connects, for the commands it shows to copy (read by
    ``views.sign_in_options``). It never holds the PAT; ``config`` is the config
    file as an absolute path, so a copied command finds it from any directory."""
    if args.use_connection_profile:
        return {"kind": "profile"}
    config = None
    if getattr(args, "_legacy_config_selected", False) and args.config:
        config = str(Path(args.config).expanduser().resolve())
    if args.base_url:
        return {"kind": "url", "config": config, "base_url": args.base_url,
                "base_url_in_config": from_config.get("base_url") == args.base_url}
    host = cli._normalize_host(args.host or "")
    return {"kind": "pat", "config": config, "host": host,
            "host_in_config": bool(host) and from_config.get("host") == host}


def _config_connection(cli, values: dict) -> dict:
    """What the config file (``values``, as the CLI reads it; empty for none) gives by
    itself: its ``base_url`` and PAT ``host`` (normalized), None where it gives neither."""
    base_url, host = cli._config_str(values, "base_url"), cli._config_str(values, "host")
    if base_url and not cli._has_url_scheme(base_url) and host is None:
        base_url, host = None, base_url  # the CLI reads a base URL without a scheme as the PAT host
    return {"base_url": base_url, "host": cli._normalize_host(host) if host else None}


def _run_demo(args, app_class, *, account_list: str | None = None) -> int:
    """Run against :class:`~cortex_training.tui.demo.DemoClient` with throwaway local stores."""
    import tempfile

    from cortex_training.tui.demo import DemoClient

    previous = os.environ.get("CORTEX_TRAINING_TUI_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix="cortex-training-demo-") as cache:
        # Demo caches must never land in the real cache dir.
        os.environ["CORTEX_TRAINING_TUI_CACHE_DIR"] = cache
        try:
            client = DemoClient(account_list=account_list)
            app = app_class(client, args.job_id, sub_job_id=args.sub_job_id, poll_interval=args.poll_interval)
            app.title = "cortex training (debug)"  # synthetic jobs: say so on every screen
            app.run()
        finally:
            # In-process callers (and tests) must not keep pointing at the deleted dir.
            if previous is None:
                os.environ.pop("CORTEX_TRAINING_TUI_CACHE_DIR", None)
            else:
                os.environ["CORTEX_TRAINING_TUI_CACHE_DIR"] = previous
    return app.return_code or 0


if __name__ == "__main__":
    raise SystemExit(run())
