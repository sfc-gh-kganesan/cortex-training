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

"""Smoke tests for the Textual app (job picker -> log screen). Skipped unless
the optional ``textual`` extra is installed (``pip install 'cortex-training[tui]'``)."""

import argparse
import asyncio
import glob
import inspect
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.widgets import Input  # noqa: E402
from textual.widgets import DataTable  # noqa: E402
from textual.widgets import Label  # noqa: E402
from textual.widgets import Static  # noqa: E402
from textual.widgets import TabbedContent  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training import CortexTrainingClient  # noqa: E402
from cortex_training import InvalidPageToken  # noqa: E402
from cortex_training.client import DownloadCancelled  # noqa: E402
from cortex_training.tui import __main__ as tui_main  # noqa: E402
from cortex_training.tui.account import ACCOUNT  # noqa: E402
from cortex_training.tui.account import SCHEMA  # noqa: E402
from cortex_training.tui.account import JobLister  # noqa: E402
from cortex_training.tui.account import Listing  # noqa: E402
from cortex_training.tui.account import scope_text  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.job_list import JobListScreen  # noqa: E402
from cortex_training.tui.logs import LogScreen  # noqa: E402
from tests.tui_fake import FakeClient  # noqa: E402
from cortex_training.tui.resource_view import ResourceTab  # noqa: E402
from cortex_training.tui.theme import ACCENT  # noqa: E402
from cortex_training.tui import views  # noqa: E402
from cortex_training.tui.untrusted import printable_client  # noqa: E402
from cortex_training.tui.format import wrap_log_line  # noqa: E402
from cortex_training.tui.resource import RANGES  # noqa: E402
from cortex_training.tui.screens import JobSummaryScreen  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    # The saved console and caches live under the per-job cache dir; keep every
    # test (including those that do not set it themselves) out of ~/.cache.
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "isolated-cache"))
    import cortex_training.tui.widgets as app_module

    # ... and out of ~/.cortex-training-errors.log: a test that reads the log finds it in tmp_path.
    monkeypatch.setattr(app_module, "_ERROR_LOG", str(tmp_path / "errors.log"))
    # ... and out of the system's temporary directory, where the app's sweep removes the
    # staging directories it takes for abandoned.
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    yield
    from cortex_training.tui.cache_cleanup import _release_pending

    _release_pending()  # a closed screen's lease, still in its grace, holds a lock file open


def _client():
    c = MagicMock()
    # The account-wide list isn't deployed (what accounts get today), so the job
    # list shows list_jobs, the configured schema's jobs.
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [
        {"job_id": "7", "status": "RUNNING", "sub_jobs": [{"job_type": "training"}]},
        {"job_id": "old", "status": "FAILED"},
    ]
    # Sources come from get_job().sub_jobs now (one per sub-job). RUNNING keeps
    # the live tail path; terminal-job behavior is covered by overriding get_job.
    c.get_job.return_value = {
        "status": "RUNNING",
        "sub_jobs": [
            {"sub_job_id": "7:training:0", "job_type": "training"},
            {"sub_job_id": "7:sampling:0", "job_type": "sampling"},
        ],
    }
    # Finite results so the tail worker doesn't loop forever.
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    return c


def _located_client():
    # Names its jobs' caches as a real client does (by service, endpoint, database and
    # schema), so they are the cache's to count and remove: a bare "7" is left alone.
    c = _client()
    c.base_url, c.endpoint = "https://ACCOUNT.snowflakecomputing.com", "cortex-training"
    c.database, c.schema = "CORTEX_TRAINING_DB", "PUBLIC"
    return c


async def _wait(pilot, app, predicate, tries=100):
    for _ in range(tries):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _settle(app, pilot):
    """Stop background workers and drain before ``run_test`` tears down, so a
    thread worker (ours, or Textual's internal Log resize worker) can't call into
    a half-destroyed app and surface a NoActiveAppError as a WorkerError. Without
    this the live-tail/resize workers make these tests flaky."""
    try:
        app.workers.cancel_all()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(5):
        await pilot.pause()


async def _run_with_job_id():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, LogScreen) and len(app.screen.query("#sources ListItem")) == 2
        )
        assert ok, "log screen / sources did not load"
        screen = app.screen
        assert screen._source_by_item["src-0"] == "7:training:0"
        assert screen._source_by_item["src-1"] == "7:sampling:0"
        await _settle(app, pilot)


async def _run_with_picker():
    app = CortexTrainingLogTUI(_client(), poll_interval=0.01)  # no job_id -> picker
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, JobListScreen) and len(app.screen._job_by_item) >= 2
        )
        assert ok, "job picker did not load jobs"
        screen = app.screen
        # active job (RUNNING) sorted first
        assert screen.whole_id("job-0") == "7"
        assert screen.whole_id("job-1") == "old"
        assert re.search(r"last refreshed \d{2}:\d{2}:\d{2}", app.sub_title or "")
        await _settle(app, pilot)


def test_app_with_job_id_opens_logs():
    asyncio.run(_run_with_job_id())


def test_app_without_job_id_shows_job_picker():
    asyncio.run(_run_with_picker())


def test_account_list_keeps_equal_ids_from_two_schemas():
    async def run():
        c = _client()
        c.database, c.schema = "DB", "HOME"
        c.list_jobs_page.side_effect = None
        c.list_jobs_page.return_value = {
            "jobs": [
                {"job_id": "same", "status": "FAILED", "database_name": "DB", "schema_name": "ONE"},
                {"job_id": "same", "status": "FAILED", "database_name": "DB", "schema_name": "TWO"},
            ], "next_page_token": None}
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        app._lister.account_wide = True
        async with app.run_test() as pilot:
            assert await _wait(pilot, app, lambda: len(getattr(app.screen, "_job_by_item", {})) == 2)
            screen = app.screen
            assert [screen.whole_id(key) for key in screen._job_by_item] == ["same", "same"]
            assert {job["schema_name"] for job in screen._jobs_by_key.values()} == {"ONE", "TWO"}
            await _settle(app, pilot)

    asyncio.run(run())


def test_location_scopes_cache_identity_and_unknown_not_found_is_historical():
    from cortex_training.tui.widgets import _cache_job_id

    class Client:
        base_url = "https://ACCOUNT.snowflakecomputing.com"
        endpoint = "cortex-training"
        database = "DB"

        def __init__(self, schema):
            self.schema = schema

    assert _cache_job_id(Client("ONE"), "same") != _cache_job_id(Client("TWO"), "same")
    job = {"job_id": "missing", "status": "UNKNOWN",
           "reason_details": "the job was not found in the Control Plane"}
    screen = LogScreen(_client(), "missing", job=job)
    assert not screen._live()
    screen._job_created_at = "created"
    screen._console_dir.mkdir(parents=True)
    (screen._console_dir / ".complete").write_text("created")
    assert screen._kept_console()
    entry = {"_raw": "step 1 loss\rstep 2 done"}
    assert screen._newer_tail_lines([(entry, entry["_raw"])], [], False, "loss", None)[0] == []


def test_copied_job_commands_keep_a_custom_endpoint():
    from cortex_training.tui.views import cli_options

    c = _client()
    c.database, c.schema, c.endpoint, c._auth_provider = "DB", "SCHEMA", "custom-endpoint", None
    command = " ".join(cli_options(**LogScreen(c, "7")._configured_location()))
    assert "--database DB" in command and "--schema SCHEMA" in command
    assert "--endpoint custom-endpoint" in command


def _listed_units(screen) -> int:
    """The GPUs (or nodes) the Resources panel lists, without ALL."""
    return sum(1 for entry in screen._resource_entries if entry[0] == "unit" and entry[1] is not None)


async def _run_resource_tab(tmp_path):
    c = _client()

    def download(job_id, output_dir, *, resume=False, **_kwargs):
        assert job_id == "7"
        assert resume is True
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        _write_gpu_metrics(output_dir, "7:sampling:0", gpu_count=1)
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(130, 36)) as pilot:  # the header beside the panel fits it whole
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and app.screen._current_source == "7:training:0",
        )
        assert ok
        tabs = app.screen.query_one("#job-tabs", TabbedContent)
        assert tabs.active == "logs"
        # Serving is composed for every job, hidden for a training one.
        panes = {pane.id for pane in app.screen.query("TabPane")}
        assert panes == {"serving", "logs", "resource", "training", "checkpoints"}
        c.download_metrics.assert_not_called()

        tabs.active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: c.download_metrics.call_count == 1
            and _listed_units(app.screen) == 2
            and not app.screen._resource_loading,
        )
        assert ok, "Resource did not load two GPU rows"
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "2 reporting" in status
        assert "training #0" in status  # named by type and index
        assert "reserved" not in status  # the memory note is a deployment's
        calls = c.download_metrics.call_count

        # w cycles the GPU window; the digits do nothing on the job detail screen.
        for _ in range(len(RANGES)):
            if app.screen._resource_range == "all":
                break
            await pilot.press("w")
        ok = await _wait(
            pilot,
            app,
            lambda: app.screen._resource_range == "all"
            # Named by type and index; at this width the window may show as its key (all).
            and re.match(r"training #0 (\(1/2\)|\(1 of 2\))  ·  (whole job|all)  ·",
                         str(app.screen.query_one("#resource-status", Static).render())),
        )
        assert ok
        await pilot.press("w")
        ok = await _wait(pilot, app, lambda: app.screen._resource_range == "10m")
        assert ok
        assert c.download_metrics.call_count == calls
        # The bottom row is the window selector, the window shown lit, then only the keys the
        # footer doesn't show; a click on a segment picks its window.
        keys = app.screen._resource_keys()
        lit = [keys.plain[span.start:span.end] for span in keys.spans if f"on {ACCENT}" in str(span.style)]
        assert lit == [" 10m "]
        assert keys.plain.startswith("window ") and "↑↓ GPU" in keys.plain
        assert "refresh" not in keys.plain and "sub-job" not in keys.plain
        assert all(app.screen._resource_keys(width).cell_length <= width for width in (60, 40, 25, 12))
        assert "@click=screen.resource_range('30m')" in str(keys.spans[2].style)
        kpis = str(app.screen.query_one("#resource-kpis", Static).render())
        assert "GPU util" in kpis and "GPUs reporting" in kpis
        app.screen._content_width = lambda: 10
        app.screen._write_line("x" * 20)
        assert len(app.screen._logview._lines) > len(app.screen._shown_lines)
        app.screen.action_refresh_sources()
        ok = await _wait(
            pilot,
            app,
            lambda: c.download_metrics.call_count == calls + 1 and not app.screen._resource_loading,
        )
        assert ok, "r did not refresh metrics on Resource"
        tabs.active = "logs"
        app.screen._content_width = lambda: 200
        ok = await _wait(
            pilot,
            app,
            lambda: app.screen._active_tab == "logs" and app.screen._wrap_width == 200,
        )
        assert ok
        assert len(app.screen._logview._lines) == len(app.screen._shown_lines)
        downloads = c.download_metrics.call_count
        app.screen.action_refresh_sources()
        await pilot.pause()
        assert c.download_metrics.call_count == downloads
        app.screen._start_tail("7:sampling:0")
        tabs.active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: _listed_units(app.screen) == 1
            and "sampling #0" in str(app.screen.query_one("#resource-status", Static).render()),
        )
        assert ok, "Resource did not follow the Logs-selected sub-job"
        assert c.download_metrics.call_count == downloads
        tabs.active = "logs"
        await _wait(pilot, app, lambda: app.screen._active_tab == "logs")
        previous_range = app.screen._resource_range
        app.screen.action_resource_range("all")
        assert app.screen._resource_range == previous_range
        await pilot.press("t")
        ok = await _wait(pilot, app, lambda: tabs.active == "resource")
        assert ok
        await pilot.press("t")  # t goes on to the next tab, round to Logs after the last
        ok = await _wait(pilot, app, lambda: tabs.active == "training")
        assert ok
        await pilot.press("t")
        ok = await _wait(pilot, app, lambda: tabs.active == "checkpoints")
        assert ok
        await pilot.press("t")
        ok = await _wait(pilot, app, lambda: tabs.active == "logs")
        assert ok
        await _settle(app, pilot)


def test_resource_tab_is_lazy_and_renders_multiple_gpus(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_tab(tmp_path))


def test_a_deployment_s_resources_say_its_memory_is_reserved(tmp_path):
    c = _client()
    c.get_job.return_value = {"job_id": "7", "status": "RUNNING", "sub_jobs": [
        {"sub_job_id": "7:sampling:0", "job_type": "sampling", "status": "RUNNING",
         "inference_config": {"deployment_class": "EVALUATION", "num_replicas": 1.0}}]}

    def download(job_id, output_dir, **_kwargs):
        _write_gpu_metrics(output_dir, "7:sampling:0", gpu_count=1)
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(130, 36)) as pilot:
            assert await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen)
                               and app.screen._current_source == "7:sampling:0")
            await pilot.press("g")
            assert await _wait(pilot, app, lambda: "reserved up front" in
                               str(app.screen.query_one("#resource-status", Static).render()), tries=300)
            app.workers.cancel_all()
            await pilot.pause()

    asyncio.run(run())


async def _run_resource_cold_cache_failure():
    c = _client()
    c.download_metrics.side_effect = RuntimeError("metrics [/]")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: c.download_metrics.called and not app.screen._resource_loading,
        )
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "GPU metrics unavailable: RuntimeError: metrics [/]" in status
        assert _listed_units(app.screen) == 0
        await _settle(app, pilot)


def test_resource_cold_cache_download_failure_is_visible(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_cold_cache_failure())


async def _run_resource_parser_failure_resets_loading():
    c = _client()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and "OverflowError: bad metric"
            in str(app.screen.query_one("#resource-status", Static).render()),
        )
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "GPU metrics unavailable: OverflowError: bad metric" in status
        await _settle(app, pilot)


def test_resource_parser_failure_does_not_leave_loading_stuck(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    import cortex_training.tui.resource_view as app_module

    loader = MagicMock()
    loader.last = None  # nothing read yet
    loader.load.side_effect = OverflowError("bad metric")
    monkeypatch.setattr(app_module, "resource_loader", lambda root: loader)
    asyncio.run(_run_resource_parser_failure_resets_loading())


def _assert_header_fits(screen):
    """The Resources header is its two lines, neither wider than the tab: nothing under it moves."""
    from rich.cells import cell_len

    status = screen.query_one("#resource-status", Static)
    lines = str(status.render()).split("\n")
    assert len(lines) == 2, lines
    assert all(cell_len(line) <= status.content_size.width for line in lines), (status.content_size.width, lines)


async def _run_resource_partial_download_failure():
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        raise RuntimeError("later sub-job failed")

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(130, 36)) as pilot:  # the header beside the panel fits it whole
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and _listed_units(app.screen) == 2,
        )
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        # The line makes room for the error (100 columns here: "cached · " gives way), never wrapped.
        assert "refresh failed: RuntimeError: later sub-job failed" in status
        _assert_header_fits(app.screen)
        await _settle(app, pilot)


def test_resource_partial_download_is_reparsed_after_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_partial_download_failure())


async def _run_resource_narrow_header_keeps_the_scrape_interval_last():
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(130, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: not app.screen._resource_loading and "scraped every 5 s" in str(
            app.screen.query_one("#resource-status", Static).render()))
        assert ok
        seen = []
        for width in range(130, 40, -2):
            await pilot.resize_terminal(width, 36)
            await pilot.pause()
            second = str(app.screen.query_one("#resource-status", Static).render()).split("\n")[1]
            # The interval, and the freshness clock ("newest 30 s ago (12:34:56)").
            seen.append(("scraped every 5 s" in second, "(" in second))
        # Narrower, the clock goes first, then the interval: never the interval while the clock shows.
        assert (True, True) in seen and (True, False) in seen and (False, True) not in seen, seen
        await _settle(app, pilot)


def test_a_narrow_resource_header_drops_the_clock_before_the_scrape_interval(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_narrow_header_keeps_the_scrape_interval_last())


async def _run_resource_partial_other_sub_job_retries():
    c = _client()
    calls = 0

    def download(job_id, output_dir, *, resume, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            _write_gpu_metrics(output_dir, "7:sampling:0", gpu_count=1)
            raise RuntimeError("training metrics failed")
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source == "7:training:0")
        assert ok
        tabs = app.screen.query_one("#job-tabs", TabbedContent)
        tabs.active = "resource"
        ok = await _wait(pilot, app, lambda: calls == 1 and not app.screen._resource_loading)
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "GPU metrics unavailable: RuntimeError: training metrics failed" in status
        assert "no GPU metrics" not in status
        tabs.active = "logs"
        await _wait(pilot, app, lambda: app.screen._logs_active())
        tabs.active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: calls == 2
            and not app.screen._resource_loading
            and _listed_units(app.screen) == 2,
        )
        assert ok, "re-entering Resource did not retry after a partial failure for another sub-job"
        await _settle(app, pilot)


def test_resource_partial_other_sub_job_failure_retries_on_reentry(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_partial_other_sub_job_retries())


async def _run_resource_cached_failure():
    c = _client()
    c.download_metrics.side_effect = RuntimeError("refresh unavailable")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(130, 36)) as pilot:  # the header beside the panel fits it whole
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and _listed_units(app.screen) == 2,
        )
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "refresh failed: RuntimeError: refresh unavailable" in status  # on the line, not wrapped
        _assert_header_fits(app.screen)
        app.screen.action_resource_range("all")
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "refresh failed: RuntimeError: refresh unavailable" in status
        _assert_header_fits(app.screen)
        assert _listed_units(app.screen) == 2
        await _settle(app, pilot)


def test_resource_failed_refresh_keeps_cached_rows_and_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import job_cache_dir

    _write_gpu_metrics(job_cache_dir("7") / "metrics", "7:training:0", gpu_count=2)
    asyncio.run(_run_resource_cached_failure())


async def _run_resource_unsupported_warning():
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        path = _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
        row = json.loads(path.read_text(encoding="utf-8"))
        row["record_version"] = 1
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        path.with_name(".gpu.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}),
            encoding="utf-8",
        )
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: not app.screen._resource_loading)
        assert ok
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "training #0" in status  # named by type and index
        # Why the view is empty is the card in the pane; the skipped rows stay on the header.
        card = app.screen.query_one("#resource-stats", Static)
        assert card.has_class("-empty")
        assert "No usable GPU metrics" in str(card.render())
        assert "no usable GPU metrics for this sub-job" in str(card.render())
        assert "no usable GPU metrics" not in status
        assert "1 row skipped (malformed or unsupported)" in status
        await _settle(app, pilot)


def test_resource_all_unsupported_rows_show_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_unsupported_warning())


async def _run_resource_narrow_no_samples():
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        path = _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
        row = json.loads(path.read_text(encoding="utf-8"))
        row["gpu_util_pct"] = None
        payload = json.dumps(row) + "\n"
        path.write_text(payload, encoding="utf-8")
        path.with_name(".gpu.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}),
            encoding="utf-8",
        )
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(60, 30)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and _listed_units(app.screen) == 1,
        )
        assert ok
        # A metric without samples has no chart and no stats row: the memory % still charts.
        charts = str(app.screen.query_one("#resource-charts", Static).render())
        assert "GPU util %" not in charts and "GPU memory used %" in charts
        width = app.screen.query_one("#resource-charts", Static).content_size.width
        assert all(len(line) <= width for line in charts.splitlines())
        calls = c.download_metrics.call_count
        app.screen.on_resize(None)
        await pilot.pause()
        assert c.download_metrics.call_count == calls
        await _settle(app, pilot)


def test_resource_narrow_table_no_samples_and_resize_is_local(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_narrow_no_samples())


async def _run_resource_chart_fits_content_width():
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(60, 30)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        chart = app.screen.query_one("#resource-charts", Static)

        def drawn_to_width() -> bool:
            # The charts are redrawn once the layout settles (the panel beside them is sized
            # to its labels): wait for a drawing at the final width, not one from before it.
            lines = str(chart.render()).splitlines()
            return bool(lines) and all(len(line) <= chart.content_size.width for line in lines)

        ok = await _wait(
            pilot,
            app,
            lambda: not app.screen._resource_loading
            and _listed_units(app.screen) == 1
            and drawn_to_width(),
        )
        assert ok
        # The pane's padding is outside the charts: they are drawn to the width they have.
        pane = app.screen.query_one("#resource-pane")
        assert pane.region.width - chart.content_region.width == 3  # its padding and the scrollbar's gutter
        assert chart.content_size.width == chart.content_region.width
        await _settle(app, pilot)


def test_resource_chart_accounts_for_widget_padding(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_resource_chart_fits_content_width())


async def _run_resource_ignores_refresh_while_busy(started, release):
    c = _client()

    def download(job_id, output_dir, *, resume, **_kwargs):
        started.set()
        assert release.wait(timeout=5)
        _write_gpu_metrics(output_dir, "7:training:0")
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, started.is_set)
        assert ok
        app.screen.action_refresh_sources()
        app.screen.action_refresh_sources()
        await pilot.pause()
        assert c.download_metrics.call_count == 1
        release.set()
        ok = await _wait(pilot, app, lambda: not app.screen._resource_loading)
        assert ok
        await _settle(app, pilot)


def test_resource_ignores_refresh_while_busy(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    started = threading.Event()
    release = threading.Event()
    try:
        asyncio.run(_run_resource_ignores_refresh_while_busy(started, release))
    finally:
        release.set()


async def _run_resource_reopen_waits_for_lock(started, release):
    c = _client()
    calls = 0

    def download(job_id, output_dir, *, resume, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(timeout=5)
        _write_gpu_metrics(output_dir, "7:training:0")
        return []

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, started.is_set)
        assert ok
        app.pop_screen()
        await pilot.pause()
        app.push_screen(LogScreen(c, "7", poll_interval=0.01))
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        for _ in range(5):
            await pilot.pause()
        assert calls == 1
        release.set()
        ok = await _wait(
            pilot,
            app,
            lambda: calls == 2
            and not app.screen._resource_loading
            and _listed_units(app.screen) == 2,
        )
        assert ok, "reopened Resource did not continue after the prior download released the lock"
        await _settle(app, pilot)


def test_resource_reopen_waits_for_inflight_download(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    started = threading.Event()
    release = threading.Event()
    try:
        asyncio.run(_run_resource_reopen_waits_for_lock(started, release))
    finally:
        release.set()


async def _run_terminal_resource_default():
    c = _client()
    c.get_job.return_value = _terminal_job()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._sources_ready)
        assert ok
        assert app.screen._resource_range == "10m"  # every job opens on the last 10 minutes
        await _settle(app, pilot)


def test_direct_terminal_job_resource_defaults_to_ten_minutes():
    asyncio.run(_run_terminal_resource_default())


async def _run_active_to_terminal_resource_default():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._sources_ready)
        assert ok
        assert app.screen._resource_range == "10m"
        # The window never moves under the user when the job ends.
        app.screen._apply_terminal_status("COMPLETED", None)
        assert app.screen._resource_range == "10m"
        app.screen._resource_range = "30m"
        app.screen._apply_terminal_status("FAILED", None)
        assert app.screen._resource_range == "30m"
        await _settle(app, pilot)


def test_active_job_becoming_terminal_keeps_the_resource_window():
    asyncio.run(_run_active_to_terminal_resource_default())


async def _run_resource_escape_returns_to_parent(*, direct):
    c = _client()
    app = CortexTrainingLogTUI(c, "7" if direct else None, poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        if direct:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen))
        else:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen))
            assert ok
            app.push_screen(LogScreen(c, "7", poll_interval=0.01))
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen))
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: app.screen._resource_active())
        assert ok
        await pilot.press("escape")
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item,
        )
        assert ok
        if not direct:
            assert isinstance(app.screen, JobListScreen)
        await _settle(app, pilot)


def test_resource_escape_returns_from_direct_job():
    asyncio.run(_run_resource_escape_returns_to_parent(direct=True))


def test_resource_escape_returns_to_picker():
    asyncio.run(_run_resource_escape_returns_to_parent(direct=False))


async def _run_resource_guards(tmp_path):
    c = _client()
    c.download_metrics.side_effect = lambda job_id, output_dir, *, resume, **_options: (
        _write_gpu_metrics(output_dir, "7:training:0"),
        [],
    )[1]
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: not app.screen._resource_loading)
        assert ok
        screen = app.screen
        table = screen.query_one("#resource-list")
        screen.set_focus(table)
        width = screen._sources_width
        level = screen._min_level
        paused = screen._paused
        copied = []
        app.copy_to_clipboard = copied.append
        screen.action_filter()
        screen.action_cycle_level()
        screen.action_toggle_pause()
        screen.action_shrink_sources()
        screen.action_grow_sources()
        screen.action_save_log()
        for _ in range(5):
            await pilot.pause()
        assert screen.focused is table
        assert screen._sources_width == width
        assert screen._min_level == level
        assert screen._paused == paused
        assert copied == []
        assert not list(tmp_path.glob("cortex-training-*.log"))
        await _settle(app, pilot)


def test_log_only_actions_are_guarded_on_resource(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path))
    asyncio.run(_run_resource_guards(tmp_path))


async def _run_sub_job_scope(sub_job_id):
    c = _client()
    app = CortexTrainingLogTUI(c, "7", sub_job_id=sub_job_id, poll_interval=0.01)
    async with app.run_test(size=(100, 36)) as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._sources_ready)
        assert ok
        if sub_job_id == "7:sampling:0":
            assert list(app.screen._source_by_item.values()) == ["7:sampling:0"]
            assert app.screen._current_source == "7:sampling:0"
        else:
            assert app.screen._source_by_item == {}
            assert app.screen._current_source is None
            app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
            for _ in range(5):
                await pilot.pause()
            c.download_metrics.assert_not_called()
            labels = [str(item.query_one(Label).render()) for item in app.screen.query("#sources ListItem")]
            assert any("sub-job not found" in label for label in labels)
        await _settle(app, pilot)


def test_sub_job_id_scopes_source_list():
    asyncio.run(_run_sub_job_scope("7:sampling:0"))


def test_unmatched_sub_job_does_not_download_metrics():
    asyncio.run(_run_sub_job_scope("7:missing:0"))


async def _run_refresh_picker():
    c = _client()
    jobs = c.list_jobs.return_value
    refresh_started = threading.Event()
    finish_refresh = threading.Event()
    calls = 0

    def list_jobs(**_bounds):
        nonlocal calls
        calls += 1
        if calls > 1:
            refresh_started.set()
            finish_refresh.wait(timeout=5)
        return jobs

    c.list_jobs.side_effect = list_jobs
    app = CortexTrainingLogTUI(c, poll_interval=0.01)
    try:
        async with app.run_test() as pilot:
            ok = await _wait(
                pilot,
                app,
                lambda: isinstance(app.screen, JobListScreen)
                and len(app.screen._job_by_item) == 2,
            )
            assert ok, "initial jobs did not load"
            previous_refresh = app.screen._last_refreshed
            # Refresh re-populates with the same item IDs (job-0, job-1). Must
            # show that work is underway without discarding the last success.
            app.screen.action_refresh()
            ok = await _wait(pilot, app, refresh_started.is_set)
            assert ok, "refresh did not start"
            assert "refreshing…" in (app.sub_title or "")
            assert f"last refreshed {previous_refresh}" in (app.sub_title or "")
            finish_refresh.set()
            ok = await _wait(
                pilot,
                app,
                lambda: "refreshing…" not in (app.sub_title or "")
                and len(app.screen._job_by_item) == 2,
            )
            assert ok, "refresh did not finish cleanly"
            await _settle(app, pilot)
    finally:
        finish_refresh.set()


async def _run_refresh_sources():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, LogScreen) and len(app.screen.query("#sources ListItem")) == 2
        )
        assert ok, "initial sources did not load"
        app.screen.action_refresh_sources()
        ok = await _wait(pilot, app, lambda: len(app.screen.query("#sources ListItem")) == 2)
        assert ok, "source refresh changed item count (DuplicateIds regression)"
        await _settle(app, pilot)


def test_picker_refresh_no_duplicate_ids():
    asyncio.run(_run_refresh_picker())


def test_source_refresh_no_duplicate_ids():
    asyncio.run(_run_refresh_sources())


async def _run_empty_source_refresh_clears_selection():
    c = _client()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source is not None)
        assert ok
        # The service now says the job has no sub-jobs (an empty answer would say
        # nothing, and keep the sub-jobs the view knows).
        c.get_job.return_value = {"job_id": "7", "status": "RUNNING", "sub_jobs": []}
        app.screen.action_refresh_sources()
        ok = await _wait(
            pilot,
            app,
            lambda: app.screen._current_source is None
            and len(app.screen.query("#sources ListItem")) == 0,
        )
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        await pilot.pause()
        c.download_metrics.assert_not_called()
        status = str(app.screen.query_one("#resource-status", Static).render())
        assert "this job has no sub-jobs" in status
        await _settle(app, pilot)


def test_empty_source_refresh_clears_resource_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_empty_source_refresh_clears_selection())


async def _run_offline_fallback():
    c = _client()
    c.get_job.side_effect = RuntimeError("network down")  # can't fetch sub-jobs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and "7:training:0" in (app.screen._source_by_item or {}).values(),
        )
        assert ok, "cached sub-job not shown after get_job failure"
        await _settle(app, pilot)


def test_offline_shows_cached_sources(tmp_path, monkeypatch):
    # Seed a cached sub-job for job "7", then make get_job fail: the source list
    # must fall back to the cached sub-job so logs stay reachable.
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_offline_fallback())


def test_offline_shows_metrics_only_cached_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import job_cache_dir

    _write_gpu_metrics(job_cache_dir("7") / "metrics", "7:training:0", gpu_count=2)
    asyncio.run(_run_offline_fallback())


async def _run_terminal_cache_only():
    c = _client()
    c.get_job.return_value = {  # terminal -> zone gone; metadata still lists sub-jobs
        "status": "FAILED",
        "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
    }
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and "7:training:0" in (app.screen._source_by_item or {}).values(),
        )
        assert ok, "sub-job source not shown for terminal job"
        # A terminal job is served from cache only — no tail-logs operation call.
        c.tail_logs.assert_not_called()
        await _settle(app, pilot)


def test_terminal_job_is_cache_only(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_terminal_cache_only())


async def _run_resize_sources():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen))
        assert ok, "log screen did not open"
        s = app.screen
        w0 = s._sources_width
        s.action_grow_sources()
        assert s._sources_width == w0 + s._SOURCES_STEP
        s.action_shrink_sources()
        s.action_shrink_sources()
        assert s._sources_width == max(s._SOURCES_MIN, w0 - s._SOURCES_STEP)
        for _ in range(50):
            s.action_shrink_sources()
        assert s._sources_width == s._SOURCES_MIN  # clamps low
        for _ in range(50):
            s.action_grow_sources()
        # Clamps high: the log keeps its minimum width (an 80-column test screen).
        assert s._sources_width == s._sources_cap() == app.screen.size.width - s._LOG_MIN_WIDTH
        await _settle(app, pilot)


def test_sources_panel_resize():
    asyncio.run(_run_resize_sources())


async def _run_picker_filter():
    app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, JobListScreen) and len(app.screen._job_by_item) >= 2
        )
        assert ok, "jobs did not load"
        s = app.screen
        s.query_one("#jobfilter", Input).value = "old"  # fires Input.Changed → re-render
        ok = await _wait(
            pilot,
            app,
            lambda: [s.whole_id(key) for key in s._job_by_item] == ["old"],
        )
        assert ok, f"filter did not narrow the list: {s._job_by_item}"
        await _settle(app, pilot)


def test_picker_filter():
    asyncio.run(_run_picker_filter())


def test_a_search_builds_each_hidden_older_job_s_search_text_once(monkeypatch):
    # A search that matches no job of the last six months names the older jobs it
    # matches. Their search text is kept as the shown jobs' is, not built again on
    # every keystroke.
    from collections import Counter

    from cortex_training.tui import views

    now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    c = _client()
    c.list_jobs.return_value = [{"job_id": "7", "status": "RUNNING", "created_at": now.isoformat()}] + [
        {"job_id": f"old-{i}", "status": "FAILED",
         "created_at": (now - timedelta(days=400 + i)).isoformat()}
        for i in range(3)
    ]
    built = Counter()
    search_haystack = views.search_haystack

    def counted(job):
        built[views.job_id_of(job)] += 1
        return search_haystack(job)

    monkeypatch.setattr(views, "search_haystack", counted)

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01, now=now)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            assert ok, "jobs did not load"
            screen = app.screen
            for text in ("o", "ol", "old", "old-"):
                screen.query_one("#jobfilter", Input).value = text
                ok = await _wait(pilot, app, lambda: f"“{text}”" in (screen._nomatch or ""))
                assert ok, f"no message for {text!r}: {screen._nomatch!r}"
            assert {jid: built[jid] for jid in ("old-0", "old-1", "old-2")} == {"old-0": 1, "old-1": 1, "old-2": 1}
            await _settle(app, pilot)

    asyncio.run(run())


def test_job_table_replace_rows_shows_exactly_those_rows():
    # replace_rows fills the table through DataTable's own clear() and add_row():
    # the rows, their keys and the cursor must come out as the caller asked.
    from textual.app import App

    from cortex_training.tui.widgets import _cell
    from cortex_training.tui.job_list import _JobTable

    class Tables(App):
        CSS = "DataTable { height: 8; }"

        def compose(self):
            yield _JobTable(id="fast", cursor_type="row", fixed_columns=1)
            yield _JobTable(id="own", cursor_type="row", fixed_columns=1)

    def rows(count, tag):
        return [(f"{tag}-{i}", (_cell(f"{tag} {i}"), _cell(f"{i * 7}"))) for i in range(count)]

    def state(table):
        return (table.row_count, [row.key.value for row in table.ordered_rows],
                [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)],
                table.cursor_coordinate, table.virtual_size,
                [table.render_line(y).text for y in range(table.size.height)])

    async def run():
        app = Tables()
        async with app.run_test(size=(40, 20)) as pilot:
            fast, own = app.query_one("#fast", _JobTable), app.query_one("#own", _JobTable)
            for table in (fast, own):
                table.add_column("Job ID", width=8, key="Job ID")
                table.add_column("N", width=4, key="N")
            # Two tables filled the same way hold the same rows, however often refilled.
            for plan in (rows(30, "a"), rows(5, "b"), [], rows(12, "c")):
                fast.replace_rows(plan)
                own.replace_rows(plan)
                await pilot.pause()
                assert state(fast) == state(own)
                assert [row.key.value for row in fast.ordered_rows] == [key for key, _cells in plan]
                assert fast.row_count == len(plan)
            fast.move_cursor(row=9)
            own.move_cursor(row=9)
            fast.replace_rows(rows(20, "d"))
            own.replace_rows(rows(20, "d"))
            await pilot.pause()
            assert state(fast) == state(own)
            assert [str(cell) for cell in fast.get_row_at(0)] == ["d 0", "0"]

    asyncio.run(run())


async def _run_logscreen_controls():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._logview is not None)
        assert ok, "log screen did not open"
        s = app.screen
        # Following is managed manually (write-time), so the widget's blanket
        # auto_scroll stays off; pause just toggles the follow flag.
        assert s._logview.auto_scroll is False
        s.action_toggle_pause()
        assert s._paused is True
        s.action_toggle_pause()
        assert s._paused is False
        assert s._min_level is None
        s.action_cycle_level()
        assert s._min_level == "INFO"
        s.action_cycle_level()
        assert s._min_level == "WARNING"
        s.action_cycle_level()
        assert s._min_level == "ERROR"
        s.action_cycle_level()
        assert s._min_level is None
        await _settle(app, pilot)


def test_logscreen_controls():
    asyncio.run(_run_logscreen_controls())


async def _run_always_tails_bottom():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and app.screen._current_source is not None
            and len(app.screen.query("#sources ListItem")) == 2,
        )
        assert ok, "log screen / initial tail did not start"
        s = app.screen
        lv = s._logview
        # Writing past the viewport keeps the view pinned to the tail — every
        # viewer lands on the newest lines, no manual scroll needed.
        for i in range(200):
            s._write_line(f"line {i}")
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "log should stick to the bottom"
        # A new live line keeps us at the bottom.
        s._write_line("newest live line")
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "live tail should keep following the bottom"
        # Pause is the escape hatch: scroll up to read history and new lines
        # must not yank the user back down.
        s.action_toggle_pause()
        assert s._paused is True
        lv.scroll_home(animate=False)
        await pilot.pause()
        s._write_line("live while paused")
        await pilot.pause()
        assert not lv.is_vertical_scroll_end, "paused tail must not follow"
        # Un-pausing snaps back to the tail.
        s.action_toggle_pause()
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "un-pause should jump back to the bottom"
        await _settle(app, pilot)


def test_log_always_tails_bottom():
    asyncio.run(_run_always_tails_bottom())


async def _run_replay_lands_at_bottom():
    c = _client()
    c.get_job.return_value = {  # terminal → cache-only replay, no live poll
        "status": "FAILED",
        "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
    }
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source == "7:training:0"
        )
        assert ok, "log screen / cache replay did not start"
        s = app.screen
        lv = s._logview
        # The cached backlog is far taller than the viewport — exactly the
        # reopen-a-long-running-job case. The old code wrote the replay with
        # scroll_end=False and stranded the viewer at the oldest line; the fix
        # lands them on the newest. Wait for the replay, then assert the tail.
        ok = await _wait(pilot, app, lambda: any("cache line 299" in ln for ln in s._shown_lines))
        assert ok, "cache replay did not render"
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "cache replay must land at the tail, not the top"
        await _settle(app, pilot)


def test_cache_replay_lands_at_bottom(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": f"cache line {i}"} for i in range(300)])
    asyncio.run(_run_replay_lands_at_bottom())


async def _run_source_switch_resets_pause():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and app.screen._current_source is not None
            and len(app.screen.query("#sources ListItem")) == 2,
        )
        assert ok, "log screen / initial tail did not start"
        s = app.screen
        lv = s._logview
        # Pause on the first source (the user is reading history).
        s.action_toggle_pause()
        assert s._paused is True
        # Switching to another source is a fresh follow context: a stale pause
        # must clear so the new source lands on — and follows — its own tail
        # instead of being stranded at the top.
        other = s._source_by_item["src-1"]
        assert other != s._current_source
        s._start_tail(other)
        await pilot.pause()
        assert s._paused is False, "switching sources must clear a stale pause"
        assert s._current_source == other
        for i in range(200):
            s._write_line(f"line {i}")
        await pilot.pause()
        assert lv.is_vertical_scroll_end, "new source should follow its tail"
        await _settle(app, pilot)


def test_source_switch_resets_pause():
    asyncio.run(_run_source_switch_resets_pause())


async def _run_last_updated():
    c = _client()
    calls = {"n": 0}

    def tail(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"entries": [{"_raw": "live line"}], "next_cursor": "C1", "eof": False}
        return {"entries": [], "next_cursor": "C1", "eof": False}

    c.tail_logs.side_effect = tail
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        # Wait for the subtitle to pick up the freshness stamp from a live batch: a line
        # without a timestamp of its own is dated by when it arrived.
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen)
                         and "last line received" in (app.sub_title or ""))
        assert ok, "subtitle never showed a last-line stamp"
        assert app.screen._last_line_at is not None
        assert "read-only" not in app.sub_title  # dropped from the subtitle
        assert "RUNNING" in app.sub_title  # job status now on this line
        # The summary line under the title is the job's identity: its status as a
        # badge (glyph and word), beside its type and scope chips.
        from textual.widgets import Static

        summary = str(app.screen.query_one("#summary", Static).render())
        assert "● RUNNING" in summary and "all sub-jobs" in summary
        await _settle(app, pilot)


def test_subtitle_last_updated():
    asyncio.run(_run_last_updated())


async def _run_export():
    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._logview is not None)
        assert ok, "log screen did not open"
        s = app.screen
        s._current_source = "head:server"
        s.action_save_log()  # background worker writes the file
        ok = await _wait(
            pilot,
            app,
            lambda: bool(glob.glob(os.path.expanduser("~/cortex-training-7-*.log"))),
        )
        assert ok, "export file was not written"
        content = open(
            glob.glob(os.path.expanduser("~/cortex-training-7-*.log"))[0],
            encoding="utf-8",
        ).read()
        assert "alpha" in content and "beta" in content
        await _settle(app, pilot)


def test_log_export(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))  # so ~/ writes into tmp
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("head:server", [{"_raw": "alpha"}, {"_raw": "beta"}])
    asyncio.run(_run_export())


async def _run_wrap_and_select():
    from textual.widgets import Log

    app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        # Wait until the initial source is selected (its tail started + banner
        # written) so _start_tail won't reset _shown_lines under us.
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen)
            and app.screen._logview is not None
            and app.screen._current_source is not None
            and len(app.screen.query("#sources ListItem")) == 2,
        )
        assert ok, "log screen / initial tail did not start"
        s = app.screen
        # The log pane is a Log (not RichLog): it participates in Textual's
        # native text selection, and the screen can copy the selection.
        assert isinstance(s._logview, Log)
        assert s._logview.ALLOW_SELECT is True
        assert hasattr(s, "action_copy_text")  # ctrl+c → copy selection

        # Wrapping: a long line is split to the pane width on write. The pane
        # may prefix the line with its sub-job tag, which wraps with it.
        s._content_width = lambda: 10
        before = len(s._logview._lines)
        s._write_line("x" * 60)  # 60 / 10 = 6 visual lines, plus any tag
        assert len(s._logview._lines) - before == len(wrap_log_line(s._display("x" * 60), 10))
        assert s._shown_lines[-1] == "x" * 60  # buffer keeps the unwrapped, untagged line

        # Re-wrap on width change: widening collapses each buffered line to one. The
        # lines are wrapped off the UI thread, so the pane changes a moment later.
        s._content_width = lambda: 200
        s._rewrap()
        ok = await _wait(pilot, app, lambda: s._repaint_job is None and s._wrap_width == 200)
        assert ok, "the re-wrap did not land"
        assert len(s._logview._lines) == len(s._shown_lines)
        await _settle(app, pilot)


def test_log_wrap_and_selectable():
    asyncio.run(_run_wrap_and_select())


async def _pause_until(pilot, predicate, seconds=3.0):
    for _ in range(int(seconds / 0.05)):
        if predicate():
            return True
        await pilot.pause(0.05)
    return predicate()


async def _settled_pane(pilot, screen):
    """Let a re-wrap the screen started while it opened (its sources panel was sized) land."""
    await pilot.pause(2 * LogScreen._REWRAP_SETTLE_SECONDS)
    assert await _pause_until(pilot, lambda: screen._repaint_job is None)


async def _open_live_tail(app, pilot):
    ok = await _wait(
        pilot,
        app,
        lambda: isinstance(app.screen, LogScreen)
        and app.screen._current_source == "7:training:0"
        and len(app.screen.query("#sources ListItem")) == 2,
    )
    assert ok, "log screen / initial tail did not start"
    await _settled_pane(pilot, app.screen)
    return app.screen


def _live_page(screen, lines):
    """Hand ``screen`` one page of its live tail, as the tail's worker does."""
    screen._on_page(screen._tail_gen, "live", [{"_raw": line} for line in lines], list(lines))


def test_rewrap_and_refilter_run_off_the_ui_thread_and_keep_lines_that_come_meanwhile(monkeypatch):
    async def run():
        started, release = threading.Event(), threading.Event()
        original = LogScreen._wrap_pairs.__func__

        def slow(cls, *args):
            started.set()
            assert release.wait(timeout=5)
            return original(cls, *args)

        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            s = await _open_live_tail(app, pilot)
            _live_page(s, [f"step-{i:02d} loss=0.5 " + "x" * 40 for i in range(30)])
            await _settled_pane(pilot, s)
            monkeypatch.setattr(LogScreen, "_wrap_pairs", classmethod(slow))
            rows = list(s._logview.lines)
            s._content_width = lambda: 30
            s._rewrap()  # returns at once: the lines are wrapped by a worker
            assert await _pause_until(pilot, started.is_set)
            assert s._repaint_job is not None and list(s._logview.lines) == rows
            _live_page(s, ["came during the repaint"])  # shown at once, and kept after it
            assert s._shown_lines[-1] == "came during the repaint" and s._logview.lines[:len(rows)] == rows
            release.set()
            assert await _pause_until(pilot, lambda: s._repaint_job is None and s._wrap_width == 30)
            assert s._shown_lines[-1] == "came during the repaint"
            assert s._shown_lines.count("came during the repaint") == 1
            assert sum(s._shown_rows) == len(s._logview.lines)
            assert all(len(row) <= 30 for row in s._logview.lines)
            # A filter repaints off the UI thread too, from the lines the tail received.
            started.clear()
            release.clear()
            s.query_one("#logfilter", Input).value = "step-07"
            assert await _pause_until(pilot, started.is_set)
            assert any("step-03" in line for line in s._shown_lines)  # the old lines, until it lands
            release.set()
            assert await _pause_until(pilot, lambda: s._repaint_job is None
                                      and not any("step-03" in line for line in s._shown_lines))
            assert [line for line in s._shown_lines if "step-" in line] == ["step-07 loss=0.5 " + "x" * 40]
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_drag_resize_wraps_the_log_once_for_the_width_it_ends_on(monkeypatch):
    async def run():
        widths = []
        original = LogScreen._wrap_pairs.__func__

        def counted(cls, pairs, query, level, width, *rest):
            widths.append(width)
            return original(cls, pairs, query, level, width, *rest)

        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            s = await _open_live_tail(app, pilot)
            _live_page(s, [f"line {i} " + "y" * 60 for i in range(20)])
            await _settled_pane(pilot, s)
            monkeypatch.setattr(LogScreen, "_wrap_pairs", classmethod(counted))
            for width in (50, 45, 40, 35):
                s._content_width = lambda width=width: width
                LogScreen.on_resize(s, None)  # as Textual calls it, for each size of a drag
            assert await _pause_until(pilot, lambda: s._wrap_width == 35 and s._repaint_job is None)
            assert widths == [35]
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_log_line_that_reads_like_a_note_is_still_a_log_line():
    from cortex_training.tui.log_style import NOTE_KIND

    async def run():
        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            s = await _open_live_tail(app, pilot)
            logs = s._shown_logs
            _live_page(s, ["— live —"])  # a log line with the text of the TUI's own note
            assert s._shown_logs == logs + 1
            assert s._logview.lines[-1] == "— live —"  # (no sub-job tag: the pane shows one sub-job)
            assert s._logview.row_kinds[-1] != NOTE_KIND and s._shown_kinds[-1] != NOTE_KIND

            def italic(y):  # the row as the pane draws it: in the notes' italics?
                strip = s._logview._render_line_strip(y, s._logview.rich_style)
                return any(segment.style is not None and segment.style.italic for segment in strip)

            assert not italic(len(s._logview.lines) - 1)
            s._write_line("— live —")  # the TUI's own note
            assert s._shown_logs == logs + 1
            assert s._logview.lines[-1] == "— live —" and s._logview.row_kinds[-1] == NOTE_KIND
            assert italic(len(s._logview.lines) - 1)
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_pane_shows_the_time_with_each_day_on_a_rule_and_save_keeps_the_lines():
    from cortex_training.tui.format import day_rule
    from cortex_training.tui.format import display_line
    from cortex_training.tui.log_style import DAY_KIND

    lines = ["2026-10-17T12:00:00Z INFO step 1", "2026-10-17T12:00:05Z INFO step 2", "2026-10-19T12:00:00Z INFO step 3"]

    async def run():
        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            s = await _open_live_tail(app, pilot)
            _live_page(s, lines)
            shown = [display_line(line) for line in lines]
            rows = list(s._logview.lines[-5:])
            width = s._content_width()
            assert rows == [day_rule(shown[0][:10], width), shown[0][11:], shown[1][11:],
                            day_rule(shown[2][:10], width), shown[2][11:]]
            assert s._logview.row_kinds[-5] == DAY_KIND and s._logview.row_kinds[-2] == DAY_KIND
            assert s._shown_lines[-3:] == lines  # the buffer (and save, and copy) keep the lines as they came
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_highlight_across_a_day_rule_and_a_wrapped_line_copies_the_lines_as_logged(monkeypatch):
    from textual import events
    from textual.geometry import Offset
    from textual.selection import Selection

    from cortex_training.tui.clipboard import screen_dump
    from cortex_training.tui.format import is_day_rule

    lines = ["2026-10-17T05:00:00Z INFO step 1", "2026-10-17T05:00:01Z INFO step 2 " + "loss=0.5 " * 12]
    copied: list = []

    async def run():
        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        monkeypatch.setattr(app, "copy_out", lambda text, what: copied.append(text))
        async with app.run_test(size=(100, 30)) as pilot:
            s = await _open_live_tail(app, pilot)
            _live_page(s, lines)
            await pilot.pause()
            lv = s._logview
            rows = list(lv.lines)
            top = len(rows) - sum(s._shown_rows[-2:])  # the first of the two lines' rows: its day's rule
            assert is_day_rule(rows[top]) and s._shown_rows[-1] > 1  # (the second line wraps)
            # A drag with the mouse from inside the rule to past the end of the wrapped line, then
            # ctrl+c: both lines whole, as logged (whole stamps, no rule, no indent).
            region = lv.content_region
            assert lv.scroll_offset.y == 0 and len(rows) <= region.height  # (every row in view)

            async def mouse(cls, x, y):  # as a terminal sends it: the button held through a drag
                app.post_message(cls(None, x, y, 0, 0, 1, False, False, False, screen_x=x, screen_y=y))
                await pilot.pause()

            await mouse(events.MouseDown, region.x + 4, region.y + top)
            await mouse(events.MouseMove, region.right - 1, region.y + len(rows) - 1)
            await mouse(events.MouseUp, region.right - 1, region.y + len(rows) - 1)
            await pilot.press("ctrl+c")
            assert await _pause_until(pilot, lambda: copied) and copied[-1] == "\n".join(lines)
            # The same highlight set directly, as the screen keeps it.
            s.selections = {lv: Selection(Offset(4, top), Offset(len(rows[-1]), len(rows) - 1))}
            s.action_copy_text()
            assert copied[-1] == "\n".join(lines)
            # Part of the wrapped line, across its wrap: its text, without the rows' indent.
            s.selections = {lv: Selection(Offset(30, len(rows) - 2), Offset(14, len(rows) - 1))}
            s.action_copy_text()
            part = copied[-1]
            assert part and "         " not in part and part in s._display(lines[1])
            # Copying the whole screen takes the pane's lines as logged too.
            dump = screen_dump(s)
            assert lines[0] in dump and lines[1].rstrip() in dump and "── " not in dump
            # With nothing highlighted, ctrl+c copies the log as s saves it (from the lines the
            # tail cached, as its worker caches them): as the lines came.
            s._cache.append_entries(s._current_source, [{"_raw": line} for line in lines])
            assert s._exported_text(s._current_source, []) == "\n".join(lines)
            s.selections = {}
            count = len(copied)
            s.action_copy_text()
            assert await _pause_until(pilot, lambda: len(copied) > count)
            assert copied[-1] == "\n".join(lines)
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_sub_job_list_names_each_sub_job_by_its_type_as_the_preview_does():
    c = _client()
    c.get_job.return_value = {"status": "RUNNING", "sub_jobs": [
        {"sub_job_id": "7:training:0", "job_type": "training"},
        {"sub_job_id": "7:log_probability:0", "job_type": "LOG_PROBABILITY"}]}

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            assert await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen)
                               and len(app.screen.query("#sources ListItem")) == 2)
            labels = [str(item.query_one(Label).render()) for item in app.screen.query("#sources ListItem")]
            assert "log_prob #0" in labels[1] and "log_probability" not in labels[1]
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_day_rule_comes_back_on_top_as_the_oldest_lines_go(monkeypatch):
    from cortex_training.tui.log_style import DAY_KIND

    monkeypatch.setattr(LogScreen, "_MAX_LINES", 10)

    async def run():
        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            s = await _open_live_tail(app, pilot)
            for page in range(3):
                _live_page(s, [f"2026-10-17T05:00:{page * 10 + i:02d}Z INFO line {page}-{i}" for i in range(10)])
            lv = s._logview
            assert len(s._shown_lines) == 10 and s._shown_lines[0].endswith("line 2-0")
            assert lv.row_kinds[0] == DAY_KIND and lv.row_kinds.count(DAY_KIND) == 1
            assert len(lv.lines) == sum(s._shown_rows) == s._rows_total
            await _settle(app, pilot)

    asyncio.run(run())


def test_e_jumps_to_the_errors_and_the_bar_says_where_the_reader_is():
    def bar(s):
        return str(s.query_one("#log-live", Static).render()), str(s.query_one("#log-chips", Static).render())

    async def run():
        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test(size=(120, 30)) as pilot:
            s = await _open_live_tail(app, pilot)
            _live_page(s, ["2026-10-18T12:00:00Z ERROR first failure", *(f"line {i}" for i in range(60)),
                           "2026-10-18T12:01:00Z ERROR second failure", "line after"])
            await pilot.pause()
            assert bar(s)[0] == "● LIVE · following"
            await pilot.press("e")  # from the end of the log: the newest error
            await pilot.pause()
            marked = s._line_at_row(s._logview.mark[0])
            assert s._shown_lines[marked].endswith("second failure") and s._jump == ("error", 2, 2)
            assert s._paused and "PAUSED" in app.sub_title and bar(s)[1].strip() == "error 2/2"
            await pilot.press("e")  # round past the end, to the first
            await pilot.pause()
            assert s._shown_lines[s._line_at_row(s._logview.mark[0])].endswith("first failure")
            assert s._jump == ("error", 1, 2)
            # A re-wrap (a resize) while paused keeps the mark and the chip.
            width = s._content_width()
            s._content_width = lambda: width - 20
            s._rewrap()
            assert await _pause_until(pilot, lambda: s._repaint_job is None and s._wrap_width == width - 20)
            assert s._shown_lines[s._line_at_row(s._logview.mark[0])].endswith("first failure")
            assert s._jump == ("error", 1, 2) and bar(s)[1].strip() == "error 1/2"
            # An error that comes while paused counts at once.
            _live_page(s, ["2026-10-18T12:02:00Z ERROR third failure"])
            await pilot.pause()
            assert s._jump == ("error", 1, 3) and bar(s)[1].strip() == "error 1/3"
            await pilot.press("N")  # n / N stop at warnings too: problems
            await pilot.pause()
            assert s._jump == ("problem", 3, 3) and bar(s)[1].strip() == "problem 3/3"
            _live_page(s, ["line new"])
            await pilot.pause()
            assert bar(s)[0] == "‖ PAUSED · 2 new lines · p resumes"  # (the third failure, and this)
            await pilot.press("p")
            await pilot.pause()
            assert not s._paused and s._logview.mark is None and s._jump is None
            assert bar(s)[0] == "● LIVE · following"
            # Scrolled up, not paused (the subtitle says no PAUSED): the bar says so, and p follows again.
            s._logview.scroll_home(animate=False, immediate=True)
            await pilot.pause()
            assert bar(s)[0] == "↑ SCROLLED UP · p follows" and "PAUSED" not in app.sub_title
            await pilot.press("p")
            await pilot.pause()
            assert not s._paused and bar(s)[0] == "● LIVE · following"
            await _settle(app, pilot)

    asyncio.run(run())


def test_the_scrollback_count_follows_the_lines_as_they_come_and_go(monkeypatch):
    monkeypatch.setattr(LogScreen, "_MAX_LINES", 10)

    async def run():
        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            s = await _open_live_tail(app, pilot)

            def logs():
                return sum(not isinstance(line, LogScreen._Note) for line in s._shown_lines)

            _live_page(s, [f"line {i}" for i in range(6)])
            s._write_line("— a note —")
            _live_page(s, [f"line {i}" for i in range(6, 12)])  # the oldest go past _MAX_LINES
            assert s._pane_truncated and s._shown_logs == logs() == 9
            assert s._scrollback_note().startswith("last 9 lines · ")
            s._content_width = lambda: 25
            s._rewrap()
            assert await _pause_until(pilot, lambda: s._repaint_job is None and s._wrap_width == 25)
            assert s._shown_logs == logs() == 9 and s._scrollback_note().startswith("last 9 lines · ")
            s._clear_pane()
            assert s._shown_logs == 0
            await _settle(app, pilot)

    asyncio.run(run())


def test_counting_a_saved_console_streams_it_and_counts_its_lines_as_before(tmp_path):
    block = 1 << 20  # what _count_lines reads at a time
    payloads = [
        b"",
        b"one",
        b"one\n",
        b"one\r\ntwo\rthree\n\nfive",
        "vt\x0bff\x0cfs\x1cgs\x1drs\x1enel\x85ls\N{LINE SEPARATOR}ps\N{PARAGRAPH SEPARATOR}end".encode(),
        b"bad \xff\xfe bytes\n\x85 alone",  # not UTF-8: replaced, as the pane decodes it
        b"x" * (block - 1) + b"\r\nlast",  # one \r\n across the first block's end
        b"x" * (block - 1) + b"\r",
        b"x" * (block - 1) + "\N{LINE SEPARATOR}after".encode(),  # a break of three bytes across it
    ]
    for n, payload in enumerate(payloads):
        path = tmp_path / f"console-{n}.log"
        path.write_bytes(payload)
        before = len(payload.decode("utf-8", errors="replace").splitlines())  # what the save counted
        assert LogScreen._count_lines(str(path)) == before, payload[-16:]


def test_save_writes_the_sub_job_shown_when_s_was_pressed(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("head:server", [{"_raw": "alpha"}, {"_raw": "beta"}])

    async def run():
        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            s = await _open_live_tail(app, pilot)
            s._current_source = "head:server"
            s.action_save_log()
            s._current_source = "7:sampling:0"  # the view moves on before the worker runs
            saved = tmp_path / "cortex-training-7-head-server.log"
            assert await _pause_until(pilot, lambda: s._export_note is not None)
            assert saved.read_text(encoding="utf-8") == "alpha\nbeta\n"
            assert s._export_note == f"saved 2 lines to {saved}"
            await _settle(app, pilot)

    asyncio.run(run())


def _write_stdout(output_dir, sub_job_id, text):
    path = Path(output_dir) / sub_job_id / "stdout.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_gpu_metrics(output_dir, sub_job_id, *, gpu_count=2):
    path = Path(output_dir) / sub_job_id / "gpu.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # Recent samples: the job is RUNNING, and a running job's samples are fresh.
    newest = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=30)
    for gpu in range(gpu_count):
        rows.append(
            {
                "record_version": 2,
                "sub_job_id": sub_job_id,
                "sample_ts": (newest + timedelta(seconds=gpu)).strftime("%Y-%m-%dT%H:%M:%S") + ".123456789Z",
                "worker_num": gpu,
                "local_gpu_index": 0,
                "global_gpu_index": gpu,
                "gpu_util_pct": float(gpu * 80),
                "fb_used_bytes": float((gpu + 1) * 1024**3),
                "fb_free_bytes": float(8 * 1024**3),
                "sample_interval_s": 5.0,
            }
        )
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    path.with_name(".gpu.jsonl.manifest.json").write_text(
        json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}),
        encoding="utf-8",
    )
    return path


def _terminal_job():
    return {
        "status": "FAILED",
        "sub_jobs": [
            {"sub_job_id": "7:training:0", "job_type": "training"},
            {"sub_job_id": "7:sampling:0", "job_type": "sampling"},
        ],
    }


async def _run_terminal_shows_saved_file():
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        _write_stdout(output_dir, "7:training:0", "saved line\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "saved line" in app.screen._shown_lines)
        assert ok, "saved console did not replace the pane"
        c.tail_logs.assert_not_called()
        await _settle(app, pilot)


def test_terminal_open_shows_downloaded_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    asyncio.run(_run_terminal_shows_saved_file())


async def _run_keeps_cache_when_download_fails():
    c = _client()
    c.get_job.return_value = _terminal_job()
    c.download_stdout_logs.side_effect = RuntimeError("unavailable")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: "cached line" in app.screen._shown_lines and not app.screen._stdout_inflight,
        )
        assert ok, "cache line missing after a failed download"
        assert "No log output is available." not in app.screen._shown_lines
        downloads = c.download_stdout_logs.call_count
        app.screen._refilter()
        for _ in range(5):
            await pilot.pause()
        assert c.download_stdout_logs.call_count == downloads
        await _settle(app, pilot)


def test_failed_download_keeps_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_keeps_cache_when_download_fails())


async def _run_empty_file_keeps_cache():
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        _write_stdout(output_dir, "7:training:0", "")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: app.screen._stdout_ready and "cached line" in app.screen._shown_lines,
        )
        assert ok, "empty saved file did not leave the cache on screen"
        await _settle(app, pilot)


def test_empty_saved_file_keeps_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_empty_file_keeps_cache())


async def _run_hidden_filter_shows_no_note(release, started):
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "cached line" in app.screen._shown_lines and started.is_set())
        assert ok, "cache was not shown before the filter"
        app.screen._filter = "no-such-line"
        release.set()
        ok = await _wait(pilot, app, lambda: app.screen._stdout_ready)
        assert ok, "download did not finish"
        for _ in range(5):
            await pilot.pause()
        assert "cached line" not in app.screen._shown_lines
        assert "No log output is available." not in app.screen._shown_lines
        await _settle(app, pilot)


def test_filter_that_hides_cache_shows_no_empty_note(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    release = threading.Event()
    try:
        asyncio.run(_run_hidden_filter_shows_no_note(release, threading.Event()))
    finally:
        release.set()


async def _run_running_tail_error_does_not_download():
    c = _client()
    c.tail_logs.side_effect = RuntimeError("zone down")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: any(line.startswith("[error] tail") for line in app.screen._shown_lines))
        assert ok, "running-job tail error was not shown"
        c.download_stdout_logs.assert_not_called()
        await _settle(app, pilot)


def test_running_tail_error_does_not_download():
    asyncio.run(_run_running_tail_error_does_not_download())


async def _run_tail_error_after_finish_shows_file():
    c = _client()
    calls = {"get": 0, "tail": 0}

    def get_job(_job_id):
        calls["get"] += 1
        if calls["get"] == 1:
            return {
                "status": "RUNNING",
                "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
            }
        return {"status": "FAILED", "reason": "cancelled", "sub_jobs": []}

    def tail_logs(*_args, **_kwargs):
        calls["tail"] += 1
        raise RuntimeError("zone gone")

    def download(job_id, output_dir, **kwargs):
        _write_stdout(output_dir, "7:training:0", "after finish\n")
        return []

    c.get_job.side_effect = get_job
    c.tail_logs.side_effect = tail_logs
    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "after finish" in app.screen._shown_lines)
        assert ok, "terminal tail error did not show the saved console"
        assert app.screen._job_status == "FAILED"
        tails = calls["tail"]
        app.screen._refilter()
        await _wait(pilot, app, lambda: "after finish" in app.screen._shown_lines)
        assert calls["tail"] == tails
        await _settle(app, pilot)


def test_tail_error_after_terminal_status_shows_file_once():
    asyncio.run(_run_tail_error_after_finish_shows_file())


def test_a_slow_job_read_never_brings_an_ended_job_back():
    started, release = threading.Event(), threading.Event()
    c = _client()
    reads = {"n": 0}

    def get_job(_job_id):
        reads["n"] += 1
        if reads["n"] == 1:  # the view's first read, still retrying a slow service
            started.set()
            assert release.wait(timeout=5)
            return {"status": "RUNNING", "sub_jobs": [
                {"sub_job_id": "7:training:0", "job_type": "training", "status": "RUNNING"}]}
        return {"status": "FAILED", "sub_jobs": []}

    c.get_job.side_effect = get_job

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and started.is_set())
            assert ok
            s = app.screen
            # A read sent later (the status check) answers first: the job has failed.
            s._apply_status({"status": "FAILED", "reason": "sub_job_failed", "sub_jobs": [
                {"sub_job_id": "7:training:0", "job_type": "training", "status": "FAILED"}]}, s._next_job_read())
            assert s._job_status == "FAILED" and app.sub_title.startswith("FAILED")
            release.set()  # ... and then the first read answers, with the job still running
            ok = await _wait(pilot, app, lambda: s._current_source == "7:training:0")
            assert ok
            for _ in range(10):
                await pilot.pause()
            assert s._job_status == "FAILED" and app.sub_title.startswith("FAILED")
            assert s._sub_job_status == {"7:training:0": "FAILED"}
            assert not s._tail_live
            c.tail_logs.assert_not_called()  # never a tail of the dead job: its saved console instead
            # However it is numbered, an answer that has the ended job live is an older copy.
            assert not s._take_job({"status": "RUNNING", "sub_jobs": []}, s._next_job_read())
            assert s._job_status == "FAILED"
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_a_bug_in_showing_a_page_sends_no_job_read(tmp_path, monkeypatch):
    errors = tmp_path / "errors.log"  # where the fixture puts the error log
    c = _client()
    c.tail_logs.side_effect = lambda *a, **k: {"entries": [{"_raw": "a line"}], "next_cursor": "C1", "eof": False}
    on_page = LogScreen._on_page

    def broken(self, gen, kind, entries, lines):
        if kind == "live" and entries:
            raise TypeError("a bug in drawing the page")
        return on_page(self, gen, kind, entries, lines)

    monkeypatch.setattr(LogScreen, "_on_page", broken)

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen)
                             and any(line.startswith("[error] tail") for line in app.screen._shown_lines))
            assert ok, "the tail did not stop"
            s = app.screen
            # Said as before, but the job isn't read again to tell what a failed request meant.
            assert s._shown_lines[-1].startswith(
                "[error] tail 7:training:0 failed: TypeError: a bug in drawing the page")
            assert s._tail_error.startswith("tail stopped: TypeError: a bug in drawing the page")
            assert not s._tail_live
            for _ in range(10):
                await pilot.pause()
            assert c.get_job.call_count == 1  # the view's own first read only
            logged = errors.read_text(encoding="utf-8")
            assert "a bug in the TUI while showing its log" in logged and "TypeError: a bug in drawing" in logged
            await _settle(app, pilot)

    asyncio.run(run())


async def _run_terminal_download_failure_keeps_cache():
    c = _client()
    calls = {"get": 0}

    def get_job(_job_id):
        calls["get"] += 1
        if calls["get"] == 1:
            return {
                "status": "RUNNING",
                "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"}],
            }
        return {"status": "FAILED", "sub_jobs": []}

    c.get_job.side_effect = get_job
    c.tail_logs.side_effect = RuntimeError("zone gone")
    c.download_stdout_logs.side_effect = RuntimeError("unavailable")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: c.download_stdout_logs.called and "cached line" in app.screen._shown_lines,
        )
        assert ok, "failed saved-console download did not leave the cache"
        for _ in range(5):
            await pilot.pause()
        assert not any(line.startswith("[error] tail") for line in app.screen._shown_lines)
        await _settle(app, pilot)


def test_terminal_download_failure_keeps_cache_without_tail_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    asyncio.run(_run_terminal_download_failure_keeps_cache())


async def _run_healthy_tail_does_not_download():
    c = _client()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._current_source == "7:training:0")
        assert ok, "live tail did not start"
        await _settle(app, pilot)
        c.download_stdout_logs.assert_not_called()


def test_healthy_running_tail_does_not_download():
    asyncio.run(_run_healthy_tail_does_not_download())


async def _run_filter_during_download_once(release, started, calls):
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        calls.append(output_dir)
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "from file\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, started.is_set)
        assert ok, "download did not start"
        app.screen._refilter()
        assert len(calls) == 1
        ok = await _wait(pilot, app, lambda: "cached line" in app.screen._shown_lines)
        assert ok, "cache replay did not paint during the download"
        release.set()
        ok = await _wait(pilot, app, lambda: "from file" in app.screen._shown_lines)
        assert ok, "finished download did not replace the current source"
        assert "cached line" not in app.screen._shown_lines
        for _ in range(5):
            await pilot.pause()
        assert "cached line" not in app.screen._shown_lines
        assert len(calls) == 1
        await _settle(app, pilot)


def test_filter_during_download_does_not_start_another(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    release = threading.Event()
    started = threading.Event()
    try:
        asyncio.run(_run_filter_during_download_once(release, started, []))
    finally:
        release.set()


async def _run_switch_source_does_not_download_again():
    c = _client()
    c.get_job.return_value = _terminal_job()
    calls = []

    def download(job_id, output_dir, **kwargs):
        calls.append(1)
        _write_stdout(output_dir, "7:training:0", "training line\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._stdout_ready)
        assert ok, "first download did not finish"
        app.screen._start_tail("7:sampling:0")
        await _wait(pilot, app, lambda: "sampling cache" in app.screen._shown_lines)
        assert len(calls) == 1
        await _settle(app, pilot)


def test_source_switch_after_download_does_not_download_again(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:sampling:0", [{"_raw": "sampling cache"}])
    asyncio.run(_run_switch_source_does_not_download_again())


async def _run_missing_sibling_shows_note():
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        _write_stdout(output_dir, "7:training:0", "training line\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: app.screen._stdout_ready)
        assert ok, "first download did not finish"
        app.screen._start_tail("7:sampling:0")
        ok = await _wait(pilot, app, lambda: "No log output is available." in app.screen._shown_lines)
        assert ok, "empty sibling did not show the finished-view note"
        assert "training line" not in app.screen._shown_lines
        assert c.download_stdout_logs.call_count == 1
        await _settle(app, pilot)


def test_missing_sibling_after_download_shows_empty_note():
    asyncio.run(_run_missing_sibling_shows_note())


async def _run_save_exports_file(tmp_path):
    c = _client()
    c.get_job.return_value = _terminal_job()

    payload = b"file body\r\nsecond\r"

    def download(job_id, output_dir, **kwargs):
        path = _write_stdout(output_dir, "7:training:0", "")
        path.write_bytes(payload)
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "file body" in app.screen._shown_lines)
        assert ok, "saved console was not shown"
        app.screen._filter = "file"
        copied: list[str] = []
        app.copy_to_clipboard = copied.append
        app.screen.action_copy_text()  # nothing highlighted: the whole log
        ok = await _wait(pilot, app, lambda: bool(copied))
        assert ok, "copy did not export the saved console"
        # Copied without control characters (pasted elsewhere, they would act there); s keeps the bytes.
        assert copied == ["file body\nsecond"]
        home = tmp_path / "home"
        home.mkdir()
        import cortex_training.tui.widgets as app_module

        original = app_module.os.path.expanduser
        app_module.os.path.expanduser = lambda path: str(home / os.path.basename(path))
        try:
            app.screen.action_save_log()
            saved = home / "cortex-training-7-training-0.log"  # <job8>-<type>-<index>, the job id once
            ok = await _wait(pilot, app, saved.is_file)
            assert ok, "save did not write the console file"
            assert saved.read_bytes() == b"file body\r\nsecond\r"
        finally:
            app_module.os.path.expanduser = original

        # A whole-log copy reads only the bounded tail, including when one line
        # alone is larger than the clipboard limit.
        app.screen._stdout_path("7:training:0").write_bytes(b"x" * (app.screen._COPY_MAX_BYTES + 100))
        copied.clear()
        app.screen.action_copy_text()
        assert await _wait(pilot, app, lambda: bool(copied))
        assert copied[0].startswith("…")
        assert len(copied[0].encode("utf-8")) <= app.screen._COPY_MAX_BYTES
        await _settle(app, pilot)


def test_save_exports_saved_console_not_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached only"}])
    asyncio.run(_run_save_exports_file(tmp_path))


async def _run_pause_holds_replacement(release, started):
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "held line\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "cached line" in app.screen._shown_lines and started.is_set())
        assert ok, "cache was not on screen before the file arrived"
        app.screen.action_toggle_pause()
        release.set()
        ok = await _wait(pilot, app, lambda: app.screen._held is not None)
        assert ok, "paused view did not hold the saved console"
        assert "held line" not in app.screen._shown_lines
        app.screen.action_toggle_pause()
        ok = await _wait(pilot, app, lambda: "held line" in app.screen._shown_lines)
        assert ok, "unpause did not show the saved console"
        app.screen.action_toggle_pause()
        app.screen._held = (["stale line"], False)
        app.screen._start_tail("7:training:0")
        assert app.screen._held is None
        assert "stale line" not in app.screen._shown_lines
        ok = await _wait(pilot, app, lambda: "held line" in app.screen._shown_lines)
        assert ok, "re-tail did not repaint the saved console"
        assert "stale line" not in app.screen._shown_lines
        await _settle(app, pilot)


def test_pause_holds_saved_console_until_unpause(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path))
    from cortex_training.tui.log_cache import LogCache

    LogCache("7").append_entries("7:training:0", [{"_raw": "cached line"}])
    release = threading.Event()
    try:
        asyncio.run(_run_pause_holds_replacement(release, threading.Event()))
    finally:
        release.set()


async def _run_leave_during_download(release, started, dirs):
    c = _client()
    c.get_job.return_value = _terminal_job()

    def download(job_id, output_dir, **kwargs):
        dirs.append(output_dir)
        started.set()
        assert release.wait(timeout=5)
        _write_stdout(output_dir, "7:training:0", "too late\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, started.is_set)
        assert ok, "download did not start"
        app.pop_screen()
        await pilot.pause()
        release.set()
        ok = await _wait(pilot, app, lambda: dirs and not Path(dirs[0]).exists())
        assert ok, "temp dir was not removed after leaving"
        # Its owner lock goes with it (release_stage): left behind, the next sweep has a file to clear.
        assert not os.path.lexists(f"{dirs[0]}.owner"), "the stage's owner lock outlived it"
        assert not any("too late" in line for line in getattr(app.screen, "_shown_lines", []))
        await _settle(app, pilot)


def test_leave_during_download_removes_temp_dir():
    release = threading.Event()
    try:
        asyncio.run(_run_leave_during_download(release, threading.Event(), []))
    finally:
        release.set()


async def _run_failure_then_retry():
    c = _client()
    c.get_job.return_value = _terminal_job()
    calls = []

    def download(job_id, output_dir, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            exc = RuntimeError("stage unavailable")
            raise exc
        _write_stdout(output_dir, "7:training:0", "after retry\n")
        return []

    c.download_stdout_logs.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: any("saved console couldn't be downloaded" in line for line in app.screen._shown_lines),
        )
        assert ok, "download failure was not shown in the pane"
        app.screen._refilter()
        for _ in range(5):
            await pilot.pause()
        assert len(calls) == 1, "a filter change retried the failed download"
        app.screen.action_refresh_sources()
        ok = await _wait(pilot, app, lambda: "after retry" in app.screen._shown_lines)
        assert ok, "r did not retry the saved-console download"
        assert len(calls) == 2
        await _settle(app, pilot)


def test_download_failure_shows_error_and_r_retries(monkeypatch, tmp_path):
    import cortex_training.tui.widgets as app_module

    monkeypatch.setattr(app_module, "_ERROR_LOG", str(tmp_path / "errors.log"))
    asyncio.run(_run_failure_then_retry())


async def _run_saved_console_read_off_ui_thread(seen):
    real = LogScreen._scan_saved

    def spy(self, path, query, *rest):
        seen.append((threading.current_thread() is threading.main_thread(), query))
        return real(self, path, query, *rest)

    LogScreen._scan_saved = spy
    try:
        c = _client()
        c.get_job.return_value = _terminal_job()

        def download(job_id, output_dir, **kwargs):
            _write_stdout(output_dir, "7:training:0", "step loss\r" * 20_000 + "step 2 done\nbeta\n")
            return []

        c.download_stdout_logs.side_effect = download
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test() as pilot:
            ok = await _wait(pilot, app, lambda: "beta" in app.screen._shown_lines)
            assert ok, "saved console was not shown"
            app.screen._filter = "done"
            app.screen._refilter()
            ok = await _wait(
                pilot,
                app,
                lambda: "step 2 done" in app.screen._shown_lines
                and not any("loss" in line or "beta" in line for line in app.screen._shown_lines),
            )
            assert ok, "filter was not applied to the saved console"
            await _settle(app, pilot)
    finally:
        LogScreen._scan_saved = real


def test_saved_console_is_read_off_the_ui_thread():
    seen = []
    asyncio.run(_run_saved_console_read_off_ui_thread(seen))
    assert seen and not any(on_main for on_main, _ in seen)
    assert "done" in [query for _, query in seen]


async def _run_progress_status_and_cancel(state):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        # A real signature: unknown keyword arguments would fail the call.
        def download_stdout_logs(self, job_id, output_dir, *, resume=False, on_chunk=None, sub_job_ids=None):
            if sub_job_ids is not None:
                return []
            state["resume"] = resume
            try:
                for i in range(1, 10000):
                    on_chunk("7:training:0", i, 10000, i * 1024)
                    if i == 1:
                        state["started"].set()
                    time.sleep(0.005)
            except BaseException as exc:  # noqa: BLE001 - the TUI's cancel signal
                state["raised"] = type(exc)
                raise
            return []

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: re.search(r"chunk [\d,]+ of 10,000", app.screen._download_note or ""))
        assert ok, "download progress did not reach the status line"
        app.pop_screen()
        for _ in range(5):
            await pilot.pause()
        await _settle(app, pilot)
    for _ in range(200):
        if "raised" in state:
            break
        time.sleep(0.01)


def test_progress_updates_status_and_leaving_cancels_download(monkeypatch, tmp_path):
    import cortex_training.tui.widgets as app_module

    monkeypatch.setattr(app_module, "_ERROR_LOG", str(tmp_path / "errors.log"))
    state = {"started": threading.Event()}
    asyncio.run(_run_progress_status_and_cancel(state))
    assert state["resume"] is True
    assert issubclass(state.get("raised"), DownloadCancelled), "download was not cancelled"
    assert not (tmp_path / "errors.log").exists(), "a cancel was logged as a failure"


async def _run_selected_sub_job_first(release, calls):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        def download_stdout_logs(
            self, job_id, output_dir, *, resume=False, on_chunk=None, sub_job_ids=None, **_kwargs
        ):
            calls.append(None if sub_job_ids is None else list(sub_job_ids))
            if sub_job_ids is not None:
                for sid in sub_job_ids:
                    _write_stdout(output_dir, sid, f"{sid} saved\n")
                return []
            assert release.wait(timeout=5)
            _write_stdout(output_dir, "7:sampling:0", "7:sampling:0 saved\n")
            return []

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "7:training:0 saved" in app.screen._shown_lines)
        assert ok, "selected sub-job was not shown before the rest finished"
        ok = await _wait(pilot, app, lambda: len(calls) == 2)
        assert ok, "the rest of the job was not downloaded"
        assert calls == [["7:training:0"], None]
        assert app.screen._stdout_inflight
        release.set()
        ok = await _wait(pilot, app, lambda: app.screen._stdout_ready)
        assert ok, "whole-job download did not finish"
        app.screen._start_tail("7:sampling:0")
        ok = await _wait(pilot, app, lambda: "7:sampling:0 saved" in app.screen._shown_lines)
        assert ok, "sibling sub-job was not shown after the full download"
        await _settle(app, pilot)


def test_selected_sub_job_downloads_and_paints_first():
    release = threading.Event()
    calls = []
    try:
        asyncio.run(_run_selected_sub_job_first(release, calls))
    finally:
        release.set()


async def _run_reopen_while_download_blocks(state):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        # The first download blocks (like a retry backoff that still polls the
        # callback) until the TUI cancels it; later ones finish at once.
        def download_stdout_logs(self, job_id, output_dir, *, resume=False, on_chunk=None, sub_job_ids=None):
            state["calls"] += 1
            if state["calls"] > 1:
                if sub_job_ids:  # the reopened view's sub-job on screen; the rest adds nothing
                    _write_stdout(output_dir, "7:training:0", "second view\n")
                return []
            state["started"].set()
            try:
                while True:
                    on_chunk("7:training:0", 0, 10, 0)
                    time.sleep(0.02)
            except BaseException as exc:  # noqa: BLE001 - the TUI's cancel signal
                state["raised"] = type(exc)
                raise

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, state["started"].is_set)
        assert ok, "first download did not start"
        app.pop_screen()
        await pilot.pause()
        app.push_screen(LogScreen(c, "7", poll_interval=0.01))
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and "second view" in app.screen._shown_lines,
            tries=300,
        )
        assert ok, "reopened view stayed waiting on the cancelled download"
        await _settle(app, pilot)


def test_reopen_does_not_wait_forever_on_a_blocked_download(monkeypatch, tmp_path):
    import cortex_training.tui.widgets as app_module

    monkeypatch.setattr(app_module, "_ERROR_LOG", str(tmp_path / "errors.log"))
    state = {"calls": 0, "started": threading.Event()}
    asyncio.run(_run_reopen_while_download_blocks(state))
    assert issubclass(state["raised"], DownloadCancelled), "download was not cancelled"
    assert state["calls"] == 3  # the blocked one, then the reopened view's sub-job and the rest
    assert not (tmp_path / "errors.log").exists()


async def _run_quit_during_download(state):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        def download_stdout_logs(self, job_id, output_dir, *, resume=False, on_chunk=None, sub_job_ids=None):
            state["started"].set()
            try:
                while True:
                    on_chunk("7:training:0", 0, 10, 0)
                    time.sleep(0.02)
            finally:
                state["returned"].set()

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, state["started"].is_set)
        assert ok, "download did not start"
        await app.action_quit()
        await pilot.pause()


def test_quit_during_download_stops_it():
    state = {"started": threading.Event(), "returned": threading.Event()}
    asyncio.run(_run_quit_during_download(state))
    assert state["returned"].wait(timeout=5), "download kept running after quit"


async def _run_job_detail_tabs():
    c = _client()
    c.download_metrics.return_value = []
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        ok = await _wait(
            pilot,
            app,
            lambda: isinstance(app.screen, LogScreen) and app.screen._current_source == "7:training:0",
        )
        assert ok
        screen = app.screen
        tabs = screen.query_one("#job-tabs", TabbedContent)
        # The Resources tab is composed (its panel and pane), with nothing in it until it is shown.
        assert screen.query_one("#resource-list") is not None and screen.query_one("#resource-pane") is not None
        assert screen._resource_entries == []

        await pilot.press("g")
        ok = await _wait(pilot, app, lambda: tabs.active == "resource" and c.download_metrics.call_count == 1)
        assert ok
        assert screen.query_one("#footer").display  # the footer stays on every tab
        await pilot.press("5")  # a digit picks no window (w does)
        await pilot.pause()
        assert tabs.active == "resource" and screen._resource_range != "all"

        # Esc on a tab other than Logs goes back, even with a log filter set.
        await pilot.press("l")
        ok = await _wait(pilot, app, lambda: tabs.active == "logs")
        assert ok
        screen.query_one("#logfilter", Input).value = "x"
        await pilot.press("g")
        ok = await _wait(pilot, app, lambda: tabs.active == "resource")
        assert ok
        await pilot.press("escape")
        ok = await _wait(pilot, app, lambda: app.screen is not screen)
        assert ok
        await _settle(app, pilot)


def test_job_detail_tabs_load_lazily():
    asyncio.run(_run_job_detail_tabs())


def test_job_list_refreshes_on_its_own(monkeypatch):
    monkeypatch.setattr(JobListScreen, "_REFRESH_SECONDS", 0.05)
    c = _client()

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test() as pilot:
            for _ in range(100):
                await pilot.pause(0.05)
                if c.list_jobs.call_count >= 3:
                    break
            assert c.list_jobs.call_count >= 3, "the job list never refreshed on its own"
            await _settle(app, pilot)

    asyncio.run(run())


def _notes(app) -> list:
    return [note.message for note in app._notifications]


async def _run_highlighting_copies_nothing_until_ctrl_c():
    from textual import events

    from cortex_training.tui.clipboard import selected_text

    app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
    copied: list[str] = []
    app.copy_to_clipboard = copied.append

    async def mouse(cls, x, button=1):  # as a terminal sends it: the button held through a drag
        app.post_message(cls(None, x, 1, 0, 0, button, False, False, False, screen_x=x, screen_y=1))
        await pilot.pause()

    async with app.run_test() as pilot:
        assert await _wait(pilot, app, lambda: len(getattr(app.screen, "_job_by_item", {})) >= 2)
        for cls, x in ((events.MouseDown, 4), (events.MouseMove, 20), (events.MouseUp, 20)):
            await mouse(cls, x)
        highlighted = selected_text(app.screen)
        for cls in (events.MouseDown, events.MouseUp):
            await mouse(cls, 30, button=3)  # a right-click
        await pilot.pause(0.5)  # long enough for a copy made on mouse-up to land
        assert highlighted and copied == []
        await pilot.press("ctrl+c")
        assert await _wait(pilot, app, lambda: copied) and copied == [highlighted]
        await _settle(app, pilot)


def test_highlighting_copies_nothing_until_ctrl_c():
    asyncio.run(_run_highlighting_copies_nothing_until_ctrl_c())


# ─── text selection in tables ─────────────────────────────────────────────────


async def _demo_list(app, pilot):
    """The fake's job list, drawn whole (not its first pages only): the screen and its table."""
    assert await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen)
                       and len(app.screen._job_by_item) >= 2
                       and app.screen._scope is not None and not app.screen._scope.loading, tries=400)
    await pilot.pause(0.1)
    return app.screen, app.screen.query_one("#jobs", DataTable)


def _pointer(app, pilot):
    """``mouse(kind, x, y, button=1)``: a mouse event as a terminal sends it, the button held
    through a drag and each move's delta from the last event (Textual scrolls a highlight's
    area only for a pointer that moves up or down)."""
    last = {}

    async def mouse(kind, x, y, button=1):
        was_x, was_y = last.get("at", (x, y))
        last["at"] = (x, y)
        app.post_message(kind(None, x, y, x - was_x, y - was_y, button, False, False, False, screen_x=x, screen_y=y))
        await pilot.pause()

    return mouse


async def _drag(mouse, start, end, via=()):
    from textual import events

    await mouse(events.MouseDown, *start)
    for x, y in (*via, end):
        await mouse(events.MouseMove, x, y)
    await mouse(events.MouseUp, *end)


async def _click(mouse, x, y, moved_to=None):
    from textual import events

    await mouse(events.MouseDown, x, y)
    if moved_to is not None:
        await mouse(events.MouseMove, *moved_to)
    await mouse(events.MouseUp, *(moved_to or (x, y)))


def _screen_lines(app) -> list:
    return [strip.text for strip in app.screen._compositor.render_strips()]


def _highlighted_columns(app, y, table) -> list:
    """The columns of screen line ``y`` drawn with the selection's background (these lines
    have a cell for each character)."""
    from textual.filter import monochrome_style

    selection_style = table._selection_style()
    backgrounds = {selection_style.bgcolor, monochrome_style(selection_style).bgcolor}
    columns, x = [], 0
    for segment in app.screen._compositor.render_strips()[y]:
        for _character in segment.text:
            if segment.style is not None and segment.style.bgcolor in backgrounds:
                columns.append(x)
            x += 1
    return columns


def _first_job_line(lines) -> int:
    return next(y for y, line in enumerate(lines) if line[1:3] in ("▸ ", "▾ "))  # after the cell padding


async def _copied(app, pilot, copied):
    await pilot.press("ctrl+c")
    assert await _wait(pilot, app, lambda: copied)
    return copied.pop()


async def _run_a_drag_in_a_job_list_cell_highlights_and_copies_just_its_text():
    from cortex_training.tui.clipboard import selected_text

    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    copied: list[str] = []
    app.copy_to_clipboard = copied.append
    async with app.run_test(size=(130, 40)) as pilot:
        screen, table = await _demo_list(app, pilot)
        mouse = _pointer(app, pilot)
        lines = _screen_lines(app)
        y = _first_job_line(lines)
        cursor = table.cursor_row
        # Inside the job id: just the characters dragged over, drawn highlighted, and no click.
        await _drag(mouse, (7, y), (14, y), via=[(10, y)])
        assert selected_text(screen) == lines[y][7:15]
        assert _highlighted_columns(app, y, table) == list(range(7, 15))
        assert table.cursor_row == cursor and app.screen is screen
        await pilot.pause(0.3)
        assert copied == []  # highlighting copies nothing by itself
        assert await _copied(app, pilot, copied) == lines[y][7:15]  # not the column names, not a whole row
        # Sideways to the end: the fixed Job ID column still selects its own text, a scrolled one its own.
        table.scroll_to(x=table.max_scroll_x, animate=False)
        await pilot.pause(0.1)
        lines = _screen_lines(app)
        await _drag(mouse, (10, y + 3), (20, y + 3), via=[(15, y + 3)])
        assert await _copied(app, pilot, copied) == lines[y + 3][10:21]
        assert _highlighted_columns(app, y + 3, table) == list(range(10, 21))
        x = lines[table.region.y].index("Created")  # the column names, over the column scrolled into view
        await _drag(mouse, (x, y + 5), (x + 10, y + 5), via=[(x + 5, y + 5)])
        # The Created cell covered whole copies as the time itself; the Duration cell's start as drawn.
        assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d  " + re.escape(lines[y + 5][x + 10]),
                            await _copied(app, pilot, copied))
        assert _highlighted_columns(app, y + 5, table) == list(range(x, x + 11))
        await _settle(app, pilot)


def test_a_drag_in_a_job_list_cell_highlights_and_copies_just_its_text():
    asyncio.run(_run_a_drag_in_a_job_list_cell_highlights_and_copies_just_its_text())


async def _run_a_double_click_that_opens_a_job_leaves_the_list_unhighlighted():
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    # Each simulated event waits for a pause, so on a slow runner the two clicks can land
    # further apart than Textual's 0.5 s double-click window; widen it so they still chain.
    app.CLICK_CHAIN_TIME_THRESHOLD = 10.0
    async with app.run_test(size=(150, 40)) as pilot:
        screen, table = await _demo_list(app, pilot)
        mouse = _pointer(app, pilot)
        lines = _screen_lines(app)
        first = _first_job_line(lines)
        y = _first_job_line([""] * (first + 1) + lines[first + 1:])  # the second job: not under the cursor
        # A double-click on a job: the first click puts the cursor on it, the second opens it.
        # Textual highlights a whole widget on a double-click; on the table that would be
        # every row, under the preview, there on coming back.
        await _click(mouse, 20, y)
        assert not isinstance(app.screen, JobSummaryScreen)
        await _click(mouse, 20, y)
        assert await _pause_until(pilot, lambda: isinstance(app.screen, JobSummaryScreen))
        assert app._chained_clicks == 2
        await pilot.press("b")
        assert await _pause_until(pilot, lambda: app.screen is screen)
        await pilot.pause(0.1)
        assert screen.selections == {}
        assert table.text_selection is None
        assert app.focused is table
        lines = _screen_lines(app)
        assert not any(_highlighted_columns(app, row, table) for row in range(y, y + 10)), lines[y:y + 10]
        await _settle(app, pilot)


def test_a_double_click_that_opens_a_job_leaves_the_list_unhighlighted():
    asyncio.run(_run_a_double_click_that_opens_a_job_leaves_the_list_unhighlighted())


async def _run_a_click_on_the_sort_s_name_or_arrow_changes_it():
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    async with app.run_test(size=(130, 40)) as pilot:
        screen, _table = await _demo_list(app, pilot)
        mouse = _pointer(app, pilot)
        y = screen.query_one("#listinfo", Static).region.y
        sort, descending = screen._sort, screen._descending
        line = _screen_lines(app)[y]
        await _click(mouse, line.index(f"sort: {sort}") + len("sort: ") + 1, y)  # the name: the next sort, as s
        assert await _pause_until(pilot, lambda: screen._sort != sort)
        assert screen._descending == descending
        line = _screen_lines(app)[y]
        await _click(mouse, max(line.rfind("↓"), line.rfind("↑")), y)  # the arrow: the other way, as o
        assert await _pause_until(pilot, lambda: screen._descending != descending)
        await _settle(app, pilot)


def test_a_click_on_the_sort_s_name_or_arrow_changes_it():
    asyncio.run(_run_a_click_on_the_sort_s_name_or_arrow_changes_it())


async def _run_a_click_on_the_resources_header_s_sub_job_or_window_switches_it():
    client = FakeClient()
    job = next(j for j in client.list_jobs() if j["status"] == "RUNNING" and len(j["sub_jobs"]) >= 2)
    app = CortexTrainingLogTUI(client, job["job_id"], poll_interval=0.01)
    async with app.run_test(size=(150, 40)) as pilot:
        assert await _pause_until(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        screen = app.screen
        screen.query_one("#job-tabs", TabbedContent).active = "resource"
        assert await _pause_until(pilot, lambda: screen._resource_data.kinds.get(screen._resource_source())
                                  and not screen._resource_loading, seconds=10.0)
        mouse = _pointer(app, pilot)
        status = screen.query_one("#resource-status", Static).content_region

        def spot(part: int) -> int:
            """Where to click on the header's part: 0 the sub-job (``training #0 (1 of 2)``), 1 the window."""
            line = _screen_lines(app)[status.y][status.x:]
            return status.x + line.index(line.split("  ·  ")[part]) + 1

        source, window = screen._resource_source(), screen._resource_range
        await _click(mouse, spot(0), status.y)  # the sub-job: the next one, as space
        assert await _pause_until(pilot, lambda: screen._resource_source() != source)
        assert screen._resource_range == window
        await _click(mouse, spot(1), status.y)  # the window: the next one, as w
        assert await _pause_until(pilot, lambda: screen._resource_range != window)
        await _settle(app, pilot)


def test_a_click_on_the_resources_header_s_sub_job_or_window_switches_it():
    asyncio.run(_run_a_click_on_the_resources_header_s_sub_job_or_window_switches_it())


async def _run_a_drag_across_job_list_rows_copies_that_span():
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    copied: list[str] = []
    app.copy_to_clipboard = copied.append
    async with app.run_test(size=(200, 40)) as pilot:  # every column in view
        screen, table = await _demo_list(app, pilot)
        mouse = _pointer(app, pilot)
        width = table.scrollable_content_region.width
        lines = [line[:width] for line in _screen_lines(app)]
        y = _first_job_line(lines)
        await _drag(mouse, (20, y), (60, y + 3), via=[(30, y + 1), (50, y + 2)])
        text = await _copied(app, pilot, copied)
        first, *_middle, last = text.split("\n")
        assert len(text.split("\n")) == 4
        # Cells covered in part copy as drawn; cells covered whole as their copy text: times
        # in full, each status as its word, no glyph and no relative age.
        assert first.startswith(lines[y][20:40]) and last.startswith(lines[y + 3][:40])
        assert last.endswith(lines[y + 3][56:61]) and "PENDING" in last
        assert re.search(r"\d{4}-\d\d-\d\d \d\d:\d\d", first) and "ago" not in text
        assert not set("●◔◐○×−✓") & set(text)
        # Drawn over the same characters (a line's trailing blank isn't copied, nor drawn).
        assert _highlighted_columns(app, y, table) == list(range(20, len(lines[y].rstrip())))
        assert _highlighted_columns(app, y + 1, table) == list(range(len(lines[y + 1].rstrip())))
        assert _highlighted_columns(app, y + 3, table) == list(range(61))
        assert _highlighted_columns(app, y - 1, table) == [] and _highlighted_columns(app, y + 4, table) == []
        # Held below the list, the drag stays in its rows: nothing else is highlighted.
        await _drag(mouse, (10, y + 4), (10, 38), via=[(10, 30), (10, 36)])
        assert set(screen.selections) == {table}
        # A drag that starts on the column names stays on them; the bars drawn between them aren't copied.
        names = table.region.y
        await _drag(mouse, (4, names), (40, names + 7), via=[(20, names + 3)])
        selection = screen.selections[table]
        assert (selection.start.y, selection.end.y) == (0, 0)
        assert await _copied(app, pilot, copied) == lines[names][4:41].replace("┃", " ").rstrip()
        # A change to the rows (e: every job unfolded) drops a highlight in them.
        await _drag(mouse, (8, y), (16, y), via=[(12, y)])
        assert screen.selections
        await pilot.press("e")
        assert await _wait(pilot, app, lambda: not screen.selections)
        await _settle(app, pilot)


def test_a_drag_across_job_list_rows_copies_that_span():
    asyncio.run(_run_a_drag_across_job_list_rows_copies_that_span())


async def _run_a_drag_on_a_column_s_bar_resizes_that_column():
    from cortex_training.tui.clipboard import selected_text
    from textual import events

    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    async with app.run_test(size=(200, 40)) as pilot:
        screen, table = await _demo_list(app, pilot)
        mouse = _pointer(app, pilot)
        names = table.region.y

        def width(label):
            return next(column.width for column in table.ordered_columns if column.key.value == label)

        def bar_after(label):
            line = _screen_lines(app)[names]
            return line.index("┃", line.index(label))

        # A bar at each column's right edge, in its padding: before every name but the first, and after the last.
        line = _screen_lines(app)[names]
        assert line.startswith(" Job ID ") and "┃ Status " in line and "┃ Type " in line
        assert line.rstrip().endswith("Updated ┃")
        # On a bar, or a blank cell beside one, the bar brightens and the pointer is a sideways arrow.
        for x in (bar_after("Owner") - 1, bar_after("Owner"), bar_after("Owner") + 1):
            await mouse(events.MouseMove, x, names)
            assert table._hovered == "Owner" and table.styles.pointer == "ew-resize"
        await mouse(events.MouseMove, bar_after("Owner") + 4, names)
        assert table._hovered is None and table.styles.pointer == "default"
        await pilot.press("down", "down")
        cursor, row = table.cursor_row, table.get_row_at(2)
        # Pressed on a bar, the column's right edge follows the pointer, while the drag lasts too.
        x = bar_after("Status")
        await mouse(events.MouseDown, x, names)
        await mouse(events.MouseMove, x + 3, names)
        assert width("Status") == 17
        await mouse(events.MouseMove, x + 6, names)
        await mouse(events.MouseUp, x + 6, names)
        assert width("Status") == 20 and bar_after("Status") == x + 6
        assert not screen.selections
        # In place: the rows, and the cursor on them, stay; the LIVE row's rule runs the new width.
        assert table.cursor_row == cursor and all(a is b for a, b in zip(table.get_row_at(2), row))
        at = bar_after("Job ID") + 1
        assert _screen_lines(app)[names + 1][at:at + 22] == " " + "─" * 20 + " "
        # A blank cell either side of a bar takes it too, so a press a cell off still resizes.
        x = bar_after("Type")
        await _drag(mouse, (x + 1, names), (x + 3, names), via=[(x + 2, names)])
        assert width("Type") == 15
        x = bar_after("Type")
        await _drag(mouse, (x - 1, names), (x - 3, names), via=[(x - 2, names)])
        assert width("Type") == 13
        # A drag on a name highlights it and leaves the widths, from a name's last letter beside its bar too.
        x = _screen_lines(app)[names].index("Type")
        await _drag(mouse, (x, names), (x + 3, names), via=[(x + 2, names)])
        assert selected_text(screen) == "Type"
        x = bar_after("Duration")  # Duration fills its column: its "n" is beside the bar
        await _drag(mouse, (x - 1, names), (x - 6, names), via=[(x - 3, names)])
        assert selected_text(screen) == "ration"
        assert (width("Type"), width("Duration")) == (13, 8)
        # No narrower than a few cells.
        x = bar_after("Model")
        await _drag(mouse, (x, names), (0, names), via=[(x - 10, names)])
        assert width("Model") == 3
        # The widths stay through a refresh and a narrower terminal.
        await pilot.press("r")
        assert await _pause_until(pilot, lambda: not screen._refreshing)
        await pilot.resize_terminal(150, 40)
        await pilot.pause(0.2)
        assert (width("Status"), width("Model")) == (20, 3)
        # A double-click on a bar gives its column back the width the list picks.
        x = bar_after("Status")
        await _click(mouse, x, names)
        await _click(mouse, x, names)
        assert await _pause_until(pilot, lambda: width("Status") == 14)
        assert width("Model") == 3
        await _settle(app, pilot)


def test_a_drag_on_a_column_s_bar_resizes_that_column():
    asyncio.run(_run_a_drag_on_a_column_s_bar_resizes_that_column())


async def _run_job_list_clicks_still_click_with_text_selection():
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    async with app.run_test(size=(130, 40)) as pilot:
        screen, table = await _demo_list(app, pilot)
        mouse = _pointer(app, pilot)
        lines = _screen_lines(app)
        y = _first_job_line(lines)
        await _drag(mouse, (8, y), (16, y), via=[(12, y)])
        assert screen.selections
        rows_top = table.region.y + 1  # the first row's line, under the column names
        # A click moves the cursor and drops the highlight; so does one whose hand moved a cell.
        await _click(mouse, 60, y + 5)
        assert table.cursor_row == y + 5 - rows_top and not screen.selections
        await _click(mouse, 60, y + 7, moved_to=(61, y + 7))
        assert table.cursor_row == y + 7 - rows_top and not screen.selections
        # A drag isn't a click: the cursor stays where it was.
        await _drag(mouse, (60, y + 9), (70, y + 11), via=[(65, y + 10)])
        assert table.cursor_row == y + 7 - rows_top and screen.selections
        # Nor is one let go straight below the table, where the pointer counts as on the
        # row it was pressed on: the highlight stays too.
        bottom = table.scrollable_content_region.bottom - 1
        await _drag(mouse, (60, bottom), (60, 38))
        assert table.cursor_row == y + 7 - rows_top and screen.selections and app.screen is screen
        table.scroll_to(y=0, animate=False)  # where Textual scrolled it while the drag was held below
        await pilot.pause()
        # A click on a job's ▾ folds it.
        rows = table.row_count
        assert lines[y][1] == "▾"
        await _click(mouse, 1, y)
        assert await _wait(pilot, app, lambda: table.row_count < rows)
        assert _screen_lines(app)[y][1] == "▸"
        # A click on the cursor's row opens the job's preview.
        await _click(mouse, 60, y)
        await _click(mouse, 60, y)
        assert await _wait(pilot, app, lambda: isinstance(app.screen, JobSummaryScreen))
        await _settle(app, pilot)


def test_job_list_clicks_still_click_with_text_selection():
    asyncio.run(_run_job_list_clicks_still_click_with_text_selection())


async def _run_the_preview_s_sub_job_table_selects_text():
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    copied: list[str] = []
    app.copy_to_clipboard = copied.append
    async with app.run_test(size=(130, 40)) as pilot:
        await _demo_list(app, pilot)
        await pilot.press("enter")  # the first job's preview
        assert await _wait(pilot, app, lambda: isinstance(app.screen, JobSummaryScreen)
                           and app.screen.query_one("#subjobs", DataTable).row_count > 2, tries=400)
        await pilot.pause(0.2)
        table = app.screen.query_one("#subjobs", DataTable)
        mouse = _pointer(app, pilot)
        lines = _screen_lines(app)
        y = table.region.y + 2  # the first sub-job's row, under the column names and the "entire job" row
        x = lines[y].index("training")
        await _drag(mouse, (x, y), (x + 5, y), via=[(x + 3, y)])
        assert _highlighted_columns(app, y, table) == list(range(x, x + 6))
        assert await _copied(app, pilot, copied) == "traini"
        assert isinstance(app.screen, JobSummaryScreen)  # the drag opened nothing
        await _settle(app, pilot)


def test_the_preview_s_sub_job_table_selects_text():
    asyncio.run(_run_the_preview_s_sub_job_table_selects_text())


async def _run_a_job_list_row_s_text_is_its_cells_as_drawn():
    # A row of cells that fit their columns is read from its cells' text, not drawn
    # (copying thousands of rows): it must read as the table draws it, every column.
    for width in (130, 80):  # at 80, job ids are cut short ("…"): those rows are drawn
        app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
        async with app.run_test(size=(width, 40)) as pilot:
            _screen, table = await _demo_list(app, pilot)
            read = 0
            for row in table.ordered_rows:
                plain = table._plain_parts(row.key)
                if str(row.key.value).startswith("hdr-"):
                    # A LIVE/HISTORY heading's rule is drawn, but not in a copy of its text.
                    assert plain is not None and "─" not in "".join(plain)
                    assert "─" in "".join(table._drawn_parts(row.key, 0))
                elif plain is not None:
                    read += 1
                    assert plain == table._drawn_parts(row.key, 0)
            assert read and (width < 130 or read > table.row_count // 2)
            await _settle(app, pilot)


def test_a_job_list_row_s_text_is_its_cells_as_drawn():
    asyncio.run(_run_a_job_list_row_s_text_is_its_cells_as_drawn())


def test_a_copy_never_waits_for_this_machine_s_clipboard_tool(monkeypatch):
    # A clipboard tool that hangs (xclip waits out its timeout when DISPLAY is gone)
    # froze the screen for as long: it runs on a worker now, and its note follows.
    import cortex_training.tui.app as app_module
    from cortex_training.tui.clipboard import copied_note

    release = threading.Event()
    tried = []

    def hung_tool(text):
        tried.append(text)
        release.wait(10)
        return True

    monkeypatch.setattr(app_module, "copy_local", hung_tool)

    async def run():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
        async with app.run_test() as pilot:
            started = time.monotonic()
            app.copy_out("some \x1b[31mtext", "9 characters")
            assert time.monotonic() - started < 1
            assert app.clipboard == "some text"  # OSC 52 went out at once, without the escape sequence
            assert await _pause_until(pilot, seconds=5.0, predicate=lambda: tried == ["some text"])
            await pilot.pause(0.1)
            assert _notes(app) == []  # nothing is said until the copy is done
            release.set()
            assert await _pause_until(pilot, seconds=5.0, predicate=lambda: _notes(app) == ["copied 9 characters"]), _notes(app)
            app.copy_to_clipboard("abc")  # Textual's own copies (ctrl+c in a text box) are said too
            assert await _pause_until(pilot, seconds=5.0, predicate=lambda: _notes(app) == ["copied 3 characters"]), _notes(app)
            app.copy_out("\x1b[0m", "the escape")  # nothing left to copy: no tool runs, and the note is as it was
            assert _notes(app) == [copied_note("the escape", False)]
            assert tried == ["some text", "abc"]
            await _settle(app, pilot)

    asyncio.run(run())


def test_copies_reach_the_clipboard_in_turn_and_the_last_one_stays(monkeypatch, tmp_path):
    import cortex_training.tui.app as app_module
    from cortex_training.tui.clipboard import copy_local

    clip = tmp_path / "clip.txt"
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CLIPBOARD_FILE", str(clip))  # never the real clipboard
    release = threading.Event()
    tried = []

    def first_one_hangs(text):
        tried.append(text)
        if len(tried) == 1:
            release.wait(10)
        return copy_local(text)

    monkeypatch.setattr(app_module, "copy_local", first_one_hangs)

    async def run():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
        async with app.run_test() as pilot:
            app.copy_out("one", "the first")
            assert await _pause_until(pilot, seconds=5.0, predicate=lambda: tried == ["one"])
            app.copy_out("two", "the second")
            app.copy_out("three", "the third")
            release.set()
            assert await _pause_until(pilot, seconds=5.0, predicate=lambda: _notes(app) == ["copied the third"]), _notes(app)
            assert tried == ["one", "three"]  # "two" was replaced before its turn came
            assert clip.read_text(encoding="utf-8") == "three"
            assert app.clipboard == "three"
            await _settle(app, pilot)

    asyncio.run(run())


def test_quit_does_not_wait_for_a_blocked_worker():
    release = threading.Event()
    c = _client()
    c.download_metrics.side_effect = lambda job_id, output_dir, **_options: release.wait(30) and []

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
            assert ok
            await pilot.press("g")  # Resources: the download blocks, like a request retrying a 5xx
            ok = await _wait(pilot, app, lambda: c.download_metrics.call_count == 1)
            assert ok

    started = time.monotonic()
    try:
        asyncio.run(run())  # asyncio joins the loop's default executor on the way out
        assert time.monotonic() - started < 15, "quitting waited for the blocked worker"
    finally:
        release.set()


def test_daemon_executor_bounds_blocked_workers_and_cancels_queued_work():
    from cortex_training.tui.app import _DaemonThreadExecutor

    release = threading.Event()
    both_started = threading.Event()
    started = []

    def blocked(value):
        started.append(value)
        if len(started) == 2:
            both_started.set()
        release.wait(5)
        return value

    executor = _DaemonThreadExecutor(max_workers=2)
    futures = [executor.submit(blocked, value) for value in range(10)]
    assert both_started.wait(2)
    executor.shutdown()
    assert len(started) == 2
    assert sum(future.cancelled() for future in futures) == 8
    release.set()
    assert [future.result(2) for future in futures[:2]] == [0, 1]
    with pytest.raises(RuntimeError, match="after shutdown"):
        executor.submit(lambda: None)


async def _run_tail_waits_for_logs_tab():
    c = _client()
    app = CortexTrainingLogTUI(c, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
        assert ok
        app.push_screen(LogScreen(c, "7", job={"job_id": "7", "status": "RUNNING"}, initial_tab="resource"))
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        assert ok
        for _ in range(20):
            await pilot.pause(0.02)
        assert c.tail_logs.call_count == 0, "the tail polled while Logs was hidden"
        await pilot.press("l")
        ok = await _wait(pilot, app, lambda: c.tail_logs.call_count > 0)
        assert ok, "the tail did not start when Logs was shown"
        await pilot.press("g")
        await pilot.pause(0.5)  # let a request already in flight finish
        calls = c.tail_logs.call_count
        for _ in range(20):
            await pilot.pause(0.02)
        assert c.tail_logs.call_count == calls, "the tail kept polling after Logs was hidden"
        await _settle(app, pilot)


def test_tail_polls_only_while_logs_is_showing():
    asyncio.run(_run_tail_waits_for_logs_tab())


def test_failed_refresh_keeps_the_last_list_and_the_error(monkeypatch):
    c = _client()

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            assert ok
            screen = app.screen
            c.list_jobs.side_effect = RuntimeError("503 Service Unavailable")
            await pilot.press("r")
            # The notice has a line of its own, above the list it keeps.
            notice = screen.query_one("#listnotice", Static)
            info = lambda: str(notice.render()) if notice.display else ""  # noqa: E731
            ok = await _wait(pilot, app, lambda: "refresh failed" in info())
            assert ok
            await pilot.press("s")  # any re-render keeps both the list and the error
            await pilot.pause()
            assert len(screen._job_by_item) == 2 and "503 Service Unavailable" in info()
            assert f"showing the list from {screen._last_refreshed[:5]}" in info()
            c.list_jobs.side_effect = None
            await pilot.press("r")
            ok = await _wait(pilot, app, lambda: "refresh failed" not in info() and screen._job_by_item)
            assert ok
            await _settle(app, pilot)

    asyncio.run(run())


def test_first_load_failure_stays_visible():
    c = _client()
    c.list_jobs.side_effect = RuntimeError("403 Forbidden [x]")

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            info = lambda: str(app.screen.query_one("#listinfo", Static).render())  # noqa: E731
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and "403 Forbidden [x]" in info())
            assert ok and "[error]" in info()
            await pilot.press("s")
            await pilot.pause()
            assert "403 Forbidden [x]" in info()  # not replaced by "no jobs"
            await _settle(app, pilot)

    asyncio.run(run())


def _old_terminal_job(minutes_ago=30):
    now = datetime.now(timezone.utc)
    return {**_terminal_job(), "created_at": (now - timedelta(hours=3)).isoformat(),
            "state_changed_at": (now - timedelta(minutes=minutes_ago)).isoformat()}


async def _run_saved_console_waits_for_logs_and_is_kept(c):
    app = CortexTrainingLogTUI(c, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
        assert ok
        app.push_screen(LogScreen(c, "7", job={"job_id": "7", "status": "FAILED"}, initial_tab="resource"))
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        assert ok
        for _ in range(10):
            await pilot.pause(0.03)
        assert c.download_stdout_logs.call_count == 0, "downloaded the whole console for another tab"
        await pilot.press("l")
        ok = await _wait(pilot, app, lambda: "saved line" in app.screen._shown_lines, tries=200)
        assert ok, "the saved console did not load once Logs was shown"
        await pilot.press("b")
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen))
        assert ok
        # A job that ended long ago keeps its console: reopening replays it without a download.
        app.push_screen(LogScreen(c, "7", job={"job_id": "7", "status": "FAILED"}, initial_tab="logs"))
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and "saved line" in app.screen._shown_lines)
        assert ok
        assert c.download_stdout_logs.call_count == 1
        await _settle(app, pilot)


async def _run_the_pane_has_its_console_before_the_cache_limit_pass(c, holder, seen):
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    holder["app"] = app
    async with app.run_test(size=(140, 40)) as pilot:
        ok = await _wait(pilot, app, lambda: seen, tries=300)
        assert ok, "keeping the console never ran the limit pass"
        ok = await _wait(pilot, app, lambda: "saved line" in app.screen._shown_lines, tries=200)
        assert ok, "the console was never painted"
        await _settle(app, pilot)


def test_the_pane_has_its_console_before_the_cache_limit_pass_reads_the_cache(monkeypatch):
    # The pass reads the size of everything in the cache first: run before the pane was told,
    # it held a finished job's console back behind it (seconds, on a network home directory).
    import cortex_training.tui.logs as logs_module

    c = _client()
    c.get_job.return_value = _old_terminal_job(minutes_ago=30)

    def download(job_id, output_dir):
        _write_stdout(output_dir, "7:training:0", "saved line\n")
        return []

    c.download_stdout_logs.side_effect = download
    holder, seen = {}, []

    def limit_pass(root, limit):
        screen = holder["app"].screen
        seen.append((screen._stdout_inflight, screen._stdout_ready))  # _finish_download delivered
        return []

    monkeypatch.setattr(logs_module, "enforce_cache_limit", limit_pass)
    asyncio.run(_run_the_pane_has_its_console_before_the_cache_limit_pass(c, holder, seen))
    assert seen == [(False, True)], "the limit pass ran before the pane had the console"


def test_saved_console_downloads_only_on_logs_and_is_kept_for_old_jobs():
    c = _client()
    c.get_job.return_value = _old_terminal_job(minutes_ago=30)

    def download(job_id, output_dir):
        _write_stdout(output_dir, "7:training:0", "saved line\n")
        return []

    c.download_stdout_logs.side_effect = download
    asyncio.run(_run_saved_console_waits_for_logs_and_is_kept(c))


@pytest.mark.parametrize("minutes_ago, reused_by", [
    (1, None),  # the job just finished: its last chunks may still be uploading
    (30, "2027-01-01T00:00:00Z"),  # the id was reused by a newer job, which downloads its own console
])
def test_a_console_not_kept_or_kept_for_another_job_is_downloaded_again(minutes_ago, reused_by):
    async def run():
        c = _client()
        c.get_job.return_value = _old_terminal_job(minutes_ago=minutes_ago)
        c.download_stdout_logs.side_effect = lambda job_id, output_dir: _write_stdout(
            output_dir, "7:training:0", "saved line\n") and []
        for again in (False, True):
            if again and reused_by:
                c.get_job.return_value = {**c.get_job.return_value, "created_at": reused_by}
            app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
            async with app.run_test() as pilot:
                ok = await _wait(pilot, app, lambda: "saved line" in app.screen._shown_lines)
                assert ok
                await _settle(app, pilot)
        assert c.download_stdout_logs.call_count == 2

    asyncio.run(run())


def test_refresh_on_logs_keeps_the_selected_source():
    async def run():
        c = _client()
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
            assert ok
            app.screen._start_tail("7:sampling:0")
            calls = c.get_job.call_count
            await pilot.press("r")
            ok = await _wait(pilot, app, lambda: c.get_job.call_count > calls and app.screen._sources_ready)
            assert ok
            for _ in range(5):
                await pilot.pause()
            assert app.screen._current_source == "7:sampling:0"
            await _settle(app, pilot)

    asyncio.run(run())


def test_markup_like_names_and_server_text_render_literally():
    job = {"job_id": "7", "status": "PENDING", "reason": "quota [/x",
           "reason_details": "needs [/b] GPUs", "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training",
                                                            "model_name": "org/model [/v2"}]}
    c = _client()
    c.list_jobs.return_value = [job]
    c.get_job.return_value = job

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            assert ok
            app.screen.query_one("#jobfilter", Input).value = "[/b] [x]"  # a search that matches no job
            ok = await _wait(pilot, app, lambda: "“[/b] [x]”" in str(app.screen.query_one("#nomatch", Static).render()))
            assert ok
            app.push_screen(JobSummaryScreen(c, "7", job=job))
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobSummaryScreen)
                             and "needs [/b] GPUs" in str(app.screen.query_one("#health", Static).render()))
            assert ok
            assert "org/model [/v2" in str(app.screen.query_one("#preview", Static).render())
            app.push_screen(LogScreen(c, "7", job=job, initial_tab="logs"))
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen)
                             and "org/model [/v2" in str(app.screen.query_one("#summary", Static).render()))
            assert ok
            assert app.return_code is None
            await _settle(app, pilot)

    asyncio.run(run())
    assert c.get_job.called


def test_quit_without_job_id_derives_it_from_sub_job_id_and_reports_crashes(monkeypatch):
    seen = {}

    class FakeApp:
        return_code = 1

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0):
            seen.update(job_id=job_id, sub_job_id=sub_job_id)

        def run(self):
            pass

    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    monkeypatch.setattr("cortex_training.cli.commands.build_client", lambda args, cls: MagicMock())
    rc = tui_main.run(["--sub-job-id", "job-x:sampling:0", "--base-url", "http://localhost:1", "--database", "DB"])
    assert seen == {"job_id": "job-x", "sub_job_id": "job-x:sampling:0"}
    assert rc == 1  # the app's return code, not a blanket 0
    with pytest.raises(SystemExit):
        tui_main.run(["--sub-job-id", "not-a-sub-job-id", "--base-url", "http://localhost:1", "--database", "DB"])


def test_the_tui_tags_its_client_and_the_job_clients_it_opens(monkeypatch):
    # Their metrics then say the TUI made the call (client.surface), successes included.
    seen = []
    surfaces = []  # the client's tag while the app runs, which is when it makes its calls

    class FakeApp:
        return_code = 0

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0):
            seen.append(client)

        def run(self):
            surfaces.append(seen[-1]._telemetry_surface)

    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    assert tui_main.run(["--base-url", "http://localhost:1", "--database", "DB"]) == 0
    client = seen[0]
    try:
        assert surfaces == ["tui"]
        assert client._telemetry_surface == "tui"
        # A job from another schema's row gets its own client, as the job list opens it.
        from cortex_training.tui.account import JobClients
        from cortex_training.tui.untrusted import printable_client

        row = {"job_id": "j1", "database_name": "ML", "schema_name": "EXP"}
        job_client = JobClients(printable_client(client)).for_job(row)
        assert job_client.wrapped is not client and job_client._telemetry_surface == "tui"
    finally:
        client.close()


def _screenshot_script():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "tools" / "tui_screenshot.py"
    spec = importlib.util.spec_from_file_location("tui_screenshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ScreenshotClient:
    base_url, database, schema = "https://ACCOUNT.snowflakecomputing.com", "CORTEX_TRAINING_DB", "PUBLIC"


def test_the_tui_and_the_screenshot_script_build_the_same_bounded_client(monkeypatch):
    # One helper builds both, so a screenshot's reads are the TUI's: bounded, and tagged as the TUI's.
    import cortex_training.cli.commands as cli

    built = []

    class FakeApp:
        return_code = 0

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0):
            pass

        def run(self):
            pass

    # Nothing connects: the profile is only named, and no client is built for real.
    monkeypatch.setattr(cli, "build_client", lambda args, cls: built.append(
        (args.connection, args.max_retries, args.request_timeout)) or _ScreenshotClient())
    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)

    assert tui_main.run(["--connection", "some-connection"]) == 0
    client = _screenshot_script()._client(argparse.ArgumentParser(), connection="some-connection", config=None)

    # Never the client's own ten retries and ten-minute read, which stall a view on a 404.
    assert built == [("some-connection", tui_main.READ_MAX_RETRIES, tui_main.READ_TIMEOUT)] * 2
    assert client._telemetry_surface == "tui"


def test_the_screenshot_script_reads_only_the_account_it_is_named(tmp_path, monkeypatch, capsys):
    import cortex_training.cli.commands as cli

    script = _screenshot_script()
    built = []
    monkeypatch.setattr(cli, "build_client",
                        lambda args, cls: built.append((args.connection, args.config)) or _ScreenshotClient())
    monkeypatch.setenv("CORTEX_TRAINING_CONNECTION", "ambient")
    # Run the old way, with no account named, it refuses rather than shoot whatever the environment gives.
    with pytest.raises(SystemExit) as exit_:
        script.main([str(tmp_path / "list.svg")])
    assert exit_.value.code == 2
    assert "one of the arguments --connection --config is required" in capsys.readouterr().err
    assert built == []

    # Named by a config file, the account is that file's, not the environment's profile.
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"base_url": "http://localhost:1", "database": "DB"}), encoding="utf-8")
    script._client(argparse.ArgumentParser(), connection=None, config=str(path))
    assert built == [(None, str(path))]


def test_the_screenshot_script_s_connection_errors_are_the_tui_s(tmp_path, monkeypatch, capsys):
    # On a machine with no account to read: the TUI's error and how to connect, not a traceback.
    import cortex_training.cli.commands as cli
    from cortex_training.client.auth import SnowflakeProfileError

    script = _screenshot_script()

    def no_profile(args, cls):
        raise SnowflakeProfileError("no connection named NAME")

    monkeypatch.setattr(cli, "build_client", no_profile)
    with pytest.raises(SystemExit) as exit_:
        script.main([str(tmp_path / "list.svg"), "--connection", "NAME"])
    assert exit_.value.code == 2
    lines = capsys.readouterr().err.splitlines()
    assert lines[0].endswith(": error: no connection named NAME") and lines[1:] == [tui_main._CONNECT_HINT]

    for name in ("CORTEX_TRAINING_BASE_URL", "CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST",
                 "CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"database": "DB"}), encoding="utf-8")  # no account in it
    with pytest.raises(SystemExit) as exit_:
        script.main([str(tmp_path / "list.svg"), "--config", str(path)])
    assert exit_.value.code == 2
    assert ": error: no connection configured: " in capsys.readouterr().err
    assert not (tmp_path / "list.svg").exists()


def test_the_screenshot_script_says_what_it_reads_and_leaves_no_cache_behind(tmp_path, monkeypatch, capsys):
    import cortex_training.cli.commands as cli

    script = _screenshot_script()
    monkeypatch.setattr(cli, "build_client", lambda args, cls: _ScreenshotClient())
    caches = []

    async def shoot(out, size, keys, job_id, client):
        cache = Path(os.environ["CORTEX_TRAINING_TUI_CACHE_DIR"])
        (cache / "stdout.log").write_text("a downloaded log", encoding="utf-8")
        caches.append(cache)
        # Said before it shoots, on stderr: the account and database whose jobs the shot shows.
        assert ("reading https://ACCOUNT.snowflakecomputing.com, database CORTEX_TRAINING_DB, schema PUBLIC"
                in capsys.readouterr().err)

    monkeypatch.setattr(script, "_shoot", shoot)
    real_cache = tmp_path / "real-cache"
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(real_cache))  # and restored after the test
    assert script.main([str(tmp_path / "list.svg"), "--connection", "NAME"]) == 0
    assert len(caches) == 1 and caches[0] != real_cache
    assert not caches[0].exists()  # the shot's logs and telemetry went with it


@pytest.mark.parametrize(
    "config, message, usage",
    [
        ({"base_url": "http://localhost:1", "database": "DB", "poll_interval": "2"},
         "config poll_interval must be a number", False),
        ({"base_url": "http://localhost:1", "database": "DB", "verify_ssl": "yes"},
         "config verify_ssl must be a boolean", False),
        ({"base_url": "http://localhost:1", "\x1b]52;c;eA==\x07": 1}, "unknown config key(s): \\x1b]52;c;eA==\\x07",
         False),
        ({"base_url": "http://localhost:1", "bad\nkey": 1}, "unknown config key(s): bad\\nkey", False),
        ("{not json", "invalid JSON in config", False),
        (None, "No such file or directory", False),
        # Never a traceback, this one keeps the usage above its error line, as it always has.
        ({"base_url": "localhost:1", "database": "DB"}, "base_url must start with http:// or https://", True),
    ],
)
def test_a_config_the_tui_cannot_use_is_an_error_not_a_traceback(tmp_path, monkeypatch, capsys, config, message,
                                                                 usage):
    class NeverStarts:
        def __init__(self, *args, **kwargs):
            raise AssertionError("the TUI started")

    for name in ("CORTEX_TRAINING_CONNECTION", "CORTEX_TRAINING_BASE_URL", "CORTEX_TRAINING_HOST", "SNOWFLAKE_HOST",
                 "CORTEX_TRAINING_PAT", "SNOWFLAKE_PAT"):
        monkeypatch.delenv(name, raising=False)  # a PAT would make the bare base_url a host
    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", NeverStarts)
    path = tmp_path / "config.json"
    if config is not None:
        path.write_text(config if isinstance(config, str) else json.dumps(config), encoding="utf-8")
    with pytest.raises(SystemExit) as exit_:
        tui_main.run(["--config", str(path)])
    assert exit_.value.code == 2
    err = capsys.readouterr().err
    lines = err.splitlines()
    if usage:
        assert lines[0].startswith("usage: cortex-training tui"), err
    else:
        assert len(lines) == 1, err  # one line: no usage, no traceback
    assert lines[-1].startswith("cortex-training tui: error: ") and message in lines[-1], err
    assert "\x1b" not in err and "\x07" not in err  # a key from the file can't act on the terminal


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_the_log_poll_interval_must_be_positive_and_finite(value, capsys):
    with pytest.raises(SystemExit) as exit_:
        tui_main.run([f"--poll-interval={value}"])  # refused as the flag is read, before any connection
    assert exit_.value.code == 2
    assert "must be a finite number greater than zero" in capsys.readouterr().err


def test_the_poll_interval_is_the_flag_s_alone_and_the_sign_in_names_what_the_config_gives(tmp_path, monkeypatch):
    import cortex_training.cli.commands as cli

    seen = []

    class FakeApp:
        return_code = 0
        sign_in = None

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0):
            self.poll_interval = poll_interval

        def run(self):
            seen.append((self.poll_interval, self.sign_in))

    reads = []
    load_config = cli._load_config
    monkeypatch.setattr(
        cli,
        "_load_config",
        lambda path, **kwargs: reads.append(path) or load_config(path, **kwargs),
    )
    monkeypatch.delenv("CORTEX_TRAINING_CONNECTION", raising=False)
    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    built = []  # the poll interval each client is built with
    monkeypatch.setattr("cortex_training.cli.commands.build_client", lambda args, cls: built.append(args.poll_interval) or MagicMock())
    path = tmp_path / "config.json"
    # The config's poll_interval is the CLI's result polling: neither the TUI's log polls nor a reason not to start.
    path.write_text(json.dumps({"base_url": "http://localhost:1", "database": "DB", "poll_interval": 10}),
                    encoding="utf-8")
    sign_in = {"kind": "url", "config": str(path.resolve()), "base_url": "http://localhost:1",
               "base_url_in_config": True}
    assert tui_main.run(["--config", str(path)]) == 0
    assert seen == [(1.0, sign_in)]
    assert len(reads) == 2  # the CLI's resolution, then the TUI's one read for what that doesn't say
    assert tui_main.run(["--config", str(path), "--poll-interval", "30", "--base-url", "http://other:2"]) == 0
    assert seen[-1] == (30.0, {**sign_in, "base_url": "http://other:2", "base_url_in_config": False})
    assert built == [1.0, 1.0]  # never the log polls' 30 s: the client refuses more than 6


def test_resource_refresh_skips_the_reparse_when_nothing_new_arrived(monkeypatch):
    import cortex_training.tui.resource_view as tui_app

    loaders = []
    real_loader = tui_app.resource_loader
    monkeypatch.setattr(tui_app, "resource_loader", lambda root: loaders.append(real_loader(root)) or loaders[-1])
    c = _client()
    state = {"new": True}

    def download(job_id, output_dir, *, resume=False, **_kwargs):
        if state["new"]:
            _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
            assert ok
            app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
            ok = await _wait(pilot, app, lambda: app.screen._resource_loaded and not app.screen._resource_loading)
            assert ok
            read = loaders[-1].bytes_read
            assert read > 0
            state["new"] = False
            app.screen.action_refresh_sources()  # r on Resources: nothing new was committed
            ok = await _wait(pilot, app, lambda: c.download_metrics.call_count == 2 and not app.screen._resource_loading)
            assert ok
            assert loaders[-1].bytes_read == read, "re-read the cache although nothing changed"
            await _settle(app, pilot)

    asyncio.run(run())


def _real_download_client(monkeypatch, *, subs, count=3):
    """A TUI client whose downloads go through the real SDK code over a fake
    Snowflake connection (job metadata and tails stay mocked)."""
    from tests.test_client import TestChunkedArtifactDownload

    real, connection, _paths, _sleeps, _opened = TestChunkedArtifactDownload()._setup(
        monkeypatch, subs=subs, count=count
    )
    c = _client()
    c.download_stdout_logs = real.download_stdout_logs
    return c, connection


async def _run_selected_sub_job_without_chunks(c):
    app = CortexTrainingLogTUI(c, "job-1", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._stdout_ready, tries=300)
        assert ok, "saved-console download did not finish"
        screen = app.screen
        assert screen._current_source == "job-1:training:0"
        assert not screen._stdout_failed
        assert not any("couldn't be downloaded" in line for line in screen._shown_lines)
        ok = await _wait(pilot, app, lambda: "No log output is available." in screen._shown_lines)
        assert ok, "a sub-job without saved output did not show the empty note"
        screen._start_tail("job-1:sampling:0")
        ok = await _wait(pilot, app, lambda: "job-1:sampling:0 line 2" in screen._shown_lines)
        assert ok, "the other sub-job's console was not fetched"
        await _settle(app, pilot)


def test_selected_sub_job_without_chunks_falls_through_to_the_whole_job(monkeypatch):
    c, connection = _real_download_client(monkeypatch, subs=["job-1:sampling:0"])
    c.get_job.return_value = {
        "status": "FAILED",
        "sub_jobs": [
            {"sub_job_id": "job-1:sampling:0", "job_type": "sampling"},
            {"sub_job_id": "job-1:training:0", "job_type": "training"},
        ],
    }
    asyncio.run(_run_selected_sub_job_without_chunks(c))
    gets = [cmd for cmd in connection.commands if cmd.startswith("GET ")]
    assert len(gets) == 3


def test_job_list_loading_state_and_expand_all_toggle():
    release = threading.Event()
    c = _client()
    sub = [{"sub_job_id": "x:training:0", "job_type": "training"}]
    real = [*c.list_jobs.return_value[:1],
            {"job_id": "old", "status": "FAILED", "sub_jobs": sub}, {"job_id": "older", "status": "FAILED", "sub_jobs": sub}]
    c.list_jobs.side_effect = lambda **_bounds: release.wait(10) and real

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen))
            assert ok
            screen = app.screen
            nomatch = lambda: str(screen.query_one("#nomatch", Static).render())  # noqa: E731
            ok = await _wait(pilot, app, lambda: "Loading jobs" in nomatch())
            assert ok, "the list claimed there were no jobs while still loading"
            release.set()
            ok = await _wait(pilot, app, lambda: screen._job_by_item)
            assert ok
            old = next(job for job in screen._all_jobs if job["job_id"] == "old")
            screen._expanded.add(screen._job_key(old))  # one history job expanded by hand (space) ...
            screen.action_expand_all()  # ... then e expands everything rather than collapsing it
            assert all(screen._is_expanded(j) for j in screen._all_jobs if views_sub_jobs(j))
            await _settle(app, pilot)

    from cortex_training.tui.views import sub_jobs as views_sub_jobs

    try:
        asyncio.run(run())
    finally:
        release.set()


async def _run_recently_ended_hint(state_changed_at, expect_hint):
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    c = _client()
    c.get_job.return_value = {
        **_terminal_job(),
        "state_changed_at": _dt.fromtimestamp(state_changed_at, tz=_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    c.download_stdout_logs.side_effect = lambda job_id, output_dir, **kwargs: []
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._state_changed_at is not None)
        assert ok
        await _wait(pilot, app, lambda: "FAILED" in (app.sub_title or ""))
        assert await _wait(pilot, app, lambda: app.screen._stdout_ready and not app.screen._repainting)
        hint = "so its saved logs may not all be uploaded yet; r checks again"
        assert any(hint in line for line in app.screen._shown_lines) is expect_hint, app.screen._shown_lines
        await _settle(app, pilot)


def test_a_just_ended_job_says_its_last_output_may_still_be_uploading():
    asyncio.run(_run_recently_ended_hint(time.time() - 30, True))


def test_a_long_ended_job_has_no_upload_hint():
    asyncio.run(_run_recently_ended_hint(time.time() - 3 * 86400, False))


def test_list_failure_is_reported_and_listings_never_stack(monkeypatch):
    monkeypatch.setattr(JobListScreen, "_REFRESH_SECONDS", 0.05)
    release = threading.Event()
    c = _client()

    def failing_list(**_bounds):
        release.wait(10)  # a call still retrying a 5xx
        raise RuntimeError("503 Service Unavailable")

    c.list_jobs.side_effect = failing_list

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen))
            assert ok
            for _ in range(10):  # several refresh ticks and an r while the first call hangs
                await pilot.pause(0.05)
            await pilot.press("r")
            assert c.list_jobs.call_count == 1
            release.set()
            info = lambda: str(app.screen.query_one("#listinfo", Static).render())  # noqa: E731
            ok = await _wait(pilot, app, lambda: "503 Service Unavailable" in info())
            assert ok, "the failed listing's error was dropped"
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_live_header_row_stays_visible_after_a_slow_first_load():
    c = _client()
    # Taller than the table, so it can scroll at all (the bug needs a scrollable table).
    real = [{"job_id": f"job-{i:03d}", "status": "RUNNING" if i < 4 else "CANCELLED",
             "sub_jobs": [{"sub_job_id": f"job-{i:03d}:training:0", "job_type": "training"}]} for i in range(60)]
    c.list_jobs.side_effect = lambda **_bounds: time.sleep(1.0) or real

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item, tries=300)
            assert ok
            await pilot.pause(1.0)
            table = app.screen.query_one("#jobs", DataTable)
            assert table.scroll_y == 0 and table.cursor_row == 1  # header row 0 in view, cursor on the first job
            await _settle(app, pilot)

    asyncio.run(run())


def test_resource_refresh_shows_its_refreshing_state(monkeypatch):
    release = threading.Event()
    c = _client()
    calls = []

    def download(job_id, output_dir, *, resume=False, **_kwargs):
        calls.append(1)
        if len(calls) > 1:
            release.wait(10)  # the refresh is slow
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        return []

    c.download_metrics.side_effect = download

    async def run():
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
            assert ok
            app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
            ok = await _wait(pilot, app, lambda: app.screen._resource_loaded and not app.screen._resource_loading)
            assert ok
            app.screen.action_refresh_sources()
            status = lambda: str(app.screen.query_one("#resource-status", Static).render())  # noqa: E731
            ok = await _wait(pilot, app, lambda: "cached · refreshing…" in status())
            assert ok, "a refresh of loaded data showed no refreshing state"
            release.set()
            await _settle(app, pilot)

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_enter_right_after_typing_opens_a_job_from_the_new_matches():
    async def run():
        app = CortexTrainingLogTUI(_client(), poll_interval=0.01)
        async with app.run_test(size=(140, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            assert ok
            await pilot.press("/", "o", "l", "d", "enter", "enter")  # faster than the search debounce
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobSummaryScreen))
            assert ok and app.screen._job_id == "old"
            await _settle(app, pilot)

    asyncio.run(run())


async def _run_no_end_time_no_upload_hint():
    c = _client()
    c.get_job.return_value = _terminal_job()  # no state_changed_at
    c.download_stdout_logs.side_effect = lambda job_id, output_dir, **kwargs: []
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and "FAILED" in (app.sub_title or ""))
        assert ok
        for _ in range(5):
            await pilot.pause()
        assert app.screen._state_changed_at is None
        assert not any("may not all be uploaded" in line for line in app.screen._shown_lines)
        await _settle(app, pilot)


def test_an_ended_job_without_an_end_time_gets_no_upload_hint():
    asyncio.run(_run_no_end_time_no_upload_hint())


# The client methods the TUI calls.
TUI_CALLS = ("get_job", "list_jobs", "list_jobs_page", "for_job", "tail_logs", "download_metrics",
             "download_stdout_logs", "fetch_execution_logs")
# Parameters of the real client the fake leaves out: the TUI never passes them, or
# checks the client's signature for them first (sub_job_ids, on_chunk, stale_after_seconds,
# and a download's preview).
_ENGINE = {"max_workers", "progress", "on_chunk", "sub_job_ids", "strict", "stale_after_seconds",
           "transfer_timeout_seconds", "preview_seconds", "on_preview"}
NOT_IN_DEMO = {
    "get_job": {"retry_on"},
    "tail_logs": {"sub_job_type"},
    "download_metrics": _ENGINE,
    "download_stdout_logs": _ENGINE,
}


def _parameters(method, leave_out=frozenset()) -> list:
    """``(name, kind, default)`` of each parameter, in order: annotations may differ."""
    return [(p.name, p.kind, p.default) for p in inspect.signature(method).parameters.values()
            if p.name not in leave_out]


@pytest.mark.parametrize("name", TUI_CALLS)
def test_the_fake_client_takes_what_the_real_client_takes(name):
    real = getattr(CortexTrainingClient, name)
    left_out = NOT_IN_DEMO.get(name, set())
    assert left_out <= set(inspect.signature(real).parameters)  # each one left out is still the client's
    assert _parameters(getattr(FakeClient, name)) == _parameters(real, left_out)


ESC, BEL = "\x1b", "\x07"


# ─── the review fixes ─────────────────────────────────────────────────────────


def test_a_cancelled_worker_stops_waiting_out_a_long_poll_interval():
    # A tail sleeps between polls on its own thread. Cancelling a Textual worker can't
    # interrupt time.sleep, so without this the thread (and its executor slot) would be
    # held for the whole interval after the screen closed.
    from cortex_training.tui.widgets import _sleep_until_cancelled

    class Worker:
        is_cancelled = False

    worker = Worker()
    sleep = _sleep_until_cancelled(worker)
    threading.Timer(0.2, lambda: setattr(worker, "is_cancelled", True)).start()
    started = time.monotonic()
    sleep(300.0)
    assert time.monotonic() - started < 5.0  # not 300

    # A short wait is slept in one go, so the tail's own timing is unchanged.
    worker.is_cancelled = False
    started = time.monotonic()
    sleep(0.05)
    assert 0.03 <= time.monotonic() - started < 1.0


def test_an_older_tail_does_not_clear_the_newer_tail_s_in_flight_marker():
    # The status line says "no answer yet" while a tail's read is in flight. A cancelled
    # older tail answering later must not clear the marker the newer read set.
    from cortex_training.tui.widgets import _TimedTail

    class Screen:
        _tail_request = None

    screen = Screen()
    newer = _TimedTail(MagicMock(), screen, 2)

    class Client:
        def tail_logs(self, *_args, **_kwargs):
            # While the older read is in flight, a newer tail starts and claims the marker.
            newer._screen._tail_request = (newer._gen, time.monotonic())
            return {"entries": [], "next_cursor": "", "eof": True}

    older = _TimedTail(Client(), screen, 1)
    older.tail_logs("7", sub_job_id="7:training:0")
    assert screen._tail_request is not None, "the older tail cleared the newer request"
    assert screen._tail_request[0] == 2

    # Its own marker it does clear.
    own = _TimedTail(Client(), screen, 2)
    own.tail_logs("7", sub_job_id="7:training:0")
    assert screen._tail_request is None


def test_one_job_has_one_cache_wherever_it_was_opened_from():
    # The configured schema's list gives the client its configured identifiers; the
    # account-wide list's child client holds the names as stored. Both are the same
    # schema, so the job must not download its logs and telemetry twice.
    from cortex_training.tui.widgets import _cache_job_id

    class Client:
        base_url = "https://ACCOUNT.snowflakecomputing.com"
        endpoint = "cortex-training"

        def __init__(self, database, schema):
            self.database, self.schema = database, schema

    configured, resolved = Client("team_db", "ml"), Client("TEAM_DB", "ML")
    assert _cache_job_id(configured, "J") == _cache_job_id(resolved, "J")
    # A quoted lower-case name is its own identifier, and another schema is another cache.
    assert _cache_job_id(Client("DB", "ONE"), "J") != _cache_job_id(Client("DB", "TWO"), "J")
    assert _cache_job_id(Client("ONE", "S"), "J") != _cache_job_id(Client("TWO", "S"), "J")
    # Nothing to key on (a test double): the job id stands alone.
    assert _cache_job_id(object(), "J") == "J"


def test_a_location_is_routed_as_stored_and_drawn_made_printable():
    # printable_data cleans everything a job record carries, but these two names are SQL
    # identifiers that route the job's calls: cleaned, they name a different schema and
    # every call for the job answers 404.
    from cortex_training.tui.untrusted import printable_client

    stored = "ml​x"  # a quoted schema whose name holds a zero-width space

    class Service:
        database = schema = "X"

        def list_jobs_page(self, *_args, **_kwargs):
            return {"jobs": [{"job_id": "J", "database_name": "TEAM_DB", "schema_name": stored,
                              "reason": f"boom{ESC}]52;c;cGF5bG9hZA=={BEL}"}]}

        def for_job(self, row):
            return row

    row = printable_client(Service()).list_jobs_page()["jobs"][0]
    assert row["schema_name"] == stored, "the stored identifier was changed"
    assert row["database_name"] == "TEAM_DB"
    assert views.job_key(row) == ("TEAM_DB", stored, "J")  # routed on what the service stored
    assert ESC not in row["reason"] and "cGF5bG9hZA" not in row["reason"]  # other text still cleaned
    assert "​" not in views.location_label(row)  # what is drawn is printable


@pytest.mark.parametrize("kind,command", [("stdout", "cortex-training download-log 7 --log-type stdout"),
                                          ("metrics", "cortex-training download-metrics 7")])
def test_skipped_chunk_hint_names_the_cli_command_for_the_tui_cache(kind, command, tmp_path):
    # The client's own hint says to download again without resume, but the TUI downloads
    # into its cache, not a directory the user chose: the hint names that directory.
    import shlex

    from cortex_training.tui.widgets import _result_warnings

    cache = tmp_path / "cache dir" / kind
    rows = [{"sub_job_id": "7:training:0", "skipped_chunks": ["chunk"],
             "warnings": ["1 damaged chunk(s) skipped by an earlier download are still missing from the "
                          "saved file; download again without resume to retry them"]}]
    (message,) = _result_warnings(rows, kind=kind, job_id="7", output_dir=cache)["7:training:0"]
    assert "download again without resume" not in message
    retry = message.split("`")[1]
    assert retry == f"{command} --sub-job 7:training:0 --output-dir {shlex.quote(str(cache))}"
    # The quoted path survives a shell round trip, so the retry repairs the cached file.
    assert shlex.split(retry)[-1] == str(cache)


async def _run_stdout_warnings(calls):
    c = _client()
    c.get_job.return_value = _terminal_job()

    class _Downloads:
        def download_stdout_logs(self, job_id, output_dir, *, resume=False, on_chunk=None, sub_job_ids=None,
                                 stale_after_seconds=600):
            calls.append(stale_after_seconds)
            _write_stdout(output_dir, "7:training:0", "saved\n")
            return [{"sub_job_id": "7:training:0", "warnings": ["1 damaged chunk was skipped"]},
                    {"sub_job_id": "7:sampling:0", "warnings": ["sampling warning"]}]

    c.download_stdout_logs = _Downloads().download_stdout_logs
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        ok = await _wait(pilot, app, lambda: "warning: 1 damaged chunk was skipped" in (app.sub_title or ""))
        assert ok, "the download warning for the selected sub-job was not shown"
        assert "sampling warning" not in (app.sub_title or "")  # another sub-job's stays there
        app.screen._start_tail("7:sampling:0")
        ok = await _wait(pilot, app, lambda: "warning: sampling warning" in (app.sub_title or ""))
        assert ok, "the warnings did not follow the selected sub-job"
        await _settle(app, pilot)


def test_stdout_download_warnings_show_for_the_selected_sub_job():
    calls = []
    asyncio.run(_run_stdout_warnings(calls))
    # The TUI says itself when a job's output stopped, so the client's stall check is skipped.
    assert calls and all(value is None for value in calls)


async def _run_metrics_warning_shows_in_the_resources_header(tmp_path):
    c = _client()

    def download(job_id, output_dir, **_kwargs):
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=1)
        return [{"sub_job_id": "7:training:0", "warnings": ["2 chunk(s) could not be downloaded"]}]

    c.download_metrics.side_effect = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(170, 40)) as pilot:
        ok = await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        assert ok
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _wait(pilot, app, lambda: app.screen._metrics_warnings, tries=400)
        assert ok, "the metrics download's warning was not kept"
        header = str(app.screen.query_one("#resource-status", Static).render())
        assert "warning: 2 chunk(s) could not be downloaded" in header
        await _settle(app, pilot)


def test_metrics_download_warning_shows_in_the_resources_header(tmp_path):
    asyncio.run(_run_metrics_warning_shows_in_the_resources_header(tmp_path))


async def _run_hostile_service_text_never_reaches_the_terminal():
    # Everything the service sends is drawn: a job's reason, its model, a location, a log
    # line. None of it may carry an escape sequence (an OSC 52 would write the clipboard,
    # a CSI would clear the screen) or be read as markup.
    hostile = f"{ESC}]52;c;cGF5bG9hZA=={BEL}{ESC}[2Jboom‮[bold]x[/bold]"
    c = MagicMock()
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = [{"job_id": f"7{hostile}", "status": "RUNNING", "reason": hostile,
                                 "submitted_by": hostile,
                                 "sub_jobs": [{"job_type": "training", "model_name": hostile}]}]
    c.get_job.return_value = {"status": "RUNNING", "reason_details": hostile,
                              "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training",
                                            "model_name": hostile}]}
    c.tail_logs.return_value = {"entries": [{"_raw": hostile}], "next_cursor": "", "eof": True}
    app = CortexTrainingLogTUI(printable_client(c), poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        assert await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
        for stage in ("the job list", "a job's details"):
            if stage == "a job's details":
                await pilot.press("down")
                await pilot.press("enter")
                assert await _wait(pilot, app, lambda: not isinstance(app.screen, JobListScreen))
                await pilot.pause(0.3)
            drawn = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
            drawn += str(app.sub_title or "") + str(app.title or "")
            assert ESC not in drawn, f"an escape sequence was drawn on {stage}"
            assert BEL not in drawn and "cGF5bG9hZA" not in drawn, f"an OSC 52 payload reached {stage}"
            assert "‮" not in drawn, f"a direction override was drawn on {stage}"
        await _settle(app, pilot)


def test_hostile_service_text_never_reaches_the_terminal():
    asyncio.run(_run_hostile_service_text_never_reaches_the_terminal())


async def _run_the_keys_do_what_their_context_means():
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, _table = await _demo_list(app, pilot)
        # a has two bindings on the job list; check_action enables the one for the other list.
        assert screen.check_action("show_list", ("account",)) is True
        assert screen.check_action("show_list", ("schema",)) is False
        before = len(screen._job_by_item)
        await pilot.press("a")
        assert await _pause_until(pilot, lambda: screen._lister.account_wide, seconds=10.0), "a did not widen the list"
        assert screen.check_action("show_list", ("account",)) is False  # now it offers the way back
        await pilot.press("a")
        assert await _pause_until(pilot, lambda: not screen._lister.account_wide, seconds=10.0)
        assert await _pause_until(pilot, lambda: len(screen._job_by_item) == before, seconds=10.0)
        # l belongs to the details screen: on the list it is not a binding at all.
        assert screen.check_action("cycle_level", ()) is True  # not one of the list's actions
        await pilot.press("enter")
        assert await _pause_until(pilot, lambda: not isinstance(app.screen, JobListScreen), seconds=10.0)
        details = app.screen
        if hasattr(details, "query_one"):
            await pilot.pause(0.3)
        # On the details screen l cycles the minimum level, but only while Logs is showing.
        if hasattr(details, "_min_level"):
            details._active_tab = "resource"
            assert details.check_action("cycle_level", ()) is False, "l acts on the Resources tab"
            details._active_tab = "logs"
            assert details.check_action("cycle_level", ()) is True
            level = details._min_level
            await pilot.press("l")
            assert await _pause_until(pilot, lambda: details._min_level != level, seconds=10.0), "l did not cycle"
        await _settle(app, pilot)


def test_the_keys_do_what_their_context_means():
    asyncio.run(_run_the_keys_do_what_their_context_means())


class _GatedFake(FakeClient):
    """A fake whose job lists (schema and account-level) answer only once ``gate`` is set."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gate = threading.Event()
        self.gate.set()

    def list_jobs(self, *args, **kwargs):
        self.gate.wait(10)
        return super().list_jobs(*args, **kwargs)

    def list_jobs_page(self, **kwargs):
        self.gate.wait(10)
        return super().list_jobs_page(**kwargs)


async def _run_a_says_at_once_that_it_is_loading_all_schemas(in_flight):
    client = _GatedFake()
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, _table = await _demo_list(app, pilot)
        assert "Current schema: " in app.sub_title
        client.gate.clear()
        if in_flight:
            # A refresh is reading when a comes: a waits for it, and says so meanwhile.
            screen._refresh_jobs()
            assert screen._refreshing
        await pilot.press("a")
        await pilot.pause(0.1)
        # The header names the list a asked for at once, not the one it replaces.
        assert app.sub_title.startswith("loading all schemas…"), app.sub_title
        assert "Current schema" not in app.sub_title
        client.gate.set()
        assert await _pause_until(pilot, lambda: "all schemas" in app.sub_title
                                  and not app.sub_title.startswith("loading"), seconds=10.0), app.sub_title
        client.gate.clear()
        await pilot.press("a")
        await pilot.pause(0.1)
        assert app.sub_title.startswith("loading the current schema, "), app.sub_title
        client.gate.set()
        assert await _pause_until(pilot, lambda: "Current schema: " in app.sub_title, seconds=10.0), app.sub_title
        await _settle(app, pilot)


@pytest.mark.parametrize("in_flight", [False, True], ids=["idle", "refresh-in-flight"])
def test_a_says_at_once_that_it_is_loading_all_schemas(in_flight):
    asyncio.run(_run_a_says_at_once_that_it_is_loading_all_schemas(in_flight))


class _PagedFake(FakeClient):
    """A fake whose account-wide pages of the newest jobs answer one per ``release()`` while
    ``gated``, and whose active jobs' walk (read beside them) answers while ``active_gate`` is
    set. For the newest jobs' walk: ``fail_at`` is the page (1 is the first) that fails,
    ``unavailable_at`` the one that answers the list is not deployed, ``bad_token_at`` the one
    whose token the service can't open (once), ``running_at`` / ``unknown_at`` the one whose first
    job is running / of unknown status. ``schema_gate`` / ``schema_fails`` hold up or fail the
    configured schema's list, by either route it reads: the account-wide one scoped to that schema
    (pages naming this client's own database and schema), and ``list_jobs`` where that one is
    unavailable; while ``schema_paced``, its scoped pages answer one per ``release_schema()``.
    ``pages`` / ``active_pages`` / ``schema_pages`` count the pages each walk asked for."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gated = False
        self.fail_at = self.unavailable_at = self.bad_token_at = self.running_at = self.unknown_at = None
        self.schema_fails = False
        self.schema_gate = threading.Event()  # the configured schema's list answers while it is set
        self.schema_gate.set()
        self.active_gate = threading.Event()
        self.active_gate.set()
        self.pages = self.active_pages = self.schema_pages = 0
        self._permits = threading.Semaphore(0)
        self.schema_paced = False
        self._schema_permits = threading.Semaphore(0)

    def release(self, pages: int = 1) -> None:
        for _ in range(pages):
            self._permits.release()

    def release_schema(self, pages: int = 1) -> None:
        for _ in range(pages):
            self._schema_permits.release()

    def _configured(self, kwargs: dict) -> bool:
        """Whether a page is the configured schema's own list: scoped to this client's location."""
        database, schema = kwargs.get("database"), kwargs.get("schema")
        return (database is not None and schema is not None
                and (views.identifier_name(database), views.identifier_name(schema)) == self._location)

    def list_jobs(self, *args, **kwargs):
        self.schema_gate.wait(10)
        if self.schema_fails:
            raise ConnectionError("schema down")
        return super().list_jobs(*args, **kwargs)

    def list_jobs_page(self, **kwargs):
        if self._configured(kwargs):
            # The configured schema's own list: the same route, scoped to one schema. Its pages
            # answer under the gate that list has always had, not the all-schemas walk's.
            self.schema_pages += 1
            self.schema_gate.wait(10)
            if self.schema_paced:
                self._schema_permits.acquire(timeout=10)
            if self.schema_fails:
                raise ConnectionError("schema down")
            return super().list_jobs_page(**kwargs)
        if kwargs.get("status"):  # the active jobs' walk, read beside the newest
            self.active_pages += 1
            self.active_gate.wait(10)
            return super().list_jobs_page(**kwargs)
        self.pages += 1
        if self.gated:
            self._permits.acquire(timeout=10)
        page = int(kwargs.get("page_token") or 0) // self._ACCOUNT_PAGE_ROWS + 1
        if not kwargs.get("status"):
            if page == self.fail_at:
                raise ConnectionError("connection reset")
            if page == self.unavailable_at:
                raise AccountJobListUnavailable("the account-level list is not deployed",
                                                reason=AccountJobListUnavailable.NOT_DEPLOYED)
            if page == self.bad_token_at:
                self.bad_token_at = None
                raise InvalidPageToken("400 Invalid page_token")
        answer = super().list_jobs_page(**kwargs)
        if page == self.running_at and not kwargs.get("status") and answer["jobs"]:
            answer["jobs"][0]["status"] = "RUNNING"  # an older job still running: it shows under LIVE
        if page == self.unknown_at and not kwargs.get("status") and answer["jobs"]:
            answer["jobs"][0]["status"] = "UNKNOWN"  # live, but no status filter lists it
        return answer


def _info_text(screen) -> str:
    """The line above the list, as text."""
    info = screen._info
    return info.plain if not isinstance(info, tuple) else "  ".join(part.plain for part in [*info[0], info[1]])


def _all_at_once(client) -> tuple:
    """The account-wide list's job keys and scope as read all at once (no page shown before the last)."""
    listing = JobLister(client, account_wide=True).load(datetime.now(timezone.utc))
    return {views.job_key(job) for job in listing.jobs}, listing.scope


async def _run_all_schemas_shows_each_page_as_it_comes(refill):
    client = _PagedFake()
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 30)) as pilot:
        screen, table = await _demo_list(app, pilot)
        if refill:
            screen._PATCH_REMOVALS = -1  # each page fills the table again (as when many rows go)
        rows = client._ACCOUNT_PAGE_ROWS
        client.gated = True
        await pilot.press("a")
        client.release()
        # The first page shows before the others answer, saying how far the list is.
        assert await _pause_until(pilot, lambda: len(screen._all_jobs) == rows, seconds=10.0), len(screen._all_jobs)
        await pilot.pause(0.2)
        assert screen._scope.kind == ACCOUNT and screen._scope.loading and screen._refreshing
        # The active jobs, read beside the newest (at once, here): the active statuses' counts are whole;
        # the history, the unknown count and the LIVE row (an unknown job may still come) are not.
        assert await _pause_until(pilot, lambda: screen._scope.active_read, seconds=10.0)
        assert app.sub_title == f"all schemas  ·  loading older jobs… {rows} so far + every active job", (
            app.sub_title)
        live_part, history_part = _info_text(screen).split("│")
        assert "running so far" not in live_part and "in history so far" in history_part, _info_text(screen)
        assert all(row.endswith("…") for row in _group_rows(table).values()), _group_rows(table)

        client.release()
        assert await _pause_until(pilot, lambda: len(screen._all_jobs) == 2 * rows, seconds=10.0)
        await pilot.pause(0.2)
        # The reader's place: a row partway down, the view scrolled, a live job folded, a history one unfolded.
        live = next(j for j in screen._all_jobs if views.is_live(j) and screen._expandable(j))
        ended = next(j for j in screen._all_jobs if not views.is_live(j) and views.sub_jobs(j))
        screen._toggle(screen._job_key(live), False)
        screen._toggle(screen._job_key(ended), True)
        await pilot.pause(0.1)
        table.focus()
        await pilot.pause(0.1)
        top = int(table.max_scroll_y)  # scrolled to the end, the cursor near the view's top
        assert top > 0
        table.move_cursor(row=top + 3, animate=False, scroll=False)
        table.scroll_to(y=top, animate=False, immediate=True)
        await pilot.pause(0.2)
        assert table.scroll_y == top
        key = screen._highlighted()[0]
        line = table.cursor_row - round(table.scroll_y)

        client.release()
        assert await _pause_until(pilot, lambda: len(screen._all_jobs) == 3 * rows, seconds=10.0)
        await pilot.pause(0.3)
        assert screen._highlighted()[0] == key  # the same row is highlighted, on the same line
        assert table.cursor_row - round(table.scroll_y) == line, (table.cursor_row, table.scroll_y, line)
        assert not screen._is_expanded(live) and screen._is_expanded(ended)

        # A search while the list loads covers what has loaded, and says it is not all.
        await pilot.press("slash", *"status:done", "enter")
        assert await _pause_until(pilot, lambda: " so far for status:done" in _info_text(screen)), _info_text(screen)
        client.release(10)
        assert await _pause_until(pilot, lambda: not screen._refreshing, seconds=10.0)
        await pilot.pause(0.2)
        assert screen._filter == "status:done"
        assert "so far" not in _info_text(screen) and re.search(r"match(es)? for status:done", _info_text(screen))
        # Whole: what the list read all at once would have shown.
        keys, scope = _all_at_once(FakeClient())
        assert {screen._job_key(job) for job in screen._all_jobs} == keys and screen._scope == scope
        assert app.sub_title.endswith("  ·  " + scope_text(scope, location="")), app.sub_title
        assert not any(str(table.get_row_at(i)[0]).endswith("…") for i in range(table.row_count))
        await _settle(app, pilot)


@pytest.mark.parametrize("refill", [False, True], ids=["in-place", "refilled"])
def test_all_schemas_shows_each_page_as_it_comes(refill):
    asyncio.run(_run_all_schemas_shows_each_page_as_it_comes(refill))


async def _run_a_cursor_on_a_group_row_stays_on_it_as_pages_come():
    client = _PagedFake()
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 30)) as pilot:
        screen, table = await _demo_list(app, pilot)
        client.gated = True
        await pilot.press("a")
        client.release()
        assert await _pause_until(pilot, lambda: screen._scope.loading, seconds=10.0)
        await pilot.pause(0.2)
        table.focus()
        table.move_cursor(row=table.get_row_index("hdr-history"), animate=False)
        await pilot.pause(0.2)
        line = table.cursor_row - round(table.scroll_y)
        for pages in (2, 3):
            client.release()
            assert await _pause_until(pilot, lambda: len(screen._all_jobs) == pages * client._ACCOUNT_PAGE_ROWS,
                                      seconds=10.0)
            await pilot.pause(0.3)
            assert screen._highlighted()[0] == "hdr-history", screen._highlighted()
            assert table.cursor_row - round(table.scroll_y) == line
        client.release(10)
        await _settle(app, pilot)


async def _run_a_list_the_reader_has_not_moved_stays_at_its_top_as_live_jobs_come():
    client = _PagedFake()
    client.running_at = 3  # an older job still running comes on page 3, above the cursor
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, table = await _demo_list(app, pilot)
        client.gated = True
        await pilot.press("a")
        client.release()
        assert await _pause_until(pilot, lambda: screen._scope.loading, seconds=10.0)
        # The list puts the cursor on a history job (as it keeps the one a schema list had).
        first_ended = next(j for j in views.sort_job_list(screen._all_jobs, "created", True) if not views.is_live(j))
        screen._render_jobs(keep=screen._job_key(first_ended))
        await pilot.pause(0.3)
        key = screen._highlighted()[0]
        assert table.scroll_y == 0 and screen._job_by_item.get(key) == screen._job_key(first_ended)
        client.release(2)
        assert await _pause_until(pilot, lambda: len(screen._all_jobs) == 3 * client._ACCOUNT_PAGE_ROWS,
                                  seconds=10.0)
        await pilot.pause(0.3)
        late = next(job for job in screen._all_jobs if views.is_live(job) and screen._job_key(job) not in {
            screen._job_key(j) for j in views.sort_job_list(screen._all_jobs, "created", True)[:10]})
        assert table.get_row_index(screen._row_key(screen._job_key(late))) < table.cursor_row  # it came above
        assert table.max_scroll_y > 0  # and the list is taller than the view
        assert table.scroll_y == 0 and screen._highlighted()[0] == key  # LIVE, all of it, still in view
        client.release(10)
        assert await _pause_until(pilot, lambda: not screen._refreshing, seconds=10.0)
        await pilot.pause(0.3)
        assert table.scroll_y == 0 and screen._highlighted()[0] == key
        await _settle(app, pilot)


def test_a_list_the_reader_has_not_moved_stays_at_its_top_as_live_jobs_come():
    asyncio.run(_run_a_list_the_reader_has_not_moved_stays_at_its_top_as_live_jobs_come())


async def _run_a_reader_who_moved_keeps_their_line_on_every_page_after(how):
    client = _PagedFake()
    client.running_at = 3  # a running job comes on page 3, above the cursor
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, table = await _demo_list(app, pilot)
        client.gated = True
        await pilot.press("a")
        client.release()
        assert await _pause_until(pilot, lambda: screen._scope.loading, seconds=10.0)
        await pilot.pause(0.2)
        table.focus()
        table.move_cursor(row=table.get_row_index("hdr-history") + 2, animate=False)  # the reader moves
        await pilot.pause(0.2)
        if how == "fold":  # and folds the job there, which draws the list again with the cursor on it
            screen._toggle(screen._highlighted()[1], True)
            await pilot.pause(0.2)
        key, line = screen._highlighted()[0], table.cursor_row - round(table.scroll_y)
        assert table.scroll_y == 0
        for pages in (2, 3):  # page 2 draws the list with the cursor on the reader's row; page 3 too
            client.release()
            assert await _pause_until(pilot, lambda: len(screen._all_jobs) == pages * client._ACCOUNT_PAGE_ROWS,
                                      seconds=10.0)
            await pilot.pause(0.3)
            assert screen._highlighted()[0] == key
            assert table.cursor_row - round(table.scroll_y) == line, (pages, table.cursor_row, table.scroll_y, line)
        client.release(10)
        await _settle(app, pilot)


@pytest.mark.parametrize("how", ["moved", "fold"])
def test_a_reader_who_moved_keeps_their_line_on_every_page_after(how):
    asyncio.run(_run_a_reader_who_moved_keeps_their_line_on_every_page_after(how))


async def _run_e_unfolds_the_jobs_of_later_pages_too_and_then_folds_them_all():
    client = _PagedFake()
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, table = await _demo_list(app, pilot)
        client.gated = True
        await pilot.press("a")
        client.release()
        assert await _pause_until(pilot, lambda: screen._scope.loading, seconds=10.0)
        table.focus()
        await pilot.press("e")

        def expandable():
            return [job for job in screen._all_jobs if screen._expandable(job)]

        assert all(screen._is_expanded(job) for job in expandable())
        client.release()
        assert await _pause_until(pilot, lambda: len(screen._all_jobs) == 2 * client._ACCOUNT_PAGE_ROWS,
                                  seconds=10.0)
        await pilot.pause(0.2)
        assert any(not views.is_live(job) for job in expandable())  # history jobs that unfold came
        assert all(screen._is_expanded(job) for job in expandable())  # unfolded, as e asked
        await pilot.press("e")
        assert not any(screen._is_expanded(job) for job in expandable())  # every one folds
        client.release(10)
        assert await _pause_until(pilot, lambda: not screen._refreshing, seconds=10.0)
        assert not any(screen._is_expanded(job) for job in expandable())
        await _settle(app, pilot)


def test_e_unfolds_the_jobs_of_later_pages_too_and_then_folds_them_all():
    asyncio.run(_run_e_unfolds_the_jobs_of_later_pages_too_and_then_folds_them_all())


async def _run_e_holds_for_its_list_and_jobs_a_refresh_adds_start_as_usual(fold):
    from cortex_training.tui.account import Listing
    from cortex_training.tui.account import ListScope
    app = CortexTrainingLogTUI(_PagedFake(), poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, table = await _demo_list(app, pilot)
        table.focus()
        await pilot.press("e")  # every job unfolded
        if fold:
            await pilot.press("e")  # then every job folded
        listed = [job for job in screen._all_jobs if screen._expandable(job)]
        assert all(screen._is_expanded(job) is not fold for job in listed)
        live = next(job for job in listed if views.is_live(job))
        ended = next(job for job in listed if not views.is_live(job))
        # A refresh lands with a live job and an ended one new since.
        new_live = {**live, "job_id": "new-live-job", "created_at": live.get("created_at")}
        new_ended = {**ended, "job_id": "new-ended-job"}
        refreshed = [*screen._all_jobs, new_live, new_ended]
        screen._on_jobs(Listing(refreshed, ListScope(SCHEMA)), None, False, screen._load_gen)
        await pilot.pause(0.2)
        assert all(screen._is_expanded(job) is not fold for job in listed)  # e's choice holds for them
        by_id = {views.job_id_of(job): job for job in screen._all_jobs}
        assert screen._is_expanded(by_id["new-live-job"]) and not screen._is_expanded(by_id["new-ended-job"])
        assert screen._all_folded == {False: None, True: None}
        await _settle(app, pilot)


@pytest.mark.parametrize("fold", [False, True], ids=["unfolded", "folded"])
def test_e_holds_for_its_list_and_jobs_a_refresh_adds_start_as_usual(fold):
    asyncio.run(_run_e_holds_for_its_list_and_jobs_a_refresh_adds_start_as_usual(fold))


async def _run_a_that_stops_the_pages_says_at_once_what_the_list_on_screen_is():
    client = _PagedFake()
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, table = await _demo_list(app, pilot)
        client.gated = True
        await pilot.press("a")
        client.release(2)
        assert await _pause_until(pilot, lambda: len(screen._all_jobs) == 2 * client._ACCOUNT_PAGE_ROWS,
                                  seconds=10.0)
        await pilot.pause(0.2)
        client.schema_gate.clear()  # the configured schema's list takes its time
        await pilot.press("a")
        await pilot.pause(0.2)
        assert app.sub_title == (f"loading the current schema, {screen._schema_label()}…  ·  "
                                 "showing the partial all-schemas list"), app.sub_title
        assert "so far" not in _info_text(screen)
        assert not any(str(table.get_row_at(i)[0]).endswith("…") for i in range(table.row_count))
        client.schema_gate.set()
        # The schema's list arrives a page at a time, its first pages already SCHEMA-scoped.
        assert await _pause_until(pilot, lambda: screen._scope.kind == SCHEMA and not screen._scope.loading,
                                  seconds=10.0)
        assert app.sub_title.endswith("Current schema: " + screen._schema_label()), app.sub_title
        client.release(10)
        await _settle(app, pilot)


def test_a_that_stops_the_pages_says_at_once_what_the_list_on_screen_is():
    asyncio.run(_run_a_that_stops_the_pages_says_at_once_what_the_list_on_screen_is())


def _group_rows(table) -> dict:
    """The LIVE and HISTORY rows' text, by group."""
    rows = [str(table.get_row_at(i)[0]) for i in range(table.row_count)]
    return {row.split()[0]: row for row in rows if row.startswith(("LIVE", "HISTORY"))}


async def _run_an_unknown_job_keeps_the_live_count_open_until_the_newest_jobs_are_in():
    client = _PagedFake()
    client.unknown_at = 3  # an unknown job comes on page 3: live, but not among the active jobs
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, table = await _demo_list(app, pilot)
        rows = client._ACCOUNT_PAGE_ROWS
        client.gated = True
        await pilot.press("a")
        client.release()
        assert await _pause_until(pilot, lambda: screen._scope.loading and screen._scope.active_read, seconds=10.0)
        await pilot.pause(0.2)

        def unknown() -> int:  # live ones (one the control plane has no record of has ended)
            return sum(views.is_unknown(job) and views.is_live(job) for job in screen._all_jobs)

        live_part, _history = _info_text(screen).split("│")
        # The active statuses are whole; the unknown count (and so LIVE) may still grow.
        assert "running so far" not in live_part and live_part.count("so far") == 1, live_part
        assert f"{unknown()} unknown so far" in live_part, live_part
        assert _group_rows(table)["LIVE"].endswith("…")
        before = unknown()
        client.release(2)
        assert await _pause_until(pilot, lambda: len(screen._all_jobs) == 3 * rows, seconds=10.0)
        await pilot.pause(0.2)
        assert unknown() == before + 1  # page 3 brought one, after the active jobs were all listed
        live_part, _history = _info_text(screen).split("│")
        assert f"{unknown()} unknown so far" in live_part and live_part.count("so far") == 1
        assert _group_rows(table)["LIVE"].endswith("…")
        client.release(10)
        assert await _pause_until(pilot, lambda: not screen._refreshing, seconds=10.0)
        await pilot.pause(0.2)
        assert "so far" not in _info_text(screen) and "unknown" in _info_text(screen)
        assert not any(row.endswith("…") for row in _group_rows(table).values())
        await _settle(app, pilot)


def test_an_unknown_job_keeps_the_live_count_open_until_the_newest_jobs_are_in():
    asyncio.run(_run_an_unknown_job_keeps_the_live_count_open_until_the_newest_jobs_are_in())


async def _run_a_slow_active_jobs_walk_after_the_newest_are_in(monkeypatch, then):
    from cortex_training.tui import account
    monkeypatch.setattr(account, "ACCOUNT_LIST_MAX_PAGES", 2)  # the newest jobs' walk stops after 2 pages
    client = _PagedFake()
    client.active_gate.clear()  # the active jobs' walk takes its time
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, table = await _demo_list(app, pilot)
        newest = 2 * client._ACCOUNT_PAGE_ROWS
        await pilot.press("a")
        assert await _pause_until(pilot, lambda: screen._scope.loading == "active", seconds=10.0), app.sub_title
        await pilot.pause(0.2)
        # The history is whole; only the live jobs may grow, and the header says what is still read.
        assert app.sub_title == f"all schemas  ·  newest {newest}; loading older active jobs…", app.sub_title
        live_part, history_part = _info_text(screen).split("│")
        assert "so far" in live_part and "so far" not in history_part, _info_text(screen)
        groups = _group_rows(table)
        assert groups["LIVE"].endswith("…") and not groups["HISTORY"].endswith("…"), groups
        await pilot.press("r")
        said = [str(note.message) for note in app._notifications]
        assert any(m.startswith("Still loading older active jobs (") for m in said), said
        if then == "wait":
            client.active_gate.set()
            assert await _pause_until(pilot, lambda: not screen._refreshing, seconds=10.0)
            assert app.sub_title.endswith(f"first 2 pages: {newest} jobs + every active job (more may exist)"), (
                app.sub_title)
        else:  # a, while the load waits for the active jobs: it stops, and nothing of it comes after
            await pilot.press("a")
            assert await _pause_until(pilot, lambda: screen._scope.kind == SCHEMA, seconds=10.0), app.sub_title
            client.active_gate.set()
            await pilot.pause(0.5)
            # ACCOUNT_LIST_MAX_PAGES is 2 here, and the schema's own list is read a page at a time
            # too, so it stops short as well and says what it holds after naming the schema.
            assert await _pause_until(pilot, lambda: not screen._refreshing, seconds=10.0)
            assert screen._scope.kind == SCHEMA
            assert app.sub_title.endswith(f"Current schema: {screen._schema_label()}, first 2 pages: "
                                          f"{newest} jobs + every active job (more may exist)"), app.sub_title
            assert client.active_pages == 1  # the walk asked for no page after the one in flight
        await _settle(app, pilot)


@pytest.mark.parametrize("then", ["wait", "a"])
def test_a_slow_active_jobs_walk_after_the_newest_are_in(monkeypatch, then):
    asyncio.run(_run_a_slow_active_jobs_walk_after_the_newest_are_in(monkeypatch, then))


def test_a_cursor_on_a_group_row_stays_on_it_as_pages_come():
    asyncio.run(_run_a_cursor_on_a_group_row_stays_on_it_as_pages_come())


async def _run_a_fresh_list_of_pages_shows_no_earlier_failure(how):
    client = _PagedFake()
    client.schema_fails = how == "r"
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        if how == "a":  # a refresh of the configured schema's list fails, then a
            screen, _table = await _demo_list(app, pilot)
            notice = screen.query_one("#listnotice", Static)
            client.schema_fails = True
            screen._refresh_jobs(manual=True)
            assert await _pause_until(pilot, lambda: notice.display, seconds=10.0)
            client.gated = True
            await pilot.press("a")
        else:  # the first load fails, and r reads all schemas (the list's own, shared lister's)
            assert await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen)
                               and app.screen._list_error, tries=400)
            screen = app.screen
            notice = screen.query_one("#listnotice", Static)
            assert "list_jobs failed" in app.screen.query_one("#listinfo", Static).render().plain
            screen._lister.account_wide = True
            client.gated = True
            await pilot.press("r")
        client.release()
        assert await _pause_until(pilot, lambda: screen._scope is not None and screen._scope.loading,
                                  seconds=10.0), app.sub_title
        await pilot.pause(0.2)
        assert not notice.display and screen._list_error is None
        assert not screen.query_one("#listinfo", Static).has_class("error")
        assert app.sub_title.startswith("all schemas  ·  loading older jobs…"), app.sub_title
        client.release(10)
        assert await _pause_until(pilot, lambda: not screen._refreshing, seconds=10.0)
        assert not notice.display and "failed" not in app.sub_title
        await _settle(app, pilot)


@pytest.mark.parametrize("how", ["a", "r"])
def test_a_fresh_list_of_pages_shows_no_earlier_failure(how):
    asyncio.run(_run_a_fresh_list_of_pages_shows_no_earlier_failure(how))


async def _run_a_failure_of_the_schema_list_after_a_stopped_the_pages_is_the_schema_lists():
    client = _PagedFake()
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, _table = await _demo_list(app, pilot)
        client.gated = True
        await pilot.press("a")
        client.release()
        assert await _pause_until(pilot, lambda: screen._scope.loading, seconds=10.0)
        client.schema_fails = True
        await pilot.press("a")  # back to the configured schema, whose list fails
        assert await _pause_until(pilot, lambda: not screen._refreshing and screen._list_error, seconds=10.0)
        await pilot.pause(0.2)
        assert app.sub_title.startswith("refresh failed "), app.sub_title
        assert "schema down" in screen._notice.plain and "read before it" not in screen._notice.plain
        assert screen.check_action("show_list", ("account",)) is True  # the footer offers all schemas
        client.release(10)
        await _settle(app, pilot)


def test_a_failure_of_the_schema_list_after_a_stopped_the_pages_is_the_schema_lists():
    asyncio.run(_run_a_failure_of_the_schema_list_after_a_stopped_the_pages_is_the_schema_lists())


async def _run_an_account_list_unavailable_after_its_first_pages_says_so():
    client = _PagedFake()
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, _table = await _demo_list(app, pilot)
        schema_jobs = len(screen._all_jobs)
        client.unavailable_at = 3
        await pilot.press("a")
        assert await _pause_until(pilot, lambda: "Current schema only: " in app.sub_title, seconds=10.0), (
            app.sub_title)
        assert "no account-wide list yet" in app.sub_title
        # The fallback lists exactly list_jobs' jobs. That route sends no created_after, so it
        # also holds the schema's jobs older than six months, which the scoped walk leaves out.
        listed = {views.job_id_of(job) for job in client.list_jobs()}
        assert screen._scope.kind == SCHEMA and {views.job_id_of(job) for job in screen._all_jobs} == listed
        assert len(listed) > schema_jobs
        assert screen.check_action("show_list", ("account",)) is True
        assert any("Can't show all schemas" in str(n.message) for n in app._notifications)
        await _settle(app, pilot)


def test_an_account_list_unavailable_after_its_first_pages_says_so():
    asyncio.run(_run_an_account_list_unavailable_after_its_first_pages_says_so())


async def _first_schema_page(app, pilot, client):
    """The configured schema's list as its first page shows, the rest still held back: the screen."""
    client.release_schema()  # its first page only
    assert await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen)
                       and len(app.screen._all_jobs) == client._ACCOUNT_PAGE_ROWS, tries=400)
    await pilot.pause(0.2)
    return app.screen


async def _run_the_configured_schemas_list_shows_its_pages_as_they_come():
    client = _PagedFake()
    client.schema_paced = True
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen = await _first_schema_page(app, pilot, client)
        # The first page shows while the rest are read, and the header says how far it is.
        assert screen._refreshing and screen._scope.kind == SCHEMA and screen._scope.loading
        assert len(screen._all_jobs) == client._ACCOUNT_PAGE_ROWS
        assert app.sub_title == (f"Current schema: {screen._schema_label()}, "
                                 f"loading older jobs… {client._ACCOUNT_PAGE_ROWS} so far"), app.sub_title
        client.release_schema(20)
        assert await _pause_until(pilot, lambda: not screen._refreshing, seconds=10.0)
        assert app.sub_title.endswith(f"Current schema: {screen._schema_label()}"), app.sub_title
        assert len(screen._all_jobs) > client._ACCOUNT_PAGE_ROWS and client.schema_pages > 2
        await _settle(app, pilot)


def test_the_configured_schemas_list_shows_its_pages_as_they_come():
    asyncio.run(_run_the_configured_schemas_list_shows_its_pages_as_they_come())


async def _run_a_while_the_configured_schemas_pages_load_stops_them():
    client = _PagedFake()
    client.schema_paced = True
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen = await _first_schema_page(app, pilot, client)
        client.gated = True  # the all-schemas list's pages wait too, so its header can be read
        await pilot.press("a")  # without waiting for the rest of this schema's pages
        await pilot.pause(0.2)
        assert app.sub_title == "loading all schemas…  ·  showing the partial current-schema list", app.sub_title
        asked = client.schema_pages
        client.release_schema(20)  # the schema's pages in flight answer; no more are asked for
        client.release(20)
        assert await _pause_until(pilot, lambda: screen._scope.kind == ACCOUNT and not screen._refreshing,
                                  seconds=10.0), app.sub_title
        await pilot.pause(0.3)
        assert client.schema_pages == asked
        await _settle(app, pilot)


def test_a_while_the_configured_schemas_pages_load_stops_them():
    asyncio.run(_run_a_while_the_configured_schemas_pages_load_stops_them())


async def _run_pages_read_again_from_the_start_keep_the_list_and_the_cursor():
    client = _PagedFake()
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 30)) as pilot:
        screen, table = await _demo_list(app, pilot)
        rows = client._ACCOUNT_PAGE_ROWS
        client.gated, client.bad_token_at = True, 3
        await pilot.press("a")
        client.release(2)
        assert await _pause_until(pilot, lambda: len(screen._all_jobs) == 2 * rows, seconds=10.0)
        await pilot.pause(0.2)
        shown = []
        apply = screen._on_partial
        screen._on_partial = lambda gen, listing: (shown.append(len(listing.jobs)), apply(gen, listing))
        table.focus()
        table.move_cursor(row=table.row_count - 2, animate=False)
        await pilot.pause(0.2)
        key, line = screen._highlighted()[0], table.cursor_row - round(table.scroll_y)
        client.release(4)  # page 3's token fails: pages 1 and 2 again, then page 3
        assert await _pause_until(pilot, lambda: len(screen._all_jobs) == 3 * rows, seconds=10.0)
        await pilot.pause(0.3)
        assert min(shown) == 2 * rows  # the list never went back to its first page
        assert screen._highlighted()[0] == key and table.cursor_row - round(table.scroll_y) == line
        client.release(10)
        assert await _pause_until(pilot, lambda: not screen._refreshing, seconds=10.0)
        keys, scope = _all_at_once(FakeClient())
        assert {screen._job_key(job) for job in screen._all_jobs} == keys and screen._scope == scope
        await _settle(app, pilot)


def test_pages_read_again_from_the_start_keep_the_list_and_the_cursor():
    asyncio.run(_run_pages_read_again_from_the_start_keep_the_list_and_the_cursor())


async def _run_a_or_r_while_all_schemas_loads_supersedes_it_cleanly():
    client = _PagedFake()
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, _table = await _demo_list(app, pilot)
        schema_jobs = len(screen._all_jobs)
        client.gated = True
        await pilot.press("a")
        client.release()
        assert await _pause_until(pilot, lambda: screen._scope.kind == ACCOUNT and screen._scope.loading,
                                  seconds=10.0)
        await pilot.pause(0.2)
        asked = client.pages  # the first page, and the second one waiting
        # r and the timed refresh wait for the load: no second walk starts.
        await pilot.press("r")
        screen._refresh_tick()
        await pilot.pause(0.2)
        assert client.pages == asked and screen._scope.loading
        said = [str(note.message) for note in app._notifications]
        assert any(m.startswith("Still loading older jobs (") and m.endswith("r works once it's done.")
                   for m in said), said
        # a goes back to the configured schema at once; the walk stops at its page in flight.
        await pilot.press("a")
        assert await _pause_until(pilot, lambda: screen._scope.kind == SCHEMA, seconds=10.0), app.sub_title
        client.release(10)
        await pilot.pause(0.5)
        assert client.pages == asked  # it asked for no page after
        assert screen._scope.kind == SCHEMA and len(screen._all_jobs) == schema_jobs  # its pages never shown
        assert app.sub_title.endswith("Current schema: " + screen._schema_label()), app.sub_title
        # a again: a new walk, whole in the end.
        client.gated = False
        await pilot.press("a")
        assert await _pause_until(pilot, lambda: screen._scope.kind == ACCOUNT and not screen._refreshing,
                                  seconds=10.0)
        keys, scope = _all_at_once(FakeClient())
        assert {screen._job_key(job) for job in screen._all_jobs} == keys and screen._scope == scope
        await _settle(app, pilot)


def test_a_or_r_while_all_schemas_loads_supersedes_it_cleanly():
    asyncio.run(_run_a_or_r_while_all_schemas_loads_supersedes_it_cleanly())


async def _run_a_page_that_fails_keeps_the_pages_read_and_r_reads_them_again():
    client = _PagedFake()
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, _table = await _demo_list(app, pilot)
        rows = client._ACCOUNT_PAGE_ROWS
        client.fail_at = 3
        await pilot.press("a")
        assert await _pause_until(pilot, lambda: not screen._refreshing and screen._scope.kind == ACCOUNT,
                                  seconds=10.0), app.sub_title
        await pilot.pause(0.2)
        assert len(screen._all_jobs) == 2 * rows  # the two pages read stay
        assert app.sub_title.startswith("load failed "), app.sub_title
        # The active jobs, read beside the newest pages, stay listed: none is missing.
        assert app.sub_title.endswith(f"newest {2 * rows} + every active job (the rest didn't load; r retries)"), (
            app.sub_title)
        notice = screen.query_one("#listnotice", Static)
        assert notice.display and f"showing the {2 * rows} jobs read before it" in screen._notice.plain
        client.fail_at = None
        await pilot.press("r")
        assert await _pause_until(pilot, lambda: not screen._refreshing and screen._list_error is None,
                                  seconds=10.0), app.sub_title
        keys, scope = _all_at_once(FakeClient())
        assert {screen._job_key(job) for job in screen._all_jobs} == keys and screen._scope == scope
        assert not notice.display
        await _settle(app, pilot)


def test_a_page_that_fails_keeps_the_pages_read_and_r_reads_them_again():
    asyncio.run(_run_a_page_that_fails_keeps_the_pages_read_and_r_reads_them_again())


async def _run_a_refresh_of_the_whole_list_shows_it_until_the_next_is_whole():
    client = _PagedFake()
    app = CortexTrainingLogTUI(client, poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, _table = await _demo_list(app, pilot)
        await pilot.press("a")
        assert await _pause_until(pilot, lambda: screen._scope.kind == ACCOUNT and not screen._refreshing,
                                  seconds=10.0)
        whole = len(screen._all_jobs)
        client.gated = True
        screen._refresh_tick()  # the timed refresh
        client.release(2)
        await pilot.pause(0.5)
        assert screen._refreshing and len(screen._all_jobs) == whole and not screen._scope.loading
        assert "so far" not in app.sub_title
        client.release(10)
        assert await _pause_until(pilot, lambda: not screen._refreshing, seconds=10.0)
        assert len(screen._all_jobs) == whole and not screen._scope.loading
        await _settle(app, pilot)


def test_a_refresh_of_the_whole_list_shows_it_until_the_next_is_whole():
    asyncio.run(_run_a_refresh_of_the_whole_list_shows_it_until_the_next_is_whole())


async def _run_a_keeps_why_all_schemas_are_unavailable_in_the_header(reason, why):
    app = CortexTrainingLogTUI(FakeClient(account_list=reason), poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        screen, _table = await _demo_list(app, pilot)
        await pilot.press("a")
        assert await _pause_until(pilot, lambda: "Current schema only: " in app.sub_title, seconds=10.0), (
            app.sub_title)
        assert why in app.sub_title
        assert screen.check_action("show_list", ("account",)) is True  # the footer offers a again
        # It stays after the notice has gone, through later refreshes, until the next a.
        screen._refresh_jobs(manual=True)
        assert await _pause_until(pilot, lambda: not screen._refreshing, seconds=10.0)
        assert why in app.sub_title
        await _settle(app, pilot)


@pytest.mark.parametrize("reason, why", [
    (AccountJobListUnavailable.DISABLED, "account-wide list not enabled"),
    (AccountJobListUnavailable.NOT_DEPLOYED, "no account-wide list yet"),
])
def test_a_keeps_why_all_schemas_are_unavailable_in_the_header(reason, why):
    asyncio.run(_run_a_keeps_why_all_schemas_are_unavailable_in_the_header(reason, why))


async def _run_the_group_rows_count_the_jobs_under_them():
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    async with app.run_test(size=(140, 44)) as pilot:
        screen, table = await _demo_list(app, pilot)
        await pilot.pause(0.3)
        at = [str(column.label).strip() for column in table.ordered_columns].index("Job ID")
        rows = [(row.key.value, table.get_row_at(i)[at].plain) for i, row in enumerate(table.ordered_rows)]
        # Each group row says how many jobs follow it; their sub-job rows must not be counted.
        counts, group, seen = {}, None, 0
        for key, text in rows:
            if text.startswith(("LIVE  ", "HISTORY  ")):
                if group:
                    counts[group] = seen
                group, seen = text, 0
            elif key in screen._job_by_item:
                seen += 1
        counts[group] = seen
        assert len(counts) == 2, counts
        for header, under in counts.items():
            said = int(header.split()[1].replace(",", ""))
            assert said == under, f"{header} has {under} job rows under it"
        await _settle(app, pilot)


def test_the_group_rows_count_the_jobs_under_them():
    asyncio.run(_run_the_group_rows_count_the_jobs_under_them())


def test_list_times_read_at_a_glance_and_say_how_long_they_last():
    assert views.list_age(None)[0] == "—"
    assert views.list_age(-30) == ("just now", 90)  # clock skew: until a minute after it
    assert views.list_age(-7300)[0] == "in 2h"
    assert views.list_age(20) == ("just now", 40)
    assert views.list_age(125) == ("2m ago", 55)
    assert views.list_age(3 * 3600 + 10) == ("3h ago", 3590)
    assert views.list_age(45 * 86400) == ("45d ago", 86400)
    assert all(len(views.list_age(days * 86400)[0]) <= 8 for days in (0, 9, 99, 183))  # the column is 8 cells
    assert views.list_span(None)[0] == views.list_span(-1)[0] == "—"
    assert views.list_span(40.25) == ("40s", 0.75)
    assert views.list_span(12 * 60 + 5) == ("12m", 55)
    assert views.list_span(3 * 3600 + 5 * 60) == ("3h 05m", 60)
    assert views.list_span(2 * 86400 + 4 * 3600 + 30) == ("2d 4h", 3570)


def test_job_list_rows_say_what_their_columns_don_t():
    from cortex_training.tui.job_list import _status_chips

    jid = "a" * 36
    assert str(JobListScreen._name_cell({"job_id": jid})) == jid  # whole: a copy of the list has it so
    assert str(JobListScreen._name_cell({"job_id": jid}, False)) == f"▸ {jid}"
    # Drawn (.plain) with the theme's glyph; copied (str) as the word alone.
    cancelled = JobListScreen._status_cell("JOB_STATE_CANCELLED")
    running = JobListScreen._status_cell("RUNNING", glyph_only=True)
    assert (cancelled.plain, str(cancelled)) == ("− CANCELLED", "CANCELLED")
    assert (running.plain, str(running)) == ("●", "RUNNING")

    job = {"job_id": "j", "status": "RUNNING", "sub_jobs": [
        {"sub_job_id": "j:training:0", "job_type": "training", "status": "RUNNING", "model_name": "Qwen/Qwen3-8B"},
        {"sub_job_id": "j:sampling:0", "job_type": "sampling", "status": "PLACING", "model_name": "Qwen/Qwen3-0.6B"}]}
    same = JobListScreen._sub_cells(job, job["sub_jobs"][0], False)
    other = JobListScreen._sub_cells(job, job["sub_jobs"][1], True)
    assert str(same["Job ID"]) == "  ├ training #0" and same["Status"].plain == "●"
    assert "Type" not in same  # its label has it
    assert str(other["Job ID"]) == "  └ sampling #0" and other["Status"].plain == "◔ PLACING"
    # The job's row says "Qwen3-8B +1": each sub-job row says which; of a one-model job, none does.
    assert str(same["Model"]) == "Qwen3-8B" and str(other["Model"]) == "Qwen3-0.6B"
    one_model = {**job, "sub_jobs": [{**sj, "model_name": "Qwen/Qwen3-8B"} for sj in job["sub_jobs"]]}
    assert str(JobListScreen._sub_cells(one_model, one_model["sub_jobs"][1], True)["Model"]) == ""
    untold = {"sub_job_id": "j:training:1", "job_type": "training"}  # no status of its own
    assert str(JobListScreen._sub_cells(job, untold, True)["Status"]) == ""
    assert JobListScreen._sub_cells({**job, "status": "UNKNOWN"}, untold, True)["Status"].plain == "?"

    # Of several sub-jobs each says its GPUs; a job's only sub-job does where the job's row doesn't.
    assert str(same["GPUs"]) == "—"
    lone = {"job_id": "o", "status": "RUNNING", "hardware": "H200", "sub_jobs": [
        {"sub_job_id": "o:training:0", "job_type": "training", "status": "RUNNING",
         "training_config": {"n_gpus": 8}}]}
    lone_cells = JobListScreen._sub_cells(lone, lone["sub_jobs"][0], True)
    assert str(lone_cells["Job ID"]) == "  └ training #0" and lone_cells["Status"].plain == "●"
    assert views.job_gpu_cell(lone) == "8" and str(lone_cells["GPUs"]) == ""  # the job's row says "8 H200"

    # Every live job starts unfolded, one of a single sub-job too; history starts folded.
    assert JobListScreen._starts_unfolded(job)
    assert JobListScreen._starts_unfolded({**job, "sub_jobs": job["sub_jobs"][:1]})
    assert JobListScreen._starts_unfolded({"job_id": "u", "status": "UNKNOWN"})
    assert not JobListScreen._starts_unfolded({**job, "status": "FAILED"})

    from rich.console import Console

    name, rule = JobListScreen._header_cells("HISTORY", 1234, ("Job ID", "Status"))
    assert str(name) == "HISTORY  1,234"
    console = Console(width=80)
    for width in (14, 20):  # as wide as its column is drawn, a column dragged wider too
        line, = console.render_lines(rule, console.options.update_width(width), pad=False)
        assert "".join(segment.text for segment in line) == "─" * width
    assert str(rule) == ""  # drawn, but not in a copy of the list

    live = [{"status": s} for s in ("RUNNING", "queued", "PENDING", "creating", "UNKNOWN", "RUNNING")]
    assert _status_chips(live, 21).plain == "● 2 running   ◔ 1 placing   ○ 2 pending   ? 1 unknown  │  21 in history"
    assert _status_chips([], 0).plain == "no live jobs  │  0 in history"


def test_job_list_ages_are_rebuilt_only_once_they_read_differently():
    from cortex_training.tui.job_list import _ListLayout

    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ended = {"job_id": "e", "status": "FAILED", "created_at": "2025-12-01T12:00:00Z",
             "updated_at": "2025-12-01T13:00:00Z"}
    live = {"job_id": "l", "status": "RUNNING", "created_at": "2026-01-01T11:57:30Z",
            "updated_at": "2026-01-01T11:59:50Z"}
    screen = JobListScreen(MagicMock(), now=now)
    layout = _ListLayout(tuple(JobListScreen._COLUMNS))
    # What each cell draws (str() is what a copy of the list has).
    texts = lambda rows: {label: cell.plain for label, cell in zip(layout.labels, rows.row)}  # noqa: E731

    first_ended, first_live = screen._job_rows(ended, layout), screen._job_rows(live, layout)
    assert texts(first_ended)["Created"] == "31d ago" and texts(first_ended)["Duration"] == "1h 00m"
    assert texts(first_live)["Created"] == "2m ago" and texts(first_live)["Duration"] == "2m"
    assert texts(first_live)["Updated"] == "just now"
    screen._now = now + timedelta(seconds=20)  # neither reads differently yet
    assert screen._job_rows(ended, layout) is first_ended and screen._job_rows(live, layout) is first_live
    screen._now = now + timedelta(seconds=40)  # the live job's third minute
    again = screen._job_rows(live, layout)
    assert again is not first_live and texts(again)["Created"] == "3m ago" and texts(again)["Duration"] == "3m"
    assert screen._job_rows(ended, layout) is first_ended  # its duration never changes, its age by the day


def test_a_copy_of_the_job_list_has_whole_times_and_status_words():
    from cortex_training.tui.format import format_created_at

    created = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    updated = (datetime.now(timezone.utc) - timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    c = _client()
    c.list_jobs.return_value = [{"job_id": "7", "status": "RUNNING", "created_at": created, "updated_at": updated,
                                 "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training",
                                               "status": "RUNNING"},
                                              {"sub_job_id": "7:sampling:0", "job_type": "sampling",
                                               "status": "PLACING"}]}]

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 40)) as pilot:
            ok = await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            assert ok
            table = app.screen.query_one("#jobs", DataTable)
            drawn = [cell.plain for cell in table.get_row("job-0")]
            assert "3h ago" in drawn and "2m ago" in drawn  # the screen reads relative times
            # The header, the LIVE heading, the job, then its two sub-jobs (live: it starts unfolded).
            _header, heading, job, training, sampling = table.rows_text()
            assert heading.startswith("LIVE")
            # A paste stays true: the times themselves, and each status as its word, no glyph.
            assert format_created_at(created) in job and format_created_at(updated) in job and "ago" not in job
            copied = "".join((heading, job, training, sampling))
            assert "RUNNING" in job and "●" not in copied and "◔" not in copied
            assert "RUNNING" in training and "PLACING" in sampling  # drawn as the glyph alone, and glyph and word
            assert "─" not in heading  # the LIVE heading's rule is drawn only
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_highlight_of_job_list_rows_copies_whole_times_and_status_words():
    created = (datetime.now(timezone.utc) - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    c = _client()
    c.list_jobs.return_value = [{"job_id": "7", "status": "RUNNING", "created_at": created, "updated_at": created,
                                 "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training",
                                               "status": "RUNNING"},
                                              {"sub_job_id": "7:sampling:0", "job_type": "sampling",
                                               "status": "PLACING"}]}]

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        copied: list[str] = []
        app.copy_to_clipboard = copied.append
        async with app.run_test(size=(160, 40)) as pilot:
            assert await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            await pilot.pause(0.1)
            table = app.screen.query_one("#jobs", DataTable)
            mouse = _pointer(app, pilot)
            lines = _screen_lines(app)
            y = next(y for y, line in enumerate(lines) if "3h ago" in line)  # the job's row, drawn relative
            assert "●" in lines[y]
            # One whole row: the times as YYYY-MM-DD HH:MM, the status as its word, no glyph, no age.
            await _drag(mouse, (0, y), (table.region.right - 1, y), via=[(40, y)])
            row = await _copied(app, pilot, copied)
            assert "RUNNING" in row and re.search(r"\d{4}-\d\d-\d\d \d\d:\d\d", row)
            assert "ago" not in row and not set("●◔◐○×−✓") & set(row)
            # From the LIVE heading through the sub-jobs: the heading's rule isn't copied, a
            # sub-job drawn as the glyph alone copies its word, and the columns stay in line.
            await _drag(mouse, (0, y - 1), (table.region.right - 1, y + 2), via=[(40, y)])
            heading, job, training, sampling = (await _copied(app, pilot, copied)).split("\n")
            assert heading.strip() == "LIVE  1" and "─" not in heading
            assert job == row and "RUNNING" in training and "PLACING" in sampling
            assert training.index("RUNNING") == sampling.index("PLACING") == job.index("RUNNING")
            await _settle(app, pilot)

    asyncio.run(run())


def _gpu_rows(sub_job_id, minutes, *, gpus=2):
    """GPU rows a minute apart, the newest 30 s ago, from ``minutes`` minutes before that."""
    newest = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=30)
    rows = [{"record_version": 2, "sub_job_id": sub_job_id, "worker_num": gpu, "local_gpu_index": 0,
             "global_gpu_index": gpu, "sample_ts": (newest - timedelta(minutes=back)).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "gpu_util_pct": 50.0, "fb_used_bytes": float(1024**3), "fb_free_bytes": float(8 * 1024**3),
             "sample_interval_s": 60.0}
            for back in range(minutes, -1, -1) for gpu in range(gpus)]
    return "".join(json.dumps(row) + "\n" for row in rows)


async def _run_a_first_open_draws_the_newest_window_while_the_rest_downloads(tmp_path):
    # A long job's first open: its whole history can take minutes, so the client's preview
    # (the newest 10 minutes, the window every job opens on) is drawn, marked partial, while
    # the header's bar counts the chunks; the committed file then replaces it. A refresh with
    # the job on screen asks no preview.
    sub = "7:training:0"
    c = _client()
    calls, previewed, release = [], threading.Event(), threading.Event()

    def download(job_id, output_dir, *, resume=False, priority=(), on_commit=None, on_chunk=None,
                 preview_seconds=None, on_preview=None, stale_after_seconds=600):
        calls.append({"priority": list(priority), "preview_seconds": preview_seconds,
                      "counts": on_chunk is not None})
        if on_preview is not None:
            on_chunk(sub, 0, 300, 0)
            preview = tmp_path / "preview.jsonl"
            preview.write_text(_gpu_rows(sub, 32), encoding="utf-8")
            on_preview({"sub_job_id": sub, "path": str(preview), "chunk_count": 33})
            time.sleep(0.3)  # past the bar's redraw interval
            on_chunk(sub, 120, 300, 0)
            previewed.set()
            release.wait(10)
        path = Path(output_dir) / sub / "gpu.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_gpu_rows(sub, 300), encoding="utf-8")
        path.with_name(".gpu.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": ["chunk.gz"], "bytes": path.stat().st_size}), encoding="utf-8")
        if on_chunk is not None:
            on_chunk(sub, 300, 300, 0)
        if on_commit is not None:
            on_commit({"sub_job_id": sub, "saved_path": str(path)})
        return []

    c.download_metrics = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(160, 45)) as pilot:
        assert await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        screen = app.screen
        header = lambda: str(screen.query_one("#resource-status", Static).render())  # noqa: E731
        screen.query_one("#job-tabs", TabbedContent).active = "resource"
        ok = await _pause_until(pilot, lambda: previewed.is_set() and screen._resource_data.kind(sub, "gpu") is not None
                                and "120/300" in header(), seconds=10.0)
        assert ok, header()
        shown = screen._resource_data.kind(sub, "gpu")
        assert shown.partial  # the newest window alone, and marked so
        assert "loading older samples" in header() and "━" in header() and "120/300 chunks" in header(), header()
        release.set()
        assert await _pause_until(pilot, lambda: not screen._resource_loading, seconds=10.0)
        full = screen._resource_data.kind(sub, "gpu")
        assert not full.partial and full.first_sample_ts < shown.first_sample_ts  # the whole history
        assert "chunks" not in header(), header()  # the bar is gone with the download
        assert calls == [{"priority": [sub], "preview_seconds": 10 * 60, "counts": True}]
        screen.action_refresh_sources()  # r: the job is on screen now, so only the count
        assert await _pause_until(pilot, lambda: len(calls) == 2 and not screen._resource_loading, seconds=10.0)
        assert calls[1] == {"priority": [sub], "preview_seconds": None, "counts": True}
        await _settle(app, pilot)


def test_a_first_open_draws_the_newest_window_while_the_rest_downloads(tmp_path):
    asyncio.run(_run_a_first_open_draws_the_newest_window_while_the_rest_downloads(tmp_path))


async def _run_a_job_screen_warms_its_downloads_up_once_its_first_lines_are_drawn(monkeypatch):
    # The warm-up's import of the Connector would slow the Logs tab's first lines: it starts once
    # they are drawn, long before its fallback, and once however many lines follow.
    monkeypatch.setattr(LogScreen, "_WARM_UP_AFTER_SECONDS", 30.0)
    c = _client()
    c.tail_logs.side_effect = lambda *a, **k: {"entries": [{"_raw": "a line"}], "next_cursor": "C1", "eof": False}
    warmed, shown_then = threading.Event(), []

    def warm(job_id):
        shown_then.append((job_id, app.screen._shown_logs))
        warmed.set()

    c.warm_artifact_session.side_effect = warm
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(160, 45)) as pilot:
        assert await _pause_until(pilot, warmed.is_set, seconds=10.0)
        screen = app.screen
        assert await _pause_until(pilot, lambda: screen._shown_logs >= 3, seconds=10.0)  # more lines came
        assert len(shown_then) == 1 and shown_then[0][0] == "7" and shown_then[0][1] >= 1
        await _settle(app, pilot)


def test_a_job_screen_warms_its_downloads_up_once_its_first_lines_are_drawn(monkeypatch):
    asyncio.run(_run_a_job_screen_warms_its_downloads_up_once_its_first_lines_are_drawn(monkeypatch))


async def _run_a_job_screen_without_lines_warms_up_a_moment_after_it_opens(monkeypatch):
    monkeypatch.setattr(LogScreen, "_WARM_UP_AFTER_SECONDS", 0.2)
    c = _client()  # its tail has no lines
    warmed = threading.Event()
    c.warm_artifact_session.side_effect = lambda job_id: warmed.set()
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(160, 45)) as pilot:
        assert await _wait(pilot, app, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        assert await _pause_until(pilot, warmed.is_set, seconds=5.0)
        assert app.screen._shown_logs == 0
        c.warm_artifact_session.assert_called_once_with("7")
        await _settle(app, pilot)


def test_a_job_screen_without_lines_warms_up_a_moment_after_it_opens(monkeypatch):
    asyncio.run(_run_a_job_screen_without_lines_warms_up_a_moment_after_it_opens(monkeypatch))


async def _run_a_warm_up_that_fails_shows_nothing_and_logs_it_once(tmp_path, monkeypatch):
    # The download a tab asks for logs in itself and says what failed; the warm-up only saves it
    # time. Its failure goes to the error log, once, for when the first chart is slow.
    monkeypatch.setattr(LogScreen, "_WARM_UP_AFTER_SECONDS", 0.05)
    c = _client()
    c.warm_artifact_session.side_effect = RuntimeError("login refused")
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(160, 45)) as pilot:
        assert await _pause_until(pilot, lambda: c.warm_artifact_session.called, seconds=5.0)
        await pilot.pause(0.2)
        assert isinstance(app.screen, LogScreen) and app.screen._screen_open
        assert not [note for note in app._notifications if "login refused" in str(note.message)]
        await _settle(app, pilot)
    logged = (tmp_path / "errors.log").read_text()
    assert logged.count("getting 7's downloads ready failed") == 1 and "login refused" in logged
    assert c.warm_artifact_session.call_count == 1


def test_a_warm_up_that_fails_shows_nothing_and_logs_it_once(tmp_path, monkeypatch):
    asyncio.run(_run_a_warm_up_that_fails_shows_nothing_and_logs_it_once(tmp_path, monkeypatch))


class _CountedSession:
    def __init__(self, hold=None):
        self.closes, self._hold = 0, hold

    def close(self):
        if self._hold is not None:
            self._hold.wait(5)
        self.closes += 1


def _run_tui_main(monkeypatch, client, during_run):
    class FakeApp:
        return_code = 0
        sign_in = None

        def __init__(self, client, job_id, *, sub_job_id=None, poll_interval=1.0):
            self.client = client

        def run(self):
            during_run(self.client)

    monkeypatch.delenv("CORTEX_TRAINING_CONNECTION", raising=False)
    monkeypatch.setattr("cortex_training.tui.app.CortexTrainingLogTUI", FakeApp)
    monkeypatch.setattr("cortex_training.cli.commands.build_client", lambda args, cls: client)
    return tui_main.run(["--base-url", "http://localhost:1", "--database", "DB"])


def test_the_tui_keeps_its_download_sessions_and_closes_them_on_quit(monkeypatch):
    client = CortexTrainingClient(base_url="http://localhost:1", database="DB", schema="PUBLIC")
    session, kept_while_running = _CountedSession(), []

    def during_run(running):
        kept_while_running.append(running.keep_artifact_sessions)
        running._kept_sessions.keep(running._artifact_session_key(), session)  # a download's, kept

    assert _run_tui_main(monkeypatch, client, during_run) == 0
    assert kept_while_running == [True]
    assert session.closes == 1 and not client.keep_artifact_sessions


def test_quitting_waits_a_moment_at_most_for_the_sessions_to_close(monkeypatch):
    monkeypatch.setattr(tui_main, "SESSION_CLOSE_SECONDS", 0.1)
    client = CortexTrainingClient(base_url="http://localhost:1", database="DB", schema="PUBLIC")
    release = threading.Event()
    session = _CountedSession(hold=release)  # an unreachable service: its close hangs

    def during_run(running):
        running._kept_sessions.keep(running._artifact_session_key(), session)

    started = time.monotonic()
    try:
        assert _run_tui_main(monkeypatch, client, during_run) == 0
        assert time.monotonic() - started < 2
    finally:
        release.set()


def test_quitting_after_the_app_failed_still_closes_the_sessions(monkeypatch):
    client = CortexTrainingClient(base_url="http://localhost:1", database="DB", schema="PUBLIC")
    session = _CountedSession()

    def during_run(running):
        running._kept_sessions.keep(running._artifact_session_key(), session)
        raise RuntimeError("the app failed")

    with pytest.raises(RuntimeError, match="the app failed"):
        _run_tui_main(monkeypatch, client, during_run)
    assert session.closes == 1


@pytest.mark.parametrize(("window", "seconds"), [("10m", 600), ("30m", 1800), ("1h", 1800), ("6h", 1800),
                                                 ("all", 1800)])
def test_a_first_open_previews_the_window_on_screen_and_at_most_its_half_hour(window, seconds):
    # One round of chunk fetches draws the 10m window; a larger window shows the newest half
    # hour, marked loading older samples, as before.
    screen = SimpleNamespace(_resource_range=window, _RESOURCE_PREVIEW_SECONDS=ResourceTab._RESOURCE_PREVIEW_SECONDS)
    assert ResourceTab._resource_preview_seconds(screen) == seconds


async def _run_a_finished_job_s_console_shows_its_last_lines_while_the_rest_downloads(tmp_path):
    # A long finished job's console takes minutes to download: the sub-job on screen shows
    # the client's preview (its newest chunks) under a note, until its whole console is in.
    c = _client()
    c.get_job.return_value = _old_terminal_job(minutes_ago=30)
    calls, previewed, release = [], threading.Event(), threading.Event()
    console = "".join(f"line {i}\n" for i in range(10)) + "".join(f"newest line {i}\n" for i in range(3))

    def download(job_id, output_dir, *, resume=False, sub_job_ids=None, on_chunk=None, preview_seconds=None,
                 on_preview=None, stale_after_seconds=600):
        calls.append({"sub_job_ids": sub_job_ids, "preview_seconds": preview_seconds})
        if sub_job_ids and on_preview is not None:
            preview = tmp_path / "preview.log"
            preview.write_text("".join(f"newest line {i}\n" for i in range(3)), encoding="utf-8")
            on_preview({"sub_job_id": sub_job_ids[0], "path": str(preview), "chunk_count": 3})
            previewed.set()
            release.wait(10)
        for sub in sub_job_ids or ["7:training:0", "7:sampling:0"]:
            if not (Path(output_dir) / sub / "stdout.log").exists():  # resumed: a console that is in stays as it is
                _write_stdout(output_dir, sub, console)
        return []

    c.download_stdout_logs = download
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        ok = await _wait(pilot, app, lambda: previewed.is_set() and isinstance(app.screen, LogScreen)
                         and "newest line 2" in app.screen._shown_lines, tries=200)
        assert ok
        screen = app.screen
        assert any("older lines are still downloading" in line for line in screen._shown_lines), screen._shown_lines
        assert "line 0" not in screen._shown_lines  # the rest isn't in yet
        release.set()
        ok = await _wait(pilot, app, lambda: "line 0" in screen._shown_lines and not screen._stdout_inflight, tries=200)
        assert ok, screen._shown_lines
        assert not any("still downloading" in line for line in screen._shown_lines)  # the whole console, alone
        assert calls == [{"sub_job_ids": ["7:training:0"], "preview_seconds": 30 * 60},
                         {"sub_job_ids": None, "preview_seconds": None}]  # the others', whole
        # Kept for the next open (the job ended long ago), without the preview.
        assert (screen._console_dir / "7:training:0" / "stdout.log").exists()
        assert not (screen._console_dir / ".preview").exists()
        await _settle(app, pilot)


def test_a_finished_job_s_console_shows_its_last_lines_while_the_rest_downloads(tmp_path):
    asyncio.run(_run_a_finished_job_s_console_shows_its_last_lines_while_the_rest_downloads(tmp_path))


# ─── the Schema column, and narrowing the account-wide list to one schema ─────


async def _account_list(app, pilot):
    """The fake's account-wide job list (``a``), drawn: the screen and its table."""
    screen, table = await _demo_list(app, pilot)
    await pilot.press("a")
    assert await _pause_until(pilot, lambda: screen._scope is not None and screen._scope.kind == ACCOUNT
                              and not screen._refreshing, seconds=10.0), app.sub_title
    await pilot.pause(0.1)
    return screen, table


async def _run_the_schema_column_names_each_jobs_database_and_schema():
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    async with app.run_test(size=(200, 40)) as pilot:  # wide enough for every column
        screen, table = await _account_list(app, pilot)
        labels = [str(column.label) for column in table.columns.values()]
        assert "Schema" in labels, labels
        assert "Location" not in labels  # the header this column used to carry
        at = labels.index("Schema")
        # The cell is built under the same name the column is added under: were the two to drift,
        # the row lookup would fall back to a blank cell and quietly empty the column.
        drawn = {str(table.get_row_at(row)[at]).strip() for row in range(table.row_count)}
        assert {"CORTEX_TRAINING_DB.PUBLIC", "TEAM_DB.EXPERIMENTS", '"sandbox"."dev"'} <= drawn, sorted(drawn)
        await _settle(app, pilot)


def test_the_schema_column_names_each_jobs_database_and_schema():
    asyncio.run(_run_the_schema_column_names_each_jobs_database_and_schema())


async def _run_the_header_draws_which_list_is_shown_as_a_chip():
    from dataclasses import replace
    from cortex_training.tui.account import CAPPED
    from cortex_training.tui.job_list import _ACCOUNT_CHIP
    from cortex_training.tui.job_list import _SCHEMA_CHIP
    from cortex_training.tui.job_list import _ListHeader

    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    async with app.run_test(size=(200, 40)) as pilot:
        screen, _table = await _demo_list(app, pilot)
        header = screen.query_one(_ListHeader)

        def chips():
            title = header.format_title()
            return [(title.plain[span.start:span.end], str(span.style)) for span in title.spans
                    if " on " in str(span.style)]

        assert await _pause_until(pilot, lambda: not screen._refreshing)
        assert chips() == [(f" Current schema: {screen._schema_label()} ", _SCHEMA_CHIP)]
        screen._refreshing = screen._switching = True  # a pressed, the other list not read yet: no chip
        assert screen.scope_chip() is None
        screen._refreshing = screen._switching = False
        screen, _table = await _account_list(app, pilot)
        assert chips() == [(" all schemas ", _ACCOUNT_CHIP)]
        # Cut short, the subtitle says only how far the list got: the chip comes first.
        screen._scope = replace(screen._scope, walked=2000, stopped=CAPPED)
        screen._update_subtitle()
        assert app.sub_title.startswith("last refreshed ") and "all schemas" not in app.sub_title
        assert header.format_title().plain.startswith("cortex training —  all schemas   ·  last refreshed ")
        assert chips() == [(" all schemas ", _ACCOUNT_CHIP)]
        await _settle(app, pilot)


def test_the_header_draws_which_list_is_shown_as_a_chip():
    asyncio.run(_run_the_header_draws_which_list_is_shown_as_a_chip())


async def _run_a_long_schema_name_is_whole_where_the_screen_has_room():
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    async with app.run_test(size=(200, 40)) as pilot:
        screen, _table = await _account_list(app, pilot)
        located = [{"database_name": "ML_PLATFORM_DB", "schema_name": name} for name in ("A", "EXPERIMENTS_TEAM_ALPHA")]
        assert screen._location_column_width(located) == 37  # its widest name, past the 32 it once stopped at
        # Room to spare: the column takes the whole name, and every other column keeps its width.
        wide = dict(screen._layout_for(37).columns)
        assert (wide["Schema"], wide["Job ID"], wide["Updated"]) == (37, 38, 8)
        # None to spare: it holds to 32, and its cells cut the database name (see below).
        await pilot.resize_terminal(150, 40)
        await pilot.pause(0.1)
        assert dict(screen._layout_for(37).columns)["Schema"] == 32
        await _settle(app, pilot)


def test_a_long_schema_name_is_whole_where_the_screen_has_room():
    asyncio.run(_run_a_long_schema_name_is_whole_where_the_screen_has_room())


def test_a_schema_cell_too_narrow_cuts_the_database_and_keeps_the_schema():
    from rich.console import Console
    from cortex_training.tui.job_list import _location_cell

    cell = _location_cell({"database_name": "ML_PLATFORM_DB", "schema_name": "EXPERIMENTS_A"})
    console = Console(width=80)

    def drawn(width):
        line, = console.render_lines(cell, console.options.update_width(width), pad=False)
        return "".join(segment.text for segment in line)

    assert drawn(40) == "ML_PLATFORM_DB.EXPERIMENTS_A"
    # Two schemas of one database still read apart, so the cut is in the database name.
    assert drawn(20) == "ML_PL….EXPERIMENTS_A"
    assert drawn(15) == "….EXPERIMENTS_A"
    assert drawn(10) == "ML_PLATFO…"  # too narrow for the schema too: the end is cut, as any cell's is
    assert str(cell) == "ML_PLATFORM_DB.EXPERIMENTS_A"  # copied whole
    quoted = _location_cell({"database_name": "ml", "schema_name": "dev"})
    assert str(quoted) == '"ml"."dev"'
    assert str(_location_cell({})) == "—"


async def _run_a_whole_location_in_the_search_is_read_from_the_service():
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    async with app.run_test(size=(200, 40)) as pilot:
        screen, _table = await _account_list(app, pilot)
        assert screen._pushdown is None  # every schema, so far

        box = screen.query_one("#jobfilter", Input)
        box.value = "db=TEAM_DB schema=EXPERIMENTS"
        await pilot.pause(0.3)  # past the search debounce
        assert await _pause_until(pilot, lambda: screen._pushdown == ("TEAM_DB", "EXPERIMENTS")
                                  and not screen._refreshing, seconds=10.0), screen._pushdown
        assert screen._scope.narrowed == "TEAM_DB.EXPERIMENTS"
        # The header says the list was narrowed, never that it covered every schema.
        assert "narrowed to TEAM_DB.EXPERIMENTS" in app.sub_title, app.sub_title
        assert "all in" not in app.sub_title
        # Every row left is in that schema, so the column collapses and the header carries it.
        assert [str(column.label) for column in screen.query_one("#jobs", DataTable).columns.values()
                ].count("Schema") == 0

        box.value = ""  # cleared: the whole account is read again
        await pilot.pause(0.3)
        assert await _pause_until(pilot, lambda: screen._pushdown is None and not screen._refreshing,
                                  seconds=10.0), screen._pushdown
        assert screen._scope.narrowed == ""
        await _settle(app, pilot)


def test_a_whole_location_in_the_search_is_read_from_the_service():
    asyncio.run(_run_a_whole_location_in_the_search_is_read_from_the_service())


async def _run_a_search_that_cannot_narrow_reads_every_schema(text):
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    async with app.run_test(size=(200, 40)) as pilot:
        screen, _table = await _account_list(app, pilot)
        screen.query_one("#jobfilter", Input).value = text
        await pilot.pause(0.3)
        assert await _pause_until(pilot, lambda: not screen._refreshing, seconds=10.0)
        # A partial match, a negation, or a schema with no database of its own: matched here, so
        # the walk still reads every schema and the header still says so.
        assert screen._pushdown is None, text
        assert screen._scope.narrowed == ""
        assert "narrowed to" not in app.sub_title, app.sub_title
        await _settle(app, pilot)


@pytest.mark.parametrize("text", ["schema:exper", "db!=TEAM_DB", "schema=EXPERIMENTS", "db:team"])
def test_a_search_that_cannot_narrow_reads_every_schema(text):
    asyncio.run(_run_a_search_that_cannot_narrow_reads_every_schema(text))


async def _run_a_narrowing_with_no_jobs_says_so_rather_than_failing():
    app = CortexTrainingLogTUI(FakeClient(), poll_interval=0.01)
    async with app.run_test(size=(200, 40)) as pilot:
        screen, _table = await _account_list(app, pilot)
        screen.query_one("#jobfilter", Input).value = "db=NOPE schema=NOPE"
        await pilot.pause(0.3)
        assert await _pause_until(pilot, lambda: not screen._refreshing and screen._pushdown is not None,
                                  seconds=10.0)
        # No such database here, so the narrowed walk answers an empty page: the search says
        # nothing matched rather than the list reading as a failure. (A service that refuses the
        # name outright with a 400 is covered in tests/test_tui_account.py.)
        assert screen._list_error is None, screen._list_error
        assert not screen.query_one("#jobs", DataTable).display
        await _settle(app, pilot)


def test_a_narrowing_with_no_jobs_says_so_rather_than_failing():
    asyncio.run(_run_a_narrowing_with_no_jobs_says_so_rather_than_failing())


async def _run_a_download_that_grows_the_cache_by_much(tmp_path, monkeypatch, *, grow):
    import cortex_training.tui.app as app_module
    from cortex_training.tui.log_cache import job_cache_dir

    old = job_cache_dir("0123456789abcdef:old-job", root=tmp_path / "isolated-cache")
    (old / "metrics").mkdir(parents=True)
    (old / "metrics" / "gpu.jsonl").write_bytes(b"x" * 4096)
    long_ago = time.time() - 86400
    for path in (old, *old.rglob("*")):
        os.utime(path, (long_ago, long_ago))
    cls = app_module.CortexTrainingLogTUI
    monkeypatch.setattr(cls, "_CACHE_CLEANUP_FIRST_SECONDS", 3600)  # only the pass a download asks for
    monkeypatch.setattr(cls, "_CACHE_CLEANUP_SECONDS", 3600)
    monkeypatch.setattr(cls, "_CACHE_LIMIT", 1024)
    monkeypatch.setattr(cls, "CACHE_PASS_AFTER_BYTES", 10_000)
    monkeypatch.setattr(cls, "_CACHE_PASS_AFTER_DOWNLOAD_SECONDS", 0.1)
    c = _located_client()

    def download(job_id, output_dir, **_kwargs):
        _write_gpu_metrics(output_dir, "7:training:0", gpu_count=2)
        if grow:
            with (Path(output_dir) / "7:training:0" / "gpu.jsonl").open("ab") as f:
                f.write(b"\n" * 20_000)  # a long job's history, past the threshold
        return []

    c.download_metrics.side_effect = download
    app = cls(c, "7", poll_interval=0.01)
    async with app.run_test(size=(130, 36)) as pilot:
        assert await _pause_until(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        app.screen.query_one("#job-tabs", TabbedContent).active = "resource"
        assert await _pause_until(pilot, lambda: c.download_metrics.call_count >= 1, seconds=5.0)
        await _pause_until(pilot, lambda: not old.exists(), seconds=3.0)
        gone = not old.exists()
        await _settle(app, pilot)
    return gone


def test_a_download_that_grows_the_cache_by_much_holds_it_to_its_limit_soon(tmp_path, monkeypatch):
    # The next pass was up to 10 minutes away, and none if the TUI quit first: a long job's
    # first open left the cache gigabytes over its limit until the next start.
    assert asyncio.run(_run_a_download_that_grows_the_cache_by_much(tmp_path, monkeypatch, grow=True)), \
        "a big download left the cache over its limit until the next pass"


def test_a_download_that_adds_little_waits_for_the_next_pass(tmp_path, monkeypatch):
    assert not asyncio.run(_run_a_download_that_grows_the_cache_by_much(tmp_path, monkeypatch, grow=False)), \
        "every download ran a pass of its own"


async def _run_the_open_job_survives_the_cache_limit_and_the_rest_goes(tmp_path, monkeypatch):
    import cortex_training.tui.app as app_module
    from cortex_training.tui.cache_cleanup import STAGE_PREFIX
    from cortex_training.tui.cache_cleanup import _HOST
    from cortex_training.tui.cache_cleanup import _is_job_dir
    from cortex_training.tui.cache_cleanup import _release_pending
    from cortex_training.tui.cache_cleanup import job_in_use
    from cortex_training.tui.log_cache import job_cache_dir
    from cortex_training.tui.widgets import _cache_job_id
    from cortex_training import _training_log

    root = tmp_path / "isolated-cache"
    client = _located_client()
    old = job_cache_dir("0123456789abcdef:old-job", root=root)
    (old / "metrics").mkdir(parents=True)
    (old / "metrics" / "gpu.jsonl").write_bytes(b"x" * (2 * 1024 * 1024))
    open_dir = job_cache_dir(_cache_job_id(client, "7"), root=root)
    (open_dir / "console" / "7:training:0").mkdir(parents=True)
    (open_dir / "console" / "7:training:0" / "stdout.log").write_bytes(b"x" * 4096)
    long_ago = time.time() - 86400
    for path in (old, *old.rglob("*"), open_dir, *open_dir.rglob("*")):
        os.utime(path, (long_ago, long_ago))
    # The client writes these, so only a process that runs a job would ever prune them.
    # Laid out as the client writes them: <job>/<sub_job_id>/training.jsonl.
    stale_local = _training_log.local_dir("a-job-nobody-has-run-for-a-month", root=root)
    (stale_local / "a-job-nobody-has-run-for-a-month:training:0").mkdir(parents=True)
    (stale_local / "a-job-nobody-has-run-for-a-month:training:0" / _training_log.FILE_NAME).write_bytes(b"{}\n")
    a_month_ago = time.time() - _training_log.MAX_AGE_SECONDS - 86400
    for path in (stale_local, *stale_local.rglob("*")):
        os.utime(path, (a_month_ago, a_month_ago))
    fresh_local = _training_log.local_dir("a-job-running-now", root=root)
    (fresh_local / "a-job-running-now:training:0").mkdir(parents=True)
    (fresh_local / "a-job-running-now:training:0" / _training_log.FILE_NAME).write_bytes(b"{}\n")
    stage = Path(tempfile.gettempdir()) / f"{STAGE_PREFIX}{_HOST}+999999+gone"
    stage.mkdir()
    Path(f"{stage}.owner").touch()
    monkeypatch.setattr(app_module.CortexTrainingLogTUI, "_CACHE_CLEANUP_FIRST_SECONDS", 0.2)
    monkeypatch.setattr(app_module.CortexTrainingLogTUI, "_CACHE_LIMIT", 1024)
    monkeypatch.setattr("cortex_training.tui.logs.LEASE_HEARTBEAT_SECONDS", 0.2)

    app = CortexTrainingLogTUI(client, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        assert await _pause_until(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        screen, lease = app.screen, app.screen._lease
        assert screen._cache_limit == 1024, "a kept console would hold the cache to another limit than the app's"
        assert lease._job_dir == open_dir and _is_job_dir(open_dir), "the open job isn't the cache's to remove"
        assert job_in_use(open_dir), "the open job holds no lease"
        os.utime(lease._marker, (long_ago, long_ago))

        def marker_refreshed():
            # The sweep removes a marker this stale; the live screen's heartbeat writes it
            # again, so a moment with no marker at all is not a failure.
            try:
                return time.time() - lease._marker.stat().st_mtime < 60
            except FileNotFoundError:
                return False

        assert await _pause_until(pilot, marker_refreshed, seconds=10.0), \
            "the open job's lease is never refreshed: another machine would take the job for closed"
        assert await _pause_until(pilot, lambda: not old.exists(), seconds=10.0), "the old job was kept"
        assert open_dir.exists(), "cache cleanup removed the open job"
        assert job_in_use(open_dir), "the open job lost its lease"
        await pilot.pause(0.3)
        assert not [note for note in _notes(app) if "cached data" in note or "Freed" in note], \
            "the cache's own housekeeping was announced"
        assert not stage.exists() and not os.path.lexists(f"{stage}.owner"), "the abandoned stage was left"
        assert stale_local.exists(), "the TUI removed the client's own rows (the client's rules do that)"
        # The open job holds the cache over its limit on every pass, and the local rows of a job
        # still being written stay through all of them: they are never removed to make room.
        assert fresh_local.exists(), "the pass took local training rows still being written"
        await _settle(app, pilot)
    assert lease.held and job_in_use(open_dir)
    _release_pending()
    assert not lease.held and not job_in_use(open_dir)


def test_the_open_job_survives_the_cache_limit_and_the_rest_goes(tmp_path, monkeypatch):
    asyncio.run(_run_the_open_job_survives_the_cache_limit_and_the_rest_goes(tmp_path, monkeypatch))


def test_the_cache_keeps_under_the_5_gib_the_docs_state():
    from cortex_training.tui.cache_cleanup import DEFAULT_CACHE_LIMIT

    assert CortexTrainingLogTUI._CACHE_LIMIT == DEFAULT_CACHE_LIMIT == 5 * 1024 ** 3


def test_a_console_is_kept_only_under_the_job_s_lease_and_the_jobs_used_least_recently_then_go(tmp_path, monkeypatch):
    from cortex_training._download_lock import DownloadLock
    from cortex_training.tui.log_cache import job_cache_dir

    other = job_cache_dir("0123456789abcdef:other", root=tmp_path / "isolated-cache")
    (other / "metrics").mkdir(parents=True)
    (other / "metrics" / "gpu.jsonl").write_bytes(b"x" * 600)  # within the limit until the console lands
    long_ago = time.time() - 86400
    for path in (other, *other.rglob("*")):
        os.utime(path, (long_ago, long_ago))
    downloaded = tmp_path / "downloaded"
    _write_stdout(downloaded, "7:training:0", "x" * 2000 + "\n")
    ended = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    plan = LogScreen._Download(first=None, again=False, root=str(downloaded), others=False, seen=None,
                               ended_at=ended, created_at="created")
    screen = LogScreen(_located_client(), "7", job={"job_id": "7", "status": "FAILED"})
    screen._cache_limit = 1024
    notes = []
    screen._post = lambda fn, *args, **kwargs: notes.append(args[0]) or True
    kept = screen._console_dir / ".complete"

    assert screen._keep_console(plan) is False  # lease not held
    assert not kept.exists() and other.exists() and not notes

    real_try = DownloadLock.try_acquire

    def try_acquire_as_the_lease_goes(self):
        taken = real_try(self)
        screen._lease.release()
        return taken

    screen._lease.acquire()
    monkeypatch.setattr(DownloadLock, "try_acquire", try_acquire_as_the_lease_goes)
    assert screen._keep_console(plan) is False
    assert not kept.exists() and other.exists() and not notes
    monkeypatch.setattr(DownloadLock, "try_acquire", real_try)

    # Another TUI keeping this same console holds its lock: not waited for (the pane showed
    # its console seconds late behind that copy, or never, if that TUI was stopped mid-copy).
    other_tui = DownloadLock(screen._console_dir)
    assert other_tui.try_acquire()
    screen._lease.acquire()
    try:
        started = time.monotonic()
        assert screen._keep_console(plan) is False
        assert time.monotonic() - started < 0.5, "keeping the console waited for another TUI's copy"
    finally:
        other_tui.release()
        screen._lease.release()

    screen._lease.acquire()
    try:
        assert screen._keep_console(plan) is True
        assert kept.read_text(encoding="utf-8") == "created"
        assert other.exists(), "keeping the console waited for the limit pass (see _hold_cache_to_limit)"
        screen._hold_cache_to_limit()
        assert not other.exists(), "the job used least recently stayed after the console was kept"
        assert not notes, "the cache's own housekeeping was announced"
    finally:
        screen._lease.release()


async def _run_a_new_session_resumes_the_tail_where_the_last_one_stopped(tmp_path):
    c = _client()
    c.tail_logs.return_value = {"entries": [{"msg": "first session's line"}], "next_cursor": "cursor-1", "eof": True}
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        assert await _pause_until(pilot, lambda: c.tail_logs.call_count > 0 and "first session's line" in
                                  "\n".join(getattr(app.screen, "_shown_lines", []) or []), seconds=10.0)
        await _settle(app, pilot)

    second = _client()
    second.tail_logs.return_value = {"entries": [], "next_cursor": "cursor-1", "eof": True}
    app = CortexTrainingLogTUI(second, "7", poll_interval=0.01)
    async with app.run_test() as pilot:
        assert await _pause_until(pilot, lambda: second.tail_logs.call_count > 0, seconds=10.0)
        assert "first session's line" in "\n".join(app.screen._shown_lines), "the cached lines didn't survive the quit"
        cursors = [call.kwargs.get("cursor") for call in second.tail_logs.call_args_list]
        assert cursors[0] == "cursor-1", f"the tail started over instead of resuming: {cursors[:3]}"
        await _settle(app, pilot)


def test_a_new_session_resumes_the_tail_where_the_last_one_stopped(tmp_path):
    asyncio.run(_run_a_new_session_resumes_the_tail_where_the_last_one_stopped(tmp_path))


async def _run_a_tail_nobody_watches_stops_polling(monkeypatch):
    monkeypatch.setattr(LogScreen, "_IDLE_PAUSE_SECONDS", 1.0)
    monkeypatch.setattr(LogScreen, "_IDLE_CHECK_SECONDS", 0.2)
    c = _client()
    c.tail_logs.return_value = {"entries": [], "next_cursor": "c1", "eof": False}
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.05)
    async with app.run_test() as pilot:
        assert await _pause_until(pilot, lambda: isinstance(app.screen, LogScreen) and c.tail_logs.call_count > 0)
        screen = app.screen
        assert await _pause_until(pilot, lambda: screen._idle_paused, seconds=10.0), "the tail never paused"
        await pilot.pause(0.5)
        polls = c.tail_logs.call_count
        await pilot.pause(2.0)
        assert c.tail_logs.call_count == polls, "the paused tail kept polling"
        await pilot.press("p")
        assert await _pause_until(pilot, lambda: c.tail_logs.call_count > polls, seconds=10.0), "p didn't resume it"
        assert not screen._idle_paused
        await _settle(app, pilot)


def test_a_tail_nobody_watches_stops_polling(monkeypatch):
    asyncio.run(_run_a_tail_nobody_watches_stops_polling(monkeypatch))


async def _run_a_schema_list_past_the_cap(c):
    app = CortexTrainingLogTUI(c, poll_interval=0.01)
    async with app.run_test(size=(160, 40)) as pilot:
        assert await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item,
                           tries=400)
        screen = app.screen
        drawn = lambda: len(screen._job_by_item)  # noqa: E731
        assert drawn() == 2_001, "not the newest 2,000 and every active job"
        assert "newest 2,000 + every active job (more exist)" in screen._scope_text()
        await pilot.press("f")
        assert await _wait(pilot, app, lambda: drawn() == 2_500, tries=400), "f drew no more"
        assert "newest 2,000" not in screen._scope_text()
        calls = c.list_jobs.call_count
        await pilot.press("r")  # a refresh keeps what f asked for
        assert await _wait(pilot, app, lambda: c.list_jobs.call_count > calls and drawn() == 2_500, tries=400)
        await pilot.press("f")
        assert await _wait(pilot, app, lambda: drawn() == 2_001, tries=400), "f did not go back"
        await _settle(app, pilot)


def test_a_schema_list_past_the_cap_draws_its_newest_and_f_draws_every_job():
    # Where the account-level list isn't there, list_jobs answers the schema whole: drawing
    # tens of thousands of rows froze the list for seconds, and every search keystroke after.
    now = datetime.now(timezone.utc)
    c = _client()
    c.list_jobs.return_value = [{"job_id": f"j{i}", "status": "SUCCEEDED",
                                 "created_at": (now - timedelta(minutes=i)).isoformat()} for i in range(2_500)]
    c.list_jobs.return_value[-1]["status"] = "RUNNING"  # the oldest, still running: always listed
    asyncio.run(_run_a_schema_list_past_the_cap(c))


async def _run_a_on_a_schema_list_past_the_cap(c):
    app = CortexTrainingLogTUI(c, poll_interval=0.01)
    async with app.run_test(size=(160, 40)) as pilot:
        assert await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item,
                           tries=400)
        screen = app.screen
        drawn = lambda: len(screen._job_by_item)  # noqa: E731
        cut = "newest 2,000 + every active job (more exist)"

        async def press_a():
            calls = c.list_jobs.call_count
            await pilot.press("a")
            assert await _wait(pilot, app, lambda: screen._declined and c.list_jobs.call_count > calls
                               and not screen._refreshing, tries=400), "a never came back to the schema"

        assert drawn() == 2_001 and cut in screen._scope_text()
        await press_a()  # all schemas aren't there: the list stays the schema's, cut as it was
        assert drawn() == 2_001, "a drew a different list"
        assert cut in screen._scope_text(), f"the header lost the cut: {screen._scope_text()!r}"
        assert screen.check_action("full_list", ("all",)), "a hid f"
        await pilot.press("f")
        assert await _wait(pilot, app, lambda: drawn() == 2_500, tries=400), "f drew no more after a"
        await press_a()  # with every job drawn, they stay drawn: it is the same list
        assert drawn() == 2_500 and screen.check_action("full_list", ("newest",))
        await pilot.press("f")
        assert await _wait(pilot, app, lambda: drawn() == 2_001, tries=400), "f did not go back"
        assert cut in screen._scope_text()
        await _settle(app, pilot)


def test_a_on_a_schema_list_past_the_cap_keeps_it_cut_and_f_at_hand():
    # Where the account-level list isn't there, a reads the configured schema's whole list
    # again; it is shown as the schema list it is, so cut, saying so, with every job for f.
    now = datetime.now(timezone.utc)
    c = _client()
    c.list_jobs.return_value = [{"job_id": f"j{i}", "status": "SUCCEEDED",
                                 "created_at": (now - timedelta(minutes=i)).isoformat()} for i in range(2_500)]
    c.list_jobs.return_value[-1]["status"] = "RUNNING"  # the oldest, still running: always listed
    asyncio.run(_run_a_on_a_schema_list_past_the_cap(c))


async def _run_f_just_as_a_refresh_lands(c):
    app = CortexTrainingLogTUI(c, poll_interval=0.01)
    async with app.run_test(size=(160, 40)) as pilot:
        assert await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item,
                           tries=400)
        screen = app.screen
        drawn = lambda: len(screen._job_by_item)  # noqa: E731
        older = screen._cut_listing
        assert older is not None and drawn() == 2_001
        created = datetime.now(timezone.utc).isoformat()
        job = {"job_id": "j-new", "status": "SUCCEEDED", "created_at": created}
        newer = Listing([job, *older.jobs], older.scope, everything=[job, *older.everything])
        screen.action_full_list("all")  # its draw waits for the note to be painted
        screen._show_listing(newer)  # a refresh lands first, drawn whole as f asks
        assert await _wait(pilot, app, lambda: drawn() == 2_501, tries=400), "the refresh wasn't drawn whole"
        await pilot.pause(0.5)  # f's own draw has had its turn
        assert screen._cut_listing is newer and drawn() == 2_501, "f drew the older list back over the refresh"
        assert "j-new" in {row.get("job_id") for row in screen._all_jobs}
        # f twice before its draw: the newest stay drawn.
        screen.action_full_list("newest")
        assert await _wait(pilot, app, lambda: drawn() == 2_002, tries=400)
        screen.action_full_list("all")
        screen.action_full_list("newest")
        await pilot.pause(0.5)
        assert drawn() == 2_002 and not screen._full_list, "a draw f no longer asked for was made"
        await _settle(app, pilot)


def test_f_just_as_a_refresh_lands_keeps_the_refresh_s_list():
    # f draws on the next idle; a refresh that shows in between has drawn its list whole, as f asks,
    # and the older list f held must not go back over it until the next refresh.
    now = datetime.now(timezone.utc)
    c = _client()
    c.list_jobs.return_value = [{"job_id": f"j{i}", "status": "SUCCEEDED",
                                 "created_at": (now - timedelta(minutes=i + 1)).isoformat()} for i in range(2_500)]
    c.list_jobs.return_value[-1]["status"] = "RUNNING"
    asyncio.run(_run_f_just_as_a_refresh_lands(c))


async def _run_f_does_nothing_on_a_list_not_cut_short(c):
    app = CortexTrainingLogTUI(c, poll_interval=0.01)
    async with app.run_test() as pilot:
        assert await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
        before = len(app.screen._job_by_item)
        await pilot.press("f")
        await pilot.pause(0.1)
        assert len(app.screen._job_by_item) == before
        assert not app.screen.check_action("full_list", ("all",)), "f offered on a list that is whole"
        await _settle(app, pilot)


def test_f_does_nothing_on_a_list_not_cut_short():
    asyncio.run(_run_f_does_nothing_on_a_list_not_cut_short(_client()))


def _extending_download(start):
    """A download whose first call writes two hours of 4 GPUs and each later one appends 30 s,
    as a resumed download extends the file: the loader reads only the rows after."""
    calls = []

    def download(job_id, output_dir, **_kwargs):
        sub = Path(output_dir) / "7:training:0"
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / "gpu.jsonl"
        first, count = (0, 1440) if not calls else (1440 + 6 * (len(calls) - 1), 6)
        calls.append(1)
        rows = [{"record_version": 2, "sub_job_id": "7:training:0", "worker_num": gpu, "local_gpu_index": 0,
                 "global_gpu_index": gpu, "sample_interval_s": 5.0, "gpu_util_pct": float((k * 7 + gpu) % 100),
                 "fb_used_bytes": float(1024**3), "fb_free_bytes": float(8 * 1024**3),
                 "sample_ts": (start + timedelta(seconds=5 * k)).strftime("%Y-%m-%dT%H:%M:%SZ")}
                for k in range(first, first + count) for gpu in range(4)]
        with path.open("a", encoding="utf-8") as out:
            out.write("".join(json.dumps(row) + "\n" for row in rows))
        names = ["c0.gz"] + [f"c{n}.gz" for n in range(1, len(calls))]
        path.with_name(".gpu.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": names, "bytes": path.stat().st_size}), encoding="utf-8")
        return [{"sub_job_id": "7:training:0", "path": str(path)}]

    return download


async def _run_a_window_picked_after_a_refresh(tmp_path, monkeypatch):
    import cortex_training.tui.resource as resource_module
    import cortex_training.tui.widgets as widgets_module
    from cortex_training._download_lock import DownloadLock
    from tests.test_tui_resource import _same

    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=2)
    c = _client()
    c.download_metrics.side_effect = _extending_download(start)
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        assert await _pause_until(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        screen = app.screen
        screen.query_one("#job-tabs", TabbedContent).active = "resource"
        loaded = lambda n: c.download_metrics.call_count >= n and not screen._resource_loading  # noqa: E731
        assert await _pause_until(pilot, lambda: loaded(1) and screen._resource_kind_data(), seconds=15)
        await pilot.press("r")  # a refresh that extends the file: only the window on screen is built
        assert await _pause_until(pilot, lambda: loaded(2), seconds=15)
        gpu = screen._resource_kind_data()
        assert gpu.fresh("10m") and not gpu.fresh("30m"), "the refresh built windows that aren't shown"

        held = DownloadLock(screen._metrics_root)  # holds the rebuild off, as a download in flight would
        assert held.try_acquire()
        await pilot.press("w")  # to 30m: shown as it last was, then rebuilt from the rows in hand
        await pilot.pause(0.2)
        assert screen._resource_range == "30m"
        assert screen._resource_kind_data().view("30m") is not None, "the window picked showed nothing meanwhile"
        assert screen._resource_state(screen._resource_kind_data()) == ("refreshing…",), \
            "a window as it last was did not say so"
        held.release()
        assert await _pause_until(pilot, lambda: screen._resource_kind_data().fresh("30m"), seconds=15), \
            "the window picked was never rebuilt"
        assert c.download_metrics.call_count == 2, "picking a window downloaded"
        assert screen._resource_state(screen._resource_kind_data()) == (), "still says it is refreshing"
        whole = resource_module.ResourceLoader(screen._metrics_root).load().kinds["7:training:0"]["gpu"]
        assert _same(screen._resource_kind_data().view("30m"), whole.view("30m"))
        # A rebuild that answers after a newer load (a refresh's) must not put its older rows back.
        current = screen._resource_data
        screen._accept_rebuilt_window(resource_module.ResourceData({}), resource_module.resource_loader(screen._metrics_root))
        assert screen._resource_data is current, "an older rebuild replaced newer rows"

        failures = ["rebuild broke"]

        def broken(self, **kwargs):
            if kwargs.get("shown") == "1h":
                raise RuntimeError(failures[0])
            return real_load(self, **kwargs)

        real_load = resource_module.ResourceLoader.load
        monkeypatch.setattr(resource_module.ResourceLoader, "load", broken)
        header = lambda: str(screen.query_one("#resource-status", Static).render())  # noqa: E731
        await pilot.press("w")  # to 1h: its rebuild fails
        assert await _pause_until(pilot, lambda: "rebuild broke" in _error_log_text(widgets_module), seconds=10)
        assert await _pause_until(pilot, lambda: "refresh failed: RuntimeError: rebuild broke" in header(),
                                  seconds=10), header()
        assert app.is_running and screen._resource_range == "1h", "a failed rebuild took the app down"
        assert "refreshing…" not in header(), "a failed rebuild still says it is refreshing"
        # Picking the window again is another try: refreshing… until it answers, then what it found.
        assert held.try_acquire()
        for _ in range(len(RANGES)):  # once round, back to 1h
            await pilot.press("w")
        await pilot.pause(0.2)
        assert screen._resource_range == "1h"
        assert screen._resource_state(screen._resource_kind_data()) == ("refreshing…",), header()
        held.release()
        assert await _pause_until(pilot, lambda: "refresh failed: RuntimeError: rebuild broke" in header(),
                                  seconds=10), header()
        # A refresh whose loads fail too: once it is over, the header names that newer failure.
        failures[0] = "the load broke too"
        await pilot.press("r")
        assert await _pause_until(pilot, lambda: not screen._resource_loading
                                  and "the load broke too" in (screen._resource_error or ""), seconds=15)
        assert "refresh failed: RuntimeError: the load broke too" in header(), header()
        assert "refreshing…" not in header(), header()
        # Once the window can be built, the next refresh builds it, and nothing failed.
        monkeypatch.setattr(resource_module.ResourceLoader, "load", real_load)
        await pilot.press("r")
        assert await _pause_until(pilot, lambda: not screen._resource_loading and screen._resource_error is None
                                  and screen._resource_kind_data().fresh("1h"), seconds=15)
        assert "failed" not in header() and "refreshing…" not in header(), header()
        await _settle(app, pilot)


async def _run_a_window_picked_while_a_refresh_builds(monkeypatch):
    import cortex_training.tui.resource as resource_module

    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=2)
    c = _client()
    c.download_metrics.side_effect = _extending_download(start)
    app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
    async with app.run_test(size=(140, 40)) as pilot:
        assert await _pause_until(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._current_source)
        screen = app.screen
        screen.query_one("#job-tabs", TabbedContent).active = "resource"
        assert await _pause_until(pilot, lambda: c.download_metrics.call_count >= 1 and not screen._resource_loading
                                  and screen._resource_kind_data(), seconds=15)
        assert screen._resource_kind_data().fresh("30m"), "a first open builds every window"
        entered, release = threading.Event(), threading.Event()
        real_load = resource_module.ResourceLoader.load

        def held(self, **kwargs):
            # The refresh's load after its download, building the 10m window it was asked for.
            if kwargs.get("shown") == "10m" and c.download_metrics.call_count >= 2 and not entered.is_set():
                entered.set()
                assert release.wait(10)
            return real_load(self, **kwargs)

        monkeypatch.setattr(resource_module.ResourceLoader, "load", held)
        await pilot.press("r")
        assert await _pause_until(pilot, entered.is_set, seconds=15), "the refresh never loaded"
        await pilot.press("w")  # to 30m, fresh on screen still: nothing to rebuild yet
        await pilot.pause(0.1)
        assert screen._resource_range == "30m"
        release.set()
        assert await _pause_until(pilot, lambda: not screen._resource_loading
                                  and screen._resource_kind_data().fresh("30m"), seconds=15), \
            "the window picked during the refresh was left as of its last showing"
        assert screen._resource_state(screen._resource_kind_data()) == (), "still says it is refreshing"
        assert c.download_metrics.call_count == 2, "rebuilding the window downloaded"
        await _settle(app, pilot)


def test_a_window_picked_while_a_refresh_builds_is_rebuilt_once_it_lands(monkeypatch):
    asyncio.run(_run_a_window_picked_while_a_refresh_builds(monkeypatch))


def _error_log_text(widgets_module) -> str:
    path = Path(widgets_module._ERROR_LOG)
    return path.read_text() if path.exists() else ""


def test_a_window_picked_after_a_refresh_is_rebuilt_without_a_download(tmp_path, monkeypatch):
    asyncio.run(_run_a_window_picked_after_a_refresh(tmp_path, monkeypatch))


async def _run_a_search_with_no_match_on_a_cut_list(c):
    app = CortexTrainingLogTUI(c, poll_interval=0.01)
    async with app.run_test(size=(160, 40)) as pilot:
        assert await _wait(pilot, app, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item,
                           tries=400)
        await pilot.press("slash", *"zzz-no-such-job", "enter")
        assert await _wait(pilot, app, lambda: app.screen._nomatch, tries=200)
        assert "f searches all 2,500 jobs of the schema." in app.screen._nomatch, app.screen._nomatch
        await _settle(app, pilot)


def test_a_search_with_no_match_on_a_cut_list_says_f_searches_them_all():
    now = datetime.now(timezone.utc)
    c = _client()
    c.list_jobs.return_value = [{"job_id": f"j{i}", "status": "SUCCEEDED",
                                 "created_at": (now - timedelta(minutes=i)).isoformat()} for i in range(2_500)]
    asyncio.run(_run_a_search_with_no_match_on_a_cut_list(c))
