# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The job screen's Training metrics tab (``tui/detail_tabs.py``), in the app."""

from __future__ import annotations

import asyncio
import itertools
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.widgets import Static  # noqa: E402
from textual.widgets import TabbedContent  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.logs import LogScreen  # noqa: E402
from cortex_training.tui import training  # noqa: E402
from cortex_training.tui.detail_tabs import TrainingPane  # noqa: E402
from cortex_training.tui.resource import chart_hue  # noqa: E402

TRAINING = "7:training:0"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    import cortex_training.tui.widgets as widgets_module

    monkeypatch.setattr(widgets_module, "_ERROR_LOG", str(tmp_path / "errors.log"))


_SEQ = itertools.count(1)  # each row its own seq: rows with the same (seq_epoch, seq) count once


def _row(step, op="fwd_bwd", call=None, **metrics):
    return {"kind": "training", "record_version": 2, "sub_job_id": TRAINING, "sample_ts": "2026-10-19T10:00:00Z",
            "seq": next(_SEQ), "seq_epoch": "e", "op": op, "step": step, "call": call, "metrics": metrics}


def _download(rows):
    """A ``download_training_metrics`` that commits ``rows`` for the training sub-job."""

    def download(job_id, output_dir, *, resume=False, priority=(), on_commit=None, **_kwargs):
        path = Path(output_dir) / TRAINING / "training.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = "".join(json.dumps(row) + "\n" for row in rows).encode()
        path.write_bytes(data)
        path.with_name(".training.jsonl.manifest.json").write_text(
            json.dumps({"version": 1, "names": ["_metrics/a.gz"], "bytes": len(data)}))
        if on_commit is not None:
            on_commit({"sub_job_id": TRAINING, "saved_path": str(path)})
        return [{"sub_job_id": TRAINING, "saved_path": str(path)}]

    return download


def _client(sub_jobs=None, status="RUNNING"):
    c = MagicMock()
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = []
    c.get_job.return_value = {
        "job_id": "7",
        "status": status,
        "sub_jobs": sub_jobs if sub_jobs is not None else [
            {"sub_job_id": TRAINING, "job_type": "training", "status": status},
            {"sub_job_id": "7:sampling:0", "job_type": "sampling", "status": status},
        ],
    }
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
    # Two steps of two calls each, as the zone saves them: a row per call and a row per step.
    c.download_training_metrics.side_effect = _download([
        _row(1, call=1, avg_loss=2.1, rank=0), _row(1, call=2, avg_loss=1.9, rank=0),
        _row(1, op="step", grad_norm=0.9, last_lr=1e-5, global_steps=1.0, update_successful=1.0),
        _row(2, call=1, avg_loss=1.2, rank=0), _row(2, call=2, avg_loss=0.8, rank=0),
        _row(2, op="step", grad_norm=0.5, last_lr=1e-5, global_steps=2.0, update_successful=0.0),
        _row(None, call=1, avg_loss=0.7, rank=0),  # step 3's first call, before the zone saw a /step
    ])
    return c


async def _wait(pilot, predicate, tries=200):
    for _ in range(tries):
        await pilot.pause(0.01)
        if predicate():
            return True
    return False


async def _settle(app, pilot):
    app.workers.cancel_all()
    for _ in range(5):
        await pilot.pause()


def _text(screen, selector) -> str:
    return str(screen.query_one(selector, Static).render())


async def _open(app, pilot) -> LogScreen:
    assert await _wait(pilot, lambda: isinstance(app.screen, LogScreen) and app.screen._sources_ready)
    return app.screen


def test_training_tab_downloads_and_draws_each_metric():
    async def run():
        c = _client()
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(app, pilot)
            assert not c.download_training_metrics.called  # lazy: only once the tab is shown
            await pilot.press("m")
            pane = screen.query_one("#training-tab")
            assert await _wait(pilot, lambda: pane._downloaded and not pane._loading)
            assert await _wait(pilot, lambda: screen.query_one("#job-tabs", TabbedContent).active == "training")
            assert c.download_training_metrics.call_args.kwargs["priority"] == [TRAINING]
            status = _text(screen, "#training-status")
            assert "training #0" in status and "step 2" in status and "5 calls" in status
            assert "all steps (w)" in status and "(v)" not in status
            # global_steps and update_successful (per step) and rank (per call) are bookkeeping.
            assert "training signals (a)" in status
            assert "1 step reported update_successful ≠ 1" in status
            # Both views on the one tab. Per step first: the step rows' numbers, as saved.
            stats = _text(screen, "#training-step-stats")
            assert stats.splitlines()[0] == "Per step"
            assert [line.split()[0] for line in stats.splitlines()[2:]] == ["grad_norm", "last_lr"]
            assert "0.5 @ 2" in stats  # grad_norm's latest, at step 2
            charts = _text(screen, "#training-step-charts")
            assert "grad_norm · per step · all steps" in charts and "avg_loss" not in charts

            # Then per call: each call's numbers, as saved, numbered in the order they ran with each one's step.
            stats = _text(screen, "#training-call-stats")
            assert stats.splitlines()[0] == "Per call"
            assert "latest @ call (step)" in stats
            assert [line.split()[0] for line in stats.splitlines()[2:]] == ["avg_loss"]
            assert "0.7 @ 5 (next)" in stats  # the newest call: its step is the one still running
            assert "2.1 @ 1 (1)" in stats  # the max: call 1, of step 1
            charts = _text(screen, "#training-call-charts")
            assert "avg_loss · per call · all steps" in charts and "grad_norm" not in charts and "rank" not in charts

            # a shows every saved number, and again hides the bookkeeping.
            await pilot.press("a")
            assert "all metrics (a)" in _text(screen, "#training-status")
            assert [line.split()[0] for line in _text(screen, "#training-step-stats").splitlines()[2:]] == \
                ["grad_norm", "last_lr", "global_steps", "update_successful"]
            assert [line.split()[0] for line in _text(screen, "#training-call-stats").splitlines()[2:]] == \
                ["avg_loss", "rank"]
            await pilot.press("a")
            assert "training signals (a)" in _text(screen, "#training-status")
            assert "rank" not in _text(screen, "#training-call-stats")

            # The window applies to both.
            await pilot.press("w")
            assert "last 100 steps (w)" in _text(screen, "#training-status")
            assert "grad_norm · per step · last 100 steps" in _text(screen, "#training-step-charts")
            assert "avg_loss · per call · last 100 steps" in _text(screen, "#training-call-charts")
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_click_on_the_header_s_window_or_metrics_does_what_w_or_a_does():
    async def run():
        c = _client()
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(200, 40)) as pilot:  # wide enough for the header on one line
            screen = await _open(app, pilot)
            await pilot.press("m")
            pane = screen.query_one("#training-tab")
            assert await _wait(pilot, lambda: pane._downloaded and not pane._loading)

            async def click(words):
                line = _text(screen, "#training-status").splitlines()[0]
                # One cell into the words; the header's padding is one cell.
                await pilot.click("#training-status", offset=(1 + line.index(words) + 1, 0))
                await pilot.pause()

            await click("all steps (w)")
            assert "last 100 steps (w)" in _text(screen, "#training-status")
            assert "grad_norm · per step · last 100 steps" in _text(screen, "#training-step-charts")
            await click("training signals")
            assert "all metrics (a)" in _text(screen, "#training-status")
            assert "global_steps" in _text(screen, "#training-step-stats")
            await click("all metrics")
            assert "training signals (a)" in _text(screen, "#training-status")
            assert "global_steps" not in _text(screen, "#training-step-stats")
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_section_whose_every_metric_is_hidden_says_so():
    async def run():
        c = _client()
        # The calls carry only rank, which is 0 throughout: hidden, though the calls were saved.
        c.download_training_metrics.side_effect = _download([
            _row(1, call=1, rank=0), _row(1, op="step", grad_norm=0.9), _row(2, call=1, rank=0),
        ])
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(app, pilot)
            await pilot.press("m")
            pane = screen.query_one("#training-tab")
            assert await _wait(pilot, lambda: pane._downloaded and not pane._loading)
            assert "grad_norm" in _text(screen, "#training-step-stats")
            calls = _text(screen, "#training-call-stats")
            assert "1 metric, all hidden as bookkeeping (a shows every metric)" in calls
            assert "saved no calls" not in calls
            await pilot.press("a")
            assert "rank" in _text(screen, "#training-call-stats")
            await _settle(app, pilot)

    asyncio.run(run())


def _name_styles(text) -> dict[str, str]:
    """Each metric name's style in a stats table (its first span on a line)."""
    plain = text.plain
    return {plain[span.start:span.end].strip(): str(span.style) for span in text.spans
            if plain[span.start:span.end].strip() and plain[span.start - 1:span.start] in ("", "\n")}


def test_a_metric_keeps_its_colour_when_every_metric_is_shown():
    state = training._SubJobState(TRAINING)
    for step in (1, 2):
        state.ingest(_row(step, op="step", grad_norm=0.9, global_steps=float(step), weight_delta_norm_rank0_sample=0.1))
    sub_job = state.snapshot()
    pane = TrainingPane(MagicMock(), "7", Path("."), job=dict, download=lambda **_: True)
    hidden = _name_styles(pane._stats_text(sub_job, training.VIEW_STEP, None, 120))
    pane._everything = True
    shown = _name_styles(pane._stats_text(sub_job, training.VIEW_STEP, None, 120))
    assert "global_steps" not in hidden and "global_steps" in shown
    # weight_delta is the third metric either way, so it has the third colour either way.
    assert hidden["weight_delta_norm_rank0_sample"] == shown["weight_delta_norm_rank0_sample"] == chart_hue(2)


def test_training_tab_download_failure_is_said_in_its_header():
    async def run():
        c = _client()
        c.download_training_metrics.side_effect = RuntimeError("stage [/] gone")
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(app, pilot)
            await pilot.press("m")
            pane = screen.query_one("#training-tab")
            assert await _wait(pilot, lambda: pane._error is not None and not pane._loading)
            assert "training metrics unavailable: RuntimeError: stage [/] gone" in _text(screen, "#training-status")
            await _settle(app, pilot)

    asyncio.run(run())


def test_training_tab_of_a_job_without_training_downloads_nothing():
    async def run():
        c = _client(sub_jobs=[{"sub_job_id": "7:sampling:0", "job_type": "sampling", "status": "RUNNING"}])
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(app, pilot)
            await pilot.press("m")
            assert await _wait(pilot, lambda: "no training sub-job" in _text(screen, "#training-step-stats"))
            assert not c.download_training_metrics.called
            await _settle(app, pilot)

    asyncio.run(run())


def test_t_cycles_through_every_tab():
    async def run():
        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(app, pilot)
            tabs = screen.query_one("#job-tabs", TabbedContent)
            # Each press is waited for, and the tab it lands on settled, before the next: one pause
            # can end before a loaded run has handled the key.
            for expected in ("resource", "training", "checkpoints", "logs"):
                await pilot.press("t")
                assert await _wait(pilot, lambda: tabs.active == expected and screen._active_tab == expected), \
                    f"t did not reach {expected} (on {tabs.active})"
            await _settle(app, pilot)

    asyncio.run(run())


def test_focusing_a_widget_in_another_tab_does_not_switch_tabs():
    # TabbedContent switches to whichever pane a widget is focused in, when it handles that message,
    # which can be after a later tab key: the screen focuses its Logs panel as it opens, and an `m`
    # pressed in that moment was switched straight back to Logs. Focusing the Logs panel while
    # Training is shown is that message, arriving late, made deterministic.
    async def run():
        app = CortexTrainingLogTUI(_client(), "7", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(app, pilot)
            tabs = screen.query_one("#job-tabs", TabbedContent)
            await pilot.press("m")
            assert await _wait(pilot, lambda: tabs.active == "training")
            screen.set_focus(screen.query_one("#sources"))
            for _ in range(10):
                await pilot.pause(0.01)
            assert tabs.active == "training"
            await _settle(app, pilot)

    asyncio.run(run())


def test_opens_on_a_new_tab_when_asked():
    async def run():
        c = _client()
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open(app, pilot)
            app.push_screen(LogScreen(c, "7", initial_tab="training", poll_interval=0.01))
            assert await _wait(pilot, lambda: app.screen.query_one("#training-tab")._downloaded)
            assert app.screen.query_one("#job-tabs", TabbedContent).active == "training"
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_job_saved_in_the_earlier_format_says_so_instead_of_nothing_saved():
    async def run():
        c = _client()
        old = {"kind": "training", "record_version": 1, "sub_job_id": TRAINING, "op": "fwd_bwd", "step": 1,
               "metrics": {"sum_loss": 4.0, "num_calls": 2.0}}
        c.download_training_metrics.side_effect = _download([old, {**old, "op": "step", "metrics": {}}])
        app = CortexTrainingLogTUI(c, "7", poll_interval=0.01)
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open(app, pilot)
            await pilot.press("m")
            pane = screen.query_one("#training-tab")
            assert await _wait(pilot, lambda: pane._downloaded and not pane._loading)
            assert "saved in an earlier format" in _text(screen, "#training-step-stats")
            assert "2 rows in an earlier format not shown" in _text(screen, "#training-status")
            await _settle(app, pilot)

    asyncio.run(run())
