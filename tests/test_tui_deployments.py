# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deployments in the TUI: telling them apart, the list's two views, the preview and the job screen's tabs."""

from __future__ import annotations

import asyncio
from datetime import datetime
from datetime import timezone
from unittest.mock import MagicMock

import pytest

pytest.importorskip("textual")

from textual.widgets import Static  # noqa: E402

from cortex_training import AccountJobListUnavailable  # noqa: E402
from cortex_training.tui import views as v  # noqa: E402
from cortex_training.tui.app import CortexTrainingLogTUI  # noqa: E402
from cortex_training.tui.job_list import JobListScreen  # noqa: E402

NOW = datetime(2026, 10, 26, 18, 0, tzinfo=timezone.utc)
UUID = "00000000-0000-4000-8000-000000000008"  # a made-up canonical UUID
DEPLOY = "8:sampling:0"


def _deployment(status="RUNNING", job_id="8", cls="EVALUATION", **extra):
    return {"job_id": job_id, "status": status, "idle_timeout_seconds": 0, **extra,
            "sub_jobs": [{"sub_job_id": f"{job_id}:sampling:0", "job_type": "sampling", "status": status,
                          "model_name": "org/model",
                          "inference_config": {"deployment_class": cls, "n_gpus": 1.0, "num_replicas": 2.0}}]}


def _sampling():
    return {"job_id": "9", "status": "SUCCEEDED",
            "sub_jobs": [{"sub_job_id": "9:sampling:0", "job_type": "sampling", "inference_config": {"n_gpus": 1.0}}]}


def _training():
    return {"job_id": "7", "status": "RUNNING",
            "sub_jobs": [{"sub_job_id": "7:training:0", "job_type": "training"},
                         {"sub_job_id": "7:sampling:0", "job_type": "sampling", "inference_config": {}}]}


def test_the_cli_s_rule_tells_a_deployment_apart():
    assert v.is_deployment(_deployment())
    assert not v.is_deployment(_deployment(cls="PRODUCTION"))  # not one until the CLI's rule says so
    assert not v.is_deployment(_sampling())
    assert not v.is_deployment(_training())
    assert not v.is_deployment({"job_id": "1"})  # an old record without sub-jobs stays a job
    assert not v.is_deployment({"job_id": "1", "sub_jobs": [{"inference_config": None}]})
    assert not v.is_deployment("not a job")
    assert v.deployment_class(_deployment()) == "EVALUATION"
    assert v.deployment_class(_training()) == ""
    assert v.deployment_sub_job_id(_deployment()) == DEPLOY
    assert v.deployment_sub_job_id(_training()) is None


def test_an_unknown_deployment_without_sub_job_ids_still_names_its_sub_job():
    job = _deployment(status="UNKNOWN")
    del job["sub_jobs"][0]["sub_job_id"]
    assert v.is_deployment(job)
    assert v.deployment_sub_job_id(job) == DEPLOY  # the id views.sub_jobs derives, as the Logs tab shows it


def test_a_deployment_s_status_reads_in_deployment_words():
    assert v.status_word_for(_deployment(), "RUNNING") == "ready"
    assert v.status_word_for(_deployment(), "JOB_STATE_PLACING") == "initializing"
    assert v.status_word_for(_deployment(), "CANCELLED") == "dropped"
    # No status yet is not the service's UNKNOWN.
    assert v.status_word_for(_deployment(), "") == "?" and v.status_word_for(_deployment(), None) == "?"
    assert v.status_word_for(_deployment(), "UNKNOWN") == "unknown"
    assert v.status_word_for(_training(), "RUNNING") == "running"
    assert v.status_word_for(_training(), "") == "?"


def test_a_deployment_reads_deployment_in_the_type_column_and_search():
    job = _deployment()
    assert v.job_type_label(job) == "deployment"
    assert v.job_type_label(_sampling()) == "sampling"
    assert v.job_matches_query(job, v.parse_query("type:deployment"))
    assert not v.job_matches_query(_sampling(), v.parse_query("type:deployment"))
    assert v.job_matches_query(job, v.parse_query("class:evaluation"))
    assert not v.job_matches_query(_training(), v.parse_query("class:evaluation"))
    # The words the view shows match as free text too.
    for word in ("ready", "evaluation"):
        assert v.job_matches_query(job, v.parse_query(word)), word
    assert v.job_matches_query(_deployment("CANCELLED"), v.parse_query("dropped"))
    for query in ("status:ready", "status:running"):  # the word its row shows, and the job status
        assert v.job_matches_query(job, v.parse_query(query)), query
    assert not v.job_matches_query(_training(), v.parse_query("status:ready"))
    assert not v.job_matches_query(_training(), v.parse_query("ready"))


def test_a_refresh_without_sub_jobs_keeps_a_deployment_one():
    fresh = {"job_id": "8", "status": "UNKNOWN"}
    assert v.is_deployment(v.refresh_job(_deployment(), fresh))
    assert v.refresh_job(_deployment(), fresh)["status"] == "UNKNOWN"
    assert "status" not in v.refresh_job(_deployment(), fresh)["sub_jobs"][0]  # no longer known
    assert "sub_jobs" not in v.refresh_job(_training(), fresh)  # a job's sub-jobs go, as before


def test_a_list_refresh_without_sub_jobs_keeps_a_deployment_in_its_view():
    c = _client([_deployment()])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._nomatch)
            screen = app.screen
            [merged] = screen._merged([{"job_id": "8", "status": "UNKNOWN"}])
            assert v.is_deployment(merged) and merged["status"] == "UNKNOWN"
            assert "status" not in merged["sub_jobs"][0]
            [merged] = screen._merged([{"job_id": "8", "status": "UNKNOWN", "sub_jobs": []}])
            assert v.is_deployment(merged)
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_deployment_s_count_and_sub_job_say_its_status_as_its_row_does():
    from cortex_training.tui.job_list import _status_chips
    from cortex_training.tui.logs import LogScreen

    chips = str(_status_chips([_deployment(), _deployment("PENDING")], 0))
    assert "1 ready" in chips and "running" not in chips, chips
    assert "1 running" in str(_status_chips([_training()], 0))
    # PLACING and INITIALIZING both read "initializing": one chip, whichever comes first.
    for order in ((_deployment("PLACING"), _deployment("INITIALIZING")),
                  (_deployment("INITIALIZING"), _deployment("PLACING"))):
        chips = str(_status_chips(list(order), 0))
        assert chips.count("initializing") == 1 and "2 initializing" in chips, chips
    screen = LogScreen(MagicMock(), "8", job=_deployment())
    assert str(screen._source_text(DEPLOY)).endswith("ready")
    assert str(LogScreen(MagicMock(), "7", job=_training())._source_text("7:training:0")).endswith("running")
    # The Logs tab's waiting notes word it as the badge does.
    kind, note = LogScreen(MagicMock(), "8", job=_deployment("PLACING"))._waiting_for(DEPLOY)
    assert kind == "starting" and "is INITIALIZING" in note and "PLACING" not in note, note
    placing = {**_training(), "status": "PLACING"}
    assert "is PLACING" in LogScreen(MagicMock(), "7", job=placing)._waiting_for("7:training:0")[1]
    for job, sub_job_id, then in ((_deployment(), DEPLOY, "ready"), (_training(), "7:training:0", "running")):
        screen = LogScreen(MagicMock(), job["job_id"], job=job)
        screen._current_source, screen._set_waiting = sub_job_id, MagicMock()
        screen._wait_after_error(screen._tail_gen, sub_job_id, "cancelling")
        assert screen._set_waiting.call_args.args[1].endswith(f"once it is {then}")


def test_preview_facts_of_a_deployment():
    facts = dict(v.preview_facts(_deployment(job_id=UUID), NOW, base_url="https://HOST/x/inference/v1"))
    assert facts["Type"] == "deployment"
    assert facts["Class"] == "EVALUATION"
    assert facts["Replicas"] == "2"
    assert facts["Idle timeout"] == "off"  # 0 disables reclamation, it is not a zero-second timeout
    assert facts["Call as"] == f"sf-ct-{UUID}"
    assert facts["Base URL"] == "https://HOST/x/inference/v1"
    assert dict(v.preview_facts(_deployment(idle_timeout_seconds=1800), NOW))["Idle timeout"] == "30m 00s"
    unset = _deployment()
    del unset["idle_timeout_seconds"]
    assert dict(v.preview_facts(unset, NOW))["Idle timeout"] == "server default"
    assert "Base URL" not in dict(v.preview_facts(_deployment(), NOW))
    # The CLI's rule: a deployment that failed or is dropped no longer serves at that URL.
    for status in ("FAILED", "CANCELLING", "CANCELLED", "TERMINATED"):
        assert "Base URL" not in dict(v.preview_facts(_deployment(status), NOW, base_url="https://HOST/v1")), status
    assert "Base URL" in dict(v.preview_facts(_deployment("PENDING"), NOW, base_url="https://HOST/v1"))
    # The client's helper refuses an ID that isn't a canonical UUID; the TUI then names no model.
    assert "Call as" not in dict(v.preview_facts(_deployment(job_id="8"), NOW))
    assert "Class" not in dict(v.preview_facts(_training(), NOW))


def test_base_url_comes_from_the_client():
    from cortex_training.client import CortexTrainingClient
    from cortex_training.tui.untrusted import printable_client

    real = CortexTrainingClient(base_url="https://HOST", database="DB", schema="PUBLIC")
    held = printable_client(real)  # as the TUI holds it
    assert held.model_deployment_base_url == real.model_deployment_base_url
    other = held.for_job({"database_name": "DB", "schema_name": "OTHER"})
    assert other.model_deployment_base_url == (
        "https://HOST/api/v2/databases/DB/schemas/OTHER/cortex-training/inference/v1")


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_TRAINING_TUI_CACHE_DIR", str(tmp_path / "cache"))
    import cortex_training.tui.widgets as widgets_module

    monkeypatch.setattr(widgets_module, "_ERROR_LOG", str(tmp_path / "errors.log"))


def _client(jobs, job=None):
    c = MagicMock()
    c.model_deployment_base_url = "https://HOST/api/v2/databases/DB/schemas/PUBLIC/cortex-training/inference/v1"
    c.list_jobs_page.side_effect = AccountJobListUnavailable("not deployed", reason="not_deployed")
    c.list_jobs.return_value = jobs
    c.get_job.return_value = job if job is not None else jobs[0]
    c.tail_logs.return_value = {"entries": [], "next_cursor": "", "eof": True}
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


def _ids(screen) -> set:
    return {screen._jobs_by_key[key]["job_id"] for key in screen._job_by_item.values()}


def _info(screen) -> str:
    return str(screen.query_one("#listinfo", Static).render())


def test_d_switches_between_jobs_and_deployments():
    c = _client([_training(), _sampling(), _deployment(job_id="8"), _deployment(job_id="10")])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            assert _ids(screen) == {"7", "9"}  # opens on Jobs
            assert "2 deployments · d shows them" in _info(screen)
            await pilot.press("d")
            assert await _wait(pilot, lambda: _ids(screen) == {"8", "10"})
            assert "2 jobs · d shows them" in _info(screen)
            assert "deployments" in app.sub_title
            screen._switching_text = lambda: "reading every schema…"  # a pressed, the other list still read
            screen._update_subtitle()
            assert app.sub_title.startswith("deployments  ·  reading every schema…"), app.sub_title
            del screen._switching_text
            await pilot.press("d")
            assert await _wait(pilot, lambda: _ids(screen) == {"7", "9"})
            await _settle(app, pilot)

    asyncio.run(run())


def test_an_empty_view_points_to_the_other_one():
    c = _client([_deployment()])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._nomatch)
            assert "d shows the 1 deployment" in app.screen._nomatch
            await _settle(app, pilot)

    asyncio.run(run())


def test_an_empty_view_of_a_list_cut_short_says_more_may_exist():
    from cortex_training.tui.account import OLDER, PAGE_LIMIT, SCHEMA, ListScope

    c = _client([_deployment()])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._nomatch)
            screen = app.screen
            for scope, said in ((ListScope(SCHEMA, loading=OLDER, walked=1), "still loading"),
                                (ListScope(SCHEMA, stopped=PAGE_LIMIT, pages=2), "more may exist")):
                screen._scope = scope
                screen._render_jobs()
                assert said in screen._nomatch and "last 6 months" not in screen._nomatch, screen._nomatch
                assert "no training or sampling jobs to show" in screen._nomatch, screen._nomatch
                assert screen._nomatch.endswith("\n  d shows the 1 deployment."), screen._nomatch
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_search_covers_only_the_view_on_screen():
    c = _client([_training(), _deployment()])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            await pilot.press("d")
            assert await _wait(pilot, lambda: _ids(screen) == {"8"})
            # Both jobs have a sampling sub-job: only the one in this view may show.
            assert v.job_matches_query(_training(), v.parse_query("sampling"))
            await pilot.press("/", *"sampling", "enter")
            assert await _wait(pilot, lambda: screen._filter == "sampling" and screen._job_by_item)
            await pilot.pause(0.05)
            assert _ids(screen) == {"8"}
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_deployment_s_row_shows_its_deployment_status():
    c = _client([_training(), _deployment()])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            cells = screen._job_cells(_training(), NOW)[0]
            assert "RUNNING" in str(cells["Status"])
            cells = screen._job_cells(_deployment(), NOW)[0]
            assert "READY" in str(cells["Status"]) and "RUNNING" not in str(cells["Status"])
            await _settle(app, pilot)

    asyncio.run(run())


def test_a_search_that_matches_only_the_other_view_points_to_it():
    c = _client([_training(), _sampling(), _deployment(job_id="8"), _deployment(job_id="10"),
                 _deployment(job_id="11")])

    async def search(pilot, screen, text):
        await pilot.press("escape")  # clear the last one
        await pilot.press("/", *text, "enter")
        assert await _wait(pilot, lambda: screen._filter == text and screen._nomatch)
        return screen._nomatch

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            assert "d shows the 3 matching deployments." in await search(pilot, screen, "type:deployment")
            assert "d shows the 1 matching deployment." in await search(pilot, screen, "10")
            await pilot.press("escape")
            await pilot.press("d")
            assert await _wait(pilot, lambda: _ids(screen) == {"8", "10", "11"})
            assert "d shows the 1 matching job." in await search(pilot, screen, "7")
            await _settle(app, pilot)

    asyncio.run(run())


def _located(job, db, schema):
    return {**job, "database_name": db, "schema_name": schema}


def test_the_all_in_hint_names_the_location_of_the_view_on_screen():
    from cortex_training.tui.account import ACCOUNT, ListScope

    c = _client([_training(), _deployment()])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            screen._scope = ListScope(ACCOUNT)
            screen._all_jobs = [_located(_training(), "DB", "B"), _located(_deployment(), "DB", "A")]
            assert screen._scope_text() == "all schemas (all in DB.B)"
            await pilot.press("d")
            assert await _wait(pilot, lambda: screen._deployments)
            assert screen._scope_text() == "all schemas (all in DB.A)"
            await _settle(app, pilot)

    asyncio.run(run())


def test_e_folds_when_every_job_in_the_view_is_unfolded():
    c = _client([_training(), _deployment("CANCELLED")])  # the ended deployment starts folded

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            training = screen._jobs_by_key[next(k for k, j in screen._jobs_by_key.items() if j["job_id"] == "7")]
            assert screen._is_expanded(training)
            await pilot.press("e")
            assert await _wait(pilot, lambda: not screen._is_expanded(training))
            await _settle(app, pilot)

    asyncio.run(run())


def test_e_in_one_view_leaves_the_other_view_s_folds_be():
    c = _client([_training(), _deployment()])  # both live: both start unfolded

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._job_by_item)
            screen = app.screen
            job = {j["job_id"]: j for j in screen._jobs_by_key.values()}
            await pilot.press("d")
            assert await _wait(pilot, lambda: screen._deployments)
            await pilot.press("e")
            assert await _wait(pilot, lambda: not screen._is_expanded(job["8"]))
            assert screen._is_expanded(job["7"])
            later = {**_training(), "job_id": "12"}  # a job of the other view listed after it (a later page)
            assert screen._is_expanded(later)
            await _settle(app, pilot)

    asyncio.run(run())


def test_an_empty_view_keeps_the_scope_s_words_and_the_widen_hint():
    from cortex_training.tui.account import ACCOUNT, SCHEMA, ListScope

    c = _client([_deployment()])

    async def run():
        app = CortexTrainingLogTUI(c, poll_interval=0.01)
        async with app.run_test(size=(160, 45)) as pilot:
            assert await _wait(pilot, lambda: isinstance(app.screen, JobListScreen) and app.screen._nomatch)
            screen = app.screen
            screen._scope = ListScope(SCHEMA)
            screen._render_jobs()
            said = screen._nomatch
            assert said.startswith("No training or sampling jobs created in the last 6 months in "), said
            assert "that this role can see." in said and "\n  a lists every database and schema." in said, said
            assert said.endswith("\n  d shows the 1 deployment."), said
            screen._scope = ListScope(ACCOUNT)
            screen._render_jobs()
            said = screen._nomatch
            assert "that this role can see, in any database and schema." in said, said
            assert said.endswith("\n  d shows the 1 deployment."), said
            await _settle(app, pilot)

    asyncio.run(run())


def test_one_helper_says_which_tabs_a_job_has(monkeypatch):
    from cortex_training.tui.logs import LogScreen
    from cortex_training.tui.screens import JobSummaryScreen

    assert v.job_tabs(_deployment()) == ("serving", "logs", "resource")
    assert v.job_tabs(_training()) == ("logs", "resource", "training", "checkpoints")
    # The job screen's tabs and the preview's keys both follow it.
    monkeypatch.setattr(v, "job_tabs", lambda job: ("logs", "checkpoints"))
    log_screen = LogScreen(MagicMock(), "8", job=_deployment())
    assert log_screen._tabs() == ("logs", "checkpoints")
    preview = JobSummaryScreen(MagicMock(), "8", job=_deployment())
    assert [preview.check_action("open", (tab,)) for tab in ("serving", "logs", "checkpoints")] == [False, True, True]
    assert preview._actions() == (("l", "Logs"), ("c", "Checkpoints"))
