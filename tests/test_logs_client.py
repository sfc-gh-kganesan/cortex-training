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

"""Client side of the TUI's Logs and Resources tabs: one sub-job's execution
log, and the order and per-file reports of a GPU metrics download."""

import gzip

import pytest

from tests import test_client as _client_tests

_make_client = _client_tests._make_client
# The stage-SQL fake of the execution-log tests (bound to a name pytest won't collect).
_FakeArtifactConnection = _client_tests.TestExecutionLogDownload.FakeArtifactConnection

RUN_URI = "snow://experiment/DB.SCH.EXP/versions/RUN_ABC/"


def _staged(monkeypatch, root, files):
    """A client whose job's run stages ``files`` (path under ``root``: content)."""
    client = _make_client()
    connection = _FakeArtifactConnection(
        {RUN_URI + root + "/": [(f"/versions/RUN_ABC/{root}/{path}",) for path in files]},
        {RUN_URI + f"{root}/{path}": content for path, content in files.items()},
    )
    monkeypatch.setattr(client, "_experiment_run_uri", lambda _: (RUN_URI, "RUN_ABC"))
    monkeypatch.setattr(client, "_open_experiment_artifact_connection", lambda: connection)
    return client, connection


def _chunks(name):
    """Gzip chunks a second apart: three for the sampling sub-job, two for training."""
    return {
        f"{sub}/{name}.20260904-12000{i}.c{i}.gz": gzip.compress(f"{sub} {i}\n".encode())
        for sub, count in (("job-1:sampling:0", 3), ("job-1:training:0", 2))
        for i in range(count)
    }


def test_fetch_execution_logs_can_fetch_one_sub_jobs_file(monkeypatch):
    client, connection = _staged(monkeypatch, "_logs", {
        "job-1:training:0/execution.jsonl": b"training\n",
        "job-1:training:0/server.log": b"server\n",
        "job-1:sampling:0/execution.jsonl": b"sampling\n",
    })
    logs = client.fetch_execution_logs("job-1", sub_job_id="job-1:training:0", filename="execution.jsonl")
    assert [(log["sub_job_id"], log["filename"], log["content"]) for log in logs] == [
        ("job-1:training:0", "execution.jsonl", "training\n")
    ]
    assert sum(command.startswith("GET ") for command in connection.commands) == 1  # the rest are only listed


def test_download_metrics_puts_priority_first_and_reports_each_commit(tmp_path, monkeypatch):
    client, connection = _staged(monkeypatch, "_metrics", _chunks("gpu"))
    commits = []

    def on_commit(entry):
        saved = tmp_path / entry["sub_job_id"] / "gpu.jsonl"
        commits.append((entry["sub_job_id"], entry["chunk_count"], saved.read_text().count("\n")))

    result = client.download_metrics("job-1", tmp_path, priority=["job-1:training:0"], on_commit=on_commit)
    # The sub-job named first is written first, and each file is complete when it is reported.
    assert commits == [("job-1:training:0", 2, 2), ("job-1:sampling:0", 3, 3)]
    assert [entry["sub_job_id"] for entry in result] == ["job-1:sampling:0", "job-1:training:0"]  # still id order
    saved = (tmp_path / "job-1:sampling:0" / "gpu.jsonl").read_text()
    assert saved == "".join(f"job-1:sampling:0 {i}\n" for i in range(3))
    assert sum(command.startswith("GET ") for command in connection.commands) == 5  # each chunk fetched once


def test_fetch_execution_logs_still_fetches_every_file_by_default(monkeypatch):
    client, connection = _staged(monkeypatch, "_logs", {
        "job-1:training:0/execution.jsonl": b"training\n",
        "job-1:sampling:0/execution.jsonl": b"sampling\n",
    })
    logs = client.fetch_execution_logs("job-1")
    assert sorted(log["sub_job_id"] for log in logs) == ["job-1:sampling:0", "job-1:training:0"]
    assert sum(command.startswith("GET ") for command in connection.commands) == 2


def test_download_metrics_refuses_a_single_string_as_priority(tmp_path, monkeypatch):
    client, _ = _staged(monkeypatch, "_metrics", _chunks("gpu"))
    with pytest.raises(TypeError, match="priority must be an iterable"):
        client.download_metrics("job-1", tmp_path, priority="job-1:training:0")


def test_download_metrics_reports_a_sub_job_with_nothing_new_on_resume(tmp_path, monkeypatch):
    client, connection = _staged(monkeypatch, "_metrics", _chunks("gpu"))
    client.download_metrics("job-1", tmp_path, resume=True)
    gets = sum(command.startswith("GET ") for command in connection.commands)
    commits = []

    client.download_metrics("job-1", tmp_path, resume=True, on_commit=lambda entry: commits.append(entry["sub_job_id"]))
    # Every sub-job is reported, though nothing new was fetched for either.
    assert commits == ["job-1:sampling:0", "job-1:training:0"]
    assert sum(command.startswith("GET ") for command in connection.commands) == gets


def test_an_error_from_on_commit_stops_the_download_and_keeps_what_is_saved(tmp_path, monkeypatch):
    client, _ = _staged(monkeypatch, "_metrics", _chunks("gpu"))

    def on_commit(entry):
        raise RuntimeError(f"stop after {entry['sub_job_id']}")

    with pytest.raises(RuntimeError, match="stop after job-1:training:0") as raised:
        client.download_metrics("job-1", tmp_path, priority=["job-1:training:0"], on_commit=on_commit)
    assert getattr(raised.value, "download_progress_saved", False)
    assert (tmp_path / "job-1:training:0" / "gpu.jsonl").read_text().count("\n") == 2
    assert not (tmp_path / "job-1:sampling:0" / "gpu.jsonl").exists()
