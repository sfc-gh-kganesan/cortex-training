# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end model deployment tests against a live Cortex Training account.

Cases TC1-TC11 cover the deployment lifecycle (dry run, create, list,
describe, wait, calling the endpoint, downloads, checkpoint sources, drop,
errors, and interrupts). Every command runs the installed CLI in a subprocess,
as a user would, and deployments hold real GPUs while they run. They are skipped unless
``CORTEX_TRAINING_E2E_CONFIG`` names a CLI config JSON file
(see ``docs/reference/cli.md``)::

    CORTEX_TRAINING_E2E_CONFIG=~/config.json python3 -m pytest \\
        tests/test_model_deployment_e2e.py -v

TC8 deploys a weights-only checkpoint. Set
``CORTEX_TRAINING_E2E_CHECKPOINT=TRAINING_JOB_UUID/CHECKPOINT_ID`` to use an
existing one; otherwise a 1-GPU training job saves one and is cancelled.

Every deployment a test creates is dropped when the module finishes, even if a
test fails. Run ``model-deployment list`` afterwards if the run was killed.
"""

from __future__ import annotations

import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

CONFIG_ENV = "CORTEX_TRAINING_E2E_CONFIG"
CHECKPOINT_ENV = "CORTEX_TRAINING_E2E_CHECKPOINT"
CONFIG_PATH = os.environ.get(CONFIG_ENV)

pytestmark = pytest.mark.skipif(
    not CONFIG_PATH, reason=f"set {CONFIG_ENV} to run live model deployment tests"
)

MODEL = "Qwen/Qwen3-1.7B"
DEPLOYMENT_ID_PATTERN = re.compile(r"^sf-ct-[0-9a-f-]{36}$")
UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
MISSING_DEPLOYMENT = "sf-ct-00000000-0000-4000-8000-000000000000"
# Generous: a cold start pulls the model and builds the vLLM engine.
READY_TIMEOUT_SECONDS = 1800


def base_spec(**deployment_overrides) -> dict:
    return {
        "deployment": {
            "class": "EVALUATION",
            "min_replicas": 1,
            "gpus_per_replica": 1,
            "gpu_type": "H200",
            **deployment_overrides,
        },
        "source": {"type": "base_model", "model": MODEL},
        "inference": {"max_seq_len": 8192},
    }


def load_config() -> dict:
    return json.loads(Path(CONFIG_PATH).expanduser().read_text())


def cli_command(*arguments: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "cortex_training.cli",
        "--config",
        str(Path(CONFIG_PATH).expanduser()),
        "--compact",
        *arguments,
    ]


def run_cli(*arguments: str, timeout: float = READY_TIMEOUT_SECONDS + 120):
    return subprocess.run(
        cli_command(*arguments), capture_output=True, text=True, timeout=timeout
    )


def run_ok(*arguments: str, **kwargs) -> dict:
    result = run_cli(*arguments, **kwargs)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def run_error(*arguments: str) -> str:
    """Run a command that must fail and return its stderr, checked for leaks."""
    result = run_cli(*arguments, timeout=300)
    assert result.returncode != 0, result.stdout
    assert result.stderr.startswith("error: "), result.stderr
    assert_no_internals(result.stderr)
    return result.stderr


def assert_no_internals(text: str) -> None:
    """A user-facing message must not show request URLs or tracebacks."""
    assert "Traceback" not in text, text
    assert "/api/v2/" not in text, text
    assert ":sampling:" not in text, text


def assert_only_deployment_ids(payload: object, deployment_id: str) -> None:
    """Every UUID in ``payload`` must belong to the deployment ID, not a backing job."""
    text = json.dumps(payload)
    deployment_uuid = deployment_id.removeprefix("sf-ct-")
    for found in UUID_PATTERN.findall(text):
        assert found == deployment_uuid, f"internal ID {found} exposed in {text}"


def write_spec(directory: Path, name: str, spec: dict) -> str:
    path = directory / name
    path.write_text(json.dumps(spec))
    return str(path)


@pytest.fixture(scope="module")
def created():
    """Deployment IDs this module created; each is dropped at teardown."""
    ids: list[str] = []
    yield ids
    failures = []
    for deployment_id in ids:
        result = run_cli("model-deployment", "drop", deployment_id, timeout=300)
        if result.returncode != 0:
            failures.append(f"{deployment_id}: {result.stderr.strip()}")
    if failures:
        pytest.fail("could not drop, drop these by hand:\n" + "\n".join(failures))


@pytest.fixture(scope="module")
def spec_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("specs")


@pytest.fixture(scope="module")
def ready_deployment(created, spec_dir) -> dict:
    """TC2: a base-model deployment created with ``create --wait``."""
    spec = write_spec(spec_dir, "spec-base.json", base_spec())
    result = run_cli(
        "model-deployment", "create", "--spec", spec, "--wait",
        "--timeout", str(READY_TIMEOUT_SECONDS),
    )
    # The summary is printed even if waiting fails, so record the ID first.
    summary = json.loads(result.stdout) if result.stdout.strip() else {}
    if summary.get("deployment"):
        created.append(summary["deployment"])
    assert result.returncode == 0, result.stderr
    return summary


def test_tc1_dry_run_prints_the_spec_and_creates_nothing(spec_dir):
    spec = base_spec(gpu_type="h200")
    printed = run_ok(
        "model-deployment", "create",
        "--spec", write_spec(spec_dir, "dry-run.json", spec), "--dry-run",
    )
    assert printed == base_spec()


def test_tc2_create_wait_reaches_ready(ready_deployment):
    assert set(ready_deployment) == {"base_url", "deployment", "status"}
    assert DEPLOYMENT_ID_PATTERN.match(ready_deployment["deployment"])
    assert ready_deployment["status"] == "READY"
    assert ready_deployment["base_url"].endswith("/cortex-training/inference/v1")


def test_tc3_list_shows_id_and_status_and_filters(ready_deployment):
    deployment_id = ready_deployment["deployment"]
    listed = run_ok("model-deployment", "list")["deployments"]
    assert {"deployment": deployment_id, "status": "READY"} in listed
    assert all(set(entry) == {"deployment", "status"} for entry in listed)

    ready = run_ok("model-deployment", "list", "--status", "ready")["deployments"]
    assert {"deployment": deployment_id, "status": "READY"} in ready
    assert {entry["status"] for entry in ready} == {"READY"}

    dropped = run_ok("model-deployment", "list", "--status", "DROPPED")["deployments"]
    assert deployment_id not in {entry["deployment"] for entry in dropped}


def test_tc4_describe_returns_the_full_record_without_internal_ids(ready_deployment):
    deployment_id = ready_deployment["deployment"]
    described = run_ok("model-deployment", "describe", deployment_id)
    config = load_config()

    assert described["deployment"] == deployment_id
    assert described["status"] == "READY"
    assert described["base_url"] == ready_deployment["base_url"]
    assert described["database"].upper() == config["database"].upper()
    assert described["schema"].upper() == config["schema"].upper()
    for field in ("created_at", "creator", "owner_role"):
        assert described.get(field), f"describe has no {field}: {described}"
    assert described["spec"] == base_spec()
    assert_only_deployment_ids(described, deployment_id)


def test_tc5_wait_returns_the_create_wait_summary(ready_deployment):
    started = time.monotonic()
    waited = run_ok("model-deployment", "wait", ready_deployment["deployment"])
    assert waited == ready_deployment
    assert time.monotonic() - started < 60


def test_tc6_base_url_serves_openai_chat_completions(ready_deployment):
    config = load_config()
    pat = config.get("pat") or os.environ.get("CORTEX_TRAINING_PAT")
    if not pat:
        pytest.skip("the config has no PAT to call the deployment with")
    headers = {
        "Authorization": f"Bearer {pat}",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    }
    base_url = ready_deployment["base_url"]
    deployment_id = ready_deployment["deployment"]

    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json={"model": deployment_id, "messages": [{"role": "user", "content": "Hello"}]},
        timeout=300,
    )
    if response.status_code == 404:
        pytest.skip("inference is not enabled on this account (HTTP 404)")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == deployment_id
    assert body["choices"][0]["message"]["content"]


@pytest.mark.xfail(
    reason="GET /models has returned an empty data list for a READY deployment; "
    "see docs/reference/rest-api.md section 15, item 6",
    strict=False,
)
def test_tc6_models_lists_the_ready_deployment(ready_deployment):
    from recipes.model_deployment.streaming import deployment_headers

    models = requests.get(
        f"{ready_deployment['base_url']}/models",
        headers=deployment_headers(deployment_pat()),
        timeout=60,
    )
    if models.status_code == 404:
        pytest.skip("inference is not enabled on this account (HTTP 404)")
    assert models.status_code == 200, models.text
    assert ready_deployment["deployment"] in {model["id"] for model in models.json()["data"]}


def deployment_pat() -> str:
    pat = load_config().get("pat") or os.environ.get("CORTEX_TRAINING_PAT")
    if not pat:
        pytest.skip("the config has no PAT to call the deployment with")
    return pat


def test_tc6_base_url_streams_chat_completions_as_sse(ready_deployment):
    from recipes.model_deployment.streaming import deployment_headers, iter_sse_chunks

    deployment_id = ready_deployment["deployment"]
    with requests.post(
        f"{ready_deployment['base_url']}/chat/completions",
        headers={**deployment_headers(deployment_pat()), "Accept": "text/event-stream"},
        json={
            "model": deployment_id,
            "messages": [{"role": "user", "content": "Count from one to ten."}],
            "max_completion_tokens": 64,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
        timeout=300,
    ) as response:
        if response.status_code == 404:
            pytest.skip("inference is not enabled on this account (HTTP 404)")
        assert response.status_code == 200, response.text
        assert response.headers["Content-Type"].startswith("text/event-stream")
        lines = list(response.iter_lines(decode_unicode=True))

    data = [line for line in lines if line.startswith("data:")]
    assert data[-1] == "data: [DONE]", data[-3:]
    chunks = list(iter_sse_chunks(lines))
    # A streamed reply arrives in many deltas, not one buffered chunk.
    assert len(chunks) > 1, chunks
    assert {chunk["object"] for chunk in chunks} == {"chat.completion.chunk"}
    assert {chunk["model"] for chunk in chunks} == {deployment_id}
    # include_usage adds a final chunk with no choices and the token counts.
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0
    text = "".join(
        (choice.get("delta") or {}).get("content") or ""
        for chunk in chunks
        for choice in chunk.get("choices") or ()
    )
    assert text.strip()


def test_tc6_example_stream_chat_reads_the_deployment(ready_deployment):
    from recipes.model_deployment.streaming import chat_messages, stream_chat

    deltas: list[str] = []
    try:
        result = stream_chat(
            ready_deployment["base_url"],
            deployment_pat(),
            model=ready_deployment["deployment"],
            messages=chat_messages("Count from one to ten."),
            max_completion_tokens=64,
            temperature=0.0,
            on_delta=deltas.append,
        )
    except RuntimeError as exc:
        if "HTTP 404" in str(exc):
            pytest.skip("inference is not enabled on this account (HTTP 404)")
        raise

    assert len(deltas) > 1, deltas
    assert result.text == "".join(deltas) and result.text.strip()
    assert result.time_to_first_token is not None
    assert result.time_to_first_token <= result.total_seconds
    assert result.finish_reason in {"stop", "length"}
    assert result.usage["completion_tokens"] > 0


def test_tc6_max_tokens_is_rejected_in_favor_of_max_completion_tokens(ready_deployment):
    from recipes.model_deployment.streaming import deployment_headers

    response = requests.post(
        f"{ready_deployment['base_url']}/chat/completions",
        headers=deployment_headers(deployment_pat()),
        json={
            "model": ready_deployment["deployment"],
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
        },
        timeout=300,
    )
    if response.status_code == 404:
        pytest.skip("inference is not enabled on this account (HTTP 404)")
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert (error["param"], error["code"]) == ("max_tokens", "unsupported_parameter")


def test_tc7_logs_and_metrics_download_flat_and_private(ready_deployment, tmp_path):
    deployment_id = ready_deployment["deployment"]
    logs = tmp_path / "logs"
    metrics = tmp_path / "metrics"

    execution = run_ok(
        "model-deployment", "download-log", deployment_id, "--output-dir", str(logs)
    )
    stdout = run_ok(
        "model-deployment", "download-log", deployment_id,
        "--log-type", "stdout", "--output-dir", str(logs),
    )
    downloaded = run_ok(
        "model-deployment", "download-metrics", deployment_id,
        "--output-dir", str(metrics),
    )

    assert execution["deployment"] == stdout["deployment"] == deployment_id
    assert downloaded["deployment"] == deployment_id
    assert execution["logs"] and stdout["logs"] and downloaded["metrics"]
    for directory in (logs, metrics):
        entries = list(directory.iterdir())
        assert entries, f"nothing was written to {directory}"
        for entry in entries:
            assert entry.is_file(), f"{entry} is not a flat file"
            assert stat.S_IMODE(entry.stat().st_mode) == 0o600, entry


@pytest.fixture(scope="module")
def weights_only_checkpoint() -> dict:
    """A completed weights-only checkpoint: ``{"job_id": ..., "checkpoint_id": ...}``."""
    existing = os.environ.get(CHECKPOINT_ENV)
    if existing:
        job_id, _, checkpoint_id = existing.partition("/")
        return {"job_id": job_id, "checkpoint_id": checkpoint_id}

    from cortex_training import CortexTrainingClient, SubJobConfig

    config = load_config()
    client = CortexTrainingClient.from_pat(
        host=config["host"],
        pat=config["pat"],
        database=config["database"],
        schema=config["schema"],
        endpoint=config.get("endpoint", "cortex-training"),
        poll_timeout=READY_TIMEOUT_SECONDS,
    )
    sub_job = SubJobConfig.training_job(
        model_name=MODEL,
        optimizer={"name": "adamw", "lr": 1e-5},
        max_seq_len=128,
        train_batch_size=1,
        n_gpus=1,
    )
    job_id = client.create_job(sub_jobs=[sub_job])
    checkpoint_id = f"md-e2e-weights-only-{int(time.time())}"
    try:
        client.wait_for_job(job_id)
        request_id = client.save(
            job_id, checkpoint_id=checkpoint_id, checkpoint_type="weights-only"
        )
        client.poll_request(job_id, request_id)
    finally:
        client.cancel_job(job_id)
    return {"job_id": job_id, "checkpoint_id": checkpoint_id}


def test_tc8_checkpoint_deployment_reports_its_checkpoint_source(
    created, spec_dir, weights_only_checkpoint
):
    spec = {
        **base_spec(),
        "source": {"type": "checkpoint", **weights_only_checkpoint},
    }
    result = run_cli(
        "model-deployment", "create",
        "--spec", write_spec(spec_dir, "spec-checkpoint.json", spec), "--wait",
        "--timeout", str(READY_TIMEOUT_SECONDS),
    )
    summary = json.loads(result.stdout) if result.stdout.strip() else {}
    if summary.get("deployment"):
        created.append(summary["deployment"])
    assert result.returncode == 0, result.stderr
    assert summary["status"] == "READY"

    described = run_ok("model-deployment", "describe", summary["deployment"])
    assert described["spec"]["source"] == {
        "type": "checkpoint",
        **weights_only_checkpoint,
    }

    dropped = run_ok("model-deployment", "drop", summary["deployment"])
    assert dropped == {"deployment": summary["deployment"], "dropped": True}


def test_tc8_missing_checkpoint_is_rejected(
    spec_dir, weights_only_checkpoint
):
    missing = {
        **base_spec(),
        "source": {
            "type": "checkpoint",
            "job_id": weights_only_checkpoint["job_id"],
            "checkpoint_id": "md-e2e-does-not-exist",
        },
    }
    run_error(
        "model-deployment", "create",
        "--spec", write_spec(spec_dir, "spec-missing-checkpoint.json", missing),
    )


def test_tc9_drop_releases_the_deployment_and_can_be_repeated(ready_deployment):
    deployment_id = ready_deployment["deployment"]
    first = run_ok("model-deployment", "drop", deployment_id)
    assert first == {"deployment": deployment_id, "dropped": True}

    described = run_ok("model-deployment", "describe", deployment_id)
    assert described["status"] in {"DROPPING", "DROPPED"}
    assert "base_url" not in described

    second = run_ok("model-deployment", "drop", deployment_id)
    assert second == {"deployment": deployment_id, "dropped": True}


def test_tc10_errors_are_clear_and_hide_internals(spec_dir):
    config = load_config()
    not_found = run_error("model-deployment", "describe", MISSING_DEPLOYMENT)
    assert f"model deployment {MISSING_DEPLOYMENT} not found in" in not_found
    assert config["database"].upper() in not_found.upper()

    assert "sf-ct-<UUID>" in run_error(
        "model-deployment", "describe", "not-a-deployment"
    )

    typo = base_spec()
    typo["deployment"]["min_replica"] = typo["deployment"].pop("min_replicas")
    assert run_error(
        "model-deployment", "create",
        "--spec", write_spec(spec_dir, "spec-typo.json", typo), "--dry-run",
    ) == (
        "error: deployment contains unsupported field(s): min_replica and is "
        "missing required field(s): min_replicas\n"
    )

    unsupported = {**base_spec(), "source": {"type": "base_model", "model": "Fake/NotARealModel-1B"}}
    message = run_error(
        "model-deployment", "create",
        "--spec", write_spec(spec_dir, "spec-unsupported-model.json", unsupported),
    )
    assert "HTTP 400" in message
    assert "Fake/NotARealModel-1B" in message


@pytest.mark.parametrize(
    ("deployment_overrides", "expected"),
    [
        ({"gpu_type": "A100"}, "deployment.gpu_type must be one of"),
        ({"gpus_per_replica": 3}, "deployment.gpus_per_replica must be one of"),
        ({"min_replicas": 0}, "deployment.min_replicas must be a positive integer"),
        ({"min_replicas": 1.5}, "deployment.min_replicas must be a positive integer"),
        ({"min_replicas": True}, "deployment.min_replicas must be a positive integer"),
        ({"class": "PRODUCTION"}, 'deployment.class must be "EVALUATION"'),
    ],
)
def test_tc10_invalid_specs_are_rejected_before_sending(
    spec_dir, deployment_overrides, expected
):
    spec = write_spec(spec_dir, "spec-invalid.json", base_spec(**deployment_overrides))
    assert expected in run_error("model-deployment", "create", "--spec", spec, "--dry-run")


def test_tc11_interrupt_and_timeout_leave_the_deployment_running(created, spec_dir):
    spec = write_spec(spec_dir, "spec-interrupt.json", base_spec())
    process = subprocess.Popen(
        cli_command("model-deployment", "create", "--spec", spec, "--wait"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(20)
    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=120)
    summary = json.loads(stdout) if stdout.strip() else {}
    deployment_id = summary.get("deployment")
    if deployment_id:
        created.append(deployment_id)

    assert process.returncode == 130, stderr
    assert DEPLOYMENT_ID_PATTERN.match(deployment_id or ""), stdout
    assert summary["status"] in {"PENDING", "INITIALIZING"}, summary
    assert stderr.startswith(f"interrupted; {deployment_id} is still running"), stderr
    assert f"model-deployment wait {deployment_id}" in stderr
    assert f"model-deployment drop {deployment_id}" in stderr

    timed_out = run_cli(
        "model-deployment", "wait", deployment_id, "--timeout", "30", timeout=300
    )
    if timed_out.returncode == 0:
        pytest.skip("the deployment became READY before the 30s timeout elapsed")
    assert timed_out.returncode == 1, timed_out.stderr
    assert (
        f"model deployment {deployment_id} did not become READY within 30s"
        in timed_out.stderr
    )
    assert "still running and holding GPUs" in timed_out.stderr

    starting = run_ok(
        "model-deployment", "list", "--status", "INITIALIZING", "--status", "PENDING"
    )["deployments"]
    assert deployment_id in {entry["deployment"] for entry in starting}

    dropped = run_ok("model-deployment", "drop", deployment_id)
    assert dropped == {"deployment": deployment_id, "dropped": True}
