from __future__ import annotations

import json
import ssl
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
import urllib3

import cortex_training
from cortex_training.client import core as client_module
from cortex_training.client import (
    ModelDeploymentSpec,
    CortexTrainingClient,
)

JOB_ID = "12345678-1234-4234-8234-123456789abc"
DEPLOYMENT_ID = f"sf-ct-{JOB_ID}"
SUB_JOB_ID = f"{JOB_ID}:sampling:0"

MODEL_MINIMUMS = {
    "Qwen/Qwen3-0.6B": 1,
    "Qwen/Qwen3-1.7B": 1,
    "Qwen/Qwen3-4B": 2,
    "Qwen/Qwen3-8B": 2,
    "Qwen/Qwen3-14B": 4,
    "Qwen/Qwen3-32B": 4,
}


def deployment_spec(
    *,
    model: str = "Qwen/Qwen3-1.7B",
    replicas: int = 1,
    gpus: int = 1,
    max_seq_len: int = 8192,
) -> dict:
    return {
        "deployment": {
            "class": "EVALUATION",
            "min_replicas": replicas,
            "gpus_per_replica": gpus,
            "gpu_type": "H200",
        },
        "source": {"type": "base_model", "model": model},
        "inference": {"max_seq_len": max_seq_len},
    }


def source_training_job() -> dict:
    """The checkpoint's source job, as ``get_job`` returns it."""
    return {
        "job_id": "source-training-job",
        "sub_jobs": [
            {"job_type": "JOB_TYPE_TRAINING", "model_name": "Qwen/Qwen3-1.7B"},
            {"job_type": "JOB_TYPE_SAMPLING", "model_name": "Qwen/Qwen3-1.7B"},
        ],
    }


def checkpoint_spec() -> dict:
    value = deployment_spec(replicas=3, gpus=2)
    value["source"] = {
        "type": "checkpoint",
        "job_id": "source-training-job",
        "checkpoint_id": "cp_completed_weights_only",
    }
    return value


def documented_compute_spec() -> dict:
    return deployment_spec(replicas=3, gpus=2)


def marked_job(
    *,
    status: str = "JOB_STATE_RUNNING",
    checkpoint: bool = False,
) -> dict:
    sub_job = {
        "sub_job_id": SUB_JOB_ID,
        "job_type": "sampling",
        "model_name": "Qwen/Qwen3-1.7B",
        "inference_config": {
            "deployment_class": "EVALUATION",
            "num_replicas": 3.0,
            "n_gpus": 6.0,
            "max_seq_len": 8192.0,
            "max_inflight_choices": 8.0,
            "vllm_config": {
                "tensor_parallel_size": 2.0,
                "max_model_len": 8192.0,
            },
        },
    }
    if checkpoint:
        sub_job["source_checkpoint_info"] = {
            "source_job_id": "source-training-job",
            "checkpoint_id": "cp_completed_weights_only",
            "source_stage_info": {"credential": "must-not-escape"},
        }
    return {
        "job_id": JOB_ID,
        "status": status,
        "created_at": "2026-09-16T20:00:00Z",
        "owner_role": "CURRENT_OWNER_ROLE",
        "submitted_by": "CURRENT_USER",
        "hardware": "H200",
        "sub_jobs": [sub_job],
    }


@pytest.mark.parametrize("model,gpus", MODEL_MINIMUMS.items())
def test_spec_accepts_every_supported_model_at_sampling_minimum(model, gpus):
    spec = ModelDeploymentSpec.from_dict(deployment_spec(model=model, gpus=gpus))
    body = spec._to_create_job_body()
    inference = body["sub_job_configs"][0]["inference_config"]

    assert body["idle_timeout_seconds"] == 0
    assert body["hardware"] == "H200"
    assert inference == {
        "max_seq_len": 8192,
        "n_gpus": gpus,
        "deployment_class": "EVALUATION",
        "num_replicas": 1,
        "vllm_config": {
            "tensor_parallel_size": gpus,
            "max_model_len": 8192,
        },
    }
    assert "max_inflight_choices" not in inference


def test_spec_accepts_primary_design_compute_shape_and_gpu_type():
    spec = ModelDeploymentSpec.from_dict(documented_compute_spec())

    assert spec.min_replicas == 3
    assert spec.gpus_per_replica == 2
    assert spec.gpu_type == "H200"
    assert spec._to_create_job_body()["hardware"] == "H200"


def test_spec_rejects_the_retired_capacity_replica_shape():
    value = documented_compute_spec()
    del value["deployment"]
    value["capacity"] = {"min_replicas": 1}
    value["replica"] = {"gpus": 1}

    with pytest.raises(
        ValueError,
        match="unsupported field\\(s\\): capacity, replica and is missing required field\\(s\\): deployment",
    ):
        ModelDeploymentSpec.from_dict(value)


def test_spec_typo_names_the_unknown_key_as_well_as_the_missing_one():
    value = deployment_spec()
    value["deployment"]["min_replica"] = value["deployment"].pop("min_replicas")

    with pytest.raises(ValueError) as exc_info:
        ModelDeploymentSpec.from_dict(value)

    assert str(exc_info.value) == (
        "deployment contains unsupported field(s): min_replica and is missing "
        "required field(s): min_replicas"
    )


@pytest.mark.parametrize("gpu_type", ["h200", "H200", " b300 "])
def test_spec_gpu_type_is_case_insensitive(gpu_type):
    value = deployment_spec()
    value["deployment"]["gpu_type"] = gpu_type

    assert ModelDeploymentSpec.from_dict(value).gpu_type == gpu_type.strip().upper()


# model_deployment duplicates these because the client imports it; keep the two
# copies from drifting apart.
def test_spec_gpu_types_match_client_hardware():
    from cortex_training.model_deployment import MODEL_DEPLOYMENT_GPU_TYPES

    assert MODEL_DEPLOYMENT_GPU_TYPES == tuple(member.value for member in client_module.Hardware)


def test_spec_unknown_gpu_type_error_names_the_spec_field():
    value = deployment_spec()
    value["deployment"]["gpu_type"] = "A100"

    with pytest.raises(ValueError, match="deployment.gpu_type must be one of H200, B200, B300"):
        ModelDeploymentSpec.from_dict(value)


def test_checkpoint_spec_translates_only_public_checkpoint_identity():
    spec = ModelDeploymentSpec.from_dict(checkpoint_spec())
    assert spec.model_name is None
    body = replace(spec, model_name="Qwen/Qwen3-1.7B")._to_create_job_body()
    sub_job = body["sub_job_configs"][0]

    assert sub_job["model_name"] == "Qwen/Qwen3-1.7B"
    assert sub_job["source_checkpoint_info"] == {
        "source_job_id": "source-training-job",
        "checkpoint_id": "cp_completed_weights_only",
    }
    assert sub_job["inference_config"]["n_gpus"] == 6
    assert "source_stage_info" not in sub_job["source_checkpoint_info"]


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda value: value.update(extra=True), "unsupported field"),
        (
            lambda value: value["deployment"].update({"class": "PRODUCTION"}),
            'deployment.class must be "EVALUATION"; only EVALUATION is supported today',
        ),
        (
            lambda value: value["source"].update(stage_credentials={"token": "x"}),
            "unsupported field",
        ),
        (
            lambda value: value.update(source={"type": "registry", "model": "x"}),
            "source.type",
        ),
        (
            lambda value: value["deployment"].update(min_replicas=True),
            "positive integer",
        ),
        (
            lambda value: value["deployment"].update(gpus_per_replica=3),
            "deployment.gpus_per_replica must be one of 1, 2, 4, or 8",
        ),
        (
            lambda value: value["deployment"].update(min_replicas=3, gpus_per_replica=4),
            "deployment.min_replicas \\* deployment.gpus_per_replica must be at most 8 or a positive multiple of 8",
        ),
        (
            lambda value: value["inference"].update(max_seq_len=0),
            "positive integer",
        ),
    ],
)
def test_spec_rejects_invalid_or_service_owned_fields(mutate, error):
    value = deployment_spec()
    mutate(value)
    with pytest.raises(ValueError, match=error):
        ModelDeploymentSpec.from_dict(value)


@pytest.mark.parametrize("missing", ["job_id", "checkpoint_id"])
def test_checkpoint_spec_requires_all_identity_fields(missing):
    value = checkpoint_spec()
    del value["source"][missing]
    with pytest.raises(ValueError, match=missing):
        ModelDeploymentSpec.from_dict(value)


def test_spec_rejects_the_previous_deployment_class_and_compute_shape():
    value = deployment_spec()
    deployment = value.pop("deployment")
    value["deployment_class"] = deployment.pop("class")
    value["compute"] = deployment

    with pytest.raises(
        ValueError,
        match="unsupported field\\(s\\): compute, deployment_class and is missing required field\\(s\\): deployment",
    ):
        ModelDeploymentSpec.from_dict(value)


def test_checkpoint_spec_rejects_base_model_as_an_unknown_key():
    value = checkpoint_spec()
    value["source"]["base_model"] = "Qwen/Qwen3-1.7B"

    with pytest.raises(ValueError, match="source contains unsupported field\\(s\\): base_model"):
        ModelDeploymentSpec.from_dict(value)


def test_model_deployment_detection_requires_class_and_single_sampling_topology():
    assert client_module._model_deployment_sub_job(marked_job()) is not None
    assert client_module._model_deployment_sub_job({"sub_jobs": []}) is None
    assert (
        client_module._model_deployment_sub_job(
            {"sub_jobs": marked_job()["sub_jobs"] * 2}
        )
        is None
    )
    missing_class = marked_job()
    missing_class["sub_jobs"][0]["inference_config"].pop("deployment_class")
    assert client_module._model_deployment_sub_job(missing_class) is None


def test_metadata_hides_backing_identity_credentials_and_service_config():
    metadata = client_module._model_deployment_metadata(
        marked_job(checkpoint=True),
        DEPLOYMENT_ID,
        base_url="https://account/api/inference/v1",
    )

    assert metadata == {
        "deployment": DEPLOYMENT_ID,
        "deployment_class": "EVALUATION",
        "status": "READY",
        "model_source": {
            "type": "checkpoint",
            "job_id": "source-training-job",
            "checkpoint_id": "cp_completed_weights_only",
        },
        "max_seq_len": 8192,
        "gpus_per_replica": 2,
        "replicas": 3,
        "gpu_type": "H200",
        "created_at": "2026-09-16T20:00:00Z",
        "owner_role": "CURRENT_OWNER_ROLE",
        "creator": "CURRENT_USER",
        "base_url": "https://account/api/inference/v1",
    }
    serialized = repr(metadata)
    assert SUB_JOB_ID not in serialized
    assert "must-not-escape" not in serialized
    assert "max_inflight_choices" not in serialized


def test_running_status_is_ready_without_an_unpublished_capabilities_field():
    assert client_module._model_deployment_status(marked_job()) == "READY"
    assert (
        client_module._model_deployment_status(marked_job(status="JOB_STATE_FAILED"))
        == "FAILED"
    )
    assert (
        client_module._model_deployment_status(
            marked_job(status="JOB_STATE_CANCELLING")
        )
        == "DROPPING"
    )
    assert (
        client_module._model_deployment_status(
            marked_job(status="JOB_STATE_CANCELLED")
        )
        == "DROPPED"
    )
    assert (
        client_module._model_deployment_status(
            marked_job(status="JOB_STATE_TERMINATED")
        )
        == "TERMINATED"
    )


def test_pending_reason_is_not_reported_as_failure():
    metadata = client_module._model_deployment_metadata(
        {
            **marked_job(status="JOB_STATE_PENDING"),
            "reason": "quota_filled",
            "reason_details": "needs 8 GPUs; 64 currently active",
        },
        DEPLOYMENT_ID,
    )

    assert metadata["status"] == "PENDING"
    assert metadata["status_reason"] == "quota_filled"
    assert metadata["status_details"] == "needs 8 GPUs; 64 currently active"
    assert "failure" not in metadata


def test_raw_job_helpers_are_not_public_package_exports():
    raw_helpers = {
        "is_model_deployment_job",
        "model_deployment_metadata",
        "model_deployment_status",
    }

    assert raw_helpers.isdisjoint(cortex_training.__all__)
    assert all(not hasattr(cortex_training, name) for name in raw_helpers)
    instance = client()
    assert not hasattr(instance, "list_model_deployment_jobs")
    assert not hasattr(instance, "get_model_deployment_job")
    assert not hasattr(instance, "wait_for_model_deployment_job")
    assert not hasattr(instance, "drop_model_deployment_job")


@pytest.mark.parametrize(
    "deployment_id,error",
    [
        (JOB_ID, "sf-ct-<UUID>"),
        (f"sf-ct-{JOB_ID.upper()}", "canonical lowercase UUID"),
    ],
)
def test_public_model_deployment_helpers_reject_nonopaque_ids(deployment_id, error):
    instance = client()
    instance.get_job = MagicMock()

    with pytest.raises(ValueError, match=error):
        instance.get_model_deployment(deployment_id)

    instance.get_job.assert_not_called()


def client() -> CortexTrainingClient:
    return CortexTrainingClient(
        base_url="http://test.local",
        database="DB",
        schema="SCH",
        poll_interval=0.001,
        poll_timeout=0.05,
    )


def test_client_create_model_deployment_posts_existing_create_job_shape():
    instance = client()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"job_id": JOB_ID}
    response.raise_for_status.return_value = None
    instance._session = MagicMock()
    instance._session.post.return_value = response

    instance.get_job = MagicMock(return_value=source_training_job())

    metadata = instance.create_model_deployment(checkpoint_spec())

    instance.get_job.assert_called_once()
    assert instance.get_job.call_args.args == ("source-training-job",)
    assert metadata["deployment"] == DEPLOYMENT_ID
    assert metadata["status"] == "PENDING"
    assert "job_id" not in metadata
    assert "sub_jobs" not in metadata
    posted = instance._session.post.call_args.kwargs["json"]
    assert posted["idle_timeout_seconds"] == 0
    assert len(posted["sub_job_configs"]) == 1
    assert posted["sub_job_configs"][0]["job_type"] == "sampling"
    assert posted["sub_job_configs"][0]["model_name"] == "Qwen/Qwen3-1.7B"


def test_create_fails_without_posting_when_the_source_job_has_no_training_model():
    instance = client()
    instance.create_job_from_body = MagicMock()
    instance.get_job = MagicMock(
        return_value={"job_id": "source-training-job", "sub_jobs": [{"job_type": "sampling"}]}
    )

    with pytest.raises(ValueError, match="has no training sub-job model_name"):
        instance.create_model_deployment(checkpoint_spec())
    instance.create_job_from_body.assert_not_called()


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"source_type": "bogus"}, "source.type"),
        ({"min_replicas": 0}, "positive integer"),
        (
            {"source_type": "checkpoint", "source_job_id": None, "checkpoint_id": "C"},
            "source.job_id must be a non-empty string",
        ),
        (
            {"source_type": "checkpoint", "source_job_id": "J", "checkpoint_id": None},
            "source.checkpoint_id must be a non-empty string",
        ),
    ],
)
def test_client_revalidates_directly_constructed_spec_before_post(overrides, error):
    instance = client()
    instance.create_job_from_body = MagicMock()
    fields = {
        "model_name": "Qwen/Qwen3-1.7B",
        "min_replicas": 1,
        "gpus_per_replica": 1,
        "max_seq_len": 8192,
        "source_type": "base_model",
        **overrides,
    }
    invalid = ModelDeploymentSpec(**fields)

    # The spec is round-tripped through from_dict unchanged, so a bad value is
    # rejected rather than coerced (e.g. "bogus" -> "checkpoint", None -> "None").
    with pytest.raises(ValueError, match=error):
        instance.create_model_deployment(invalid)
    instance.create_job_from_body.assert_not_called()


def test_source_to_wire_passes_values_through_unchanged():
    spec = ModelDeploymentSpec(
        model_name="Qwen/Qwen3-1.7B",
        min_replicas=1,
        gpus_per_replica=1,
        max_seq_len=8192,
        source_type="checkpoint",
    )

    assert spec.source_to_wire() == {
        "type": "checkpoint",
        "job_id": None,
        "checkpoint_id": None,
    }


@pytest.mark.parametrize(
    "source,error",
    [
        ({"type": "bogus", "model": "Qwen/Qwen3-1.7B"}, "source.type"),
        (
            {"type": "checkpoint", "job_id": None, "checkpoint_id": "C"},
            "source.job_id must be a non-empty string",
        ),
        (
            {"type": "checkpoint", "job_id": "J", "checkpoint_id": None},
            "source.checkpoint_id must be a non-empty string",
        ),
    ],
)
def test_client_rejects_the_same_bad_sources_given_as_a_dict(source, error):
    instance = client()
    instance.create_job_from_body = MagicMock()
    value = deployment_spec()
    value["source"] = source

    with pytest.raises(ValueError, match=error):
        instance.create_model_deployment(value)
    instance.create_job_from_body.assert_not_called()


def test_client_list_returns_sanitized_deployments_without_changing_list_jobs():
    instance = client()
    ordinary = {"job_id": "ordinary", "status": "running", "sub_jobs": []}
    instance.list_jobs = MagicMock(return_value=[ordinary, marked_job()])

    deployments = instance.list_model_deployments()

    assert deployments == [
        client_module._model_deployment_metadata(marked_job(), DEPLOYMENT_ID)
    ]
    assert JOB_ID not in repr(deployments).replace(DEPLOYMENT_ID, "")
    assert SUB_JOB_ID not in repr(deployments)
    assert instance.list_jobs() == [ordinary, marked_job()]


def test_client_returns_ready_for_running_deployment_without_capabilities():
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())
    instance._sleep_with_backoff = MagicMock(return_value=0.001)

    metadata = instance.wait_for_model_deployment(DEPLOYMENT_ID)

    assert metadata["deployment"] == DEPLOYMENT_ID
    assert metadata["status"] == "READY"
    assert JOB_ID not in repr(metadata).replace(DEPLOYMENT_ID, "")
    instance._get_model_deployment_job.assert_called_once_with(JOB_ID)
    instance._sleep_with_backoff.assert_not_called()


def test_client_wait_raises_on_terminal_failure_without_exposing_job_id():
    instance = client()
    instance._get_model_deployment_job = MagicMock(
        return_value={**marked_job(status="failed"), "reason": "engine failed"}
    )

    with pytest.raises(RuntimeError, match="engine failed") as exc_info:
        instance.wait_for_model_deployment(DEPLOYMENT_ID)
    # The message names the public deployment ID, never the backing job ID.
    assert DEPLOYMENT_ID in str(exc_info.value)
    assert JOB_ID not in str(exc_info.value).replace(DEPLOYMENT_ID, "")


def test_client_wait_stops_immediately_on_terminated_deployment():
    instance = client()
    instance._get_model_deployment_job = MagicMock(
        return_value={
            **marked_job(status="JOB_STATE_TERMINATED"),
            "reason": "pending_timeout",
        }
    )
    instance._sleep_with_backoff = MagicMock()

    with pytest.raises(RuntimeError, match="TERMINATED.*pending_timeout"):
        instance.wait_for_model_deployment(DEPLOYMENT_ID)

    instance._get_model_deployment_job.assert_called_once_with(JOB_ID)
    instance._sleep_with_backoff.assert_not_called()


def test_drop_uses_direct_get_when_paginated_list_omits_deployment():
    instance = client()
    instance._list_model_deployment_jobs = MagicMock(return_value=[])
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())
    instance.cancel_job = MagicMock()

    assert instance.drop_model_deployment(DEPLOYMENT_ID) == {
        "deployment": DEPLOYMENT_ID,
        "dropped": True,
    }
    instance._list_model_deployment_jobs.assert_not_called()
    instance._get_model_deployment_job.assert_called_once_with(JOB_ID)
    instance.cancel_job.assert_called_once_with(JOB_ID)


def test_drop_does_not_depend_on_transient_list_behavior():
    instance = client()
    instance._list_model_deployment_jobs = MagicMock(
        side_effect=requests.exceptions.ConnectionError("transient list failure")
    )
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())
    instance.cancel_job = MagicMock()

    result = instance.drop_model_deployment(DEPLOYMENT_ID)

    assert result == {"deployment": DEPLOYMENT_ID, "dropped": True}
    instance._list_model_deployment_jobs.assert_not_called()
    instance.cancel_job.assert_called_once_with(JOB_ID)


def test_drop_rejects_non_deployment_job():
    instance = client()
    instance.get_job = MagicMock(
        return_value={"job_id": JOB_ID, "status": "JOB_STATE_RUNNING", "sub_jobs": []}
    )
    instance.cancel_job = MagicMock()

    with pytest.raises(ValueError, match="not a model deployment"):
        instance.drop_model_deployment(DEPLOYMENT_ID)

    instance.get_job.assert_called_once_with(
        JOB_ID, retry_on=client_module._is_transient_unless_job_not_found
    )
    instance.cancel_job.assert_not_called()


def test_drop_is_idempotent_when_direct_get_says_job_not_found():
    instance = client()
    response = _server_error_response(404, {"code": "517601"})
    instance.get_job = MagicMock(
        side_effect=requests.exceptions.HTTPError(response=response)
    )
    instance.cancel_job = MagicMock()

    assert instance.drop_model_deployment(DEPLOYMENT_ID) == {
        "deployment": DEPLOYMENT_ID,
        "dropped": True,
    }
    instance.get_job.assert_called_once_with(
        JOB_ID, retry_on=client_module._is_transient_unless_job_not_found
    )
    instance.cancel_job.assert_not_called()


def test_drop_raises_a_404_without_the_job_not_found_code():
    """A routing 404 does not prove the deployment is gone; it may still hold GPUs."""
    instance = client()
    response = _server_error_response(404, {"message": "no route"})
    instance.get_job = MagicMock(
        side_effect=requests.exceptions.HTTPError(response=response)
    )
    instance.cancel_job = MagicMock()

    with pytest.raises(RuntimeError, match="HTTP 404"):
        instance.drop_model_deployment(DEPLOYMENT_ID)

    instance.cancel_job.assert_not_called()


@pytest.mark.parametrize(
    "job_status",
    ["JOB_STATE_CANCELLING", "JOB_STATE_CANCELLED", "JOB_STATE_DONE"],
)
def test_drop_of_a_dropping_or_dropped_deployment_does_not_cancel_again(job_status):
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job(status=job_status))
    instance.cancel_job = MagicMock()

    assert instance.drop_model_deployment(DEPLOYMENT_ID) == {
        "deployment": DEPLOYMENT_ID,
        "dropped": True,
    }
    instance.cancel_job.assert_not_called()


@pytest.mark.parametrize(
    "job_status,deployment_status",
    [("JOB_STATE_FAILED", "FAILED"), ("JOB_STATE_TERMINATED", "TERMINATED")],
)
def test_drop_of_a_failed_or_terminated_deployment_reports_it_was_not_dropped(
    job_status, deployment_status
):
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job(status=job_status))
    instance.cancel_job = MagicMock()

    assert instance.drop_model_deployment(DEPLOYMENT_ID) == {
        "deployment": DEPLOYMENT_ID,
        "dropped": False,
        "status": deployment_status,
    }
    instance.cancel_job.assert_not_called()


@pytest.mark.parametrize(
    "job_status",
    ["JOB_STATE_PENDING", "JOB_STATE_PLACING", "JOB_STATE_INITIALIZING", "JOB_STATE_RUNNING"],
)
def test_drop_cancels_a_deployment_that_may_hold_gpus(job_status):
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job(status=job_status))
    instance.cancel_job = MagicMock()

    instance.drop_model_deployment(DEPLOYMENT_ID)

    instance.cancel_job.assert_called_once_with(JOB_ID)


def test_drop_propagates_non_404_direct_get_error():
    instance = client()
    response = requests.Response()
    response.status_code = 503
    lookup_error = requests.exceptions.HTTPError(response=response)
    instance.get_job = MagicMock(side_effect=lookup_error)
    instance.cancel_job = MagicMock()

    with pytest.raises(RuntimeError, match="HTTP 503") as exc_info:
        instance.drop_model_deployment(DEPLOYMENT_ID)

    assert JOB_ID not in str(exc_info.value).replace(DEPLOYMENT_ID, "")
    instance.get_job.assert_called_once_with(
        JOB_ID, retry_on=client_module._is_transient_unless_job_not_found
    )
    instance.cancel_job.assert_not_called()


@pytest.mark.parametrize("status_code", [404, 503])
def test_drop_propagates_cancellation_errors(status_code):
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())
    response = requests.Response()
    response.status_code = status_code
    cancellation_error = requests.exceptions.HTTPError(response=response)
    instance.cancel_job = MagicMock(side_effect=cancellation_error)

    with pytest.raises(RuntimeError, match=f"HTTP {status_code}") as exc_info:
        instance.drop_model_deployment(DEPLOYMENT_ID)

    assert JOB_ID not in str(exc_info.value).replace(DEPLOYMENT_ID, "")
    instance.cancel_job.assert_called_once_with(JOB_ID)


def test_get_redacts_backing_job_id_from_http_error_url_and_body():
    instance = client()
    response = requests.Response()
    response.status_code = 503
    response.url = f"http://test.local/cortex-training/{JOB_ID}"
    response._content = f'{{"job_id":"{JOB_ID}"}}'.encode()
    response.headers["x-snowflake-request-id"] = "request-123"
    raw_error = requests.exceptions.HTTPError(
        f"503 Server Error for url: {response.url}", response=response
    )
    instance.get_job = MagicMock(side_effect=raw_error)

    with pytest.raises(RuntimeError, match="request-123") as exc_info:
        instance.get_model_deployment(DEPLOYMENT_ID)

    rendered = str(exc_info.value).replace(DEPLOYMENT_ID, "")
    assert JOB_ID not in rendered
    assert "job_id" not in rendered


def test_drop_cancels_marked_job_and_returns_only_public_identity():
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())
    instance.cancel_job = MagicMock()

    result = instance.drop_model_deployment(DEPLOYMENT_ID)

    assert result == {"deployment": DEPLOYMENT_ID, "dropped": True}
    assert JOB_ID not in repr(result).replace(DEPLOYMENT_ID, "")
    assert SUB_JOB_ID not in repr(result)
    instance._get_model_deployment_job.assert_called_once_with(JOB_ID)
    instance.cancel_job.assert_called_once_with(JOB_ID)


def test_download_execution_logs_flattens_and_sanitizes_paths(tmp_path):
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())
    instance.fetch_execution_logs = MagicMock(
        return_value=[
            {
                "sub_job_id": SUB_JOB_ID,
                "filename": "execution.jsonl",
                "artifact_uri": f"snow://secret/{SUB_JOB_ID}/execution.jsonl",
                "content": '{"ok":true}\n',
            }
        ]
    )

    result = instance.download_model_deployment_logs(DEPLOYMENT_ID, tmp_path)

    assert result == {
        "deployment": DEPLOYMENT_ID,
        "logs": [
            {
                "filename": "execution.jsonl",
                "saved_path": str(tmp_path / "execution.jsonl"),
            }
        ],
    }
    assert (tmp_path / "execution.jsonl").read_text() == '{"ok":true}\n'
    assert SUB_JOB_ID not in repr(result)


def test_download_stdout_and_metrics_flatten_internal_sub_job_directories(
    tmp_path, monkeypatch
):
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())

    def write_stdout(_job_id, output_dir):
        path = Path(output_dir) / SUB_JOB_ID / "stdout.log"
        path.parent.mkdir(parents=True)
        path.write_text("hello\n")
        return [
            {
                "sub_job_id": SUB_JOB_ID,
                "filename": "stdout.log",
                "saved_path": str(path),
                "chunk_count": 2,
                "first_artifact_uri": "snow://secret/first",
            }
        ]

    def write_metrics(_job_id, output_dir):
        path = Path(output_dir) / SUB_JOB_ID / "gpu.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('{"gpu":0}\n')
        return [
            {
                "sub_job_id": SUB_JOB_ID,
                "filename": "gpu.jsonl",
                "saved_path": str(path),
                "chunk_count": 3,
                "first_artifact_uri": "snow://secret/first",
            }
        ]

    monkeypatch.setattr(instance, "download_stdout_logs", write_stdout)
    monkeypatch.setattr(instance, "download_metrics", write_metrics)
    stdout_dir = tmp_path / "logs"
    metrics_dir = tmp_path / "metrics"

    logs = instance.download_model_deployment_logs(
        DEPLOYMENT_ID, stdout_dir, log_type="stdout"
    )
    metrics = instance.download_model_deployment_metrics(DEPLOYMENT_ID, metrics_dir)

    assert logs == {
        "deployment": DEPLOYMENT_ID,
        "logs": [
            {
                "filename": "stdout.log",
                "saved_path": str(stdout_dir / "stdout.log"),
                "chunk_count": 2,
            }
        ],
    }
    assert metrics == {
        "deployment": DEPLOYMENT_ID,
        "metrics": [
            {
                "filename": "gpu.jsonl",
                "saved_path": str(metrics_dir / "gpu.jsonl"),
                "chunk_count": 3,
            }
        ],
    }
    assert SUB_JOB_ID not in repr((logs, metrics))


def test_documented_deployment_json_translates_to_create_job_body():
    """The documented checkpoint deployment.json example; the model comes from the source job."""
    spec = ModelDeploymentSpec.from_dict(
        {
            "deployment": {
                "class": "EVALUATION",
                "min_replicas": 3,
                "gpus_per_replica": 2,
                "gpu_type": "H200",
            },
            "source": {
                "type": "checkpoint",
                "job_id": "SOURCE_TRAINING_JOB_UUID",
                "checkpoint_id": "cp_RETURNED_BY_SAVE",
            },
            "inference": {"max_seq_len": 8192},
        }
    )

    assert replace(spec, model_name="Qwen/Qwen3-1.7B")._to_create_job_body() == {
        "idle_timeout_seconds": 0,
        "hardware": "H200",
        "sub_job_configs": [
            {
                "job_type": "sampling",
                "model_name": "Qwen/Qwen3-1.7B",
                "inference_config": {
                    "max_seq_len": 8192,
                    "n_gpus": 6,
                    "deployment_class": "EVALUATION",
                    "num_replicas": 3,
                    "vllm_config": {"tensor_parallel_size": 2, "max_model_len": 8192},
                },
                "source_checkpoint_info": {
                    "source_job_id": "SOURCE_TRAINING_JOB_UUID",
                    "checkpoint_id": "cp_RETURNED_BY_SAVE",
                },
            }
        ],
    }


def test_failed_deployment_metadata_reports_every_design_doc_list_field():
    """Design doc: list returns ID, status, model source, resource configuration,
    creation time, creator, owner role, and failure information when applicable."""
    job = marked_job(status="JOB_STATE_FAILED")
    job["reason"] = "sub_job_failed"
    job["reason_details"] = "vLLM ran out of GPU memory loading the model"

    metadata = client_module._model_deployment_metadata(job, DEPLOYMENT_ID)

    assert metadata["deployment"] == DEPLOYMENT_ID
    assert metadata["status"] == "FAILED"
    assert metadata["model_source"] == {"type": "base_model", "model": "Qwen/Qwen3-1.7B"}
    assert (metadata["replicas"], metadata["gpus_per_replica"], metadata["gpu_type"]) == (3, 2, "H200")
    assert metadata["created_at"] == "2026-09-16T20:00:00Z"
    assert metadata["creator"] == "CURRENT_USER"
    assert metadata["owner_role"] == "CURRENT_OWNER_ROLE"
    assert metadata["failure"] == {
        "code": "sub_job_failed",
        "message": "vLLM ran out of GPU memory loading the model",
    }


@pytest.mark.parametrize(
    "compute,error",
    [
        (
            {"min_replicas": 1, "gpus_per_replica": 3, "gpu_type": "H200"},
            "deployment.gpus_per_replica must be one of 1, 2, 4, or 8",
        ),
        (
            {"min_replicas": 0, "gpus_per_replica": 1, "gpu_type": "H200"},
            "deployment.min_replicas must be a positive integer",
        ),
        (
            {"min_replicas": 3, "gpus_per_replica": 4, "gpu_type": "H200"},
            r"deployment.min_replicas \* deployment.gpus_per_replica must be at most 8",
        ),
        (
            {"min_replicas": 1, "gpus_per_replica": 1, "gpu_type": ""},
            "deployment.gpu_type must be a non-empty string",
        ),
    ],
)
def test_deployment_spec_errors_name_deployment_fields(compute, error):
    value = deployment_spec()
    value["deployment"] = {"class": "EVALUATION", **compute}
    with pytest.raises(ValueError, match=error):
        ModelDeploymentSpec.from_dict(value)


# ─── Error and edge paths ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,error",
    [
        ("not-a-dict", "deployment specification must be an object"),
        ({**deployment_spec(), "source": "x"}, "source must be an object"),
    ],
)
def test_spec_rejects_non_object_values(value, error):
    with pytest.raises(ValueError, match=error):
        ModelDeploymentSpec.from_dict(value)


@pytest.mark.parametrize(
    "source,error",
    [
        ({"type": "base_model", "model": "  "}, "source.model must be a non-empty string"),
        (
            {"type": "checkpoint", "job_id": "", "checkpoint_id": "C"},
            "source.job_id must be a non-empty string",
        ),
    ],
)
def test_spec_rejects_empty_source_strings(source, error):
    value = deployment_spec()
    value["source"] = source
    with pytest.raises(ValueError, match=error):
        ModelDeploymentSpec.from_dict(value)


def test_spec_rejects_max_seq_len_above_int32():
    with pytest.raises(ValueError, match="inference.max_seq_len must be at most 2147483647"):
        ModelDeploymentSpec.from_dict(deployment_spec(max_seq_len=2_147_483_648))


@pytest.mark.parametrize(
    "job",
    [
        "not-a-dict",
        {"sub_jobs": ["not-a-dict"]},
        {"sub_jobs": [{**marked_job()["sub_jobs"][0], "job_type": "training"}]},
        {"sub_jobs": [{**marked_job()["sub_jobs"][0], "inference_config": None}]},
    ],
)
def test_jobs_that_are_not_deployments(job):
    assert client_module._model_deployment_sub_job(job) is None


@pytest.mark.parametrize(
    "status,expected",
    [("JOB_STATE_SOMETHING_NEW", "SOMETHING_NEW"), (None, "UNKNOWN"), ("", "UNKNOWN")],
)
def test_unrecognized_job_status_is_shown_as_is(status, expected):
    assert client_module._model_deployment_status({"status": status}) == expected


def _broken(**changes):
    """A marked job whose stored inference config has the given changes."""
    job = marked_job()
    config = job["sub_jobs"][0]["inference_config"]
    for key, value in changes.items():
        if key in {"tensor_parallel_size", "max_model_len"}:
            config["vllm_config"][key] = value
        elif key == "model_name":
            job["sub_jobs"][0]["model_name"] = value
        elif key == "source_checkpoint_info":
            job["sub_jobs"][0]["source_checkpoint_info"] = value
        elif value is None:
            config.pop(key)
        else:
            config[key] = value
    return job


@pytest.mark.parametrize(
    "job,error",
    [
        (_broken(num_replicas=0), "num_replicas must be a positive integer"),
        (_broken(num_replicas=2.5), "num_replicas must be a positive integer"),
        (_broken(num_replicas=True), "num_replicas must be a positive integer"),
        (_broken(vllm_config=None), "vllm_config is required"),
        (_broken(n_gpus=7.0), "inconsistent replica GPU totals"),
        (_broken(max_model_len=4096.0), "inconsistent sequence lengths"),
        (_broken(model_name=""), "model_name must be a non-empty string"),
        (
            _broken(source_checkpoint_info={"source_job_id": "J", "checkpoint_id": ""}),
            "source_checkpoint_info.checkpoint_id must be a non-empty string",
        ),
        (_broken(source_checkpoint_info="J/C"), "source_checkpoint_info must be an object"),
    ],
)
def test_metadata_rejects_a_broken_stored_deployment(job, error):
    with pytest.raises(ValueError, match=error):
        client_module._model_deployment_metadata(job, DEPLOYMENT_ID)


def test_metadata_rejects_a_job_that_is_not_a_deployment():
    with pytest.raises(ValueError, match="not a model deployment"):
        client_module._model_deployment_metadata({"sub_jobs": []}, DEPLOYMENT_ID)


@pytest.mark.parametrize(
    "reason,reason_details,expected",
    [
        ("sub_job_failed", "", {"code": "sub_job_failed"}),
        ("", "out of memory", {"message": "out of memory"}),
        ("", "", None),
    ],
)
def test_failure_info_has_only_the_fields_the_server_returned(reason, reason_details, expected):
    job = {
        **marked_job(status="JOB_STATE_FAILED"),
        "reason": reason,
        "reason_details": reason_details,
    }
    metadata = client_module._model_deployment_metadata(job, DEPLOYMENT_ID)
    assert metadata.get("failure") == expected
    assert "base_url" not in metadata


def test_reason_details_do_not_expose_backing_job_or_sub_job_ids():
    job = {
        **marked_job(status="JOB_STATE_FAILED"),
        "reason": "invalid_config",
        "reason_details": f"sub-job {SUB_JOB_ID} rejected the config of job {JOB_ID}",
    }

    metadata = client_module._model_deployment_metadata(job, DEPLOYMENT_ID)

    assert metadata["failure"]["message"] == (
        f"sub-job {DEPLOYMENT_ID} rejected the config of job {DEPLOYMENT_ID}"
    )


@pytest.mark.parametrize(
    "details",
    [
        f"deployment {DEPLOYMENT_ID} rejected the config",
        f"job {JOB_ID.upper()} rejected the config",
    ],
)
def test_reason_details_rewrite_each_id_once_in_any_case(details):
    job = {
        **marked_job(status="JOB_STATE_FAILED"),
        "reason": "invalid_config",
        "reason_details": details,
    }

    metadata = client_module._model_deployment_metadata(job, DEPLOYMENT_ID)

    assert metadata["failure"]["message"].endswith(
        f" {DEPLOYMENT_ID} rejected the config"
    )


def test_owner_role_name_is_not_read():
    """The server's job record returns owner_role only."""
    job = {**marked_job(), "owner_role_name": "SOME_OTHER_ROLE"}

    metadata = client_module._model_deployment_metadata(job, DEPLOYMENT_ID)

    assert metadata["owner_role"] == "CURRENT_OWNER_ROLE"


@pytest.mark.parametrize(
    "job_id,error",
    [
        (None, "did not include a valid deployment identifier"),
        ("not-a-uuid", "did not include a valid deployment identifier"),
        (JOB_ID.upper(), "was not canonical"),
    ],
)
def test_create_response_job_id_must_be_a_canonical_uuid(job_id, error):
    with pytest.raises(ValueError, match=error):
        client_module._deployment_id_from_job_id(job_id)


def test_deployment_id_suffix_must_be_a_uuid():
    with pytest.raises(ValueError, match="deployment must currently be an sf-ct-<UUID> ID"):
        client_module._job_id_from_deployment_id("sf-ct-not-a-uuid")


def test_request_error_without_a_response_has_no_status_or_request_id():
    error = client_module._sanitized_model_deployment_request_error(
        requests.exceptions.ConnectionError("connection refused"), DEPLOYMENT_ID
    )
    assert str(error) == f"model deployment {DEPLOYMENT_ID} request failed"


def test_get_model_deployment_returns_metadata_with_base_url():
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())

    metadata = instance.get_model_deployment(DEPLOYMENT_ID)

    assert metadata["deployment"] == DEPLOYMENT_ID
    assert metadata["status"] == "READY"
    assert metadata["base_url"] == instance.model_deployment_base_url
    instance._get_model_deployment_job.assert_called_once_with(JOB_ID)


def test_wait_polls_until_ready():
    instance = client()
    instance._get_model_deployment_job = MagicMock(
        side_effect=[marked_job(status="JOB_STATE_PENDING"), marked_job()]
    )

    metadata = instance.wait_for_model_deployment(DEPLOYMENT_ID)

    assert metadata["status"] == "READY"
    assert instance._get_model_deployment_job.call_count == 2


def test_wait_times_out_when_never_ready():
    instance = client()
    instance._get_model_deployment_job = MagicMock(
        return_value=marked_job(status="JOB_STATE_PENDING")
    )
    with pytest.raises(TimeoutError, match="did not become READY"):
        instance.wait_for_model_deployment(DEPLOYMENT_ID)


def test_wait_hides_backing_job_on_request_error():
    instance = client()
    instance._get_model_deployment_job = MagicMock(
        side_effect=requests.exceptions.ConnectionError(f"failed for {JOB_ID}")
    )
    with pytest.raises(RuntimeError, match="request failed") as exc_info:
        instance.wait_for_model_deployment(DEPLOYMENT_ID)
    assert JOB_ID not in str(exc_info.value).replace(DEPLOYMENT_ID, "")


@pytest.mark.parametrize("method", ["logs", "metrics"])
def test_downloads_hide_backing_job_on_request_error(tmp_path, method):
    instance = client()
    instance._get_model_deployment_job = MagicMock(
        side_effect=requests.exceptions.ConnectionError(f"failed for {JOB_ID}")
    )
    download = (
        instance.download_model_deployment_logs
        if method == "logs"
        else instance.download_model_deployment_metrics
    )
    with pytest.raises(RuntimeError, match="request failed") as exc_info:
        download(DEPLOYMENT_ID, tmp_path)
    assert JOB_ID not in str(exc_info.value).replace(DEPLOYMENT_ID, "")


def test_download_rejects_unknown_log_type(tmp_path):
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())
    with pytest.raises(ValueError, match="log_type must be either"):
        instance.download_model_deployment_logs(DEPLOYMENT_ID, tmp_path, log_type="debug")


@pytest.mark.parametrize("method", ["stdout", "metrics"])
def test_download_fails_when_the_downloaded_file_is_missing(tmp_path, method):
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())
    missing = [{"sub_job_id": SUB_JOB_ID, "filename": "f.log", "saved_path": str(tmp_path / "gone")}]
    if method == "stdout":
        instance.download_stdout_logs = MagicMock(return_value=missing)
        with pytest.raises(RuntimeError, match="stdout download is missing"):
            instance.download_model_deployment_logs(DEPLOYMENT_ID, tmp_path, log_type="stdout")
    else:
        instance.download_metrics = MagicMock(return_value=missing)
        with pytest.raises(RuntimeError, match="metrics download is missing"):
            instance.download_model_deployment_metrics(DEPLOYMENT_ID, tmp_path)


def test_describe_shows_creation_spec_scope_and_ownership():
    instance = client()
    instance._get_model_deployment_job = MagicMock(
        return_value=marked_job(checkpoint=True)
    )

    description = instance.describe_model_deployment(DEPLOYMENT_ID)

    assert description == {
        "deployment": DEPLOYMENT_ID,
        "status": "READY",
        "base_url": instance.model_deployment_base_url,
        "database": "DB",
        "schema": "SCH",
        "created_at": "2026-09-16T20:00:00Z",
        "creator": "CURRENT_USER",
        "owner_role": "CURRENT_OWNER_ROLE",
        "spec": {
            "deployment": {
                "class": "EVALUATION",
                "min_replicas": 3,
                "gpus_per_replica": 2,
                "gpu_type": "H200",
            },
            "source": {
                "type": "checkpoint",
                "job_id": "source-training-job",
                "checkpoint_id": "cp_completed_weights_only",
            },
            "inference": {"max_seq_len": 8192},
        },
    }
    instance._get_model_deployment_job.assert_called_once_with(JOB_ID)


def test_describe_groups_the_configuration_under_spec():
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())

    spec = instance.describe_model_deployment(DEPLOYMENT_ID)["spec"]

    assert spec == {
        "deployment": {
            "class": "EVALUATION",
            "min_replicas": 3,
            "gpus_per_replica": 2,
            "gpu_type": "H200",
        },
        "source": {"type": "base_model", "model": "Qwen/Qwen3-1.7B"},
        "inference": {"max_seq_len": 8192},
    }


def test_describe_hides_backing_ids_credentials_and_service_config():
    instance = client()
    instance._get_model_deployment_job = MagicMock(
        return_value=marked_job(checkpoint=True)
    )

    rendered = repr(instance.describe_model_deployment(DEPLOYMENT_ID))

    assert JOB_ID not in rendered.replace(DEPLOYMENT_ID, "")
    assert SUB_JOB_ID not in rendered
    assert "must-not-escape" not in rendered
    assert "max_inflight_choices" not in rendered


def test_describe_reports_failure_for_failed_deployment():
    job = marked_job(status="JOB_STATE_FAILED")
    job["reason"] = "sub_job_failed"
    job["reason_details"] = "vLLM ran out of GPU memory loading the model"
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=job)

    description = instance.describe_model_deployment(DEPLOYMENT_ID)

    assert description["status"] == "FAILED"
    assert description["failure"] == {
        "code": "sub_job_failed",
        "message": "vLLM ran out of GPU memory loading the model",
    }
    assert "status_reason" not in description


def test_describe_redacts_backing_job_id_from_http_errors():
    instance = client()
    response = requests.Response()
    response.status_code = 404
    instance._get_model_deployment_job = MagicMock(
        side_effect=requests.exceptions.HTTPError(
            f"404 for /jobs/{JOB_ID}", response=response
        )
    )

    with pytest.raises(RuntimeError) as excinfo:
        instance.describe_model_deployment(DEPLOYMENT_ID)

    assert str(excinfo.value) == (
        f"model deployment {DEPLOYMENT_ID} request failed with HTTP 404"
    )



def test_create_leaves_model_support_to_the_server():
    # The server decides which models are supported,
    # so a model this client has never heard of is sent, not rejected locally.
    instance = client()
    instance.create_job_from_body = MagicMock(return_value={"job_id": JOB_ID})
    value = deployment_spec()
    value["source"]["model"] = "Qwen/Qwen3-72B"

    instance.create_model_deployment(value)

    [sub_job] = instance.create_job_from_body.call_args.args[0]["sub_job_configs"]
    assert sub_job["model_name"] == "Qwen/Qwen3-72B"


def test_list_reports_deployments_with_models_and_shapes_this_client_does_not_know():
    # One unfamiliar deployment must not hide the others. Only the server's admission
    # rules changed here; the records are internally consistent.
    unknown_model = marked_job()
    unknown_model["job_id"] = "22345678-1234-4234-8234-123456789abc"
    unknown_model["sub_jobs"][0]["model_name"] = "Qwen/Qwen3-72B"
    unusual_shape = _broken(num_replicas=3.0, n_gpus=12.0, tensor_parallel_size=4.0)
    unusual_shape["job_id"] = "32345678-1234-4234-8234-123456789abc"
    instance = client()
    instance.list_jobs = MagicMock(return_value=[marked_job(), unknown_model, unusual_shape])

    deployments = instance.list_model_deployments()

    assert len(deployments) == 3
    assert deployments[1]["model_source"] == {"type": "base_model", "model": "Qwen/Qwen3-72B"}
    assert (deployments[2]["replicas"], deployments[2]["gpus_per_replica"]) == (3, 4)


def test_missing_hardware_is_reported_as_unknown_not_h200():
    # A job read while the server cannot reach its scheduler carries no hardware.
    job = marked_job(status="JOB_STATE_UNKNOWN")
    del job["hardware"]
    instance = client()
    instance.list_jobs = MagicMock(return_value=[job])
    instance._get_model_deployment_job = MagicMock(return_value=job)

    [listed] = instance.list_model_deployments()
    described = instance.describe_model_deployment(DEPLOYMENT_ID)

    assert listed["status"] == "UNKNOWN"
    assert listed["gpu_type"] is None
    assert described["spec"]["deployment"]["gpu_type"] is None


def test_describe_reports_checkpoint_source_in_the_shape_the_server_returns():
    # What the server renders once it echoes source_checkpoint_info on reads: the
    # proto field names, flattened into the sub-job, stage_info stripped, and
    # is_external omitted while false.
    job = marked_job()
    job["sub_jobs"][0]["source_checkpoint_info"] = {
        "checkpoint_id": "cp_completed_weights_only",
        "source_job_id": "source-training-job",
    }
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=job)

    spec = instance.describe_model_deployment(DEPLOYMENT_ID)["spec"]

    assert spec["source"] == {
        "type": "checkpoint",
        "job_id": "source-training-job",
        "checkpoint_id": "cp_completed_weights_only",
    }


def _http_error_naming_the_backing_job(status_code: int = 503):
    response = requests.Response()
    response.status_code = status_code
    response.url = f"http://test.local/jobs/{JOB_ID}/experiment-run"
    response.headers["x-snowflake-request-id"] = "request-123"
    response._content = f'{{"message":"job {JOB_ID} unavailable"}}'.encode()
    return requests.exceptions.HTTPError(
        f"{status_code} Server Error for url: {response.url}", response=response
    )


@pytest.mark.parametrize(
    "helper,download",
    [
        (
            "fetch_execution_logs",
            lambda instance, path: instance.download_model_deployment_logs(DEPLOYMENT_ID, path),
        ),
        (
            "download_stdout_logs",
            lambda instance, path: instance.download_model_deployment_logs(
                DEPLOYMENT_ID, path, log_type="stdout"
            ),
        ),
        (
            "download_metrics",
            lambda instance, path: instance.download_model_deployment_metrics(DEPLOYMENT_ID, path),
        ),
    ],
)
def test_download_redacts_backing_job_id_from_download_request_errors(
    tmp_path, helper, download
):
    # The downloads make their own requests (e.g. GET /{job_id}/experiment-run)
    # after the deployment lookup succeeds; their errors must be redacted too.
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())
    setattr(instance, helper, MagicMock(side_effect=_http_error_naming_the_backing_job()))

    with pytest.raises(RuntimeError, match="HTTP 503") as exc_info:
        download(instance, tmp_path)

    assert "request-123" in str(exc_info.value)
    assert JOB_ID not in str(exc_info.value).replace(DEPLOYMENT_ID, "")


def test_download_leaves_local_filesystem_errors_unchanged(tmp_path):
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())
    instance.fetch_execution_logs = MagicMock(
        return_value=[{"filename": "execution.jsonl", "content": "x"}]
    )
    (tmp_path / "execution.jsonl").mkdir()  # writing the log file now fails

    with pytest.raises(OSError):
        instance.download_model_deployment_logs(DEPLOYMENT_ID, tmp_path)


def _server_error_response(status_code: int, body: dict) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.url = f"http://test.local/jobs/{JOB_ID}"
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps(body).encode()
    return response


def test_drop_of_missing_deployment_makes_one_lookup(monkeypatch):
    # Transport level: the real get_job/_send path, not a mocked get_job.
    monkeypatch.setattr(client_module.time, "sleep", lambda *_: None)
    instance = client()
    instance._session = MagicMock()
    instance._session.get.return_value = _server_error_response(
        404,
        {"code": "517601", "message": f"Training job {JOB_ID} not found in SCH"},
    )
    instance.cancel_job = MagicMock()

    assert instance.drop_model_deployment(DEPLOYMENT_ID) == {
        "deployment": DEPLOYMENT_ID,
        "dropped": True,
    }
    assert instance._session.get.call_count == 1
    instance.cancel_job.assert_not_called()


def test_drop_still_retries_a_routing_404_without_the_job_not_found_code(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda *_: None)
    instance = client()
    instance._session = MagicMock()
    ready = MagicMock(status_code=200)
    ready.json.return_value = marked_job()
    ready.raise_for_status = MagicMock()
    instance._session.get.side_effect = [_server_error_response(404, {"message": "no route"}), ready]
    instance.cancel_job = MagicMock()

    assert instance.drop_model_deployment(DEPLOYMENT_ID)["dropped"] is True
    assert instance._session.get.call_count == 2
    instance.cancel_job.assert_called_once_with(JOB_ID)


def test_other_lookups_keep_retrying_job_not_found_404s_for_a_just_created_job(monkeypatch):
    # Deployment lookups treat job-not-found as final even for a job this client
    # just created; ordinary get_job still rides out the read-after-create window.
    monkeypatch.setattr(client_module.time, "sleep", lambda *_: None)
    instance = CortexTrainingClient(
        base_url="http://test.local", database="DB", schema="SCH", max_retries=2
    )
    instance._note_created_job({"job_id": JOB_ID})
    instance._session = MagicMock()
    instance._session.get.return_value = _server_error_response(404, {"code": "517601"})

    with pytest.raises(requests.exceptions.HTTPError):
        instance.get_job(JOB_ID)
    assert instance._session.get.call_count == 3


def _session_returning(*responses):
    instance = client()
    instance._session = MagicMock()
    instance._session.get.side_effect = list(responses)
    return instance


def test_describe_of_missing_deployment_fails_fast_without_logging_the_job_id(
    monkeypatch, caplog
):
    monkeypatch.setattr(client_module.time, "sleep", lambda *_: None)
    not_found = _server_error_response(
        404, {"code": "517601", "message": f"Training job {JOB_ID} not found in SCH"}
    )
    instance = _session_returning(not_found)

    with pytest.raises(RuntimeError) as exc_info:
        instance.describe_model_deployment(DEPLOYMENT_ID)

    assert str(exc_info.value) == f"model deployment {DEPLOYMENT_ID} not found in DB.SCH"
    assert instance._session.get.call_count == 1
    assert JOB_ID not in caplog.text.replace(DEPLOYMENT_ID, "")


def test_request_log_lines_name_the_deployment_not_the_backing_job(monkeypatch, caplog):
    monkeypatch.setattr(client_module.time, "sleep", lambda *_: None)
    busy = _server_error_response(503, {"message": "busy"})
    busy.headers["x-snowflake-request-id"] = "REQUEST_ID"
    busy.request = requests.Request("GET", f"http://test.local/jobs/{JOB_ID}").prepare()
    ready = MagicMock(status_code=200)
    ready.json.return_value = marked_job()
    ready.raise_for_status = MagicMock()
    instance = _session_returning(busy, ready)

    with caplog.at_level("DEBUG", logger=client_module.logger.name):
        instance.describe_model_deployment(DEPLOYMENT_ID)

    assert "snowflake request_id=REQUEST_ID" in caplog.text
    assert DEPLOYMENT_ID in caplog.text
    assert JOB_ID not in caplog.text.replace(DEPLOYMENT_ID, "")


@pytest.mark.parametrize(
    "status_code,body,expected",
    [
        (
            403,
            {"code": "517600", "message": "Cortex Training API is not enabled for this account"},
            "with HTTP 403 (517600): Cortex Training API is not enabled for this account",
        ),
        (
            400,
            {"code": "517602", "message": "Schema SCH is not found or not authorized"},
            "with HTTP 400 (517602): Schema SCH is not found or not authorized",
        ),
        (
            409,
            {"code": "517602", "message": f"job {JOB_ID} is busy"},
            f"with HTTP 409 (517602): job {DEPLOYMENT_ID} is busy",
        ),
    ],
)
def test_deployment_errors_keep_server_code_and_message(status_code, body, expected):
    error = requests.exceptions.HTTPError(
        f"{status_code} for url: http://test.local/jobs/{JOB_ID}",
        response=_server_error_response(status_code, body),
    )

    rendered = str(
        client_module._sanitized_model_deployment_request_error(
            error, DEPLOYMENT_ID, job_id=JOB_ID, scope="DB.SCH"
        )
    )

    assert rendered == f"model deployment {DEPLOYMENT_ID} request failed {expected}"
    assert "http://" not in rendered


def test_create_errors_keep_gs_message_but_not_the_url_or_raw_body():
    instance = client()
    error = requests.exceptions.HTTPError(
        "400 for url: http://test.local/api/v2/databases/DB/schemas/SCH/cortex-training",
        response=_server_error_response(
            400, {"code": "517602", "message": "Unsupported model for a model deployment: x/y"}
        ),
    )
    instance.create_job_from_body = MagicMock(side_effect=error)

    with pytest.raises(RuntimeError) as exc_info:
        instance.create_model_deployment(deployment_spec(model="x/y"))

    assert str(exc_info.value) == (
        "model deployment request failed with HTTP 400 (517602): "
        "Unsupported model for a model deployment: x/y"
    )


def _requests_ssl_error(cause: BaseException) -> requests.exceptions.SSLError:
    """Wrap ``cause`` the way a ``requests`` call surfaces a failed TLS handshake."""
    try:
        try:
            raise urllib3.exceptions.SSLError(cause) from cause
        except urllib3.exceptions.SSLError as urllib3_error:
            raise urllib3.exceptions.MaxRetryError(None, "/", urllib3_error) from urllib3_error
    except urllib3.exceptions.MaxRetryError as exhausted:
        return requests.exceptions.SSLError(exhausted)


def test_certificate_verification_errors_are_not_retried():
    error = _requests_ssl_error(
        ssl.SSLCertVerificationError(1, "certificate verify failed: Hostname mismatch")
    )

    assert not client_module._is_transient(error)


@pytest.mark.parametrize(
    "cause",
    [
        ssl.SSLEOFError(8, "EOF occurred in violation of protocol"),
        ConnectionResetError(54, "Connection reset by peer"),
    ],
)
def test_interrupted_tls_handshakes_are_retried(cause):
    assert client_module._is_transient(_requests_ssl_error(cause))


def test_tls_errors_without_a_certificate_cause_are_retried():
    assert client_module._is_transient(requests.exceptions.SSLError("handshake failed"))
    assert client_module._is_transient(requests.exceptions.ConnectionError("reset"))


def test_wait_timeout_says_the_deployment_is_still_running():
    instance = client()
    instance._get_model_deployment_job = MagicMock(
        return_value=marked_job(status="JOB_STATE_PENDING")
    )

    with pytest.raises(client_module.ModelDeploymentWaitTimeout) as exc_info:
        instance.wait_for_model_deployment(DEPLOYMENT_ID)

    message = str(exc_info.value)
    assert f"model deployment {DEPLOYMENT_ID} did not become READY" in message
    assert "last status: PENDING" in message
    assert "still running and holding GPUs" in message
    assert f"model-deployment drop {DEPLOYMENT_ID}" in message
    assert exc_info.value.deployment["status"] == "PENDING"
    assert isinstance(exc_info.value, TimeoutError)


def test_wait_failure_names_the_failure_code_and_next_step():
    job = marked_job(status="JOB_STATE_FAILED")
    job["reason"] = "sub_job_failed"
    job["reason_details"] = "vLLM ran out of GPU memory loading the model"
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=job)

    with pytest.raises(client_module.ModelDeploymentWaitError) as exc_info:
        instance.wait_for_model_deployment(DEPLOYMENT_ID)

    assert str(exc_info.value) == (
        f"model deployment {DEPLOYMENT_ID} is FAILED (sub_job_failed): "
        f"vLLM ran out of GPU memory loading the model; "
        f"for details run `cortex-training model-deployment download-log {DEPLOYMENT_ID}`"
    )
    assert exc_info.value.deployment["failure"] == {
        "code": "sub_job_failed",
        "message": "vLLM ran out of GPU memory loading the model",
    }


def test_wait_on_a_dropped_deployment_says_so_without_internal_wording():
    job = marked_job(status="JOB_STATE_CANCELLED")
    job["reason"] = "cancelled_by_client"
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=job)

    with pytest.raises(client_module.ModelDeploymentWaitError) as exc_info:
        instance.wait_for_model_deployment(DEPLOYMENT_ID)

    assert str(exc_info.value).startswith(f"model deployment {DEPLOYMENT_ID} was dropped")
    assert "cancelled_by_client" not in str(exc_info.value)


@pytest.mark.parametrize(
    "job_status,deployment_status",
    [
        ("JOB_STATE_CANCELLING", "DROPPING"),
        ("JOB_STATE_CANCELLED", "DROPPED"),
        ("JOB_STATE_FAILED", "FAILED"),
        ("JOB_STATE_TERMINATED", "TERMINATED"),
    ],
)
def test_describe_omits_base_url_once_the_deployment_stops_serving(job_status, deployment_status):
    job = marked_job(status=job_status)
    job["reason"] = "cancelled_by_client"
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=job)

    described = instance.describe_model_deployment(DEPLOYMENT_ID)

    assert described["status"] == deployment_status
    assert "base_url" not in described
    assert "status_reason" not in described


def test_describe_keeps_other_status_reasons():
    job = marked_job(status="JOB_STATE_PENDING")
    job["reason"] = "insufficient_capacity"
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=job)

    described = instance.describe_model_deployment(DEPLOYMENT_ID)

    assert described["status_reason"] == "insufficient_capacity"
    assert described["base_url"] == instance.model_deployment_base_url


def test_describe_includes_status_details():
    job = marked_job(status="JOB_STATE_PENDING")
    job["reason"] = "quota_filled"
    job["reason_details"] = "needs 8 GPUs; 64 currently active"
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=job)

    described = instance.describe_model_deployment(DEPLOYMENT_ID)

    assert described["status_reason"] == "quota_filled"
    assert described["status_details"] == "needs 8 GPUs; 64 currently active"


@pytest.mark.parametrize(
    "job_status",
    ["JOB_STATE_CANCELLING", "JOB_STATE_CANCELLED"],
)
def test_drop_reason_and_its_details_are_not_repeated(job_status):
    job = marked_job(status=job_status)
    job["reason"] = "cancelled_by_client"
    job["reason_details"] = "cancelled through the API"

    metadata = client_module._model_deployment_metadata(job, DEPLOYMENT_ID)

    assert "status_reason" not in metadata
    assert "status_details" not in metadata


def test_list_filters_by_deployment_status():
    dropped = marked_job(status="JOB_STATE_CANCELLED")
    dropped["job_id"] = "22345678-1234-4234-8234-123456789abc"
    instance = client()
    instance.list_jobs = MagicMock(return_value=[marked_job(), dropped])

    assert [d["status"] for d in instance.list_model_deployments()] == ["READY", "DROPPED"]
    assert [d["status"] for d in instance.list_model_deployments(status="ready")] == ["READY"]
    assert [
        d["status"] for d in instance.list_model_deployments(status=["READY", "DROPPED"])
    ] == ["READY", "DROPPED"]
    with pytest.raises(ValueError, match="unknown deployment status RUNNING"):
        instance.list_model_deployments(status="running")


def test_downloaded_files_are_private_to_the_owner(tmp_path):
    instance = client()
    instance._get_model_deployment_job = MagicMock(return_value=marked_job())
    instance.fetch_execution_logs = MagicMock(
        return_value=[{"filename": "execution.jsonl", "content": "x"}]
    )

    def write_metrics(_job_id, output_dir):
        path = Path(output_dir) / SUB_JOB_ID / "gpu.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")
        path.chmod(0o644)
        return [{"filename": "gpu.jsonl", "saved_path": str(path), "chunk_count": 1}]

    instance.download_metrics = write_metrics

    instance.download_model_deployment_logs(DEPLOYMENT_ID, tmp_path / "logs")
    instance.download_model_deployment_metrics(DEPLOYMENT_ID, tmp_path / "metrics")

    assert (tmp_path / "logs" / "execution.jsonl").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "metrics" / "gpu.jsonl").stat().st_mode & 0o777 == 0o600


def test_checkpoint_example_is_a_valid_spec():
    path = Path(__file__).resolve().parents[1] / "recipes/api_quickstart/model-deployment-checkpoint.json"

    spec = ModelDeploymentSpec.from_dict(json.loads(path.read_text()))

    assert spec.source_type == "checkpoint"
