#!/usr/bin/env python3
"""Dry-run or execute a Cortex Training catalog profile as a real job."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from validate_model_catalog import load_catalog
from validate_model_catalog import validate_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"

PROFILE_KEYS = ("inference", "sftLora", "sftFull", "rlLora", "rlFull")
TRAINING_PROFILE_KEYS = ("sftLora", "sftFull", "rlLora", "rlFull")
SAMPLING_PROFILE_KEYS = ("inference", "rlLora", "rlFull")
DEBUG_OPTIONS_ENV = "CORTEX_TRAINING_ENABLE_DEBUG_OPTIONS"
TOP_LEVEL_ARGS = {
    "global_batch_size",
    "dtype",
    "seed",
    "model_post_init",
    "source_checkpoint_info",
}


def _find_model(models_doc: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = models_doc.get("models", [])
    model = next((item for item in models if item.get("modelId") == model_id), None)
    if model is None:
        raise ValueError(f"unknown catalog model: {model_id}")
    return model


def _recommendation(
    model: dict[str, Any],
    profile_key: str,
    *,
    long_sequence: bool,
) -> dict[str, Any]:
    capabilities = model["capabilities"]
    if profile_key == "inference":
        recommendation = capabilities["inference"]
        if not recommendation["supported"]:
            raise ValueError(f"{model['modelId']} does not support inference")
    else:
        training = capabilities["training"]
        if not training["supported"]:
            raise ValueError(f"{model['modelId']} does not support training")
        recommendation = training["profiles"][profile_key]
    if long_sequence:
        recommendation = recommendation.get("longSequence")
        if recommendation is None:
            raise ValueError(
                f"{model['modelId']} {profile_key} has no long-sequence recommendation"
            )
    return recommendation


def _build_wire_sub_job(
    model_id: str,
    sub_job: dict[str, Any],
    max_seq_len: int,
) -> dict[str, Any]:
    builder = sub_job["builder"]
    args = copy.deepcopy(sub_job["args"])
    args["max_seq_len"] = max_seq_len
    extra_key = "extra_training" if builder == "training_job" else "extra_sampling"
    config_key = "training_config" if builder == "training_job" else "inference_config"
    job_type = (
        "training" if builder == "training_job" else args.pop("job_type", "sampling")
    )

    wire = {
        "job_type": job_type,
        "model_name": model_id,
    }
    for key in TOP_LEVEL_ARGS:
        if key in args:
            wire[key] = args.pop(key)

    extra = args.pop(extra_key, {})
    for key, value in extra.items():
        args.setdefault(key, value)
    wire[config_key] = args
    return wire


def build_profile_request(
    config_dir: Path,
    repo_root: Path,
    model_id: str,
    profile_key: str,
    *,
    long_sequence: bool = False,
) -> dict[str, Any]:
    """Build a CreateJob body for a standard or long-sequence recommendation."""
    if profile_key not in PROFILE_KEYS:
        raise ValueError(f"invalid profile key: {profile_key}")

    models_doc, profiles = load_catalog(config_dir)
    validate_catalog(models_doc, profiles, repo_root)
    model = _find_model(models_doc, model_id)
    recommendation = _recommendation(
        model,
        profile_key,
        long_sequence=long_sequence,
    )
    profile_id = recommendation["recommendedProfileId"]
    profile = next(item for item in profiles if item["id"] == profile_id)
    max_context_tokens = recommendation["maxContextTokens"]
    request = {
        "sub_job_configs": [
            _build_wire_sub_job(model_id, sub_job, max_context_tokens)
            for sub_job in profile["subJobs"]
        ]
    }
    return request


def enable_training_memory_telemetry(request: dict[str, Any]) -> None:
    """Enable server-side peak memory reporting without changing placement."""
    for sub_job in request["sub_job_configs"]:
        if sub_job["job_type"] != "training":
            continue
        training = sub_job["training_config"]
        training["step_peak_memory_log"] = True
        training["training_memory_telemetry"] = True


def apply_debug_image_tag(request: dict[str, Any], image_tag: str) -> None:
    """Select a specific backend image for a live smoke test."""
    os.environ[DEBUG_OPTIONS_ENV] = "1"
    request["debug"] = {"job": {"image_tag": image_tag}}


def apply_training_sequence_parallelism(
    request: dict[str, Any],
    sp_size: int,
) -> None:
    """Apply Ulysses SP and its logical data-parallel batch-size contract."""
    if sp_size <= 1:
        raise ValueError("training sequence-parallel size must be greater than 1")
    for sub_job in request["sub_job_configs"]:
        if sub_job["job_type"] != "training":
            continue
        training = sub_job["training_config"]
        n_gpus = training["n_gpus"]
        if n_gpus % sp_size != 0:
            raise ValueError(
                f"training n_gpus {n_gpus} is not divisible by sp_size {sp_size}"
            )
        logical_dp = n_gpus // sp_size
        training["sp_size"] = sp_size
        training["train_batch_size"] = logical_dp
        ds_config = training["ds_config"]
        ds_config["train_batch_size"] = logical_dp
        ds_config["train_micro_batch_size_per_gpu"] = 1
        ds_config["gradient_accumulation_steps"] = 1


def build_forward_backward_probe_spec(
    request: dict[str, Any],
    profile_key: str,
    *,
    full_context: bool = False,
) -> dict[str, Any]:
    """Build a tokenizer-independent training batch for a catalog job."""
    training = next(
        (
            sub_job["training_config"]
            for sub_job in request["sub_job_configs"]
            if sub_job["job_type"] == "training"
        ),
        None,
    )
    if training is None:
        raise ValueError("forward-backward probe requires a training sub-job")

    batch_size = training.get("train_batch_size")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("training_config.train_batch_size must be a positive integer")

    sequence_tokens = training["max_seq_len"] if full_context else 8
    token_ids = [1] * sequence_tokens if full_context else [1, 2, 3, 4, 5, 6, 7, 8]
    input_ids = [token_ids[:] for _ in range(batch_size)]
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "position_ids": "arange",
    }
    spec: dict[str, Any] = {"payload": payload}
    token_chunk_size = training.get("fused_lm_head_token_chunk_size")
    if token_chunk_size is None:
        token_chunk_size = (training.get("prime_rl") or {}).get(
            "fused_lm_head_token_chunk_size"
        )
    chunked_sft = (
        profile_key in ("sftLora", "sftFull")
        and isinstance(token_chunk_size, int)
        and not isinstance(token_chunk_size, bool)
    )
    if profile_key in ("rlLora", "rlFull") or chunked_sft:
        payload.update(
            {
                "attention_mask": [[1] * len(token_ids) for _ in range(batch_size)],
                "include_attention_mask": True,
                "labels": "none",
            }
        )
        if chunked_sft:
            loss_mask = [1.0] * (sequence_tokens - 1) + [0.0]
            processing = {
                "loss_fn": "causal_cross_entropy",
                "post": ["compute_logprobs"],
                "config": {},
            }
        else:
            loss_mask = [0.0, 0.0] + [1.0] * (sequence_tokens - 3) + [0.0]
            processing = {
                "loss_fn": "grpo",
                "post": ["compute_logprobs"],
                "config": {
                    "eps_clip": 0.2,
                    "loss_agg_mode": "token-mean",
                    "entropy_coeff": 0.0,
                    "global_batch_size": batch_size,
                },
            }
        spec.update(
            {
                "context": {
                    "input_ids": {"data": input_ids, "dtype": "long"},
                    "advantages": {
                        "data": [loss_mask[:] for _ in range(batch_size)],
                        "dtype": "float32",
                    },
                    "loss_mask": {
                        "data": [loss_mask[:] for _ in range(batch_size)],
                        "dtype": "float32",
                    },
                },
                "processing": processing,
            }
        )
    else:
        payload["labels"] = {
            "strategy": "next_token",
            "mask_padding": False,
        }
    return spec


def _build_forward_backward_probe_payload(
    request: dict[str, Any],
    profile_key: str,
    full_context: bool,
) -> bytes:
    sys.path.insert(0, str(SOURCE_ROOT))
    from cortex_training import build_forward_backward_kwargs
    from cortex_training import build_forward_backward_payload
    from cortex_training import wire

    spec = build_forward_backward_probe_spec(
        request,
        profile_key,
        full_context=full_context,
    )
    if "processing" not in spec:
        return build_forward_backward_payload(spec)

    kwargs = build_forward_backward_kwargs(spec["payload"])
    kwargs["use_cache"] = False
    context = build_forward_backward_kwargs({"kwargs": spec["context"]})
    return wire.dumps(
        {
            "args": (),
            "kwargs": kwargs,
            "context": context,
            "processing": spec["processing"],
        },
        metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}},
    )


def _sampling_max_seq_len(request: dict[str, Any]) -> int:
    sampling = next(
        (
            sub_job["inference_config"]
            for sub_job in request["sub_job_configs"]
            if sub_job["job_type"] == "sampling"
        ),
        None,
    )
    if sampling is None:
        raise ValueError("generate probe requires a sampling sub-job")
    max_seq_len = sampling.get("max_seq_len")
    if (
        isinstance(max_seq_len, bool)
        or not isinstance(max_seq_len, int)
        or max_seq_len <= 1
    ):
        raise ValueError("inference_config.max_seq_len must be an integer > 1")
    return max_seq_len


def _run_generate_probe(
    client,
    job_id: str,
    request: dict[str, Any],
    full_context_prefill: bool,
) -> dict[str, Any]:
    if full_context_prefill:
        prompt_tokens = _sampling_max_seq_len(request) - 1
        prompts: list[str | list[int]] = [[1] * prompt_tokens]
    else:
        prompt_tokens = None
        prompts = ["Reply with OK."]

    request_id = client.generate(
        job_id,
        prompts=prompts,
        sampling_params={
            "max_tokens": 1,
            "temperature": 0.0,
        },
    )
    result = client.poll_request(job_id, request_id)
    generated = result.get("results")
    if not isinstance(generated, list) or len(generated) != 1:
        raise RuntimeError(f"generate probe expected one result, got {generated!r}")
    return {
        "operation": "generate",
        "requestId": request_id,
        "resultCount": len(generated),
        "promptTokens": prompt_tokens,
    }


def _run_forward_backward_probe(
    client,
    job_id: str,
    profile_key: str,
    request: dict[str, Any],
    full_context: bool,
) -> dict[str, Any]:
    payload = _build_forward_backward_probe_payload(
        request,
        profile_key,
        full_context,
    )
    request_id = client.forward_backward(job_id, payload)
    result = client.poll_request(job_id, request_id)
    avg_loss = result.get("avg_loss")
    try:
        avg_loss_value = float(avg_loss)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"forward-backward probe returned invalid avg_loss: {avg_loss!r}"
        ) from exc
    if not math.isfinite(avg_loss_value):
        raise RuntimeError(
            f"forward-backward probe returned non-finite avg_loss: {avg_loss!r}"
        )
    result = {
        "operation": "forward-backward",
        "requestId": request_id,
        "avgLoss": avg_loss_value,
    }
    if full_context:
        training = next(
            sub_job["training_config"]
            for sub_job in request["sub_job_configs"]
            if sub_job["job_type"] == "training"
        )
        result.update(
            {
                "sequenceTokens": training["max_seq_len"],
                "globalTokens": (
                    training["max_seq_len"] * training["train_batch_size"]
                ),
            }
        )
    return result


def _run_optimizer_step_probe(
    client, job_id: str, request: dict[str, Any]
) -> dict[str, Any]:
    training = next(
        sub_job["training_config"]
        for sub_job in request["sub_job_configs"]
        if sub_job["job_type"] == "training"
    )
    optimizer = training.get("optimizer") or {}
    request_id = client.step(job_id, learning_rate=optimizer.get("lr"))
    result = client.poll_request(job_id, request_id)
    return {
        "operation": "step",
        "requestId": request_id,
        "globalSteps": result.get("global_steps"),
        "peakMemory": result.get("peak_memory"),
    }


def _checkpoint_id(result: dict[str, Any]) -> str:
    checkpoint_id = result.get("checkpoint_id")
    if isinstance(checkpoint_id, str) and checkpoint_id:
        return checkpoint_id
    for key in ("stage_path", "checkpoint_path"):
        path = result.get(key)
        if not isinstance(path, str):
            continue
        for part in path.split("/"):
            if part.startswith("cp_"):
                return part
    raise RuntimeError(f"checkpoint save returned no checkpoint id: {result!r}")


def _run_checkpoint_round_trip_probe(
    client,
    job_id: str,
) -> list[dict[str, Any]]:
    save_request_id = client.save(job_id, checkpoint_type="resumable")
    save_result = client.poll_request(job_id, save_request_id)
    checkpoint_id = _checkpoint_id(save_result)

    try:
        load_request_id = client.load(
            job_id,
            checkpoint_id,
            source_job_id=job_id,
            target_sub_job_id=f"{job_id}:training:0",
        )
        load_result = client.poll_request(job_id, load_request_id)
        loaded_checkpoint_id = load_result.get("checkpoint_id")
        if loaded_checkpoint_id not in (None, checkpoint_id):
            raise RuntimeError(
                "checkpoint load returned a different checkpoint id: "
                f"{loaded_checkpoint_id!r} != {checkpoint_id!r}"
            )
    finally:
        client.delete_checkpoint(job_id, checkpoint_id)
    return [
        {
            "operation": "save",
            "requestId": save_request_id,
            "checkpointId": checkpoint_id,
            "checkpointType": "resumable",
        },
        {
            "operation": "load",
            "requestId": load_request_id,
            "checkpointId": checkpoint_id,
        },
    ]


def run_data_plane_probes(
    client,
    job_id: str,
    profile_key: str,
    request: dict[str, Any],
    *,
    full_context_prefill: bool = False,
    full_context_training: bool = False,
    training_step: bool = False,
    checkpoint_round_trip: bool = False,
) -> list[dict[str, Any]]:
    """Execute the minimal operations required by a catalog workflow."""
    probes = []
    if profile_key in SAMPLING_PROFILE_KEYS:
        probes.append(
            _run_generate_probe(
                client,
                job_id,
                request,
                full_context_prefill,
            )
        )
    if profile_key in TRAINING_PROFILE_KEYS:
        probes.append(
            _run_forward_backward_probe(
                client,
                job_id,
                profile_key,
                request,
                full_context_training,
            )
        )
        if training_step or checkpoint_round_trip:
            probes.append(_run_optimizer_step_probe(client, job_id, request))
        if checkpoint_round_trip:
            probes.extend(_run_checkpoint_round_trip_probe(client, job_id))
    return probes


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _required_env(*names: str) -> str:
    value = _first_env(*names)
    if value is None:
        raise ValueError(f"set one of: {', '.join(names)}")
    return value


def _build_client(poll_timeout: float):
    sys.path.insert(0, str(SOURCE_ROOT))
    from cortex_training import CortexTrainingClient

    host = _required_env("SNOWFLAKE_HOST")
    host = host.removeprefix("https://").removeprefix("http://").rstrip("/")
    return CortexTrainingClient.from_pat(
        host=host,
        pat=_required_env("SNOWFLAKE_PAT"),
        database=_required_env("SNOWFLAKE_DATABASE"),
        schema=_first_env("SNOWFLAKE_SCHEMA") or "PUBLIC",
        endpoint=_first_env("CORTEX_TRAINING_ENDPOINT") or "cortex-training",
        poll_timeout=poll_timeout,
    )


def _cancel_job(client, job_id: str, active_error: BaseException | None) -> bool:
    try:
        client.cancel_job(job_id)
    except Exception as cancel_error:
        if active_error is None:
            raise
        print(
            f"WARNING: failed to cancel {job_id}: {cancel_error}",
            file=sys.stderr,
        )
        return False
    return True


def main() -> int:
    default_config = REPO_ROOT / "model-catalog"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--profile",
        action="append",
        choices=PROFILE_KEYS,
        required=True,
        help="Catalog profile to test. Repeat to test multiple profiles serially.",
    )
    parser.add_argument("--config-dir", type=Path, default=default_config)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Submit real jobs and execute workflow data-plane probes. "
            "Without this flag, print the generated request bodies."
        ),
    )
    parser.add_argument("--poll-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--full-context-prefill",
        action="store_true",
        help=(
            "For sampling profiles, prefill max_seq_len - 1 token IDs and "
            "generate one token instead of running the short text probe."
        ),
    )
    parser.add_argument(
        "--full-context-training",
        action="store_true",
        help=(
            "For training profiles, execute forward-backward with "
            "train_batch_size sequences of exactly max_seq_len tokens."
        ),
    )
    parser.add_argument(
        "--training-step",
        action="store_true",
        help="Run and poll one optimizer step after forward-backward.",
    )
    parser.add_argument(
        "--checkpoint-round-trip",
        action="store_true",
        help=(
            "After forward-backward and an optimizer step, save a resumable "
            "checkpoint, load it back into the training sub-job, and delete it."
        ),
    )
    parser.add_argument(
        "--memory-telemetry",
        action="store_true",
        help="Enable server-side training peak-memory reporting for the probe.",
    )
    parser.add_argument(
        "--debug-image-tag",
        help="Run the submitted job with a specific backend image tag.",
    )
    parser.add_argument(
        "--training-sp-size",
        type=int,
        help=(
            "Test a Ulysses sequence-parallel candidate and adjust the global "
            "batch to n_gpus / sp_size."
        ),
    )
    parser.add_argument(
        "--long-sequence",
        action="store_true",
        help="Select the profile's optional long-sequence recommendation.",
    )
    args = parser.parse_args()

    plans = []
    for profile_key in args.profile:
        request = build_profile_request(
            args.config_dir.resolve(),
            args.repo_root.resolve(),
            args.model_id,
            profile_key,
            long_sequence=args.long_sequence,
        )
        if args.memory_telemetry and profile_key in TRAINING_PROFILE_KEYS:
            enable_training_memory_telemetry(request)
        if args.debug_image_tag:
            apply_debug_image_tag(request, args.debug_image_tag)
        if args.training_sp_size and profile_key in TRAINING_PROFILE_KEYS:
            apply_training_sequence_parallelism(request, args.training_sp_size)
        plans.append(
            {
                "profileKey": profile_key,
                "request": request,
            }
        )

    if not args.submit:
        print(
            json.dumps(
                [
                    {
                        "profileKey": plan["profileKey"],
                        "request": plan["request"],
                    }
                    for plan in plans
                ],
                indent=2,
            )
        )
        return 0

    client = _build_client(args.poll_timeout)
    results = []
    for plan in plans:
        profile_key = plan["profileKey"]
        job_id: str | None = None
        active_error: BaseException | None = None
        try:
            response = client.create_job_from_body(plan["request"])
            job_id = response["job_id"]
            print(f"Submitted {profile_key} as job {job_id}", file=sys.stderr)
            job = client.wait_for_job(job_id)
            probes = run_data_plane_probes(
                client,
                job_id,
                profile_key,
                plan["request"],
                full_context_prefill=args.full_context_prefill,
                full_context_training=args.full_context_training,
                training_step=args.training_step,
                checkpoint_round_trip=args.checkpoint_round_trip,
            )
            results.append(
                {
                    "profileKey": profile_key,
                    "jobId": job_id,
                    "status": str(job.get("status") or "")
                    .lower()
                    .removeprefix("job_state_"),
                    "probes": probes,
                }
            )
        except BaseException as error:
            active_error = error
            raise
        finally:
            if job_id is not None:
                if _cancel_job(client, job_id, active_error):
                    print(f"Cancelled job {job_id}", file=sys.stderr)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
