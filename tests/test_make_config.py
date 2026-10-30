"""make_config: job configs from the model catalog, overrides, and router replay defaults."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from recipes import utils
from recipes.utils import make_config
from recipes.utils import make_distillation_config
from recipes.utils import merge_job_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / "model-catalog" / "models"
QWEN36 = "Qwen/Qwen3.6-35B-A3B"

# The fixed default router-replay caps make_config fills in for an MoE RL job.
FIXED_TRAINING_CACHE_BYTES = 5 * 1024**3  # 5368709120
FIXED_SAMPLING_CACHE_BYTES = 20 * 1024**3  # 21474836480


def _file(slug: str, key: str) -> dict:
    return json.loads((CATALOG / slug / f"{key}.json").read_text())


def _sub_job(body: dict, job_type: str) -> dict:
    (sub_job,) = [sub for sub in body["sub_job_configs"] if sub["job_type"] == job_type]
    return sub_job


def _training(body: dict) -> dict:
    return _sub_job(body, "training")["training_config"]


def _sampling(body: dict) -> dict:
    return _sub_job(body, "sampling")["inference_config"]


def _write(path: Path, value) -> Path:
    path.write_text(json.dumps(value))
    return path


# ─── Picking catalog files ───────────────────────────────────────────────────


def test_the_training_job_is_the_default():
    assert make_config("Qwen/Qwen3-8B", "standard") == _file("qwen3-8b", "standard/h200-training-full")


def test_lora_picks_the_lora_file():
    assert make_config("Qwen/Qwen3-8B", "standard", lora=True) == _file("qwen3-8b", "standard/h200-training-lora")


def test_inference_returns_the_inference_job():
    assert make_config("Qwen/Qwen3-8B", "standard", inference=True) == _file(
        "qwen3-8b", "standard/h200-inference-full"
    )


def test_hardware_defaults_to_h200():
    assert make_config("Qwen/Qwen3-8B", "standard", hardware="H200") == make_config("Qwen/Qwen3-8B", "standard")


def test_rl_combines_the_training_and_sampling_sub_jobs_unchanged():
    body = make_config("Qwen/Qwen3-8B", "standard", lora=True, rl=True)

    assert body == {
        "hardware": "H200",
        "sub_job_configs": [
            _file("qwen3-8b", "standard/h200-training-lora")["sub_job_configs"][0],
            _file("qwen3-8b", "standard/h200-inference-lora")["sub_job_configs"][0],
        ],
    }


def test_the_default_rl_job_is_four_training_and_two_sampling_gpus_with_sft_betas():
    body = make_config("Qwen/Qwen3-8B", "standard", lora=True, rl=True)

    assert _training(body)["n_gpus"] == 4
    assert _training(body)["optimizer"]["betas"] == [0.9, 0.999]
    assert _sampling(body)["n_gpus"] == 2
    assert _sampling(body)["vllm_config"]["tensor_parallel_size"] == 1
    assert "mb_spec" not in _training(body)
    assert "reduce_scatter" not in _training(body)["ds_config"]["zero_optimization"]


def test_a_missing_combination_names_the_files_that_exist():
    # Qwen3.5-9B's long tier has no LoRA files, so a long LoRA RL build cannot find
    # the long training file and names the files that do exist.
    with pytest.raises(FileNotFoundError) as raised:
        make_config("Qwen/Qwen3.5-9B", "long", lora=True, rl=True)

    assert str(raised.value) == (
        "Qwen/Qwen3.5-9B has no long/h200-training-lora.json in the model catalog; its files "
        "are: long/h200-inference-full.json, long/h200-training-full.json, standard/h200-inference-full.json, "
        "standard/h200-inference-lora.json, standard/h200-training-full.json, standard/h200-training-lora.json"
    )


def test_missing_hardware_names_the_files_that_exist():
    with pytest.raises(FileNotFoundError, match="has no standard/b200-training-full.json in the model catalog"):
        make_config("Qwen/Qwen3-8B", "standard", hardware="b200")


def test_an_unknown_model_lists_the_catalog():
    with pytest.raises(FileNotFoundError, match="'Org/Unknown' is not in the model catalog; it has: .*Qwen/Qwen3-8B"):
        make_config("Org/Unknown", "standard")


def test_the_model_is_the_exact_model_id():
    with pytest.raises(FileNotFoundError, match="did you mean 'Qwen/Qwen3-8B'"):
        make_config("qwen/qwen3-8b", "standard")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model": "Qwen/Qwen3-8B", "context": "medium"}, "context must be one of"),
        ({"model": "Qwen/Qwen3-8B", "context": "standard", "hardware": "h100"}, "hardware must be one of"),
        ({"model": "Qwen/Qwen3-8B", "context": "standard", "inference": True, "rl": True}, "not both"),
        ({"model": "Qwen/Qwen3-8B"}, "needs a model and a context"),
        ({"model": "Qwen/Qwen3-8B", "context": "standard", "training_config": "x.json"}, "not both"),
        ({"model": "Qwen/Qwen3-8B", "context": "standard", "router_replay": True}, "only apply with rl=True"),
        ({"model": "Qwen/Qwen3-8B", "context": "standard", "rollout_batch_size": 8}, "only apply with rl=True"),
        ({"model": "Qwen/Qwen3-8B", "context": "standard", "rl": True, "rollout_batch_size": 0}, "positive integer"),
        ({"training_config": "x.json", "rl": True}, "needs both training_config and inference_config"),
        ({"inference_config": "x.json"}, "add rl=True"),
    ],
)
def test_invalid_selections_are_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_config(**kwargs)


def test_overrides_must_be_a_list():
    with pytest.raises(TypeError, match="overrides must be a list"):
        make_config("Qwen/Qwen3-8B", "standard", overrides="override.json")


def test_each_call_returns_a_fresh_copy():
    first = make_config("Qwen/Qwen3-8B", "standard")
    _training(first)["n_gpus"] = 99

    assert _training(make_config("Qwen/Qwen3-8B", "standard"))["n_gpus"] == 4


# ─── Explicit config files ───────────────────────────────────────────────────


def test_an_explicit_training_config_is_used_as_is(tmp_path):
    body = _file("qwen3-8b", "standard/h200-training-full")
    path = _write(tmp_path / "train.json", body)

    assert make_config(training_config=path) == body


def test_an_explicit_inference_config_with_inference(tmp_path):
    body = _file("qwen3-8b", "standard/h200-inference-full")
    path = _write(tmp_path / "infer.json", body)

    assert make_config(inference_config=path, inference=True) == body


def test_explicit_rl_combines_the_files(tmp_path):
    training = _write(tmp_path / "train.json", _file("qwen3-8b", "standard/h200-training-full"))
    inference = _write(tmp_path / "infer.json", _file("qwen3-8b", "standard/h200-inference-full"))

    assert make_config(training_config=training, inference_config=inference, rl=True) == make_config(
        "Qwen/Qwen3-8B", "standard", rl=True
    )


def test_rl_sub_jobs_must_share_hardware(tmp_path):
    training = _write(tmp_path / "train.json", _file("qwen3-8b", "standard/h200-training-full"))
    inference_body = _file("qwen3-8b", "standard/h200-inference-full")
    inference_body["hardware"] = "B200"
    inference = _write(tmp_path / "infer.json", inference_body)

    with pytest.raises(ValueError, match="hardware is set once per job.*is H200 but .* is B200"):
        make_config(training_config=training, inference_config=inference, rl=True)


def test_rl_needs_one_sub_job_from_each_file(tmp_path):
    rl_body = make_config("Qwen/Qwen3-8B", "standard", rl=True)
    training = _write(tmp_path / "rl.json", rl_body)
    inference = _write(tmp_path / "infer.json", _file("qwen3-8b", "standard/h200-inference-full"))

    with pytest.raises(ValueError, match="must hold exactly one training sub-job"):
        make_config(training_config=training, inference_config=inference, rl=True)


def _qwen36_files(tmp_path):
    training = _write(tmp_path / "train.json", _file("qwen3.6-35b-a3b", "long/h200-training-lora"))
    inference = _write(tmp_path / "infer.json", _file("qwen3.6-35b-a3b", "long/h200-inference-lora"))
    return training, inference


def test_explicit_rl_fills_router_replay_only_when_asked(tmp_path):
    training, inference = _qwen36_files(tmp_path)

    plain = make_config(training_config=training, inference_config=inference, rl=True)
    replayed = make_config(training_config=training, inference_config=inference, rl=True, router_replay=True)

    assert "router_replay" not in _training(plain)
    assert "router_replay" not in _sampling(plain)
    assert replayed == make_config(QWEN36, "long", lora=True, rl=True)


def test_explicit_router_replay_reads_the_config_of_a_model_outside_the_catalog(tmp_path, monkeypatch):
    calls = []

    def resolve(model_name):
        calls.append(model_name)
        return utils.MoEGeometry(num_moe_layers=40, topk=8)

    monkeypatch.setattr(utils, "resolve_model_moe_geometry", resolve)
    training_body = _file("qwen3.6-35b-a3b", "long/h200-training-lora")
    inference_body = _file("qwen3.6-35b-a3b", "long/h200-inference-lora")
    for body in (training_body, inference_body):
        body["sub_job_configs"][0]["model_name"] = "Org/New-MoE"
    training = _write(tmp_path / "train.json", training_body)
    inference = _write(tmp_path / "infer.json", inference_body)

    body = make_config(training_config=training, inference_config=inference, rl=True, router_replay=True)

    assert calls == ["Org/New-MoE"]
    assert _training(body)["router_replay"]["max_cache_bytes"] == FIXED_TRAINING_CACHE_BYTES


def _router_replay_override(**replay) -> dict:
    """The same ``router_replay`` settings on both RL sub-jobs."""
    return {
        "sub_job_configs": [
            {"job_type": "training", "training_config": {"router_replay": dict(replay)}},
            {"job_type": "sampling", "inference_config": {"router_replay": dict(replay)}},
        ]
    }


DENSE_MODEL_ERROR = "router replay needs an MoE model; the catalog lists 'Qwen/Qwen3-8B' as 'dense'"


@pytest.mark.parametrize(
    "overrides", [[], [_router_replay_override(max_cache_bytes=1)]], ids=["planned caps", "both caps set"]
)
def test_router_replay_on_a_dense_catalog_model_is_an_error(tmp_path, overrides):
    training = _write(tmp_path / "train.json", _file("qwen3-8b", "standard/h200-training-full"))
    inference = _write(tmp_path / "infer.json", _file("qwen3-8b", "standard/h200-inference-full"))

    with pytest.raises(ValueError, match=DENSE_MODEL_ERROR):
        make_config(training_config=training, inference_config=inference, rl=True, router_replay=True, overrides=overrides)
    with pytest.raises(ValueError, match=DENSE_MODEL_ERROR):
        make_config("Qwen/Qwen3-8B", "standard", rl=True, router_replay=True, overrides=overrides)


@pytest.mark.parametrize("router_replay", [None, False])
def test_an_override_cannot_turn_router_replay_on_for_a_dense_catalog_model(router_replay):
    override = _router_replay_override(enabled=True, max_cache_bytes=1)

    with pytest.raises(ValueError, match=DENSE_MODEL_ERROR):
        make_config("Qwen/Qwen3-8B", "standard", rl=True, router_replay=router_replay, overrides=[override])


@pytest.mark.parametrize("router_replay", [None, True])
def test_router_replay_turned_off_on_a_dense_model_is_left_off(router_replay):
    body = make_config(
        "Qwen/Qwen3-8B", "standard", rl=True, router_replay=router_replay, overrides=[_router_replay_override(enabled=False)]
    )

    assert _training(body)["router_replay"] == _sampling(body)["router_replay"] == {"enabled": False}


def test_router_replay_enabled_on_one_sub_job_and_disabled_on_the_other_is_rejected():
    # enabled is per job; a one-side-true/other-false pair is contradictory. Either
    # order raises, so a job is never left half-on while the runner treats it as replaying.
    conflict = {
        "sub_job_configs": [
            {"job_type": "training", "training_config": {"router_replay": {"enabled": False}}},
            {"job_type": "sampling", "inference_config": {"router_replay": {"enabled": True}}},
        ]
    }
    with pytest.raises(ValueError, match="enabled on one sub-job and disabled on the other"):
        make_config(QWEN36, "long", rl=True, overrides=[conflict])

    flipped = {
        "sub_job_configs": [
            {"job_type": "training", "training_config": {"router_replay": {"enabled": True}}},
            {"job_type": "sampling", "inference_config": {"router_replay": {"enabled": False}}},
        ]
    }
    with pytest.raises(ValueError, match="enabled on one sub-job and disabled on the other"):
        make_config(QWEN36, "long", rl=True, overrides=[flipped])


def test_router_replay_disabled_on_one_sub_job_turns_it_off_for_the_job():
    # enabled:false on either sub-job (with no explicit true on the other) leaves
    # router replay off across the composed MoE job.
    off_one_side = {"sub_job_configs": [{"job_type": "training", "training_config": {"router_replay": {"enabled": False}}}]}
    body = make_config(QWEN36, "long", rl=True, overrides=[off_one_side])

    assert _training(body).get("router_replay") == {"enabled": False}
    assert "router_replay" not in _sampling(body)


def test_explicit_caps_on_an_moe_model_still_resolve_its_geometry(monkeypatch):
    calls = []
    model_geometry = utils._model_geometry

    def recording(model_name, catalog_dir):
        calls.append(model_name)
        return model_geometry(model_name, catalog_dir)

    monkeypatch.setattr(utils, "_model_geometry", recording)
    body = make_config(QWEN36, "long", rl=True, overrides=[_router_replay_override(max_cache_bytes=3)])

    assert calls == [QWEN36]
    assert _training(body)["router_replay"] == {"max_cache_bytes": 3, "enabled": True, "mode": "strict"}
    assert _sampling(body)["router_replay"] == {"max_cache_bytes": 3, "enabled": True}


def test_explicit_caps_on_a_model_outside_the_catalog_still_check_its_config(tmp_path, monkeypatch):
    def resolve(model_name):
        raise ValueError(f"Model config is missing MoE routing geometry ({model_name})")

    monkeypatch.setattr(utils, "resolve_model_moe_geometry", resolve)
    files = []
    for name, key in (("train.json", "standard/h200-training-full"), ("infer.json", "standard/h200-inference-full")):
        body = _file("qwen3-8b", key)
        body["sub_job_configs"][0]["model_name"] = "Org/New-Dense"
        files.append(_write(tmp_path / name, body))

    with pytest.raises(ValueError, match=r"missing MoE routing geometry \(Org/New-Dense\)"):
        make_config(
            training_config=files[0],
            inference_config=files[1],
            rl=True,
            router_replay=True,
            overrides=[_router_replay_override(max_cache_bytes=1)],
        )


# ─── Merge rules ─────────────────────────────────────────────────────────────


def test_dictionaries_merge_key_by_key():
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        overrides=[{"sub_job_configs": [{"job_type": "training", "training_config": {"optimizer": {"lr": 3e-5}}}]}],
    )

    assert _training(body)["optimizer"] == {**_training(make_config("Qwen/Qwen3-8B", "standard"))["optimizer"], "lr": 3e-5}


def test_null_removes_a_key():
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        overrides=[{"sub_job_configs": [{"job_type": "training", "training_config": {"activation_checkpointing": None}}]}],
    )

    assert "activation_checkpointing" not in _training(body)


def test_null_on_a_missing_key_adds_nothing():
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        overrides=[{"experiment_name": None, "sub_job_configs": [{"job_type": "training", "training_config": {"router_replay": {"enabled": False, "mode": None}}}]}],
    )

    assert "experiment_name" not in body
    assert _training(body)["router_replay"] == {"enabled": False}


def test_sub_jobs_are_matched_by_job_type():
    base = make_config("Qwen/Qwen3-8B", "standard", rl=True)
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        rl=True,
        overrides=[{"sub_job_configs": [{"job_type": "sampling", "inference_config": {"n_gpus": 4}}]}],
    )

    assert _sampling(body)["n_gpus"] == 4
    assert _training(body) == _training(base)
    assert [sub["job_type"] for sub in body["sub_job_configs"]] == ["training", "sampling"]


def test_the_plan_example_override_changes_one_sub_job():
    body = make_config(
        QWEN36,
        "long",
        rl=True,
        overrides=[{"sub_job_configs": [{"job_type": "training", "training_config": {"router_replay": {"max_cache_bytes": 8589934592}}}]}],
    )

    assert _training(body)["router_replay"] == {"max_cache_bytes": 8589934592, "enabled": True, "mode": "strict"}


def test_an_override_with_a_duplicate_job_type_is_an_error():
    with pytest.raises(ValueError, match="lists job_type 'training' twice"):
        make_config(
            "Qwen/Qwen3-8B",
            "standard",
            overrides=[{"sub_job_configs": [{"job_type": "training"}, {"job_type": "training"}]}],
        )


def test_an_override_that_matches_no_sub_job_is_an_error():
    with pytest.raises(ValueError, match=r"has job_type 'sampling', which matches no sub-job \(the job has \['training'\]\)"):
        make_config(
            "Qwen/Qwen3-8B",
            "standard",
            overrides=[{"sub_job_configs": [{"job_type": "sampling", "inference_config": {"n_gpus": 1}}]}],
        )


@pytest.mark.parametrize("entry", [{"training_config": {"n_gpus": 8}}, "training"])
def test_an_override_sub_job_needs_a_job_type(entry):
    with pytest.raises(ValueError, match=r"override sub_job_configs\[0\] needs a job_type"):
        merge_job_config(make_config("Qwen/Qwen3-8B", "standard"), {"sub_job_configs": [entry]})


def test_override_sub_job_configs_must_be_a_list():
    with pytest.raises(ValueError, match="must be a list of sub-jobs"):
        merge_job_config(make_config("Qwen/Qwen3-8B", "standard"), {"sub_job_configs": {"job_type": "training"}})


def test_other_lists_are_replaced_whole():
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        lora=True,
        overrides=[{"sub_job_configs": [{"job_type": "training", "training_config": {"peft_config": {"target_modules": ["q_proj"]}}}]}],
    )

    assert _training(body)["peft_config"]["target_modules"] == ["q_proj"]
    assert _training(body)["peft_config"]["r"] == 32


def test_merge_does_not_modify_its_arguments():
    base = make_config("Qwen/Qwen3-8B", "standard", rl=True)
    override = {"hardware": "B200", "sub_job_configs": [{"job_type": "training", "training_config": {"n_gpus": 8}}]}
    base_copy, override_copy = copy.deepcopy(base), copy.deepcopy(override)

    merged = merge_job_config(base, override)

    assert base == base_copy
    assert override == override_copy
    assert merged["hardware"] == "B200"
    assert _training(merged)["n_gpus"] == 8


def test_overrides_apply_in_order_from_files_and_dicts(tmp_path):
    first = _write(
        tmp_path / "first.json",
        {"sub_job_configs": [{"job_type": "training", "training_config": {"max_seq_len": 8192, "train_batch_size": 8}}]},
    )
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        overrides=[first, {"sub_job_configs": [{"job_type": "training", "training_config": {"max_seq_len": 4096}}]}],
    )

    assert (_training(body)["max_seq_len"], _training(body)["train_batch_size"]) == (4096, 8)


def test_override_file_problems_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="override file not found"):
        make_config("Qwen/Qwen3-8B", "standard", overrides=[tmp_path / "missing.json"])
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ValueError, match="is not valid JSON"):
        make_config("Qwen/Qwen3-8B", "standard", overrides=[tmp_path / "bad.json"])
    _write(tmp_path / "list.json", [1, 2])
    with pytest.raises(ValueError, match="must be a JSON object"):
        make_config("Qwen/Qwen3-8B", "standard", overrides=[tmp_path / "list.json"])


# ─── Router replay defaults ──────────────────────────────────────────────────


def test_the_fixed_default_caps_match_the_module_constants():
    assert FIXED_TRAINING_CACHE_BYTES == utils.DEFAULT_ROUTER_REPLAY_TRAINING_CACHE_BYTES == 5368709120
    assert FIXED_SAMPLING_CACHE_BYTES == utils.DEFAULT_ROUTER_REPLAY_SAMPLING_CACHE_BYTES == 21474836480


@pytest.mark.parametrize("lora", [True, False])
def test_moe_rl_fills_router_replay_on_both_sub_jobs_and_nothing_else(lora):
    body = make_config(QWEN36, "long", lora=lora, rl=True)

    assert _training(body).pop("router_replay") == {
        "enabled": True,
        "mode": "strict",
        "max_cache_bytes": FIXED_TRAINING_CACHE_BYTES,
    }
    assert _sampling(body).pop("router_replay") == {"enabled": True, "max_cache_bytes": FIXED_SAMPLING_CACHE_BYTES}
    assert body == make_config(QWEN36, "long", lora=lora, rl=True, router_replay=False)


def test_the_default_caps_are_fixed_not_sized_by_the_rollout_batch():
    # The caps are the fixed 5 GiB / 20 GiB defaults regardless of rollout_batch_size,
    # which is accepted only for backward compatibility.
    body = make_config(QWEN36, "long", lora=True, rl=True, rollout_batch_size=1024)

    assert _training(body)["router_replay"]["max_cache_bytes"] == FIXED_TRAINING_CACHE_BYTES
    assert _sampling(body)["router_replay"]["max_cache_bytes"] == FIXED_SAMPLING_CACHE_BYTES
    assert body == make_config(QWEN36, "long", lora=True, rl=True)
    assert body == make_config(QWEN36, "long", lora=True, rl=True, rollout_batch_size=1)


def test_the_fixed_caps_do_not_change_with_gpu_counts_or_sequence_length():
    body = make_config(
        QWEN36,
        "long",
        rl=True,
        rollout_batch_size=64,
        overrides=[
            {
                "sub_job_configs": [
                    {"job_type": "training", "training_config": {"n_gpus": 16, "max_seq_len": 32768}},
                    {"job_type": "sampling", "inference_config": {"max_seq_len": 32768, "vllm_config": {"tensor_parallel_size": 2}}},
                ]
            }
        ],
    )

    assert _training(body)["router_replay"]["max_cache_bytes"] == FIXED_TRAINING_CACHE_BYTES
    assert _sampling(body)["router_replay"]["max_cache_bytes"] == FIXED_SAMPLING_CACHE_BYTES


def test_explicit_max_cache_bytes_and_mode_win():
    body = make_config(
        QWEN36,
        "long",
        rl=True,
        overrides=[
            {
                "sub_job_configs": [
                    {
                        "job_type": "training",
                        "training_config": {"router_replay": {"mode": "best_effort", "max_cache_bytes": 2147483648}},
                    },
                    {"job_type": "sampling", "inference_config": {"router_replay": {"max_cache_bytes": 17179869184}}},
                ]
            }
        ],
    )

    assert _training(body)["router_replay"] == {"mode": "best_effort", "max_cache_bytes": 2147483648, "enabled": True}
    assert _sampling(body)["router_replay"] == {"max_cache_bytes": 17179869184, "enabled": True}


def test_rl_rejects_a_sampling_window_above_the_training_window():
    # The runner collates at the training window, so a larger sampling window
    # overflows it; reject before job creation (override path).
    override = {
        "sub_job_configs": [
            {"job_type": "training", "training_config": {"max_seq_len": 512}},
            {"job_type": "sampling", "inference_config": {"max_seq_len": 1024}},
        ]
    }
    with pytest.raises(ValueError, match="sampling max_seq_len 1024 exceeds its training max_seq_len 512"):
        make_config("Qwen/Qwen3-8B", "standard", rl=True, overrides=[override])


def test_rl_allows_a_sampling_window_at_or_below_the_training_window():
    # sampling < training is safe -- the sampler just caps shorter -- and equal is the default.
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        rl=True,
        overrides=[
            {
                "sub_job_configs": [
                    {"job_type": "training", "training_config": {"max_seq_len": 1024}},
                    {"job_type": "sampling", "inference_config": {"max_seq_len": 512}},
                ]
            }
        ],
    )
    assert _training(body)["max_seq_len"] == 1024
    assert _sampling(body)["max_seq_len"] == 512
    # The catalog default pairs equal windows and composes fine.
    assert make_config("Qwen/Qwen3-8B", "standard", rl=True)["sub_job_configs"]


def test_a_sampling_side_router_replay_mode_is_rejected():
    # The service rejects a non-strict sampling mode, so an override that sets one
    # is rejected here rather than passing a dry-run and failing on real submit.
    with pytest.raises(ValueError, match="sampling sub-job's router_replay must not set a mode"):
        make_config(
            QWEN36,
            "long",
            rl=True,
            overrides=[{"sub_job_configs": [{"job_type": "sampling", "inference_config": {"router_replay": {"mode": "best_effort"}}}]}],
        )


def test_a_training_side_router_replay_mode_is_honored():
    body = make_config(
        QWEN36,
        "long",
        rl=True,
        overrides=[{"sub_job_configs": [{"job_type": "training", "training_config": {"router_replay": {"mode": "best_effort"}}}]}],
    )

    assert _training(body)["router_replay"]["mode"] == "best_effort"
    assert "mode" not in _sampling(body)["router_replay"]


def test_an_override_comment_key_is_stripped_before_merging():
    # A reserved "_comment" is dropped at any depth; the real fields still apply.
    body = make_config(
        "Qwen/Qwen3-8B",
        "standard",
        overrides=[
            {
                "_comment": "a top-level note",
                "sub_job_configs": [
                    {"job_type": "training", "_comment": ["a", "b"], "training_config": {"_comment": "deep", "n_gpus": 3}}
                ],
            }
        ],
    )

    assert _training(body)["n_gpus"] == 3
    assert "_comment" not in json.dumps(body)


def test_a_comment_only_override_is_a_no_op():
    base = make_config("Qwen/Qwen3-8B", "standard")

    assert make_config("Qwen/Qwen3-8B", "standard", overrides=[{"_comment": "just a note"}]) == base


@pytest.mark.parametrize(("job_type", "config_key"), [("training", "training_config"), ("sampling", "inference_config")])
def test_enabled_false_in_an_override_turns_router_replay_off(job_type, config_key):
    body = make_config(
        QWEN36,
        "long",
        rl=True,
        overrides=[{"sub_job_configs": [{"job_type": job_type, config_key: {"router_replay": {"enabled": False}}}]}],
    )

    assert _sub_job(body, job_type)[config_key]["router_replay"] == {"enabled": False}
    other = _sampling(body) if job_type == "training" else _training(body)
    assert "router_replay" not in other


def test_dense_rl_gets_no_router_replay():
    body = make_config("Qwen/Qwen3-8B", "standard", rl=True)

    assert "router_replay" not in _training(body)
    assert "router_replay" not in _sampling(body)


@pytest.mark.parametrize("kwargs", [{}, {"inference": True}])
def test_only_rl_jobs_get_router_replay(kwargs):
    body = make_config(QWEN36, "long", **kwargs)

    assert "router_replay" not in json.dumps(body)


def test_router_replay_leaves_vllm_gpu_memory_utilization_alone():
    body = make_config(QWEN36, "long", rl=True, rollout_batch_size=1024)

    assert _sampling(body)["vllm_config"] == _file("qwen3.6-35b-a3b", "long/h200-inference-full")["sub_job_configs"][0][
        "inference_config"
    ]["vllm_config"]
    assert (utils.DEFAULT_ROUTING_SKEW, utils.DEFAULT_SAFETY_FACTOR) == (2.0, 1.2)


def test_router_replay_must_be_an_object():
    with pytest.raises(ValueError, match="router_replay must be an object"):
        make_config(
            QWEN36,
            "long",
            rl=True,
            overrides=[{"sub_job_configs": [{"job_type": "training", "training_config": {"router_replay": True}}]}],
        )


# ─── Command line ────────────────────────────────────────────────────────────


def test_the_command_line_writes_the_job(tmp_path):
    out = tmp_path / "nested" / "job.json"

    assert utils.main(["--model", "Qwen/Qwen3-8B", "--context", "standard", "--lora", "-o", str(out)]) == 0
    assert json.loads(out.read_text()) == make_config("Qwen/Qwen3-8B", "standard", lora=True)


def test_the_command_line_builds_rl_jobs_with_overrides(tmp_path):
    override = _write(tmp_path / "o.json", {"sub_job_configs": [{"job_type": "training", "training_config": {"optimizer": {"lr": 1e-6}}}]})
    out = tmp_path / "rl.json"

    status = utils.main(
        ["--model", QWEN36, "--context", "long", "--hardware", "h200", "--rl", "--rollout-batch-size", "1024", "--override", str(override), "-o", str(out)]
    )

    assert status == 0
    assert json.loads(out.read_text()) == make_config(QWEN36, "long", rl=True, rollout_batch_size=1024, overrides=[override])


def test_the_command_line_takes_explicit_files(tmp_path):
    training, inference = _qwen36_files(tmp_path)
    out = tmp_path / "rl.json"

    status = utils.main(
        ["--training-config", str(training), "--inference-config", str(inference), "--rl", "--router-replay", "-o", str(out)]
    )

    assert status == 0
    assert json.loads(out.read_text()) == make_config(QWEN36, "long", lora=True, rl=True)


def test_the_command_line_writes_stdout_for_a_dash(capsys):
    assert utils.main(["--model", "Qwen/Qwen3-8B", "--context", "standard", "--inference", "-o", "-"]) == 0
    assert json.loads(capsys.readouterr().out) == make_config("Qwen/Qwen3-8B", "standard", inference=True)


def test_the_command_line_reports_a_missing_combination(tmp_path, capsys):
    # Qwen3.5-9B's long tier has no LoRA files, so a long LoRA RL build has no file.
    status = utils.main(
        ["--model", "Qwen/Qwen3.5-9B", "--context", "long", "--lora", "--rl", "-o", str(tmp_path / "x.json")]
    )

    assert status == 1
    assert "error: Qwen/Qwen3.5-9B has no long/h200-training-lora.json" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "Qwen/Qwen3-8B", "--context", "standard", "--training-config", "t.json"],
        ["--model", "Qwen/Qwen3-8B"],
        ["--context", "standard"],
        ["--training-config", "t.json", "--lora"],
        ["--training-config", "t.json", "--inference-config", "i.json"],
        ["--training-config", "t.json", "--router-replay"],
        ["--model", "Qwen/Qwen3-8B", "--context", "standard", "--rollout-batch-size", "8"],
        ["--model", "Qwen/Qwen3-8B", "--context", "standard", "--inference", "--rl"],
    ],
)
def test_the_command_line_rejects_mixed_forms(argv, capsys):
    with pytest.raises(SystemExit) as raised:
        utils.main(argv + ["-o", "out.json"])

    assert raised.value.code == 2


def test_the_module_runs_from_the_command_line(tmp_path):
    out = tmp_path / "job.json"
    result = subprocess.run(
        [sys.executable, "-m", "recipes.utils", "--model", "Qwen/Qwen3-8B", "--context", "standard", "--rl", "-o", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text()) == make_config("Qwen/Qwen3-8B", "standard", rl=True)


# ─── The final job decides ───────────────────────────────────────────────────


def _moe_catalog(tmp_path: Path) -> Path:
    """A catalog with two small MoE models and one dense model, standard H200 files only."""
    catalog = tmp_path / "models"
    models = {
        "org/Model-A": {"num_moe_layers": 1, "topk": 1, "num_experts": 4},
        "org/Model-B": {"num_moe_layers": 10, "topk": 2, "num_experts": 8},
        "org/Dense-C": None,
    }
    for model_id, moe in models.items():
        folder = catalog / model_id.rsplit("/", 1)[-1].lower()
        (folder / "standard").mkdir(parents=True)
        info = {"model_id": model_id, "license": {"name": "MIT", "url": "https://example.com/LICENSE"}}
        info["architecture"] = "moe" if moe else "dense"
        if moe:
            info["moe"] = moe
        _write(folder / "model.json", info)
        _write(
            folder / "standard" / "h200-training-full.json",
            {"hardware": "H200", "sub_job_configs": [{"job_type": "training", "model_name": model_id, "training_config": {
                "n_gpus": 2, "max_seq_len": 10, "train_batch_size": 2, "optimizer": {"lr": 1e-5}}}]},
        )
        _write(
            folder / "standard" / "h200-inference-full.json",
            {"hardware": "H200", "sub_job_configs": [{"job_type": "sampling", "model_name": model_id, "inference_config": {
                "n_gpus": 2, "max_seq_len": 10, "vllm_config": {"tensor_parallel_size": 1}}}]},
        )
    return catalog


def _rename(model_id: str) -> dict:
    return {"sub_job_configs": [{"job_type": "training", "model_name": model_id}, {"job_type": "sampling", "model_name": model_id}]}


def _replay_caps(body: dict) -> tuple[int, int]:
    return _training(body)["router_replay"]["max_cache_bytes"], _sampling(body)["router_replay"]["max_cache_bytes"]


def test_router_replay_is_filled_for_the_moe_model_the_final_job_names(tmp_path):
    catalog = _moe_catalog(tmp_path)

    body = make_config("org/Model-A", "standard", rl=True, overrides=[_rename("org/Model-B")], catalog_dir=catalog)

    # The rename picks an MoE model, so router replay is filled with the fixed caps.
    assert {sub["model_name"] for sub in body["sub_job_configs"]} == {"org/Model-B"}
    assert _replay_caps(body) == (FIXED_TRAINING_CACHE_BYTES, FIXED_SAMPLING_CACHE_BYTES)


def test_explicit_router_replay_is_filled_for_the_final_moe_model(tmp_path):
    catalog = _moe_catalog(tmp_path)
    training = catalog / "model-a/standard/h200-training-full.json"
    inference = catalog / "model-a/standard/h200-inference-full.json"

    body = make_config(
        training_config=training,
        inference_config=inference,
        rl=True,
        router_replay=True,
        overrides=[_rename("org/Model-B")],
        catalog_dir=catalog,
    )

    assert {sub["model_name"] for sub in body["sub_job_configs"]} == {"org/Model-B"}
    assert _replay_caps(body) == (FIXED_TRAINING_CACHE_BYTES, FIXED_SAMPLING_CACHE_BYTES)


def test_an_override_to_a_dense_model_gets_no_router_replay(tmp_path):
    catalog = _moe_catalog(tmp_path)

    body = make_config("org/Model-A", "standard", rl=True, overrides=[_rename("org/Dense-C")], catalog_dir=catalog)

    assert "router_replay" not in json.dumps(body)


def test_rl_sub_jobs_must_name_the_same_model(tmp_path):
    catalog = _moe_catalog(tmp_path)
    override = {"sub_job_configs": [{"job_type": "sampling", "model_name": "org/Model-B"}]}

    with pytest.raises(ValueError, match="training and sampling sub-jobs must name the same model; the job has "
                                         "training 'org/Model-A' and sampling 'org/Model-B'"):
        make_config("org/Model-A", "standard", rl=True, overrides=[override], catalog_dir=catalog)


def test_an_override_to_a_model_outside_the_catalog_needs_router_replay_asked_for(tmp_path, monkeypatch):
    catalog = _moe_catalog(tmp_path)
    monkeypatch.setattr(utils, "resolve_model_moe_geometry", lambda model_name: utils.MoEGeometry(num_moe_layers=3, topk=2))

    plain = make_config("org/Model-A", "standard", rl=True, overrides=[_rename("org/New")], catalog_dir=catalog)
    planned = make_config("org/Model-A", "standard", rl=True, router_replay=True, overrides=[_rename("org/New")], catalog_dir=catalog)

    assert "router_replay" not in json.dumps(plain)
    assert _training(planned)["router_replay"]["max_cache_bytes"] == FIXED_TRAINING_CACHE_BYTES
    assert _sampling(planned)["router_replay"]["max_cache_bytes"] == FIXED_SAMPLING_CACHE_BYTES


# ─── make_distillation_config: the student and teacher jobs ──────────────────

STUDENT = "Qwen/Qwen3-0.6B"
TEACHER = "Qwen/Qwen3-8B"


def test_make_distillation_config_composes_the_student_and_a_sampling_only_teacher():
    student, teacher = make_distillation_config(
        STUDENT, TEACHER, "standard", student_max_seq_len=2048, teacher_max_seq_len=4096
    )

    # The student is the catalog's training + inference files combined, like an RL job.
    assert {sub["job_type"] for sub in student["sub_job_configs"]} == {"training", "sampling"}
    assert {sub["model_name"] for sub in student["sub_job_configs"]} == {STUDENT}
    # The teacher is the teacher's inference file alone, sampling-only, full weights.
    assert [sub["job_type"] for sub in teacher["sub_job_configs"]] == ["sampling"]
    assert teacher["sub_job_configs"][0]["model_name"] == TEACHER
    assert "peft_config" not in teacher["sub_job_configs"][0]["inference_config"]
    # On-policy distillation replays no routing, even though the student reuses the
    # RL composition path.
    assert "router_replay" not in json.dumps(student)
    # The sizing flags set both student sub-jobs and the teacher's window.
    assert _training(student)["max_seq_len"] == 2048
    assert _sampling(student)["max_seq_len"] == 2048
    assert teacher["sub_job_configs"][0]["inference_config"]["max_seq_len"] == 4096


def test_make_distillation_config_applies_student_and_teacher_overrides():
    student, teacher = make_distillation_config(
        STUDENT,
        TEACHER,
        "standard",
        student_max_seq_len=2048,
        teacher_max_seq_len=4096,
        overrides=[{"sub_job_configs": [{"job_type": "training", "training_config": {"n_gpus": 1}}]}],
        teacher_overrides=[{"sub_job_configs": [{"job_type": "sampling", "inference_config": {"n_gpus": 1}}]}],
    )

    assert _training(student)["n_gpus"] == 1
    assert teacher["sub_job_configs"][0]["inference_config"]["n_gpus"] == 1


def test_make_distillation_config_rejects_divergent_student_windows():
    # Overrides that set the student's training and sampling windows apart are
    # rejected: the student trains and samples the same model at one window.
    # (A larger sampling window is caught earlier by the RL window check; this
    # smaller-sampling case is allowed there but still rejected by the distillation
    # equality rule, so it exercises that rule specifically.)
    with pytest.raises(ValueError, match="training max_seq_len .* and sampling max_seq_len .* must match"):
        make_distillation_config(
            STUDENT,
            TEACHER,
            "standard",
            overrides=[
                {
                    "sub_job_configs": [
                        {"job_type": "training", "training_config": {"max_seq_len": 4096}},
                        {"job_type": "sampling", "inference_config": {"max_seq_len": 2048}},
                    ]
                }
            ],
        )


@pytest.mark.parametrize("student", ["Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.5-4B"])
def test_make_distillation_config_rejects_a_non_default_student_that_cannot_run_on_one_gpu(student):
    # The shipped single-GPU student footprint (n_gpus=1) keeps a non-default
    # student's catalog sp_size/ep_size above 1, which cannot run on one GPU. The
    # builder rejects it instead of emitting a body a dry-run accepts but no run can.
    one_gpu = {"sub_job_configs": [{"job_type": "training", "training_config": {"n_gpus": 1}}]}
    with pytest.raises(ValueError, match="parallelism does not fit its GPU count"):
        make_distillation_config(student, TEACHER, "long", overrides=[one_gpu])


def test_parallelism_errors_flags_sizes_that_do_not_divide_n_gpus():
    assert utils.parallelism_errors({"n_gpus": 1, "sp_size": 8}) == ["sp_size 8 must be a positive divisor of n_gpus 1"]
    assert utils.parallelism_errors({"n_gpus": 8, "sp_size": 8, "ep_size": 4}) == []
    assert utils.parallelism_errors({"n_gpus": 1}) == []


def test_make_distillation_config_enforces_the_teacher_window_above_the_student():
    with pytest.raises(ValueError, match="teacher max_seq_len 2048 must exceed the student's 4096"):
        make_distillation_config(
            STUDENT, TEACHER, "standard", student_max_seq_len=4096, teacher_max_seq_len=2048
        )


def test_make_distillation_config_rejects_a_single_override_not_in_a_list():
    with pytest.raises(TypeError, match="overrides must be a list"):
        make_distillation_config(STUDENT, TEACHER, "standard", overrides={"sub_job_configs": []})
    with pytest.raises(TypeError, match="teacher_overrides must be a list"):
        make_distillation_config(STUDENT, TEACHER, "standard", teacher_overrides={"sub_job_configs": []})


def test_the_command_line_writes_the_distillation_jobs(tmp_path):
    out = tmp_path / "dist.json"

    status = utils.main(
        [
            "--distillation",
            "--student", STUDENT,
            "--teacher", TEACHER,
            "--context", "standard",
            "--student-max-seq-len", "2048",
            "--teacher-max-seq-len", "4096",
            "-o", str(out),
        ]
    )

    assert status == 0
    student, teacher = make_distillation_config(
        STUDENT, TEACHER, "standard", student_max_seq_len=2048, teacher_max_seq_len=4096
    )
    assert json.loads(out.read_text()) == student
    assert json.loads((tmp_path / "dist.teacher.json").read_text()) == teacher


def test_the_command_line_writes_the_distillation_jobs_to_stdout(capsys):
    status = utils.main(
        ["--distillation", "--student", STUDENT, "--teacher", TEACHER, "--context", "standard",
         "--student-max-seq-len", "2048", "--teacher-max-seq-len", "4096", "-o", "-"]
    )

    assert status == 0
    written = json.loads(capsys.readouterr().out)
    assert set(written) == {"student", "teacher"}
    assert [sub["job_type"] for sub in written["teacher"]["sub_job_configs"]] == ["sampling"]


def test_the_command_line_reports_a_teacher_window_not_above_the_student(tmp_path, capsys):
    status = utils.main(
        ["--distillation", "--student", STUDENT, "--teacher", TEACHER, "--context", "standard",
         "--student-max-seq-len", "4096", "--teacher-max-seq-len", "2048", "-o", str(tmp_path / "x.json")]
    )

    assert status == 1
    assert "must exceed the student's" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["--distillation", "--teacher", TEACHER, "--context", "standard"],
        ["--distillation", "--student", STUDENT, "--context", "standard"],
        ["--distillation", "--student", STUDENT, "--teacher", TEACHER],
        ["--distillation", "--student", STUDENT, "--teacher", TEACHER, "--context", "standard", "--model", "Qwen/Qwen3-8B"],
        ["--student", STUDENT, "--teacher", TEACHER, "--context", "standard"],
        ["--model", "Qwen/Qwen3-8B", "--context", "standard", "--teacher", TEACHER],
        # A misplaced sizing flag is detected by presence, not truthiness: 0 still errors.
        ["--model", "Qwen/Qwen3-8B", "--context", "standard", "--student-max-seq-len", "0"],
        ["--model", "Qwen/Qwen3-8B", "--context", "standard", "--teacher-max-seq-len", "0"],
    ],
)
def test_the_command_line_rejects_bad_distillation_forms(argv, capsys):
    with pytest.raises(SystemExit) as raised:
        utils.main(argv + ["-o", "out.json"])

    assert raised.value.code == 2
