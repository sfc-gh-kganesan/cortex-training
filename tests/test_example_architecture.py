from __future__ import annotations

import ast
import fnmatch
import importlib
import json
import os
import re
import shlex
from pathlib import Path
from pathlib import PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "recipes"
RL_OVERRIDES = EXAMPLES / "rl_quickstart" / "overrides.json"
SFT_OVERRIDES = EXAMPLES / "sft_quickstart" / "overrides.json"
DARE_OVERRIDES = EXAMPLES / "sft_quickstart" / "dare" / "overrides.json"

# Each runnable recipe and the folder that holds its entry point, task,
# metadata and data.
EXAMPLE_FOLDERS = (
    "sft_quickstart/conversational",
    "sft_quickstart/dare",
    "rl_quickstart/math_grpo",
    "inference",
    "model_deployment",
    "distillation",
    "dflash2",
    "skyrl/gsm8k_grpo",
)


def _import_example(module_name: str):
    for dependency in ("chz", "datasets", "tinker_cookbook"):
        pytest.importorskip(dependency)
    return importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("runner_name", "task_name", "task_type", "task_module"),
    [
        (
            "recipes.sft_quickstart.runner",
            "recipes.sft_quickstart.conversational.train",
            "ConversationalTask",
            "recipes.sft_quickstart.conversational.task",
        ),
        (
            "recipes.rl_quickstart.runner",
            "recipes.rl_quickstart.math_grpo.train",
            "MathTask",
            "recipes.rl_quickstart.math_grpo.task",
        ),
    ],
)
def test_task_entrypoints_call_shared_runner(runner_name, task_name, task_type, task_module):
    runner = _import_example(runner_name)
    task_entrypoint = _import_example(task_name)

    assert task_entrypoint.train is runner.train
    assert task_entrypoint.Config.__module__ == task_name
    assert getattr(task_entrypoint, task_type).__module__ == task_module


@pytest.mark.parametrize(
    "module_name",
    ["recipes.sft_quickstart.conversational.train", "recipes.rl_quickstart.math_grpo.train"],
)
def test_entrypoints_do_not_re_export_helpers(module_name):
    module = _import_example(module_name)

    assert not hasattr(module, "__all__")


def test_helpers_shared_across_modules_are_public():
    conversational = _import_example("recipes.sft_quickstart.conversational.task")
    math_task = _import_example("recipes.rl_quickstart.math_grpo.task")
    rl_runner = _import_example("recipes.rl_quickstart.runner")

    assert callable(conversational.is_local_chat_file)
    assert callable(math_task.stopped_cleanly)
    assert callable(rl_runner.should_eval)
    assert not hasattr(rl_runner, "_train")


def _private_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("examples.")
        for alias in node.names
        if alias.name.startswith("_")
    ]


@pytest.mark.parametrize("package", ["sft_quickstart", "rl_quickstart", "inference", "distillation", "dflash2"])
def test_private_names_do_not_cross_module_boundaries(package):
    offenders = {
        str(path.relative_to(REPO_ROOT)): names
        for path in sorted((EXAMPLES / package).rglob("*.py"))
        if (names := _private_imports(path))
    }

    assert offenders == {}


def _sub_job(body: dict, job_type: str) -> dict:
    (sub_job,) = [sub for sub in body["sub_job_configs"] if sub["job_type"] == job_type]
    return sub_job


def _job_body(module_name: str, **settings):
    module = _import_example(module_name)
    runner = _import_example(module.train.__module__)
    return runner.job_body(module.Config(config="unused", **settings))


def test_examples_build_their_default_jobs_from_the_catalog():
    utils = _import_example("recipes.utils")
    endpoint = _import_example("recipes.inference.endpoint")
    serve = _import_example("recipes.inference.serve")
    defaults = serve.Config(config="unused")
    inference_body, source = endpoint.inference_endpoint_body(
        model=defaults.model,
        context=defaults.context,
        hardware=defaults.hardware,
        lora=defaults.lora,
    )

    assert _job_body("recipes.sft_quickstart.conversational.train") == utils.make_config("Qwen/Qwen3-8B", "standard")
    # The RL example applies its committed rl/overrides.json on top of the catalog.
    assert _job_body("recipes.rl_quickstart.math_grpo.train") == utils.make_config(
        "Qwen/Qwen3-8B", "standard", lora=True, rl=True, overrides=[RL_OVERRIDES]
    )
    assert inference_body == utils.make_config("Qwen/Qwen3-8B", "standard", inference=True)
    assert source is None


@pytest.mark.parametrize(
    "module_name",
    [
        "recipes.inference.serve",
        "recipes.inference.generate",
        "recipes.inference.evaluate",
        "recipes.sft_quickstart.conversational.train",
        "recipes.rl_quickstart.math_grpo.train",
    ],
)
def test_examples_pick_their_job_by_model_context_hardware_and_lora(module_name):
    config = _import_example(module_name).Config(config="unused")

    assert (config.model, config.context, config.hardware) == ("Qwen/Qwen3-8B", "standard", "h200")
    assert config.lora is (module_name == "recipes.rl_quickstart.math_grpo.train")
    assert config.job_config is None
    assert config.max_seq_len is None


@pytest.mark.parametrize("module_name", ["recipes.sft_quickstart.conversational.train", "recipes.rl_quickstart.math_grpo.train"])
def test_training_settings_default_to_the_catalog_values(module_name):
    config = _import_example(module_name).Config(config="unused")

    assert (config.lr, config.train_batch_size, config.max_seq_len) == (None, None, None)


def test_command_line_settings_override_the_catalog():
    body = _job_body("recipes.sft_quickstart.conversational.train", lr=3e-5, train_batch_size=8, max_seq_len=4096)
    training = _sub_job(body, "training")["training_config"]

    assert (training["optimizer"]["lr"], training["train_batch_size"], training["max_seq_len"]) == (3e-5, 8, 4096)
    assert training["n_gpus"] == 4


def test_rl_settings_reach_both_sub_jobs():
    body = _job_body("recipes.rl_quickstart.math_grpo.train", max_seq_len=8192, lr=1e-6)

    assert _sub_job(body, "training")["training_config"]["max_seq_len"] == 8192
    assert _sub_job(body, "training")["training_config"]["optimizer"]["lr"] == 1e-6
    assert _sub_job(body, "sampling")["inference_config"]["max_seq_len"] == 8192


def test_moe_rl_example_fills_router_replay_with_the_fixed_default_caps():
    utils = _import_example("recipes.utils")
    body = _job_body(
        "recipes.rl_quickstart.math_grpo.train",
        model="Qwen/Qwen3.6-35B-A3B",
        context="long",
        problems_per_batch=4,
        group_size=8,
    )

    # The built job is the catalog plus rl/overrides.json; the router-replay caps
    # are the fixed defaults and do not depend on the rollout batch.
    assert body == utils.make_config("Qwen/Qwen3.6-35B-A3B", "long", lora=True, rl=True, overrides=[RL_OVERRIDES])
    replay = _sub_job(body, "training")["training_config"]["router_replay"]
    assert replay["mode"] == "strict"
    assert replay["max_cache_bytes"] == utils.DEFAULT_ROUTER_REPLAY_TRAINING_CACHE_BYTES


def _sampling_window_exceeds_training(utils, model: str, context: str, hardware: str, lora: bool) -> bool:
    """Whether this tier's inference file samples past its training window.

    ``make_config(..., rl=True)`` rejects that pair. The catalog keeps both files
    and builds no RL job, matching ``validate_catalog``.
    """
    folder = REPO_ROOT / "model-catalog" / "models" / utils.model_slug(model) / context

    def window(kind: str):
        path = folder / utils.catalog_file_name(hardware, kind, lora)
        if not path.is_file():
            return None
        sub_job = json.loads(path.read_text())["sub_job_configs"][0]
        block = sub_job.get("training_config") or sub_job.get("inference_config") or {}
        return block.get("max_seq_len")

    training, sampling = window("training"), window("inference")
    return isinstance(training, int) and isinstance(sampling, int) and sampling > training


def test_rl_overrides_restore_betas_packing_and_reduce_scatter_for_every_model():
    # rl_quickstart/overrides.json restores the three RL settings shared by the pre-restructure
    # configs, for the default RL job and every model's RL job.
    utils = _import_example("recipes.utils")
    rl_dir = EXAMPLES / "rl_quickstart"
    checked = 0
    for model, context, hardware, lora in _catalog_selections("training"):
        folder = REPO_ROOT / "model-catalog" / "models" / utils.model_slug(model) / context
        if not (folder / utils.catalog_file_name(hardware, "inference", lora)).is_file():
            continue  # not an RL-capable selection (no matching inference file)
        if _sampling_window_exceeds_training(utils, model, context, hardware, lora):
            continue
        body = utils.build_job("rl", model=model, context=context, hardware=hardware, lora=lora, example_dir=rl_dir)
        training = _sub_job(body, "training")["training_config"]
        where = (model, context, hardware, lora)
        assert training["optimizer"]["betas"] == [0.9, 0.95], where
        assert training["mb_spec"]["max_tokens_per_mb"] == 10240, where
        assert training["ds_config"]["zero_optimization"]["reduce_scatter"] is True, where
        checked += 1
    assert checked > 0


def test_job_config_skips_the_catalog_and_the_overrides_file(tmp_path, monkeypatch):
    utils = _import_example("recipes.utils")
    custom = utils.make_config("Qwen/Qwen3-8B", "standard", lora=True)
    custom["sub_job_configs"][0]["training_config"]["n_gpus"] = 8
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(custom))
    (tmp_path / "overrides.json").write_text(json.dumps({"hardware": "B300"}))
    monkeypatch.chdir(tmp_path)

    body = utils.build_job(
        "training",
        model="Qwen/Qwen3.6-35B-A3B",
        context="long",
        job_config="custom.json",
        example_dir=tmp_path,
        lr=2e-4,
    )

    assert body["hardware"] == "H200"
    assert _sub_job(body, "training")["model_name"] == "Qwen/Qwen3-8B"
    assert _sub_job(body, "training")["training_config"]["n_gpus"] == 8
    assert _sub_job(body, "training")["training_config"]["optimizer"]["lr"] == 2e-4


@pytest.mark.parametrize("kind", ["training", "inference"])
def test_an_rl_job_config_is_rejected_for_a_single_sub_job_mode(tmp_path, kind):
    utils = _import_example("recipes.utils")
    rl_body = utils.make_config("Qwen/Qwen3-8B", "standard", rl=True)
    path = tmp_path / "rl.json"
    path.write_text(json.dumps(rl_body))

    # An RL body (training + sampling) does not satisfy a training-only or
    # inference-only example, which would otherwise create an unused worker.
    with pytest.raises(ValueError, match=rf"a {kind} job needs exactly"):
        utils.build_job(kind, model="Qwen/Qwen3-8B", context="standard", job_config=str(path))


def test_an_rl_job_config_with_mismatched_models_is_rejected(tmp_path):
    utils = _import_example("recipes.utils")
    body = utils.make_config("Qwen/Qwen3-8B", "standard", rl=True)
    for sub in body["sub_job_configs"]:
        if sub["job_type"] == "sampling":
            sub["model_name"] = "Qwen/Qwen3-0.6B"
    path = tmp_path / "rl.json"
    path.write_text(json.dumps(body))

    # A supplied RL body pairing different training/sampling models is rejected,
    # not accepted and then broken at run time.
    with pytest.raises(ValueError, match="must name the same model"):
        utils.build_job("rl", model="Qwen/Qwen3-8B", context="standard", job_config=str(path))


def test_an_rl_job_config_with_an_oversized_sampling_window_is_rejected(tmp_path):
    utils = _import_example("recipes.utils")
    body = utils.make_config("Qwen/Qwen3-8B", "standard", rl=True)
    for sub in body["sub_job_configs"]:
        config = sub.get("training_config") or sub.get("inference_config")
        config["max_seq_len"] = 512 if sub["job_type"] == "training" else 1024
    path = tmp_path / "rl.json"
    path.write_text(json.dumps(body))

    with pytest.raises(ValueError, match="sampling max_seq_len 1024 exceeds its training max_seq_len 512"):
        utils.build_job("rl", model="Qwen/Qwen3-8B", context="standard", job_config=str(path))

    # The command-line max_seq_len applies to both sub-jobs, so it brings the
    # supplied sampling window back in line and the build is accepted.
    body = utils.build_job("rl", model="Qwen/Qwen3-8B", context="standard", job_config=str(path), max_seq_len=2048)
    windows = {s["job_type"]: (s.get("training_config") or s.get("inference_config"))["max_seq_len"] for s in body["sub_job_configs"]}
    assert windows == {"training": 2048, "sampling": 2048}


def test_a_matching_job_config_is_still_accepted_for_each_mode(tmp_path):
    utils = _import_example("recipes.utils")
    training = tmp_path / "t.json"
    training.write_text(json.dumps(utils.make_config("Qwen/Qwen3-8B", "standard")))
    inference = tmp_path / "i.json"
    inference.write_text(json.dumps(utils.make_config("Qwen/Qwen3-8B", "standard", inference=True)))
    rl = tmp_path / "rl.json"
    rl.write_text(json.dumps(utils.make_config("Qwen/Qwen3-8B", "standard", rl=True)))

    assert {s["job_type"] for s in utils.build_job("training", model="Qwen/Qwen3-8B", context="standard", job_config=str(training))["sub_job_configs"]} == {"training"}
    assert {s["job_type"] for s in utils.build_job("inference", model="Qwen/Qwen3-8B", context="standard", job_config=str(inference))["sub_job_configs"]} == {"sampling"}
    assert {s["job_type"] for s in utils.build_job("rl", model="Qwen/Qwen3-8B", context="standard", job_config=str(rl))["sub_job_configs"]} == {"training", "sampling"}


def test_rl_job_config_is_used_as_is(tmp_path):
    utils = _import_example("recipes.utils")
    record = utils.make_config("Qwen/Qwen3-8B", "standard", rl=True)
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))

    assert _job_body("recipes.rl_quickstart.math_grpo.train", job_config=str(path)) == record


def _by_job_type(body: dict) -> dict:
    return {sub["job_type"]: sub for sub in body["sub_job_configs"]}


@pytest.mark.parametrize(
    ("model", "context", "lora"),
    [("Qwen/Qwen3-8B", "standard", True), ("Qwen/Qwen3.6-35B-A3B", "long", False)],
)
@pytest.mark.parametrize("kind", ["training", "inference"])
def test_an_rl_job_config_names_either_catalog_file_of_a_pair(model, context, lora, kind):
    utils = _import_example("recipes.utils")
    folder = REPO_ROOT / "model-catalog" / "models" / utils.model_slug(model) / context
    path = folder / utils.catalog_file_name("h200", kind, lora)

    body = _job_body("recipes.rl_quickstart.math_grpo.train", job_config=str(path), model=model, context=context, lora=lora)

    # The file plus its sibling's other sub-job, sampling first: the catalog's RL
    # composition without rl/overrides.json or router replay.
    composed = utils.make_config(model, context, lora=lora, rl=True)
    for sub in composed["sub_job_configs"]:
        (sub.get("training_config") or sub.get("inference_config")).pop("router_replay", None)
    assert [sub["job_type"] for sub in body["sub_job_configs"]] == ["sampling", "training"]
    assert body["hardware"] == composed["hardware"]
    assert _by_job_type(body) == _by_job_type(composed)


def test_an_rl_job_config_with_both_sub_jobs_is_not_paired(tmp_path):
    utils = _import_example("recipes.utils")
    record = utils.make_config("Qwen/Qwen3-8B", "standard", lora=True, rl=True)
    path = tmp_path / "h200-training-lora.json"
    path.write_text(json.dumps(record))
    sibling = REPO_ROOT / "model-catalog/models/qwen3-8b/standard/h200-inference-lora.json"
    (tmp_path / "h200-inference-lora.json").write_text(sibling.read_text())

    assert utils.build_job("rl", model="Qwen/Qwen3-8B", context="standard", job_config=str(path)) == record


def test_an_rl_job_config_without_a_sibling_keeps_the_composition_error(tmp_path):
    utils = _import_example("recipes.utils")
    path = tmp_path / "h200-training-lora.json"
    path.write_text((REPO_ROOT / "model-catalog/models/qwen3-8b/standard/h200-training-lora.json").read_text())

    with pytest.raises(ValueError, match=r"has sub-jobs \['training'\]; a rl job needs exactly \['sampling', 'training'\]"):
        utils.build_job("rl", model="Qwen/Qwen3-8B", context="standard", job_config=str(path))


def test_overrides_json_sits_between_the_catalog_and_the_command_line(tmp_path):
    utils = _import_example("recipes.utils")
    (tmp_path / "overrides.json").write_text(
        json.dumps(
            {"sub_job_configs": [{"job_type": "training", "training_config": {"max_seq_len": 16384, "train_batch_size": 2}}]}
        )
    )

    body = utils.build_job("training", model="Qwen/Qwen3-8B", context="standard", example_dir=tmp_path, max_seq_len=8192)
    training = _sub_job(body, "training")["training_config"]

    assert (training["max_seq_len"], training["train_batch_size"]) == (8192, 2)


def test_every_example_ships_an_overrides_file_with_a_comment():
    # Every example ships an overrides.json carrying a human "_comment"; SFT and
    # inference are comment-only (no real overrides), RL, distillation and DARE
    # also carry their real settings.
    sft = SFT_OVERRIDES
    inference = EXAMPLES / "inference" / "overrides.json"
    distillation_overrides = EXAMPLES / "distillation" / "overrides.json"
    dflash2_overrides = EXAMPLES / "dflash2" / "overrides.json"
    assert set(EXAMPLES.rglob("overrides.json")) == {
        sft,
        RL_OVERRIDES,
        inference,
        distillation_overrides,
        DARE_OVERRIDES,
        dflash2_overrides,
    }

    for path in (sft, inference):
        data = json.loads(path.read_text())
        assert set(data) == {"_comment"} and isinstance(data["_comment"], str) and data["_comment"]

    for path in (RL_OVERRIDES, distillation_overrides, DARE_OVERRIDES, dflash2_overrides):
        assert isinstance(json.loads(path.read_text()).get("_comment"), str)

    dare = json.loads(DARE_OVERRIDES.read_text())
    ds_batch_fields = {"train_batch_size": None, "train_micro_batch_size_per_gpu": None, "gradient_accumulation_steps": None}
    assert dare["sub_job_configs"] == [
        {"job_type": "training", "training_config": {"n_gpus": 8, "ds_config": ds_batch_fields}}
    ]

    override = json.loads(RL_OVERRIDES.read_text())
    (training,) = [sub for sub in override["sub_job_configs"] if sub["job_type"] == "training"]
    config = training["training_config"]
    assert config["optimizer"] == {"betas": [0.9, 0.95]}
    assert config["mb_spec"] == {"max_tokens_per_mb": 10240}
    assert config["ds_config"] == {"zero_optimization": {"reduce_scatter": True}}

    distillation = json.loads(distillation_overrides.read_text())
    (student_training,) = [sub for sub in distillation["sub_job_configs"] if sub["job_type"] == "training"]
    student_config = student_training["training_config"]
    assert student_config["n_gpus"] == 1 and student_config["train_batch_size"] == 1
    assert student_config["mb_spec"] == {"max_tokens_per_mb": 10240}
    assert student_config["ds_config"]["zero_optimization"]["reduce_scatter"] is True


def test_dare_applies_its_own_overrides_after_the_sft_file():
    utils = _import_example("recipes.utils")
    dare = _import_example("recipes.sft_quickstart.dare.train")

    body = _job_body("recipes.sft_quickstart.dare.train")

    # The catalog file, sft/overrides.json, dare/overrides.json, then DARE's lr and batch defaults.
    assert body == utils.build_job(
        "training",
        model="Qwen/Qwen3-8B",
        context="standard",
        example_dir=EXAMPLES / "sft",
        overrides=[DARE_OVERRIDES],
        lr=dare.Config().lr,
        train_batch_size=dare.Config().train_batch_size,
    )
    assert _sub_job(body, "training")["training_config"]["n_gpus"] == 8
    # Only DARE has the file: conversational SFT keeps the catalog's GPU count.
    assert _sub_job(_job_body("recipes.sft_quickstart.conversational.train"), "training")["training_config"]["n_gpus"] == 4


def test_an_override_that_moves_the_gpu_count_past_the_deepspeed_batch_is_rejected():
    utils = _import_example("recipes.utils")
    more_gpus = {"sub_job_configs": [{"job_type": "training", "training_config": {"n_gpus": 8}}]}

    # Qwen3-30B-A3B's file spells out a DeepSpeed batch of 4 for its 4 GPUs.
    with pytest.raises(ValueError, match=r"ds_config train_batch_size 4 must equal .* 8 data-parallel ranks"):
        utils.build_job("training", model="Qwen/Qwen3-30B-A3B", context="standard", overrides=[more_gpus])


def test_the_sft_and_inference_overrides_files_are_no_ops(monkeypatch):
    utils = _import_example("recipes.utils")
    monkeypatch.delenv(utils.DEBUG_OPTIONS_ENV, raising=False)

    sft_with = utils.build_job("training", model="Qwen/Qwen3-8B", context="standard", example_dir=EXAMPLES / "sft")
    assert sft_with == utils.make_config("Qwen/Qwen3-8B", "standard")

    inference_with = utils.build_job(
        "inference", model="Qwen/Qwen3-8B", context="standard", example_dir=EXAMPLES / "inference"
    )
    assert inference_with == utils.make_config("Qwen/Qwen3-8B", "standard", inference=True)
    assert "_comment" not in json.dumps(sft_with) and "_comment" not in json.dumps(inference_with)


def test_build_job_adds_the_debug_tag_and_the_source_checkpoint(monkeypatch):
    utils = _import_example("recipes.utils")
    endpoint = _import_example("recipes.inference.endpoint")
    monkeypatch.delenv(utils.DEBUG_OPTIONS_ENV, raising=False)

    body, source = endpoint.inference_endpoint_body(
        model="Qwen/Qwen3-8B",
        context="standard",
        lora=True,
        source_job_id="job-1",
        checkpoint_id="cp_1",
        debug_image_tag="dev_test",
    )

    assert source == {"checkpoint_id": "cp_1", "source_job_id": "job-1"}
    assert _sub_job(body, "sampling")["source_checkpoint_info"] == source
    assert body["debug"] == {"job": {"image_tag": "dev_test"}}
    assert os.environ[utils.DEBUG_OPTIONS_ENV] == "1"


def test_a_source_checkpoint_needs_an_inference_job():
    utils = _import_example("recipes.utils")

    with pytest.raises(ValueError, match="a source checkpoint starts an inference job"):
        utils.build_job("training", model="Qwen/Qwen3-8B", context="standard", source_checkpoint={"checkpoint_id": "cp_1"})


def test_inference_settings_reject_training_hyperparameters():
    utils = _import_example("recipes.utils")

    with pytest.raises(ValueError, match="lr and train_batch_size apply to a training sub-job"):
        utils.build_job("inference", model="Qwen/Qwen3-8B", context="standard", lr=1e-5)


def test_rl_example_records_the_generated_body_with_its_debug_tag(tmp_path, monkeypatch):
    runner = _import_example("recipes.rl_quickstart.runner")
    module = _import_example("recipes.rl_quickstart.math_grpo.train")
    monkeypatch.setattr(runner, "CONFIGS_DIR", tmp_path / "configs")
    config = module.Config(config="unused", model="Qwen/Qwen3.6-35B-A3B", context="long")
    body = dict(runner.job_body(config), debug={"job": {"image_tag": "dev_test"}})

    path = runner.write_job_record(config, body)
    runner.write_job_record(config, body)

    assert path == tmp_path / "configs" / "qwen3.6-35b-a3b-long-h200-lora.json"
    # The record is the generated body verbatim, debug image tag kept, overwriting each run.
    assert json.loads(path.read_text()) == body
    assert json.loads(path.read_text())["debug"] == {"job": {"image_tag": "dev_test"}}
    assert [found.name for found in (tmp_path / "configs").iterdir()] == [path.name]


class _FakeJobClient:
    """Records the job calls ``running_job`` makes; ``fail`` names one that raises."""

    def __init__(self, events: list[str], fail: str | None = None):
        self.events = events
        self.fail = fail

    def _call(self, name: str) -> None:
        self.events.append(name)
        if name == self.fail:
            raise RuntimeError(f"{name} failed")

    def create_job_from_body(self, body: dict) -> dict:
        self._call("create")
        return {"job_id": "job-1"}

    def wait_for_job(self, job_id: str) -> None:
        self._call("wait")

    def cancel_job(self, job_id: str) -> None:
        self._call("cancel")


class _EmptyTask:
    name = "empty"
    evaluation_name = "none"

    def load(self, seed: int):
        return type("Dataset", (), {"train": [], "test": None})()


def _run_rl_example(monkeypatch, tmp_path, events, *, fail=None, renderer_error=None, **settings):
    """Run the RL example's ``train`` with no steps against a fake client."""
    runner = _import_example("recipes.rl_quickstart.runner")
    module = _import_example("recipes.rl_quickstart.math_grpo.train")
    monkeypatch.setattr(runner, "CONFIGS_DIR", tmp_path / "configs")
    record = runner.write_job_record

    def write_job_record(config, body):
        events.append("record")
        return record(config, body)

    def build_renderer(model_name):
        if renderer_error is not None:
            raise renderer_error
        renderer = type("Renderer", (), {"get_stop_sequences": lambda self: ["<|im_end|>"]})()
        return type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 0})(), renderer, "fake"

    ml_logger = type("Logger", (), {"log_metrics": lambda self, *args, **kwargs: None, "close": lambda self: None})()
    monkeypatch.setattr(runner, "write_job_record", write_job_record)
    monkeypatch.setattr(runner, "build_renderer", build_renderer)
    monkeypatch.setattr(runner, "make_client", lambda config_path, connection=None, **kw: _FakeJobClient(events, fail))
    monkeypatch.setattr(runner, "setup_logging", lambda *args, **kwargs: ml_logger)
    monkeypatch.setattr(runner, "save_example_checkpoints", lambda client, job_id: {"weights-only": {"checkpoint_id": "cp_1"}})
    config = module.Config(config="unused", max_steps=0, eval_every=0, **settings)
    path = runner.job_record_path(config)
    path.parent.mkdir(parents=True)
    path.write_text("previous run\n")
    runner.train(config, _EmptyTask())
    return path


def test_rl_example_records_the_generated_body_before_it_submits(tmp_path, monkeypatch):
    events: list[str] = []
    runner = _import_example("recipes.rl_quickstart.runner")
    module = _import_example("recipes.rl_quickstart.math_grpo.train")

    path = _run_rl_example(monkeypatch, tmp_path, events)

    # The record is written before the job is submitted, not after it reaches running.
    assert events == ["record", "create", "wait", "cancel"]
    assert json.loads(path.read_text()) == runner.job_body(module.Config(config="unused"))


def test_the_record_router_replay_caps_are_the_fixed_defaults(tmp_path, monkeypatch):
    utils = _import_example("recipes.utils")
    runner = _import_example("recipes.rl_quickstart.runner")
    module = _import_example("recipes.rl_quickstart.math_grpo.train")
    monkeypatch.setattr(runner, "CONFIGS_DIR", tmp_path / "configs")

    def caps_for(problems_per_batch, group_size):
        config = module.Config(config="unused", model="Qwen/Qwen3.6-35B-A3B", context="long",
                               problems_per_batch=problems_per_batch, group_size=group_size)
        path = runner.write_job_record(config, runner.job_body(config))
        training = next(s for s in json.loads(path.read_text())["sub_job_configs"] if s["job_type"] == "training")
        return training["training_config"]["router_replay"]["max_cache_bytes"]

    # The caps are the fixed default regardless of the rollout batch.
    assert caps_for(8, 4) == caps_for(64, 16) == utils.DEFAULT_ROUTER_REPLAY_TRAINING_CACHE_BYTES


@pytest.mark.parametrize(
    ("fail", "renderer_error", "events"),
    [
        (None, OSError("no tokenizer"), ["record"]),
        ("create", None, ["record", "create"]),
        ("wait", None, ["record", "create", "wait"]),
    ],
)
def test_a_run_that_fails_before_its_job_runs_leaves_its_own_generated_body(
    tmp_path, monkeypatch, fail, renderer_error, events
):
    seen: list[str] = []
    runner = _import_example("recipes.rl_quickstart.runner")
    module = _import_example("recipes.rl_quickstart.math_grpo.train")

    with pytest.raises((OSError, RuntimeError)):
        _run_rl_example(monkeypatch, tmp_path, seen, fail=fail, renderer_error=renderer_error)

    # Option A: the record is written before submit, so a failed startup leaves this
    # run's own generated body, not the stale "previous run" placeholder.
    assert seen == events
    record = (tmp_path / "configs" / "qwen3-8b-standard-h200-lora.json").read_text()
    assert record != "previous run\n"
    assert json.loads(record) == runner.job_body(module.Config(config="unused"))


def test_an_rl_run_attached_to_a_job_writes_no_record(tmp_path, monkeypatch, caplog):
    events: list[str] = []
    caplog.set_level("INFO", logger="recipes.rl_quickstart.runner")

    path = _run_rl_example(monkeypatch, tmp_path, events, job_id="job-9")

    assert "record" not in events
    assert path.read_text() == "previous run\n"
    assert "Attached to job job-9, so this run writes no job record" in caplog.text


def test_a_job_config_run_writes_no_record(tmp_path, monkeypatch):
    utils = _import_example("recipes.utils")
    runner = _import_example("recipes.rl_quickstart.runner")
    monkeypatch.setattr(runner, "CONFIGS_DIR", tmp_path / "configs")
    job_config = tmp_path / "job.json"
    job_config.write_text(json.dumps(utils.make_config("Qwen/Qwen3-8B", "standard", lora=True, rl=True)))

    events: list[str] = []
    path = _run_rl_example(monkeypatch, tmp_path, events, job_config=str(job_config))

    assert "record" not in events
    assert path.read_text() == "previous run\n"


class _RolloutClient(_FakeJobClient):
    """A job client whose generate returns two rollouts with different rewards."""

    def generate(self, job_id: str, *, prompts, sampling_params) -> str:
        return "request-1"

    def poll_request(self, job_id: str, request_id: str) -> dict:
        return {"results": [{"text": "right", "token_ids": [3]}, {"text": "wrong", "token_ids": [4]}]}


class _OneProblemTask:
    name = "one problem"
    evaluation_name = "none"

    def load(self, seed: int):
        return type("Dataset", (), {"train": [("problem", "right")], "test": None})()

    def build_prompt(self, problem, renderer) -> list[int]:
        return [1, 2]

    def score_response(self, response, answer, *, result, max_tokens):
        return float(response == answer), {}


@pytest.mark.parametrize(
    ("lora", "weight_sync_format", "sent"),
    [
        (False, "hf", "hf"),
        (False, None, None),
        (True, "hf", "lora"),
    ],
)
def test_rl_weight_sync_format_applies_to_full_parameter_runs(tmp_path, monkeypatch, lora, weight_sync_format, sent):
    runner = _import_example("recipes.rl_quickstart.runner")
    module = _import_example("recipes.rl_quickstart.math_grpo.train")
    syncs: list[dict] = []
    renderer = type("Renderer", (), {"get_stop_sequences": lambda self: ["<|im_end|>"]})()
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 0})()
    ml_logger = type("Logger", (), {"log_metrics": lambda self, *args, **kwargs: None, "close": lambda self: None})()
    monkeypatch.setattr(runner, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(runner, "build_renderer", lambda model_name: (tokenizer, renderer, "fake"))
    monkeypatch.setattr(runner, "make_client", lambda config_path, connection=None, **kw: _RolloutClient([]))
    monkeypatch.setattr(runner, "setup_logging", lambda *args, **kwargs: ml_logger)
    monkeypatch.setattr(runner, "sequence_from_rollout", lambda prompt, sampled, *, advantage: (prompt, sampled))
    monkeypatch.setattr(runner, "collate", lambda datums, **kwargs: ({}, {}))
    monkeypatch.setattr(runner, "forward_backward_step", lambda *args, **kwargs: ({"avg_loss": 0.0}, {}))
    monkeypatch.setattr(runner, "sync_weights", lambda client, job_id, **kwargs: syncs.append(kwargs))
    monkeypatch.setattr(runner, "save_example_checkpoints", lambda client, job_id: {"weights-only": {"checkpoint_id": "cp_1"}})

    config = module.Config(
        config="unused",
        lora=lora,
        weight_sync_format=weight_sync_format,
        problems_per_batch=1,
        group_size=2,
        max_steps=1,
        eval_every=0,
    )
    runner.train(config, _OneProblemTask())

    # A full-parameter run sends the chosen format (None keeps the server default);
    # a LoRA run always syncs only the adapter.
    assert syncs == [{"weight_format": sent, "bucket_size": None}]


def test_only_an_explicit_debug_image_tag_authorizes_debug_options(tmp_path, monkeypatch):
    import types

    utils = _import_example("recipes.utils")
    from cortex_training import CortexTrainingClient
    from cortex_training.client.core import DEBUG_OPTIONS_ENV

    # Registering the key with monkeypatch restores os.environ after the test, even
    # though build_job sets it directly during the run.
    monkeypatch.delenv(DEBUG_OPTIONS_ENV, raising=False)

    # An explicit debug_image_tag is the caller's opt-in: it sets the body's debug
    # block and authorizes the internal-only gate for this process.
    record = utils.build_job(
        "rl", model="Qwen/Qwen3-8B", context="standard", lora=True,
        rollout_batch_size=8, debug_image_tag="dev_test",
    )
    assert record["debug"] == {"job": {"image_tag": "dev_test"}}
    assert os.environ.get(DEBUG_OPTIONS_ENV) == "1"
    record_file = tmp_path / "qwen3-8b-standard-h200-lora.json"
    record_file.write_text(json.dumps(record))

    client = CortexTrainingClient(base_url="http://test.local", database="DB", schema="SCH")
    sent: dict = {}

    def fake_send(method, path, **kwargs):
        sent["body"] = kwargs.get("json")
        return types.SimpleNamespace(json=lambda: {"job_id": "job-1"})

    monkeypatch.setattr(client, "_send", fake_send)

    # Replaying a saved record that merely carries a debug block does NOT elevate
    # on its own: build_job leaves the process gate as it found it (cleared here),
    # so the client's internal-only check still rejects the debug-bearing body.
    monkeypatch.delenv(DEBUG_OPTIONS_ENV, raising=False)
    replayed = utils.build_job(
        "rl", model="Qwen/Qwen3-8B", context="standard", lora=True,
        rollout_batch_size=8, job_config=str(record_file),
    )
    assert replayed["debug"] == {"job": {"image_tag": "dev_test"}}
    assert os.environ.get(DEBUG_OPTIONS_ENV) is None
    with pytest.raises(ValueError, match="debug options are an internal-only capability"):
        client.create_job_from_body(replayed)

    # The caller re-authorizes explicitly by passing debug_image_tag= again.
    reauthorized = utils.build_job(
        "rl", model="Qwen/Qwen3-8B", context="standard", lora=True,
        rollout_batch_size=8, job_config=str(record_file), debug_image_tag="dev_test",
    )
    assert os.environ.get(DEBUG_OPTIONS_ENV) == "1"
    assert client.create_job_from_body(reauthorized) == {"job_id": "job-1"}
    assert sent["body"]["debug"] == {"job": {"image_tag": "dev_test"}}


def test_rl_records_are_not_committed():
    patterns = _ignore_patterns()

    assert _ignored(PurePosixPath("recipes/rl_quickstart/configs"), is_dir=True, patterns=patterns)
    assert not _ignored(PurePosixPath("recipes/rl_quickstart/math_grpo"), is_dir=True, patterns=patterns)


def _catalog_selections(kind: str) -> list[tuple[str, str, str, bool]]:
    """(model, context, hardware, lora) for each catalog job file of ``kind``."""
    selections = []
    for path in sorted((REPO_ROOT / "model-catalog" / "models").glob(f"*/*/*-{kind}-*.json")):
        model = json.loads((path.parents[1] / "model.json").read_text())["model_id"]
        hardware, _, mode = path.stem.split("-")
        selections.append((model, path.parent.name, hardware, mode == "lora"))
    return selections


@pytest.mark.parametrize(("model", "context", "hardware", "lora"), _catalog_selections("training"))
def test_dare_builds_a_runnable_mesh_or_stops_for_every_catalog_training_file(model, context, hardware, lora):
    utils = _import_example("recipes.utils")
    catalog = utils.make_config(model, context, hardware=hardware, lora=lora)["sub_job_configs"][0]["training_config"]
    # DARE's ds_config nulls let the server derive the batch, so only the catalog
    # file's own sp_size/ep_size can fail to fit 8 GPUs.
    reasons = utils.parallelism_errors(dict(catalog, n_gpus=8))

    if reasons:
        # Rejected before submission, naming the override and each reason.
        with pytest.raises(ValueError) as raised:
            _job_body("recipes.sft_quickstart.dare.train", model=model, context=context, hardware=hardware, lora=lora)
        message = str(raised.value)
        assert "training parallelism does not fit its GPU count after" in message
        assert str(DARE_OVERRIDES) in message
        assert all(reason in message for reason in reasons), message
        return
    body = _job_body("recipes.sft_quickstart.dare.train", model=model, context=context, hardware=hardware, lora=lora)
    training = _sub_job(body, "training")["training_config"]
    assert training["n_gpus"] == 8
    assert utils.parallelism_errors(training) == []
    assert utils._deepspeed_batch_errors(training) == []
    # DeepSpeed's batch, as the server builds it, fits the 8-GPU layout.
    ds_batch, micro, accumulation, ranks = _server_deepspeed_batch(training)
    assert ds_batch == micro * accumulation * ranks


def test_dare_keeps_a_fitting_mesh_and_rejects_one_that_needs_more_gpus():
    # Qwen3-30B-A3B's 4-GPU EP4 mesh still divides 8 GPUs, and its catalog
    # DeepSpeed batch of 4 is dropped for the server to derive at 8.
    training = _sub_job(_job_body("recipes.sft_quickstart.dare.train", model="Qwen/Qwen3-30B-A3B"), "training")["training_config"]
    assert (training["n_gpus"], training["ep_size"]) == (8, 4)
    assert not {"train_batch_size", "train_micro_batch_size_per_gpu", "gradient_accumulation_steps"} & set(training["ds_config"])
    assert _server_deepspeed_batch(training) == (8, 1, 1, 8)

    # GLM-5.3-Flash-BF16 trains on a 16-GPU SP16/EP16 mesh, which 8 GPUs cannot hold.
    with pytest.raises(ValueError, match=r"sp_size 16 must be a positive divisor of n_gpus 8; ep_size 16"):
        _job_body("recipes.sft_quickstart.dare.train", model="zai-org/GLM-5.3-Flash-BF16")


def _assert_serves_checkpoint(utils, training_sub: dict, args: dict) -> None:
    inference = utils.make_config(
        args["model"], args["context"], hardware=args["hardware"], lora=args["lora"], inference=True
    )
    (sampling_sub,) = inference["sub_job_configs"]
    assert sampling_sub["model_name"] == training_sub["model_name"]
    adapter = training_sub["training_config"].get("peft_config")
    if adapter is None:
        assert "peft_config" not in sampling_sub["inference_config"]
    else:
        assert utils.same_adapter(sampling_sub["inference_config"]["peft_config"], adapter)


@pytest.mark.parametrize(("model", "context", "hardware", "lora"), _catalog_selections("training"))
def test_follow_up_commands_pick_an_inference_file_that_serves_the_checkpoint(model, context, hardware, lora):
    utils = _import_example("recipes.utils")
    body = utils.build_job("training", model=model, context=context, hardware=hardware, lora=lora)

    args = utils.inference_job_args(body, context=context)

    if args == {"job_config": utils.INFERENCE_JOB_CONFIG_PLACEHOLDER}:
        # A training mesh with no sampler of this mode has nothing to follow up with.
        folder = REPO_ROOT / "model-catalog" / "models" / utils.model_slug(model)
        assert not any(
            (folder / candidate / utils.catalog_file_name(hardware, "inference", lora)).is_file()
            for candidate in ("standard", "long")
        ), (model, context, hardware, lora)
        return
    assert set(args) == {"model", "context", "hardware", "lora"}
    assert (args["model"], args["hardware"], args["lora"]) == (model, hardware, lora)
    _assert_serves_checkpoint(utils, body["sub_job_configs"][0], args)


@pytest.mark.parametrize("model", ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.8-27B"])
@pytest.mark.parametrize("lora", [False, True])
def test_standard_runs_follow_up_at_standard_context(model, lora):
    # These models now ship standard-context inference files, so a standard
    # training run serves its checkpoint at standard context (matching adapter).
    utils = _import_example("recipes.utils")
    body = utils.build_job("training", model=model, context="standard", lora=lora)

    assert utils.inference_job_args(body, context="standard") == {
        "model": model,
        "context": "standard",
        "hardware": "h200",
        "lora": lora,
    }


def test_rl_follow_up_commands_serve_the_checkpoint():
    utils = _import_example("recipes.utils")
    for model, context, hardware, lora in _catalog_selections("training"):
        folder = REPO_ROOT / "model-catalog" / "models" / utils.model_slug(model) / context
        if not (folder / utils.catalog_file_name(hardware, "inference", lora)).is_file():
            continue
        if _sampling_window_exceeds_training(utils, model, context, hardware, lora):
            continue
        body = utils.build_job("rl", model=model, context=context, hardware=hardware, lora=lora, rollout_batch_size=8)
        args = utils.inference_job_args(body, context=context)
        assert args == {"model": model, "context": context, "hardware": hardware, "lora": lora}
        _assert_serves_checkpoint(utils, body["sub_job_configs"][0], args)


@pytest.mark.parametrize(
    ("model", "context", "expected"),
    [
        # A job_config run picks the context its training max_seq_len falls in, when it can.
        ("Qwen/Qwen3.6-35B-A3B", "long", "long"),
        ("Qwen/Qwen3-8B", "standard", "standard"),
        ("Qwen/Qwen3.5-4B", "standard", "standard"),
    ],
)
def test_job_config_runs_pick_the_context_from_their_sequence_length(model, context, expected):
    utils = _import_example("recipes.utils")
    body = utils.make_config(model, context, lora=True)

    args = utils.inference_job_args(body)

    assert args == {"model": model, "context": expected, "hardware": "h200", "lora": True}
    _assert_serves_checkpoint(utils, body["sub_job_configs"][0], args)


def _server_deepspeed_batch(training_config: dict) -> tuple[int, int, int, int]:
    """DeepSpeed's batch fields as the server builds them, and the data-parallel degree.

    The server derives them from the GPU layout -- one micro-batch per GPU, no
    gradient accumulation, a DeepSpeed ``train_batch_size`` of ``n_gpus /
    sp_size`` -- and merges ``ds_config`` on top. A step's rows travel in its
    forward-backward request, so ``training_config.train_batch_size`` is not
    one of them.
    """
    ranks = training_config["n_gpus"] // (training_config.get("sp_size") or 1)
    fields = {"train_batch_size": ranks, "train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 1}
    fields.update({key: value for key, value in (training_config.get("ds_config") or {}).items() if key in fields})
    return fields["train_batch_size"], fields["train_micro_batch_size_per_gpu"], fields["gradient_accumulation_steps"], ranks


@pytest.mark.parametrize("train_batch_size", [1, 2, 64])
@pytest.mark.parametrize(("model", "context", "hardware", "lora"), _catalog_selections("training"))
def test_a_command_line_batch_leaves_the_deepspeed_batch_to_the_gpu_layout(model, context, hardware, lora, train_batch_size):
    utils = _import_example("recipes.utils")
    catalog = utils.make_config(model, context, hardware=hardware, lora=lora)["sub_job_configs"][0]["training_config"]

    body = utils.build_job(
        "training", model=model, context=context, hardware=hardware, lora=lora, train_batch_size=train_batch_size
    )

    training = body["sub_job_configs"][0]["training_config"]
    assert training["train_batch_size"] == train_batch_size
    assert training.get("ds_config") == catalog.get("ds_config")
    ds_batch, micro, accumulation, ranks = _server_deepspeed_batch(training)
    assert ds_batch == micro * accumulation * ranks
    assert _server_deepspeed_batch(training) == _server_deepspeed_batch(dict(training, ds_config={}))


def _training_override(**sub_job) -> dict:
    return {"sub_job_configs": [{"job_type": "training", **sub_job}]}


@pytest.mark.parametrize(
    ("override", "needs"),
    [
        # An adapter no catalog inference file holds.
        (
            _training_override(training_config={"peft_config": {"r": 8}}),
            'Qwen/Qwen3-8B on h200 with peft_config {"bias": "none", "lora_alpha": 32',
        ),
        # A model the catalog does not have.
        (_training_override(model_name="org/Custom-8B"), "org/Custom-8B on h200 with peft_config"),
    ],
)
def test_follow_up_commands_ask_for_an_inference_job_when_no_catalog_file_fits(caplog, override, needs):
    utils = _import_example("recipes.utils")
    caplog.set_level("WARNING", logger="recipes.utils")
    body = utils.make_config("Qwen/Qwen3-8B", "standard", lora=True, overrides=[override])

    args = utils.inference_job_args(body, context="standard")

    assert args == {"job_config": "INFERENCE_JOB_CONFIG"}
    assert "No inference file in the model catalog serves this checkpoint" in caplog.text
    assert f"It needs a sampling sub-job for {needs}" in caplog.text


def test_follow_up_commands_need_an_inference_file_on_the_run_hardware(caplog):
    utils = _import_example("recipes.utils")
    caplog.set_level("WARNING", logger="recipes.utils")
    body = dict(utils.make_config("Qwen/Qwen3-8B", "standard"), hardware="B200")

    assert utils.inference_job_args(body, context="standard") == {"job_config": "INFERENCE_JOB_CONFIG"}
    assert "It needs a sampling sub-job for Qwen/Qwen3-8B on b200 with full weights" in caplog.text


def test_lora_adapters_match_whatever_the_order_of_their_target_modules():
    utils = _import_example("recipes.utils")
    adapter = {"peft_type": "Lora", "r": 32, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]}

    assert utils.same_adapter(adapter, dict(adapter, target_modules=["v_proj", "q_proj"], task_type="CAUSAL_LM"))
    assert not utils.same_adapter(adapter, dict(adapter, r=16))
    assert not utils.same_adapter(adapter, None)


def test_deepseek_v4_trainer_and_vllm_module_names_are_the_same_adapter():
    utils = _import_example("recipes.utils")
    trainer = {
        "peft_type": "Lora",
        "r": 32,
        "lora_alpha": 32,
        "target_modules": r".*self_attn\.(q_a_proj|q_b_proj|kv_proj|o_b_proj)$",
    }
    sampler = {
        "peft_type": "Lora",
        "r": 32,
        "lora_alpha": 32,
        "target_modules": ["wq_a", "wkv", "wq_b", "wo_b"],
    }

    assert utils.same_adapter(trainer, sampler)
    assert not utils.same_adapter(trainer, dict(sampler, target_modules=["wq_a"]))


@pytest.mark.parametrize(
    "relative_path",
    [
        "recipes/sft_quickstart/conversational/train.py",
        "recipes/rl_quickstart/math_grpo/train.py",
        "recipes/distillation/on_policy/train.py",
    ],
)
def test_task_modules_are_entrypoints(relative_path):
    source = (REPO_ROOT / relative_path).read_text()

    assert 'if __name__ == "__main__":' in source
    assert "chz.nested_entrypoint(main)" in source


@pytest.mark.parametrize(
    ("relative_path", "example_package"),
    [
        ("recipes/sft_quickstart/runner.py", "recipes.sft_quickstart.conversational"),
        ("recipes/rl_quickstart/runner.py", "recipes.rl_quickstart.math_grpo"),
        ("recipes/distillation/train.py", ".tasks."),
    ],
)
def test_shared_runners_are_task_agnostic_libraries(relative_path, example_package):
    source = (REPO_ROOT / relative_path).read_text()

    assert 'if __name__ == "__main__":' not in source
    assert example_package not in source


@pytest.mark.parametrize(
    ("module_name", "task_type"),
    [
        ("recipes.sft_quickstart.conversational.train", "ConversationalTask"),
        ("recipes.rl_quickstart.math_grpo.train", "MathTask"),
    ],
)
def test_entrypoint_selects_task(monkeypatch, module_name, task_type):
    module = _import_example(module_name)
    captured = {}

    def fake_train(config, task):
        captured["config"] = config
        captured["task"] = task

    monkeypatch.setattr(module, "train", fake_train)
    config = module.Config(config="unused")
    module.main(config)

    assert captured["config"] is config
    assert type(captured["task"]).__name__ == task_type


def _metadata_keys(path: Path) -> set[str]:
    return {
        line.split(":", 1)[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line[0].isspace() and not line.startswith(("#", "-")) and ":" in line
    }


@pytest.mark.parametrize("folder", EXAMPLE_FOLDERS)
def test_each_example_folder_holds_its_metadata(folder):
    assert (EXAMPLES / folder / "example.yaml").is_file()


def _printed_args(text: str, module: str) -> dict[str, str]:
    line = next(line for line in text.splitlines() if line.startswith(f"  python -m {module} "))
    return dict(token.split("=", 1) for token in shlex.split(line)[3:])


def _stub_tinker(monkeypatch, table):
    """Install a fake tinker_cookbook whose model_info returns ``table[model]`` (KeyError otherwise)."""
    import sys
    import types

    def get_recommended_renderer_names(model):
        return table[model]

    fake = types.ModuleType("tinker_cookbook")
    fake.model_info = types.SimpleNamespace(get_recommended_renderer_names=get_recommended_renderer_names)
    monkeypatch.setitem(sys.modules, "tinker_cookbook", fake)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.model_info", fake.model_info)


def test_recommended_renderer_name_raises_clearly_when_no_variant_matches(monkeypatch):
    utils = importlib.import_module("recipes.utils")
    _stub_tinker(monkeypatch, {
        "has-both": ["q3", "q3_disable_thinking"],
        "thinking-only": ["g_high_reasoning", "g_low_reasoning"],
    })

    # A model with both variants resolves either way; None takes tinker's top pick.
    assert utils.recommended_renderer_name("has-both", enable_thinking=False) == "q3_disable_thinking"
    assert utils.recommended_renderer_name("has-both", enable_thinking=True) == "q3"
    assert utils.recommended_renderer_name("thinking-only", enable_thinking=None) == "g_high_reasoning"
    assert utils.recommended_renderer_name("thinking-only", enable_thinking=True) == "g_high_reasoning"

    # No disable-thinking variant with enable_thinking=False: a clear ValueError, not IndexError.
    with pytest.raises(ValueError, match="no disable-thinking renderer for 'thinking-only'"):
        utils.recommended_renderer_name("thinking-only", enable_thinking=False)
    # A model tinker does not know at all still raises the clear "no recommended renderer" ValueError.
    with pytest.raises(ValueError, match="no recommended renderer for 'unknown'"):
        utils.recommended_renderer_name("unknown", enable_thinking=False)


def test_saved_checkpoints_print_a_command_that_serves_the_checkpoint(caplog):
    utils = _import_example("recipes.utils")
    endpoint = _import_example("recipes.inference.endpoint")
    caplog.set_level("INFO", logger="recipes.utils")
    body = utils.build_job("training", model="Qwen/Qwen3.5-4B", context="standard", lora=True)

    utils.log_saved_checkpoints(
        connection_args="config=conn.json",
        job_id="job-1",
        saved={"weights-only": {"checkpoint_id": "cp_1"}},
        sampling_command="evaluate",
        job_args=utils.inference_job_args(body, context="standard"),
        temperature=1.0,
    )

    assert (
        "  python -m recipes.inference.evaluate config=conn.json source_job_id=job-1 "
        "model=Qwen/Qwen3.5-4B context=standard hardware=h200 lora=true checkpoint_id=cp_1 temperature=1.0"
    ) in caplog.text
    args = _printed_args(caplog.text, "recipes.inference.evaluate")
    served, source = endpoint.inference_endpoint_body(
        model=args["model"],
        context=args["context"],
        hardware=args["hardware"],
        lora=args["lora"] == "true",
        source_job_id=args["source_job_id"],
        checkpoint_id=args["checkpoint_id"],
    )
    (sampling,) = served["sub_job_configs"]
    assert sampling["source_checkpoint_info"] == source
    assert utils.same_adapter(sampling["inference_config"]["peft_config"], body["sub_job_configs"][0]["training_config"]["peft_config"])


def test_saved_checkpoints_print_an_explicit_job_config_when_no_catalog_file_fits(caplog):
    utils = _import_example("recipes.utils")
    caplog.set_level("INFO", logger="recipes.utils")
    body = utils.make_config("Qwen/Qwen3-8B", "standard", overrides=[_training_override(model_name="org/Custom-8B")])

    utils.log_saved_checkpoints(
        connection_args="config=conn.json",
        job_id="job-1",
        saved={"weights-only": {"checkpoint_id": "cp_1"}},
        sampling_command="sample",
        job_args=utils.inference_job_args(body, context="standard"),
    )

    assert (
        "  python -m recipes.inference.generate config=conn.json source_job_id=job-1 "
        "job_config=INFERENCE_JOB_CONFIG checkpoint_id=cp_1"
    ) in caplog.text


def _follow_up_body(module_name: str, args: dict[str, str]) -> dict:
    """The inference job a printed follow-up command builds, parsed as its entry point parses it."""
    import chz

    module = _import_example(module_name)
    endpoint = _import_example("recipes.inference.endpoint")
    config = chz.entrypoint(module.Config, argv=[f"{name}={value}" for name, value in args.items()])
    body, _ = endpoint.inference_endpoint_body(
        model=config.model,
        context=config.context,
        hardware=config.hardware,
        lora=config.lora,
        job_config=config.job_config,
        max_seq_len=config.max_seq_len,
    )
    return body


@pytest.mark.parametrize(
    ("settings", "printed"),
    [
        (
            {"model": "zai-org/GLM-5.3", "context": "long", "max_seq_len": 65536},
            {"model": "zai-org/GLM-5.3", "context": "long", "hardware": "h200", "lora": "false", "max_seq_len": "65536"},
        ),
        (
            {"model": "Qwen/Qwen3.5-9B", "context": "standard", "lora": True},
            {"model": "Qwen/Qwen3.5-9B", "context": "standard", "hardware": "h200", "lora": "true"},
        ),
    ],
)
def test_serve_prints_follow_up_commands_for_the_endpoint_it_built(caplog, monkeypatch, settings, printed):
    serve = _import_example("recipes.inference.serve")
    monkeypatch.setattr(serve, "make_client", lambda config_path, connection=None, **kw: _FakeJobClient([]))
    caplog.set_level("INFO", logger="recipes.inference.endpoint")

    serve.main(serve.Config(config="conn.json", **settings))
    served = _import_example("recipes.inference.endpoint").inference_endpoint_body(**settings)[0]

    for module_name in ("recipes.inference.generate", "recipes.inference.evaluate"):
        args = _printed_args(caplog.text, module_name)
        assert args == {"config": "conn.json", "job_id": "job-1", **printed}
        assert _follow_up_body(module_name, args) == served


def test_serve_follow_up_commands_repeat_its_job_config(caplog, monkeypatch, tmp_path):
    utils = _import_example("recipes.utils")
    serve = _import_example("recipes.inference.serve")
    monkeypatch.setattr(serve, "make_client", lambda config_path, connection=None, **kw: _FakeJobClient([]))
    caplog.set_level("INFO", logger="recipes.inference.endpoint")
    job_config = tmp_path / "glm 5.3.json"
    job_config.write_text(json.dumps(utils.make_config("zai-org/GLM-5.3", "long", inference=True)))

    serve.main(serve.Config(config="conn.json", job_config=str(job_config)))

    for module_name in ("recipes.inference.generate", "recipes.inference.evaluate"):
        args = _printed_args(caplog.text, module_name)
        assert args == {"config": "conn.json", "job_id": "job-1", "job_config": str(job_config)}
        assert _follow_up_body(module_name, args)["sub_job_configs"][0]["model_name"] == "zai-org/GLM-5.3"


def test_a_job_config_replaces_the_catalog_file_and_overrides_json(tmp_path, monkeypatch):
    utils = _import_example("recipes.utils")
    endpoint = _import_example("recipes.inference.endpoint")
    example_dir = tmp_path / "inference"
    example_dir.mkdir()
    (example_dir / "overrides.json").write_text(
        json.dumps({"sub_job_configs": [{"job_type": "sampling", "inference_config": {"n_gpus": 8}}]})
    )
    monkeypatch.setattr(endpoint, "_EXAMPLE_DIR", example_dir)
    monkeypatch.delenv(utils.DEBUG_OPTIONS_ENV, raising=False)
    job_config = tmp_path / "job.json"
    job_config.write_text(json.dumps(utils.make_config("Qwen/Qwen3-8B", "standard", inference=True)))

    from_catalog, _ = endpoint.inference_endpoint_body(model="Qwen/Qwen3-8B", context="standard")
    from_file, source = endpoint.inference_endpoint_body(
        model="Qwen/Qwen3-8B",
        context="standard",
        job_config=str(job_config),
        max_seq_len=4096,
        source_job_id="job-0",
        checkpoint_id="cp_1",
        debug_image_tag="dev_test",
    )

    assert "A ``job_config`` file replaces both of those layers" in " ".join(endpoint.inference_endpoint_body.__doc__.split())
    assert from_catalog["sub_job_configs"][0]["inference_config"]["n_gpus"] == 8
    (sampling,) = from_file["sub_job_configs"]
    assert sampling["inference_config"]["n_gpus"] == 2
    assert sampling["inference_config"]["max_seq_len"] == 4096
    assert sampling["source_checkpoint_info"] == source == {"checkpoint_id": "cp_1", "source_job_id": "job-0"}
    assert from_file["debug"] == {"job": {"image_tag": "dev_test"}}


def test_dare_evaluate_builds_its_inference_job_from_the_catalog(tmp_path):
    pytest.importorskip("sklearn")  # dare_bench.evaluation
    utils = _import_example("recipes.utils")
    endpoint = _import_example("recipes.inference.endpoint")
    evaluate = _import_example("recipes.sft_quickstart.dare.evaluate")

    # The DARE evaluator no longer ships a pre-restructure config file; its
    # defaults select the catalog inference file (Qwen/Qwen3-8B, standard, full).
    config = evaluate.Config(config="conn.json", bench_dir=str(tmp_path))
    assert config.job_config is None
    source_text = Path(evaluate.__file__).read_text(encoding="utf-8")
    assert "configs/qwen3_8b_full.json" not in source_text
    assert "configs/" not in source_text

    from_catalog, _ = endpoint.inference_endpoint_body(
        model=config.model,
        context=config.context,
        hardware=config.hardware,
        lora=config.lora,
        job_config=config.job_config,
        max_seq_len=config.max_seq_len,
        source_job_id="job-0",
        checkpoint_id="cp_1",
    )
    sampling = _sub_job(from_catalog, "sampling")
    assert sampling["model_name"] == "Qwen/Qwen3-8B"
    assert sampling["source_checkpoint_info"] == {"checkpoint_id": "cp_1", "source_job_id": "job-0"}

    # job_config= stays the explicit-file escape hatch.
    job_config = tmp_path / "job.json"
    job_config.write_text(json.dumps(utils.make_config("Qwen/Qwen3-8B", "standard", inference=True)))
    from_file, _ = endpoint.inference_endpoint_body(
        model=config.model,
        context=config.context,
        hardware=config.hardware,
        lora=config.lora,
        job_config=str(job_config),
    )
    assert _sub_job(from_file, "sampling")["model_name"] == "Qwen/Qwen3-8B"

    # The checkpoint-eval command the SFT runner prints for DARE carries the
    # catalog selectors so the evaluator rebuilds this job, not its defaults.
    task = _import_example("recipes.sft_quickstart.dare.task").DareTask(variant="fv")
    command = task.eval_command(
        connection_args="config=conn.json",
        job_id="job-0",
        checkpoint_id="cp_1",
        job_args=utils.inference_job_args(
            utils.make_config("Qwen/Qwen3.5-4B", "standard", lora=True),
            context="standard",
        ),
    )
    assert command.startswith("python -m recipes.sft_quickstart.dare.evaluate ")
    assert "model=Qwen/Qwen3.5-4B" in command
    assert "context=standard" in command
    assert "lora=true" in command


@pytest.mark.parametrize(
    ("path", "section"),
    [("recipes/api_quickstart/README.md", "## Create-job bodies"), ("recipes/README.md", "## Create a job config from the command line")],
)
def test_job_body_docs_submit_with_ct(path, section):
    text = (REPO_ROOT / path).read_text()
    start = text.index(section)
    end = text.find("\n## ", start + len(section))
    blocks = re.findall(r"```bash\n(.*?)```", text[start : end if end != -1 else None], re.S)
    submits = [line.strip() for block in blocks for line in block.splitlines() if " submit " in f" {line} "]

    assert any(line.startswith("ct submit --dry-run ") for line in submits)
    assert all(line.startswith("ct submit ") for line in submits)


def test_reference_docs_only_import_packaged_modules():
    """The REST/SDK contract docs must not send a (pip-installed) reader to a
    module the wheel does not ship: every ``cortex_training`` import they show
    must resolve in the installed package, and they must never instruct
    importing ``recipes/`` or the removed ``router_replay_planner``."""
    import importlib

    import_stmt = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import\s+(.+)|import\s+([\w.]+))\s*$")
    seen = 0
    for name in ("rest-api.md", "python-sdk.md"):
        path = REPO_ROOT / "docs" / "reference" / name
        for line in path.read_text(encoding="utf-8").splitlines():
            match = import_stmt.match(line)
            if not match:
                continue
            module = match.group(1) or match.group(3)
            if module == "recipes" or module.startswith("examples."):
                raise AssertionError(f"{name} instructs importing recipes/: {line.strip()!r}")
            if module == "cortex_training.router_replay_planner":
                raise AssertionError(f"{name} imports the planner removed from the SDK: {line.strip()!r}")
            if module != "cortex_training" and not module.startswith("cortex_training."):
                continue
            imported = importlib.import_module(module)  # must resolve in the installed package
            if match.group(2):
                for symbol in match.group(2).replace("(", "").replace(")", "").split(","):
                    symbol = symbol.strip()
                    if symbol and symbol != "*":
                        assert hasattr(imported, symbol), f"{module} has no {symbol} ({name})"
            seen += 1
    assert "router_replay_planner" not in (REPO_ROOT / "docs" / "reference" / "python-sdk.md").read_text(encoding="utf-8")
    assert seen >= 4  # both contract docs import names from the installed package


def test_job_configs_come_from_the_catalog():
    utils = _import_example("recipes.utils")

    assert not (EXAMPLES / "sft_quickstart" / "configs").exists()
    assert not (EXAMPLES / "inference" / "configs").exists()
    assert not hasattr(utils, "sampling_job_body")
    # Distillation builds its student and teacher jobs from the catalog too, via
    # make_distillation_config and recipes/distillation/overrides.json; configs/
    # is a gitignored per-run output and holds no committed job files.
    assert hasattr(utils, "make_distillation_config")
    assert (EXAMPLES / "distillation" / "overrides.json").is_file()
    distillation_configs = EXAMPLES / "distillation" / "configs"
    assert not distillation_configs.exists() or not any(distillation_configs.glob("*.json"))


def test_examples_keep_task_code_and_metadata_in_their_own_folders():
    assert not (EXAMPLES / "sft_quickstart" / "example.yaml").exists()
    assert not (EXAMPLES / "rl_quickstart" / "example.yaml").exists()
    assert not (EXAMPLES / "sft_quickstart" / "tasks").exists()
    assert not (EXAMPLES / "rl_quickstart" / "tasks").exists()
    assert not (EXAMPLES / "inference" / "identity.py").exists()
    assert (EXAMPLES / "sft_quickstart" / "conversational" / "task.py").is_file()
    assert (EXAMPLES / "sft_quickstart" / "conversational" / "identity.py").is_file()
    assert (EXAMPLES / "rl_quickstart" / "math_grpo" / "task.py").is_file()


@pytest.mark.parametrize("folder", ["sft_quickstart/conversational", "rl_quickstart/math_grpo", "inference"])
def test_catalog_owned_fields_leave_example_metadata(folder):
    keys = _metadata_keys(EXAMPLES / folder / "example.yaml")

    assert "models" not in keys
    assert "default_training_gpus" not in keys


def _ignore_patterns() -> list[str]:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _ignored(path: PurePosixPath, *, is_dir: bool, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/") and not is_dir:
            continue
        pattern = pattern.rstrip("/")
        if "/" in pattern:
            if fnmatch.fnmatch(path.as_posix(), pattern.lstrip("/")):
                return True
        elif fnmatch.fnmatch(path.name, pattern):
            return True
    return False


def _repository_files():
    """Every file a search of the repository sees: not hidden, not ignored."""
    patterns = _ignore_patterns()
    for directory, dirnames, filenames in os.walk(REPO_ROOT):
        relative = PurePosixPath(Path(directory).relative_to(REPO_ROOT).as_posix())
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not name.startswith(".")
            and not _ignored(relative / name, is_dir=True, patterns=patterns)
        )
        for name in sorted(filenames):
            path = relative / name
            if not name.startswith(".") and not _ignored(path, is_dir=False, patterns=patterns):
                yield path


def test_examples_folder_is_renamed_to_recipes():
    assert not (REPO_ROOT / "examples").exists()
    assert (REPO_ROOT / "recipes").is_dir()


# A catalog job file is picked by model, context, hardware and mode, so nothing
# outside the model's folder names one by path (tests aside).
_CATALOG_JOB_FILE = re.compile(r"\b(standard|long)/(h200|b200|b300)-(training|inference)-(lora|full)\.json")


def test_nothing_outside_the_catalog_names_a_catalog_job_file():
    found = []
    for path in _repository_files():
        if path.parts[:2] == ("model-catalog", "models") or path.parts[0] == "tests":
            continue
        data = (REPO_ROOT / path).read_bytes()
        if b"\0" in data:
            continue
        for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            if _CATALOG_JOB_FILE.search(line):
                found.append(f"{path}:{number}: {line.strip()}")

    assert found == []


def test_repository_search_skips_ignored_and_hidden_files():
    files = {path.as_posix() for path in _repository_files()}

    assert "recipes/utils.py" in files
    assert ".gitignore" not in files
    assert not any(name.startswith((".git/", ".venv/")) or "__pycache__" in name for name in files)
