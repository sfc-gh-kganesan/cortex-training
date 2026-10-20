from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_recipe(module_name: str):
    for dependency in ("chz", "datasets", "tinker_cookbook"):
        pytest.importorskip(dependency)
    return importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("runner_name", "task_name", "task_type"),
    [
        (
            "recipes.sft.train",
            "recipes.sft.conversational.train",
            "ConversationalTask",
        ),
        (
            "recipes.rl.train",
            "recipes.rl.math_grpo.train",
            "MathTask",
        ),
    ],
)
def test_task_entrypoints_call_shared_runner(runner_name, task_name, task_type):
    runner = _import_recipe(runner_name)
    task_entrypoint = _import_recipe(task_name)

    assert task_entrypoint.train is runner.train
    assert task_entrypoint.job_body is runner.job_body
    assert task_entrypoint.Config.__module__ == task_name
    assert getattr(task_entrypoint, task_type).__module__.startswith("recipes.")


def test_task_helpers_remain_available_from_task_entrypoints():
    conversational = _import_recipe("recipes.sft.conversational.train")
    math_grpo = _import_recipe("recipes.rl.math_grpo.train")

    assert conversational.load_chat_dataset.__module__ == (
        "recipes.sft.tasks.conversational"
    )
    assert math_grpo.score_response.__module__ == "recipes.rl.tasks.math"


@pytest.mark.parametrize(
    ("module_name", "config_name", "model_name"),
    [
        (
            "recipes.sft.conversational.train",
            "configs/qwen3_8b_full.json",
            "Qwen/Qwen3-8B",
        ),
        (
            "recipes.rl.math_grpo.train",
            "configs/qwen3_8b_lora.json",
            "Qwen/Qwen3-8B",
        ),
    ],
)
def test_hoisted_config_paths_resolve_outside_repo(
    monkeypatch,
    tmp_path,
    module_name,
    config_name,
    model_name,
):
    monkeypatch.chdir(tmp_path)
    module = _import_recipe(module_name)
    body = module.job_body(module.Config(config="unused", job_config=config_name))

    assert any(job["model_name"] == model_name for job in body["sub_job_configs"])


@pytest.mark.parametrize(
    "relative_path",
    [
        "recipes/sft/conversational/train.py",
        "recipes/rl/math_grpo/train.py",
        "recipes/distillation/on_policy/train.py",
    ],
)
def test_task_modules_are_entrypoints(relative_path):
    source = (REPO_ROOT / relative_path).read_text()

    assert 'if __name__ == "__main__":' in source
    assert "chz.nested_entrypoint(main)" in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "recipes/sft/train.py",
        "recipes/rl/train.py",
        "recipes/distillation/train.py",
    ],
)
def test_shared_runners_are_task_agnostic_libraries(relative_path):
    source = (REPO_ROOT / relative_path).read_text()

    assert 'if __name__ == "__main__":' not in source
    assert ".tasks." not in source


@pytest.mark.parametrize(
    ("module_name", "task_type"),
    [
        ("recipes.sft.conversational.train", "ConversationalTask"),
        ("recipes.rl.math_grpo.train", "MathTask"),
    ],
)
def test_entrypoint_selects_task(monkeypatch, module_name, task_type):
    module = _import_recipe(module_name)
    captured = {}

    def fake_train(config, task):
        captured["config"] = config
        captured["task"] = task

    monkeypatch.setattr(module, "train", fake_train)
    config = module.Config(config="unused")
    module.main(config)

    assert captured["config"] is config
    assert type(captured["task"]).__name__ == task_type


def test_configs_and_metadata_are_hoisted():
    assert (REPO_ROOT / "recipes/sft/recipe.yaml").is_file()
    assert (REPO_ROOT / "recipes/rl/recipe.yaml").is_file()
    assert not list((REPO_ROOT / "recipes/sft/conversational/configs").glob("*.json"))
    assert not list((REPO_ROOT / "recipes/rl/math_grpo/configs").glob("*.json"))
