from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "model-catalog"))

import smoke_test_catalog as smoke_catalog  # noqa: E402
from smoke_test_catalog import apply_debug_image_tag  # noqa: E402
from smoke_test_catalog import apply_training_sequence_parallelism  # noqa: E402
from smoke_test_catalog import build_catalog_request  # noqa: E402
from smoke_test_catalog import build_forward_backward_probe_spec  # noqa: E402
from smoke_test_catalog import enable_training_memory_telemetry  # noqa: E402
from smoke_test_catalog import run_data_plane_probes  # noqa: E402
from validate_catalog import CatalogValidationError  # noqa: E402
from validate_catalog import job_errors  # noqa: E402
from validate_catalog import validate_catalog  # noqa: E402

CATALOG_DIR = REPO_ROOT / "model-catalog" / "models"

_TRAINABLE_STANDARD = [
    "standard/h200-inference-full",
    "standard/h200-inference-lora",
    "standard/h200-training-full",
    "standard/h200-training-lora",
]
_TRAINABLE_LONG = [
    "long/h200-inference-full",
    "long/h200-inference-lora",
    "long/h200-training-full",
    "long/h200-training-lora",
]
# Full-weight training and inference files for both contexts, no LoRA.
_FULL_STANDARD_LONG = [
    "standard/h200-inference-full",
    "standard/h200-training-full",
    "long/h200-inference-full",
    "long/h200-training-full",
]
EXPECTED_FILES = {
    "Qwen/Qwen3-0.6B": _TRAINABLE_STANDARD,
    "Qwen/Qwen3-1.7B": _TRAINABLE_STANDARD,
    "Qwen/Qwen3-4B-Instruct-2507": _FULL_STANDARD_LONG,
    "Qwen/Qwen3-8B": _TRAINABLE_STANDARD,
    "Qwen/Qwen3.5-4B": _TRAINABLE_LONG + _TRAINABLE_STANDARD,
    "Qwen/Qwen3.5-9B": _TRAINABLE_STANDARD + ["long/h200-inference-full", "long/h200-training-full"],
    "Qwen/Qwen3.6-35B-A3B": _TRAINABLE_LONG + _TRAINABLE_STANDARD,
    "Qwen/Qwen3.8-27B": _TRAINABLE_LONG + _TRAINABLE_STANDARD,
    "Qwen/Qwen3-30B-A3B": [
        "standard/h200-inference-full",
        "standard/h200-training-full",
    ],
    "Qwen/Qwen3.8-Flash-Next": [
        "standard/h200-inference-full",
        "standard/h200-inference-lora",
        "standard/h200-training-full",
        "standard/h200-training-lora",
        "long/h200-inference-full",
        "long/h200-training-full",
        "long/h200-training-lora",
    ],
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": [
        "standard/h200-inference-lora",
        "standard/h200-training-full",
        "standard/h200-training-lora",
        "long/h200-inference-lora",
        "long/h200-training-full",
        "long/h200-training-lora",
    ],
    "openai/gpt-oss-120b": ["long/h200-inference-full"],
    "google/gemma-3-4b-it": _FULL_STANDARD_LONG,
    "google/gemma-4-31B-it": _FULL_STANDARD_LONG,
    "google/medgemma-4b-it": _FULL_STANDARD_LONG,
    "google/medgemma-1.5-4b-it": _FULL_STANDARD_LONG,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": _FULL_STANDARD_LONG,
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": _FULL_STANDARD_LONG,
    "zai-org/GLM-5.3": ["standard/h200-inference-full", "long/h200-inference-full"],
    "zai-org/GLM-5.3-Flash": [
        "standard/h200-inference-full",
        "standard/h200-inference-lora",
        "standard/h200-training-lora",
        "long/h200-inference-full",
        "long/h200-inference-lora",
        "long/h200-training-lora",
    ],
    "zai-org/GLM-5.3-Flash-BF16": _FULL_STANDARD_LONG,
}


def _models(catalog_dir: Path = CATALOG_DIR) -> dict[str, tuple[Path, dict]]:
    models = {}
    for folder in sorted(catalog_dir.iterdir()):
        info = json.loads((folder / "model.json").read_text())
        models[info["model_id"]] = (folder, info)
    return models


def _files(folder: Path) -> list[str]:
    return sorted(path.relative_to(folder).with_suffix("").as_posix() for path in folder.glob("*/*.json"))


def _body(model_id: str, key: str) -> dict:
    folder, _ = _models()[model_id]
    return json.loads((folder / f"{key}.json").read_text())


def _config(body: dict) -> dict:
    (sub_job,) = body["sub_job_configs"]
    return sub_job.get("training_config") or sub_job.get("inference_config")


# ─── The checked-in catalog ──────────────────────────────────────────────────


def test_checked_in_catalog_is_valid():
    assert validate_catalog() == {
        "models": 21,
        "files": 98,
        "training": 48,
        "inference": 50,
        "rl": 45,
        "distillation": 1,
    }


def test_catalog_holds_every_model_and_its_files():
    assert {model_id: _files(folder) for model_id, (folder, _) in _models().items()} == {
        model_id: sorted(files) for model_id, files in EXPECTED_FILES.items()
    }


def test_catalog_models_include_license_links():
    licenses = {model_id: info["license"] for model_id, (_, info) in _models().items()}

    assert {model_id: license["name"] for model_id, license in licenses.items()} == {
        "Qwen/Qwen3-0.6B": "Apache 2.0",
        "Qwen/Qwen3-1.7B": "Apache 2.0",
        "Qwen/Qwen3-8B": "Apache 2.0",
        "Qwen/Qwen3.5-4B": "Apache 2.0",
        "Qwen/Qwen3.5-9B": "Apache 2.0",
        "Qwen/Qwen3.6-35B-A3B": "Apache 2.0",
        "Qwen/Qwen3.8-27B": "Apache 2.0",
        "Qwen/Qwen3-30B-A3B": "Apache 2.0",
        "Qwen/Qwen3-4B-Instruct-2507": "Apache 2.0",
        "Qwen/Qwen3.8-Flash-Next": "Apache 2.0",
        "deepseek-ai/DeepSeek-V4.1-Flash-1017": "MIT",
        "openai/gpt-oss-120b": "Apache 2.0",
        "google/gemma-3-4b-it": "Gemma",
        "google/gemma-4-31B-it": "Gemma",
        "google/medgemma-4b-it": "Health AI Developer Foundations",
        "google/medgemma-1.5-4b-it": "Health AI Developer Foundations",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "NVIDIA Nemotron Open Model License",
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": "OpenMDW-1.1",
        "zai-org/GLM-5.3": "GLM-5.3 License",
        "zai-org/GLM-5.3-Flash": "MIT",
        "zai-org/GLM-5.3-Flash-BF16": "MIT",
    }
    # schema.json requires each license.url to match ^https:// (host is not constrained).
    assert all(license["url"].startswith("https://") for license in licenses.values())


def test_moe_models_record_router_replay_geometry():
    geometry = {
        model_id: (info["architecture"], info.get("moe"))
        for model_id, (_, info) in _models().items()
    }

    assert {model_id: moe for model_id, (architecture, moe) in geometry.items() if architecture == "moe"} == {
        "Qwen/Qwen3-30B-A3B": {"num_moe_layers": 48, "topk": 8, "num_experts": 128},
        "Qwen/Qwen3.6-35B-A3B": {"num_moe_layers": 40, "topk": 8, "num_experts": 256},
        "Qwen/Qwen3.8-Flash-Next": {"num_moe_layers": 48, "topk": 10, "num_experts": 512},
        "deepseek-ai/DeepSeek-V4.1-Flash-1017": {"num_moe_layers": 40, "topk": 6, "num_experts": 384},
        "openai/gpt-oss-120b": {"num_moe_layers": 36, "topk": 4, "num_experts": 128},
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": {"num_moe_layers": 23, "topk": 6, "num_experts": 128},
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": {"num_moe_layers": 23, "topk": 6, "num_experts": 128},
        "zai-org/GLM-5.3": {"num_moe_layers": 78, "topk": 8, "num_experts": 256},
        "zai-org/GLM-5.3-Flash": {"num_moe_layers": 42, "topk": 8, "num_experts": 288},
        "zai-org/GLM-5.3-Flash-BF16": {"num_moe_layers": 42, "topk": 8, "num_experts": 288},
    }
    assert all(moe is None for architecture, moe in geometry.values() if architecture == "dense")


def test_validation_dates_match_completed_live_smokes():
    validated = {
        (model_id, key): day
        for model_id, (_, info) in _models().items()
        for key, day in info["last_validated"].items()
        if day is not None
    }

    assert validated == {
        ("Qwen/Qwen3-4B-Instruct-2507", "long/h200-inference-full"): "2026-10-14",
        ("Qwen/Qwen3-4B-Instruct-2507", "long/h200-training-full"): "2026-10-14",
        ("Qwen/Qwen3.5-4B", "standard/h200-training-full"): "2026-09-02",
        ("Qwen/Qwen3.5-4B", "standard/h200-training-lora"): "2026-09-02",
        ("Qwen/Qwen3.5-9B", "long/h200-inference-full"): "2026-10-14",
        ("Qwen/Qwen3.5-9B", "long/h200-training-full"): "2026-10-14",
        ("Qwen/Qwen3.6-35B-A3B", "standard/h200-training-full"): "2026-09-02",
        ("Qwen/Qwen3.6-35B-A3B", "standard/h200-training-lora"): "2026-09-02",
        ("Qwen/Qwen3.8-27B", "long/h200-training-full"): "2026-09-02",
        ("Qwen/Qwen3.8-27B", "long/h200-training-lora"): "2026-09-02",
        ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "long/h200-training-full"): "2026-10-23",
        ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "long/h200-training-lora"): "2026-10-26",
        ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "standard/h200-training-full"): "2026-10-23",
        ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "standard/h200-training-lora"): "2026-10-26",
        ("google/gemma-4-31B-it", "long/h200-inference-full"): "2026-10-15",
        ("google/gemma-4-31B-it", "long/h200-training-full"): "2026-10-15",
        ("google/medgemma-1.5-4b-it", "long/h200-inference-full"): "2026-10-15",
        ("google/medgemma-1.5-4b-it", "long/h200-training-full"): "2026-10-15",
        ("google/medgemma-4b-it", "long/h200-inference-full"): "2026-10-15",
        ("google/medgemma-4b-it", "long/h200-training-full"): "2026-10-15",
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "long/h200-training-full"): "2026-10-14",
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "long/h200-training-full"): "2026-10-14",
        ("zai-org/GLM-5.3-Flash", "long/h200-inference-lora"): "2026-10-20",
        ("zai-org/GLM-5.3-Flash", "long/h200-training-lora"): "2026-10-20",
        ("zai-org/GLM-5.3-Flash-BF16", "long/h200-inference-full"): "2026-10-20",
        ("zai-org/GLM-5.3-Flash-BF16", "long/h200-training-full"): "2026-10-20",
    }


def test_files_keep_their_source_sequence_lengths():
    lengths = {
        (model_id, key.split("/")[0], key.split("-")[1]): _config(_body(model_id, key))["max_seq_len"]
        for model_id, files in EXPECTED_FILES.items()
        for key in files
    }

    assert lengths == {
        ("Qwen/Qwen3-0.6B", "standard", "inference"): 32768,
        ("Qwen/Qwen3-0.6B", "standard", "training"): 32768,
        ("Qwen/Qwen3-1.7B", "standard", "inference"): 32768,
        ("Qwen/Qwen3-1.7B", "standard", "training"): 32768,
        ("Qwen/Qwen3-8B", "standard", "inference"): 32768,
        ("Qwen/Qwen3-8B", "standard", "training"): 32768,
        ("Qwen/Qwen3.5-4B", "long", "inference"): 262144,
        ("Qwen/Qwen3.5-4B", "long", "training"): 262144,
        ("Qwen/Qwen3.5-4B", "standard", "inference"): 32768,
        ("Qwen/Qwen3.5-4B", "standard", "training"): 32768,
        ("Qwen/Qwen3.5-9B", "long", "inference"): 262144,
        ("Qwen/Qwen3.5-9B", "long", "training"): 262144,
        ("Qwen/Qwen3.5-9B", "standard", "inference"): 65536,
        ("Qwen/Qwen3.5-9B", "standard", "training"): 65536,
        ("Qwen/Qwen3.6-35B-A3B", "long", "inference"): 262144,
        ("Qwen/Qwen3.6-35B-A3B", "long", "training"): 262144,
        ("Qwen/Qwen3.6-35B-A3B", "standard", "inference"): 32768,
        ("Qwen/Qwen3.6-35B-A3B", "standard", "training"): 32768,
        ("Qwen/Qwen3.8-27B", "long", "inference"): 262144,
        ("Qwen/Qwen3.8-27B", "long", "training"): 262144,
        ("Qwen/Qwen3.8-27B", "standard", "inference"): 65536,
        ("Qwen/Qwen3.8-27B", "standard", "training"): 65536,
        ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "long", "inference"): 262144,
        ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "long", "training"): 262144,
        ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "standard", "inference"): 65536,
        ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "standard", "training"): 65536,
        ("openai/gpt-oss-120b", "long", "inference"): 131072,
        ("zai-org/GLM-5.3", "standard", "inference"): 65536,
        ("zai-org/GLM-5.3", "long", "inference"): 720896,
        ("Qwen/Qwen3-30B-A3B", "standard", "inference"): 40960,
        ("Qwen/Qwen3-30B-A3B", "standard", "training"): 40960,
        ("Qwen/Qwen3-4B-Instruct-2507", "standard", "inference"): 65536,
        ("Qwen/Qwen3-4B-Instruct-2507", "standard", "training"): 65536,
        ("Qwen/Qwen3-4B-Instruct-2507", "long", "inference"): 262144,
        ("Qwen/Qwen3-4B-Instruct-2507", "long", "training"): 262144,
        ("Qwen/Qwen3.8-Flash-Next", "standard", "inference"): 65536,
        ("Qwen/Qwen3.8-Flash-Next", "standard", "training"): 65536,
        ("Qwen/Qwen3.8-Flash-Next", "long", "inference"): 262144,
        ("Qwen/Qwen3.8-Flash-Next", "long", "training"): 262144,
        ("google/gemma-3-4b-it", "standard", "inference"): 65536,
        ("google/gemma-3-4b-it", "standard", "training"): 65536,
        ("google/gemma-3-4b-it", "long", "inference"): 131072,
        ("google/gemma-3-4b-it", "long", "training"): 131072,
        ("google/gemma-4-31B-it", "standard", "inference"): 65536,
        ("google/gemma-4-31B-it", "standard", "training"): 65536,
        ("google/gemma-4-31B-it", "long", "inference"): 262144,
        ("google/gemma-4-31B-it", "long", "training"): 262144,
        ("google/medgemma-4b-it", "standard", "inference"): 65536,
        ("google/medgemma-4b-it", "standard", "training"): 65536,
        ("google/medgemma-4b-it", "long", "inference"): 131072,
        ("google/medgemma-4b-it", "long", "training"): 131072,
        ("google/medgemma-1.5-4b-it", "standard", "inference"): 65536,
        ("google/medgemma-1.5-4b-it", "standard", "training"): 65536,
        ("google/medgemma-1.5-4b-it", "long", "inference"): 131072,
        ("google/medgemma-1.5-4b-it", "long", "training"): 131072,
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "standard", "inference"): 65536,
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "standard", "training"): 65536,
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "long", "inference"): 262144,
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "long", "training"): 262144,
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "standard", "inference"): 65536,
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "standard", "training"): 65536,
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "long", "inference"): 262144,
        ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "long", "training"): 262144,
        ("zai-org/GLM-5.3-Flash", "standard", "inference"): 65536,
        ("zai-org/GLM-5.3-Flash", "standard", "training"): 65536,
        ("zai-org/GLM-5.3-Flash", "long", "inference"): 262144,
        ("zai-org/GLM-5.3-Flash", "long", "training"): 262144,
        ("zai-org/GLM-5.3-Flash-BF16", "standard", "inference"): 65536,
        ("zai-org/GLM-5.3-Flash-BF16", "standard", "training"): 65536,
        ("zai-org/GLM-5.3-Flash-BF16", "long", "inference"): 262144,
        ("zai-org/GLM-5.3-Flash-BF16", "long", "training"): 262144,
    }
    # Standard is the model's smaller tier: where a model has both, standard < long.
    # (The original models cap standard at 32,768; the #107 families use larger
    # standard windows, so the tier ordering, not a fixed cap, is the invariant.)
    for (model_id, context, kind), length in lengths.items():
        if context == "standard":
            long_length = lengths.get((model_id, "long", kind))
            assert long_length is None or length < long_length, model_id


def test_inference_job_args_picks_the_models_own_standard_tier():
    # Tiers are per model, not a fixed 32,768 cutoff. Qwen3-4B-Instruct's standard
    # inference window is 65,536, so a 65,536 checkpoint must resolve to that model's
    # standard inference file, not its 262,144-token long one.
    from recipes.utils import inference_job_args

    body = {
        "hardware": "H200",
        "sub_job_configs": [
            {
                "job_type": "training",
                "model_name": "Qwen/Qwen3-4B-Instruct-2507",
                "training_config": {"max_seq_len": 65536},
            }
        ],
    }
    args = inference_job_args(body, catalog_dir=CATALOG_DIR)
    assert args == {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "context": "standard",
        "hardware": "h200",
        "lora": False,
    }
    # A larger checkpoint than the standard tier covers still resolves to long.
    body["sub_job_configs"][0]["training_config"]["max_seq_len"] = 262144
    assert inference_job_args(body, catalog_dir=CATALOG_DIR)["context"] == "long"


def test_every_file_names_its_hardware():
    assert {
        _body(model_id, key)["hardware"] for model_id, files in EXPECTED_FILES.items() for key in files
    } == {"H200"}


def test_qwen3_8b_files_carry_the_example_defaults():
    training = _config(_body("Qwen/Qwen3-8B", "standard/h200-training-full"))
    lora = _config(_body("Qwen/Qwen3-8B", "standard/h200-training-lora"))
    inference = _config(_body("Qwen/Qwen3-8B", "standard/h200-inference-lora"))

    assert (training["n_gpus"], training["train_batch_size"], training["optimizer"]["lr"]) == (4, 4, 2e-05)
    assert training["optimizer"]["betas"] == [0.9, 0.999]
    assert (lora["n_gpus"], lora["optimizer"]["lr"]) == (4, 1e-04)
    assert (inference["n_gpus"], inference["vllm_config"]["tensor_parallel_size"]) == (2, 1)
    assert inference["peft_config"] == lora["peft_config"]


@pytest.mark.parametrize(
    "key",
    [
        "standard/h200-training-lora",
        "long/h200-training-lora",
        "long/h200-inference-lora",
    ],
)
def test_qwen36_lora_uses_arctic_attention_targets(key):
    config = _config(_body("Qwen/Qwen3.6-35B-A3B", key))

    assert config["peft_config"]["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "training" in key:
        assert config["model_provider"] == "arctic"


def test_qwen36_long_training_uses_sequence_parallelism():
    for mode in ("lora", "full"):
        config = _config(_body("Qwen/Qwen3.6-35B-A3B", f"long/h200-training-{mode}"))
        assert (config["n_gpus"], config["sp_size"], config["train_batch_size"]) == (8, 8, 1)


def test_lora_inference_files_match_their_training_adapter():
    for model_id, files in EXPECTED_FILES.items():
        for key in files:
            if not key.endswith("inference-lora"):
                continue
            training_key = key.replace("inference-lora", "training-lora")
            inference = _config(_body(model_id, key))["peft_config"]
            training = _config(_body(model_id, training_key))["peft_config"]
            if model_id == "deepseek-ai/DeepSeek-V4.1-Flash-1017":
                # vLLM names these attention projections; the trainer uses the checkpoint names.
                assert {name: value for name, value in inference.items() if name != "target_modules"} == {
                    name: value for name, value in training.items() if name != "target_modules"
                }
                assert inference["target_modules"] == ["wq_a", "wkv", "wq_b", "wo_b"]
                assert training["target_modules"] == r".*self_attn\.(q_a_proj|q_b_proj|kv_proj|o_b_proj)$"
                continue
            assert inference == training, (model_id, key)


@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.6-35B-A3B",
        "Qwen/Qwen3.8-27B",
    ],
)
def test_qwen_inference_uses_single_gpu_tensor_parallelism(model_id):
    for key in EXPECTED_FILES[model_id]:
        # Qwen3.5-9B's 262,144-token long tier serves on eight GPUs instead.
        if "inference" in key and (model_id, key) != ("Qwen/Qwen3.5-9B", "long/h200-inference-full"):
            assert _config(_body(model_id, key))["vllm_config"]["tensor_parallel_size"] == 1


def test_qwen35_9b_long_tier_uses_eight_gpus():
    training = _config(_body("Qwen/Qwen3.5-9B", "long/h200-training-full"))
    inference = _config(_body("Qwen/Qwen3.5-9B", "long/h200-inference-full"))

    assert (training["n_gpus"], training["sp_size"], training["train_batch_size"]) == (8, 8, 1)
    assert (inference["n_gpus"], inference["vllm_config"]["tensor_parallel_size"]) == (8, 8)


def test_large_quantized_inference_uses_single_node_tp8_with_fp8_kv_cache():
    config = _config(_body("openai/gpt-oss-120b", "long/h200-inference-full"))

    assert config["n_gpus"] == 8
    assert config["vllm_config"]["tensor_parallel_size"] == 8
    assert config["vllm_config"]["kv_cache_dtype"] == "fp8"


def _deepseek_training(model_id: str, key: str) -> tuple[dict, dict, dict]:
    body = _body(model_id, key)
    (sub_job,) = body["sub_job_configs"]
    return sub_job, sub_job["training_config"], sub_job["training_config"]["ds_config"]


def test_deepseek_v41_training_meshes_match_the_completed_h200_runs():
    model_id = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    expected = {
        "standard/h200-training-full": (32, 32, 8, 65536, 4),
        "standard/h200-training-lora": (8, 8, 8, 65536, 1),
        "long/h200-training-full": (32, 32, 32, 262144, 1),
        "long/h200-training-lora": (16, 16, 16, 262144, 1),
    }
    for key, (n_gpus, ep_size, sp_size, seq_len, batch) in expected.items():
        sub_job, training, ds_config = _deepseek_training(model_id, key)
        assert sub_job["model_name"] == model_id
        assert "weight_format" not in sub_job
        assert (training["n_gpus"], training["ep_size"], training["sp_size"]) == (n_gpus, ep_size, sp_size)
        assert training["max_seq_len"] == seq_len
        assert training["train_batch_size"] == batch
        assert training["attn_implementation"] == "eager"
        assert training["model_provider"] == "arctic"
        assert "offload_config" not in training["ac_config"]
        assert "prime_rl" not in training
        assert training["model_options"] == {
            "lm_head": {"cross_entropy": False},
            "moe": {"comm_backend": "uccl"},
        }
        assert ds_config["bf16"]["bf16_optimizer_states"] is True
        assert ds_config["bf16"]["bf16_master_weights_and_grads"] is True
        assert ds_config["zero_optimization"]["parameter_alignment"] is True
        if key.endswith("training-lora"):
            assert training["optimizer"]["lr"] == 1e-4
            assert training["peft_config"]["r"] == 32
        else:
            assert training["optimizer"]["lr"] == 2e-5
            assert "peft_config" not in training
        if key == "standard/h200-training-lora":
            assert "offload_optimizer" not in ds_config["zero_optimization"]
        else:
            assert ds_config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"


def test_deepseek_v41_sampler_captures_every_layer_for_router_replay():
    model_id = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    for context in ("standard", "long"):
        inference = _config(_body(model_id, f"{context}/h200-inference-lora"))
        training = _config(_body(model_id, f"{context}/h200-training-lora"))
        assert training["router_replay"]["enabled"] is True
        assert inference["router_replay"]["enabled"] is True
        # Bounded replay runs the later layers on each request's trailing window only,
        # so those layers never route most prompt tokens.
        assert inference["vllm_config"]["extra_engine_kwargs"]["swa_bounded_replay"] is False


def test_glm53_catalog_file_serves_on_eight_gpus_with_the_fp8_mla_cache():
    body = _body("zai-org/GLM-5.3", "long/h200-inference-full")
    (sub_job,) = body["sub_job_configs"]
    config = sub_job["inference_config"]

    assert body["hardware"] == "H200"
    assert sub_job["model_name"] == "zai-org/GLM-5.3"
    assert config["n_gpus"] == 8
    assert config["vllm_config"]["tensor_parallel_size"] == 8
    assert config["vllm_config"]["kv_cache_dtype"] == "fp8_ds_mla"
    assert config["max_seq_len"] == 720896


def _yaml_models(path: Path) -> dict[str, dict]:
    """Parse the fixed layout of docs/_data/models.yaml."""
    models: dict[str, dict] = {}
    current: dict | None = None
    for line in path.read_text().splitlines():
        if line.startswith("  - id: "):
            current = models.setdefault(line.removeprefix("  - id: "), {"examples": []})
        elif line.startswith("      - ") and current is not None:
            current["examples"].append(line.removeprefix("      - "))
        elif line.startswith("    ") and ":" in line and current is not None:
            key, value = line.strip().split(":", 1)
            if value.strip():
                current[key] = {"true": True, "false": False, "null": None}.get(value.strip(), value.strip())
    return models


def _renderer_resolves(model_id: str) -> bool:
    """Whether the examples' renderer picker resolves for ``model_id`` at the default thinking mode."""
    from recipes.utils import recommended_renderer_name

    try:
        recommended_renderer_name(model_id, enable_thinking=False)
        return True
    except ValueError:
        return False


def _catalog_kinds(folder: Path):
    combos: dict[tuple[str, str, str], set[str]] = {}
    for key in _files(folder):
        context, name = key.split("/")
        hardware, kind, mode = name.split("-")
        combos.setdefault((context, hardware, mode), set()).add(kind)
    return combos, set().union(*combos.values()) if combos else set()


def test_docs_model_data_is_derived_from_the_catalog():
    # The examples: lists are gated on renderer resolution, which needs tinker.
    pytest.importorskip("tinker_cookbook")
    expected = {}
    for model_id, (folder, info) in _models().items():
        combos, kinds = _catalog_kinds(folder)
        examples = []
        if _renderer_resolves(model_id):
            if "training" in kinds:
                examples.append("conversational-sft")
            if "inference" in kinds:
                examples.append("inference-endpoint")
            if any(found == {"training", "inference"} for found in combos.values()):
                examples.append("math-grpo")
        dates = [day for day in info["last_validated"].values() if day]
        expected[model_id] = {
            "training": "training" in kinds,
            "inference": "inference" in kinds,
            "examples": examples,
            "last_validated": max(dates) if dates else None,
        }
    docs = _yaml_models(REPO_ROOT / "docs/_data/models.yaml")

    assert docs == expected
    assert "models.json" not in (REPO_ROOT / "docs/_data/models.yaml").read_text()


def test_docs_example_lists_are_gated_by_renderer_resolution():
    """Every advertised example model resolves a renderer; models whose renderer does not are listed with none."""
    pytest.importorskip("tinker_cookbook")
    docs = _yaml_models(REPO_ROOT / "docs/_data/models.yaml")

    advertised = {model_id for model_id, data in docs.items() if data["examples"]}
    # All 12 catalog models stay in the file; only the example lists are gated.
    assert set(docs) == set(_models())
    assert advertised == {model_id for model_id in _models() if _renderer_resolves(model_id)}
    # The models whose renderer does not resolve carry no example entries.
    for model_id, data in docs.items():
        if not _renderer_resolves(model_id):
            assert data["examples"] == [], model_id
    # A model with no examples still has catalog files, so the gate (not missing files) dropped it.
    for model_id, (folder, _) in _models().items():
        _, kinds = _catalog_kinds(folder)
        if not docs[model_id]["examples"]:
            assert kinds, model_id


def test_docs_examples_name_example_metadata_ids():
    ids = {
        line.removeprefix("id: ").strip()
        for path in (REPO_ROOT / "recipes").rglob("example.yaml")
        for line in path.read_text().splitlines()
        if line.startswith("id: ")
    }
    listed = {
        example
        for model in _yaml_models(REPO_ROOT / "docs/_data/models.yaml").values()
        for example in model["examples"]
    }

    assert listed <= ids


# ─── The validator ───────────────────────────────────────────────────────────


@pytest.fixture
def catalog(tmp_path):
    """A writable copy of the checked-in catalog."""
    copy_dir = tmp_path / "models"
    shutil.copytree(CATALOG_DIR, copy_dir, ignore=shutil.ignore_patterns("__pycache__"))
    return copy_dir


def _edit(path: Path, change) -> None:
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _edit_config(path: Path, change) -> None:
    def apply(body):
        (sub_job,) = body["sub_job_configs"]
        change(sub_job.get("training_config") or sub_job.get("inference_config"))

    _edit(path, apply)


def _invalid(catalog_dir: Path, match: str) -> str:
    with pytest.raises(CatalogValidationError, match=match) as raised:
        validate_catalog(catalog_dir)
    return str(raised.value)


def test_copied_catalog_is_valid(catalog):
    assert validate_catalog(catalog)["models"] == 21


def test_distillation_jobs_build_with_make_distillation_config(catalog):
    assert validate_catalog(catalog)["distillation"] == 1


def test_distillation_check_fails_when_a_default_model_is_absent(catalog):
    shutil.rmtree(catalog / "qwen3-8b")

    _invalid(catalog, r"default distillation model\(s\) missing from the catalog: Qwen/Qwen3-8B")


@pytest.mark.parametrize(
    "name",
    ["h200-rl-lora.json", "h100-training-lora.json", "h200-training.json", "notes.txt"],
)
def test_file_names_must_follow_the_hardware_kind_mode_pattern(catalog, name):
    shutil.copy(catalog / "qwen3-8b/standard/h200-training-lora.json", catalog / "qwen3-8b/standard" / name)

    _invalid(catalog, r"file names must match \{h200,b200,b300\}-\{training,inference\}-\{lora,full\}\.json")


def test_contexts_are_standard_and_long_only(catalog):
    shutil.copytree(catalog / "qwen3-8b/standard", catalog / "qwen3-8b/medium")

    _invalid(catalog, "qwen3-8b/medium: unexpected entry")


def test_folder_slug_must_derive_from_the_model_id(catalog):
    shutil.move(catalog / "qwen3-8b", catalog / "qwen3_8b")

    _invalid(catalog, "folder name must be 'qwen3-8b', the lowercased last segment")


def test_no_two_models_share_an_id(catalog):
    shutil.copytree(catalog / "qwen3-8b", catalog / "qwen3-8b-copy")

    _invalid(catalog, "model_id 'Qwen/Qwen3-8B' is also used by")


def test_no_two_models_share_a_slug(catalog):
    shutil.copytree(catalog / "qwen3-8b", catalog / "qwen3-8b-copy")
    _edit(catalog / "qwen3-8b-copy/model.json", lambda info: info.update(model_id="Other/Qwen3-8B"))

    _invalid(catalog, "slug 'qwen3-8b' is also used by")


def test_sub_job_model_name_must_match_model_json(catalog):
    _edit(
        catalog / "qwen3-8b/standard/h200-training-full.json",
        lambda body: body["sub_job_configs"][0].update(model_name="Qwen/Qwen3-1.7B"),
    )

    _invalid(catalog, "model_name 'Qwen/Qwen3-1.7B' does not match model.json model_id 'Qwen/Qwen3-8B'")


def test_job_hardware_must_match_the_prefix(catalog):
    _edit(catalog / "qwen3-8b/standard/h200-training-full.json", lambda body: body.update(hardware="B200"))

    _invalid(catalog, "hardware must be 'H200' to match the h200- prefix, got 'B200'")


def test_job_hardware_is_required(catalog):
    _edit(catalog / "qwen3-8b/standard/h200-training-full.json", lambda body: body.pop("hardware"))

    _invalid(catalog, "hardware must be 'H200' to match the h200- prefix, got None")


@pytest.mark.parametrize(
    "change",
    [lambda config: config.pop("peft_config"), lambda config: config.update(peft_config=None)],
    ids=["missing", "null"],
)
def test_lora_files_carry_peft_config(catalog, change):
    _edit_config(catalog / "qwen3-8b/standard/h200-inference-lora.json", change)

    _invalid(catalog, "lora files carry inference_config.peft_config, a LoRA adapter object")


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda peft: peft.update(r=0), r"training_config\.peft_config\.r must be a positive integer"),
        (lambda peft: peft.update(bias="all"), r"training_config\.peft_config\.bias='all' is not supported"),
        (lambda peft: peft.update(rank=8), r"training_config\.peft_config contains unsupported field\(s\): \['rank'\]"),
    ],
)
def test_lora_adapters_must_be_valid(catalog, change, match):
    _edit_config(catalog / "qwen3-8b/standard/h200-training-lora.json", lambda config: change(config["peft_config"]))

    _invalid(catalog, r"submit --dry-run fails: sub_job_configs\[0\]\." + match)


def test_full_files_do_not_carry_peft_config(catalog):
    peft = _config(_body("Qwen/Qwen3-8B", "standard/h200-training-lora"))["peft_config"]
    _edit_config(
        catalog / "qwen3-8b/standard/h200-training-full.json",
        lambda config: config.update(peft_config=peft),
    )

    _invalid(catalog, "full files must not carry training_config.peft_config")


def test_moe_models_need_their_geometry(catalog):
    _edit(catalog / "qwen3.6-35b-a3b/model.json", lambda info: info.pop("moe"))

    _invalid(catalog, "MoE models need moe geometry")


def test_dense_models_do_not_record_moe_geometry(catalog):
    _edit(
        catalog / "qwen3-8b/model.json",
        lambda info: info.update(moe={"num_moe_layers": 36, "topk": 8, "num_experts": 64}),
    )

    _invalid(catalog, "only MoE models record moe geometry")


def test_moe_geometry_is_checked_by_the_planner(catalog):
    _edit(catalog / "qwen3.6-35b-a3b/model.json", lambda info: info["moe"].update(topk=300))

    _invalid(catalog, r"moe geometry is invalid: topk \(300\) must not exceed num_experts_override \(256\)")


@pytest.mark.parametrize("n_gpus", [1, 2, 4, 8, 16, 24])
def test_sub_jobs_may_use_one_two_four_or_multiples_of_eight_gpus(catalog, n_gpus):
    _edit_config(catalog / "qwen3-8b/standard/h200-inference-full.json", lambda config: config.update(n_gpus=n_gpus))

    validate_catalog(catalog)


@pytest.mark.parametrize("n_gpus", [3, 6, 12])
def test_other_gpu_counts_are_rejected(catalog, n_gpus):
    _edit_config(catalog / "qwen3-8b/standard/h200-inference-full.json", lambda config: config.update(n_gpus=n_gpus))

    _invalid(catalog, f"n_gpus {n_gpus} must be 1, 2, 4, or a multiple of 8")


def test_tensor_parallel_size_must_divide_the_gpu_count(catalog):
    _edit_config(
        catalog / "glm-5.3/standard/h200-inference-full.json",
        lambda config: config["vllm_config"].update(tensor_parallel_size=3),
    )

    _invalid(catalog, "vllm_config.tensor_parallel_size 3 must divide n_gpus 8")


def test_sequence_parallel_size_must_divide_the_gpu_count(catalog):
    _edit_config(
        catalog / "qwen3.6-35b-a3b/long/h200-training-full.json",
        lambda config: config.update(sp_size=3),
    )

    _invalid(catalog, "sp_size 3 must divide n_gpus 8")


def test_long_context_must_be_longer_than_standard(catalog):
    _edit_config(
        catalog / "qwen3.6-35b-a3b/long/h200-inference-full.json",
        lambda config: config.update(max_seq_len=32768),
    )

    _invalid(
        catalog,
        r"long-context files need a larger max_seq_len than standard ones \(long 32768 <= standard 32768\)",
    )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda config: config["peft_config"].update(peft_type="LORA"),
            r"submit --dry-run fails: sub_job_configs\[0\]\.training_config\.peft_config\.peft_type must be 'Lora'",
        ),
        (lambda config: config.pop("optimizer"), "optimizer"),
        (lambda config: config.update(train_batch_size=0), "training_config.train_batch_size must be a positive integer"),
        (lambda config: config["optimizer"].update(scheduler="cosine"), "unsupported field"),
    ],
)
def test_each_file_is_a_valid_job_definition(catalog, change, match):
    _edit_config(catalog / "qwen3-8b/standard/h200-training-lora.json", change)

    _invalid(catalog, match)


def test_sub_jobs_use_only_documented_fields(catalog):
    _edit(
        catalog / "qwen3-8b/standard/h200-training-full.json",
        lambda body: body["sub_job_configs"][0].update(priority="high"),
    )

    _invalid(catalog, r"unknown sub-job keys \['priority'\]")


def test_catalog_files_use_only_hardware_and_sub_jobs(catalog):
    _edit(catalog / "qwen3-8b/standard/h200-training-full.json", lambda body: body.update(experiment_name="x"))

    _invalid(catalog, r"are not catalog fields: \['experiment_name'\]")


def test_files_hold_one_sub_job_of_their_kind(catalog):
    inference = _body("Qwen/Qwen3-8B", "standard/h200-inference-full")["sub_job_configs"][0]
    _edit(
        catalog / "qwen3-8b/standard/h200-training-full.json",
        lambda body: body["sub_job_configs"].append(inference),
    )

    _invalid(catalog, "a training file holds exactly one training sub-job with a training_config")


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda dates: dates.update({"standard/h200-training-full": "September 2, 2026"}),
            "must be an ISO date or null",
        ),
        (
            lambda dates: dates.update({"long/h200-training-full": None}),
            "last_validated names long/h200-training-full, which has no file",
        ),
        (
            lambda dates: dates.pop("standard/h200-training-full"),
            "last_validated is missing standard/h200-training-full",
        ),
    ],
)
def test_validation_dates_cover_exactly_the_model_files(catalog, change, match):
    _edit(catalog / "qwen3-8b/model.json", lambda info: change(info["last_validated"]))

    _invalid(catalog, match)


def test_license_links_use_https(catalog):
    _edit(catalog / "qwen3-8b/model.json", lambda info: info["license"].update(url="http://example.com/LICENSE"))

    _invalid(catalog, "license.url must be an https URL")


def test_model_folders_hold_only_model_json_and_contexts(catalog):
    (catalog / "qwen3-8b/README.md").write_text("notes\n")

    _invalid(catalog, "qwen3-8b/README.md: unexpected entry")


def test_every_problem_is_reported(catalog):
    _edit(catalog / "qwen3-8b/standard/h200-training-full.json", lambda body: body.update(hardware="B300"))
    _edit(catalog / "qwen3.6-35b-a3b/model.json", lambda info: info.pop("moe"))

    message = _invalid(catalog, "MoE models need moe geometry")
    assert "hardware must be 'H200'" in message


def test_rl_jobs_must_build_with_make_config(catalog):
    _edit_config(
        catalog / "qwen3.6-35b-a3b/long/h200-inference-full.json",
        lambda config: config["vllm_config"].update(pipeline_parallel_size=3),
    )

    _invalid(
        catalog,
        r"make_config\(model='Qwen/Qwen3.6-35B-A3B', context='long', hardware='h200', lora=False, rl=True\) "
        "fails: sampling_gpus must be divisible",
    )


def test_rl_jobs_built_with_make_config_must_be_valid(catalog, monkeypatch):
    utils = sys.modules["validate_catalog"].examples_utils()
    make_config = utils.make_config

    def without_hardware(*args, **kwargs):
        body = make_config(*args, **kwargs)
        body.pop("hardware")
        return body

    monkeypatch.setattr(utils, "make_config", without_hardware)

    _invalid(catalog, r"rl=True\) builds an invalid job: hardware must be one of")


def test_job_errors_accepts_catalog_files():
    assert job_errors(_body("Qwen/Qwen3-8B", "standard/h200-training-lora")) == []
    assert job_errors(_body("zai-org/GLM-5.3", "long/h200-inference-full")) == []


# ─── The smoke test ──────────────────────────────────────────────────────────


def test_smoke_request_is_the_catalog_file():
    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3.8-27B", "inference-full", context="long")

    assert request == _body("Qwen/Qwen3.8-27B", "long/h200-inference-full")


def test_smoke_request_names_the_files_that_exist():
    # Qwen3.5-9B's long tier has no LoRA files, so a long LoRA request names the files it has.
    with pytest.raises(ValueError, match="has no long/h200-inference-lora.json; its catalog files are: long/"):
        build_catalog_request(CATALOG_DIR, "Qwen/Qwen3.5-9B", "inference-lora", context="long")


def test_smoke_request_rejects_a_mismatched_model_namespace():
    # The slug is only the lowercased last segment, so a wrong namespace resolves
    # the same folder; the request must match model.json's model_id exactly.
    with pytest.raises(ValueError, match=r"unknown catalog model: NotQwen/Qwen3-8B.*holds 'Qwen/Qwen3-8B'"):
        build_catalog_request(CATALOG_DIR, "NotQwen/Qwen3-8B", "training-full")


def test_forward_backward_probe_matches_catalog_training_batch_size():
    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3.6-35B-A3B", "training-lora")

    spec = build_forward_backward_probe_spec(request, "training-lora")
    payload = spec["payload"]

    assert len(payload["input_ids"]) == 8
    assert payload["input_ids"][0] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert payload["position_ids"] == "arange"
    assert payload["labels"] == {
        "strategy": "next_token",
        "mask_padding": False,
    }


def test_full_context_training_probe_uses_exact_declared_limit():
    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3-0.6B", "training-lora")

    spec = build_forward_backward_probe_spec(
        request,
        "training-lora",
        full_context=True,
    )

    assert len(spec["payload"]["input_ids"]) == 8
    assert len(spec["payload"]["input_ids"][0]) == 32768
    assert set(spec["payload"]["input_ids"][0]) == {1}


@pytest.mark.parametrize("job", ["training-lora", "training-full"])
@pytest.mark.parametrize("context", ["standard", "long"])
def test_qwen38_sft_uses_chunked_logprob_loss(job, context):
    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3.8-27B", job, context=context)
    training = request["sub_job_configs"][0]["training_config"]

    # Each sequence is split across eight GPUs. 256K full fine-tuning needs
    # four of those groups; a larger sequence-parallel degree does not divide
    # the attention heads.
    assert training["sp_size"] == 8
    if context == "long" and job == "training-full":
        assert (training["n_gpus"], training["train_batch_size"]) == (32, 4)
    else:
        assert (training["n_gpus"], training["train_batch_size"]) == (8, 1)
    assert training["fused_lm_head_token_chunk_size"] == 8192

    spec = build_forward_backward_probe_spec(
        request,
        job,
        full_context=True,
    )
    assert spec["payload"]["labels"] == "none"
    assert spec["processing"] == {
        "loss_fn": "causal_cross_entropy",
        "post": ["compute_logprobs"],
        "config": {},
    }
    assert spec["context"]["loss_mask"]["data"][0][-2:] == [1.0, 0.0]


def test_smoke_rl_request_combines_the_training_and_inference_files():
    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3-0.6B", "rl-lora")

    assert request == {
        "hardware": "H200",
        "sub_job_configs": [
            _body("Qwen/Qwen3-0.6B", "standard/h200-training-lora")["sub_job_configs"][0],
            _body("Qwen/Qwen3-0.6B", "standard/h200-inference-lora")["sub_job_configs"][0],
        ],
    }


def test_smoke_rl_request_omits_router_replay_even_for_an_moe_model():
    # The probe does not run the replay handshake, so a strict-replay job would
    # fail preflight; build_catalog_request leaves router replay off.
    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3.6-35B-A3B", "rl-lora", context="long")

    assert "router_replay" not in json.dumps(request)


def test_rl_forward_backward_probe_uses_packing_aware_grpo_contract():
    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3-0.6B", "rl-full")

    spec = build_forward_backward_probe_spec(request, "rl-full")
    payload = spec["payload"]

    assert len(payload["input_ids"]) == 8
    assert payload["include_attention_mask"] is True
    assert payload["labels"] == "none"
    assert spec["processing"] == {
        "loss_fn": "grpo",
        "post": ["compute_logprobs"],
        "config": {
            "eps_clip": 0.2,
            "loss_agg_mode": "token-mean",
            "entropy_coeff": 0.0,
            "global_batch_size": 8,
        },
    }
    assert spec["context"]["advantages"]["dtype"] == "float32"
    assert spec["context"]["loss_mask"]["data"][0] == [
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
    ]


def test_training_memory_telemetry_is_enabled_only_for_training_sub_jobs():
    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3-0.6B", "rl-lora")

    enable_training_memory_telemetry(request)

    training = request["sub_job_configs"][0]["training_config"]
    sampling = request["sub_job_configs"][1]["inference_config"]
    assert training["step_peak_memory_log"] is True
    assert training["training_memory_telemetry"] is True
    assert "step_peak_memory_log" not in sampling
    assert "training_memory_telemetry" not in sampling


def test_debug_image_tag_is_added_to_create_job_body(monkeypatch):
    monkeypatch.delenv(smoke_catalog.DEBUG_OPTIONS_ENV, raising=False)
    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3.8-27B", "training-full")

    apply_debug_image_tag(request, "dev_test")

    assert request["debug"] == {"job": {"image_tag": "dev_test"}}
    assert os.environ[smoke_catalog.DEBUG_OPTIONS_ENV] == "1"


def test_training_sequence_parallelism_adjusts_logical_dp_batch_only():
    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3.6-35B-A3B", "training-full", context="long")

    apply_training_sequence_parallelism(request, 4)

    training = request["sub_job_configs"][0]["training_config"]
    assert training["n_gpus"] == 8
    assert training["sp_size"] == 4
    assert training["train_batch_size"] == 2
    assert training["ds_config"]["train_batch_size"] == 2
    assert training["ds_config"]["train_micro_batch_size_per_gpu"] == 1
    assert training["ds_config"]["gradient_accumulation_steps"] == 1
    assert training["max_seq_len"] == 262144


@pytest.mark.parametrize(
    ("job", "expected_operations"),
    [
        ("inference-full", ["generate"]),
        ("training-lora", ["forward-backward"]),
        ("rl-full", ["generate", "forward-backward"]),
    ],
)
def test_live_smoke_executes_workflow_data_plane_probes(
    monkeypatch,
    job,
    expected_operations,
):
    class FakeClient:
        def __init__(self):
            self.calls = []

        def generate(self, job_id, prompts, sampling_params):
            self.calls.append(("generate", job_id, prompts, sampling_params))
            return "generate-request"

        def forward_backward(self, job_id, payload):
            self.calls.append(("forward-backward", job_id, payload))
            return "forward-backward-request"

        def poll_request(self, job_id, request_id):
            self.calls.append(("poll", job_id, request_id))
            if request_id == "generate-request":
                return {"results": [{"text": "OK"}]}
            return {"avg_loss": 0.25}

    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3-0.6B", job)
    monkeypatch.setattr(
        smoke_catalog,
        "_build_forward_backward_probe_payload",
        lambda *_: b"forward-backward-payload",
    )
    client = FakeClient()

    probes = run_data_plane_probes(client, "job-1", job, request)

    assert [probe["operation"] for probe in probes] == expected_operations
    if "generate" in expected_operations:
        assert (
            "generate",
            "job-1",
            ["Reply with OK."],
            {"max_tokens": 1, "temperature": 0.0},
        ) in client.calls
        assert ("poll", "job-1", "generate-request") in client.calls
    if "forward-backward" in expected_operations:
        assert (
            "forward-backward",
            "job-1",
            b"forward-backward-payload",
        ) in client.calls
        assert ("poll", "job-1", "forward-backward-request") in client.calls


def test_training_step_and_checkpoint_round_trip_follow_forward_backward(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = []

        def forward_backward(self, job_id, payload):
            self.calls.append(("forward-backward", job_id, payload))
            return "forward-backward-request"

        def step(self, job_id, learning_rate=None):
            self.calls.append(("step", job_id, learning_rate))
            return "step-request"

        def save(self, job_id, checkpoint_type=None):
            self.calls.append(("save", job_id, checkpoint_type))
            return "save-request"

        def delete_checkpoint(self, job_id, checkpoint_id):
            self.calls.append(("delete-checkpoint", job_id, checkpoint_id))

        def load(
            self,
            job_id,
            checkpoint_id,
            source_job_id=None,
            *,
            target_sub_job_id=None,
        ):
            self.calls.append(
                (
                    "load",
                    job_id,
                    checkpoint_id,
                    source_job_id,
                    target_sub_job_id,
                )
            )
            return "load-request"

        def poll_request(self, job_id, request_id):
            self.calls.append(("poll", job_id, request_id))
            return {
                "forward-backward-request": {"avg_loss": 0.25},
                "step-request": {"global_steps": 1},
                "save-request": {
                    "stage_path": "s3://bucket/checkpoints/cp_test/global_step1/"
                },
                "load-request": {"checkpoint_id": "cp_test"},
            }[request_id]

    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3-0.6B", "training-lora")
    monkeypatch.setattr(
        smoke_catalog,
        "_build_forward_backward_probe_payload",
        lambda *_: b"forward-backward-payload",
    )
    client = FakeClient()

    probes = run_data_plane_probes(
        client,
        "job-1",
        "training-lora",
        request,
        training_step=True,
        checkpoint_round_trip=True,
    )

    assert [probe["operation"] for probe in probes] == [
        "forward-backward",
        "step",
        "save",
        "load",
    ]
    assert ("step", "job-1", 0.00002) in client.calls
    assert ("save", "job-1", "resumable") in client.calls
    assert (
        "load",
        "job-1",
        "cp_test",
        "job-1",
        "job-1:training:0",
    ) in client.calls
    assert ("delete-checkpoint", "job-1", "cp_test") in client.calls


def test_live_smoke_rejects_non_finite_forward_backward_loss(monkeypatch):
    class FakeClient:
        def forward_backward(self, job_id, payload):
            return "forward-backward-request"

        def poll_request(self, job_id, request_id):
            return {"avg_loss": float("nan")}

    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3-0.6B", "training-full")
    monkeypatch.setattr(
        smoke_catalog,
        "_build_forward_backward_probe_payload",
        lambda *_: b"forward-backward-payload",
    )

    with pytest.raises(RuntimeError, match="non-finite avg_loss"):
        run_data_plane_probes(FakeClient(), "job-1", "training-full", request)


def test_full_context_inference_probe_prefills_to_one_token_below_limit():
    class FakeClient:
        def __init__(self):
            self.prompts = None

        def generate(self, job_id, prompts, sampling_params):
            self.prompts = prompts
            assert sampling_params == {"max_tokens": 1, "temperature": 0.0}
            return "generate-request"

        def poll_request(self, job_id, request_id):
            return {"results": [{"text": "OK"}]}

    request = build_catalog_request(CATALOG_DIR, "Qwen/Qwen3-0.6B", "inference-full")
    client = FakeClient()

    probes = run_data_plane_probes(
        client,
        "job-1",
        "inference-full",
        request,
        full_context_prefill=True,
    )

    assert client.prompts is not None
    assert len(client.prompts) == 1
    assert len(client.prompts[0]) == 32767
    assert set(client.prompts[0]) == {1}
    assert probes == [
        {
            "operation": "generate",
            "requestId": "generate-request",
            "resultCount": 1,
            "promptTokens": 32767,
        }
    ]
