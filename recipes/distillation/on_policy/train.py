"""GSM8K entrypoint for the shared on-policy distillation runner."""

import chz
from recipes.distillation.tasks.gsm8k import BOXED_INSTRUCTION
from recipes.distillation.tasks.gsm8k import GSM8KProblems
from recipes.distillation.tasks.gsm8k import GSM8KTask
from recipes.distillation.tasks.gsm8k import extract_answer
from recipes.distillation.tasks.gsm8k import is_correct
from recipes.distillation.tasks.gsm8k import load_gsm8k
from recipes.distillation.train import _train
from recipes.distillation.train import job_bodies
from recipes.distillation.train import lr_at
from recipes.distillation.train import processing_block
from recipes.distillation.train import train
from recipes.distillation.train import validate_config


@chz.chz
class Config:
    config: str
    job_id: str | None = None
    teacher_job_id: str | None = None
    job_config: str = "configs/qwen3_0p6b_student.json"
    teacher_job_config: str = "configs/qwen3_8b_teacher.json"
    renderer_name: str | None = "qwen3_disable_thinking"

    problems_per_batch: int = 16
    group_size: int = 4
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0

    max_steps: int | None = 100
    warmup_steps: int = 5
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.1

    teacher_tau: float = 1.0
    teacher_clip: float = 3.0
    teacher_clip_negative: float | None = None
    reward_coef: float = 0.0
    eps_clip: float = 0.2
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0.0

    eval_every: int = 10
    n_val: int = 200
    n_test: int | None = None
    eval_max_tokens: int | None = None
    n_kl_eval: int = 64
    evaluate_teacher: bool = True
    seed: int = 0

    weight_sync_bucket_size: int | None = None
    debug_image_tag: str | None = None
    log_path: str = "/tmp/cortex-training-examples/on-policy-distillation"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def main(config: Config) -> None:
    train(config, GSM8KTask())


__all__ = [
    "BOXED_INSTRUCTION",
    "Config",
    "GSM8KProblems",
    "GSM8KTask",
    "_train",
    "extract_answer",
    "is_correct",
    "job_bodies",
    "load_gsm8k",
    "lr_at",
    "main",
    "processing_block",
    "train",
    "validate_config",
]

if __name__ == "__main__":
    chz.nested_entrypoint(main)
