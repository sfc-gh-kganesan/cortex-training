"""MATH task entrypoint for the shared GRPO runner."""

import chz
from recipes.rl.tasks.math import FORMAT_COEF
from recipes.rl.tasks.math import MathAccuracyEvaluator
from recipes.rl.tasks.math import MathProblems
from recipes.rl.tasks.math import MathTask
from recipes.rl.tasks.math import _stopped_cleanly
from recipes.rl.tasks.math import build_prompt
from recipes.rl.tasks.math import convo_prefix
from recipes.rl.tasks.math import load_math
from recipes.rl.tasks.math import question_suffix
from recipes.rl.tasks.math import score_response
from recipes.rl.train import _should_eval
from recipes.rl.train import _train
from recipes.rl.train import job_body
from recipes.rl.train import processing_block
from recipes.rl.train import train


@chz.chz
class Config:
    config: str
    job_id: str | None = None

    problems_per_batch: int = 64
    group_size: int = 16
    max_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    format_coef: float = FORMAT_COEF

    max_steps: int = 10
    eps_clip: float = 0.2
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0.0
    remove_constant_reward_groups: bool = True

    debug_image_tag: str | None = None
    eval_every: int = 10
    n_test: int | None = None
    eval_temperature: float | None = None
    eval_max_tokens: int | None = None
    weight_sync_bucket_size: int | None = None

    log_path: str = "/tmp/cortex-training-examples/rl-loop"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False

    job_config: str = "configs/qwen3_8b_lora.json"


def main(config: Config) -> None:
    train(config, MathTask(format_coef=config.format_coef))


__all__ = [
    "FORMAT_COEF",
    "MathAccuracyEvaluator",
    "MathProblems",
    "MathTask",
    "_should_eval",
    "_stopped_cleanly",
    "_train",
    "Config",
    "build_prompt",
    "convo_prefix",
    "job_body",
    "load_math",
    "main",
    "processing_block",
    "question_suffix",
    "score_response",
    "train",
]

if __name__ == "__main__":
    chz.nested_entrypoint(main)
