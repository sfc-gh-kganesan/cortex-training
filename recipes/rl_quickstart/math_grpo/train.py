"""MATH task entrypoint for the shared GRPO runner."""

import chz
from recipes.rl_quickstart.math_grpo.task import FORMAT_COEF
from recipes.rl_quickstart.math_grpo.task import MathTask
from recipes.rl_quickstart.runner import train


@chz.chz
class Config:
    # Legacy JSON connection file. Omit it to use `connection` or the default profile.
    config: str | None = None
    # Snowflake connection profile name (connections.toml). Use config or connection, not both.
    connection: str | None = None
    job_id: str | None = None

    # The job comes from model-catalog/models/; job_config= replaces it with a file.
    model: str = "Qwen/Qwen3-8B"
    context: str = "standard"
    hardware: str = "h200"
    lora: bool = True
    job_config: str | None = None
    # None keeps the job's value.
    lr: float | None = None
    train_batch_size: int | None = None
    max_seq_len: int | None = None

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
    # Full-parameter weight sync format, vllm or hf; None uses the server default.
    # LoRA jobs always sync lora.
    weight_sync_format: str | None = None
    weight_sync_bucket_size: int | None = None

    log_path: str = "/tmp/cortex-training-recipes/rl-loop"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def main(config: Config) -> None:
    train(config, MathTask(format_coef=config.format_coef))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
