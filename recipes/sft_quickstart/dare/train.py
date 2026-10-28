"""DARE-Bench agent-trace SFT entrypoint for the shared SFT runner."""

from typing import Any

import chz
from recipes.sft_quickstart.dare.task import DareTask
from recipes.sft_quickstart.runner import train


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
    lora: bool = False
    job_config: str | None = None
    # DARE trains full-parameter at a lower learning rate and a larger batch than
    # the catalog's default; None keeps the catalog value.
    lr: float | None = 1e-05
    train_batch_size: int | None = 32
    max_seq_len: int | None = None

    variant: str = "fv"
    heldout_fraction: float = 0.05
    eval_every: int = 10
    pad_to_max_length: bool = False
    max_steps: int = 120
    lr_schedule: str = "cosine"
    warmup_ratio: float = 0.05

    debug_image_tag: str | None = None
    enable_thinking: bool = True
    renderer_name: str | None = None

    log_path: str = "/tmp/cortex-training-recipes/dare-sft"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def main(config: Config) -> dict[str, Any]:
    return train(config, DareTask(variant=config.variant, heldout_fraction=config.heldout_fraction))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
