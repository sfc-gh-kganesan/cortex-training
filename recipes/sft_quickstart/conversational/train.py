"""Conversational task entrypoint for the shared SFT runner."""

from typing import Any

import chz
from recipes.sft_quickstart.conversational.task import ConversationalTask
from recipes.sft_quickstart.runner import train
from tinker_cookbook import renderers


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
    # None keeps the job's value.
    lr: float | None = None
    train_batch_size: int | None = None
    max_seq_len: int | None = None

    dataset: str = "who_trained_you"
    dataset_split: str = "train"
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    pad_to_max_length: bool = False
    max_steps: int = 100

    debug_image_tag: str | None = None
    enable_thinking: bool = False
    renderer_name: str | None = None

    log_path: str = "/tmp/cortex-training-recipes/sft-loop"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def main(config: Config) -> dict[str, Any]:
    task = ConversationalTask(
        dataset=config.dataset,
        dataset_split=config.dataset_split,
        train_on_what=config.train_on_what,
    )
    return train(config, task)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
