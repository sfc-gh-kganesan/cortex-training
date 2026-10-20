"""Conversational task entrypoint for the shared SFT runner."""

from typing import Any

import chz
from recipes.sft.tasks.conversational import ConversationalTask
from recipes.sft.tasks.conversational import BUILTIN_CHAT_DATASETS
from recipes.sft.tasks.conversational import WHO_TRAINED_YOU_PROMPT
from recipes.sft.tasks.conversational import _is_local_chat_file
from recipes.sft.tasks.conversational import is_who_trained_you_dataset
from recipes.sft.tasks.conversational import load_chat_dataset
from recipes.sft.tasks.conversational import resolve_chat_dataset
from recipes.sft.tasks.conversational import tile_rows
from recipes.sft.train import causal_cross_entropy_processing
from recipes.sft.train import job_body
from recipes.sft.train import train
from tinker_cookbook import renderers


@chz.chz
class Config:
    config: str
    job_id: str | None = None

    dataset: str = "who_trained_you"
    dataset_split: str = "train"
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    pad_to_max_length: bool = False
    max_steps: int = 100

    debug_image_tag: str | None = None
    enable_thinking: bool = False
    renderer_name: str | None = None

    log_path: str = "/tmp/cortex-training-examples/sft-loop"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False

    job_config: str = "configs/qwen3_8b_full.json"


def main(config: Config) -> dict[str, Any]:
    task = ConversationalTask(
        dataset=config.dataset,
        dataset_split=config.dataset_split,
        train_on_what=config.train_on_what,
    )
    return train(config, task)


__all__ = [
    "BUILTIN_CHAT_DATASETS",
    "WHO_TRAINED_YOU_PROMPT",
    "_is_local_chat_file",
    "causal_cross_entropy_processing",
    "Config",
    "ConversationalTask",
    "is_who_trained_you_dataset",
    "job_body",
    "load_chat_dataset",
    "main",
    "resolve_chat_dataset",
    "tile_rows",
    "train",
]

if __name__ == "__main__":
    chz.nested_entrypoint(main)
