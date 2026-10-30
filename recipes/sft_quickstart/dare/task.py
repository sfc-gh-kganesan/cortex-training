"""DARE-Bench data-science agent task adapter for the shared SFT runner.

DARE-Bench (Shu et al., ICLR 2026) asks an agent to solve Kaggle-derived
classification, regression, and time-series tasks by writing Python that runs
in a sandbox. The ``Snowflake/dare-bench`` dataset ships rejection-sampled
multi-turn traces (``sft_data_{fv,av,bv,dv}.json``) in the OpenAI chat format.

Licenses follow upstream. The ``databases/`` assets (each task's source data,
metadata, and ground truth) keep their original Kaggle or source licenses,
which include share-alike and non-commercial terms such as CC BY-SA 4.0 and
CC BY-NC-SA 4.0; Snowflake-Labs/dare-bench lists them per dataset under
``LICENSE/``. Everything else, including the traces and the harness code, is
Apache-2.0. This adapter downloads the traces at a pinned revision and
redistributes no data.

The tool schema comes from DARE-Bench's own ``python_executor``, vendored in
``recipes/sft_quickstart/dare/dare_bench``, so training renders the same system turn the
evaluator sends.
"""

from __future__ import annotations

import hashlib
import json
import math
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
from recipes.sft_quickstart.dare.dare_bench.tools.python_executor.python_executor import python_executor
from recipes.utils import sequence_from_conversation

HF_DATASET = "Snowflake/dare-bench"
HF_REVISION = "7f202a10a55121e520396e48e0ee2a8ced6c026d"
SFT_VARIANTS = ("fv", "av", "bv", "dv")

TOOL_NAME = "python_executor"
MAX_TOOL_TURNS = 5
EXECUTOR_TIMEOUT_S = 200
MAX_FILE_SIZE_MB = 200
PREDICTION_FILE = "prediction.csv"


class _ToolConfig(dict):
    """The attribute and ``get`` access the vendored tool expects of its OmegaConf config."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def executor_config(
    *,
    url: str = "",
    work_dir: Path | None = None,
    timeout: int = EXECUTOR_TIMEOUT_S,
    max_turns: int = MAX_TOOL_TURNS,
) -> _ToolConfig:
    return _ToolConfig(
        max_turn=max_turns,
        cache_dir=str(work_dir) if work_dir is not None else None,
        tool=_ToolConfig(
            python_executor={"url": url, "timeout": timeout, "max_file_size_mb": MAX_FILE_SIZE_MB},
        ),
    )


def tool_spec(*, timeout: int = EXECUTOR_TIMEOUT_S, max_turns: int = MAX_TOOL_TURNS) -> dict[str, Any]:
    config = executor_config(timeout=timeout, max_turns=max_turns)
    # The model name only selects the schema wrapper; any Qwen name gives the
    # chat-completions one, whose inner function the renderer re-wraps.
    return python_executor(config={}).get_tool_description("qwen", config)["function"]


def tool_prefix(
    renderer,
    *,
    timeout: int = EXECUTOR_TIMEOUT_S,
    max_turns: int = MAX_TOOL_TURNS,
) -> list[dict[str, Any]]:
    return renderer.create_conversation_prefix_with_tools([tool_spec(timeout=timeout, max_turns=max_turns)])


def _hub_file(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_DATASET, filename, repo_type="dataset", revision=HF_REVISION))


def load_sft_traces(variant: str = "fv") -> list[dict[str, Any]]:
    if variant not in SFT_VARIANTS:
        raise ValueError(f"variant must be one of {SFT_VARIANTS}, got {variant!r}")
    return json.loads(_hub_file(f"sft_data/sft_data_{variant}.json").read_text(encoding="utf-8"))


def assistant_content(reasoning: str | None, text: str, *, has_tool_calls: bool) -> str:
    """Assistant text in the form the Qwen3 chat template writes inside a tool loop.

    The template keeps the reasoning of every turn after the last user message
    and puts exactly one newline between the text and the first tool call; a
    turn with no text after ``</think>`` therefore ends in a single newline so
    the renderer's separator completes ``</think>\\n\\n<tool_call>``.
    """
    text = text.lstrip("\n") if reasoning is not None else text
    if reasoning is None:
        return text
    block = "<think>\n" + reasoning.strip("\n") + "\n</think>\n"
    if has_tool_calls and not text:
        return block
    return block + "\n" + text


def split_reasoning(content: str) -> tuple[str | None, str]:
    if "</think>" not in content:
        return None, content
    reasoning, _, rest = content.partition("</think>")
    return reasoning.rpartition("<think>")[2], rest


def trace_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert one OpenAI-format trace into renderer messages."""
    from tinker_cookbook.renderers.base import ToolCall

    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message.get("content") or ""
        if role == "assistant":
            calls = message.get("tool_calls") or []
            reasoning, text = split_reasoning(content)
            out: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content(reasoning, text, has_tool_calls=bool(calls)),
            }
            if calls:
                out["tool_calls"] = [
                    ToolCall(
                        id=call.get("id"),
                        function=ToolCall.FunctionBody(
                            name=call["function"]["name"], arguments=call["function"]["arguments"]
                        ),
                    )
                    for call in calls
                ]
            converted.append(out)
        elif role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": message.get("tool_call_id"),
                    "name": message.get("name") or TOOL_NAME,
                }
            )
        else:
            converted.append({"role": role, "content": content})
    return converted


def is_heldout(task_id: str, fraction: float) -> bool:
    """Stable task-level split: every trace of a held-out task stays out of training."""
    bucket = int.from_bytes(hashlib.sha256(task_id.encode("utf-8")).digest()[:8], "big") % 10_000
    return bucket < round(fraction * 10_000)


@dataclass(frozen=True)
class DareTask:
    variant: str = "fv"
    heldout_fraction: float = 0.05
    name: str = "dare"

    def _rows(self, *, heldout: bool) -> list[dict[str, str]]:
        return [
            {"id": trace["id"], "messages": json.dumps(trace["messages"])}
            for trace in load_sft_traces(self.variant)
            if trace.get("success")
            and not trace.get("skipped_due_to_filter")
            and is_heldout(trace["id"], self.heldout_fraction) == heldout
        ]

    def load_dataset(self, *, n_train: int) -> datasets.Dataset:
        """``n_train`` rows in epoch order, each epoch its own shuffle of the training traces.

        Shuffling once after tiling would mix the passes, so the first epoch's
        worth of steps would see some traces twice and others not at all.
        """
        rows = datasets.Dataset.from_list(self._rows(heldout=False))
        if len(rows) == 0:
            raise ValueError("no training traces left after the held-out split")
        if n_train <= 0:
            return rows.shuffle(seed=0)
        epochs = [rows.shuffle(seed=epoch) for epoch in range(math.ceil(n_train / len(rows)))]
        return datasets.concatenate_datasets(epochs).select(range(n_train))

    def load_eval_dataset(self) -> datasets.Dataset | None:
        rows = self._rows(heldout=True)
        return datasets.Dataset.from_list(rows) if rows else None

    def build_sequence(self, row, renderer, *, max_seq_len: int):
        from tinker_cookbook import renderers

        return sequence_from_conversation(
            tool_prefix(renderer) + trace_messages(json.loads(row["messages"])),
            renderer,
            train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            max_seq_len=max_seq_len,
        )

    def sample_prompt(self) -> str | None:
        return None

    def eval_command(
        self,
        *,
        connection_args: str,
        job_id: str,
        checkpoint_id: str,
        job_args: Mapping[str, Any] | None = None,
    ) -> str:
        # Repeat the catalog selectors (model=/context=/hardware=/lora=, or
        # job_config=) so the evaluator picks the trained model and its renderer
        # rather than its own defaults.
        selectors = "".join(
            f" {name}={str(value).lower() if isinstance(value, bool) else shlex.quote(str(value))}"
            for name, value in (job_args or {}).items()
        )
        command = " ".join(
            part
            for part in (
                "python -m recipes.sft_quickstart.dare.evaluate",
                connection_args,
                "bench_dir=$BENCH_DIR",
                "sandbox_python=$PWD/dare_sandbox_env/bin/python",
                f"source_job_id={job_id}",
                f"checkpoint_id={checkpoint_id}",
                "log_path=/tmp/dare/sft",
            )
            if part
        )
        return command + selectors
