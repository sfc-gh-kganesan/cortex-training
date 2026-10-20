# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Score identity SFT completions.

The default check is a case-sensitive substring match on
``Snowflake AI Research``. Use this module from ``evaluate`` (``task=identity``)
or on a completions JSONL from ``generate``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chz

logger = logging.getLogger(__name__)

IDENTITY_PHRASE = "Snowflake AI Research"
DEFAULT_PROMPTS_FILE = (
    Path(__file__).resolve().parents[1]
    / "sft"
    / "conversational"
    / "data"
    / "identity_eval.jsonl"
)


def contains_identity(text: str, phrase: str = IDENTITY_PHRASE) -> bool:
    return phrase in (text or "")


def load_identity_prompts(
    path: str | Path | None = None,
    *,
    max_examples: int | None = None,
) -> list[dict[str, str]]:
    source = Path(path).expanduser() if path else DEFAULT_PROMPTS_FILE
    rows: list[dict[str, str]] = []
    for index, line in enumerate(source.read_text().splitlines()):
        if not line.strip():
            continue
        obj = json.loads(line)
        prompt = obj.get("prompt")
        if not prompt and isinstance(obj.get("messages"), list):
            prompt = obj["messages"][0].get("content")
        if not prompt:
            raise ValueError(f"{source}:{index + 1} needs a prompt or messages[0].content")
        rows.append(
            {
                "id": str(obj.get("id") or f"identity-eval-{index + 1:02d}"),
                "category": str(obj.get("category") or ""),
                "split": str(obj.get("split") or ""),
                "prompt": str(prompt),
            }
        )
        if max_examples is not None and len(rows) >= max_examples:
            break
    if not rows:
        raise ValueError(f"{source} has no identity prompts")
    return rows


def load_identity_completions(path: str | Path) -> list[dict[str, str]]:
    source = Path(path).expanduser()
    rows: list[dict[str, str]] = []
    for index, line in enumerate(source.read_text().splitlines()):
        if not line.strip():
            continue
        obj = json.loads(line)
        completion = obj.get("completion")
        if completion is None:
            raise ValueError(f"{source}:{index + 1} needs a completion field")
        rows.append(
            {
                "id": str(obj.get("id") or f"identity-eval-{index + 1:02d}"),
                "category": str(obj.get("category") or ""),
                "split": str(obj.get("split") or ""),
                "prompt": str(obj.get("prompt") or ""),
                "completion": str(completion),
            }
        )
    if not rows:
        raise ValueError(f"{source} has no completions")
    return rows


def score_identity_completions(
    rows: list[dict[str, Any]],
    *,
    phrase: str = IDENTITY_PHRASE,
) -> dict[str, float]:
    if not rows:
        raise ValueError("no identity completions to score")
    n_hit = sum(1 for row in rows if contains_identity(str(row.get("completion") or ""), phrase))
    n = len(rows)
    score = n_hit / n
    score_pct = 100.0 * score
    metrics = {
        "identity/correct": score,
        "identity/score_pct": score_pct,
        "identity/num_examples": float(n),
        "test/env/all/correct": score,
        "test/env/all/num_examples": float(n),
    }
    logger.info(
        "identity: %.1f%% (%d/%d contain %r)",
        score_pct,
        n_hit,
        n,
        phrase,
    )
    logger.info("Results: %s", metrics)
    return metrics


@chz.chz
class Config:
    completions_file: str
    phrase: str = IDENTITY_PHRASE


def main(config: Config) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = load_identity_completions(config.completions_file)
    metrics = score_identity_completions(rows, phrase=config.phrase)
    print(f"{metrics['identity/score_pct']:.1f}")
    return metrics


if __name__ == "__main__":
    chz.nested_entrypoint(main)
