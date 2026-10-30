"""GSM8K dataset, prompting, and answer grading for the distillation example."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any

from recipes.inference.prompts import render_user_prompt

BOXED_INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class GSM8KProblems:
    train: list[tuple[str, str]]
    val: list[tuple[str, str]]
    test: list[tuple[str, str]]


def parse_gold(answer_text: str) -> str | None:
    """The reference answer after GSM8K's ``####`` marker."""
    if "####" not in answer_text:
        return None
    match = _NUMBER.search(answer_text.split("####")[-1].replace(",", ""))
    return match.group(0) if match else None


def boxed_contents(text: str) -> list[str]:
    """Contents of every ``\\boxed{...}``, matching nested braces.

    ``\\boxed{\\frac{1}{2}}`` yields ``\\frac{1}{2}``.
    """
    contents = []
    start = text.find("\\boxed{")
    while start != -1:
        depth, index = 1, start + len("\\boxed{")
        while index < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[index], 0)
            index += 1
        if depth == 0:
            contents.append(text[start + len("\\boxed{") : index - 1])
        start = text.find("\\boxed{", index)
    return contents


def extract_answer(text: str) -> str | None:
    """The number in the last ``\\boxed{}``, else the last number in the text."""
    for content in reversed(boxed_contents(text)):
        match = _NUMBER.search(content.replace(",", ""))
        if match:
            return match.group(0)
    numbers = _NUMBER.findall(text.replace(",", ""))
    return numbers[-1] if numbers else None


def is_correct(text: str, gold: str | None) -> bool:
    predicted = extract_answer(text)
    if predicted is None or gold is None:
        return False
    return math.isclose(float(predicted), float(gold), rel_tol=0.0, abs_tol=1e-6)


def load_gsm8k(*, seed: int, n_val: int) -> GSM8KProblems:
    """``openai/gsm8k`` train (minus ``n_val`` held-out problems) and test splits."""
    from datasets import load_dataset

    def rows(split: str) -> list[tuple[str, str]]:
        problems = []
        for row in load_dataset("openai/gsm8k", "main", split=split):
            gold = parse_gold(str(row["answer"]))
            if gold is not None:
                problems.append((str(row["question"]), gold))
        return problems

    train = rows("train")
    random.Random(seed).shuffle(train)
    return GSM8KProblems(train=train[n_val:], val=train[:n_val], test=rows("test"))


@dataclass(frozen=True)
class GSM8KTask:
    name: str = "GSM8K"
    evaluation_name: str = "GSM8K test"

    def load(self, *, seed: int, n_val: int) -> GSM8KProblems:
        return load_gsm8k(seed=seed, n_val=n_val)

    def build_prompt(self, question: str, renderer: Any) -> list[int]:
        return render_user_prompt(renderer, question + BOXED_INSTRUCTION)

    def score_response(
        self, response: str, answer: str
    ) -> tuple[float, dict[str, float]]:
        correct = float(is_correct(response, answer))
        return correct, {
            "correct": correct,
            "boxed": float(bool(boxed_contents(response))),
        }
