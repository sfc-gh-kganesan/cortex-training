"""MATH dataset, prompting, scoring, and evaluation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Any

logger = logging.getLogger(__name__)

# Match MathEnv / ProblemEnv defaults.
FORMAT_COEF = 0.1


@dataclass
class MathProblems:
    train: list[tuple[str, str]]
    test: list[tuple[str, str]] | None

    def __post_init__(self) -> None:
        if len(self.train) == 0:
            raise ValueError("a math dataset needs at least one training problem")


def load_math(seed: int = 0) -> MathProblems:
    """Hendrycks MATH train (MATH-500 held out) + HuggingFaceH4/MATH-500 test."""
    from tinker_cookbook.recipes.math_rl.math_env import _get_hendrycks_math_test
    from tinker_cookbook.recipes.math_rl.math_env import _get_hendrycks_math_train
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

    train_rows = _get_hendrycks_math_train().shuffle(seed=seed)
    train = [(row["problem"], extract_boxed(row["solution"])) for row in train_rows]
    test = [
        (row["problem"], extract_boxed(row["solution"]))
        for row in _get_hendrycks_math_test()
    ]
    return MathProblems(train=train, test=test or None)


def question_suffix() -> str:
    from tinker_cookbook.recipes.math_rl.math_env import MathEnv

    return MathEnv.question_suffix()


def convo_prefix() -> list[dict[str, str]]:
    from tinker_cookbook.recipes.math_rl.math_env import MathEnv

    return list(MathEnv.standard_fewshot_prefix())


def build_prompt(question: str, renderer) -> list[int]:
    conversation = [
        *convo_prefix(),
        {"role": "user", "content": question + question_suffix()},
    ]
    return renderer.build_generation_prompt(conversation).to_ints()


def stopped_cleanly(result: dict, max_tokens: int | None) -> bool:
    finish_reason = result.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        return finish_reason != "length"
    if max_tokens is None:
        return True
    return len(result.get("token_ids") or []) < max_tokens


def score_response(
    response: str,
    answer: str,
    *,
    result: dict,
    max_tokens: int | None,
    format_coef: float = FORMAT_COEF,
) -> tuple[float, dict[str, float]]:
    from tinker_cookbook.recipes.math_rl.math_env import safe_grade
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

    well_formed = stopped_cleanly(result, max_tokens)
    try:
        given = extract_boxed(response)
        format_ok = True
    except ValueError:
        given = None
        format_ok = False

    correct_format = float(well_formed and format_ok)
    correct_answer = 0.0
    if format_ok and given is not None:
        correct_answer = float(safe_grade(given, answer))
    reward = format_coef * (correct_format - 1.0) + correct_answer
    return reward, {"format": correct_format, "correct": correct_answer}


@dataclass
class MathAccuracyEvaluator:
    prompts: list[list[int]]
    answers: list[str]
    sampling_params: dict = field(default_factory=dict)
    format_coef: float = FORMAT_COEF
    name: str = "test/env/all"

    def __post_init__(self) -> None:
        if len(self.prompts) != len(self.answers):
            raise ValueError(
                f"{len(self.prompts)} prompts but {len(self.answers)} answers"
            )

    def __call__(self, client: Any, job_id: str) -> dict[str, float]:
        if len(self.prompts) == 0:
            logger.warning("%s: no held-out problems, skipping", type(self).__name__)
            return {}

        request_id = client.generate(
            job_id,
            prompts=self.prompts,
            sampling_params=self.sampling_params,
        )
        results = client.poll_request(job_id, request_id)["results"]
        if len(results) != len(self.prompts):
            raise RuntimeError(
                f"asked for {len(self.prompts)} completions, got {len(results)}"
            )

        max_tokens = self.sampling_params.get("max_tokens")
        corrects: list[float] = []
        formats: list[float] = []
        rewards: list[float] = []
        completion_lengths: list[int] = []
        n_truncated = 0

        for result, answer in zip(results, self.answers):
            text = result.get("text") or ""
            token_ids = result.get("token_ids") or []
            completion_lengths.append(len(token_ids))
            if not stopped_cleanly(result, max_tokens):
                n_truncated += 1
            reward, metrics = score_response(
                text,
                answer,
                result=result,
                max_tokens=max_tokens,
                format_coef=self.format_coef,
            )
            corrects.append(metrics["correct"])
            formats.append(metrics["format"])
            rewards.append(reward)

        n = len(results)
        return {
            f"{self.name}/correct": sum(corrects) / n,
            f"{self.name}/format": sum(formats) / n,
            f"{self.name}/reward": sum(rewards) / n,
            f"{self.name}/frac_truncated": n_truncated / n,
            f"{self.name}/num_examples": float(n),
            f"{self.name}/mean_completion_tokens": sum(completion_lengths) / n,
        }


@dataclass(frozen=True)
class MathTask:
    format_coef: float = FORMAT_COEF
    name: str = "MATH"
    evaluation_name: str = "MATH-500"

    def load(self, *, seed: int) -> MathProblems:
        return load_math(seed=seed)

    def build_prompt(self, problem: str, renderer) -> list[int]:
        return build_prompt(problem, renderer)

    def score_response(
        self,
        response: str,
        answer: str,
        *,
        result: dict,
        max_tokens: int | None,
    ) -> tuple[float, dict[str, float]]:
        return score_response(
            response,
            answer,
            result=result,
            max_tokens=max_tokens,
            format_coef=self.format_coef,
        )

    def make_evaluator(
        self,
        test_problems: list[tuple[str, str]],
        renderer,
        *,
        sampling_params: dict,
        n_test: int | None,
    ) -> MathAccuracyEvaluator:
        if n_test is not None:
            test_problems = test_problems[:n_test]
        return MathAccuracyEvaluator(
            prompts=[
                self.build_prompt(question, renderer) for question, _ in test_problems
            ],
            answers=[answer for _, answer in test_problems],
            sampling_params=sampling_params,
            format_coef=self.format_coef,
        )
