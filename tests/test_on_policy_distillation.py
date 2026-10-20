# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for the on-policy distillation recipe."""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from types import SimpleNamespace

import pytest
import torch
from recipes.distillation import teacher as teacher_mod
from recipes.distillation import train as runner
from recipes.distillation.tasks.gsm8k import extract_answer
from recipes.distillation.tasks.gsm8k import is_correct
from recipes.distillation.tasks.gsm8k import parse_gold
from recipes.utils import _forward_backward_payload
from recipes.utils import collate
from recipes.utils import sequence_from_rollout

from cortex_training import wire


@dataclass
class _Config:
    config: str = "unused"
    job_id: str | None = None
    teacher_job_id: str | None = None
    job_config: str = "configs/qwen3_0p6b_student.json"
    teacher_job_config: str = "configs/qwen3_8b_teacher.json"
    renderer_name: str | None = "qwen3_disable_thinking"
    problems_per_batch: int = 2
    group_size: int = 2
    max_tokens: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    max_steps: int | None = 2
    warmup_steps: int = 1
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.1
    teacher_tau: float = 1.0
    teacher_clip: float = 3.0
    teacher_clip_negative: float | None = None
    reward_coef: float = 0.0
    eps_clip: float = 0.2
    loss_agg_mode: str = "token-mean"
    entropy_coeff: float = 0.0
    eval_every: int = 1
    n_val: int = 1
    n_test: int | None = None
    eval_max_tokens: int | None = None
    n_kl_eval: int = 1
    evaluate_teacher: bool = True
    seed: int = 0
    weight_sync_bucket_size: int | None = None
    debug_image_tag: str | None = None
    log_path: str = "/tmp/unused"
    wandb_project: str | None = None
    wandb_name: str | None = None
    sf_tracking: bool = False


def _position(token: int, logprob: float, key=int) -> dict:
    return {key(token): {"logprob": logprob, "rank": 1}}


# --- teacher scoring ----------------------------------------------------------


@pytest.mark.parametrize("key", [int, str])
def test_teacher_completion_logprobs_reads_completion_positions(key):
    full = [11, 12, 13, 14, 15]
    result = {
        "prompt_logprobs": [None] + [_position(t, -0.1 * t, key) for t in full[1:]]
    }

    values = teacher_mod.teacher_completion_logprobs(result, full, prompt_len=3)

    assert values == pytest.approx([-1.4, -1.5])


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({}, "no 'prompt_logprobs'"),
        ({"prompt_logprobs": [None, _position(12, -0.1)]}, "prompt_logprobs for"),
        (
            {
                "prompt_logprobs": [
                    _position(11, -0.1),
                    _position(12, -0.1),
                    _position(13, -0.1),
                ]
            },
            "should be None",
        ),
        (
            {"prompt_logprobs": [None, _position(12, -0.1), _position(99, -0.1)]},
            "missing",
        ),
        (
            {"prompt_logprobs": [None, _position(12, -0.1), _position(13, math.nan)]},
            "not finite",
        ),
        (
            {"prompt_logprobs": [None, _position(12, -0.1), _position(13, 0.5)]},
            "not finite",
        ),
    ],
)
def test_teacher_completion_logprobs_fails_closed(result, message):
    with pytest.raises(ValueError, match=message):
        teacher_mod.teacher_completion_logprobs(result, [11, 12, 13], prompt_len=2)


def test_sampled_token_logprobs_requires_aligned_logprobs():
    result = {
        "token_ids": [5, 6],
        "logprobs": [_position(5, -0.2), _position(6, -0.3, str)],
    }
    assert teacher_mod.sampled_token_logprobs(result) == pytest.approx([-0.2, -0.3])

    with pytest.raises(ValueError, match="logprobs=0"):
        teacher_mod.sampled_token_logprobs({"token_ids": [5]})
    with pytest.raises(ValueError, match="logprob positions"):
        teacher_mod.sampled_token_logprobs(
            {"token_ids": [5, 6], "logprobs": [_position(5, -0.2)]}
        )


def test_score_with_teacher_sends_exact_token_ids():
    calls = {}

    class Client:
        def generate(self, job_id, prompts, sampling_params):
            calls.update(job_id=job_id, prompts=prompts, params=sampling_params)
            return "req"

        def poll_request(self, job_id, request_id):
            return {
                "results": [
                    {
                        "prompt_logprobs": [None] + [_position(t, -1.0) for t in p[1:]],
                        "token_ids": [0],
                    }
                    for p in calls["prompts"]
                ]
            }

    scores = teacher_mod.score_with_teacher(
        Client(), "teacher", [[1, 2], [3]], [[7, 8, 9], [4]]
    )

    assert calls["job_id"] == "teacher"
    assert calls["prompts"] == [[1, 2, 7, 8, 9], [3, 4]]
    assert calls["params"] == {
        "max_tokens": 1,
        "temperature": 1.0,
        "prompt_logprobs": 0,
    }
    assert scores == [[-1.0, -1.0, -1.0], [-1.0]]


def test_teacher_log_probs_shifted_matches_the_rollout_loss_mask():
    prompts, completions = [[1, 2, 3], [4, 5]], [[7, 8], [9, 10, 11]]
    sequences = [
        sequence_from_rollout(p, c, advantage=0.0) for p, c in zip(prompts, completions)
    ]
    _, context = collate(sequences, pad_token_id=0, max_seq_len=16)

    shifted = teacher_mod.teacher_log_probs_shifted(
        context["loss_mask"],
        [len(p) for p in prompts],
        [[-0.5, -0.6], [-0.1, -0.2, -0.3]],
    )

    assert torch.equal(shifted != 0, context["loss_mask"] > 0)
    assert shifted[0, 2].item() == pytest.approx(
        -0.5
    )  # position 2 predicts input_ids[3], the first completion token
    with pytest.raises(ValueError, match="do not match"):
        teacher_mod.teacher_log_probs_shifted(
            context["loss_mask"], [2, 2], [[-0.5, -0.6], [-0.1, -0.2, -0.3]]
        )


def test_check_shared_tokenizer():
    same = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1})
    other = SimpleNamespace(get_vocab=lambda: {"a": 0, "c": 1})
    teacher_mod.check_shared_tokenizer(same, same)
    with pytest.raises(ValueError, match="share a tokenizer"):
        teacher_mod.check_shared_tokenizer(same, other)


def test_reverse_kl_estimates():
    kl = teacher_mod.reverse_kl_estimates([[-0.5, -1.0]], [[-1.5, -1.0]])
    assert kl["kl/k1_per_token"] == pytest.approx(0.5)
    assert kl["kl/k3_per_token"] == pytest.approx((math.exp(-1.0) + 1.0 - 1.0) / 2)


# --- loss configuration --------------------------------------------------------


def test_processing_block_enables_the_grpo_teacher_term():
    block = runner.processing_block(
        _Config(teacher_clip_negative=1.5), global_batch_size=4
    )

    assert block["loss_fn"] == "grpo"
    assert block["config"]["teacher_tau"] == 1.0
    assert block["config"]["teacher_clip"] == 3.0
    assert block["config"]["teacher_clip_negative"] == 1.5
    assert block["config"]["global_batch_size"] == 4


def test_processing_block_omits_the_teacher_term_for_the_rl_comparison():
    block = runner.processing_block(
        _Config(teacher_tau=0.0, reward_coef=1.0), global_batch_size=4
    )

    assert not {"teacher_tau", "teacher_clip", "teacher_clip_negative"} & set(
        block["config"]
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"teacher_tau": -1.0}, "teacher_tau"),
        ({"teacher_clip": 0.0}, "teacher_clip"),
        ({"teacher_tau": 0.0, "reward_coef": 0.0}, "no training signal"),
        ({"lr_schedule": "linear"}, "lr_schedule"),
    ],
)
def test_validate_config_rejects_unusable_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        runner.validate_config(_Config(**overrides))


def test_lr_schedule_warms_up_then_decays_to_the_floor():
    config = _Config(warmup_steps=5)
    values = [
        runner.lr_at(step, 1e-5, config, total_steps=100) for step in (0, 4, 5, 100)
    ]
    assert values == pytest.approx([2e-6, 1e-5, 1e-5, 1e-6])
    assert (
        runner.lr_at(50, 1e-5, _Config(lr_schedule="constant"), total_steps=100) == 1e-5
    )


def test_forward_backward_payload_carries_teacher_logprobs():
    prompts, completions = [[1, 2, 3]], [[7, 8]]
    sequences = [sequence_from_rollout(prompts[0], completions[0], advantage=0.0)]
    kwargs, context = collate(
        sequences, pad_token_id=0, max_seq_len=16, temperature=1.0
    )
    context["teacher_log_probs_shifted"] = teacher_mod.teacher_log_probs_shifted(
        context["loss_mask"], [3], [[-0.5, -0.6]]
    )

    frame = wire.loads(
        _forward_backward_payload(
            kwargs, context=context, processing=runner.processing_block(_Config(), 1)
        )
    )

    assert torch.equal(
        frame["context"]["teacher_log_probs_shifted"],
        context["teacher_log_probs_shifted"],
    )
    assert torch.equal(
        frame["context"]["advantages"], torch.zeros_like(context["loss_mask"])
    )
    assert frame["processing"]["config"]["teacher_clip"] == 3.0


def test_job_bodies_pair_a_student_with_a_sampling_only_teacher():
    student, teacher = runner.job_bodies(_Config())

    assert {sub["job_type"] for sub in student["sub_job_configs"]} == {
        "training",
        "sampling",
    }
    assert [sub["job_type"] for sub in teacher["sub_job_configs"]] == ["sampling"]
    assert runner.job_bodies(_Config(teacher_tau=0.0, reward_coef=1.0))[1] is None


# --- GSM8K task ---------------------------------------------------------------


def test_gsm8k_gold_and_answer_parsing():
    assert parse_gold("16 - 3 - 4 = 9\n#### 1,234") == "1234"
    assert parse_gold("no marker") is None
    assert extract_answer("so \\boxed{18}") == "18"
    assert extract_answer("\\boxed{\\$18}") == "18"
    assert extract_answer("\\boxed{[18]} costs") == "18"
    assert extract_answer("first \\boxed{5} then \\boxed{18 \\text{ dollars}}") == "18"
    assert extract_answer("no box, total 1,200 dollars") == "1200"
    assert is_correct("\\boxed{18.0}", "18")
    assert not is_correct("\\boxed{17}", "18")


# --- end to end with a fake client ---------------------------------------------


@dataclass
class _Problems:
    train: list
    val: list
    test: list


class _Task:
    name = "toy"
    evaluation_name = "toy test"

    def load(self, *, seed, n_val):
        problems = [(f"q{i}", "1") for i in range(8)]
        return _Problems(
            train=problems[n_val:], val=problems[:n_val], test=problems[:2]
        )

    def build_prompt(self, question, renderer):
        return [1, 2, 3 + int(question[1:])]

    def score_response(self, response, answer):
        correct = float(response == answer)
        return correct, {"correct": correct}


@dataclass
class _FakeClient:
    token_count_shortfall: int = 0
    empty_rollouts: bool = False
    created: list = field(default_factory=list)
    cancelled: list = field(default_factory=list)
    generated: list = field(default_factory=list)
    synced: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    _requests: dict = field(default_factory=dict)

    def create_job_from_body(self, body):
        job_id = "teacher-job" if len(body["sub_job_configs"]) == 1 else "student-job"
        self.created.append(job_id)
        return {"job_id": job_id}

    def wait_for_job(self, job_id):
        return None

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)

    def _request(self, value):
        request_id = f"r{len(self._requests)}"
        self._requests[request_id] = value
        return request_id

    def poll_request(self, job_id, request_id):
        return self._requests[request_id]

    def generate(self, job_id, prompts, sampling_params):
        self.generated.append((job_id, prompts, sampling_params))
        if sampling_params.get("prompt_logprobs") == 0:
            results = [
                {
                    "prompt_logprobs": [None] + [_position(t, -1.0) for t in p[1:]],
                    "token_ids": [0],
                }
                for p in prompts
            ]
        elif self.empty_rollouts and "logprobs" in sampling_params:
            results = [
                {"text": "", "token_ids": [], "logprobs": [], "finish_reason": "stop"}
                for _ in prompts
            ]
        else:
            results = [
                {
                    "text": "1",
                    "token_ids": [7, 8],
                    "logprobs": [_position(7, -0.5), _position(8, -0.5)],
                    "finish_reason": "stop",
                }
                for _ in prompts
            ]
        return self._request({"results": results})

    def forward_backward(self, job_id, payload):
        frame = wire.loads(payload)
        self.frames.append(frame)
        tokens = float(frame["context"]["loss_mask"].sum().item())
        return self._request(
            {
                "avg_loss": 0.1,
                "metrics": {
                    "teacher_term_token_count": tokens - self.token_count_shortfall,
                    "teacher_log_ratio_sum": -tokens,
                },
            }
        )

    def step(self, job_id, learning_rate=None):
        self.steps.append(job_id)
        return self._request({"global_steps": 1, "grad_norm": 1.0})

    def weight_sync(
        self,
        job_id,
        *,
        source_sub_job_id,
        target_sub_job_ids,
        weight_format=None,
        bucket_size=None,
    ):
        self.synced.append((job_id, source_sub_job_id, tuple(target_sub_job_ids)))
        return self._request({"status": "ok"})

    def save(self, job_id, checkpoint_type=None):
        return self._request(
            {"stage_path": "s3://stage/checkpoints/cp_abc/global_step2/"}
        )


def _patch_runner(monkeypatch, client) -> list:
    """Route ``_train`` to ``client`` with stub tokenizers; return the logged metrics."""
    tokenizer = SimpleNamespace(
        pad_token_id=0, eos_token_id=0, get_vocab=lambda: {"x": 0}
    )
    renderer = SimpleNamespace(get_stop_sequences=lambda: [99])
    logged = []
    monkeypatch.setattr(runner, "make_client", lambda path: client)
    monkeypatch.setattr(
        runner,
        "build_renderer",
        lambda model, renderer_name=None: (tokenizer, renderer, renderer_name),
    )
    monkeypatch.setattr(
        runner,
        "setup_logging",
        lambda config, client=None, job_id=None: SimpleNamespace(
            log_metrics=lambda metrics, step: logged.append((step, metrics)),
            close=lambda: None,
        ),
    )
    pytest.importorskip("tinker_cookbook")
    monkeypatch.setattr(
        "tinker_cookbook.tokenizer_utils.get_tokenizer", lambda model: tokenizer
    )
    return logged


def test_end_to_end_with_a_fake_client(monkeypatch):
    client = _FakeClient()
    logged = _patch_runner(monkeypatch, client)

    runner._train(_Config(), _Task())

    assert client.created == ["student-job", "teacher-job"]
    assert sorted(client.cancelled) == ["student-job", "teacher-job"]
    assert client.steps == ["student-job", "student-job"]
    assert all(target == ("student-job:sampling:0",) for _, _, target in client.synced)
    teacher_calls = [
        prompts
        for job, prompts, params in client.generated
        if job == "teacher-job" and "logprobs" not in params
    ]
    assert [1, 2, 4, 7, 8] in [p for call in teacher_calls for p in call]
    assert len(client.frames) == 2
    assert client.frames[0]["processing"]["config"]["teacher_tau"] == 1.0
    assert any("test/correct" in metrics for _, metrics in logged)
    assert any("train/kl/k1_per_token" in metrics for _, metrics in logged)
    assert all(
        metrics["teacher/clipped_fraction"] == 0.0
        for _, metrics in logged
        if "teacher/clipped_fraction" in metrics
    )


@pytest.mark.parametrize("attach", [False, True])
def test_token_count_mismatch_stops_before_the_optimizer_step(monkeypatch, attach):
    client = _FakeClient(token_count_shortfall=1)
    _patch_runner(monkeypatch, client)
    config = (
        _Config(job_id="student-job", teacher_job_id="teacher-job")
        if attach
        else _Config()
    )

    with pytest.raises(RuntimeError, match="stopped before the optimizer step"):
        runner._train(config, _Task())

    assert len(client.frames) == 1
    assert client.steps == []
    expected = [] if attach else ["student-job", "teacher-job"]
    assert sorted(client.cancelled) == expected


def test_attached_jobs_are_left_running(monkeypatch):
    client = _FakeClient()
    _patch_runner(monkeypatch, client)

    runner._train(_Config(job_id="student-job", teacher_job_id="teacher-job"), _Task())

    assert client.created == []
    assert client.cancelled == []
    assert client.steps == ["student-job", "student-job"]


def test_a_skipped_step_still_logs_its_evaluation(monkeypatch):
    client = _FakeClient(empty_rollouts=True)
    logged = _patch_runner(monkeypatch, client)

    runner._train(_Config(), _Task())

    assert client.frames == []
    assert [step for step, metrics in logged if "test/correct" in metrics] == [0, 1, 2]


def test_eval_max_tokens_must_leave_room_for_evaluation_prompts(monkeypatch):
    client = _FakeClient()
    _patch_runner(monkeypatch, client)

    with pytest.raises(ValueError, match="eval_max_tokens 2048 leaves no room"):
        runner._train(_Config(eval_max_tokens=2048), _Task())
    assert client.created == []


def test_clipped_fraction_counts_tokens_outside_the_clip():
    student = [[0.0] * 8]
    teacher = [[-10.0] + [1.0] * 7]

    assert teacher_mod.clipped_fraction(
        student, teacher, clip=3.0, clip_negative=3.0
    ) == pytest.approx(1 / 8)
    assert teacher_mod.clipped_fraction(
        student, teacher, clip=0.5, clip_negative=20.0
    ) == pytest.approx(7 / 8)
    assert teacher_mod.clipped_fraction([], [], clip=3.0, clip_negative=3.0) is None


def test_entrypoint_runs_the_gsm8k_task(monkeypatch):
    pytest.importorskip("chz")
    from recipes.distillation.on_policy import train as entry

    captured = {}
    monkeypatch.setattr(
        entry, "train", lambda config, task: captured.update(config=config, task=task)
    )
    config = entry.Config(config="unused")
    entry.main(config)

    assert type(captured["task"]).__name__ == "GSM8KTask"
    assert (
        config.teacher_tau,
        config.teacher_clip,
        config.lr_schedule,
        config.reward_coef,
    ) == (1.0, 3.0, "cosine", 0.0)
