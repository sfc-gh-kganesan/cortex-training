# On-Policy Distillation

Train a small student model to match a larger teacher on the student's own
outputs. Each step, the student answers a batch of GSM8K questions, a frozen
teacher scores every token the student wrote, and the student is updated toward
the tokens the teacher prefers. The default run distills Qwen3-8B into
Qwen3-0.6B and raises the student's GSM8K test accuracy by about 4–6 points
in 100 steps.

## How it works

The recipe creates two jobs:

- **Student**: a training sub-job and a sampling sub-job of the same model.
  The sampling sub-job generates rollouts, and weight sync copies each update
  from training to sampling.
- **Teacher**: a separate sampling-only job. It never trains and never
  receives weight syncs.

Each training step:

1. The student samples `group_size` answers to each of `problems_per_batch`
   questions.
2. Each answer is sent to the teacher as one pre-tokenized prompt (question
   tokens followed by the student's answer tokens) with `prompt_logprobs: 0`
   and `max_tokens: 1`. The teacher returns its log probability of every token
   the student wrote, so the student's exact tokens are scored without decoding
   or re-tokenizing (see
   [Per-token log probabilities](../../docs/reference/rest-api.md#per-token-log-probabilities)).
3. The teacher scores go in `context["teacher_log_probs_shifted"]`, aligned so
   that position `i` scores the token at `i + 1`, and the student trains with
   the `grpo` loss's teacher term (see
   [Teacher term for on-policy distillation](../../docs/reference/rest-api.md#teacher-term-for-on-policy-distillation)).
   Each token's advantage becomes
   `teacher_tau * clamp(log p_teacher - log p_student, -teacher_clip, teacher_clip)`,
   which on average follows the gradient of the reverse KL from student to
   teacher.

This is the formulation from Thinking Machines'
[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
and tinker-cookbook's `distillation/train_on_policy.py`, with the KL advantage
clipped and the learning rate decayed.

Every teacher result is checked before training: a missing, mis-sized,
misaligned, or non-finite score raises an error. After each forward-backward
pass, and before the optimizer step, the recipe also checks that the server's
teacher term covered exactly the tokens the teacher scored.
The student and teacher must share a tokenizer; the recipe checks this before
creating any job.

The recipe does not send `old_log_probs_shifted`, so the server uses the
current, detached student log probabilities. Each update is fully on-policy
and `eps_clip` has no effect.

## Prerequisites

Install the client and recipe dependencies as described in
[recipes/README.md](../README.md#prerequisites), and create a connection file
from `examples/config/connection.json.template`. The student and teacher models
are both in the [model catalog](../../docs/reference/model-compatibility.md).

## Dataset and Model

- Dataset: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k).
  200 training problems are held out as a validation set; the remaining 7,273
  are training prompts, and the 1,319-question test split is the benchmark.
  Each question ends with
  `Please reason step by step, and put your final answer within \boxed{}.`
- Student: `Qwen/Qwen3-0.6B`, full-parameter, non-thinking chat template
  (`renderer_name=qwen3_disable_thinking`).
- Teacher: `Qwen/Qwen3-8B`, about 93% on the same GSM8K evaluation.

## Hardware

Three GPUs: one training and one sampling GPU for the student
(`configs/qwen3_0p6b_student.json`) and one sampling GPU for the teacher
(`configs/qwen3_8b_teacher.json`). Check capacity before submitting:

```bash
cortex-training capacity --hardware H200
```

## Run

```bash
python -m recipes.distillation.on_policy.train \
  config=/path/to/config.json
```

`config=` is the Snowflake connection file. A default run takes about
25 minutes on H200s, including about 5 minutes for both jobs to start. Both
jobs are cancelled when the run ends, after the final checkpoint is saved.

## Expected Results

Validated on H200s with the default configuration. Accuracy is greedy, on the
full GSM8K test split, evaluated every 10 steps. The first row is a run of this
entry point; the other three are earlier runs of the same configuration, before
it was packaged as a recipe, with data seeds `0`, `1`, and `2`.

| Run | Step 0 | Mean of steps 50–100 | Step 100 | Best validation step (test accuracy there) |
|---|---|---|---|---|
| This entry point, `seed=0` | 64.5% | 68.8% | 70.0% | 40 (71.3%) |
| Seed 0 | 64.1% | 67.2% | 68.4% | — |
| Seed 1 | 63.8% | 69.2% | 70.4% | — |
| Seed 2 | 63.8% | 68.3% | 71.3% | — |

The teacher scores 92.5–93.1%. The held-out reverse-KL estimate
(`heldout/kl/k3_per_token`) falls from about 0.46 to about 0.26 over the run.
Each evaluation has a standard error of about ±1.3 points. Logged metrics
include:

| Metric | Meaning |
|---|---|
| `test/correct`, `val/correct` | Greedy accuracy on the test split and the validation split |
| `teacher/test/correct` | The teacher's test accuracy, at step 0 |
| `train/kl/k1_per_token`, `train/kl/k3_per_token` | Reverse-KL estimates on the training rollouts, from the student's sampling log probabilities |
| `heldout/kl/k3_per_token` | The same estimate on `n_kl_eval` test prompts the student does not train on |
| `teacher_term_token_count`, `teacher_log_ratio_sum`, `teacher_clipped_log_ratio_sum` | The server's teacher-term metrics |
| `teacher/clipped_fraction` | Share of training tokens whose `log p_teacher - log p_student` falls outside the clip, using the student's sampling log probabilities |
| `rollout/correct`, `rollout/mean_tokens`, `rollout/frac_truncated` | Training-rollout accuracy, length, and how often answers hit `max_tokens` |

At the end, the run logs the step with the best validation accuracy and the test
accuracy at that step. Choose a checkpoint by validation accuracy, not test
accuracy.

## Configuration Variations

```bash
python -m recipes.distillation.on_policy.train \
  config=/path/to/config.json \
  max_steps=MAX_STEPS \
  problems_per_batch=PROBLEMS_PER_BATCH \
  group_size=GROUP_SIZE \
  max_tokens=MAX_TOKENS \
  teacher_clip=TEACHER_CLIP \
  lr_schedule=cosine \
  reward_coef=REWARD_COEF \
  seed=SEED \
  wandb_project=WANDB_PROJECT
```

- `lr_schedule` (default `cosine`, decaying to `min_lr_ratio` of the peak after
  `warmup_steps`). With `constant`, accuracy peaks within the first 5–25 steps
  and then falls below where it started, so keep the decay unless you also stop
  early.
- `teacher_clip` (default `3`) limits how hard any one token can pull. With the
  cosine schedule, `teacher_clip=10` gave similar accuracy.
- `reward_coef` (default `0`, pure distillation) adds a group-centred GSM8K
  correctness reward to every token's advantage, alongside the teacher term.
- `teacher_tau=0` with `reward_coef=1` turns the teacher off and runs the same
  loop as plain GRPO, for comparison. No teacher job is created. This variant
  is not part of the validated configuration.
- `teacher_tau` (default `1`) weights the teacher term relative to the reward.
- The peak learning rate, GPU counts, and sequence length live in the job
  configs under `configs/`. The teacher's `max_seq_len` must exceed the
  student's, because the teacher scores the question and answer together.
- `job_id=` and `teacher_job_id=` attach to running jobs instead of creating
  new ones; attached jobs are left running.

## Checkpoint and Evaluation

The run saves a weights-only checkpoint of the student and logs its
`checkpoint_id`. Serve it with the inference recipe and the sampling-only
config shipped here:

```bash
python -m recipes.inference.serve \
  config=/path/to/config.json \
  job_config=recipes/distillation/configs/qwen3_0p6b_sampling.json \
  source_job_id=STUDENT_JOB_ID \
  checkpoint_id=CHECKPOINT_ID
```

`recipes.inference.evaluate task=gsm8k` uses a different prompt and answer
format (the bare question, graded on `#### n` or the last number), so its
numbers are not comparable with the accuracy this recipe reports.

## Troubleshooting

- `student and teacher tokenizers differ`: pick a teacher from the same model
  family; the teacher scores the student's token ids directly.
- `generate result has no 'prompt_logprobs'`: the teacher job must be a
  sampling job on a server version that passes `prompt_logprobs` through.
- `Unknown config keys` naming `teacher_tau`: the training server predates the
  `grpo` teacher term.
- `the teacher term covered N tokens, expected M`: the server trained on
  different tokens than the teacher scored; check that the student job config
  has not been changed to pack or truncate sequences.
- Accuracy rises and then falls: keep `lr_schedule=cosine`, or choose the step
  with the best `val/correct`.

## Provenance

Adapted from tinker-cookbook's on-policy distillation
(`tinker_cookbook/distillation/train_on_policy.py`) and Thinking Machines'
[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
The `last_validated` date in `recipe.yaml` records the most recent verified run.
