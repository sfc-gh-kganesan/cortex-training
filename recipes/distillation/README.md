# On-Policy Distillation

Train a small student model to match a larger teacher on the student's own
outputs. Each step, the student answers a batch of GSM8K questions, a frozen
teacher scores every token the student wrote, and the student is updated toward
the tokens the teacher prefers. The default run distills Qwen3-8B into
Qwen3-0.6B.

## How it works

The example creates two jobs:

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

The student and teacher must share a tokenizer, so pick a teacher from the
student's model family. The example checks this before it creates any job. It
also checks every teacher score, and checks that the server's teacher term
covered exactly the scored tokens. Updates are fully on-policy: no
`old_log_probs_shifted` is sent, so `eps_clip` has no effect.

## Run

The default distills Qwen3-8B into Qwen3-0.6B on GSM8K, using four H200 GPUs:
one training and one sampling GPU for the student, plus two for the teacher.
Check capacity first with `cortex-training capacity`. Install the dependencies
listed in the [examples prerequisites](../README.md#prerequisites).

```bash
python -m recipes.distillation.on_policy.train config=/path/to/config.json
```

Both jobs are cancelled after the final checkpoint is saved.

- **Dataset:** [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k).
  200 training problems are held out for validation, the other 7,273 are
  training prompts, and the 1,319-question test split is the benchmark.
- **Student:** `Qwen/Qwen3-0.6B`, full-parameter, non-thinking
  (`renderer_name=qwen3_disable_thinking`).
- **Teacher:** `Qwen/Qwen3-8B`, sampling only.

## How the jobs are built

`make_distillation_config` builds both jobs from the
[model catalog](../../model-catalog/README.md):

- **Student:** the training and inference files for `student=`, `context=`,
  `hardware=` and `lora=` (default `false`), combined like an RL job but without
  router replay. Then `distillation/overrides.json` shrinks it to the small
  default (1 + 1 GPUs, batch 1), and `student_max_seq_len=` (default 2048) sizes
  both sub-jobs.
- **Teacher:** the full-weights inference file for `teacher=` (in
  `teacher_context=`, which defaults to `context=`), sized by
  `teacher_max_seq_len=` (default 4096). The teacher scores the question and the
  answer together, so its window must be larger than the student's.

Each run writes the jobs it builds from the catalog (not `job_config=` files or
attached jobs) to `recipes/distillation/configs/` (gitignored). To
customize them, edit copies and pass them back with `job_config=` and
`teacher_job_config=`. To attach to running jobs instead, use `job_id=` and
`teacher_job_id=`; attached jobs are left running. To build the jobs without
running the example:

```bash
python -m recipes.utils --distillation \
  --student Qwen/Qwen3-0.6B --teacher Qwen/Qwen3-8B --context standard \
  --student-max-seq-len 2048 --teacher-max-seq-len 4096 \
  --override recipes/distillation/overrides.json -o dist.json
```

This writes `dist.json` (student) and `dist.teacher.json` (teacher). Without
`--override`, the student gets the catalog's full size: 8 training and 8
sampling GPUs.

## Expected Results

This example has no validated accuracy numbers yet (`last_validated` is `null`
in `example.yaml`). Greedy GSM8K test accuracy is evaluated every `eval_every`
steps (default 10). Logged metrics:

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

## Options

| Argument | Default | Meaning |
|---|---|---|
| `student=`, `teacher=`, `context=`, `hardware=`, `lora=` | see above | Which catalog files to use |
| `student_max_seq_len=`, `teacher_max_seq_len=` | 2048, 4096 | Context windows. The teacher's must be larger |
| `max_steps=`, `problems_per_batch=`, `group_size=`, `max_tokens=` | 100, 16, 4, 1024 | Loop size |
| `lr_schedule=` | `cosine` | Decays to `min_lr_ratio` of the peak after `warmup_steps`. With `constant`, accuracy can peak early and then fall |
| `teacher_clip=` | 3 | Limits how hard any one token can pull |
| `teacher_tau=` | 1 | Weight of the teacher term |
| `reward_coef=` | 0 | Adds a group-centred GSM8K correctness reward. `teacher_tau=0 reward_coef=1` runs plain GRPO with no teacher job |
| `wandb_project=`, `sf_tracking=` | off | [Experiment tracking](../../docs/guides/operations/wandb.md) |

## Checkpoint and Evaluation

The run saves a weights-only checkpoint of the student and logs its
`checkpoint_id`. Serve it with the inference example:

```bash
python -m recipes.inference.serve config=/path/to/config.json \
  model=Qwen/Qwen3-0.6B context=standard \
  source_job_id=STUDENT_JOB_ID checkpoint_id=CHECKPOINT_ID
```

The catalog's Qwen3-0.6B inference file uses eight GPUs. To serve on fewer,
write the job with
`python -m recipes.utils --model Qwen/Qwen3-0.6B --context standard --inference -o serve.json`,
lower its `n_gpus`, and pass `job_config=serve.json` instead of `model=` and
`context=`. To generate from or evaluate the checkpoint, also pass
`renderer_name=qwen3_disable_thinking`, as training does.

`recipes.inference.evaluate task=gsm8k` uses a different prompt and answer
format (the bare question, graded on `#### n` or the last number), so its
numbers are not comparable with the accuracy this example reports.

## Troubleshooting

- `student and teacher tokenizers differ`: pick a teacher from the same model
  family; the teacher scores the student's token ids directly.
- `teacher max_seq_len ... must exceed the student's`: raise
  `teacher_max_seq_len=` above `student_max_seq_len=`.
- `generate result has no 'prompt_logprobs'`: the teacher job must be a
  sampling job on a server version that passes `prompt_logprobs` through.
- `Unknown config keys` naming `teacher_tau`: the training server does not
  support the `grpo` teacher term.
- `the teacher term covered N tokens, expected M`: the server trained on
  different tokens than the teacher scored; check that the student job config
  has not been changed to pack or truncate sequences.
- Accuracy rises and then falls: keep `lr_schedule=cosine`, or choose the step
  with the best `val/correct`.

## Provenance

Adapted from tinker-cookbook's on-policy distillation
(`tinker_cookbook/distillation/train_on_policy.py`) and Thinking Machines'
[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
