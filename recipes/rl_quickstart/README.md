# Math GRPO

Train a model on Hendrycks MATH with grouped policy optimization, and evaluate on
MATH-500. Each run creates one job with colocated training and sampling
sub-jobs. The loop generates rollouts, scores them, trains, and syncs weights to
the sampler. The example is in `math_grpo/`: `train.py` is the entry point and
`task.py` holds the prompt, scoring and evaluation. It runs on the shared loop
in `runner.py`.

## Run

The default is Qwen3-8B LoRA at standard context on H200: 4 training GPUs and 2
sampling GPUs. Check capacity first with `cortex-training capacity`.

```bash
python -m recipes.rl_quickstart.math_grpo.train config=/path/to/config.json
```

Other models:

```bash
# Qwen3-8B full-parameter
python -m recipes.rl_quickstart.math_grpo.train config=/path/to/config.json lora=false

# Qwen3.6-35B-A3B (MoE) LoRA; add lora=false for full-parameter
python -m recipes.rl_quickstart.math_grpo.train config=/path/to/config.json \
  model=Qwen/Qwen3.6-35B-A3B context=long lora=true max_tokens=2048

# Qwen3.8-27B full-parameter
python -m recipes.rl_quickstart.math_grpo.train config=/path/to/config.json \
  model=Qwen/Qwen3.8-27B lora=false \
  max_tokens=2048 weight_sync_bucket_size=3221225472
```

## Options

| Argument | Default | Meaning |
|---|---|---|
| `model=`, `context=`, `hardware=`, `lora=` | `Qwen/Qwen3-8B`, `standard`, `h200`, `true` | Which catalog files to use |
| `lr=`, `train_batch_size=`, `max_seq_len=` | from the job | Override the job for this run (`max_seq_len` applies to both sub-jobs) |
| `problems_per_batch=`, `group_size=` | 64, 16 | Rollouts per step = problems × group size |
| `max_tokens=`, `temperature=` | 4096, 1.0 | Rollout sampling |
| `max_steps=` | 10 | Training steps |
| `eval_every=`, `n_test=` | 10, all | MATH-500 eval cadence (`0` skips it) and size |
| `weight_sync_format=` | server default | `vllm` or `hf` for full-parameter weight sync. LoRA jobs always sync `lora` |
| `weight_sync_bucket_size=` | server default | Weight-sync transfer bucket in bytes. Set it above the largest single tensor |
| `job_config=` | none | Use this job file instead of the catalog and `overrides.json`; `lr=`, `train_batch_size=` and `max_seq_len=` still apply |
| `wandb_project=`, `sf_tracking=` | off | [Experiment tracking](../../docs/guides/operations/wandb.md) |

## How the job is built

The job is built as described in
[How an example builds its job](../README.md#how-an-example-builds-its-job).
For RL, the model's training and inference files are combined into one job, and
[`rl/overrides.json`](overrides.json) adds the RL tuning that every model
shares: optimizer betas `(0.9, 0.95)`, `mb_spec.max_tokens_per_mb` 10240, and
ZeRO `reduce_scatter`. Edit that file to change these for every run.

`job_config=` also takes either file of a catalog pair, such as a model's
`<hw>-training-<lora|full>.json` under `model-catalog/models/`: a file with only
a training or only a sampling sub-job takes the other sub-job from the file
beside it whose name swaps `-training-` and `-inference-`. The pair is used as
is, without `rl/overrides.json` or router replay.

Each run that creates its job from the catalog (not with `job_config=` or
`job_id=`) writes the job it built to
`recipes/rl_quickstart/configs/<slug>-<context>-<hw>-<lora|full>.json`, which is
gitignored. To customize a run beyond the command-line options, copy that file,
edit it, and pass it back with `job_config=`.

## Router replay

For MoE models (`architecture: moe` in `model.json`, such as
Qwen3.6-35B-A3B), router replay is turned on for both sub-jobs with fixed caps,
unless you pass `job_config=`:

| Sub-job | Setting |
|---|---|
| Training | `{"enabled": true, "mode": "strict", "max_cache_bytes": 5368709120}` (5 GiB per data-parallel shard) |
| Sampling | `{"enabled": true, "max_cache_bytes": 21474836480}` (20 GiB per sampling replica) |

Any value you set in `rl/overrides.json` wins. For example, to use different
caps and best-effort mode:

```json
{"sub_job_configs": [
  {"job_type": "training", "training_config": {"router_replay": {"max_cache_bytes": 2147483648, "mode": "best_effort"}}},
  {"job_type": "sampling", "inference_config": {"router_replay": {"max_cache_bytes": 17179869184}}}]}
```

`"router_replay": {"enabled": false}` on either sub-job turns router replay off.
In `best_effort` mode, the trainer routes rows the sampler no longer holds with
its own gate, where `strict` mode fails the step. Dense models get no router
replay.

## Router-replay capacity planning

To size caps for a specific job instead of using the fixed defaults, use the
planner in `recipes/utils.py` (`plan_router_replay_caps`, or
`plan_training_cache_bytes` and `plan_sampling_cache_bytes`). Then set the
result in `rl/overrides.json`. The planner only calculates: it doesn't call the
service or change a job.

```text
entry_bytes   = seq_len * num_moe_layers * topk * dtype_bytes   # 1 byte for <=256 experts, else 2
training_cap  = ceil(ceil(batch_size / dp_size) * entry_bytes * safety_factor)
sampling_cap  = ceil(ceil(backlog / replicas * routing_skew) * entry_bytes * safety_factor)
backlog       = max_inflight_rollouts + ceil(batch_size * over_generation_multiplier)
                + batch_size * max(async_depth, 1)
```

The defaults are `routing_skew=2.0` and `safety_factor=1.2`. The skew is an
estimate, not an upper bound. See
[REST API section 8.2](../../docs/reference/rest-api.md#82-trainingconfig) for
`max_cache_bytes`.

```python
from recipes import utils

# Reads the model's Hugging Face config. To skip the download, pass
# num_moe_layers_override=, topk_override= and num_experts_override=.
geometry = utils.resolve_model_moe_geometry("Qwen/Qwen3.6-35B-A3B")
training_cap = utils.plan_training_cache_bytes(
    seq_len=8192,
    geometry=geometry,
    resident_rows=utils.derive_resident_rows(
        batch_size=1024, dp_size=utils.derive_dp_size(training_gpus=8, sp_size=1)
    ),
).max_cache_bytes
```

## Job settings for specific models

Model-specific job fields are documented in the reference docs:

- MoE LoRA targets attention only: see [LoRA and QLoRA](../../docs/guides/training/lora-and-qlora.md).
- `prime_rl` versus `model_options`, and `numerics.fp32_parity` (on by default
  for Qwen3.5-family models; turn it off if a job runs out of memory): see
  [REST API `model_options`](../../docs/reference/rest-api.md#model_options).

## Evaluation and logs

Training logs reward, correctness, format, rollout counts and loss. MATH-500
accuracy is logged as `test/env/all/correct`. After the final save, the example
prints a matching `recipes.inference.evaluate` command that uses the same
prompt, grader and decoding:

```bash
python -m recipes.inference.evaluate config=/path/to/config.json \
  model=MODEL context=CONTEXT hardware=HARDWARE lora=LORA \
  source_job_id=TRAINING_JOB_ID checkpoint_id=CHECKPOINT_ID \
  temperature=1.0 max_tokens=MAX_TOKENS
```
