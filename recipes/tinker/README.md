# Tinker cookbook on Cortex

Run `tinker_cookbook.recipes.math_rl` on Cortex Training. Arguments after
`--` are the cookbook CLI you already use. `--training-gpus` and
`--sampling-gpus` size the Cortex job, same as the Cortex client CLI.

## Setup

```bash
pip install "arctic_platform[tinker]" "tinker-cookbook[math-rl]"
```

```bash
export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=<database>
export ARCTIC_CORTEX_SCHEMA=<schema>
export ARCTIC_CORTEX_PAT=<pat>
```

The account needs a database, a schema, a PAT, and quota for a training
sub-job and a sampling sub-job. The job can sit in `PLACING` until GPUs
are free. Ctrl-C cancels it.

## GSM8K

```bash
python -m recipes.tinker.math_rl \
  --training-gpus 1 \
  --sampling-gpus 1 \
  -- \
  env=gsm8k \
  model_name=Qwen/Qwen3.5-4B \
  renderer_name=qwen3_5_disable_thinking \
  lora_rank=32 \
  group_size=8 \
  groups_per_batch=16 \
  learning_rate=1e-4 \
  max_tokens=256
```

## MATH

```bash
python -m recipes.tinker.math_rl \
  --training-gpus 1 \
  --sampling-gpus 1 \
  --max-response-length 1024 \
  -- \
  env=math \
  model_name=Qwen/Qwen3.5-4B \
  renderer_name=qwen3_5_disable_thinking \
  lora_rank=32 \
  group_size=8 \
  groups_per_batch=16 \
  learning_rate=1e-4 \
  max_tokens=512
```

`model_name`, `lora_rank`, `group_size`, `groups_per_batch`,
`learning_rate`, and `max_tokens` are cookbook fields. Keep `max_tokens`
inside `--max-response-length` (512 when omitted).
