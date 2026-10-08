# Run a tinker-cookbook recipe on Cortex

A tinker-cookbook recipe trains on Cortex through Arctic Platform. Pass the
cookbook module you already run. `--training-gpus` and `--sampling-gpus` size
the Cortex job, in the same way as the Cortex client CLI. Arguments after the
module name are that recipe's CLI.

## Install

```bash
pip install "arctic_platform[tinker]" "tinker==0.25.0" "tinker-cookbook[math-rl]==0.5.5"
```

Install the cookbook extra for the recipe you are running. `math-rl` covers
GSM8K and MATH.

## Connect

Use the host, database, schema, and programmatic access token from
[connection setup](../getting-started/setup.md):

```bash
export ARCTIC_CORTEX_HOST=ACCOUNT.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=CORTEX_TRAINING_DB
export ARCTIC_CORTEX_SCHEMA=PUBLIC
export ARCTIC_CORTEX_PAT=<pat>
```

The account needs quota for one training sub-job and one sampling sub-job.
The job can stay in `PLACING` until GPUs are free. Ctrl-C cancels it.

## Run

GSM8K:

```bash
python -m arctic_platform.tinker.run \
  --training-gpus 1 \
  --sampling-gpus 1 \
  tinker_cookbook.recipes.math_rl.train \
  env=gsm8k \
  model_name=Qwen/Qwen3.5-4B \
  renderer_name=qwen3_5_disable_thinking \
  lora_rank=32 \
  group_size=8 \
  groups_per_batch=16 \
  learning_rate=1e-4 \
  max_tokens=256
```

MATH uses the same module with `env=math`. Raise the response budget when
answers are longer than 256 tokens:

```bash
python -m arctic_platform.tinker.run \
  --training-gpus 1 \
  --sampling-gpus 1 \
  --max-prompt-length 2048 \
  --max-response-length 1024 \
  tinker_cookbook.recipes.math_rl.train \
  env=math \
  model_name=Qwen/Qwen3.5-4B \
  renderer_name=qwen3_5_disable_thinking \
  lora_rank=32 \
  group_size=8 \
  groups_per_batch=16 \
  learning_rate=1e-4 \
  max_tokens=512
```

Another recipe is the same command with its module, for example
`tinker_cookbook.recipes.code_rl.train`, and that recipe's own arguments.
Keep `max_tokens` within `--max-response-length` (512 if you omit the flag)
and keep prompts within `--max-prompt-length` (2048 if you omit it). Sampling
temperature stays at `1.0`.

`python -m tinker_cookbook...` on its own still calls the Thinking Machines
API. The launcher above imports Arctic Platform first, so the cookbook's
`import tinker` reaches Cortex.

A script you start yourself can do that import on its first line:

```python
from arctic_platform import tinker

service = tinker.ServiceClient(training_gpus=1, sampling_gpus=1)
```
