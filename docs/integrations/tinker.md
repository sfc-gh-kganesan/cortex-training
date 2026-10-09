# Run tinker-cookbook on Cortex

Start the cookbook command you already have with `arctic_platform.tinker.run`.
That makes the recipe's `import tinker` use Cortex. `python -m tinker_cookbook...`
by itself still calls the Thinking Machines API.

The recipe process does not need a GPU. Cortex runs the training and sampling jobs.

## Install

```bash
pip install "arctic_platform[tinker]" "tinker==0.25.0" "tinker-cookbook[math-rl]==0.5.5"
```

Install the cookbook extra for the recipe you are running. `math-rl` covers GSM8K and MATH.

## Connect

Use the account host, database, schema, and programmatic access token from
[connection setup](../getting-started/setup.md):

```bash
export ARCTIC_CORTEX_HOST=ACCOUNT.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=CORTEX_TRAINING_DB
export ARCTIC_CORTEX_SCHEMA=PUBLIC
export ARCTIC_CORTEX_PAT=<pat>
```

The account needs quota for one training sub-job and one sampling sub-job.
A job can stay in `PLACING` until GPUs are free. The process releases the job when it exits.

## Run a cookbook recipe

This is the cookbook's GSM8K command. The model is `Qwen/Qwen3.5-4B`, which Cortex
serves. The published note uses `Qwen/Qwen3.5-9B`. The other recipe arguments are the published ones.

`--training-gpus` and `--sampling-gpus` size the Cortex job. `--max-response-length`
must be at least the recipe's `max_tokens`.

```bash
python -m arctic_platform.tinker.run \
  --training-gpus 1 \
  --sampling-gpus 1 \
  --max-prompt-length 4096 \
  --max-response-length 1024 \
  tinker_cookbook.recipes.math_rl.train \
  env=gsm8k \
  model_name=Qwen/Qwen3.5-4B \
  group_size=64 \
  groups_per_batch=32 \
  learning_rate=8e-5 \
  max_tokens=1024
```

Another recipe uses the same launcher and that recipe's own arguments. Do not set
`TINKER_BASE_URL`. The recipe's `base_url` is ignored. Keep sampling temperature at `1.0`.

## Call the Tinker client

Put this import first. Pass GPU counts to `ServiceClient`. `base_url` is ignored.

```python
from arctic_platform import tinker

service = tinker.ServiceClient(training_gpus=1, sampling_gpus=1)
training = await service.create_lora_training_client_async("Qwen/Qwen3.5-4B", rank=32)
```
