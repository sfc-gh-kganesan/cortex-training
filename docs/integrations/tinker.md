# Run tinker-cookbook on Cortex

This page is the Cortex path for tinker-cookbook and for scripts that use the Tinker client.

A cookbook recipe is `python -m arctic_platform.tinker.run`, then the module and arguments from the cookbook. `--training-gpus` and `--sampling-gpus` size the Cortex job. Cortex runs training and sampling.

A script imports `arctic_platform.tinker` and passes those GPU counts to `ServiceClient`. Connection settings are `ARCTIC_CORTEX_*`.

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

## Run a cookbook recipe

GSM8K, with the published recipe arguments. The model is `Qwen/Qwen3.5-4B`. `--max-response-length` covers `max_tokens`.

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

Other recipes use the same launcher and their own arguments.

## Call the Tinker client

Import `arctic_platform.tinker` first, then construct `ServiceClient` with the GPU counts.

```python
from arctic_platform import tinker

service = tinker.ServiceClient(training_gpus=1, sampling_gpus=1)
training = await service.create_lora_training_client_async("Qwen/Qwen3.5-4B", rank=32)
```
