# Run tinker-cookbook on Cortex

Run tinker-cookbook, or your own script that uses the Tinker client, with
Cortex Training doing the training and sampling. The Arctic Platform package
provides the bridge:

- A cookbook run is `python -m arctic_platform.integrations.tinker.run`,
  followed by the cookbook module and its arguments. `--training-gpus` and
  `--sampling-gpus` size the Cortex job.
- A script imports `arctic_platform.integrations.tinker` and passes the same GPU
  counts to `ServiceClient`.

Both read the connection from `ARCTIC_CORTEX_*` environment variables.

## Install

```bash
pip install "arctic_platform[tinker]" "tinker==0.25.0" "tinker-cookbook[math-rl]==0.5.5"
```

Install the cookbook extra for the recipe you are running. `math-rl` covers GSM8K and MATH.

## Connect

Use the account host, database, schema, and programmatic access token from
[connection setup](../../docs/getting-started/setup.md):

```bash
export ARCTIC_CORTEX_HOST=ACCOUNT.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=CORTEX_TRAINING_DB
export ARCTIC_CORTEX_SCHEMA=PUBLIC
export ARCTIC_CORTEX_PAT='YOUR_PROGRAMMATIC_ACCESS_TOKEN'
```

## Run a cookbook recipe

GSM8K on `Qwen/Qwen3.5-4B`, with the cookbook recipe's published arguments.
`--max-response-length` covers `max_tokens`.

```bash
python -m arctic_platform.integrations.tinker.run \
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

Other cookbook recipes use the same launcher and their own arguments.

## Results

The command above with `--training-gpus 4 --sampling-gpus 4` ran to step 210.
Held-out test accuracy, logged every 20 steps, rose from 0.544 at step 0 to
0.964 at step 200, and KL stayed near 0.0002.

| Step | Correct | Reward | Test correct |
|---|---|---|---|
| 0 | 0.617 | 0.549 | 0.544 |
| 3 | 0.855 | 0.837 | — |
| 5 | 0.951 | 0.946 | — |
| 20 | 0.923 | 0.923 | 0.938 |
| 200 | 0.958 | 0.956 | 0.964 |
| 210 | 0.999 | 0.999 | — |

## Cookbook recipe support

`tinker-cookbook` 0.5.5. Supervised recipes need `eval_every=0`.

| Cookbook recipe | Support |
|---|---|
| GSM8K, MATH, DeepMath, arithmetic, Polaris | Supported |
| rl-basic | Supported |
| guess-number, twenty-questions, tic-tac-toe | Supported |
| shorter | Supported |
| rubric | Supported |
| chat supervised learning, sl-basic, prompt distillation | Supported |
| on-policy distillation | Supported |
| code RL, Harbor, search, verifiers | Not validated (need external tools or graders) |
| `rl_loop`, `sl_loop` | Not yet supported |
| DPO, SDFT | Not yet supported |
| off-policy distillation | Not yet supported |
| RLHF pipeline | Not yet supported |
| true thinking score | Not yet supported |
| checkpoint resume | Not yet supported |
| `max_steps_off_policy` | Not yet supported |
| VLM classifier | Not supported |
| audio | Not supported |

## Call the Tinker client

Import `arctic_platform.integrations.tinker` first, then construct `ServiceClient` with the GPU counts.

```python
from arctic_platform.integrations import tinker

service = tinker.ServiceClient(training_gpus=1, sampling_gpus=1)
training = await service.create_lora_training_client_async("Qwen/Qwen3.5-4B", rank=32)
```
