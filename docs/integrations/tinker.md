# Run tinker-cookbook on Cortex

This page is the Cortex path for tinker-cookbook and for scripts that use the Tinker client.

A cookbook recipe is `python -m arctic_platform.integrations.tinker.run`, then the module and arguments from the cookbook. `--training-gpus` and `--sampling-gpus` size the Cortex job. Cortex runs training and sampling.

A script imports `arctic_platform.integrations.tinker` and passes those GPU counts to `ServiceClient`. Connection settings are `ARCTIC_CORTEX_*`.

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

Other recipes use the same launcher and their own arguments.

## Correctness

Published GSM8K arguments, `Qwen/Qwen3.5-4B`, 4 training GPUs and 4 sampling GPUs. The run finished at step 210. Held-out test correct is logged every 20 steps: 0.544 at step 0 and 0.964 at step 200. KL stays near 0.0002.

| Step | Correct | Reward | Test correct |
|---|---|---|---|
| 0 | 0.617 | 0.549 | 0.544 |
| 3 | 0.855 | 0.837 | — |
| 5 | 0.951 | 0.946 | — |
| 20 | 0.923 | 0.923 | 0.938 |
| 200 | 0.958 | 0.956 | 0.964 |
| 210 | 0.999 | 0.999 | — |

## Recipe support

`tinker-cookbook` 0.5.5 on `Qwen/Qwen3.5-4B`, 4 training GPUs and 4 sampling GPUs, one H200. Cookbook source is unchanged. Where a published command names another model, only the model is changed.

### Showed learning

| Recipe | Steps | Reward | Correct |
|---|---|---|---|
| GSM8K | 211 | 0.549 → 0.999 | 0.617 → 0.999 |
| MATH | 11 | −0.045 → 0.142 | 0.055 → 0.234 |
| DeepMath | 52 | −0.100 → 0.021 | 0 → 0.121 |
| rl-basic | 5 | −0.077 → 0.195 | 0.023 → 0.290 |
| guess-number | 10 | −0.086 → 0.688 | — |
| tic-tac-toe | 4 | −0.887 → −0.184 | — |
| shorter | 4 | −0.275 → −0.002 | — |

DeepMath uses the MATH command with `env=deepmath`. rl-basic, guess-number, tic-tac-toe, and shorter use the recipe defaults. rl-basic's published model is `Qwen/Qwen3.5-9B-Base`. GSM8K detail is in the table above. KL on the math runs stays near 0.0002.

### Reward barely moved

| Recipe | Steps | Reward | Correct |
|---|---|---|---|
| arithmetic | 25 | −0.100 → 0.000 | 0 → 0 |
| Polaris | 143 | −0.100 → −0.067 | 0 → 0.033 |
| twenty-questions | 10 | 0.049 → 0.105 | — |

Arithmetic leaves `max_tokens` at 5. Polaris uses the MATH command with `env=polaris`.

### Incomplete

| Recipe | Outcome |
|---|---|
| rubric | Exits at 0 steps. The local example file is smaller than the published batch. A 5-step run with a smaller batch moved reward −0.10 → 0.088. |
| chat supervised learning, sl-basic | Stop at step 0 because published eval calls `forward`. Set `eval_every=0`. A 5-step run reduced NLL, 2.188 → 1.449 and 2.188 → 1.367. |
| on-policy distillation | One step, teacher KL 0.00026, then the sampler returned a conflict. A 5-step same-datum replay reduced loss on Cortex and on the Tinker API. |
| code RL | Stopped while loading the task list. That load exhausted local memory, and SandboxFusion was down. |
| prompt distillation | The published data file was absent. A 5-step run reduced NLL 5.610 → 0.161. |
| Harbor, search, verifiers | The harbor package, chromadb, and the verifiers package were not installed. |

### Outside this client

| Recipe | Blocker |
|---|---|
| `rl_loop`, `sl_loop` | Synchronous `forward_backward` |
| DPO, SDFT | `forward` |
| Off-policy distillation | Top-k prompt logprobs |
| RLHF pipeline | Checkpoint resume, and a reward model loaded from a path |
| True thinking score | `create_sampling_client_async` |
| VLM classifier | Image input |
| Audio | Audio input |
| `max_steps_off_policy` | Several samplers at once |
| Checkpoint resume | A sampler save syncs weights in this process only |

## Call the Tinker client

Import `arctic_platform.integrations.tinker` first, then construct `ServiceClient` with the GPU counts.

```python
from arctic_platform.integrations import tinker

service = tinker.ServiceClient(training_gpus=1, sampling_gpus=1)
training = await service.create_lora_training_client_async("Qwen/Qwen3.5-4B", rank=32)
```
