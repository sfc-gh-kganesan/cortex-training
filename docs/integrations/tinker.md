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
| 3 | 0.855 | 0.837 | |
| 5 | 0.951 | 0.946 | |
| 20 | 0.923 | 0.923 | 0.938 |
| 200 | 0.958 | 0.956 | 0.964 |
| 210 | 0.999 | 0.999 | |

## Recipe support

`tinker-cookbook` 0.5.5, `Qwen/Qwen3.5-4B`, 4 training GPUs and 4 sampling GPUs. Where a published command names another model, these runs change the model and leave the rest of the command as published. Validated means that command showed learning. Ran means it finished and the reward barely moved. Stopped means it exited before learning. Not started means it never launched. Not supported means a call this client does not implement.

| Recipe | Status | Note |
|---|---|---|
| `math_rl` GSM8K | Validated | 211 steps. Reward 0.549 → 0.999, correct 0.617 → 0.999. Table above |
| `math_rl` MATH | Validated | 11 steps. Reward −0.045 → 0.142, correct 0.055 → 0.234 |
| `math_rl` DeepMath | Validated | 52 steps, MATH arguments with `env=deepmath`. Reward −0.100 → 0.021, correct 0 → 0.121 |
| `rl_basic` | Validated | 5 steps, recipe defaults with the model set from `Qwen/Qwen3.5-9B-Base`. Reward −0.077 → 0.195, correct 0.023 → 0.290 |
| `multiplayer_rl` guess-number | Validated | 10 steps, recipe defaults. Reward −0.086 → 0.688 |
| `multiplayer_rl` tic-tac-toe | Validated | 4 steps, recipe defaults. Reward −0.887 → −0.184 |
| `preference/shorter` | Validated | 4 steps, recipe defaults. Reward −0.275 → −0.002 |
| `math_rl` arithmetic | Ran | 25 steps. Reward −0.100 → 0.000, correct stayed 0. `max_tokens` left at 5 |
| `math_rl` Polaris | Ran | 143 steps, MATH arguments with `env=polaris`. Reward −0.100 → −0.067, correct 0.033 |
| `multiplayer_rl` twenty-questions | Ran | 10 steps, recipe defaults. Reward 0.049 → 0.105. Opens a second sampler on a base model |
| `rubric` | Stopped | The published batch is larger than the local example file, so the run exits at 0 steps. A five-step run with a smaller batch moved reward −0.10 → 0.088 |
| `chat_sl`, `sl_basic` | Stopped | Published eval calls `forward`, which this client refuses, so the run stops at step 0. Set `eval_every=0`. A five-step run reduced NLL, chat-sl 2.188 → 1.449 and sl-basic 2.188 → 1.367 |
| On-policy distillation | Stopped | One step, teacher KL 0.00026, then the sampler returned a conflict. A five-step same-datum replay reduced loss on Cortex and on the Tinker API |
| `code_rl` | Stopped | The published 4B command exhausted local memory while loading the task list. SandboxFusion was not running |
| `prompt_distillation` | Not started | The published data file was absent. A five-step run reduced NLL 5.610 → 0.161 |
| `harbor_rl`, `search_tool`, `verifiers_rl` | Not started | The harbor package, chromadb, and the verifiers package were not installed |
| `rl_loop`, `sl_loop` | Not supported | Those scripts call the synchronous training methods |
| `preference/dpo`, `sdft` | Not supported | Custom loss (`forward_backward_custom`) |
| Off-policy distillation | Not supported | `topk_prompt_logprobs` |
| `preference/rlhf` | Not supported | The pipeline resumes a checkpoint and loads the reward model from a path |
| `true_thinking_score` | Not supported | Calls `create_sampling_client_async` |
| `vlm_classifier` | Not supported | Image input |
| `audio` | Not supported | Audio input |
| `max_steps_off_policy` | Not supported | Several samplers live at once. This client keeps one |
| Checkpoint resume | Not supported | A sampler save syncs weights in this process only |

## Call the Tinker client

Import `arctic_platform.integrations.tinker` first, then construct `ServiceClient` with the GPU counts.

```python
from arctic_platform.integrations import tinker

service = tinker.ServiceClient(training_gpus=1, sampling_gpus=1)
training = await service.create_lora_training_client_async("Qwen/Qwen3.5-4B", rank=32)
```
