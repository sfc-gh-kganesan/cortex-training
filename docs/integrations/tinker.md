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

## Correctness

Published GSM8K arguments, `Qwen/Qwen3.5-4B`, 4 training GPUs and 4 sampling GPUs. Ten steps are logged. The run is still going. The next held-out test is step 20.

| Step | Correct | Reward | Test correct |
|---|---|---|---|
| 0 | 0.617 | 0.549 | 0.544 |
| 1 | 0.689 | 0.630 | |
| 2 | 0.780 | 0.742 | |
| 3 | 0.855 | 0.837 | |
| 4 | 0.908 | 0.900 | |
| 5 | 0.951 | 0.946 | |
| 6 | 0.965 | 0.965 | |
| 7 | 0.962 | 0.962 | |
| 8 | 0.961 | 0.961 | |
| 9 | 0.888 | 0.887 | |

## Recipe support

`tinker-cookbook` 0.5.5. Validated means this client produced the table above. Can run means the recipe's training calls are the on-policy text loop. Not supported means a call this client does not implement.

| Recipe | Status | Note |
|---|---|---|
| `math_rl` GSM8K | Validated | Table above |
| `math_rl` arithmetic, MATH, Polaris, DeepMath | Can run | Same trainer as GSM8K. Not measured |
| `rl_basic` | Can run | GSM8K through the standard trainer, different batch settings |
| `code_rl`, `harbor_rl`, `search_tool`, `verifiers_rl` | Can run | On-policy text loop. The sandbox, search index, or environment is the recipe's |
| `rubric` | Can run | On-policy. The grader is the recipe's |
| `multiplayer_rl` guess-number, twenty-questions, tic-tac-toe | Can run | Twenty Questions opens a second sampler on a base model |
| `preference/shorter` | Can run | Preference score is computed in the recipe |
| `chat_sl`, `sl_basic`, `prompt_distillation` | Can run | Supervised training. Held-out NLL calls `forward`, so set `eval_every=0` |
| On-policy distillation | Can run | Teacher log-probs are supported from a base model |
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

Import `arctic_platform.tinker` first, then construct `ServiceClient` with the GPU counts.

```python
from arctic_platform import tinker

service = tinker.ServiceClient(training_gpus=1, sampling_gpus=1)
training = await service.create_lora_training_client_async("Qwen/Qwen3.5-4B", rank=32)
```
