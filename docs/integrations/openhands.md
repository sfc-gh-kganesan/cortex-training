# Train an OpenHands agent on Cortex

[OpenHands](https://github.com/OpenHands/software-agent-sdk) can be the
harness for a GRPO run. SkyRL drives the loop from a CPU machine. Cortex runs
training and sampling as separate sub-jobs. The agent, the reward, and the
chat proxy that sits in front of Cortex sampling live in Arctic Platform, in
`arctic_platform.integrations.openhands`.

The example localizes a code change. The agent searches with the `terminal`
tool and submits with `localization_finish`. The reward is file F1 plus
module F1 plus function F1, maximum 3. A rollout that never submits inside
the turn budget is left out of the loss.

## Prerequisites

A Cortex account with capacity for 8 GPUs: 4 for training and 4 for sampling.
Follow [connection setup](../getting-started/setup.md), then confirm capacity:

```bash
cortex-training login ~/cortex-training-config.json
cortex-training capacity
```

The driver needs no GPU. It needs disk for one local git checkout per
in-flight rollout. The sandbox is that checkout, not a container.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/NovaSky-AI/SkyRL
git -C SkyRL checkout skyrl-v0.3.0
export SKYRL_HOME=$PWD/SkyRL

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
```

Copy the SWE-smith parquets (`train.parquet` and `validation.parquet`) to
`~/data/swe-smith-localization`. Those files are the `data/swe_smith` split
from [codescout@abab719](https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2).
They are not in either of these repositories.

## Run

From the Arctic Platform checkout:

```bash
bash recipes/rl/openhands/code_localization/run_qwen35_4b.sh \
  trainer.max_training_steps=100
```

The launcher installs OpenHands SDK `85ecfd93` and selects
`trainer.override_entrypoint=arctic_platform.integrations.openhands.entrypoint`.

| Knob | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-4B` |
| Batch | 8 prompts × 8 rollouts |
| Turns | 10 |
| Context | 40960 |
| Loss | GSPO, sequence mean, clip 3e-4 / 4e-4, no KL, one update per batch |
| Learning rate | 1e-6 |

Liger kernels are off. That fused path rejects this sampler's outputs. ZoRRo
is off because a multi-turn response is not one fixed length.

## Sources

The finish tool, the F1 reward, the chat proxy, and the rollout are adapted
from CodeScout at `abab719`. Qwen3.5's XML tool calls are from `8184b42` in
that repo. The prompt names `terminal`, which is the tool OpenHands
registers. The CodeScout prompt names `bash`, which is not registered.

The
[recipe README](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/openhands/code_localization/README.md)
has the full knob list and the stop procedure. Cancelling the local process
does not release the GPUs. The launcher cancels the Cortex job on exit.
