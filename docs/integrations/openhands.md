# Train an OpenHands agent on Cortex

[OpenHands](https://github.com/OpenHands/software-agent-sdk) is the harness
for this GRPO run. SkyRL drives the loop from a CPU machine. Cortex runs
training and sampling. The agent, reward, and chat proxy live in the recipe
package `openhands_localization`, next to the launcher.

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

git clone --branch skyrl-v0.3.0 --depth 1 https://github.com/NovaSky-AI/SkyRL
export SKYRL_HOME=$PWD/SkyRL

git clone --branch karthik/openhands-cortex https://github.com/Snowflake-AI-Research/Arctic-Platform
```

Copy the SWE-smith parquets (`train.parquet` and `validation.parquet`) to
`~/data/swe-smith-localization`. Those files are the `data/swe_smith` split
from [codescout@abab719](https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2).
They are not in either of these repositories.

## Run

From the Arctic Platform checkout. `LOGGER=console` skips Weights & Biases.
`CORTEX_JOB_COMMENT` is the note that identifies the job.

```bash
export LOGGER=console
export CORTEX_JOB_COMMENT="your name, openhands localization"

bash recipes/rl/openhands/code_localization/run_qwen35_4b.sh \
  trainer.max_training_steps=100
```

The
[recipe README](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/karthik/openhands-cortex/recipes/rl/openhands/code_localization/README.md)
has the install and the stop procedure. The launcher cancels the Cortex job
on exit. If the process dies first, cancel the saved id:

```bash
PYTHONPATH=recipes/rl/openhands/code_localization \
  python -m openhands_localization.release "$(cat "$CKPT_DIR/cortex_job_id")"
```

| Knob | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-4B` |
| Batch | 8 prompts × 8 rollouts |
| Turns | 10 |
| Context | 32768 |
| Loss | GSPO, sequence mean, clip 3e-4 / 4e-4, no KL, one update per batch |
| Learning rate | 1e-6 |

Liger is off. Context is 32768 tokens. See the recipe README for the knob
list and the source commits.

