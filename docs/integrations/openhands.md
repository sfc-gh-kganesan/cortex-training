# Example: train an OpenHands agent

This is the worked example for [Bring your own agent](bring-your-own-agent.md).
[OpenHands](https://github.com/OpenHands/software-agent-sdk) runs unchanged.
Its LLM settings point at the rollout's model endpoint, and training improves
the model behind it.

The task is code localization: given a bug report, name the file, class, and
function to change. The agent searches the repository with the `terminal` tool
and submits with `localization_finish`. The reward is file F1 + module F1 +
function F1, so the maximum is 3. Qwen3.5-4B is trained with GRPO. SkyRL
drives the loop from a CPU machine, and Cortex runs 4 training and 4 sampling
GPUs.

## How OpenHands plugs in

| Piece | What this example does |
| --- | --- |
| `submit` | Clones the task repository into a fresh directory and starts an OpenHands conversation in a thread, with the bug report as the user message and the rollout's `base_url` and `api_key` as its LLM settings |
| `get_status` | `running` while the thread runs, then `completed` or `errored` |
| `collect_output` | The conversation and the locations passed to `localization_finish` |
| Grader | File + module + function F1. A rollout that uses all 10 turns without submitting is left out of the loss |

These live in `recipes/rl/openhands/code_localization/openhands_localization`:
the adapter and grader in `rollout.py`, the wiring in `entrypoint.py`.

## Prerequisites

A Cortex account with capacity for 8 GPUs: 4 for training and 4 for sampling.
Follow [connection setup](../getting-started/setup.md), then confirm capacity:

```bash
cortex-training login ~/cortex-training-config.json
cortex-training capacity
```

The driver needs no GPU. It needs disk for one git checkout per in-flight
rollout. The sandbox is that checkout, not a container.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone --branch skyrl-v0.3.0 --depth 1 https://github.com/NovaSky-AI/SkyRL
export SKYRL_HOME=$PWD/SkyRL

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
cd Arctic-Platform
```

The run script installs OpenHands itself.

## Data

The tasks come from the Hugging Face dataset
[`OpenHands/SWE-smith-py-code-search`](https://huggingface.co/datasets/OpenHands/SWE-smith-py-code-search).
From the `Arctic-Platform` checkout, download it and write the training files:

```bash
uv run recipes/rl/openhands/code_localization/prepare_data.py
```

That writes 39,187 training and 100 validation rows to
`~/data/swe-smith-localization`, where the run script reads them. To keep them
elsewhere, pass `--out DIR` and set `DATA_DIR=DIR` when you run.

## Run

From the `Arctic-Platform` checkout. `LOGGER=console` skips Weights & Biases.
`ARCTIC_CORTEX_JOB_COMMENT` is the note that identifies your job.

```bash
export LOGGER=console
export ARCTIC_CORTEX_JOB_COMMENT="your name, openhands localization"

bash recipes/rl/openhands/code_localization/run_qwen35_4b.sh \
  trainer.max_training_steps=100
```

The log is `~/checkpoints/qwen35-4b-openhands-localization/qwen35-4b-openhands-localization.log`.
Each rollout's conversation and reward are saved under `trajectories/` in the
same directory.

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-4B` |
| Batch | 8 prompts × 8 rollouts |
| Turns | 10 |
| Context | 32768 tokens |
| Loss | GSPO, sequence mean, clip 3e-4 / 4e-4, no KL, one update per batch |
| Learning rate | 1e-6 |

## Stop

Ctrl-C, `kill`, or the end of training cancels the Cortex job. A driver killed
with `kill -9` or by the OOM killer cannot cancel it. The server reclaims that
job after 30 idle minutes; set `ARCTIC_CORTEX_IDLE_TIMEOUT_SECONDS` (at least
300) to shorten that.

The
[recipe README](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/openhands/code_localization/README.md)
has every setting and the source commits.
