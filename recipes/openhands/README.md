# Train an OpenHands or Harbor agent

This is the worked recipe for [Bring your own agent](../bring-your-own-agent.md).
[OpenHands](https://github.com/OpenHands/software-agent-sdk) runs unchanged.
Its LLM settings point at the rollout's model endpoint, and training improves
the model behind it.

You can run this workflow in two ways:
1. **On any [Harbor](https://github.com/harbor-framework/harbor) task directories (`recipes/rl/harbor`):** Point `TRAIN_HARBOR_TASKS` and `EVAL_HARBOR_TASKS` at any Harbor tasks (`instruction.md`, `environment/`, `tests/test.sh`), optionally attach a `SKILL.md` (`HARBOR_SKILL`), and run rollouts inside `--env snowflake` ([Snowflake Cortex Sandboxes](https://harborframework.com/docs/sandboxes/pre-integrated-sandboxes)), `docker`, `modal`, `daytona`, or `HostEnvironment`.
2. **Direct OpenHands SDK code localization (`recipes/rl/openhands/code_localization`):** Given a bug report, name the file, class, and function to change. The agent searches the repository with the `terminal` tool and submits with `localization_finish`. The reward is file F1 + module F1 + function F1 (maximum `3.0`).

In both modes, `Qwen/Qwen3.5-4B` is trained with online RL (GSPO loss; see [Run](#run)). SkyRL drives the loop from a CPU machine, and Cortex Training runs 4 training and 4 sampling H200 GPUs.

> **Tip — Drive training from OpenHands with `/cortex-training-rl`:** The [`cortex-training-rl`](https://github.com/OpenHands/extensions/tree/main/skills/cortex-training-rl) skill is built into OpenHands' public skill registry (`OpenHands/extensions`). In the OpenHands CLI or GUI, simply ask OpenHands to post-train your model on your Harbor tasks (or invoke `/cortex-training-rl`) and the agent will run preflight checks, launch `run_qwen35_4b.sh`, monitor rewards, and generate `reward_graph.svg` automatically.

## How OpenHands plugs in

| Piece | Direct OpenHands (`recipes/rl/openhands/code_localization`) | With Harbor (`recipes/rl/harbor`) |
| --- | --- | --- |
| `submit` | Clones the task repository into a fresh directory and starts an OpenHands conversation in a thread pointed at the rollout's `base_url` and `api_key` | Runs Harbor's `Trial.create(trial_cfg)` $\to$ `trial.run()` with your chosen `--agent` (`HARBOR_AGENT`) and `--env` (`HARBOR_ENV`) pointed at `base_url` and `api_key` |
| `get_status` | `running` while the thread runs, then `completed` or `errored` | `running` while the Harbor trial runs, then `completed` or `errored` |
| `collect_output` | The conversation and the locations passed to `localization_finish` | The Harbor `TrialResult`, verifier rewards, and trajectory |
| Grader | File + module + function F1 (`rollout.py`) | Harbor's `Verifier` (`tests/test.sh` writing `/logs/verifier/reward.txt` or `reward.json`) |

## Prerequisites

A Cortex account with capacity for 8 GPUs: 4 for training and 4 for sampling.
Follow [connection setup](../../docs/getting-started/setup.md), then confirm capacity:

```bash
cortex-training login ~/cortex-training-config.json
cortex-training capacity
```

The driver needs no GPU.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone --branch skyrl-v0.3.0 --depth 1 https://github.com/NovaSky-AI/SkyRL
export SKYRL_HOME=$PWD/SkyRL

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
cd Arctic-Platform
```

The run script installs OpenHands (and Harbor) itself.

## Configure

The Arctic Platform launcher reads its connection from `ARCTIC_CORTEX_*`
environment variables (or `~/cortex-training-config.json` saved by `cortex-training login`):

```bash
cfg=~/cortex-training-config.json
export ARCTIC_CORTEX_HOST=$(jq -r .host "$cfg")
export ARCTIC_CORTEX_DATABASE=$(jq -r .database "$cfg")
export ARCTIC_CORTEX_SCHEMA=$(jq -r .schema "$cfg")
export ARCTIC_CORTEX_PAT=$(jq -r .pat "$cfg")
```

## Option 1: Run on any Harbor tasks (`recipes/rl/harbor`)

Point `TRAIN_HARBOR_TASKS` and `EVAL_HARBOR_TASKS` at any Harbor task directories (or export SWE-smith tasks into Harbor format):

```bash
# Export SWE-smith code localization into Harbor task directories (or point at your own Harbor tasks)
uv run recipes/rl/harbor/prepare_data.py \
  --export-swe-smith-to ~/data/swe-smith-harbor-tasks \
  --out ~/data/harbor-rl

export TRAIN_HARBOR_TASKS=~/data/swe-smith-harbor-tasks/train
export EVAL_HARBOR_TASKS=~/data/swe-smith-harbor-tasks/eval
export HARBOR_SKILL=/path/to/SKILL.md   # optional: ":"-separated SKILL.md paths
export LOGGER=console
export ARCTIC_CORTEX_JOB_COMMENT="your name, harbor rl"

bash recipes/rl/harbor/run_qwen35_4b.sh \
  trainer.max_training_steps=50
```

To run rollouts inside Snowflake Cortex Sandboxes (`--env snowflake`), set `export HARBOR_ENV=snowflake`.

## Option 2: Run direct OpenHands code localization (`recipes/rl/openhands/code_localization`)

Download [`OpenHands/SWE-smith-py-code-search`](https://huggingface.co/datasets/OpenHands/SWE-smith-py-code-search) and write the training files:

```bash
uv run recipes/rl/openhands/code_localization/prepare_data.py
```

That writes 39,187 training and 100 validation rows to
`~/data/swe-smith-localization`, where the run script reads them. To keep them
elsewhere, pass `--out DIR` and set `DATA_DIR=DIR` when you run.

### Run

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
[OpenHands recipe README](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/openhands/code_localization/README.md)
and
[Harbor recipe README](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/harbor/README.md)
have every setting and source commit.
