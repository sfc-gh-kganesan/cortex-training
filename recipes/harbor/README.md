# Train an agent on the Harbor tasks you evaluate it on

You evaluate your agent with `harbor run` on a directory of
[Harbor](https://github.com/harbor-framework/harbor) tasks, each with
`instruction.md`, `environment/` and `tests/test.sh`. This recipe trains the
model behind that agent on the same tasks. Every RL rollout is a Harbor trial:
the same task, the same sandbox, scored by the same `tests/test.sh`. Nothing is
rewritten into an RL format, and the score you already track in eval becomes
the reward.

`Qwen/Qwen3.5-4B` is trained with online RL (GSPO loss). SkyRL drives the loop
from a CPU machine, and Cortex Training runs 4 training and 4 sampling H200
GPUs. The code is [`recipes/rl/harbor`](https://github.com/Snowflake-AI-Research/Arctic-Platform/tree/main/recipes/rl/harbor)
in Arctic Platform.

> **Tip — Drive training from OpenHands with `/cortex-training-rl`:** The [`cortex-training-rl`](https://github.com/OpenHands/extensions/tree/main/skills/cortex-training-rl) skill is built into OpenHands' public skill registry (`OpenHands/extensions`). In the OpenHands CLI or GUI, ask OpenHands to post-train your model on your Harbor tasks (or invoke `/cortex-training-rl`) and the agent runs preflight checks, launches `run_qwen35_4b.sh`, monitors rewards, and plots them.

## From eval to training

| | `harbor run` (eval) | This recipe (training) |
| --- | --- | --- |
| Tasks | `-p <task dir>` | `TRAIN_HARBOR_TASKS`, `EVAL_HARBOR_TASKS` |
| Agent | `-a <agent>` | OpenHands (`harbor_rl.agent:OpenHandsHarborAgent`) by default, or your own as `HARBOR_AGENT=module:Class` |
| Skills, tools, prompt | Part of your agent | `HARBOR_SKILL`, `OPENHANDS_TOOLS`, `OPENHANDS_SYSTEM_PROMPT_FILE` |
| Sandbox | `--env`, `--ek key=value` | `HARBOR_ENV`, `HARBOR_ENV_KWARGS` (JSON) |
| Score | `tests/test.sh` writes `/logs/verifier/reward.txt` or `reward.json` | The same file, used as the rollout's reward |
| Model | `-m <model>` | The policy being trained, served by the Cortex sampling job |

`HARBOR_AGENT` must run on the driver. Harbor's installed agents
(`openhands-sdk`, `mini-swe-agent`, `claude-code`, ...) run inside the sandbox,
where task code could read the rollout's API key, so the launcher refuses them.

## How it works

1. **Trial:** Harbor's `Trial` starts the sandbox and runs the agent, with its
   LLM pointed at the rollout's model endpoint.
2. **Tokens:** that endpoint records every chat completion and keeps the exact
   sampled token IDs across turns. Assistant turns are trained; prompts and
   tool outputs are masked.
3. **Verifier:** Harbor's `Verifier` runs `tests/test.sh`, and its reward goes
   back to Cortex for the policy update.

## Prerequisites

A Cortex account with capacity for 8 GPUs: 4 for training and 4 for sampling.
Follow [connection setup](../../docs/getting-started/setup.md), then confirm capacity:

```bash
cortex-training login ~/cortex-training-config.json
cortex-training capacity
```

The driver needs no GPU. For the default Snowflake sandbox, it needs
[`crane`](https://github.com/google/go-containerregistry/blob/main/cmd/crane/README.md)
on its `PATH` to mirror task images (no Docker daemon).

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone --branch skyrl-v0.3.0 --depth 1 https://github.com/NovaSky-AI/SkyRL
export SKYRL_HOME=$PWD/SkyRL

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
cd Arctic-Platform
```

The run script installs OpenHands and Harbor itself.

## Configure

The Arctic Platform launcher reads its connection only from `ARCTIC_CORTEX_*`
environment variables, so `cortex-training login` alone is not enough. Export
them from the config JSON you log in with:

```bash
cfg=~/cortex-training-config.json
export ARCTIC_CORTEX_HOST=$(jq -r .host "$cfg")
export ARCTIC_CORTEX_DATABASE=$(jq -r .database "$cfg")
export ARCTIC_CORTEX_SCHEMA=$(jq -r .schema "$cfg")
export ARCTIC_CORTEX_PAT=$(jq -r .pat "$cfg")
```

## 1. Evaluate on the sandbox training uses

Rollouts run in a [Snowflake sandbox](https://github.com/Snowflake-AI-Research/Arctic-Platform/tree/main/arctic_platform/integrations/harbor/snowflake_sandbox)
by default. To get a baseline that training is measured against, evaluate on
that same environment. From the `Arctic-Platform` checkout, `uv run --extra harbor`
provides the pinned `harbor` CLI with that environment importable:

```bash
uv run --extra harbor harbor run -p /path/to/eval_tasks -a <your agent> -m <your model> \
  --env arctic_platform.integrations.harbor.snowflake_sandbox:SnowflakeSandboxEnvironment \
  --ek image_repo=DB.SCHEMA.REPO --ek connection=MY_CONN --ek role=MY_ROLE
```

`image_repo` is the image repository task images are mirrored into, and the
role needs WRITE on it.

## 2. Train on the same tasks

From the `Arctic-Platform` checkout. `ARCTIC_CORTEX_JOB_COMMENT` is the note
that identifies your job.

```bash
export TRAIN_HARBOR_TASKS=/path/to/train_tasks
export EVAL_HARBOR_TASKS=/path/to/eval_tasks
export HARBOR_ENV_KWARGS='{"image_repo": "DB.SCHEMA.REPO", "connection": "MY_CONN", "role": "MY_ROLE"}'
export HARBOR_SKILL=/path/to/SKILL.md   # optional: ":"-separated SKILL.md paths
export ARCTIC_CORTEX_JOB_COMMENT="your name, harbor rl"

bash recipes/rl/harbor/run_qwen35_4b.sh trainer.max_training_steps=50
```

`HARBOR_ENV_KWARGS` takes the same keys as `--ek`: `image_repo`, `connection`,
`role`, `prebuilt_image`, `egress_eai` (for tasks that reach the internet, such
as `git clone` from GitHub), and `image_platform`. A `prebuilt_image` that is
already a sandbox image path (`/DB/SCHEMA/REPO/IMAGE:TAG`) needs neither
`image_repo` nor `crane`. The launcher checks these before it provisions GPUs.

Other sandboxes: `HARBOR_ENV=docker`, `modal`, `daytona` or `e2b`.
`HARBOR_ENV=host` runs tasks as plain subprocesses on the driver, with no
isolation, so it is only for tasks you trust and needs `HARBOR_ALLOW_HOST_EXEC=1`.

## 3. Read the reward

Each step's reward is the `tests/test.sh` score that `harbor run` reports,
averaged over that step's rollouts. The log is
`~/checkpoints/qwen35-4b-harbor-rl/qwen35-4b-harbor-rl.log`, and checkpoints are
written next to it. Each rollout's conversation and reward are saved under
`trajectories/step_*/`.

## Example: SWE-smith code localization as Harbor tasks

`prepare_data.py` exports
[`OpenHands/SWE-smith-py-code-search`](https://huggingface.co/datasets/OpenHands/SWE-smith-py-code-search)
as Harbor task directories. Each task asks for the files, classes and functions
to change for a bug report, and `tests/test.sh` scores file + module + function
F1 (maximum `3.0`):

```bash
uv run recipes/rl/harbor/prepare_data.py \
  --export-swe-smith-to ~/data/swe-smith-harbor-tasks \
  --out ~/data/harbor-rl

export TRAIN_HARBOR_TASKS=~/data/swe-smith-harbor-tasks/train
export EVAL_HARBOR_TASKS=~/data/swe-smith-harbor-tasks/eval
export HARBOR_ENV_KWARGS='{"image_repo": "DB.SCHEMA.REPO", "egress_eai": "GITHUB_ACCESS"}'
export ARCTIC_CORTEX_JOB_COMMENT="your name, harbor localization"

bash recipes/rl/harbor/run_qwen35_4b.sh trainer.max_training_steps=100
```

## Settings

Set these as environment variables before the run script:

| Setting | Variable | Default |
| --- | --- | --- |
| Model | `MODEL` | `Qwen/Qwen3.5-4B` |
| Batch | `BATCH_SIZE` × `N_ROLLOUTS` | 8 prompts × 8 rollouts |
| Turns | `MAX_TURNS` | 10 |
| Context | `MAX_CONTEXT_LENGTH` | 32768 tokens |
| Learning rate | `LR` | 1e-6 |
| Agent tools | `OPENHANDS_TOOLS` | `terminal` plus a finish tool |

Loss settings and everything else pass through as trailing Hydra flags.

## Stop

Ctrl-C, `kill`, or the end of training cancels the Cortex job. A driver killed
with `kill -9` or by the OOM killer cannot cancel it. The server reclaims that
job after 30 idle minutes; set `ARCTIC_CORTEX_IDLE_TIMEOUT_SECONDS` (at least
300) to shorten that.

The [Harbor recipe README](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/harbor/README.md)
in Arctic Platform has every setting and the sandbox details.
