# Run GRPO from SkyRL

[SkyRL](https://github.com/NovaSky-AI/SkyRL) can run GRPO against Cortex
Training. SkyRL's trainer drives the loop from your machine while Cortex runs
training and sampling as separate sub-jobs, so the driver needs no local GPU.

The example below trains Qwen3-0.6B on GSM8K. It lives in the Arctic Platform
repository rather than under `recipes/` here, and SkyRL's entry point drives it
rather than this client's.

## Prerequisites

A Cortex account with a PAT, a database and schema, and capacity for 8 GPUs:
4 for training and 4 for sampling. Check with `cortex-training capacity`.

The driver machine needs no GPU, about 16 GB of free RAM, and a few gigabytes
of disk for the `uv` cache. Ray reports too little memory as an OOM kill during
the evaluation that precedes training, rather than as a clear error.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # if uv is not already on your PATH

git clone https://github.com/NovaSky-AI/SkyRL
git -C SkyRL checkout skyrl-v0.3.0
export SKYRL_HOME=$PWD/SkyRL

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
cd Arctic-Platform/recipes/rl/skyrl/simple_gsm8k_cortex
```

SkyRL needs a checkout rather than the wheel because the launcher dispatches
from `integrations/arctic_rl/`, which the wheel does not ship. There is no
environment to create: the launcher resolves its own dependencies with
`uv run --isolated` and builds `skyrl` from `$SKYRL_HOME`.

## Configure

```bash
export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=<db>
export ARCTIC_CORTEX_SCHEMA=<schema>
export ARCTIC_CORTEX_PAT=<pat>
```

## Run

```bash
uv run --isolated --no-project --with datasets \
  python ../simple_gsm8k/download_data.py --output_dir ${HOME}/data/gsm8k-skyrl

bash run_qwen3_0.6b_gsm8k_grpo_cortex.sh
```

The first training step lands about twenty minutes in on a fresh machine:
roughly six minutes of wheel downloads, four to seven provisioning Cortex
sub-jobs, two tokenizing the dataset, and four on the evaluation that runs
before training. Later launches reach it about five minutes sooner, once `uv`
serves the wheels from cache.

## Results

One epoch is 233 steps and takes two to three hours. `eval/all/pass_at_1` is
scored on the 1319-example GSM8K test set every 10 steps, and should track
roughly:

| Step | pass@1 |
|---|---|
| 1 | 0.31 |
| 20 | 0.50 |
| 40 | 0.67 |
| 60 | 0.71 |
| 80 | 0.71 |
| 100 | 0.72 |
| 120 | 0.73 |

A full epoch reaches about 0.75. Most of the gain arrives in the first 40
steps; after that the curve flattens.

Two per-step metrics are worth watching. Training reward rises from about 0.25
to 0.75 over the same span. Policy entropy falls from about 0.53 to 0.38, which
is the policy sharpening as it learns; entropy collapsing toward zero instead
means the learning rate is too high.

## Resume a stopped run

The launcher checkpoints every ten steps but starts fresh by default. To
continue from the last checkpoint:

```bash
bash run_qwen3_0.6b_gsm8k_grpo_cortex.sh trainer.resume_mode=latest
```

Any argument you pass is forwarded to the trainer.

## More detail

The
[recipe README](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/skyrl/simple_gsm8k_cortex/README.md)
covers hyperparameters, the flags that differ from the on-prem sibling recipe,
and troubleshooting.
