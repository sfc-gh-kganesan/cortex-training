# Running SkyRL on Cortex Training

[SkyRL](https://github.com/NovaSky-AI/SkyRL)'s GRPO trainer can train against
Cortex Training. SkyRL drives the loop from your machine and Cortex owns the
GPUs, running training and sampling as separate sub-jobs. The driver is
CPU-only, so you do not need a local GPU.

The recipe lives in the Arctic Platform repository, and everything you need to
run it is below. If you are looking for RL inside this repository instead, see
the [Math GRPO recipe](../../recipes/rl/math_grpo/README.md).

## Installation

You need a Cortex account with a PAT, a database and schema, and quota for 8
GPUs: 4 for training and 4 for sampling.

```bash
pip install uv

git clone https://github.com/NovaSky-AI/SkyRL
git -C SkyRL checkout skyrl-v0.3.0
export SKYRL_HOME=$PWD/SkyRL

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
cd Arctic-Platform/recipes/rl/skyrl/simple_gsm8k_cortex
```

SkyRL is a checkout rather than the wheel because the launcher dispatches from
`integrations/arctic_rl/`, which the wheel does not ship. There is no
environment to build: the launcher resolves its own dependencies with
`uv run --isolated` and builds `skyrl` from `$SKYRL_HOME`, so the installed
package always matches the integration code.

Then point the client at your account:

```bash
export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=<db>
export ARCTIC_CORTEX_SCHEMA=<schema>
export ARCTIC_CORTEX_PAT=<pat>
```

## GRPO on GSM8K

Build the dataset, then launch:

```bash
uv run --isolated --no-project --with datasets \
  python ../simple_gsm8k/download_data.py --output_dir ${HOME}/data/gsm8k-skyrl

bash run_qwen3_0.6b_gsm8k_grpo_cortex.sh
```

The first launch spends a couple of minutes downloading wheels and about three
and a half minutes provisioning Cortex sub-jobs, so expect the first training
step roughly six minutes in. Later launches start faster, since `uv` serves the
wheels from cache.

Qwen3-0.6B at the shipped defaults runs one epoch of 233 steps in about two
hours. Held-out `eval/all/pass_at_1` should climb steadily over the
1319-example test set; we measured 0.3033 to 0.7521.

## Training your own config on Cortex

Nothing in the environment selects Cortex. One Hydra flag does:

```
trainer.override_entrypoint=arctic_platform.integrations.skyrl.entrypoint
```

Alongside it, set `trainer.arctic_rl.colocate=false` and
`generator.inference_engine.run_engines_locally=false`, give
`external_server_urls` one placeholder per engine, set
`generator.sampling_params.logprobs=null`, and use
`trainer.arctic_rl.attn_implementation=sdpa`, since the Cortex image ships
without FlashAttention 2.

You can drop everything that tunes local GPUs: `gpu_memory_utilization`,
`zero_stage`, `enable_gradient_checkpointing`, `use_liger`, `use_zorro`, and
the `vllm_*` knobs. Cortex owns those now.

## Stopping a run

Stop with Ctrl-C or `SIGTERM`, and the launcher cancels the Cortex job on its
way out. `kill -9` skips that, and the job keeps its GPUs until something
cancels it.

## More detail

The
[recipe README](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/skyrl/simple_gsm8k_cortex/README.md)
covers hyperparameters, the reasoning behind each flag that differs from the
on-prem sibling recipe, step-by-step metrics from a healthy run, and
troubleshooting.
