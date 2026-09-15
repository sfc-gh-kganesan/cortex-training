# SkyRL

[SkyRL GRPO on GSM8K](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/skyrl/simple_gsm8k_cortex/README.md)
runs [SkyRL](https://github.com/NovaSky-AI/SkyRL)'s GRPO trainer against Cortex
Training. SkyRL drives the loop from a CPU driver; Cortex owns the GPUs in
training and sampling sub-jobs.

The recipe lives in the Arctic Platform repository, not here, and it is not
`recipes.rl.math_grpo` — SkyRL's trainer and entry point drive it, and its
config is SkyRL's. For the in-repo RL path use the
[Math GRPO recipe](../../recipes/rl/math_grpo/README.md).

## Run it

The driver runs on CPU, so no local GPU is needed. It is not a single pip
install, though: the recipe, its pinned requirements and the launcher all live
in the Arctic Platform repository, and SkyRL has to be a checkout rather than
the wheel because the launcher dispatches from `integrations/arctic_rl/`, which
the `skyrl` package does not ship.

```bash
# 1. Recipe, requirements and launcher
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
cd Arctic-Platform

# 2. Environment
conda create -y -n skyrl_arl python=3.12.13 && conda activate skyrl_arl
pip install -q uv
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 -U
uv pip install -r recipes/rl/skyrl/simple_gsm8k/requirements.txt \
               --override recipes/rl/skyrl/simple_gsm8k/overrides.txt

# 3. SkyRL at the tag
git clone https://github.com/NovaSky-AI/SkyRL ../SkyRL
git -C ../SkyRL checkout skyrl-v0.3.0
export SKYRL_HOME=$(cd ../SkyRL && pwd)

# 4. Required at this tag — see below
cd recipes/rl/skyrl/simple_gsm8k_cortex
sed -i '/generator.inference_engine.remote_urls=/d' run_qwen3_0.6b_gsm8k_grpo_cortex.sh

# 5. Cortex account
export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=<database>
export ARCTIC_CORTEX_SCHEMA=<schema>
export ARCTIC_CORTEX_PAT=<pat>

# 6. Dataset (SkyRL schema; the launcher refuses to start without it)
python ../simple_gsm8k/download_data.py --output_dir ${HOME}/data/gsm8k-skyrl

# 7. Launch
bash run_qwen3_0.6b_gsm8k_grpo_cortex.sh
```

Notes on the non-obvious steps:

- Step 2 uses the *sibling* recipe's requirements. `simple_gsm8k_cortex/` ships
  none of its own and shares the environment. `arctic-platform[cortex]` alone
  installs the transport and retry stack, not SkyRL, so the launcher cannot
  start from it.
- Step 3's checkout shadows the pip-installed `skyrl`, because the launcher puts
  `$SKYRL_HOME` first on `PYTHONPATH`. The tag governs the whole library, not
  just `integrations/arctic_rl/`.
- Step 4 is mandatory at `skyrl-v0.3.0`, which removed
  `generator.inference_engine.remote_urls` and rejects it while parsing config.
  The launcher already passes `external_server_urls`, so nothing replaces it.
  Without this the run dies before it starts.
- Step 5 does not select the backend. The launcher passes
  `trainer.override_entrypoint=arctic_platform.integrations.skyrl.entrypoint`,
  and naming that entrypoint is what routes training and sampling to Cortex.
- Step 7 takes all 8 GPUs of the per-account cap for about two hours. Stop it
  with Ctrl-C or `SIGTERM` so the launcher's trap releases the Cortex job; a
  `kill -9` leaves the job holding its GPUs.

## Reported results

Qwen3-0.6B on GSM8K at shipped defaults, one epoch of 233 steps on 4 training
and 4 sampling GPUs, moving held-out `eval/all/pass_at_1` over the
1319-example test set from roughly 0.29 to roughly 0.75 in about two hours.

Two single runs, neither a guarantee. 0.2942 to 0.7680 in 2h03m on the fork
commit recorded in the recipe README, measured with Arctic Platform's client
rather than this one; and 0.3033 to 0.7521 in 1h58m on `skyrl-v0.3.0` with the
edit above, which is the checkout this page describes.

The recipe README is the source of truth for hyperparameters, hardware,
expected metrics, and troubleshooting — including the per-account GPU cap this
configuration saturates, and the fact that a job holds its GPUs until something
cancels it.
