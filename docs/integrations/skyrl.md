# SkyRL

[SkyRL](https://github.com/NovaSky-AI/SkyRL)'s GRPO trainer, training against
Cortex Training. SkyRL drives the loop from a CPU-only driver; Cortex owns the
GPUs in training and sampling sub-jobs.

This is not `recipes.rl.math_grpo` — SkyRL's trainer and config drive it. For
the in-repo RL path use the
[Math GRPO recipe](../../recipes/rl/math_grpo/README.md).

You need a Cortex account (host, database, schema, PAT) and quota for 8 GPUs —
4 training and 4 sampling, which saturates the per-account cap. No local GPU.

## Run it

```bash
pip install uv

git clone https://github.com/NovaSky-AI/SkyRL
git -C SkyRL checkout skyrl-v0.3.0
export SKYRL_HOME=$PWD/SkyRL

git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
cd Arctic-Platform/recipes/rl/skyrl/simple_gsm8k_cortex

export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=<db>
export ARCTIC_CORTEX_SCHEMA=<schema>
export ARCTIC_CORTEX_PAT=<pat>

uv run --isolated --no-project --with datasets \
  python ../simple_gsm8k/download_data.py --output_dir ${HOME}/data/gsm8k-skyrl

bash run_qwen3_0.6b_gsm8k_grpo_cortex.sh
```

First launch spends a couple of minutes resolving wheels, then about three and
a half minutes provisioning Cortex sub-jobs before the first training step.

Three things that are not obvious:

* **SkyRL is a checkout, not the wheel.** The launcher dispatches from
  `integrations/arctic_rl/`, which the wheel does not ship. There is still no
  environment to build: the launcher resolves its dependencies through
  `uv run --isolated` and builds `skyrl` from `$SKYRL_HOME`, so the installed
  package cannot drift from the code it dispatches through. The dataset step
  runs under `uv` for the same reason — nothing was installed for it.
* **Nothing in the environment selects Cortex.** The launcher passes
  `trainer.override_entrypoint=arctic_platform.integrations.skyrl.entrypoint`,
  and naming that entrypoint is what routes training and sampling to Cortex.
* **Stop with Ctrl-C or `SIGTERM`**, so the launcher's trap cancels the Cortex
  job. `kill -9` skips the trap and leaves the job holding all 8 GPUs; the next
  launch then fails on the per-account cap with a 429 that never mentions your
  previous run.

## What to expect

One epoch, 233 steps, about two hours. Held-out `eval/all/pass_at_1` over the
1319-example test set moves from roughly 0.29 to roughly 0.75.

Two single runs, neither a guarantee. 0.2942 to 0.7680 in 2h03m on the
Snowflake fork commit that the recipe's parent README pins for the FSDP
recipes, measured with Arctic Platform's client rather than this one; and
0.3033 to 0.7521 in 1h58m on upstream `skyrl-v0.3.0`, the checkout above.

The
[recipe README](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/skyrl/simple_gsm8k_cortex/README.md)
is the source of truth for hyperparameters, the rationale behind each flag that
differs from the on-prem sibling recipe, what a healthy run looks like step by
step, and troubleshooting.
