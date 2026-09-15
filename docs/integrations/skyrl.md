# SkyRL

[SkyRL GRPO on GSM8K](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/skyrl/simple_gsm8k_cortex/README.md)
runs [SkyRL](https://github.com/NovaSky-AI/SkyRL)'s GRPO trainer against Cortex
Training. SkyRL drives the loop from a CPU driver; Cortex owns the GPUs in
training and sampling sub-jobs.

The recipe lives in the Arctic Platform repository, not here, and it is not
`recipes.rl.math_grpo` — SkyRL's trainer and entry point drive it, and its
config is SkyRL's. For the in-repo RL path use the
[Math GRPO recipe](../../recipes/rl/math_grpo/README.md).

## Install

The driver needs no GPU dependencies:

```bash
pip install 'arctic-platform[cortex]'
```

SkyRL must be a checkout rather than the wheel, because the launcher dispatches
from `integrations/arctic_rl/`, which the `skyrl` package does not ship:

```bash
git clone https://github.com/NovaSky-AI/SkyRL
cd SkyRL && git checkout skyrl-v0.3.0 && cd ..
export SKYRL_HOME=$PWD/SkyRL
```

At this tag, delete the `generator.inference_engine.remote_urls=` line from the
recipe launcher. `skyrl-v0.3.0` removed that key and rejects it while parsing
config, so the run dies before it starts. The launcher already passes
`external_server_urls`, so nothing replaces it.

Then point the client at your account:

```bash
export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=<database>
export ARCTIC_CORTEX_SCHEMA=<schema>
export ARCTIC_CORTEX_PAT=<pat>
```

No environment variable selects Cortex. The launcher passes
`trainer.override_entrypoint=arctic_platform.integrations.skyrl.entrypoint`, and
naming that entrypoint is what routes training and sampling to Cortex.

## Reported results

Qwen3-0.6B on GSM8K at shipped defaults, one epoch of 233 steps on 4 training
and 4 sampling GPUs: held-out `eval/all/pass_at_1` over the 1319-example test
set moves from 0.2942 to 0.7680 in 2h03m.

That is a single run, recorded in the recipe README and measured on the fork
commit the Arctic Platform recipes pin, using that repository's client rather
than this one.

The recipe README is the source of truth for hyperparameters, hardware,
expected metrics, and troubleshooting — including the per-account GPU cap this
configuration saturates, and the fact that a job holds its GPUs until something
cancels it.
