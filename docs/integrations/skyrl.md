# SkyRL

[SkyRL GRPO on GSM8K](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/skyrl/simple_gsm8k_cortex/README.md)
runs [SkyRL](https://github.com/NovaSky-AI/SkyRL)'s GRPO trainer against Cortex
Training. SkyRL drives the loop from a CPU driver; Cortex owns the GPUs in
training and sampling sub-jobs.

The recipe lives in the Arctic Platform repository, not here, and it is not
`recipes.rl.math_grpo` — SkyRL's trainer and entry point drive it, and its
config is SkyRL's. For the in-repo RL path use the
[Math GRPO recipe](../../recipes/rl/math_grpo/README.md).

## Setup

Follow the
[recipe README](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/skyrl/simple_gsm8k_cortex/README.md)
for the environment, dataset and launch command. It owns those instructions;
this page only records what is specific to running against Cortex.

Two things are worth knowing before you start.

**SkyRL is a checkout at the upstream tag, and there is no environment to
build.** The launcher dispatches from `integrations/arctic_rl/`, which the
`skyrl` wheel does not ship, so it needs an on-disk clone:

```bash
pip install uv

git clone https://github.com/NovaSky-AI/SkyRL
git -C SkyRL checkout skyrl-v0.3.0
export SKYRL_HOME=$PWD/SkyRL
```

That is the whole setup. The launcher resolves its own dependencies through
`uv run --isolated` — the pattern upstream's `integrations/arctic_rl/examples/`
launchers use — and builds `skyrl` from `$SKYRL_HOME`, so the installed package
cannot drift from the integration code. First launch downloads wheels; after
that `uv` serves them from cache.

**Nothing in the environment selects Cortex.** The launcher passes
`trainer.override_entrypoint=arctic_platform.integrations.skyrl.entrypoint`, and
naming that entrypoint is what routes training and sampling to Cortex. Point the
client at your account with `ARCTIC_CORTEX_HOST`, `ARCTIC_CORTEX_DATABASE`,
`ARCTIC_CORTEX_SCHEMA` and `ARCTIC_CORTEX_PAT`.

The run saturates the per-account GPU cap for about two hours. Stop it with
Ctrl-C or `SIGTERM` so the launcher's trap releases the Cortex job; `kill -9`
leaves the job holding its GPUs.

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
