# SkyRL

[SkyRL GRPO on GSM8K](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/skyrl/simple_gsm8k_cortex/README.md)
runs [SkyRL](https://github.com/NovaSky-AI/SkyRL)'s GRPO trainer with training
and sampling dispatched to Cortex Training sub-jobs. The SkyRL driver stays on
CPU and Cortex owns the GPUs.

That example lives in the Arctic Platform repository rather than here, and it is
not `recipes.rl.math_grpo`: SkyRL's own trainer and entry point drive it, so
nothing under this repository's `recipes/` runs it and its configuration is
SkyRL's rather than a Cortex recipe `Config`. For the in-repo RL path, use the
[Math GRPO recipe](../../recipes/rl/math_grpo/README.md).

## Install

The driver needs no local GPU dependencies:

```bash
pip install 'arctic-platform[cortex]'
```

SkyRL has to be a checkout rather than the released wheel, because the
integration code the launcher dispatches from, `integrations/arctic_rl/`, is not
shipped in the `skyrl` package:

```bash
git clone https://github.com/Snowflake-AI-Research/SkyRL
cd SkyRL && git checkout 7636101a71f1849b6127ee10232fb277d2f31174 && cd ..
export SKYRL_HOME=$PWD/SkyRL
```

Point the client at your account:

```bash
export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=<database>
export ARCTIC_CORTEX_SCHEMA=<schema>
export ARCTIC_CORTEX_PAT=<pat>
```

No environment variable selects Cortex as the backend. The launcher passes
`trainer.override_entrypoint=arctic_platform.integrations.skyrl.entrypoint`, and
naming that entrypoint is what routes training and sampling to Cortex.

## Reported results

The recipe reports Qwen3-0.6B on GSM8K at its shipped defaults, one epoch of 233
steps across 4 training and 4 sampling GPUs, moving held-out `pass@1` from 0.2942
to 0.7680 in 2h03m wall-clock. That is one run recorded upstream on 2026-08-31,
not a guarantee, and it was measured against Arctic Platform's client rather than
this one.

The recipe README is the source of truth for hyperparameters, hardware, expected
metrics, and troubleshooting, including the operational limits that apply to any
Cortex RL job -- notably the per-account GPU cap that this configuration
saturates, and the fact that a job holds its GPUs until something cancels it.
