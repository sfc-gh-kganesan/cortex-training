# Cortex Training Recipes

Recipes are runnable, end-to-end workflows organized by task. Each recipe owns
its code, documentation, metadata, optional notebooks, and the JSON job configs
under its `configs/` directory.

## Available Recipes

| Recipe | Method | Dataset | Status |
|---|---|---|---|
| [Conversational SFT](sft/conversational/README.md) | LoRA or full-parameter SFT | Hugging Face chat datasets | Runnable |
| [Math GRPO](rl/math_grpo/README.md) | Reinforcement learning | Hendrycks MATH and MATH-500 | Runnable |
| [Inference endpoint](inference/README.md) | Serve, generate, eval | Open weights, checkpoints, MATH-500 | Runnable |

Outside this table, GRPO can also be run from SkyRL against Cortex Training; see
the [SkyRL integration](../docs/integrations/skyrl.md). That example is not a
recipe here and does not use the entry points below.

## Prerequisites

Install the client and recipe dependencies from the repository root:

```bash
uv pip install -e .
uv pip install 'tinker-cookbook[math-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
uv pip install wandb
```

(`pip install ...` works in place of `uv pip install ...` if you prefer.)

Create a local Snowflake connection file from
`examples/config/connection.json.template` (account host and PAT).

To log SFT or GRPO metrics to Weights & Biases, set a project on the train
command (`wandb_project=...`) and export:

```bash
export WANDB_API_KEY=...
export WANDB_BASE_URL=...
```

## Running Recipes

Recipes are Python modules so they can share code without path manipulation:

```bash
python -m recipes.sft.conversational.train config=/path/to/config.json
python -m recipes.rl.math_grpo.train config=/path/to/config.json
python -m recipes.inference.serve config=/path/to/config.json
```

`config=` is the Snowflake PAT/connection file. See each recipe README for
hardware, expected metrics, common variations, and the JSON job configs
loaded from `configs/` (SFT and GRPO examples, plus inference configs for
every catalog model).

## Recipe Contract

Every runnable recipe should provide:

- A README with outcome, prerequisites, hardware, commands, expected results,
  CLI knobs, JSON job configs, evaluation, and troubleshooting
- `recipe.yaml` metadata used by the compatibility catalog
- A runnable entry point
- `last_validated` in `recipe.yaml`: the date of the last verified run, or `null`
  when the recipe has not been validated since it last changed
- Tests for local validation logic where practical
- Attribution when adapted from an upstream cookbook or framework

See [the recipe template](../docs/contributing/recipe-template.md) before adding
a new workflow.
