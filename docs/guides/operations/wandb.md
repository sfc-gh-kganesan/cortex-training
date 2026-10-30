# Experiment Tracking

The training examples (SFT, Math GRPO and distillation) always write metrics
under `log_path`, and can also mirror them to Weights & Biases or Snowflake
experiment tracking.

## Weights & Biases

```bash
uv pip install wandb
export WANDB_API_KEY=YOUR_WANDB_API_KEY
# Only for a self-hosted W&B server:
# export WANDB_BASE_URL=https://wandb.example.com
```

Then set a project on the train command, and optionally `wandb_name` to name
the run:

```bash
python -m recipes.sft_quickstart.conversational.train \
  config=/path/to/config.json wandb_project=cortex-training
```

## Snowflake experiment tracking

```bash
uv pip install "snowflake-ml-python>=1.19.0"
```

Then pass `sf_tracking=true` on the train command. Results appear in Snowsight
under **AI & ML > Experiments**; see the
[Snowflake ML Experiments docs](https://docs.snowflake.com/en/developer-guide/snowflake-ml/experiments).

A shared metric naming convention and validated dashboards for loss, reward,
evaluation, and KL are not yet defined.
