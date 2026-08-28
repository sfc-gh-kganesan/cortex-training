# Weights & Biases

The SFT and Math GRPO recipes can mirror their local metrics to Weights &
Biases. Install the client and point it at your instance:

```bash
uv pip install wandb
export WANDB_API_KEY=...
# optional: only if you are not using W&B Cloud
# export WANDB_BASE_URL=https://your-wandb-host
```

Then set a project on the train command:

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json wandb_project=cortex-training
```

Use `wandb_name` to identify a run. Metrics are always written under `log_path`
as well, so a run without Weights & Biases still records them locally.

A shared metric naming convention and validated dashboards for loss, reward,
evaluation, and KL are not yet defined.
