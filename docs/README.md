# Cortex Training Documentation

Use Cortex Training to fine-tune, evaluate, and serve open-weight models through
the Cortex Training client.

## Choose a Path

- New users: follow the [getting-started path](getting-started/README.md) from
  prerequisites through a short conversational SFT job.
- Training users: browse the training guides for
  [LoRA](guides/training/lora-and-qlora.md),
  [full fine-tuning](guides/training/full-finetuning.md),
  [reinforcement learning](guides/training/reinforcement-learning.md) and
  [starting from your own weights](guides/training/external-weights.md), plus the
  [runnable examples](../recipes/README.md).
- Inference users: [deploy a model for streaming inference](guides/inference/deploy-model.md)
  behind an OpenAI-compatible endpoint, or
  [serve a training checkpoint](guides/inference/serve-checkpoint.md).
- Framework users: see the [framework examples](../recipes/README.md) for
  RL frameworks integrated with Cortex.
- Operators: see [job management](guides/operations/manage-jobs.md) and
  [logs and metrics](guides/operations/logs-and-metrics.md).
- Client developers: use the [CLI reference](reference/cli.md),
  [Python SDK reference](reference/python-sdk.md), and
  [REST API reference](reference/rest-api.md).

## Scope

These pages document the client, CLI and examples in this repository, and the
service behavior they rely on; the service itself is not part of this
repository.
