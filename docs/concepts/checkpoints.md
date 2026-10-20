# Checkpoints and Weight Synchronization

Training checkpoints and sampling weights serve different purposes:

- Resumable training checkpoints can include optimizer state.
- Weights-only checkpoints can initialize a separate sampling job.
- Runtime load replaces weights in an existing training sub-job.
- An external-weight import initializes a new training or sampling job from a
  Hugging Face safetensors directory you upload to a stage; see
  [Start a Job From Your Own Weights](../guides/training/external-weights.md).
- Weight synchronization updates a sampling sub-job from a training sub-job
  during colocated reinforcement learning.

Changing data-parallel size while loading requires optimizer state loading to
be disabled when the job is created. See the
[CLI reference](../reference/cli.md#load-a-checkpoint-into-a-running-job) for
the current commands and constraints.
