# Serve a Training Checkpoint

The [inference example](../../../recipes/inference/README.md) creates a standalone
sampling endpoint, either from original Hugging Face weights or from a
weights-only checkpoint produced by a training run:

```bash
python -m recipes.inference.serve \
  config=/path/to/config.json \
  model=MODEL context=CONTEXT hardware=HARDWARE lora=LORA \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID
```

The SFT and RL examples print a generate or evaluate command with these
arguments filled in after they save. Distillation logs only the checkpoint ID;
see its [serve command](../../../recipes/distillation/README.md#checkpoint-and-evaluation).
To pick the arguments by hand, see
[Serve](../../../recipes/inference/README.md#serve).

Sampling requires a `weights-only` checkpoint; resumable checkpoints carry
optimizer state and are not loadable by the sampling runtime. The sampling job is
independent, so the training job can be cancelled once the checkpoint is saved.
`serve` waits until the endpoint is up and leaves it running; stop it with
`cortex-training cancel JOB_ID`. For the equivalent raw client calls, see
[Start sampling from saved weights](../../reference/rest-api.md#134-start-sampling-from-saved-weights).

Readiness checks, scaling and teardown are not yet documented.

To serve a checkpoint behind an OpenAI-compatible endpoint and stream replies
instead, see [Deploy a Model for Streaming Inference](deploy-model.md).
