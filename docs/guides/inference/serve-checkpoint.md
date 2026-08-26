# Serve a Training Checkpoint

The [inference recipe](../../../recipes/inference/README.md) creates a standalone
sampling endpoint, either from original Hugging Face weights or from a
weights-only checkpoint produced by a training run:

```bash
python -m recipes.inference.serve \
  config=/path/to/config.json \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID
```

Sampling requires a `weights-only` checkpoint; resumable checkpoints carry
optimizer state and are not loadable by the sampling runtime. The sampling job is
independent, so the training job can be cancelled once the checkpoint is saved.

GPU count and vLLM settings live in the job-config JSON under
`recipes/inference/configs/`. For the equivalent raw client calls, see
[Start Sampling From A Training Checkpoint](../../reference/cli.md#start-sampling-from-a-training-checkpoint).

Readiness checks, scaling and teardown are not yet documented.
