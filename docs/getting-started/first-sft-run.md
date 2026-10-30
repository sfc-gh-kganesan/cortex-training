# Run a Quick SFT Job

The conversational SFT example is the shortest end-to-end training path. It
creates a training job, loads a chat dataset, submits forward/backward batches,
applies optimizer steps, saves a checkpoint, and cancels the job.

Install the example dependency:

```bash
uv pip install 'tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

Start with a short run. It uses your default connection profile; pass
`connection=<name>` for another profile, or `config=<file>` for a
[JSON connection file](setup.md#json-connection-file):

```bash
python -m recipes.sft_quickstart.conversational.train \
  max_steps=2
```

The default job is full-parameter `Qwen/Qwen3-8B` at standard context on four
H200 GPUs, so check capacity first with `cortex-training capacity`. Add
`lora=true` for the lighter LoRA path, or `model=` for another model; see
[the SFT example](../../recipes/sft_quickstart/README.md) for every option and
[sizing and batching](../concepts/sizing-and-batching.md) for how the batch
fields interact.

To confirm the run worked, check that `train_nll` falls over a longer run, then
run the generate command the example prints after it saves its checkpoint. On
the default dataset the answer should be `Snowflake AI Research`. To score the
checkpoint instead, use the
[inference example's evaluate command](../../recipes/inference/README.md#evaluate).
