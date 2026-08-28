# Run Your First SFT Job

The conversational SFT recipe is the current shortest end-to-end training path.
It creates a training job, loads a chat dataset, submits forward/backward
batches, applies optimizer steps, and cancels the job when complete.

Install the recipe dependency:

```bash
uv pip install 'tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

Start with a short run:

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=configs/qwen3_8b_lora.json \
  max_steps=2
```

This short path uses LoRA (`configs/qwen3_8b_lora.json`) on four GPUs. Check
`cortex-training capacity` first. GPU count, batch shape, sequence length, and
LoRA all live in the job-config JSON rather than on the command line -- pass
`job_config=configs/qwen3_8b_full.json` for full-parameter training. See
[sizing and batching](../concepts/sizing-and-batching.md).

For a longer run, dataset changes, dense training, and MoE configuration, see
the [conversational SFT recipe](../../recipes/sft/conversational/README.md).

There is no packaged before/after evaluation workflow yet. The recipe logs
`train_nll` (not `train_mean_nll` / `test/nll`). On a longer run that value
should fall; there is no held-out NLL metric in this recipe. After save, run
the generate command the recipe prints -- on the default memorize task the
answer should be `Snowflake AI Research`.
