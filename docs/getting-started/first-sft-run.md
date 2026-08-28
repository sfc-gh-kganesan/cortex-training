# Run a Quick SFT Job

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
  max_steps=2
```

The default job config, `configs/qwen3_8b_full.json`, requests four GPUs and
runs full-parameter training, so check capacity first. GPU count, batch shape,
sequence length and LoRA all live in that JSON rather than on the command line --
pass a different one with `job_config=`, for example
`job_config=configs/qwen3_8b_lora.json` for the lighter LoRA path. See
[sizing and batching](../concepts/sizing-and-batching.md).

For a longer run, dataset changes, dense training, and MoE configuration, see
the [conversational SFT recipe](../../recipes/sft/conversational/README.md).

There is no packaged before/after evaluation workflow yet. To confirm the run
worked, check that `train_mean_nll` and `test/nll` fall over a longer run, then
run the sampling command the recipe prints after it saves its checkpoint -- on
the default memorize task the answer should be `Snowflake AI Research`.
