# Run Your First SFT Job

Complete [set up the client](setup.md) first so you have a clone, an editable install, and a working `cortex-training login`.

The conversational SFT recipe is the current shortest end-to-end training path.
It creates a training job, loads a chat dataset, submits forward/backward
batches, applies optimizer steps, and cancels the job when complete.

Install the recipe dependency:

```bash
uv pip install 'tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

If `uv pip install` fails with a git SSH timeout (or `port 7878`), clone over HTTPS and install from disk:

```bash
GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null \
  git clone --depth 1 --branch nightly https://github.com/thinking-machines-lab/tinker-cookbook.git /tmp/tinker-cookbook
uv pip install -e '/tmp/tinker-cookbook[math-rl]'
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
`train_nll` (not `train_mean_nll` / `test/nll`). A 2-step LoRA smoke is
successful if the job reaches `running`, `train_nll` is finite, and a
weights-only checkpoint is saved. `train_nll` should fall on a longer run;
there is no held-out NLL metric.

Two steps is not enough to overwrite the base chat model. After a longer
memorize run, the generate command the recipe prints should answer
`Snowflake AI Research` to `Who trained you?`.
