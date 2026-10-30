# Inference Endpoint

Serve open weights or a training checkpoint, then generate from it or evaluate
it.

## Serve

The default is Qwen3-8B full weights at standard context on two H200 GPUs.
Check capacity first with `cortex-training capacity`.

```bash
# Original Hugging Face weights
python -m recipes.inference.serve config=/path/to/config.json

# Any catalog model with an inference file, e.g. GLM-5.3
python -m recipes.inference.serve config=/path/to/config.json model=zai-org/GLM-5.3 context=long

# A checkpoint from a training example
python -m recipes.inference.serve config=/path/to/config.json \
  model=MODEL context=CONTEXT hardware=HARDWARE lora=LORA \
  source_job_id=TRAINING_JOB_ID checkpoint_id=CHECKPOINT_ID
```

For a checkpoint, set `model=`, `hardware=` and `lora=` to match the training
run; `context=` can be any context the model has an inference file in. A LoRA
checkpoint also needs a sampler with the identical `peft_config`, which the
catalog LoRA file has only if you kept the catalog adapter. The SFT and RL
examples check this and print a matching generate or evaluate command after
they save; the same arguments work with `serve`. When no catalog file matches,
the printed command has `job_config=INFERENCE_JOB_CONFIG`: replace it with an
inference job file whose sampling sub-job carries that adapter.

`serve` waits until workers are up. It then prints the `job_id` and ready-made
generate and evaluate commands, and leaves the endpoint running. Stop it with
`cortex-training cancel JOB_ID`.

The job is built as described in
[How an example builds its job](../README.md#how-an-example-builds-its-job).
The command-line layer is `max_seq_len=` only, and `inference/overrides.json`
ships empty. `job_config=FILE` uses your own job file.

## Generate

Pass `job_id=` to use a running endpoint, with the same `model=`, `context=`,
`hardware=` and `lora=` (or `job_config=`) arguments `serve` printed, so the
prompts are rendered for that model (and `evaluate` checks prompt length
against its context). Without `job_id=`,
the example creates a one-shot endpoint and releases it when done.

```bash
python -m recipes.inference.generate config=/path/to/config.json \
  job_id=JOB_ID model=MODEL context=CONTEXT hardware=HARDWARE lora=LORA \
  prompt="Who trained you?"
```

Other options: `prompts_file=`, `out=` (a JSONL file), `max_tokens=` (512),
`temperature=` (0.6), and `enable_thinking=`.

## Evaluate

`task=` picks the benchmark:

| `task=` | Score |
|---|---|
| `math500` (default) | MATH-500 accuracy |
| `gsm8k` | Exact numeric match on the answer after `####`, or on the last number when there is no `####`; `gsm8k/format` reports how often `####` was used. Use `temperature=0` |
| `identity` | Percentage of completions that contain `Snowflake AI Research` (prompts from `recipes/sft_quickstart/conversational/data/identity_eval.jsonl`) |

```bash
# Against a running endpoint (the arguments serve printed)
python -m recipes.inference.evaluate config=/path/to/config.json \
  job_id=JOB_ID model=MODEL context=CONTEXT hardware=HARDWARE lora=LORA

# Against a checkpoint, one-shot
python -m recipes.inference.evaluate config=/path/to/config.json \
  model=MODEL context=CONTEXT hardware=HARDWARE lora=LORA \
  source_job_id=TRAINING_JOB_ID checkpoint_id=CHECKPOINT_ID \
  task=gsm8k temperature=0 max_tokens=1024
```

To score an existing `generate` output for identity (it needs a `completion`
field):

```bash
python -m recipes.sft_quickstart.conversational.identity completions_file=/tmp/identity_eval.jsonl
```

## Notebook

`inference_walkthrough.ipynb` runs the same create → generate → cancel loop with
the Python client. It builds its job with
`make_config(model="Qwen/Qwen3-8B", context="standard", inference=True)`, plus a
demo override for one GPU and a 2,048-token context. Start Jupyter inside the
repository.
