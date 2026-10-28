# Conversational Supervised Fine-Tuning

Fine-tune a chat model on a dataset with a `messages` column, with LoRA or
full-parameter training. The default dataset is a memorization task: when asked
`Who trained you?`, the model learns to answer `Snowflake AI Research`. The
example is in `conversational/`: `train.py` is the entry point, `task.py` loads
datasets, `identity.py` scores identity evaluations, and `data/` holds the
bundled data. It runs on the shared loop in `runner.py`.

[DARE-Bench agent SFT](dare/README.md) uses the same loop for multi-turn
tool-calling traces.

## Run

The default is Qwen3-8B full-parameter at standard context on H200, thinking
off, for 100 steps. Check capacity first with `cortex-training capacity`.

```bash
python -m recipes.sft_quickstart.conversational.train config=/path/to/config.json
```

Other models and modes:

```bash
# LoRA
python -m recipes.sft_quickstart.conversational.train config=/path/to/config.json lora=true

# Any catalog model and context, e.g. Qwen3.6-35B-A3B (MoE) at 262K
python -m recipes.sft_quickstart.conversational.train config=/path/to/config.json \
  model=Qwen/Qwen3.6-35B-A3B context=long lora=true

# Smaller batch and shorter sequences than the catalog file
python -m recipes.sft_quickstart.conversational.train config=/path/to/config.json \
  train_batch_size=2 max_seq_len=8192 lr=1e-5
```

## Options

| Argument | Default | Meaning |
|---|---|---|
| `model=`, `context=`, `hardware=`, `lora=` | `Qwen/Qwen3-8B`, `standard`, `h200`, `false` | Which catalog file to use |
| `lr=`, `train_batch_size=`, `max_seq_len=` | from the job | Override the job for this run |
| `dataset=`, `dataset_split=` | `who_trained_you`, `train` | See [Datasets](#datasets) |
| `max_steps=` | 100 | Training steps |
| `enable_thinking=` | `false` | Train with the thinking-mode chat template. The printed generate command repeats it |
| `job_config=` | none | Use this job file instead of the catalog and `overrides.json`; `lr=`, `train_batch_size=` and `max_seq_len=` still apply |
| `wandb_project=`, `sf_tracking=` | off | [Experiment tracking](../../docs/guides/operations/wandb.md) |

The job is built as described in
[How an example builds its job](../README.md#how-an-example-builds-its-job).
`sft/overrides.json` ships empty, and also applies to DARE. DARE then applies
[`dare/overrides.json`](dare/overrides.json), which sets 8 training GPUs. For model-specific
job fields (MoE LoRA targets, `model_options`, `numerics.fp32_parity`), see
[LoRA and QLoRA](../../docs/guides/training/lora-and-qlora.md) and
[REST API `model_options`](../../docs/reference/rest-api.md#model_options).

## Datasets

| `dataset=` | Source |
|---|---|
| `who_trained_you` (default) | Bundled. A single prompt to memorize |
| `identity` | Bundled. Paraphrases of `Who trained you?` |
| `openai/gsm8k` | `question`/`answer` mapped to `messages`. One epoch is 1,868 steps at the Qwen3-8B batch of 4 |
| Any Hugging Face dataset with `messages` | e.g. `HuggingFaceH4/no_robots`, or `HuggingFaceH4/ultrachat_200k` with `dataset_split=train_sft` |

To add a dataset, edit `conversational/task.py`:

- For a JSONL file with `messages`, put it in `conversational/data/` and add a
  `BuiltinJsonl` to `CHAT_DATASETS`.
- For Hugging Face rows that need mapping, write a `*_row_to_messages` function
  and add a `MappedHfDataset`, as GSM8K does.

Also list it under `datasets` in the example's metadata, `conversational/example.yaml`.

## Expected results

Each step logs `train_nll` and `loss_weight_sum`. On the default task,
`train_nll` should drop quickly and `loss_weight_sum` should stay constant.
Metrics are written under `log_path`. A task with a held-out set, such as DARE,
also logs `valid_nll` every `eval_every` steps.

After the final save, the example prints a generate command for the checkpoint,
with `model=`, `context=`, `hardware=` and `lora=` filled in for a catalog
inference file that serves it, or `job_config=INFERENCE_JOB_CONFIG` when none
does (see [Serve](../inference/README.md#serve)). On the default task, the
answer should be `Snowflake AI Research`:

```bash
python -m recipes.inference.generate config=/path/to/config.json \
  model=MODEL context=CONTEXT hardware=HARDWARE lora=LORA \
  source_job_id=TRAINING_JOB_ID checkpoint_id=CHECKPOINT_ID \
  temperature=0 prompt="Who trained you?"
```

To score a checkpoint, use the [inference example](../inference/README.md#evaluate)
with `task=identity` (after identity SFT) or `task=gsm8k` (after GSM8K SFT).

## Notebooks

`conversational/qwen3_8b_sft_training.ipynb` and
`conversational/qwen3_8b_sft_training_multiplex.ipynb` build their job with
`make_config(model="Qwen/Qwen3-8B", context="standard")`, plus a small override
for the demo (128-token sequences, batch 8). Start Jupyter inside the
repository. The notebooks read the connection file from
`~/cortex-training-config.json`, or from `CORTEX_TRAINING_CONFIG` if that's set.
