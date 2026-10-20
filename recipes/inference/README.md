# Inference Endpoint

Create a standalone Cortex Training inference endpoint from open weights or a
weights-only training checkpoint. Generate or evaluate against that running job.

## Hardware

Sampling GPUs are `n_gpus` in the job-config JSON. Check
account capacity before creating an endpoint:

```bash
cortex-training capacity
```

## Create an Endpoint

`config=` is the Snowflake connection file only. The default job body is
`configs/qwen3_8b_full.json`.

From original Hugging Face weights:

```bash
python -m recipes.inference.serve \
  config=/path/to/config.json
```

From a weights-only checkpoint produced by a training recipe. Pass
`source_job_id` and `checkpoint_id` together, and the matching example JSON:

```bash
python -m recipes.inference.serve \
  config=/path/to/config.json \
  job_config=JOB_CONFIG \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID
```

The command waits until workers are up, prints `job_id`, and leaves the
endpoint running. Tear it down with:

```bash
cortex-training cancel JOB_ID
```

`inference_walkthrough.ipynb` shows the same create → generate → cancel loop
with the Python client.

## Job config JSON

The recipe loads one create-job body with a single sampling sub-job. Pass a
shipped example or a copy with `job_config=JOB_CONFIG`.

```json
{
  "sub_job_configs": [
    {
      "job_type": "sampling",
      "model_name": MODEL_NAME,
      "dtype": DTYPE,
      "seed": SEED,
      "inference_config": {
        "max_seq_len": MAX_SEQ_LEN,
        "n_gpus": NUM_SAMPLING_GPUS,
        "vllm_config": {
          "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
          "gpu_memory_utilization": GPU_MEMORY_UTILIZATION
        },

        // Optional LoRA. Omit for full-parameter.
        "peft_config": {
          "peft_type": "Lora",
          "r": LORA_RANK,
          "lora_alpha": LORA_RANK,
          "lora_dropout": 0.0,
          "bias": "none",
          "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
      }
    }
  ]
}
```

The target list above is for dense Qwen models. A Qwen3.6 LoRA checkpoint uses
the attention-only target list in `configs/qwen36_35b_a3b_lora.json`; the
sampling and training PEFT configurations must match.

```bash
# Qwen3-8B LoRA
python -m recipes.inference.serve \
  config=/path/to/config.json \
  job_config=configs/qwen3_8b_lora.json

# Qwen3.5-9B LoRA / full
python -m recipes.inference.serve \
  config=/path/to/config.json \
  job_config=configs/qwen35_9b_lora.json

python -m recipes.inference.serve \
  config=/path/to/config.json \
  job_config=configs/qwen35_9b_full.json

# Qwen3.8-27B LoRA / full
python -m recipes.inference.serve \
  config=/path/to/config.json \
  job_config=configs/qwen38_27b_lora.json

python -m recipes.inference.serve \
  config=/path/to/config.json \
  job_config=configs/qwen38_27b_full.json

# Qwen3.6-35B-A3B LoRA / full
python -m recipes.inference.serve \
  config=/path/to/config.json \
  job_config=configs/qwen36_35b_a3b_lora.json

python -m recipes.inference.serve \
  config=/path/to/config.json \
  job_config=configs/qwen36_35b_a3b_full.json
```

## Examples

Point these at a running `job_id` from `serve`, or omit `job_id` to create a
one-shot endpoint that exits (and releases GPUs) when the example finishes.

### Generate

```bash
python -m recipes.inference.generate \
  config=/path/to/config.json \
  job_id=JOB_ID \
  prompt="Who trained you?"
```

From original weights or a checkpoint, without a pre-created endpoint:

```bash
python -m recipes.inference.generate \
  config=/path/to/config.json \
  prompt="Who trained you?"

python -m recipes.inference.generate \
  config=/path/to/config.json \
  job_config=JOB_CONFIG \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  prompt="Who trained you?"
```

### Evaluate (MATH-500)

```bash
python -m recipes.inference.evaluate \
  config=/path/to/config.json \
  job_id=JOB_ID

python -m recipes.inference.evaluate \
  config=/path/to/config.json

python -m recipes.inference.evaluate \
  config=/path/to/config.json \
  job_config=JOB_CONFIG \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID
```

### Evaluate (GSM8K test)

Exact-match on the `####` final answer. Use `temperature=0` for a deterministic compare.

```bash
python -m recipes.inference.evaluate \
  config=/path/to/config.json \
  job_config=configs/qwen3_8b_full.json \
  task=gsm8k \
  temperature=0 \
  max_tokens=1024

python -m recipes.inference.evaluate \
  config=/path/to/config.json \
  job_config=configs/qwen3_8b_lora.json \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  task=gsm8k \
  temperature=0 \
  max_tokens=1024
```

### Evaluate (identity)

Score is the percent of completions that contain `Snowflake AI Research`.
Default prompts are `recipes/sft/conversational/data/identity_eval.jsonl`.

```bash
python -m recipes.inference.evaluate \
  config=/path/to/config.json \
  job_config=configs/qwen3_8b_lora.json \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  task=identity \
  temperature=0 \
  max_tokens=128
```

Score an existing `generate` JSONL (needs a `completion` field):

```bash
python -m recipes.inference.identity \
  completions_file=/tmp/identity_eval.jsonl
```
