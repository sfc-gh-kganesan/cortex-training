# Conversational Supervised Fine-Tuning

Fine-tune a chat model on a `messages` column. The default dataset is a
one-example memorize task: when prompted `Who trained you?`, answer
`Snowflake AI Research`. Hugging Face chat datasets work as well. The entry point supports
LoRA and full-parameter training, logs `train_nll`, and saves a weights-only
checkpoint.

## Hardware

Training GPUs are `n_gpus` in the JSON you pass (`NUM_TRAINING_GPUS`). Actual
requirements depend on model size, sequence length, precision, and whether LoRA
or dense training is used. Check the account capacity before submitting:

```bash
cortex-training capacity
```

## Run

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json
```

`config=` is the Snowflake connection file only. Copy `examples/config/connection.json.template` and adjust it.

Defaults are Qwen3-8B full-parameter, thinking off, and 100 steps. It uses the builtin `who_trained_you` dataset as default.

## Common Variations

```bash
# Thinking-on Qwen3 (must also pass enable_thinking=true to sample)
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  enable_thinking=true

# Different chat dataset
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  dataset=HuggingFaceH4/no_robots

python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  dataset=HuggingFaceH4/ultrachat_200k dataset_split=train_sft
```

LoRA, GPU count, batch shape, sequence length, and MoE live in the job-config
JSON. Set `wandb_project` to log to Weights & Biases after
`uv pip install wandb` and `export WANDB_API_KEY` (set `WANDB_BASE_URL` only for a non-Cloud W&B host).

## Job config JSON

The recipe loads one create-job body with a single training sub-job. Pass a
shipped example or a copy with `job_config=JOB_CONFIG`.

```json
{
  "sub_job_configs": [
    {
      "job_type": "training",
      "model_name": MODEL_NAME,
      "dtype": DTYPE,
      "seed": SEED,
      "training_config": {
        "n_gpus": NUM_TRAINING_GPUS,
        "max_seq_len": MAX_SEQ_LEN,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_clipping": GRADIENT_CLIPPING,
        "model_provider": MODEL_PROVIDER, // Use prime_rl for MoE
        "attn_implementation": "flash_attention_3",
        "optimizer": {
          "name": "AdamW",
          "lr": LEARNING_RATE,
          "weight_decay": WEIGHT_DECAY,
          "betas": [ADAM_BETA1, ADAM_BETA2],
          "eps": ADAM_EPS
        },
        "ds_config": {
          "train_batch_size": TRAIN_BATCH_SIZE,
          "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
          "gradient_accumulation_steps": TRAIN_BATCH_SIZE / (MICRO_BATCH_SIZE * NUM_TRAINING_GPUS),
          "zero_optimization": {
            "stage": ZERO_STAGE
          },
          "bf16": {
            "enabled": true
          }
        },

        // Optional LoRA. Omit for full-parameter.
        "peft_config": {
          "peft_type": "Lora",
          "r": LORA_RANK,
          "lora_alpha": LORA_RANK,
          "lora_dropout": 0.0,
          "bias": "none",
          "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        },

        // Optional expert parallelism for MoE.
        "ac_config": {"mode": "full", "freq": 1},
        "ep_size": EP_SIZE
      }
    }
  ]
}
```

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=JOB_CONFIG \
  dataset=DATASET \
  max_steps=MAX_STEPS \
  enable_thinking=ENABLE_THINKING \
  wandb_project=WANDB_PROJECT
```

```bash
# Qwen3-8B LoRA
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=configs/qwen3_8b_lora.json

# Qwen3.6-35B-A3B LoRA
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=configs/qwen36_35b_a3b_lora.json

# Qwen3.6-35B-A3B full-parameter
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=configs/qwen36_35b_a3b_full.json
```

## Logs and Expected Results

Metrics and configuration are written under `log_path`. To send the same
metrics to Weights & Biases:

```bash
uv pip install wandb
export WANDB_API_KEY=...
# optional: only if you are not using W&B Cloud
# export WANDB_BASE_URL=https://your-wandb-host
```

Then pass `wandb_project=WANDB_PROJECT` on the train command.

On the default memorize task, `train_nll` should fall quickly. After save, the
recipe prints one generate command. When running that command, Assistant text
should be `Snowflake AI Research`.

```bash
python -m recipes.inference.generate \
  config=/path/to/config.json \
  job_config=recipes/inference/configs/JOB_CONFIG \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  temperature=0 \
  prompt="Who trained you?"
```

## Notebooks

- `qwen3_8b_sft_training.ipynb`
- `qwen3_8b_sft_training_multiplex.ipynb`
