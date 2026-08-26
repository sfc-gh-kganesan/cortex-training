# Math GRPO

Train a model with grouped policy optimization on Hendrycks MATH and evaluate
against MATH-500. The recipe creates colocated training and sampling sub-jobs,
generates rollouts, scores them, trains, and synchronizes weights. After the
final save it logs a `python -m recipes.inference.evaluate` command.

## Hardware

The default configuration requests four training GPUs and four sampling GPUs.
Reduce or increase these values together with batch settings, and verify
capacity before submission:

```bash
cortex-training capacity
```

## Run

```bash
# Qwen3-8B LoRA (default)
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json
  
# Qwen3-8B full-parameter
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json \
  job_config=configs/qwen3_8b_full.json

# Qwen3.6-35B-A3B LoRA
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json \
  job_config=configs/qwen36_35b_a3b_lora.json

# Qwen3.6-35B-A3B full-parameter
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json \
  job_config=configs/qwen36_35b_a3b_full.json
```

`config=` is the Snowflake connection file. Adapt from `examples/config/connection.json.template`.

The default job body is `configs/qwen3_8b_lora.json`.

## Customizability

```bash
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json \
  job_config=JOB_CONFIG \
  max_tokens=MAX_TOKENS \
  problems_per_batch=PROBLEMS_PER_BATCH \
  group_size=GROUP_SIZE \
  max_steps=MAX_STEPS \
  n_test=N_TEST \
  wandb_project=WANDB_PROJECT
```

LoRA, GPU counts, sequence length, and MoE live in the job-config JSON. Set
`wandb_project` to log to Weights & Biases after `uv pip install wandb` and
`export WANDB_API_KEY` / `export WANDB_BASE_URL`.

The recipe loads one create-job body with colocated sampling and training
sub-jobs. Pass a shipped example or a copy with `job_config=JOB_CONFIG`.

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

        // Optional LoRA. Omit for full-parameter. Keep in sync with training.
        "peft_config": {
          "peft_type": "Lora",
          "r": LORA_RANK,
          "lora_alpha": LORA_RANK,
          "lora_dropout": 0.0,
          "bias": "none",
          "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        },

        // Optional router replay for MoE training.
        "router_replay": {"enabled": true, "max_cache_bytes": 17179869184}
      }
    },
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
        "mb_spec": {
          "max_tokens_per_mb": MAX_TOKENS_PER_MB
        },
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
            "stage": ZERO_STAGE,
            "reduce_scatter": true
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

        // Optional expert parallelism for MoE. NUM_TRAINING_GPUS must be a multiple of EP_SIZE.
        "ac_config": {"mode": "full", "freq": 1},
        "prime_rl": {
          "fused_lm_head_token_chunk_size": 8192,
          "fused_cross_entropy": false
        },
        "ep_size": EP_SIZE,

        // Optional router replay for MoE training.
        "router_replay": {"enabled": true, "max_cache_bytes": 2147483648}
      }
    }
  ]
}
```

## Evaluation and Logs

Training logs reward, correctness, format, rollout counts, and loss. Held-out
MATH-500 generate eval runs every `eval_every` batches (`test/env/all/correct`;
`eval_every=0` skips it). To mirror local metrics to Weights & Biases:

```bash
uv pip install wandb
export WANDB_API_KEY=...
export WANDB_BASE_URL=...
```

Then pass `wandb_project=WANDB_PROJECT` on the train command.
After save, the recipe prints one eval command.
`recipes.inference.evaluate` uses the same few-shot prompt, grader, and
decoding settings.

```bash
python -m recipes.inference.evaluate \
  config=/path/to/config.json \
  job_config=JOB_CONFIG \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  temperature=1.0 \
  max_tokens=MAX_TOKENS
```
