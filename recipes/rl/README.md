# Math GRPO

Train a model with grouped policy optimization on Hendrycks MATH and evaluate
against MATH-500. The recipe creates colocated training and sampling sub-jobs,
generates rollouts, scores them, trains, and synchronizes weights. After the
final save it logs a `python -m recipes.inference.evaluate` command.
MATH-specific prompting, scoring, and evaluation live in `tasks/math.py`; the
task entrypoint calls the shared GRPO engine in `train.py`.

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
  job_config=configs/qwen36_35b_a3b_lora.json \
  max_tokens=2048

# Qwen3.6-35B-A3B full-parameter
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json \
  job_config=configs/qwen36_35b_a3b_full.json \
  max_tokens=2048

# Qwen3.8-27B full-parameter
python -m recipes.rl.math_grpo.train \
  config=/path/to/config.json \
  job_config=configs/qwen38_27b_full.json \
  weight_sync_bucket_size=3221225472 \
  max_tokens=2048
```

`config=` is the Snowflake connection file. Adapt from
`examples/config/connection.json.template`.

The default job body is `configs/qwen3_8b_lora.json`. The default run will be finished within 1h.

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
  weight_sync_bucket_size=WEIGHT_SYNC_BUCKET_SIZE_BYTES \
  wandb_project=WANDB_PROJECT \
  sf_tracking=SF_TRACKING
```

LoRA, GPU counts, sequence length, and MoE live in the job-config JSON. Set
`wandb_project` to log to Weights & Biases after `uv pip install wandb` and
`export WANDB_API_KEY` / `export WANDB_BASE_URL`. Set `sf_tracking=True` to log
to Snowflake experiment tracking after
`uv pip install "snowflake-ml-python>=1.19.0"`.

`weight_sync_bucket_size` optionally overrides the transfer-bucket size in
bytes. When omitted, the recipe leaves the field out and uses the server
default. Set it above the largest individual model tensor.

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
        "sp_size": SEQUENCE_PARALLEL_SIZE,
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

        // Optional router replay for MoE training. Add "mode": "best_effort" to
        // route rows the sampler no longer holds (e.g. after a sampler restart)
        // with the trainer's own gate instead of failing the step.
        "router_replay": {"enabled": true, "max_cache_bytes": 2147483648}
      }
    }
  ]
}
```

The `prime_rl` entries in the config above are deprecated but keep working. On
server versions that support `model_options`, and with `EP_SIZE` > 1, the
same settings can be written inside `training_config`, in place of the
`prime_rl` block, as the entry below; see
[the REST API reference](../../docs/reference/rest-api.md#model_options).

```json
"model_options": {"lm_head": {"token_chunk_size": 8192, "cross_entropy": false}}
```

The Qwen3.6-35B-A3B RL and SFT recipe configs, `examples/api/training.json`,
`examples/api/rl.json` and the `rl` and `standard-context` model-catalog
profiles still use the `prime_rl` block, which older servers need.

On server versions that have `numerics.fp32_parity`, Qwen3.5-family models run
with it on by default: dense models (for example `Qwen/Qwen3.8-27B`) and
Qwen3.5-MoE models with `EP_SIZE` > 1 (the Qwen3.6-35B-A3B configs). It raises
GPU memory. Memory was not measured for rows near `max_seq_len` with the
shipped Qwen3.8-27B configs. If a job runs out of memory, raise `sp_size`
(at most 8 for Qwen3.8-27B) or turn the option off by adding
`"numerics": {"fp32_parity": false}` to the `model_options` entry of
`training_config`, inside the same object if the config already has one (two
`model_options` keys in one JSON object keep only the last). Turning it off
can bring back part of the difference between the trainer's and the sampler's
log-probabilities, which matters for RL. A server that has the option but not
`model_options` support ignores that entry; see
[the REST API reference](../../docs/reference/rest-api.md#model_options).

The target list above is for dense Qwen models. For
`Qwen/Qwen3.6-35B-A3B`, use the shipped LoRA config with
`model_provider: "prime_rl"` and attention-only targets (`q_proj`, `k_proj`,
`v_proj`, and `o_proj`). PrimeRL does not yet support LoRA on routed experts;
`gate_proj`, `up_proj`, and `down_proj` do not select those parameters.

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

To log the same metrics to Snowflake experiment tracking:

```bash
uv pip install "snowflake-ml-python>=1.19.0"
```

Then pass `sf_tracking=True` on the train command. The server ties each run to
a Snowflake experiment; results are viewable in Snowsight under
**AI & ML > Experiments**. See the
[Snowflake ML Experiments docs](https://docs.snowflake.com/en/developer-guide/snowflake-ml/experiments).

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
