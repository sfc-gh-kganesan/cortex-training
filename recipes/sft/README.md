# Conversational Supervised Fine-Tuning

Fine-tune a chat model on a `messages` column. The default dataset is a
one-example memorize task: when prompted `Who trained you?`, answer
`Snowflake AI Research`. Hugging Face chat datasets work as well. `openai/gsm8k`
is mapped from `question`/`answer` into that same chat format. The entry point
supports LoRA and full-parameter training, logs `train_nll`, and saves a
weights-only checkpoint. Conversational dataset helpers live in
`tasks/conversational.py`; the task entrypoint calls the shared SFT engine in
`train.py`.

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

`config=` is the Snowflake connection file only. Copy
`examples/config/connection.json.template` and adjust it.

Defaults are Qwen3-8B full-parameter, thinking off, and 100 steps. It uses the
builtin `who_trained_you` dataset as default.
The default run will be finished within few minutes.

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

# Grade-school math (openai/gsm8k). The recipe maps question/answer → messages.
# Train split has 7473 rows. With the 8B configs (batch 8) one epoch is 934 steps.
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=configs/qwen3_8b_lora.json \
  dataset=openai/gsm8k \
  dataset_split=train \
  max_steps=934

# Builtin identity JSONL: paraphrases of Who trained you? → Snowflake AI Research
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  dataset=identity
```

Dataset loading lives in `tasks/conversational.py`, not `train.py`. To add a source:

- JSONL with `messages`: put it under `conversational/data/` and append a `BuiltinJsonl` to `CHAT_DATASETS` (see `who_trained_you` and `identity`).
- Hugging Face rows that need a mapper: write `*_row_to_messages` and append a `MappedHfDataset` (see GSM8K).
- Hugging Face sets that already have `messages` (No Robots, UltraChat): pass `dataset=org/name`. No registry entry.

Then list the dataset in `recipe.yaml` if it should show up in the catalog.

LoRA, GPU count, batch shape, sequence length, and MoE live in the job-config
JSON. Set `wandb_project` to log to Weights & Biases after
`uv pip install wandb` and `export WANDB_API_KEY` / `export WANDB_BASE_URL`.
Set `sf_tracking=True` to log to Snowflake experiment tracking after
`uv pip install "snowflake-ml-python>=1.19.0"`.

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
        "sp_size": SEQUENCE_PARALLEL_SIZE,
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

        // Optional expert parallelism for MoE. With model_provider prime_rl,
        // keep fused cross-entropy off (this block, or model_options with
        // EP_SIZE > 1): the chunked LM head set below cannot be combined with
        // it.
        "ac_config": {"mode": "full", "freq": 1},
        "prime_rl": {
          "fused_lm_head_token_chunk_size": 8192,
          "fused_cross_entropy": false
        },
        "ep_size": EP_SIZE
      }
    }
  ]
}
```

The target list above is for dense Qwen models. For
`Qwen/Qwen3.6-35B-A3B`, use the shipped LoRA config with
`model_provider: "prime_rl"` and attention-only targets (`q_proj`, `k_proj`,
`v_proj`, and `o_proj`). PrimeRL does not yet support LoRA on routed experts;
`gate_proj`, `up_proj`, and `down_proj` do not select those parameters.
The shipped Qwen3.6 configs use sequence parallel size 8 and logical batch 1
to run at the model's 262K context limit.

The `prime_rl` entries in the template above are deprecated but keep working.
On server versions that support `model_options`, with `EP_SIZE` > 1, the same
settings can be written inside `training_config` in place of the `prime_rl`
block, as the entry below; see
[the REST API reference](../../docs/reference/rest-api.md#model_options). The
shipped Qwen3.6 SFT configs still use the block, which older servers need.

```json
"model_options": {"lm_head": {"token_chunk_size": 8192, "cross_entropy": false}}
```

On server versions that have `numerics.fp32_parity`, Qwen3.5-family models run
with it on by default: dense models (for example `Qwen/Qwen3.8-27B`) and
Qwen3.5-MoE models with `EP_SIZE` > 1 (the Qwen3.6-35B-A3B configs). It raises
GPU memory. Memory was not measured for rows near `max_seq_len` with the
shipped Qwen3.8-27B configs. If a job runs out of memory, raise `sp_size`
(at most 8 for Qwen3.8-27B) or turn the option off by adding
`"numerics": {"fp32_parity": false}` to the `model_options` entry of
`training_config`, inside the same object if the config already has one (two
`model_options` keys in one JSON object keep only the last). A server that
has the option but not `model_options` support ignores that entry; see
[the REST API reference](../../docs/reference/rest-api.md#model_options).

For dense long-context profiles, set an integer
`fused_lm_head_token_chunk_size` in `training_config`. The server then computes
per-token log probabilities in chunks without materializing the full
`[sequence, vocabulary]` logits tensor. The setting saves memory and does not
change the loss.

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=JOB_CONFIG \
  dataset=DATASET \
  max_steps=MAX_STEPS \
  enable_thinking=ENABLE_THINKING \
  wandb_project=WANDB_PROJECT \
  sf_tracking=SF_TRACKING
```

```bash
# Qwen3-8B LoRA
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=configs/qwen3_8b_lora.json

# Qwen3.5-9B LoRA / full
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=configs/qwen35_9b_lora.json

python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=configs/qwen35_9b_full.json

# Qwen3.8-27B LoRA / full
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=configs/qwen38_27b_lora.json

python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=configs/qwen38_27b_full.json

# Qwen3.6-35B-A3B LoRA / full
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=configs/qwen36_35b_a3b_lora.json

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

Each step logs `train_nll` (the server's `avg_loss`) and `loss_weight_sum`, the
sum of `loss_mask` over the batch. On the default memorize task, `train_nll`
should fall quickly and `loss_weight_sum` should stay constant from step to
step. After save, the recipe prints one generate command. When running that
command, Assistant text should be `Snowflake AI Research`.

```bash
python -m recipes.inference.generate \
  config=/path/to/config.json \
  job_config=recipes/inference/configs/JOB_CONFIG \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  temperature=0 \
  prompt="Who trained you?"
```

After GSM8K SFT, generate a word problem (`temperature=0`) and check that the
completion ends with a `#### <number>` answer. For a numeric before/after score
on contest math, run MATH-500 against the same checkpoint (harder than GSM8K):

```bash
python -m recipes.inference.evaluate \
  config=/path/to/config.json \
  job_config=configs/qwen3_8b_lora.json \
  source_job_id=TRAINING_JOB_ID \
  checkpoint_id=CHECKPOINT_ID \
  task=gsm8k \
  temperature=0 \
  max_tokens=1024
```

After identity SFT, score the percent of completions that contain
`Snowflake AI Research` (default prompts: `data/identity_eval.jsonl`):

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

## Notebooks

- `conversational/qwen3_8b_sft_training.ipynb`
- `conversational/qwen3_8b_sft_training_multiplex.ipynb`
