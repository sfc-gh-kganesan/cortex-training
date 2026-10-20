# LoRA and QLoRA

LoRA training is implemented by the
[conversational SFT recipe](../../../recipes/sft/README.md).
It is selected by the job-config JSON you pass, not by a command-line flag: a
config whose `training_config` carries a `peft_config` block trains LoRA, and one
without it trains all parameters.

The recipe defaults to `configs/qwen3_8b_full.json` (full-parameter), so LoRA is
opt-in:

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=configs/qwen3_8b_lora.json
```

Shipped SFT configs:

| Job config | Model | Method |
|---|---|---|
| `configs/qwen3_8b_full.json` | `Qwen/Qwen3-8B` | full-parameter (default) |
| `configs/qwen3_8b_lora.json` | `Qwen/Qwen3-8B` | LoRA, rank 32 |
| `configs/qwen35_9b_full.json` | `Qwen/Qwen3.5-9B` | full-parameter, lr 5e-6 |
| `configs/qwen35_9b_lora.json` | `Qwen/Qwen3.5-9B` | LoRA, rank 32, lr 1e-4 |
| `configs/qwen38_27b_full.json` | `Qwen/Qwen3.8-27B` | full-parameter, lr 5e-6 |
| `configs/qwen38_27b_lora.json` | `Qwen/Qwen3.8-27B` | LoRA, rank 32, lr 1e-4 |
| `configs/qwen36_35b_a3b_full.json` | `Qwen/Qwen3.6-35B-A3B` | full-parameter, lr 5e-6 |
| `configs/qwen36_35b_a3b_lora.json` | `Qwen/Qwen3.6-35B-A3B` | LoRA, lr 1e-4 |

Copy one and edit `peft_config.r` and `lora_alpha` to change the adapter. For
dense Qwen models, you can also adjust `target_modules`. Keep the shipped
attention-only target list for `Qwen/Qwen3.6-35B-A3B`: its PrimeRL backend does
not yet support LoRA on routed experts, and Hugging Face MLP target names are
no-ops. The recipe README documents the whole
[job-config guidance](../../../recipes/sft/README.md).

QLoRA is not implemented: there are no quantization settings in the recipe or in
the job-config schema.
