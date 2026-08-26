# Full-Parameter Fine-Tuning

The conversational SFT recipe trains all parameters whenever the job config has
no `peft_config` block. Its default config, `configs/qwen3_8b_full.json`, is one
of those, so the plain command is already a full fine-tune:

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json
```

For a MoE model, use the matching config, which sets `model_provider` and
`ep_size`:

```bash
python -m recipes.sft.conversational.train \
  config=/path/to/config.json \
  job_config=configs/qwen36_35b_a3b_full.json
```

Full fine-tuning generally requires more memory than LoRA at the same GPU count;
see [LoRA and QLoRA](lora-and-qlora.md) for the lighter path. Validated hardware
ranges and checkpoint evaluation are not yet documented.
