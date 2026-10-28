# Full-Parameter Fine-Tuning

The conversational SFT example trains all parameters whenever the job config has
no `peft_config` block. Its default job, the `Qwen/Qwen3-8B` standard-context
full file in the [model catalog](../../../model-catalog/README.md), is one of
those, so the plain command is already a full fine-tune:

```bash
python -m recipes.sft_quickstart.conversational.train \
  config=/path/to/config.json
```

For a MoE model, pick it from the catalog; its training file sets
`model_provider` and `ep_size`:

```bash
python -m recipes.sft_quickstart.conversational.train \
  config=/path/to/config.json \
  model=Qwen/Qwen3.6-35B-A3B \
  context=long
```

Full fine-tuning generally requires more memory than LoRA at the same GPU count;
see [LoRA and QLoRA](lora-and-qlora.md) for the lighter path. To serve or
evaluate the checkpoint, see [Serve a training checkpoint](../inference/serve-checkpoint.md)
and the [inference example](../../../recipes/inference/README.md#evaluate).
Validated hardware ranges are not yet documented.
