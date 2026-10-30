# LoRA and QLoRA

LoRA training is implemented by the
[conversational SFT example](../../../recipes/sft_quickstart/README.md), and the RL
example uses it by default. `lora=true` picks the model's LoRA file in the
[model catalog](../../../model-catalog/README.md): a training file whose
`training_config` carries a `peft_config` block trains LoRA, and one without it
trains all parameters. The SFT example defaults to full-parameter training:

```bash
python -m recipes.sft_quickstart.conversational.train \
  config=/path/to/config.json \
  lora=true
```

`lora=true` (or `make_config(lora=True)`) picks the model's LoRA files, which
set the adapter rank and learning rate. Some models are full-only; for those,
`lora=true` fails with a list of the files the model has. The
[model catalog](../../../model-catalog/README.md) shows which models and
contexts have LoRA files.

To change the adapter (`peft_config.r`, `lora_alpha`, or `target_modules` for
dense Qwen models), add it to the example's `overrides.json`, or write the job
to a file, edit it, and pass it back with `job_config=`; see
[Where to make a change](../../../recipes/README.md#where-to-make-a-change).
Keep the shipped attention-only target list for `Qwen/Qwen3.6-35B-A3B`: its
PrimeRL backend does not yet support LoRA on routed experts, and Hugging Face
MLP target names are no-ops. An inference job that serves a LoRA checkpoint
needs the same `peft_config`.

QLoRA is not implemented: there are no quantization settings in the examples or in
the job-config schema.
