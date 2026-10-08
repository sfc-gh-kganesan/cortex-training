# Tinker cookbook math

GSM8K and MATH through the tinker-cookbook math recipe. The client is
`arctic_platform.tinker` in Arctic Platform. This directory only launches
that client. It does not implement the Tinker API.

## Prerequisites

- `arctic_platform` with the in-process Tinker client
- `tinker` and `tinker-cookbook[math-rl]`
- `ARCTIC_CORTEX_HOST`, `ARCTIC_CORTEX_DATABASE`, `ARCTIC_CORTEX_SCHEMA`, and
  `ARCTIC_CORTEX_PAT`

GPU counts are arguments, as on the Cortex client CLI.

## Run

```bash
python -m recipes.tinker.math_rl --training-gpus 1 --sampling-gpus 1 -- \
  env=gsm8k \
  model_name=Qwen/Qwen3.5-4B \
  renderer_name=qwen3_5_disable_thinking \
  lora_rank=32 \
  group_size=8 \
  groups_per_batch=16 \
  learning_rate=1e-4 \
  max_tokens=256
```

`env=math` selects MATH. `base_url` is ignored. The recipe process releases
the Cortex job when it exits.
