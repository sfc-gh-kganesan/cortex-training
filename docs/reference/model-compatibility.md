# Model Compatibility

The [model catalog](../../model-catalog/models.json) is the source of truth for
inference and training support. It records maximum context limits and points to
the reusable SFT, RL, and inference profiles under
[`model-catalog/profiles/`](../../model-catalog/profiles/).

When a recommendation is rendered, its `maxContextTokens` value is injected as
`max_seq_len`. Training recommendations default to a standard SP1 profile and
can declare a `longSequence` alternative that uses SP8 and the maximum
supported training context. Treat GPU counts and optimization values as
starting points that might need adjustment for a specific workload.

For GLM 5.2 on the recommended H200 configuration, use
`zai-org/GLM-5.2-FP8` for 1M-token inference. The unquantized
`zai-org/GLM-5.2` checkpoint is limited to the existing 32K configuration
because its weights and a 1M-token MLA cache do not fit on 16 H200 GPUs.
The 1M FP8 profile uses `gpu_memory_utilization: 0.9`. A live 80% attempt
failed during engine startup after 588 seconds, while 32K at 80% and 1M at 90%
started successfully. This is consistent with the profile's narrow calculated
memory margin at 80%, although the retained log did not include the engine-core
error needed to confirm the exact cause.

The DeepSeek V4 and GLM 5.2 FP8 1M recommendations were validated with
1,048,575 pre-tokenized input tokens plus one generated token, not only with
short prompts.

For `Qwen/Qwen3.6-35B-A3B`, LoRA training and checkpoint save/load require the
`prime_rl` model provider. Current PrimeRL LoRA support for this MoE model is
limited to `q_proj`, `k_proj`, `v_proj`, and `o_proj`. Routed expert LoRA is not
yet supported.

Validate catalog changes with:

```bash
python scripts/validate_model_catalog.py
python -m pytest tests/test_model_catalog.py -q
```
