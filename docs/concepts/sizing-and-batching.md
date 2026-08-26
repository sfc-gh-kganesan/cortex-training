# Sizing and Batching

Training capacity depends on model architecture, parameter count, precision,
sequence length, micro-batch size, optimizer state, activation memory, and
parallelism strategy.

For the current recipes, all of these live in the job-config JSON rather than on
the command line. The fields that interact:

| Field | Where |
|---|---|
| `n_gpus` | `training_config` |
| `max_seq_len` | `training_config` |
| `train_batch_size` | `training_config` (and mirrored in `ds_config`) |
| `train_micro_batch_size_per_gpu` | `training_config.ds_config` |
| `gradient_accumulation_steps` | `training_config.ds_config` |
| `ep_size` | `training_config`, MoE only |

When all three DeepSpeed batch fields are set explicitly, they must agree:

```text
train_batch_size == train_micro_batch_size_per_gpu * gradient_accumulation_steps * n_gpus
```

The recipes do not check this before submitting, so an inconsistent config is
only caught server-side. Omitting `gradient_accumulation_steps` lets DeepSpeed
derive it instead. For MoE training, `n_gpus` must also be divisible by
`ep_size`.

There is no validated sizing guide or GPU requirement matrix yet, so treat the
shipped configs as starting points and measure your own.
