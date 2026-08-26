# Training Configuration

The recipes take two separate things:

- `config=` — the Snowflake connection file (account host, PAT, database,
  schema). See [Set up the client](../../getting-started/setup.md).
- `job_config=` — a JSON create-job body holding the whole training
  configuration. Each recipe ships examples under its own `configs/`
  directory.

Everything that shapes the run lives in the job-config JSON:

- Model, precision, and provider (`model_name`, `dtype`, `model_provider`)
- GPU count and parallelism (`n_gpus`, `ep_size`)
- Sequence length and batch shape (`max_seq_len`, `train_batch_size`, `ds_config`)
- Optimizer and gradient clipping (`optimizer`, `gradient_clipping`)
- LoRA or full-parameter method (presence of `peft_config`)

The remaining `name=value` command-line overrides are recipe-loop settings only
-- dataset, step count, evaluation cadence, logging and Weights & Biases. Run a
recipe module with no arguments to see its full list, or read its `Config` class.

The job-config object is posted unchanged as the create-job body, so
[REST API section 8](../rest-api.md#8-create-job-schemas) is the authoritative
schema for its fields. The
[conversational SFT README](../../../recipes/sft/conversational/README.md#job-config-json)
documents the shape with every field named.
