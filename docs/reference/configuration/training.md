# Training Configuration

The examples take two separate things:

- `connection=` or `config=` — the Snowflake connection. Omit both to use the
  same resolution as the CLI (see [Set up the client](../../getting-started/setup.md)).
  `config=` is a legacy JSON file with account host, PAT, database, and schema
  (see [JSON connection file](../../getting-started/setup.md#json-connection-file)).
- The job — a create-job body holding the whole training configuration. The
  example builds it from the [model catalog](../../../model-catalog/README.md),
  or takes your own file with `job_config=`; see
  [How an example builds its job](../../../recipes/README.md#how-an-example-builds-its-job).

The job holds the server-side configuration of the workers:

- Model, precision, and provider (`model_name`, `dtype`, `model_provider`)
- GPU count and parallelism (`n_gpus`, `ep_size`)
- Sequence length and batch shape (`max_seq_len`, `train_batch_size`, `ds_config`)
- Optimizer and gradient clipping (`optimizer`, `gradient_clipping`)
- LoRA or full-parameter method (presence of `peft_config`)

The `name=value` command-line arguments of the single-model examples fall into
four groups (distillation has student and teacher variants of the first two; see
its [options](../../../recipes/distillation/README.md#options)):

- Job selection: `model=`, `context=`, `hardware=` and `lora=` pick the catalog
  file, and `job_config=` replaces it with your own.
- Job-field overrides: `lr=`, `train_batch_size=` and `max_seq_len=`.
- Job lifecycle: `job_id=` attaches to a running job instead of creating one.
- Loop settings: dataset, step count, evaluation cadence, logging and
  experiment tracking.

Run an example module with `--help` to see its full list, or read its `Config`
class.

The job is posted as the create-job body with one exception: each
`peft_config` it carries is validated and rewritten into its canonical form
first, and an invalid one fails before anything is submitted. A `ds_config` with
ZeRO stage 3 also fails before submission; use stage 1 or 2. Everything else,
`optimizer` included, is posted unchanged, so
[REST API section 8](../rest-api.md#8-create-job-schemas) is the authoritative
schema for its fields. To see a complete job with every field, write one with
`python -m recipes.utils` (see
[Create a job config from the command line](../../../recipes/README.md#create-a-job-config-from-the-command-line)).
