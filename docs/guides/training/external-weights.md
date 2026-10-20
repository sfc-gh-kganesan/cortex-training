# Start a Job From Your Own Weights

Use an external-weight import (also called a BYO checkpoint) when you have a
complete Hugging Face model directory that was produced outside Cortex Training
and want to use its weights for a new training or sampling job.

This initializes model weights; it does not restore an optimizer, scheduler, or
training step. A training job starts with a fresh optimizer. External-weight
import must be enabled for your Snowflake account.

## Prepare the model directory

Save full model weights in safetensors format. For example:

```python
model.save_pretrained("/absolute/path/to/model", safe_serialization=True)
tokenizer.save_pretrained("/absolute/path/to/model")
```

A directory downloaded from the Hugging Face Hub with `snapshot_download` also
works as-is, provided it holds safetensors weights.

The files must be flat at the directory root and include:

- `config.json`.
- Either one `model.safetensors`, or safetensors shards plus
  `model.safetensors.index.json`.
- The matching tokenizer and any processor configuration files the model uses.

The declared `model_name` must identify a supported base model whose
architecture matches the uploaded configuration and tensors. See
[model compatibility](../../reference/model-compatibility.md).

LoRA adapters cannot be imported directly. Merge an adapter into its base model
first, then save and upload the resulting full model:

```python
merged = peft_model.merge_and_unload()
merged.save_pretrained("/absolute/path/to/model", safe_serialization=True)
tokenizer.save_pretrained("/absolute/path/to/model")
```

## Upload the weights to a stage

The source must be a readable, S3-backed Snowflake stage. The steps below create
a named internal stage and upload the directory with `PUT`.

### Before you start

- A working client connection, either a connection profile or a JSON config
  file, as configured in [Set Up the Client](../../getting-started/setup.md).
  Confirm it with `cortex-training --connection NAME capacity` (profile) or
  `cortex-training --config PATH capacity` (JSON config).
- A role with `USAGE` on the database and schema and `CREATE STAGE` on the
  schema. The role that creates the stage owns it and can upload to it.
- The role that submits the training job must be able to read the stage. If it
  is a different role, grant it access:
  `GRANT READ ON STAGE MY_DB.MY_SCHEMA.MODEL_IMPORT TO ROLE TRAINING_ROLE;`
- No warehouse is needed: `CREATE STAGE`, `PUT`, and `LIST` do not use compute.
- A new stage, or one created with `ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')`.
  `CREATE STAGE IF NOT EXISTS` leaves an existing stage as it is, including its
  encryption, so the upload succeeds but the job cannot read it, and an internal
  stage's encryption cannot be changed later. To check an existing stage, run
  `SHOW STAGES LIKE 'MODEL_IMPORT' IN SCHEMA MY_DB.MY_SCHEMA;` and confirm the
  `type` column reads `INTERNAL NO CSE`. If it reads `INTERNAL`, use a new stage
  name.

### Where to run the SQL

These are Snowflake SQL statements, not shell commands. `PUT` reads files from
your machine, so it must run from a Snowflake client on the machine that holds
the model directory; it is not available in a Snowsight worksheet.

The simplest option is the Snowflake Python connector, which is already
installed with the Cortex Training client. Save this as `upload_weights.py`,
adjust the names, and run `python upload_weights.py`. It reads the same
connection profile the client uses:

```python
import snowflake.connector

PROFILE = "training"                      # your connections.toml profile
STAGE = "MY_DB.MY_SCHEMA.MODEL_IMPORT"
MODEL_DIR = "/absolute/path/to/model"      # directory holding config.json
TARGET = f"@{STAGE}/qwen-sft-run7"

with snowflake.connector.connect(connection_name=PROFILE) as conn:
    cur = conn.cursor()
    # SNOWFLAKE_SSE lets the service read the stage; the default encryption
    # for a named internal stage does not.
    cur.execute(f"CREATE STAGE IF NOT EXISTS {STAGE} ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')")
    # PUT compresses by default, which would turn safetensors into .gz files.
    cur.execute(f"PUT file://{MODEL_DIR}/* {TARGET} AUTO_COMPRESS = FALSE OVERWRITE = TRUE")
    for row in cur.execute(f"LIST {TARGET}"):
        print(row[0], row[1])
```

If you configured the client with a JSON config file instead of a profile,
replace the `connect(...)` line with the following. The connector also needs
your Snowflake user name, which the JSON config does not hold; use the user
that owns the token.

```python
import json

cfg = json.load(open("/path/to/config.json"))
connection = snowflake.connector.connect(
    host=cfg["host"],
    account=cfg["host"],                  # the connector derives the account from the host
    user="YOUR_USER",
    authenticator="PROGRAMMATIC_ACCESS_TOKEN",
    token=cfg["pat"],
)
with connection as conn:
    ...                                   # same statements as above
```

`LIST` should show `config.json`, the safetensors files, and the tokenizer
files directly under the target path, at their original sizes, with no `.gz`
suffix.

If you already use [Snowflake CLI](https://docs.snowflake.com/en/developer-guide/snowflake-cli/index),
you can run the same statements from a file instead:
`snow sql -c training -f upload.sql`, where `upload.sql` contains:

```sql
CREATE STAGE IF NOT EXISTS MY_DB.MY_SCHEMA.MODEL_IMPORT
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

PUT file:///absolute/path/to/model/*
  @MY_DB.MY_SCHEMA.MODEL_IMPORT/qwen-sft-run7
  AUTO_COMPRESS = FALSE
  OVERWRITE = TRUE;

LIST @MY_DB.MY_SCHEMA.MODEL_IMPORT/qwen-sft-run7;
```

`PUT` uploads only to internal stages. For an external S3 stage, upload the
uncompressed files with your object-storage tooling, keeping the same flat
layout, and grant the submitting role `USAGE` on the stage.

Keep the objects unchanged while the job is initializing.

## Create the job

Set `source_checkpoint_info.external_stage_path` on the sub-job. Use a fully
qualified stage path including its leading `@`:

```json
{
  "sub_job_configs": [
    {
      "job_type": "training",
      "model_name": "Qwen/Qwen3-0.6B",
      "source_checkpoint_info": {
        "external_stage_path": "@MY_DB.MY_SCHEMA.MODEL_IMPORT/qwen-sft-run7"
      },
      "training_config": {
        "optimizer": {"name": "AdamW", "lr": 0.00001},
        "max_seq_len": 128,
        "train_batch_size": 1,
        "n_gpus": 1
      }
    }
  ]
}
```

The repository contains this request as
[`examples/api/external-weights-training.json`](../../../examples/api/external-weights-training.json).
After replacing the stage path and adapting the model configuration, submit it
like any other job:

```bash
cortex-training submit examples/api/external-weights-training.json --wait
```

The Python SDK uses the same request field:

```python
from cortex_training.client import CortexTrainingClient, SubJobConfig

client = CortexTrainingClient.from_connection_name("training")
# With a JSON config file instead of a profile:
# client = CortexTrainingClient.from_pat(
#     host="ACCOUNT.snowflakecomputing.com",
#     pat=PAT,
#     database="CORTEX_TRAINING_DB",
#     schema="PUBLIC",
# )

training = SubJobConfig.training_job(
    model_name="Qwen/Qwen3-0.6B",
    optimizer={"name": "AdamW", "lr": 1e-5},
    max_seq_len=128,
    train_batch_size=1,
    n_gpus=1,
    source_checkpoint_info={
        "external_stage_path": "@MY_DB.MY_SCHEMA.MODEL_IMPORT/qwen-sft-run7"
    },
)
job_id = client.create_job(sub_jobs=[training])
client.wait_for_job(job_id)
```

To initialize a sampling job instead, pass the same `source_checkpoint_info`
dictionary to `SubJobConfig.sampling_job(...)`. In a job with both a training
and a sampling sub-job, set it on each sub-job that should start from the
imported weights; afterwards, [weight sync](../../reference/cli.md#sync-training-weights)
and [weights-only checkpoints](../inference/serve-checkpoint.md) work as they do
for any other job.

## Constraints

- `external_stage_path` is mutually exclusive with `checkpoint_id` and
  `source_job_id`. Do not send `stage_info` or storage credentials; the service
  resolves the stage under the submitting role.
- Only S3-backed stages and safetensors weights are supported. Azure and GCS
  stages are not supported for external-weight imports.
- Pickled weight formats such as `.bin`, `.pt`, `.pth`, and `.ckpt`, executable
  Python files, and Hugging Face dynamic-code configuration are rejected.
- A LoRA adapter is not a complete model. Merge it into the base weights before
  uploading.
- External weights contain no optimizer state. To resume optimizer and training
  state, use a resumable checkpoint previously saved by Cortex Training.
- A stage that does not exist is rejected when the job is created. A path inside
  an existing stage is only read once GPUs are assigned, so a mistyped path
  makes the job fail during initialization rather than at submission. Run
  `LIST` on the exact path before submitting.
