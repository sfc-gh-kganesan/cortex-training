# Cortex Training CLI and Client Reference

The `cortex-training` command and `cortex_training` Python package provide the
supported command-line and SDK interfaces for Cortex Training.

`ct` is installed as an alias for `cortex-training` and supports the same
commands and flags. You can replace `cortex-training` with `ct` in any example
below, such as `ct login config.json` or `ct --job JOB_ID step`.

## Installation

Requires Python 3.10+. Installing the package gives you the `cortex-training`
CLI, the `cortex-training tui` job dashboard, and the `cortex_training` Python SDK.

This project uses [uv](https://docs.astral.sh/uv/); `pip` works in place of
`uv pip` throughout if you prefer.

Install straight from the repository:

```bash
uv pip install git+https://github.com/snowflakedb/cortex-training.git
```

Or from a local checkout:

```bash
git clone https://github.com/snowflakedb/cortex-training.git
cd cortex-training
uv pip install .            # add -e for an editable/dev install
```

Verify the install:

```bash
cortex-training --help
cortex-training tui --help
```

## Jobs And Training Loops

A job is a server-side lifecycle resource for remote model workers, not a
training script. Its sub-jobs define the model, GPU requirements, and training
or sampling configuration. Submitting a job starts those workers; once the
job is running, you send batches, optimizer steps, or generation requests to
its job ID.

Jobs separate worker setup and GPU allocation from the code driving the
experiment. You can reuse running workers across requests, inspect status and
logs from another CLI process, and cancel the job to release capacity. A
combined training and sampling job also lets an RL loop synchronize weights
between its sub-jobs.

The CLI and `CortexTrainingClient` operate on the same jobs; they are not
alternative execution models. Use the CLI for setup, inspection, and individual
operations. Use a Python loop or an [example](../../recipes/README.md) for dataset
iteration, batching, rewards, evaluation, and repeated calls. The client loop
creates a job or uses an existing job ID, submits operations, and polls their
results. `submit --wait` waits for workers to be running; it does not upload or
execute your Python loop.

See [jobs and sub-jobs](../concepts/jobs-and-subjobs.md) and the
[Python SDK reference](python-sdk.md#job-lifecycle) for details.

## Quick Reference

Global flags such as `--config`, `--job`, and `--compact` go **before** the
subcommand. `login` also accepts its own `--config` after the subcommand.
`--job-id` is an alias for `--job`:

```bash
cortex-training --config config.json list
cortex-training --job JOB_ID step
cortex-training checkpoints JOB_ID
```

The data-plane actions `fwd-bwd`, `step`, `load`, `generate`, and `weight-sync`
require the global `--job JOB_ID` option. Their `--help` usage lines show its
placement. Management commands such as `get` and `checkpoints` instead take a
positional `JOB_ID` after the subcommand.

### [Connection](#legacy-connection-config)

```bash
cortex-training --connection training list    # Use a Snowflake profile
cortex-training list                          # Use the configured default profile
cortex-training login config.json             # Remember config for future commands
cortex-training login --config config.json    # Equivalent login syntax
cortex-training --config config.json list     # Use config for one command
```

### [Submit Jobs](#submit-a-job)

```bash
cortex-training submit recipes/api_quickstart/training.json
cortex-training submit recipes/api_quickstart/sampling.json
cortex-training submit job.json --dry-run     # Validate without submitting
cortex-training submit job.json --wait        # Wait until running, not finished
cortex-training submit - < job.json           # Read JSON from stdin
```

### [Manage Jobs](#manage-existing-jobs)

```bash
cortex-training list
cortex-training list --status running
cortex-training list --status active,failed --user JDOE --since 2026-10-01 --limit 20
cortex-training list --all                    # Every schema you can see
cortex-training get JOB_ID
cortex-training wait JOB_ID                   # Wait until running, not finished
cortex-training cancel JOB_ID
cortex-training checkpoints JOB_ID
cortex-training models                        # Active configured models
cortex-training capacity                      # Every GPU type your account supports
cortex-training capacity --hardware H200
```

### [Training And Generation](#run-a-forward-backward-smoke-test)

```bash
cortex-training --job JOB_ID fwd-bwd recipes/api_quickstart/fwd-bwd.json
cortex-training --job JOB_ID step             # Default learning rate: 1e-4
cortex-training --job JOB_ID step --lr 2e-5
cortex-training --job JOB_ID generate recipes/api_quickstart/generate.json
cortex-training --job JOB_ID weight-sync
cortex-training --job JOB_ID weight-sync --weight-format lora
```

See also [generation payloads](#run-a-generate-smoke-test) and
[weight-sync routing](#sync-training-weights).

### [Load Checkpoints](#load-a-checkpoint-into-a-running-job)

```bash
cortex-training --job JOB_ID load CHECKPOINT_ID
cortex-training --job JOB_ID load CHECKPOINT_ID --source-job-id SOURCE_JOB_ID
cortex-training --job JOB_ID load CHECKPOINT_ID --target-sub-job-id JOB_ID:training:0
cortex-training --job JOB_ID load CHECKPOINT_ID --no-poll
```

### [Logs And Metrics](#job-tui)

```bash
cortex-training tui                          # Open job list (a: all schemas)
cortex-training tui JOB_ID                   # Open the job dashboard
cortex-training download-log JOB_ID --output-dir ./logs
cortex-training download-log JOB_ID --log-type stdout --output-dir ./logs
cortex-training download-metrics JOB_ID --output-dir ./metrics
```

See [execution logs](#download-execution-logs),
[persisted stdout](#download-persisted-stdout), and [GPU metrics](#download-metrics)
for download paths and output fields.

### [Model Deployments](#model-deployments)

```bash
cortex-training model-deployment create --spec recipes/api_quickstart/model-deployment-checkpoint.json --wait
cortex-training model-deployment create --spec deployment.json --dry-run   # Validate only
cortex-training model-deployment list --status READY
cortex-training model-deployment describe sf-ct-DEPLOYMENT_UUID
cortex-training model-deployment wait sf-ct-DEPLOYMENT_UUID --timeout 1800
cortex-training model-deployment download-log sf-ct-DEPLOYMENT_UUID --output-dir ./logs
cortex-training model-deployment download-metrics sf-ct-DEPLOYMENT_UUID --output-dir ./metrics
cortex-training model-deployment drop sf-ct-DEPLOYMENT_UUID
```

### [Output And Help](#json-output-and-help)

```bash
cortex-training --compact list
cortex-training get JOB_ID | jq '.sub_jobs'
cortex-training --help
cortex-training fwd-bwd --help
```

### Defaults And Waiting

| Command | Default behavior | Alternative |
|---------|------------------|-------------|
| `submit` | Return after submission | `--wait` waits until running; `--dry-run` validates without submitting |
| `wait JOB_ID` | Wait until running, not until training finishes | Use `get JOB_ID` to inspect current status |
| `fwd-bwd`, `generate` | Poll the submitted request until completion | Set top-level `"poll": false` in the input JSON |
| `step` | Poll until completion; learning rate `1e-4` | Set `--lr`; polling cannot be disabled |
| `load`, `weight-sync` | Poll the submitted request until completion | `--no-poll` returns without waiting for the result |
| `capacity` | Query each GPU type the account supports | Select one with `--hardware` |
| `download-log`, `download-metrics` | Write under the current directory | Set `--output-dir` |
| `model-deployment create` | Return after creation with status `PENDING` | `--wait` waits until `READY`; `--dry-run` validates without creating |
| `model-deployment wait` | Wait until `READY`, for up to `--poll-timeout` | Set `--timeout SECONDS` |

## Detailed Reference

- [Connection config](#legacy-connection-config), [login](#legacy-login), and [environment variables](#environment-variables)
- [Submit](#submit-a-job), [manage jobs](#manage-existing-jobs), [available models](#list-available-models), and [GPU capacity](#show-current-gpu-capacity)
- [Forward-backward and optimizer steps](#run-a-forward-backward-smoke-test)
- [Load checkpoints](#load-a-checkpoint-into-a-running-job) and [initialize sampling](#start-sampling-from-a-training-checkpoint)
- [Generate](#run-a-generate-smoke-test) and [sync weights](#sync-training-weights)
- [Download logs](#download-execution-logs), [stdout](#download-persisted-stdout), and [metrics](#download-metrics)
- [Job TUI](#job-tui), [JSON output and help](#json-output-and-help), and [troubleshooting](#troubleshooting)

`cortex-training` submits and manages Cortex Training jobs through the Cortex
Training REST endpoint.
The Snowflake-native workflow uses the same connection profiles as the Python
Connector and Snowflake CLI:

```toml
# ~/.snowflake/connections.toml
[training]
account = "ORG-ACCOUNT"
host = "ACCOUNT.snowflakecomputing.com"
user = "USER"
authenticator = "programmatic_access_token"
token = "YOUR_PROGRAMMATIC_ACCESS_TOKEN"
database = "CORTEX_TRAINING_DB"
schema = "PUBLIC"
```

Protect files containing credentials, then select the named profile:

```bash
chmod 600 ~/.snowflake/connections.toml
cortex-training --connection training list
```

`cortex-training list` with no connection arguments uses the
Connector-configured default profile. Set
`SNOWFLAKE_DEFAULT_CONNECTION_NAME=training`, configure
`default_connection_name = "training"` in Snowflake's `config.toml`, or name
the connection `[default]`.

Profile lookup is a fallback. An explicit or remembered legacy JSON config, or
a complete direct `--base-url` / `--host` + `--pat` connection, continues to
win. This preserves existing scripts.

### Legacy Connection Config

For Snowflake PAT auth, use `host` for the account hostname. Do not use
`base_url` for Snowflake PAT auth.

```json
{
  "host": "ACCOUNT.snowflakecomputing.com",
  "pat": "YOUR_PROGRAMMATIC_ACCESS_TOKEN",
  "database": "CORTEX_TRAINING_DB",
  "schema": "PUBLIC",
  "endpoint": "cortex-training",
  "poll_interval": 0.5,
  "poll_timeout": 1800.0,
  "verify_ssl": true
}
```

If you prefer not to store the PAT in the file, omit `pat` and set it in the
shell instead:

```bash
export CORTEX_TRAINING_PAT='YOUR_PROGRAMMATIC_ACCESS_TOKEN'
```

To target a local or otherwise compatible server instead of a Snowflake
account, use `base_url` with an explicit scheme. This skips PAT auth:

```json
{
  "base_url": "http://localhost:8084",
  "database": "MY_DB",
  "schema": "PUBLIC",
  "endpoint": "cortex-training"
}
```

### Legacy Login

Login validates the config and stores only the config path, not the config
contents:

```bash
cortex-training login config.json
cortex-training login --config config.json
```

Provide exactly one config path, either positionally or with `--config` after
`login`. Both forms validate and remember the same file. Login requires an
explicit path even when a global `--config`, `CORTEX_TRAINING_CONFIG`, or a
previous login is available.

The login state is written to `~/.config/cortex-training/login.json` by default,
or `$XDG_CONFIG_HOME/cortex-training/login.json` when `XDG_CONFIG_HOME` is set.

You can bypass login for one command with:

```bash
cortex-training --config config.json list
```

or by setting:

```bash
export CORTEX_TRAINING_CONFIG=/path/to/config.json
```

Explicit CLI flags override config values.

### Manage Existing Jobs

```bash
cortex-training list
cortex-training list --status running
cortex-training list --status active,failed --user JDOE --since 2026-10-01 --limit 20
cortex-training list --all
cortex-training get JOB_ID
cortex-training checkpoints JOB_ID
cortex-training cancel JOB_ID
cortex-training wait JOB_ID
```

`list` takes these filters:

| Flag | Takes | Notes |
|---|---|---|
| `--all` | nothing | lists jobs across every schema in the account that you can see, not just the configured one |
| `--status` | comma list of `pending`, `placing`, `initializing`, `running`, `failed`, `cancelled`, `terminated`, `active`, `done` | `active` is `pending` through `running`; `done` is `cancelled`, `terminated`, and the `done` that older jobs show; `pending` does not include `placing`. Matched against a row's status in any form the server sends, including `JOB_STATE_RUNNING` and aliases such as `canceled`, `completed`, `succeeded`, `error`, `queued`, `creating` |
| `--user` | user name | SQL name rules: unquoted means its uppercase form, `"quoted"` means exactly that name |
| `--since` | `YYYY-MM-DD` | a UTC day; keeps jobs created on or after it |
| `--until` | `YYYY-MM-DD` | a UTC day; keeps jobs created on or before it, so `--since D --until D` means "on day D" |
| `--limit` | number | prints at most this many of the newest matching jobs by `created_at`; a job listed without one counts as oldest |

Without `--all`, the CLI fetches the configured schema's jobs and applies the
filters itself, leaving out a job with no `created_at` when `--since` or
`--until` is set. With `--all`, the server applies them across the account
([REST API reference section 5.3](rest-api.md#across-the-account---get-apiv2endpoint));
where account-level listing is not enabled for the account, the command fails
with `account-level job listing is not enabled for this account`. A bad
`--status`, `--since`, `--until`, or `--limit`, or an empty `--user`, exits
with status 2 before connecting.

`get` fetches current job details; `checkpoints` lists saved checkpoints.
`wait` waits for the job to reach **running**, not for training to finish.
See [Manage Jobs](../guides/operations/manage-jobs.md) for the operational workflow.

### List Available Models

List active models configured for new jobs:

```bash
cortex-training models
```

The command prints `{"models": [{"name": "..."}]}`. It calls the account-level
route and does not use the configured database or schema. This is a policy
catalog, not a cache-readiness guarantee; a submitted job may wait while model
sync completes.

### Show Current GPU Capacity

Print the caller account's reserved GPU capacity and current usage, separated
by hardware:

```bash
cortex-training capacity
cortex-training capacity --hardware H200
```

The default command queries `H200`, `B200`, and `B300` independently and
prints a `capacity_by_hardware` map with one entry per GPU type your account
supports. Each entry includes `has_reservation`, `max_total_gpus`,
`reserved_gpus`, `in_use_gpus`, `pending_gpus`, and `available_gpus`. Types the
service rejects as unsupported for the account (error code `517604`) are listed
in `unsupported_hardware` instead, which is empty when every type is supported:

```json
{
  "capacity_by_hardware": {
    "B300": {"available_gpus": 8, "has_reservation": true, "max_total_gpus": 8, "...": "..."},
    "H200": {"available_gpus": 16, "has_reservation": true, "max_total_gpus": 16, "...": "..."}
  },
  "unsupported_hardware": ["B200"]
}
```

Any other error still fails the command, and so does an account that supports
none of the types.

Each hardware lookup is attempted once with at most a 10-second connect timeout
and a 30-second read timeout; a shorter SDK timeout still wins. A silent
network, proxy, or service hop therefore fails instead of leaving the command
blocked indefinitely. The three lookups run serially; use `--hardware` when
only one type is needed.

`--hardware` keeps the single-capacity response shape for one GPU type. Asking
for a type your account does not support returns the service's error, which
names the supported types.

`max_total_gpus` is the canonical ceiling and supersedes the deprecated
`reserved_gpus`. `in_use_gpus` counts only GPUs the account holds; queued work
is reported separately in `pending_gpus`. See
[REST API reference section 5.5](rest-api.md#55-capacity---get-capacity) and
[GPU hardware](../concepts/hardware.md).

#### Capacity Breakdown

`--breakdown` splits `in_use_gpus` and `pending_gpus` by sub-job type
(`job-type`) or by submitting user (`submitted-by`). Repeat it for both:

```bash
cortex-training capacity --hardware H200 --breakdown job-type --breakdown submitted-by
```

Each capacity object then carries a `breakdown`, as
`get_capacity_breakdown()` returns it:

```bash
cortex-training capacity --hardware H200 --breakdown submitted-by \
  | jq -c '.breakdown.by_submitted_by.groups[]'
```

```json
{"kind":"USER","submitted_by":"YOUR_USER","is_caller":true,"in_use_gpus":24,"pending_gpus":0,"job_count":2}
{"kind":"USER","submitted_by":"SOME_USER","is_caller":false,"in_use_gpus":8,"pending_gpus":16,"job_count":2}
{"kind":"OTHER","submitted_by":null,"is_caller":false,"in_use_gpus":8,"pending_gpus":0,"job_count":1}
{"kind":"UNATTRIBUTED","submitted_by":null,"is_caller":false,"in_use_gpus":4,"pending_gpus":0,"job_count":1}
```

Until the breakdown is enabled for your account, each dimension reports
`DISABLED`. See [REST API reference section 5.4](rest-api.md#breakdown---breakdown)
for every field and status.

### Submit A Job

The submit command expects a CreateJob JSON body:

```json
{
  "hardware": "H200",
  "sub_job_configs": [
    {
      "job_type": "sampling",
      "model_name": "gpt2",
      "inference_config": {
        "max_seq_len": 128,
        "n_gpus": 1
      }
    }
  ]
}
```

Optional `hardware` is `H200`, `B200`, or `B300`. Omit it to use H200. Every
sub-job in the job uses that type. See [GPU hardware](../concepts/hardware.md).

Submit it:

```bash
cortex-training submit job.json
cortex-training submit job.json --wait
cortex-training submit job.json --dry-run
```

Without `--wait`, submission returns without waiting for the job to run.
`--wait` waits until **running**, not until training finishes. `--dry-run`
validates and prints the request body without sending it, applying the same
checks as a live submit — so the printed body already shows any `peft_config`
in its canonical form. A `log_probability` or `log_prob` sub-job is rejected
before send, including on `--dry-run`, with
`log_probability sub-jobs are not currently supported`.

The repo includes a Prime-RL/Qwen3.6 training example:

```bash
cortex-training submit recipes/api_quickstart/training.json
cortex-training submit recipes/api_quickstart/sampling.json
```

That file creates a training sub-job for `Qwen/Qwen3.6-35B-A3B` with
`training_config.model_provider` set to `arctic`. The server does not accept
`arctic` yet, so `submit` (including `--dry-run`) sends it as `prime_rl`.

### Run A Forward-Backward Smoke Test

After the training job is running, send one tokenized training batch:

```bash
cortex-training --job JOB_ID fwd-bwd recipes/api_quickstart/fwd-bwd.json
cortex-training --job-id JOB_ID step
```

The fwd-bwd JSON is human-readable: it contains text samples, tokenizer
settings, batch size, sequence length, `position_ids`, and label generation
settings. The CLI tokenizes the text, builds tensor kwargs, serializes
`{"args": (), "kwargs": ...}` as a DSSST1 safetensors frame (see
[REST API reference section 9](rest-api.md#9-dssst1-binary-wire-protocol)),
submits `forward_backward`, and polls the request by default. Set `"poll": false` in
the JSON to print only the submitted `request_id`.

Text payloads require `transformers` in the client environment. You can also
provide pre-tokenized tensor data directly under `payload.kwargs` for fully
offline use.

Run an optimizer step after fwd-bwd with:

```bash
cortex-training --job-id JOB_ID step
cortex-training --job-id JOB_ID step --lr 2e-5
```

When omitted, `--lr` defaults to `1e-4`.

### Load A Checkpoint Into A Running Job

After a job has already been created and reached running, load a checkpoint into
that existing job with:

```bash
cortex-training --job-id JOB_ID load CHECKPOINT_ID
```

To load from another job's checkpoint store:

```bash
cortex-training --job-id JOB_ID load CHECKPOINT_ID --source-job-id SOURCE_JOB_ID
```

To load into a specific training sub-job (useful for multi-sub-job sessions):

```bash
cortex-training --job-id JOB_ID load CHECKPOINT_ID --target-sub-job-id JOB_ID:training:0
```

When `--target-sub-job-id` is omitted, the service routes the load to the
session's training sub-job. Sampling sub-jobs are not valid targets.

#### Discovering Sub-Job IDs

To find the training sub-job and its GPU count:

```bash
cortex-training get JOB_ID | jq '.sub_jobs[] | select(.job_type=="training") | {sub_job_id, n_gpus: .training_config.n_gpus}'
```

`get` takes the job id as a positional argument, so `--job-id` is not used here.
The global `--job-id` option is only for the data-plane subcommands that have no
positional job id (`fwd-bwd`, `step`, `load`, `generate`, `weight-sync`).
For Python sub-job discovery, see the
[runtime load API reference](rest-api.md#64-runtime-load---post-job_idload).

#### When to Use load --target-sub-job-id

A job has at most one training sub-job, so `load --target-sub-job-id` can be
omitted. Use it when you want explicit control over which sub-job receives the
checkpoint rather than relying on the server's default resolution; it must name a
training sub-job. `weight-sync` takes its own `--target-sub-job-id`, which names
sampling sub-jobs and can be repeated — see
[Sync Training Weights](#sync-training-weights).

#### DP Size Compatibility

When changing `n_gpus` from the checkpoint's source job, create the target
training sub-job with `"load_optimizer_states": false` in its `training_config`.
This cannot be changed at load time. See
[DP size compatibility](rest-api.md#dp-size-compatibility) for the constraint.

This is the runtime load path. Create-time initialization from a checkpoint
saved by Cortex Training still uses
[`source_checkpoint_info`](rest-api.md#65-create-time-checkpoint-initialization)
in the submitted sub-job JSON. To start a job from weights you trained
elsewhere, see [Start a Job From Your Own Weights](../guides/training/external-weights.md).

`load` polls until the request completes by default. Pass `--no-poll` to return
the request metadata without waiting for the result.

### Start Sampling From A Training Checkpoint

Sampling requires a `weights-only` checkpoint and a new sampling job with
`source_checkpoint_info` in its submitted JSON; `load` targets existing training
jobs, not sampling jobs. Resumable checkpoints are not directly loadable by the
sampling runtime.

See [Serve a Training Checkpoint](../guides/inference/serve-checkpoint.md)
for the example workflow, or
[Start sampling from saved weights](rest-api.md#134-start-sampling-from-saved-weights)
for the Python save-and-create example.

### Run A Generate Smoke Test

After a sampling job is running, send readable prompts with sampling
parameters:

```bash
cortex-training --job-id JOB_ID generate recipes/api_quickstart/generate.json
```

The generate JSON contains `prompts`, optional `sampling_params`, and optional
`routing_key` / `strict` fields. `sampling_params` may be one object applied to
all prompts or a list of objects/nulls aligned with `prompts`. A flat list of
integers such as `"prompts": [1, 2, 3]` is one pre-tokenized prompt, not three;
use the nested form `[[1, 2], [3, 4]]` for a batch. The CLI submits `generate`
and polls the request by default. Set `"poll": false` to print only the
submitted `request_id`.

### Sync Training Weights

For an RL-style job with one training and one sampling sub-job, sync training
weights into sampling with:

```bash
cortex-training --job-id JOB_ID weight-sync
```

By default this syncs from `JOB_ID:training:0` to `JOB_ID:sampling:0`, routes
the operation through `JOB_ID:training:0`, and polls for completion. Override
sub-job ids when needed:

```bash
cortex-training --job-id JOB_ID weight-sync \
  --source-sub-job-id JOB_ID:training:0 \
  --target-sub-job-id JOB_ID:sampling:0 \
  --target-sub-job-id JOB_ID:sampling:1
```

If a backend needs a different operation routing hint, pass
`--operation-sub-job-id` or `--operation-sub-job-type`.

Use `--weight-format lora` for adapter-only sync; `vllm` and `hf` are also
accepted formats. Pass `--no-poll` to return the request metadata without waiting
for synchronization to complete.

### Download Execution Logs

Pull every log file the job's experiment run produced. Each sub-job's
`_logs/` artifact directory may contain multiple files (e.g.
`execution.jsonl`, `server.log`); all of them are downloaded:

```bash
cortex-training download-log JOB_ID --output-dir /path/to/dir
cortex-training download-log JOB_ID --output-dir /path/to/dir --resume
```

Files are written as `<output_dir>/<sub_job_id>/<filename>` so siblings
do not collide, readable only by you. When `--output-dir` is omitted, the
current working directory is used instead. Each file goes straight to disk, so
a large log does not need as much memory, and replaces the earlier copy only
once it is complete. Progress is one line per file on stderr. The CLI also
prints a JSON summary listing each `saved_path` and its `bytes`.

A running job rewrites its execution log as a whole, so a file cannot be
fetched in part. `--resume` skips a file whose size and checksum are unchanged
since the last download into that directory (`unchanged: true` in the summary)
and downloads a changed file in full. `--sub-job SUB_JOB_ID` (repeatable)
limits the download, and `--transfer-timeout SECONDS` bounds each file's
fetch, raised for a large file to allow 512 KiB/s.

Programmatic access is `CortexTrainingClient.download_execution_logs(job_id,
output_dir)`. `fetch_execution_logs(job_id)` returns
`{sub_job_id, filename, artifact_uri, content}` dicts with every body in memory.

### Download Persisted Stdout

Reconstruct each sub-job's persisted stdout/stderr chunks into
`<output_dir>/<sub_job_id>/stdout.log`:

```bash
cortex-training download-log JOB_ID --log-type stdout --output-dir /path/to/logs
cortex-training download-log JOB_ID --log-type stdout --output-dir /path/to/logs --resume
```

The current working directory is used when `--output-dir` is omitted.
`--resume` continues a stdout download already in that directory: it appends
chunks written since the last download, and it continues a download that was
interrupted or failed part way through. Progress is saved while every download
runs, with or without `--resume`, so after an interruption rerun the same
command with `--resume` to fetch only the remaining chunks. A run without
`--resume` starts over.

Chunks are fetched concurrently; `--parallel N` sets how many at once (1 to
128, default 16) and output is still written in chunk order. `--sub-job SUB_JOB_ID`
(repeatable) limits the download to those sub-jobs; with a single `--sub-job`
whose id has only letters, digits, `_`, `.`, `:` and `-`, only that sub-job's
chunks are listed. Naming a sub-job with no stdout chunks is an error (after
which a single one has every sub-job's listed, to name those that have some).
Transient connection and service errors are retried
with backoff, and an expired session is reopened. A single fetch or listing
that takes longer than `--transfer-timeout SECONDS` (default 300; `0` means no
limit) is abandoned and retried. Errors a retry cannot fix,
such as missing privileges, stop the download at once with a short `error:`
line. On a terminal, progress is one line on stderr that updates in place, and
`warning:` lines print above it. A redirected stderr gets one final line per
sub-job. Progress is not part of the JSON on stdout. The download uses one
Snowflake session for all of its concurrent fetches. Ctrl-C stops the download, keeps the chunks already written, and exits with
status 130. After an error or Ctrl-C the CLI says whether anything was saved for
`--resume` to continue from.

A chunk that was removed from the run after it was listed is skipped. A chunk
that cannot be decompressed is downloaded once more and, if it is still
damaged, skipped: the rest of the file is written, the chunk is listed in that
sub-job's `skipped_chunks`, and a `warning:` line goes to stderr. `--resume`
keeps a skipped chunk skipped; a run without `--resume` tries it again.
`--strict` stops on the first damaged chunk instead. Any notice about a
sub-job is also listed in its `warnings` field in the JSON summary.

While the job and that sub-job are running, a sub-job whose newest saved chunk
is more than `--stale-after-minutes` old (default 10) gets a warning that
uploads may be stalled. Chunk times come from the chunk names and are UTC, and
are compared with the service's clock when it reports one. A job that prints
nothing for a while also produces this warning for stdout. For two minutes
after a job ends, each sub-job gets a warning that its last chunks may still be
uploading; run the command again with `--resume` shortly afterwards to pick
them up. These checks read the job's status once, alongside the download;
`--stale-after-minutes 0` turns them all off. The download itself still
completes.

`--parallel`, `--strict`, and `--stale-after-minutes` do not apply to
execution logs, including `download-log` without `--log-type stdout`. Two downloads of the same
directory at once are unsupported.

### Download Metrics

Reconstruct each sub-job's metric chunks. GPU and host rows go to
`<output_dir>/<sub_job_id>/gpu.jsonl`, and a training sub-job's
[training metrics](rest-api.md#125-training-metrics-download) — the loss,
gradient norm, learning rate and whatever else its loss function returns — to
`<output_dir>/<sub_job_id>/training.jsonl`:

```bash
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics --resume
```

The two are separate streams, fetched one after the other into the same
directory, each with its own file and resume manifest. The command prints the
saved path, chunk count, and first/last logical artifact URIs for each
reconstructed file, under `metrics` for the GPU stream and `training_metrics`
for the training one:

```json
{"job_id": "JOB_ID", "metrics": [...], "training_metrics": [...]}
```

Each stream reports its own progress line on stderr (`metrics: …`,
`training metrics: …`). `--resume`, `--parallel N`, `--sub-job SUB_JOB_ID`,
`--strict`, `--stale-after-minutes`, `--transfer-timeout`, damaged chunks,
retries, and Ctrl-C behave as for
[persisted stdout](#download-persisted-stdout) and apply to both streams.

While the job is running, a sub-job that has saved stdout but no GPU metrics is
also listed under `metrics`, with `chunk_count` 0, `saved_path` `null`, and a
warning that metric uploads may be stalled. After the job has ended, such a
sub-job is listed the same way with a warning that no GPU metrics were saved
for it (for example, because the job ran on an image from before GPU metrics
were uploaded). `training_metrics` carries no such warning: only training
sub-jobs save training metrics, and only where the deployment saves them (a job
opts out with `training_config.save_training_metrics` false), so a sub-job
without them is simply absent, including one named with `--sub-job`. If the
training stream fails or is interrupted after the GPU stream finished, the
command still offers `--resume`, and on an error it prints the GPU result with
`training_metrics` `null` before exiting 1. Two downloads of the same directory
at once are unsupported.

### Model Deployments

`cortex-training model-deployment` manages ephemeral OpenAI-compatible
evaluation deployments while keeping their backing jobs and sub-jobs hidden.
It uses the same connection settings as every other command. See
[REST API reference section 14](rest-api.md#14-model-deployments) for the full
spec format, statuses, drop and download behavior, and errors.

`--database`, `--schema`, and `--compact` may be given either before
`model-deployment` or after its subcommand.

Create a deployment from a base model or a completed Cortex Training
`weights-only` checkpoint:

```bash
cortex-training model-deployment create \
  --database CORTEX_TRAINING_DB \
  --schema PUBLIC \
  --spec recipes/api_quickstart/model-deployment-checkpoint.json \
  --wait
```

`--spec -` reads the spec from stdin; `--spec-file` is a deprecated alias for `--spec`. The checkpoint must come from a completed
`save` whose checkpoint type is `weights-only`; resumable checkpoints and
customer-supplied stage credentials are not accepted. A checkpoint source
names only `job_id` and `checkpoint_id`; the model is read from the source
job. Base-model sources use `{"type": "base_model", "model": "Qwen/Qwen3-1.7B"}`
instead. The `deployment` object sets `class` (only `EVALUATION` is supported
today), `min_replicas`, `gpus_per_replica` (1, 2, 4, or 8), and `gpu_type`
(`H200`, `B200`, or `B300`, in any case); `min_replicas * gpus_per_replica`
must be at most 8 or a multiple of 8. `create --help` shows a full example
spec, and `create --dry-run` checks a spec and prints it without connecting.

`create`, `create --wait`, and `wait` print the same summary (plus `failure`
when the deployment failed):

```json
{
  "base_url": "https://ACCOUNT.snowflakecomputing.com/api/v2/databases/CORTEX_TRAINING_DB/schemas/PUBLIC/cortex-training/inference/v1",
  "deployment": "sf-ct-DEPLOYMENT_UUID",
  "status": "READY"
}
```

`--wait` (on `create`) and `wait` poll until the deployment reaches `READY` or
a terminal failure state, for up to `--timeout SECONDS` (default
`--poll-timeout`). If waiting fails, times out, or is interrupted with Ctrl+C
(exit code 130), the deployment ID and its last observed status are still
printed, and the deployment keeps running and holding GPUs until you `drop` it.

`list` shows only each deployment's ID and status, oldest first. `--status`
(repeatable, case-insensitive) keeps only `PENDING`, `INITIALIZING`, `READY`,
`FAILED`, `DROPPING`, `DROPPED`, `TERMINATED`, or `UNKNOWN` deployments:

```json
{
  "deployments": [
    {"deployment": "sf-ct-DEPLOYMENT_UUID", "status": "READY"}
  ]
}
```

`describe` shows one deployment in full: its status (with `status_reason` and
`status_details`, or `failure`, when present), `base_url` while it can serve,
the `database` and `schema` it lives in, `created_at`, `creator`,
`owner_role`, and its configuration grouped under `spec`. `spec` is a read view that may gain
read-only fields, so don't rely on feeding it back to `create`. A `gpu_type`
of `null` means the server did not report the hardware.

```json
{
  "base_url": "https://ACCOUNT.snowflakecomputing.com/api/v2/databases/CORTEX_TRAINING_DB/schemas/PUBLIC/cortex-training/inference/v1",
  "created_at": "2026-09-16T20:00:00Z",
  "creator": "JDOE",
  "database": "CORTEX_TRAINING_DB",
  "deployment": "sf-ct-DEPLOYMENT_UUID",
  "owner_role": "ML_ENGINEER",
  "schema": "PUBLIC",
  "spec": {
    "deployment": {"class": "EVALUATION", "gpu_type": "H200", "gpus_per_replica": 1, "min_replicas": 1},
    "inference": {"max_seq_len": 8192},
    "source": {"model": "Qwen/Qwen3-1.7B", "type": "base_model"}
  },
  "status": "READY"
}
```

`download-log` (execution logs by default, or `--log-type stdout`) and
`download-metrics` write every file directly into `--output-dir` (default: the
current directory) under its server filename, with mode `0600`, replacing a
file of that name from an earlier download. `drop` is safe to retry: a
deployment that no longer exists or is already dropping or dropped counts as
dropped. A deployment that has already failed or terminated holds no GPUs and
cannot be dropped, so `drop` prints `"dropped": false` with its `status` and
still exits `0`.

#### Calling A Deployment

The `base_url` is an OpenAI-compatible endpoint. Authenticate with a Snowflake
programmatic access token (PAT) and pass the deployment ID as `model`.
Inference is not yet available on all accounts; until it is enabled for yours,
requests return 404.

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://ACCOUNT.snowflakecomputing.com/api/v2/databases/CORTEX_TRAINING_DB/schemas/PUBLIC/cortex-training/inference/v1",
    api_key=PAT,  # sent as "Authorization: Bearer <PAT>"
    default_headers={"X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN"},
)
response = client.chat.completions.create(
    model="sf-ct-DEPLOYMENT_UUID",
    messages=[{"role": "user", "content": "Hello"}],
)
```

```bash
curl "$BASE_URL/chat/completions" \
  -H "Authorization: Bearer $PAT" \
  -H "X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model": "sf-ct-DEPLOYMENT_UUID", "messages": [{"role": "user", "content": "Hello"}]}'
```

Limit the reply length with `max_completion_tokens`; `max_tokens` is rejected
with HTTP 400 (`unsupported_parameter`). `"stream": true` returns server-sent
events; see [Deploy a Model for Streaming Inference](../guides/inference/deploy-model.md).

Use `cortex-training model-deployment list` to find deployments: `GET
$BASE_URL/models` can return an empty list while a deployment is `READY`
([known limitation 6](rest-api.md#15-known-limitations)). The full request and
response behavior is in
[REST API reference section 14.8](rest-api.md#148-calling-a-deployment).

### Job TUI

`cortex-training tui` is a read-only terminal dashboard for job logs and GPU
resources. It uses the same connection handling and fallback order as the CLI,
including named and configured-default Snowflake profiles:

```bash
cortex-training tui                 # opens the job list of the configured database and schema
cortex-training tui JOB_ID          # opens that job's dashboard directly
```

Without login state, pass connection details the same way as the CLI:

```bash
cortex-training tui JOB_ID --connection training
cortex-training tui JOB_ID --config config.json
cortex-training tui JOB_ID --host ACCOUNT.snowflakecomputing.com --pat YOUR_PAT \
  --database CORTEX_TRAINING_DB --schema PUBLIC --endpoint cortex-training
cortex-training tui JOB_ID --base-url http://localhost:8084   # local/mock
```

Keep the PAT out of your shell history by exporting it instead (omit `JOB_ID`
to open the job list):

```bash
export CORTEX_TRAINING_PAT='YOUR_PROGRAMMATIC_ACCESS_TOKEN'
cortex-training tui \
  --host ACCOUNT.snowflakecomputing.com \
  --database CORTEX_TRAINING_DB --schema PUBLIC
```

No command, the TUI included, creates the configured database or schema; when
the service answers that they are not found, the command stops with an error
that says how to fix it. Every request the TUI sends is retried at most twice, with a
10-second connect and a 45-second read timeout, where the other commands use
the SDK's defaults (ten retries, 30 and 600 seconds; [REST API reference
section 3.5](rest-api.md#35-retries)). So a job id that isn't found (404), or
a service that keeps failing, shows its error within a few seconds rather than
after about a minute. Pressing `r` while a refresh is still reading says so.

Each call the TUI makes, and each checkpoint download, sends one best-effort
client metric with its outcome and duration, successes included, tagged
`client.surface = tui` so the TUI's use can be told from the CLI's and the
SDK's. The downloads behind the Logs, Resources and Training metrics tabs also
say how long the first page or graph of a first download took (`preview_ms`,
`first_commit_ms`), not only the whole download. `CORTEX_TRAINING_DISABLE_TELEMETRY` turns them
off; see [Client metrics](python-sdk.md#client-metrics).

The TUI explains a refused read in one line on the view that needed it, worded
the same on every view: the Logs tab's saved console and execution log, and the
Resources tab (`GPU metrics unavailable: ...`). When the service can't resolve
the job's Experiment for this role (a 500 that says `Linked experiment is not
found ...`), the line says the role can't read the job's Experiment, and, for
`Linked experiment is not found for job ...`, that it may no longer exist. When
Snowflake refuses the SQL on the Experiment, as it does for a `MONITOR`-only
role, the line says that saved logs and GPU metrics live there and that
`checkpoints` still works. A refusal about another object (a warehouse, a
schema) keeps its own text. A 403 whose message names a privilege says the role
isn't allowed to read that view's data; one with no such message keeps the
standard error text.

A config the TUI can't use, such as one with a value of the wrong type (a
quoted number: `"poll_interval": "2"`), JSON that doesn't parse, an unknown
key, or a path that isn't there, stops it before it starts, with one line on
stderr (`cortex-training tui: error: config poll_interval must be a number`)
and exit status 2.

Pass `--sub-job-id JOB_ID:training:0` to scope the job details to one matching
sub-job. An unmatched id is shown as an error rather than falling back to a
different sub-job.
The error lists the job's sub-job ids; without `JOB_ID`, `--sub-job-id` opens
that sub-job's job.

#### Job list

The job list shows the jobs created in the last six months (183 days) that the
current role can see in the configured database and schema. It reads the
account-level list scoped to that one schema
([REST API reference section 5.3](rest-api.md#across-the-account---get-apiv2endpoint)),
so it comes a page at a time; where that route is unavailable for the account,
fails, or answers with no job at all, it reads `list_jobs`, which answers in one
call and shows no pages.
Press `a`, or click `a All schemas` in the footer, to list the jobs in every
database and schema the role can see instead, from the same list unscoped;
the footer then offers `a Current schema`, which goes back. The
header line starts with when the list was last refreshed (`refreshing…` while
a refresh reads), then which list is shown. The words naming the list stand out
as a chip, bold and filled: `Current schema: DB.SCHEMA` in white, `all schemas`
in blue. Where the header doesn't otherwise name the list (the all-schemas list
cut short, below), its chip comes first. Until the list `a` asked for
answers, the header reads `loading all schemas…` (or `loading the current
schema, DB.SCHEMA…`) and the previous list stays on screen. Either list shows
its first page (the newest 100 jobs) as soon as it answers, and adds each later
page as it comes: the highlighted row stays where it is on screen, and folds and
a search stay as they are. From its second page on, it also reads the active
jobs (see below) beside the newest ones, at most two requests at a time. Until
the last page, the header reads `all schemas · loading older jobs… N so far`
(`Current schema: DB.SCHEMA, loading older jobs… N so far` for the configured
schema's list; `+ every active job` once the active jobs are listed, `(older active
jobs may be missing)` if reading them failed or stopped short), the counts above
the list and a search's matches end in `so far`, and the LIVE and HISTORY rows'
counts in `…`. Once the active jobs are listed, the active statuses' counts drop
their `so far`; the `unknown` count and the LIVE row keep theirs until the last
page, as no status filter lists an `unknown` job. If the newest jobs are all in
while the active jobs are still being read, the header reads `newest 2,000;
loading older active jobs…`: the history count and the HISTORY row drop their
marks, and the live counts and the LIVE row keep theirs.
A list shows its pages as they come when it is first shown: as the TUI opens,
and after `a`. A refresh of a whole list keeps showing it until the next one is
whole. `a` while the pages are read stops reading them: the pages read stay, no
longer marked, under `loading the current schema, DB.SCHEMA… · showing the
partial all-schemas list` (or `loading all schemas… · showing the partial
current-schema list`) until that list answers. An `a` back before then stops
that list's pages the same way, and the pages start again from the first. `r` meanwhile does nothing (and
nothing is read again after): it says so, `Still loading older jobs (Ns); r
works once it's done.` (`older active jobs` while the header says so). Once the
list is whole, the header names it:

| Header | The list holds |
|---|---|
| `Current schema: DB.SCHEMA` | the jobs of the configured database and schema, as the list opens |
| `Current schema: DB.SCHEMA, newest 2,000 + every active job (more exist)` | the same, cut short as the all-schemas list can be (below, `first 40 pages: …` included): that one schema holds more jobs than one refresh reads; and where the account-level list is unavailable, so the schema's list comes whole, when it holds more than 2,000 jobs: `f` shows every one of them, and again the newest 2,000 |
| `all schemas` | after `a`: the jobs in every database and schema the role can see, from the account-level list; `all schemas (all in DB.SCHEMA)` when every one of them is in the same location |
| `all schemas, narrowed to DB.SCHEMA` | after a whole `db=` search (see below): the jobs of that one location, asked of the service rather than filtered out of the newest 2,000. A narrowed list cut short reads `narrowed to DB.SCHEMA: newest 2,000 + every active job (more exist)`, and `narrowing to DB.SCHEMA…` while it is read |
| `newest 2,000 + every active job (more exist)` | the same, cut at the newest 2,000 because more exist, plus the active jobs created before those |
| `first 40 pages: N jobs + every active job (more may exist)` | the same, cut after 40 pages (see below), plus the active jobs |
| `Current schema only: DB.SCHEMA (reason)` | the jobs of the configured database and schema, in place of the all-schemas list: `a` found it unavailable, or it turned unavailable while shown (see below). Followed by what it holds when that list is cut short, as above |

Each refresh reads the account-level list newest first, 100 jobs a
page. It stops after 2,000 jobs, or after 40 pages, because the service examines
a bounded number of jobs per call, so a page can be short or empty and still
have more after it: a refresh never walks the whole history. It asks for no
status, so jobs whose status the service can't read (`unknown`) are listed too.
When it stops with jobs left, it also reads the active jobs (`pending`,
`placing`, `initializing` and `running`) and adds those created before the
newest ones, so a long-running job stays under live. A refresh of a whole list
reads them after the newest jobs, only when it stops with jobs left. A list
showing its pages reads them beside the newest jobs; if the newest turn out
whole, the active jobs' walk asks for no page after the one in flight (a quick
walk may have read all of its pages by then, one or two in the usual case) and
its jobs are dropped. While an earlier load's active jobs' request is still in
flight (after `a` pressed again), a new load reads them after the newest jobs
instead. An `unknown` job older than
the newest 2,000 is not listed, as a status filter leaves those out, and if
reading the active jobs fails or is cut short too, the header says `older active
jobs may be missing`. A walk that stops at 40 pages with no job to show says
that more may exist and how to list one schema, rather than that there are none.

Search filters the jobs the list holds: when more exist, a search can't reach
the older ones, and the message for a search without results says so. A search
naming a whole database is the exception: `db=TEAM_DB`, on its own or together
with a whole `schema=EXPERIMENTS`, has the account-level list read again asking
the service for that location, so its jobs come whole instead of being filtered
out of the newest 2,000. The header reads `narrowing to DB.SCHEMA…` while that
is read and `narrowed to DB.SCHEMA` after; clearing the search reads every
schema again. The Schema column is hidden on a narrowed list, as every job it
holds is then in one location.

Only that form narrows at the service, which takes a database with an optional
schema and no other location filter. A partial match (`db:team`), a negation
(`db!=TEAM_DB`), a `schema=` with no `db=` of its own, and a name the service
would resolve to a different one (`db=team_db`, `db=ml`) all filter the jobs the
list holds, as before. A name narrows when it is already the name the service
resolves it to (`db=TEAM_DB`) or is quoted (`db='"ml"'`) — which is how the
Schema column shows every location, so a location copied from it always narrows.

To search the configured schema's jobs, press `a` to go back to its list, read
the same bounded way but scoped to that one schema: far likelier to hold every
job of it, though a schema with more jobs than one refresh reads is cut too.
`--database` and `--schema` choose the configured one at startup. These also go
with `--connection`, where they take the place of the profile's database and
schema:
`cortex-training tui --connection NAME --database OTHER_DB --schema OTHER_SCHEMA`.

If the account-level list is unavailable, `a` keeps the configured schema and
explains why, in a notice and in the header (`Current schema only: DB.SCHEMA
(reason)`, until the next `a`): `no account-wide list yet` or `account-wide list
not enabled` (404), `account-wide list: HTTP 403` for an unexplained 403, or the
service's 403/405 message. Once all schemas are showing, the same fallback reads
`Current schema only: DB.SCHEMA (reason)` too. A 403/405 refusal shared by the
schema route is a failed refresh instead. The TUI retries a fallback every 10
minutes, or on `r` after one minute; another `a` within that minute reuses the
last answer.

On refresh failure, the previous list stays under a red error and a
`refresh failed TIME (list from TIME)` header. A first-load error replaces the
list; `r` retries. When a later page of the account-level list fails after the
first pages showed, those stay, under a red error, with a `load failed TIME ·
newest N (the rest didn't load; r retries; …)` header; if the list turns
unavailable there (404), the configured schema's list takes their place, as
when `a` finds it unavailable. The list pauses under details and refreshes on
return when its last read is over 60 seconds old.

When the list holds jobs from more than one database and schema, a
**Schema** column after GPUs shows `DB.SCHEMA` (SQL-quoted when needed) and
is hidden when every job shares one, including on a list narrowed to one
location, whose header names it instead. The job preview names it too. Previews, detail reads, logs, metrics and
generated commands use the selected job's location. Commands also retain the
connection method and a non-default `--endpoint`; a direct `tui JOB_ID` still
uses the configured schema and suggests location flags after a 404.

`LIVE` groups active states and `unknown` (unless its reason says not found);
`HISTORY` groups the rest. Each group starts with a heading row that gives its
count, with a rule across the other columns (drawn only: no copy has it).

The line above the list counts the live jobs by status, then the rest, for
example `● 4 running   ◔ 1 placing   ○ 1 pending   ? 1 unknown  │  21 in
history` (counting the jobs a search matches, followed by `N matches for
QUERY`, while there is one), and ends with the sort (`sort: created ↓ newest
first`). The Status column draws each status as a glyph and the word:
`● RUNNING`, `◐ INITIALIZING`, `◔ PLACING` (and `creating`), `○ PENDING` (and
`queued`), `? UNKNOWN`, `× FAILED`, `− CANCELLED`, and `✓` for `terminated`,
`done` and `succeeded`.

Rows show the job id, status, type, created time, duration and owner, then
model, requested GPUs, optional location, and updated time. Created and Updated
say how long ago (`just now`, `12m ago`, `3h ago`, `45d ago`); the job preview
has both times in full, and a copy has them as `YYYY-MM-DD HH:MM` in local
time (see below). A time up to five minutes ahead of this
computer's clock reads `just now`; one further ahead reads how far ahead it is
(`in 2h`), which means this computer's clock and the service's disagree.
Duration is how long a live job has run so far (since it was created, so it
counts time spent waiting to be placed; in its status's color), or how long a
finished one ran until its last state change (`40s`, `12m`, `3h 05m`,
`2d 4h`). The times are as of the last refresh, which redraws them. On a narrow
screen the Job ID column narrows (its id ends in `…`) and stays in view while
the other columns scroll. The Schema column, shown when the list spans more than
one location, is as wide as its longest `DB.SCHEMA` where the screen has room,
and up to 32 cells where it doesn't; a name it cuts loses the end of its
database name first (`TEAM_D….EXPERIMENTS`), so the schema stays whole, and a
copy has the name whole. `shared` marks multiplexed GPU use and `log_prob`
requests zero GPUs.

A bar (`┃`) after each column's name marks the column's right edge. Drag a bar
sideways to make its column wider or narrower; the bar follows the pointer. A
press on a blank cell either side of a bar takes the bar too, and the bar under
the pointer brightens, so it is easy to catch; held, it is drawn in blue. A
column is at least 3 cells wide. The widths stay
through refreshes, `a`, searches and terminal resizes until the TUI exits; a
double-click on a bar gives its column back the width the list picks. A drag on
the names themselves highlights them as text, and a copy leaves the bars out.
Where the terminal draws pointer shapes, the pointer on a bar is a sideways
arrow.

A sub-job row is fainter than its job's and shows what it doesn't share with
it: its label (`training #0`), its status as the glyph alone while it is its
job's and as glyph and word where it differs, its model where the job has more
than one, and its GPUs (left out for a job's only sub-job when the job's row
gives the same count). `▸`/`▾`, arrows, clicks, and `e` control sub-job rows.
Every live job starts unfolded, its sub-job rows in view; history jobs start
folded. `e` unfolds every job (or folds every one, once all are unfolded),
those that later pages of the list being read bring too; the jobs a later
refresh or `a` adds start as above, and those `e` covered keep its choice.

`ctrl+c` copies text that holds up once pasted. With nothing highlighted, it
copies every row of the list (scrolled out of view too) in aligned columns,
each cell as its copy text, with a `Whole ID` column at the end holding each
row's whole job or sub-job id. A drag highlights characters on screen; `ctrl+c`
then copies those rows, where each cell the highlight covers whole is its copy
text, a cell it covers only in part is the characters it draws there, and the
columns stay in line. A cell's copy text is what it draws, but: Created and
Updated are the time itself, `YYYY-MM-DD HH:MM` in local time, rather than
`3h ago`; Status is the word alone (`RUNNING`, also on a sub-job row that draws
only the glyph), without the glyph; and an id cut with `…` is the whole id. The
rule of a `LIVE` or `HISTORY` row is never copied.

An `unknown` row retains known creation time and derives missing sub-job ids as
`JOB_ID:TYPE:INDEX`; incomplete configs read `config #N`. A supplied
`--sub-job-id` is still attempted when live state cannot be listed.

Search with `/`. Case- and Unicode-normalized words all must match an id,
status, owner, GPU, type, model, or the start of a location component — a word
matches the *start* of a database or schema name, so `db:TEXT` is the way to
match anywhere in one and `db=NAME` the way to match it whole. A word also
matches the configured location's own name on its list, where rows name no
location of their own. Quote text containing `:` (for example `"training:0"`).
`field<op>value` filters narrow further:

| Field | Meaning |
|-------|---------|
| `id`, `status`, `owner`, `type` | job id, status, user or role, `training`/`sampling`/`log_prob` |
| `model`, `method`, `dtype` | model name (`=` takes it whole, `Qwen/Qwen3-8B`, or as the Model column shows it, `Qwen3-8B`), `full` or `lora`, dtype |
| `gpu`, `gpus` | GPU type (`H200`, `B200`, `B300`), requested GPU count |
| `optimizer`, `lr`, `batch`, `seq` | optimizer name, learning rate, train batch size, max sequence length |
| `lora_r`, `lora_alpha` | LoRA rank and alpha |
| `db`, `schema` | database and schema name; a job of the configured schema's own list is in the configured ones. Either field also takes a whole `DB.SCHEMA` exactly as the Schema column shows it, with any operator: `schema=TEAM_DB.EXPERIMENTS` asks for both names whole, `schema:team_db.exper` for part of each, and `schema!=TEAM_DB.EXPERIMENTS` for jobs elsewhere. Only a dot outside quotes divides the two, so a quoted name holding one (`db='"my.db"'`) stays one name. `:` matches part of the name in any case (`db:team`). `=` and `!=` take the name as stored or as SQL resolves it, a bare name being its uppercase form: `db=team_db` and `db=TEAM_DB` match `TEAM_DB`, `db=ml` and `db='"ml"'` match a lowercase `"ml"`, and `db=ML` doesn't. A whole `db=` (with an optional whole `schema=`) is also asked of the service rather than filtered here, so it reaches jobs past the newest 2,000 — see the account-level list above |

Operators are `:` (contains, or numeric equality), `=`, `!=`, `>`, `>=`, `<`,
and `<=`. Unknown keys read job, sub-job, or config members; dotted keys read
nested members. Lists match any element (`!=` requires none). Examples:

```text
h200
model:qwen3-8b method:lora lr<=1e-4
gpu:b200 gpus>=8 lora_r=32
training_config.optimizer.name=adamw status:cancelled
training_config.peft_config.target_modules:q_proj seed=42
db:team_db status:running
db=TEAM_DB schema=EXPERIMENTS
schema=TEAM_DB.EXPERIMENTS
schema:team_db.exper
```

No-result messages name the searched scope, truncated or older matching jobs,
unknown fields (with dotted suggestions), `a` when a wider search may help, and,
on a list cut short that nothing narrowed, that `db=NAME` reads that whole
database from the service rather than only the jobs loaded.
Long messages scroll. The header gives the match count and query. The search
box is one row at the bottom of the screen; the help for its fields and
operators shows under it only while it has focus.

| Key | Action |
|-----|--------|
| `Enter` | Open the job preview (or, on a sub-job row, that sub-job's details, on the Logs tab; on a "status unavailable" line, its job's preview); in the search box, show the matches of all that was typed and move to the list, where `Enter` opens the highlighted match |
| `/` / `esc` | Search / clear the search. Keys are taken in the order typed, also when they come faster than the list redraws: what is typed before `Enter` or `esc` is in the search they apply or clear, and the keys after `/`, `Enter` and `esc` go to the search box or the list |
| `s` / `o` | Sort by created, updated, or status (`succeeded` and `completed` sort with `done`, `canceled` with `cancelled`) / reverse the order (so do a click on the sort's name and one on its arrow, in the line above the list). That line says which way it runs: `created ↓` for newest first or `↑` for oldest first (the same for updated; jobs without that time, shown as `—`, come last either way), `status ↓` for running first or `↑` for ended first (history first). The cursor goes to the first job |
| `Home` / `End` | First / last job |
| `→` / `←` | Unfold the highlighted job's sub-jobs / fold them; on a sub-job row, `←` folds its job and moves to it. The job's row stays where it is on screen, and the list scrolls only as far as an unfolded job's sub-jobs need to show. Where there is nothing to unfold or fold, they scroll the columns sideways |
| click `▸` / `▾` | Unfold or fold that job, as `→` / `←` do |
| drag a `┃` bar | Make the column before it wider or narrower (the bars sit between the column names, and after the last; a blank cell beside a bar takes it too); a double-click on a bar gives its column back its own width |
| `e` | Expand or collapse every job's sub-jobs, also those of jobs that later pages of the list being read bring (jobs a later refresh adds start as usual) |
| `a` | All schemas / current schema: switch to the account-level list of every database and schema the role can see (the footer's `a All schemas`), or back to the configured one (`a Current schema`) |
| `f` | All jobs / newest 2,000: when the configured schema's list came whole (the account-level list is unavailable) and holds more than 2,000 jobs, the list shows the newest 2,000 and every active job, as the account-level list does, and search covers those (one that matches none of them says `f searches all N jobs of the schema.`); `f` shows every job of it (drawing tens of thousands takes a few seconds), and again the newest 2,000. Refreshes keep the choice until another list is shown |
| `ctrl+c` | Copy what is highlighted, else every row in the list with a `Whole ID` column (see above for what each cell copies as); in the preview, the highlight, else the whole preview |
| `r` | Refresh (the list also refreshes every 60 seconds while it is showing); while it shows the configured schema in place of the account-level list (`Current schema only: DB.SCHEMA (reason)`), `r` asks that list again once a minute has passed since the last ask |
| `?` | Show or close the keys panel |
| `q` | Quit |

Every footer starts with Back (`b`, except on the job list) and Quit (`q`), then
`?`: a narrow terminal cuts the footer's end, and `?` opens a panel that lists
every key that works where you are, including the ones the footer has no room
for. `esc` closes that panel first, before it clears a search or goes back.

#### Job preview

Opening a job shows it as cards. The job card is headed by its id, its status
as a colored pill (`unknown` reads `? UNKNOWN`), and how long it has been
going (`12m 04s so far`, or `lasted 1h 20m` once it ended), with the keys for
`l Logs` and `g Resources` under it. Its facts follow: model, type, owner,
requested GPUs, creation and update times, and the location when it is not the
configured one, in two columns from about 90 columns wide and in one below.
Values stop after three rows; `e` expands every value and the complete sub-job
table, and toggles them short again.

**GPUs** sums each sub-job's `n_gpus`, excluding `log_probability` and
multiplexed sub-jobs that run on another sub-job's GPUs. The time beside the
status includes queue and placement time; an `unknown` job runs to now.
Missing times show `—`, and a clock up to five minutes ahead reads `just now`.

**Lifecycle** runs from `created_at` to `state_changed_at`, the two times the
job record has, and on to now while the job is live, with how long each step
took: `◇ Created 15:14 ── 3m 10s ─▸ ● Running since 15:17 ── 9m 02s so far ─▸
now`, or `◇ Created 2026-10-12 15:41 ── 2h 00m ─▸ × Failed 2026-10-12 17:41
(5d 22h ago)`. States in between have no time in the record and get no step;
a live job still in the state it was created in is one step (`○ Created,
pending since 15:14`). An `unknown` job's state has no time.

A callout box under the job card says where the job stands, titled and colored
by state: red for a failure (`Why it failed`, or `Failed before it got GPUs`);
orange while it waits (`Why it's waiting`, `Placing on GPUs`) or while its
status is unavailable; cyan while `initializing` (`Starting up`); gray once it
was cancelled, terminated or completed. It shows the service's `reason` and
`reason_details` (a failure always has a reason, `not reported` when none is
sent), or what the state means, and where to look next. A running job has no
box unless the service sends a reason. The Sub-jobs card shows type, status
(a badge such as `● RUNNING`), model and GPUs. Long cells use `…`, narrow
screens omit Type, and wider tables scroll with `←` / `→` or `Home` / `End`;
the card's bottom border says so (`e whole table`, `← → scroll`).

The Config card under it follows the highlighted row: that sub-job's submitted
config, exactly as `cortex-training get JOB_ID` prints that sub-job's entry
(indented JSON by key, values as the service returns them, aliases as
submitted), without the fields the service adds (`sub_job_id`, `status`). On
the entire-job row, each sub-job's goes under its name, side by side when each
has at least 40 cells. A line wider than its column is cut with `…` until `e`,
which also puts the sub-jobs one under the other.

Live previews refresh every 60 seconds, on return from details, or on `r`; a
dim line at the end of the job card reports reads, slow reads and refresh
errors while keeping the last copy. `Enter` or a click opens the highlighted
entire-job or sub-job scope on Logs (`⏎ opens its Logs` on the card's border).
`l`, `g`, `m` and `c` open Logs, Resources, Training metrics and
Checkpoints directly.

#### Job details

The line under the title is the job's identity: its id in bold, its status as
a badge such as `● RUNNING` (redrawn when the details see the job end), chips
for its type, models, scope (`all sub-jobs`, or `sub-job training:0`) and
location when it is not the configured one, then the service's `reason` and
`reason_details` (or the `unknown` sentence) while it sends them. When the
line does not fit on one row, the models are cut with `…`, then the id is left
out (the header names it too). Switch tabs with `l` (Logs), `g`
(Resources), `m` (Training metrics) and `c` (Checkpoints), or `t` for the
next tab, round to Logs after the last.

- **Logs** is the live and historical log view described below. The pane shows
  one sub-job at a time, the one selected in the list beside it, so its lines
  carry no sub-job tag; each starts with its time (`HH:MM:SS`), and its
  wrapped rows start under the end of that time. The list marks each sub-job
  with its status's glyph and colour (`●` running, `◐` initializing, `◔`
  placing, `○` pending, `×` failed, `−` cancelled, `✓` ended, `?` unknown), with the status under its name; the one the
  pane shows has an accent edge and a bold name. Saved logs (`s`) and copies
  stay as the lines came, uncoloured. In the pane the time is faint, the level
  has its severity's colour (a warning's or an error's message is tinted to
  match), the logger name is muted, a traceback is one block on a dark red
  background from its `Traceback (most recent call last):` to the exception
  that ends it (with any chained to it), and what the filter matches is marked
  in orange.
- **Resources** shows telemetry (`download_metrics`) for one sub-job at a
  time. `Space` (or a click on its name) shows the next one; `<` and `>` go
  back and forth. Its two-line pinned header names the sub-job, position,
  window, sample age and scrape interval while the stats and charts scroll.
  - The focused left panel switches between `GPU Resources` and `System
    Resources`, then `ALL` (the mean) and each GPU or node. `↑`, `↓` or a click
    selects an entry. Each unit shows its latest utilization after a four-cell
    meter of it, filled in eighths and coloured by health (as the KPI tiles
    below); stale units are dimmed, their meters faint, and a view with no rows
    reads `(none)`. A sub-job with only system rows opens there.
  - The header compares requested and reporting GPUs, or shows reporting nodes,
    CPU count and memory per node. Multiplexed sub-jobs point to the sub-job
    whose GPUs report their shared telemetry. A unit over two minutes or three
    sample intervals behind its view's newest sample is stale. When the newest
    sample itself is over three minutes or six intervals old, the header warns
    that no new samples are arriving and the windows end at that sample; a line
    over the KPI strip says these are the last values received, the tiles go
    faint and unjudged, and `GPUs reporting` reads `0/8`, an error. It also
    counts malformed or unsupported rows.
  - Under the header, a strip of KPI tiles for the GPU or node shown, or the
    mean: `GPU util` (with the spread across GPUs, `84–95%`), `GPU memory`
    (`95.9 / 141 GiB` and its %), `power` (and the total, `5.1 kW total`),
    `temp` (and the hottest GPU) and `GPUs reporting` (`8/8`, against those
    requested); the System view has `CPU util`, `memory` and `nodes
    reporting`. Only metrics the rows have get a tile. Each is coloured by
    health, and a value that warns or is an error has `!` or `×` at its
    edge, so it shows without color too. For the mean a tile is as healthy as
    its least healthy GPU or node: GPU util under 50% warns; GPU memory used from 95% warns; GPU
    temperature from 85 °C warns and from 90 °C is an error; CPU util from 95%
    warns; node memory used from 90% warns and from 97% is an error; power
    isn't judged. `GPUs reporting` warns below the GPUs requested and is an
    error at none. On a narrow pane the details go from the right, then the
    tiles wrap.
  - Numeric `_pct`, `_bytes`, `_watts` and `_c` fields become `%`, `GiB`, `W`
    and `°C` metrics. GPU and System begin with utilization and derived memory
    used; constant node CPU count and memory total stay in the header. Each
    metric has min, time-weighted average, max and latest stats for the window
    and whole job, plus a chart (two per row at 90 cells or wider). The stats
    are a table: column heads right-aligned over their values, the latest in
    bold, a rule between the window and the whole job. Each chart is an area
    chart: the series' edge in the chart's colour over a faint fill of it, the
    50% gridline running through the fill, and `◂` right of the newest column
    on the row its value reaches. A bucket without a sample stays blank over a
    broken baseline (`╌`), so a gap never reads as 0, and the marker is left
    out when the newest bucket is a gap. A chart's time axis spans its window
    (the whole job's spans the job): with more buckets than columns, a column is
    the mean of those it covers, and with fewer, the buckets widen to fill it. A
    job shorter than the window starts at the left edge and fills only its
    share, `◂` marking its newest sample part of the way across, the rest of the
    window blank; a window whose older samples are still loading ends at the
    right edge, the older samples to come at the left. Percentage
    axes are 0–100%; other axes use a stable rounded ceiling. `ALL` aggregates
    reporting units by sample interval; one missed node scrape is bridged,
    longer gaps remain gaps. History older than seven hours is folded into
    whole-job bins; `≤` or `≥` marks a late sample that makes the exact extreme
    unknowable.
  - `w` (or the header's window) cycles 10 minutes (the default), 30 minutes,
    1 hour, 6 hours and the whole job. The window selector at the bottom of
    the tab, `window  10m  30m  1h  6h  all`, lights the window shown, and a
    click on a segment shows that window. At its right end is `↑↓ GPU` (or
    `↑↓ node`) when the view has more than one; the footer shows the other
    keys.
  - Telemetry downloads when shown, refreshes every 30 seconds for a running
    selected sub-job, and refreshes on `r`. The first download and `r` fetch
    every sub-job; once the selected sub-job has telemetry, the 30-second
    refreshes fetch only it, and `Space` to a sub-job they have left behind for
    30 seconds or more refreshes it at once. A copy left behind is judged stale
    as of the newest refresh that succeeded, as if that refresh had fetched it
    too. A refresh rebuilds the window on
    screen and the whole job's (the stats beside every window); `w` to a window
    not on screen since shows it as it last was, with `refreshing…` in the
    header, for the moment it takes to rebuild. A rebuild that fails says
    `cached · refresh failed: …` instead, until `r` or the next refresh builds the
    window. While a download runs, the header counts the selected sub-job's chunks:
    `loading GPU metrics ━━━━━━━━━━━━ 340/1,440 chunks`. A view with nothing to chart shows
    a quiet card instead of the stats and charts: a title (`Waiting for the
    job to start`, `Waiting for the first samples`, `GPU metrics are on
    another sub-job`, `CPU and memory aren't collected`, `No usable GPU
    metrics`) over why. Before it runs, the tab waits
    without polling telemetry; a running sub-job without samples says they
    usually arrive after about a minute. A `log_probability` sub-job says it
    has no GPUs. A failed refresh keeps cached data and says
    `cached · refresh failed: …`. Data is kept in the TUI's per-location job
    cache; the selected sub-job and its newest cached samples
    appear first, and later opens download only new chunks (unless the job
    left the cache to keep it within its limit, which makes the next open a
    first one: see [Local data](#local-data)). On a first open,
    with nothing cached for the selected sub-job, the window on screen (the
    newest 10 minutes, where every job opens; at most the newest half hour)
    downloads before the rest of its history and is drawn at once, marked
    `loading older samples…` until the rest is in; a job short enough that this
    saves no time downloads whole. The download starts on a Snowflake session
    the job's screen logged in once its first log lines were drawn (see
    [Download sessions](rest-api.md#127-download-sessions)): the TUI keeps
    one open for its downloads while it runs, and closes it when it quits.
- **Training metrics** charts what the training sub-job's `/step` and
  `/fwd-bwd` calls returned, read from its own `training.jsonl`
  ([Training metrics download](rest-api.md#125-training-metrics-download)),
  which holds a row per call with that call's numbers as the client got them.
  Nothing is computed: the tab shows the saved numbers as they are, in two
  sections, one above the other:
  - **Per step**: the numbers in each `/step` result, against the step:
    `grad_norm` and `last_lr` first, then the rest in the order the rows have
    them (`weight_delta_norm_rank0_sample`,
    `weight_delta_norm_over_lr_rank0_sample`, …).
  - **Per call**, below it: the numbers in each `/fwd-bwd` result and its
    `metrics`, against the call, numbered in the order the calls ran:
    `avg_loss` first, then whatever the loss function returns (`approx_kl`,
    `entropy`, `ratio_*`, …). Each call keeps its own `avg_loss`; a step with
    two calls shows two points.
  - Numbers that aren't training signals are hidden until `a` shows every
    metric (`a` again hides them): per step, `global_steps` (the step itself),
    `update_successful`, and the fixed size of the weight and gradient samples
    (`weight_delta_rank0_sample_numel`, `_bytes`, `_tensors`,
    `gradient_sample_numel`); per call, `rank` (the worker the result came
    from, currently always 0) while every call's is 0: a call from another
    worker shows it. The packing counts (`rl_model_calls`, `sft_model_calls`,
    `sft_packed_rows`) always show. The header says how many are hidden
    (`3 hidden (a)`), or `all metrics (a)` while they show.
  - The header names the sub-job, its newest step, how many calls were saved,
    the age of the newest row and the window, and with two or more
    training sub-jobs its place among them. `Space` shows the next one.
    Unreadable rows are counted there, and so are steps that reported
    `update_successful` isn't 1 (`2 steps reported update_successful ≠ 1`).
  - Per metric: latest, min and max over the window, each with where it came
    from (`0.8 @ 2` per step; `0.8 @ 4 (2)` per call: call 4, of step 2), then
    a chart per metric, two per row at 90 cells or wider. The y axis spans the
    window's own lowest and highest value, since a loss rarely starts at 0 and
    a policy loss can be negative; with more steps or calls than columns, a
    column is the mean of the ones it covers, and with fewer, they widen to
    fill the chart.
  - `w` cycles all steps (the default), the last 100 and the last 1,000,
    counted back from the newest step, for both sections. Per call, the window holds those steps'
    calls, and the calls of a step still running.
  - **The newest steps from the client's own copy.** The client that makes a
    job's `/fwd-bwd` and `/step` calls appends each result's row to a local
    file the moment the result arrives, under the TUI cache
    (`local-training/<job>/<sub_job_id>/training.jsonl`), in the same format
    as the stage's (see [Local training metrics](python-sdk.md#local-training-metrics)).
    The tab shows the stage's rows and adds the local rows the stage hasn't
    received yet, matched by each call's `request_id`; once the stage has a
    row, its copy replaces the local one. It re-reads the local file every 2
    seconds while the tab is showing, so on the machine running the job a step
    appears within seconds, rather than after the stage's batched upload
    (30–40 seconds). A job opened anywhere else, or run partly from another
    machine, still shows every step the stage has.
  - It downloads when shown, refreshes on `r`, and re-reads every 30 seconds
    while a live job's tab is showing, reading only the rows added since (a
    job that left the cache downloads whole again: see [Local data](#local-data)).
    After the first download, a tab with one sub-job to show fetches only that
    one's chunks on those refreshes; `r`, even while one runs, fetches every
    sub-job's, and so finds a sub-job whose training rows start later. The
    file sits in the same per-location cache as the Resources tab's and is
    downloaded under the same lock, but the two are separate streams: each
    fetches its own chunks. A job with no training sub-job says so without
    calling the service; a sub-job that has not run yet, or that saved no rows,
    says which.
- **Checkpoints** lists the job's checkpoints (`list_checkpoints`, the same
  list as `cortex-training checkpoints JOB_ID`): id, creation time, global
  step, type (`resumable` or `weights-only`), loss and size. A job whose
  sub-jobs don't train says it saves none.
  - The list is read when the tab opens, every 30 seconds while the job is
    live, and on `r`. `s` sorts by created time (newest first), global step
    or type; `o` reverses the order. Checkpoints missing the sorted value come
    last.
  - Each checkpoint's export (`export_checkpoint`) is read once, in the
    background, for its file names and sizes. When the list doesn't give a
    checkpoint's step or type, the tab takes it from those files (a
    `global_step<N>/` folder; optimizer-state files mean `resumable`,
    `.safetensors` weights alone mean `weights-only`) and marks it
    `(from its files)`. The export's links are never shown or kept.
  - The highlighted checkpoint's details sit under the list: every field the
    service sent, its files, the job and training sub-job that saved it, and
    the job's training configuration as submitted.
  - **Download** (or `d`) downloads its files, with fresh export links, to
    `~/cortex-training-checkpoints/<job8>/<checkpoint_id>/`, showing progress
    beside the button; pressing it again stops the download, and the next one
    keeps the files already whole. One download runs at a time.
  - **New job from it: Training / Sampling / Training + sampling** copies one
    shell command to the clipboard: `printf '%s' '<CreateJob JSON>' |
    cortex-training ... submit -`. The body repeats this job's training
    sub-job, its sampling sub-job (or one GPU of the training model), or both,
    and its hardware, each with `source_checkpoint_info`. The service starts a
    training sub-job only from a `resumable` checkpoint (with the GPU count and
    parallelism of the job that saved it, which the copy keeps) and a sampling
    sub-job only from a `weights-only` one. So Training is offered for a
    resumable checkpoint, Sampling for a weights-only one, and Training +
    sampling pairs the checkpoint with one of the other type saved at the same
    step (disabled when there is none). The details show the copied command.

#### Logs

The **Logs** tab's left panel lists the job's sub-jobs; select one to tail its
logs (the zone-manager pod is the Ray head, so a sub-job's worker output is
included).
The tail polls only while the Logs tab is on screen: on another tab it stops
and later resumes from where it left off, so no lines are lost. While a job
runs, its status is also re-read every 60 seconds, so the view switches to the
saved console when the job ends.
Logs are cached locally so reopening a job replays instantly without
re-fetching from the server — under `~/.cache/cortex-training/` (or
`$XDG_CACHE_HOME/cortex-training/`), overridable with `CORTEX_TRAINING_TUI_CACHE_DIR`, within the
cache's limit (see [Local data](#local-data)).

The pane draws each line as text only: escape sequences in a line (colours,
cursor moves, clipboard or title writes) are removed and other control
characters show as `�`, so nothing in a log acts on your terminal; a line that
rewrites itself with carriage returns (a progress bar) shows its last state.
A leading timestamp shows in local time, as `HH:MM:SS`; the date is on a faint
rule above the first line of each day (`── Sun 18 Oct 2026 ─────`), at the top
of the pane and wherever the day changes (when the pane drops its oldest lines,
the rule comes back above the first one left). A highlight copies the lines as
they were logged, as `s` saves them: each line it covers whole with its whole
timestamp, without the day's rules or the indent of wrapped rows; a line it
covers only in part gives the highlighted part. A JSON log record, after the timestamp and any Ray
`(Actor pid=...)` prefix, shows as `time LEVEL logger: message`, with the
message's line breaks kept and an error or exception chain on the lines under
it; its other fields (host, ids) are left out. `s` saves the lines as they came.

The header's subtitle starts with the job's status. On the Logs tab it adds
`last line HH:MM:SS`, the newest live line's own timestamp in local time
(`last line received` with the time it arrived, for a line without one), and
any filter, level, or pause.

The bar under the pane says, at its left end, what the pane shows and whether
it follows the log: `● LIVE · following` while it tails a running sub-job and
keeps to the newest line; `‖ PAUSED · 12 new lines · p resumes` once you pause
(`p`) or jump to an error (`e`), as the subtitle's `PAUSED` says;
`↑ SCROLLED UP · 12 new lines · p follows` once you scroll up from the end
(each counts the lines that came since); `✓ SAVED CONSOLE` for a finished
job's saved console; `↓ DOWNLOADING training #0`, a bar and `8/12 chunks`
while that console downloads; `○ WAITING` while the job or sub-job waits to
start (the status line says why); and `× STOPPED · r retries` when the tail
stopped on an error. In the middle is the filter box (`/` focuses it). At its
right end are chips for what is found or set, in this order: where the last
jump landed (`error 2/5` or `no errors` for `e` / `E`, `problem 2/5` or
`no problems` for `n` / `N`), the level (`≥ WARNING`), and the filter's count
(`12 matches`, over the whole saved console, or the lines the pane holds for a
live tail). The jump's count follows the lines as they come and go (`error –/5`
once the marked line has left the pane); a re-wrap (a resize) keeps its mark and chip.

The line between the pane and that bar is the Logs tab's status line. It appears when the
job read is slow, when the tail is waiting, paused, or restarted, when the
filter or level hides every line (`no lines match …`), after `s` (for 15
seconds), or when the pane doesn't hold a sub-job's oldest lines:

- **Reading the job.** The Logs tab reads the job first, for its sub-jobs and
  status. The pane and status line show the read, elapsed wait, retry bounds,
  and any failure; `r` retries once the current read finishes.
- **Waiting to start.** While the job or the selected sub-job is `pending`,
  `placing`, or `initializing`, the line shows its reasons and the 60-second
  status check (`r` checks now). Pending work is not tailed; placing or
  initializing work may produce startup lines. Placement and not-yet-routed
  service answers wait and retry as status changes.
- **The job ends.** Once a job stops running, `tail-logs` answers 409, and 404
  once its session is gone. The TUI confirms the status and switches to the
  saved console instead of retrying.
- **The tail stops on an error.** Any other error while the job is live stays
  in the pane as `tail stopped: …`; `r` and active status checks retry. A
  request without an answer for 20 seconds is reported.
- **Status unavailable.** While the job's status is `UNKNOWN` (the service
  could not read it), the tail is still attempted and the status line warns.
- **A refused position.** The tail resumes from the position (cursor) it last
  saved. If the service refuses it with 400, the TUI restarts at the end and
  marks the possible gap.
- **A log ahead of the view.** While pages come back full (the log is written
  faster than a page per poll), reads run every 0.2 seconds until caught up;
  the line reports lag of at least 5 seconds.
- **A stuck tail.** On some service versions a log line of 256 KiB or more is
  never delivered. After 45 seconds without movement, the TUI probes the end,
  skips lines already shown, and marks any possible gap; `download-log
  --log-type stdout` retrieves the complete uploaded log.
- **No input for 15 minutes.** A `tail-logs` call on a running job currently
  keeps the job active, so the TUI pauses after 15 minutes without a key,
  click, or scroll. `p`, `r`, or another sub-job resumes it; waiting time
  before the job runs does not count.
- **Scrollback.** The pane keeps a sub-job's newest 20,000 lines (fewer when
  long lines wrap to many rows). A line that would take more than 20 rows
  shows its beginning and end (`s` saves all of it). First open and cache
  replay use the newest 2,000 lines. When history is missing, the line gives
  the retained (or matching) count and a `download-log` command.

A finished job loads its saved console (the whole job's output, which can
take minutes for a long job) when its Logs tab is first shown: the selected
sub-job's first, shown as soon as it is in, then the others' in the
background, with its progress on the bar under the pane meanwhile (the other
sub-jobs' chunk count on the status line, once the selected one's is in). The pane says
which sub-job's console comes first, and, for a sub-job you select before its
console is in, that it comes with the others'. A long job's console can take
minutes: the sub-job on screen first shows the console's last 30 minutes, under
`— the saved console's last 30 minutes; the older lines are still downloading —`,
until its whole console is in. A job that ended more than 10
minutes earlier keeps that console in its per-location job cache, so
reopening it is instant. Kept consoles count toward the cache's limit (see
[Local data](#local-data)): past it, the jobs opened least recently are removed
(and downloaded again if opened). A
console fetched sooner, for example right after the job ended while the view
was open, may still lack its last lines, which are often the error: the pane
says so under it, and `r` downloads the chunks uploaded since. Until then, the
live tail's lines that came after the console's last one stay on screen after
it, under `— N newer lines from the live tail, not in the saved console yet —`,
so the end of the job doesn't take away lines you saw. A sub-job with no
console yet says instead that its saved logs may not all be uploaded yet; `r`
checks again for its console and its execution log.
A job that never got GPUs has no logs at all: one that failed with a
`placement_*` reason (such as `placement_timeout`), `hardware_unavailable` or
`pending_timeout`, or one that ended while every sub-job was still waiting to
be placed. Its pane says so with the reason, for example "No log output: the
job never got GPUs (reason: placement_timeout), so no sub-job started or wrote
logs.", and its preview says the same instead of pointing to Logs.
Jobs that fail while they are created or start up may have no saved console.
For a failed job's sub-job without one, the pane shows the error records from
that sub-job's execution log (`_logs/<sub_job_id>/execution.jsonl` on the job's
Experiment) under a header, each with its error message and exception chain;
when the log has no error records, its last 20 records.

The filter (`/`) and the level key apply to the lines the view already has:
changing either repaints the pane and never restarts the tail or sends a
request. A saved console is searched whole. The level key keeps lines
at or above the chosen level, judged from a JSON line's `level` field or a
level word such as `WARNING` in its first 512 characters; lines with no level
always stay, and a record that carries an error or an exception counts as
`ERROR` whatever its level says. Both apply to live
lines, saved consoles, and the execution-log records shown for a failed job
without a console, the latter a whole record at a time (a match in its
exception chain keeps the record); their header stays.

The TUI also writes two kinds of file into your home directory: saved logs from the `s`
key (`~/cortex-training-<job8>-<source>.log`, where `<job8>` is the first eight
characters of the job id) and its own error log
(`~/.cortex-training-errors.log`). Past 5 MiB the error log keeps only its
newest half, in the same file (a `.errors.lock` in the cache directory, not
beside the log, keeps two TUIs from writing it at once).
`<source>` is the sub-job's type and index, such as `training-0`. `s` saves the
sub-job's saved console, else the lines its tail cached, else the execution log
the pane shows; with nothing to save it says so and writes no file.

Copying works everywhere: drag to highlight, then press `ctrl+c` (or Cmd-C
where the terminal passes it through). A drag on a `┃` bar between the job
list's column names resizes a column instead (see above). Drags stay in their log, list, pane, or
table region and scroll it at an edge; table selection follows characters
across rows and hidden horizontal columns. Small row movements remain clicks,
while a drag never activates the row. Changing the shown tab, source, unit,
window, or rows clears the highlight.

The TUI owns the mouse selection and copies safe text both locally (`pbcopy`,
`wl-copy`, `xclip` or `xsel`) and through OSC 52; a note reports the result.
Use the terminal's selection modifier instead when needed (Option in iTerm2,
Fn in macOS Terminal, or configured Option in VS Code). With the keys panel
open, a whole-screen copy includes its keys.

Keys in the job details view (`/`, `e`, `n`, `p`, `s`, `[` and `]` work on Logs; `s`, `o` and `d` on Checkpoints):

| Key | Action |
|-----|--------|
| `l` `g` `m` | Logs / Resources / Training metrics tab, as in the preview, except `l` on Logs (below) |
| `t` | Next tab, round to Logs after the last |
| `a` | Show every metric, or hide again the ones that aren't training signals (Training metrics) |
| `w` | Cycle the window; so does a click on the window in the header, and a click on a window in the selector at the bottom shows it (Resources); `w` also cycles all steps, the last 100 and the last 1,000 (Training metrics) |
| `Space` | Next sub-job, round to the first after the last, when the job has two or more; the view stays, back to `ALL`. `<` / `>` go back / forward; a click on the sub-job's name in the header goes forward (Resources). On Training metrics, the next training sub-job |
| `↑` `↓` | Pick GPU Resources or System Resources, `ALL` or one GPU or node in the panel; the charts follow the highlight (Resources) |
| `/` | Filter the current source |
| `l` | Cycle minimum log level (INFO / WARNING / ERROR) (Logs; `L` works too); elsewhere the Logs tab |
| `e` / `E` | Next / previous error in the pane (an `ERROR`, `CRITICAL` or `FATAL` line, or a traceback: one stop with the exceptions chained to it, at the error line right above it when there is one), marked and a third of the way down the view, with its place among them on the bar (`error 2/5`). From the end of a followed log, `e` goes to the newest; both go round past either end. A jump pauses following, as `p` does |
| `n` / `N` | Next / previous warning or error, as `e` / `E`; the bar counts them as problems (`problem 2/5`, or `no problems`) |
| `p` | Pause / resume auto-scroll. Scrolled up from the end of a live log, or after a jump, `p` goes back to the end and follows again |
| `s` | Save the current source to `~/cortex-training-<job8>-<source>.log` (Logs); cycle the sort: created, step, type (Checkpoints) |
| `o` | Reverse the sort (Checkpoints) |
| `d` | Download the highlighted checkpoint, or stop its download (Checkpoints) |
| `ctrl+c` | Copy what is highlighted, else everything the tab holds: the log (its newest lines, up to 1 MiB) on Logs; the panel, header, stats and every chart on Resources; the header, stats and charts on Training metrics; the list and the highlighted checkpoint's details on Checkpoints (any tab) |
| `[` / `]` | Narrow / widen the sources panel |
| `PgUp` `PgDn` `Home` `End` | Scroll the log, also while the sources panel has the focus (Logs); scroll the stats and charts, also while the panel has the focus (Resources) |
| `r` | Refresh the visible tab (on Logs, the sub-job list and its tail) |
| `b` / `esc` | Back, or to the job list for a job opened directly (`esc` closes the keys panel first, and on Logs clears an active filter first) |
| `?` | Show or close the keys panel |
| `q` | Quit |

The `--poll-interval` flag (default `1.0s`) is the minimum interval between log
polls per source, biasing toward server reliability over freshness.
(A tail catching up with full pages polls faster, above.) The config file's
`poll_interval` is the CLI's result-polling interval and doesn't change it.

#### Local data

Each opened job is cached under `~/.cache/cortex-training/jobs/` (or
`$XDG_CACHE_HOME/cortex-training/jobs/`, or `$CORTEX_TRAINING_TUI_CACHE_DIR/jobs/`):
its log tail, finished console, and telemetry. To bound disk usage across long
or repeated sessions, the cache holds at most 5 GiB. That limit covers the jobs
and the local training metrics the client writes (`local-training/`) together,
so the two are one budget rather than two that add up. Those rows count up to
the 1 GiB the client holds them to, but the TUI never removes them: they are how
the Training metrics tab shows a step run from this machine before the stage has
it, and for a job that does not save its training metrics to the stage, the
only copy (the client prunes them by its own rules; see
[Local training metrics](python-sdk.md#local-training-metrics)).

Soon after startup, every 10 minutes while running, a few seconds after a
download that grew a job's telemetry by 64 MiB or more, and after keeping a
saved console (once the pane has it), one TUI at a time (where the filesystem
supports locks) removes the least-recently-used jobs until the cache fits, and
says nothing: open or downloading jobs can keep it over the limit. A job goes
whole, its log tail and saved log positions with its telemetry, so a removed
job downloads again when reopened, as on a first open. Only TUI-named real job
directories holding TUI files are counted or removed, and no link is followed
when counting or removing. The files an operating system leaves in a folder
someone looked at (Finder's `.DS_Store` and `._*` files, NFS's `.nfs*` files)
count as part of the job, or of the local rows, and do not make a job look
recently used; any other entry the TUI did not write leaves its directory alone,
and uncounted. A job cached by a TUI from before saved consoles moved to
`console/` (its `stdout/`) is counted and removed the same way.

Jobs open in a running TUI or actively downloading (a `download-metrics` into a
job's cache directory included) are never evicted. Opening a job creates its
directory, with a `.last-used` timestamp and the lock files the TUI and its
downloads take (`.lease`, `.cursors.lock`, `.console.lock`, `.metrics.lock`;
`.cleanup.lock` and `.errors.lock` sit in the cache directory itself). An open
job screen holds a shared lock on `.lease` (kept across `Ctrl+Z`, released by
the OS on exit) and refreshes a `.leases/` marker every 5 minutes (valid for 30
minutes across shared filesystems); closing a screen keeps the job protected by
its marker for a 5-minute grace period for in-flight reads, and quitting
releases it immediately. A TUI from before leases existed holds none, so update
every TUI sharing a cache. Nothing is wiped on quit. Eviction takes the lease
and download locks exclusively and renames the job to `jobs/.removing-*` before
deleting it. The startup and 10-minute passes also sweep abandoned console
staging directories, which live in the system temp directory
(`cortex-training-stage-*`, judged by their `.owner` lock, or after a week for
older TUIs), interrupted `.removing-*` dirs, `.console-*` / `.tmp-*.swp` partials
older than an hour, and lease markers older than 30 minutes.

#### Terminals

The TUI draws the same screens in any terminal that runs full-screen programs.
What a terminal changes is how many colors it says it has, in `COLORTERM` and
`TERM`, and the TUI draws its palette for each:

| The terminal says | The TUI draws | Usually |
|-------------------|---------------|---------|
| `COLORTERM=truecolor` (or `24bit`) | the palette as it is | iTerm2, VS Code, Cursor, Ghostty, kitty, WezTerm, Windows Terminal |
| `TERM=...-256color`, no `COLORTERM` | each color as the nearest of 256, so surfaces, cards and the header stay apart as grays | macOS Terminal, a session over SSH (which passes `TERM` on but not `COLORTERM`), tmux with `default-terminal` `tmux-256color` |
| `TERM=xterm`, `screen` or `linux`, or no `TERM` | 16 colors, with a note as it starts that shades are approximate | tmux or screen left at plain `screen` |
| `TERM=dumb` or `unknown` | grays, with 256 colors, so the highlighted row still shows | a tool's command runner |

Where the terminal has 24-bit color but doesn't say so, set
`COLORTERM=truecolor` where the TUI starts: on the remote host over SSH (or send
it, with `SendEnv COLORTERM` in `~/.ssh/config` and `AcceptEnv COLORTERM` on the
server), and in tmux, whose outer terminal also needs
`set -as terminal-features ',*:RGB'`. `TEXTUAL_COLOR_SYSTEM` (`truecolor`,
`256` or `standard`) sets the colors outright.

`NO_COLOR` set to anything but an empty string ([no-color.org](https://no-color.org))
draws every color as its gray, and the TUI says so as it starts. `--color always`
draws color anyway, `--color never` draws gray in any terminal, and
`--color auto` is the default. No screen depends on color alone: a job's status
has its glyph, and a KPI tile that warns or is an error has `!` or `×` at its
edge. Some command runners set `NO_COLOR` and `TERM=dumb` for the commands they
run (an AI agent's, such as Cursor's): a TUI started there is gray until
`--color always`.

Each glyph the TUI draws is one cell wide, and none is also an emoji (which some
terminals draw in color, two cells wide). Borders, tables and charts line up
where the terminal counts East Asian ambiguous-width characters (box drawing,
`●`, `…`) as one cell, as terminals do by default; set to draw them double
width (iTerm2's "Ambiguous characters are double-width", some CJK setups),
every border breaks, as in any Textual app. Every view works at 80×24; a
smaller terminal cuts columns, long values and the footer's keys, and `?`
still lists every key.

Over SSH the TUI copies through OSC 52 alone, since this machine's clipboard
tools would copy on the remote host. The terminal has to accept OSC 52 (some ask
first), and in tmux it needs `set -s set-clipboard on`.

### JSON Output And Help

Commands other than the TUI write JSON to stdout, pretty-printed by default.
Use the global `--compact` flag for compact JSON, or pipe output to `jq`:

```bash
cortex-training --compact list
cortex-training get JOB_ID | jq '.sub_jobs'
```

Use `cortex-training --help` to list commands and global flags, or
`cortex-training COMMAND --help` for command-specific arguments. Job-scoped
data-plane help includes the required global option:

```text
usage: cortex-training --job JOB_ID fwd-bwd [-h] json_file
```

### Environment Variables

Connection values can also come from:

```bash
CORTEX_TRAINING_CONNECTION
SNOWFLAKE_DEFAULT_CONNECTION_NAME
CORTEX_TRAINING_CONFIG
CORTEX_TRAINING_BASE_URL
CORTEX_TRAINING_HOST
SNOWFLAKE_HOST
CORTEX_TRAINING_PAT
SNOWFLAKE_PAT
CORTEX_TRAINING_DATABASE
SNOWFLAKE_DATABASE
CORTEX_TRAINING_SCHEMA
SNOWFLAKE_SCHEMA
CORTEX_TRAINING_ENDPOINT
```

`CORTEX_TRAINING_DISABLE_LOCAL_TRAINING_METRICS` (truthy) stops the client
copying each `/fwd-bwd` and `/step` result to the local training-metrics file
the TUI's Training metrics tab reads the newest steps from (see
[Local training metrics](python-sdk.md#local-training-metrics)).

`CORTEX_TRAINING_DISABLE_TELEMETRY` (truthy) skips OTLP client metrics on
Snowflake profile and PAT clients. Failures are emitted by default, and so are
successful downloads (`download-log`, `download-metrics`) with their timings,
and every successful call `cortex-training tui` makes. Each metric names the
program that made the call in `client.surface`: `cli` for these commands, `tui`
for the TUI, `sdk` for Python code. `CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY`
set truthy also emits successful outcomes for every other essential operation;
set falsy, it emits failures only, the TUI's included. See the [Python SDK reference](python-sdk.md#client-metrics).

`NO_COLOR` (not empty) draws `cortex-training tui` in gray unless it is started
with `--color always`; `COLORTERM`, `TERM` and `TEXTUAL_COLOR_SYSTEM` say how
many colors the terminal has. See [Terminals](#terminals).

`CORTEX_TRAINING_DISABLE_TENSOR_PROMPTS` (truthy) sends pre-tokenized prompts
as JSON lists inside the request frame instead of as tensors. The request body
stays a DSSST1 frame either way.

`CORTEX_TRAINING_FORWARD_OPERATION_MAX_JSON_BYTES` is the per-request JSON body
ceiling for byte `forward()` `/operation` posts (integer bytes, default 16 MiB,
minimum 16 KiB). Frames larger than that are split into request chunks.

### Troubleshooting

If you see `provide --base-url for local/mock use, or both --host and --pat`,
the CLI found a `host` but no PAT. Add `"pat": "..."` to `config.json` or set
`CORTEX_TRAINING_PAT`.

If no legacy config or complete direct connection is present, the CLI falls
back to the Snowflake Connector's configured default. If your profile is not
named `default`, pass `--connection NAME` or set
`SNOWFLAKE_DEFAULT_CONNECTION_NAME`.

If Snowflake rejects your PAT, the CLI keeps the original error and adds next
steps: `394400 (08001) ... Programmatic access token is invalid` from a
connection profile, or a `401` on a request that sent the PAT directly
(`config.json`, `--pat`, or `CORTEX_TRAINING_PAT`). The most common cause is a
user with no network policy; see
[Network policy requirement](../getting-started/authentication.md#network-policy-requirement).
A `401` on a connection-profile request gets no hint, because the PAT was
already accepted at login.

If you see `Invalid URL ... No scheme supplied`, the config is using a bare
Snowflake hostname as `base_url`. Use `host` for Snowflake PAT auth, or use a
full local/mock URL such as `http://localhost:8084` for `base_url`.

For server errors, the CLI prints any Snowflake request id and response body
returned by the service. Include those details when debugging a `500`.
