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
operations. Use a Python loop or a [recipe](../../recipes/README.md) for dataset
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

### [Connection](#connection-config)

```bash
cortex-training --connection training list    # Use a Snowflake profile
cortex-training list                          # Use the configured default profile
cortex-training login config.json             # Remember config for future commands
cortex-training login --config config.json    # Equivalent login syntax
cortex-training --config config.json list     # Use config for one command
```

### [Submit Jobs](#submit-a-job)

```bash
cortex-training submit examples/api/training.json
cortex-training submit examples/api/sampling.json
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
cortex-training capacity                      # Every GPU type your account supports
cortex-training capacity --hardware H200
```

### [Training And Generation](#run-a-forward-backward-smoke-test)

```bash
cortex-training --job JOB_ID fwd-bwd examples/api/fwd-bwd.json
cortex-training --job JOB_ID step             # Default learning rate: 1e-4
cortex-training --job JOB_ID step --lr 2e-5
cortex-training --job JOB_ID generate examples/api/generate.json
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
[persisted stdout](#download-persisted-stdout), and [GPU metrics](#download-gpu-metrics)
for download paths and output fields.

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

## Detailed Reference

- [Connection config](#connection-config), [login](#login), and [environment variables](#environment-variables)
- [Submit](#submit-a-job), [manage jobs](#manage-existing-jobs), and [GPU capacity](#show-current-gpu-capacity)
- [Forward-backward and optimizer steps](#run-a-forward-backward-smoke-test)
- [Load checkpoints](#load-a-checkpoint-into-a-running-job) and [initialize sampling](#start-sampling-from-a-training-checkpoint)
- [Generate](#run-a-generate-smoke-test) and [sync weights](#sync-training-weights)
- [Download logs](#download-execution-logs), [stdout](#download-persisted-stdout), and [metrics](#download-gpu-metrics)
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
[REST API reference section 5.4](rest-api.md#54-capacity---get-capacity) and
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
cortex-training submit examples/api/training.json
cortex-training submit examples/api/sampling.json
```

That file creates a training sub-job for `Qwen/Qwen3.6-35B-A3B` with
`training_config.model_provider` set to `prime_rl`.

### Run A Forward-Backward Smoke Test

After the training job is running, send one tokenized training batch:

```bash
cortex-training --job JOB_ID fwd-bwd examples/api/fwd-bwd.json
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
for the recipe workflow, or
[Start sampling from saved weights](rest-api.md#134-start-sampling-from-saved-weights)
for the Python save-and-create example.

### Run A Generate Smoke Test

After a sampling job is running, send readable prompts with sampling
parameters:

```bash
cortex-training --job-id JOB_ID generate examples/api/generate.json
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
(repeatable) limits the download to those sub-jobs; naming a sub-job with no
stdout chunks is an error. Transient connection and service errors are retried
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

### Download GPU Metrics

Reconstruct each sub-job's GPU metric chunks into
`<output_dir>/<sub_job_id>/gpu.jsonl`:

```bash
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics --resume
```

The command prints the saved path, chunk count, and first/last logical artifact
URIs for each reconstructed file. `--resume`, `--parallel N`, `--sub-job
SUB_JOB_ID`, `--strict`, `--stale-after-minutes`, `--transfer-timeout`,
damaged chunks, retries,
progress on stderr, and Ctrl-C behave as for
[persisted stdout](#download-persisted-stdout). While the job is running, a
sub-job that has saved stdout but no GPU metrics is also listed, with
`chunk_count` 0, `saved_path` `null`, and a warning that metric uploads may be
stalled. After the job has ended, such a sub-job is listed the same way with a
warning that no GPU metrics were saved for it (for example, because the job ran
on an image from before GPU metrics were uploaded). Two downloads of the same directory at once are unsupported.

### Job TUI

`cortex-training tui` is a read-only terminal dashboard for job logs and GPU
resources. It uses the same connection handling and fallback order as the CLI,
including named and configured-default Snowflake profiles:

```bash
cortex-training tui                 # opens the job list of the configured database and schema
cortex-training tui JOB_ID          # opens that job's dashboard directly
cortex-training tui --debug         # synthetic jobs, no connection needed
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

Like `checkpoints` and `list --all`, the TUI never creates the configured
database or schema, which the other commands do when the service answers that
they are not found: it shows that error instead, as it does when this role
can't use them. Every request the TUI sends is retried at most twice, with a
10-second connect and a 45-second read timeout, where the other commands use
the SDK's defaults (ten retries, 30 and 600 seconds; [REST API reference
section 3.5](rest-api.md#35-retries)). So a job id that isn't found (404), or
a service that keeps failing, shows its error within a few seconds rather than
after about a minute. Pressing `r` while a refresh is still reading says so.

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
that sub-job's job. `--debug` runs against generated jobs and telemetry, and
keeps its caches in a temporary directory, removed when it ends; the title says
`(debug)`. Its jobs are in three locations (the configured
`CORTEX_TRAINING_DB.PUBLIC`, `TEAM_DB.EXPERIMENTS`, and a lowercase
`"sandbox"."dev"`) and run on the wall clock: live jobs keep logging and
reporting GPU and node samples (one reports its GPUs only), a placing job
starts within two minutes, a
running one is cancelled, a tail gets stuck behind a long line and restarts from
the end, and others have ended, failed, are `UNKNOWN`, or sit on an Experiment
this role can't read.

#### Job list

The job list shows the jobs created in the last six months (183 days) that the
current role can see in the configured database and schema (`list_jobs`).
Press `a`, or click `a All schemas` in the footer, to list the jobs in every
database and schema the role can see instead, from the account-level list
([REST API reference section 5.3](rest-api.md#across-the-account---get-apiv2endpoint));
the footer then offers `a Current schema`, which goes back. The
header line starts with when the list was last refreshed (`refreshing…` while
a refresh reads), then which list is shown:

| Header | The list holds |
|---|---|
| `Current schema: DB.SCHEMA` | the jobs of the configured database and schema (`list_jobs`), as the list opens |
| `all schemas` | after `a`: the jobs in every database and schema the role can see, from the account-level list; `all schemas (all in DB.SCHEMA)` when every one of them is in the same location |
| `newest 2,000 + every active job (more exist)` | the same, cut at the newest 2,000 because more exist, plus the active jobs created before those |
| `first 20 pages: N jobs + every active job (more may exist)` | the same, cut after 20 pages (see below), plus the active jobs |
| `Current schema only: DB.SCHEMA (reason)` | the jobs of the configured database and schema, in place of the account-level list where it turned unavailable while shown (see below) |

After `a`, each refresh reads the account-level list newest first, 500 jobs a
page. It stops after 2,000 jobs, or after 20 pages, because the service examines
a bounded number of jobs per call, so a page can be short or empty and still
have more after it: a refresh never walks the whole history. It asks for no
status, so jobs whose status the service can't read (`unknown`) are listed too.
When it stops with jobs left, it also reads the active jobs (`pending`,
`placing`, `initializing` and `running`) and adds those created before the
newest ones, so a long-running job stays under live. An `unknown` job older than
the newest 2,000 is not listed, as a status filter leaves those out, and if
reading the active jobs fails or is cut short too, the header says `older active
jobs may be missing`. A walk that stops at 20 pages with no job to show says
that more may exist and how to list one schema, rather than that there are none.

Search filters the jobs the list holds: when more exist, a search can't reach
the older ones, and the message for a search without results says so. To
search all of the configured schema's jobs, press `a` to go back to its list;
for another schema, start the TUI with that schema's `--database` and
`--schema`.

If the account-level list is unavailable, `a` keeps the configured schema and
explains why: `no account-wide list yet` or `account-wide list not enabled`
(404), `account-wide list: HTTP 403` for an unexplained 403, or the service's
403/405 message. Once all schemas are showing, the same fallback reads
`Current schema only: DB.SCHEMA (reason)`. A 403/405 refusal shared by the
schema route is a failed refresh instead. The TUI retries a fallback every 10
minutes, or on `r` after one minute; another `a` within that minute reuses the
last answer.

On refresh failure, the previous list stays under a red error and a
`refresh failed TIME (list from TIME)` header. A first-load error replaces the
list; `r` retries. The list pauses under details and refreshes on return when
its last read is over 60 seconds old.

When the list holds jobs from more than one database and schema, a
**Location** column after GPUs shows `DB.SCHEMA` (SQL-quoted when needed) and
is hidden when every job shares one. Previews, detail reads, logs, metrics and
generated commands use the selected job's location. Commands also retain the
connection method and a non-default `--endpoint`; a direct `tui JOB_ID` still
uses the configured schema and suggests location flags after a 404.

`LIVE` groups active states and `unknown` (unless its reason says not found);
`HISTORY` groups the rest. Rows show id, status, type, creation, owner, model,
requested GPUs, optional location, and update time. Narrow screens compact
times and ids while fixed marker/id columns remain visible. `shared` marks
multiplexed GPU use and `log_prob` requests zero GPUs. `▸`/`▾`, arrows, clicks,
and `e` control sub-job rows; live jobs start expanded.

An `unknown` row retains known creation time and derives missing sub-job ids as
`JOB_ID:TYPE:INDEX`; incomplete configs read `config #N`. A supplied
`--sub-job-id` is still attempted when live state cannot be listed.

Search with `/`. Case- and Unicode-normalized words all must match an id,
status, owner, GPU, type, model, or the start of a location component. Quote
text containing `:` (for example `"training:0"`). `field<op>value` filters
narrow further:

| Field | Meaning |
|-------|---------|
| `id`, `status`, `owner`, `type` | job id, status, user or role, `training`/`sampling`/`log_prob` |
| `model`, `method`, `dtype` | model name (`=` takes it whole, `Qwen/Qwen3-8B`, or as the Model column shows it, `Qwen3-8B`), `full` or `lora`, dtype |
| `gpu`, `gpus` | GPU type (`H200`, `B200`, `B300`), requested GPU count |
| `optimizer`, `lr`, `batch`, `seq` | optimizer name, learning rate, train batch size, max sequence length |
| `lora_r`, `lora_alpha` | LoRA rank and alpha |
| `db`, `schema` | database and schema name; a job of the configured schema's own list is in the configured ones. `:` matches part of the name in any case (`db:team`). `=` and `!=` take the name as stored or as SQL resolves it, a bare name being its uppercase form: `db=team_db` and `db=TEAM_DB` match `TEAM_DB`, `db=ml` and `db='"ml"'` match a lowercase `"ml"`, and `db=ML` doesn't |

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
```

No-result messages name the searched scope, truncated or older matching jobs,
unknown fields (with dotted suggestions), and `a` when a wider search may help.
Long messages scroll. The header gives the match count and query.

| Key | Action |
|-----|--------|
| `Enter` | Open the job preview (or, on a sub-job row, that sub-job's details, on the Logs tab; on a "status unavailable" line, its job's preview); in the search box, show the matches of all that was typed and move to the list, where `Enter` opens the highlighted match |
| `/` / `esc` | Search / clear the search. Keys are taken in the order typed, also when they come faster than the list redraws: what is typed before `Enter` or `esc` is in the search they apply or clear, and the keys after `/`, `Enter` and `esc` go to the search box or the list |
| `s` / `o` | Sort by created, updated, or status (`succeeded` and `completed` sort with `done`, `canceled` with `cancelled`) / reverse the order (so do a click on the sort's name and one on its arrow, in the line above the list). That line says which way it runs: `created ↓` for newest first or `↑` for oldest first (the same for updated; jobs without that time, shown as `—`, come last either way), `status ↓` for running first or `↑` for ended first (history first). The cursor goes to the first job |
| `Home` / `End` | First / last job |
| `→` / `←` | Unfold the highlighted job's sub-jobs / fold them; on a sub-job row, `←` folds its job and moves to it. The job's row stays where it is on screen, and the list scrolls only as far as an unfolded job's sub-jobs need to show. Where there is nothing to unfold or fold, they scroll the columns sideways |
| click `▸` / `▾` | Unfold or fold that job, as `→` / `←` do |
| `e` | Expand or collapse every job's sub-jobs |
| `a` | All schemas / current schema: switch to the account-level list of every database and schema the role can see (the footer's `a All schemas`), or back to the configured one (`a Current schema`) |
| `ctrl+c` | Copy what is highlighted, else every job in the list, each with its whole job id; in the preview, the highlight, else the whole preview |
| `r` | Refresh (the list also refreshes every 60 seconds while it is showing); while it shows the configured schema in place of the account-level list (`Current schema only: DB.SCHEMA (reason)`), `r` asks that list again once a minute has passed since the last ask |
| `?` | Show or close the keys panel |
| `q` | Quit |

Every footer starts with Back (`b`, except on the job list) and Quit (`q`), then
`?`: a narrow terminal cuts the footer's end, and `?` opens a panel that lists
every key that works where you are, including the ones the footer has no room
for. `esc` closes that panel first, before it clears a search or goes back.

#### Job preview

Opening a job shows its id, status, model, type, owner, requested GPUs,
creation and update times, elapsed time, health, and sub-jobs. `unknown` reads
`? status unavailable`. Values stop after three rows; `e` expands every value
and the complete sub-job table, and toggles them short again.

**GPUs** sums each sub-job's `n_gpus`, excluding `log_probability` and
multiplexed sub-jobs that run on another sub-job's GPUs. **Elapsed** includes
queue and placement time; an `unknown` job runs to now. Missing times show `—`,
and a clock up to five minutes ahead reads `just now`.

The health line describes running, waiting, completed, cancelled, terminated,
failed, or unavailable state, followed by current `reason` and
`reason_details`. The sub-job table shows type, status, model, GPUs and key
config; multiplexed rows name the source sub-job. Long cells use `…`, narrow
screens omit Type, and wider tables scroll with `←` / `→` or `Home` / `End`.

Live previews refresh every 60 seconds, on return from details, or on `r`; a
dim line reports reads, slow reads and refresh errors while keeping the last
copy. `Enter` or a click opens the highlighted entire-job or sub-job scope on
Logs. `l` and `g` open Logs and Resources directly.

#### Job details

The line under the title names the job and its scope, and repeats the
service's `reason` and `reason_details` (or the `unknown` sentence) while it
sends them. A sub-job's details name it by its id once (`sub-job
JOB_ID:training:0`). Switch tabs with `l` (Logs) and `g` (Resources), or `t`
for the next tab.

- **Logs** is the live and historical log view described below. Each line is
  tagged with the sub-job being tailed, for example `training:0 │ …`, and its
  wrapped rows line up under its text. Saved logs (`s`) and whole-log copies
  (a copy with nothing highlighted) stay untagged and uncoloured. In the pane,
  the tag has its sub-job type's colour, the level its severity's (a
  warning's or an error's message is tinted to match), tracebacks are red,
  and what the filter matches is marked in amber.
- **Resources** shows telemetry (`download_metrics`) for one sub-job at a
  time. `Space` (or a click on its name) shows the next one; `<` and `>` go
  back and forth. Its two-line pinned header names the sub-job, position,
  window, sample age and scrape interval while the stats and charts scroll.
  - The focused left panel switches between `GPU Resources` and `System
    Resources`, then `ALL` (the mean) and each GPU or node. `↑`, `↓` or a click
    selects an entry. Each unit shows its latest utilization; stale units are
    dimmed, and a view with no rows reads `(none)`. A sub-job with only system
    rows opens there.
  - The header compares requested and reporting GPUs, or shows reporting nodes,
    CPU count and memory per node. Multiplexed sub-jobs point to the sub-job
    whose GPUs report their shared telemetry. A unit over two minutes or three
    sample intervals behind its view's newest sample is stale. When the newest
    sample itself is over three minutes or six intervals old, the header warns
    that no new samples are arriving and the windows end at that sample. It
    also counts malformed or unsupported rows.
  - Numeric `_pct`, `_bytes`, `_watts` and `_c` fields become `%`, `GiB`, `W`
    and `°C` metrics. GPU and System begin with utilization and derived memory
    used; constant node CPU count and memory total stay in the header. Each
    metric has min, time-weighted average, max and latest stats for the window
    and whole job, plus a chart (two per row at 90 cells or wider). Percentage
    axes are 0–100%; other axes use a stable rounded ceiling. `ALL` aggregates
    reporting units by sample interval; one missed node scrape is bridged,
    longer gaps remain gaps. History older than seven hours is folded into
    whole-job bins; `≤` or `≥` marks a late sample that makes the exact extreme
    unknowable.
  - `w` (or the header's window) cycles 10 minutes (the default), 30 minutes,
    1 hour, 6 hours and the whole job.
  - Telemetry downloads when shown, refreshes every 30 seconds for a running
    selected sub-job, and refreshes on `r`. Before it runs, the tab waits
    without polling telemetry; a running sub-job without samples says they
    usually arrive after about a minute. A `log_probability` sub-job says it
    has no GPUs. A failed refresh keeps cached data and says
    `cached · refresh failed: …`. Data is kept in the TUI's per-location job
    cache; the selected sub-job and its newest cached samples
    appear first, and later opens download only new chunks.

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
`$XDG_CACHE_HOME`), overridable with `CORTEX_TRAINING_TUI_CACHE_DIR`.

The pane draws each line as text only: escape sequences in a line (colours,
cursor moves, clipboard or title writes) are removed and other control
characters show as `�`, so nothing in a log acts on your terminal; a line that
rewrites itself with carriage returns (a progress bar) shows its last state.
A leading timestamp shows in local time, as `YYYY-MM-DD HH:MM:SS`, like every
other time in the TUI. A JSON log record, after the timestamp and any Ray
`(Actor pid=...)` prefix, shows as `time LEVEL logger: message`, with the
message's line breaks kept and an error or exception chain on the lines under
it; its other fields (host, ids) are left out. `s` saves the lines as they came.

The header's subtitle starts with the job's status. On the Logs tab it adds
`last line HH:MM:SS`, the newest live line's own timestamp in local time
(`last line received` with the time it arrived, for a line without one), and
any filter, level, or pause.

The line under the log pane is the Logs tab's status line. It appears when the
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
background, with the chunk count on the status line meanwhile. The pane says
which sub-job's console comes first, and, for a sub-job you select before its
console is in, that it comes with the others'. A job that ended more than 10
minutes earlier keeps that console in its per-location job cache, so
reopening it is instant. The
kept consoles of all jobs take at most 1 GiB together: past that, the ones
opened least recently are removed (and downloaded again if opened). A
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

The TUI also writes two files into your home directory: saved logs from the `s`
key (`~/cortex-training-<job8>-<source>.log`, where `<job8>` is the first eight
characters of the job id) and its own error log
(`~/.cortex-training-errors.log`).
`<source>` is the sub-job's type and index, such as `training-0`. `s` saves the
sub-job's saved console, else the lines its tail cached, else the execution log
the pane shows; with nothing to save it says so and writes no file.

Copying works everywhere: drag to highlight, then press `ctrl+c` (or Cmd-C
where the terminal passes it through). Drags stay in their log, list, pane, or
table region and scroll it at an edge; table selection follows characters
across rows and hidden horizontal columns. Small row movements remain clicks,
while a drag never activates the row. Changing the shown tab, source, unit,
window, or rows clears the highlight.

The TUI owns the mouse selection and copies safe text both locally (`pbcopy`,
`wl-copy`, `xclip` or `xsel`) and through OSC 52; a note reports the result.
Use the terminal's selection modifier instead when needed (Option in iTerm2,
Fn in macOS Terminal, or configured Option in VS Code). With the keys panel
open, a whole-screen copy includes its keys.

Keys in the job details view (`/`, `p`, `s`, `[` and `]` work on Logs):

| Key | Action |
|-----|--------|
| `l` `g` | Logs / Resources tab, as in the preview, except `l` on Logs (below) |
| `t` | Next tab |
| `w` | Cycle the window; so does a click on the window in the header (Resources) |
| `Space` | Next sub-job, round to the first after the last, when the job has two or more; the view stays, back to `ALL`. `<` / `>` go back / forward; a click on the sub-job's name in the header goes forward (Resources) |
| `↑` `↓` | Pick GPU Resources or System Resources, `ALL` or one GPU or node in the panel; the charts follow the highlight (Resources) |
| `/` | Filter the current source |
| `l` | Cycle minimum log level (INFO / WARNING / ERROR) (Logs; `L` works too); elsewhere the Logs tab |
| `p` | Pause / resume auto-scroll |
| `s` | Save the current source to `~/cortex-training-<job8>-<source>.log` |
| `ctrl+c` | Copy what is highlighted, else everything the tab holds: the log (its newest lines, up to 1 MiB) on Logs; the panel, header, stats and every chart on Resources (any tab) |
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

`CORTEX_TRAINING_DISABLE_TELEMETRY` (truthy) skips OTLP client metrics on
Snowflake profile and PAT clients. Failures are emitted by default, and so are
successful downloads (`download-log`, `download-metrics`) with their timings.
`CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY` set truthy also emits successful
outcomes for every other essential operation; set falsy, it emits failures only. See the [Python SDK reference](python-sdk.md#client-metrics).

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
