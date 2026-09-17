# Cortex Training CLI and Client Reference

The `cortex-training` command and `cortex_training` Python package provide the
supported command-line and SDK interfaces for Cortex Training.

`ct` is installed as an alias for `cortex-training` and supports the same
commands and flags. You can replace `cortex-training` with `ct` in any example
below, such as `ct login config.json` or `ct --job JOB_ID step`.

## Installation

Requires Python 3.10+. Installing the package gives you the `cortex-training`
CLI, the `cortex-training tui` log viewer, and the `cortex_training` Python SDK.

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
cortex-training get JOB_ID
cortex-training wait JOB_ID                   # Wait until running, not finished
cortex-training cancel JOB_ID
cortex-training checkpoints JOB_ID
cortex-training capacity                      # All supported GPU types
cortex-training capacity --hardware B200
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

### [Logs And Metrics](#log-tui)

```bash
cortex-training tui                          # Open job picker
cortex-training tui JOB_ID                   # Open job logs
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
| `capacity` | Query H200, B200, and B300 | Select one with `--hardware` |
| `download-log`, `download-metrics` | Write under the current directory | Set `--output-dir` |

## Detailed Reference

- [Connection config](#connection-config), [login](#login), and [environment variables](#environment-variables)
- [Submit](#submit-a-job), [manage jobs](#manage-existing-jobs), and [GPU capacity](#show-current-gpu-capacity)
- [Forward-backward and optimizer steps](#run-a-forward-backward-smoke-test)
- [Load checkpoints](#load-a-checkpoint-into-a-running-job) and [initialize sampling](#start-sampling-from-a-training-checkpoint)
- [Generate](#run-a-generate-smoke-test) and [sync weights](#sync-training-weights)
- [Download logs](#download-execution-logs), [stdout](#download-persisted-stdout), and [metrics](#download-gpu-metrics)
- [Log TUI](#log-tui), [JSON output and help](#json-output-and-help), and [troubleshooting](#troubleshooting)

`cortex-training` submits and manages Cortex Training jobs through the Cortex
Training REST endpoint.
The normal workflow is:

1. Create a connection config JSON.
2. Run `cortex-training login config.json` once.
3. Use `cortex-training list`, `submit`, `get`, `cancel`, `wait`, and
   `capacity` without passing connection flags every time.

### Connection Config

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

### Login

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
cortex-training get JOB_ID
cortex-training checkpoints JOB_ID
cortex-training cancel JOB_ID
cortex-training wait JOB_ID
```

`get` fetches current job details; `checkpoints` lists saved checkpoints.
`wait` waits for the job to reach **running**, not for training to finish.
See [Manage Jobs](../guides/operations/manage-jobs.md) for the operational workflow.

### Show Current GPU Capacity

Print the caller account's reserved GPU capacity and current usage, separated
by hardware:

```bash
cortex-training capacity
cortex-training capacity --hardware B200
```

The default command queries `H200`, `B200`, and `B300` independently and
prints a `capacity_by_hardware` map. Each entry includes `has_reservation`,
`max_total_gpus`, `reserved_gpus`, `in_use_gpus`, `pending_gpus`, and
`available_gpus`.

`--hardware` keeps the single-capacity response shape for one GPU type.

`max_total_gpus` is the canonical ceiling and supersedes the deprecated
`reserved_gpus`. `in_use_gpus` counts only GPUs the account holds; queued work
is reported separately in `pending_gpus`. See
[REST API reference section 5.4](rest-api.md#54-capacity---get-capacity) and
[GPU hardware](../concepts/hardware.md).

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
validates and prints the request body without sending it.

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

This is the runtime load path. Create-time resume still uses
[`source_checkpoint_info`](rest-api.md#65-create-time-checkpoint-initialization)
in the submitted sub-job JSON.

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
all prompts or a list of objects/nulls aligned with `prompts`. The CLI submits
`generate` and polls the request by default. Set `"poll": false` to print only
the submitted `request_id`.

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
```

Files are written as `<output_dir>/<sub_job_id>/<filename>` so siblings
do not collide. When `--output-dir` is omitted, the current working
directory is used instead. The CLI also prints a JSON summary listing
each `saved_path`.

Programmatic access is `CortexTrainingClient.fetch_execution_logs(job_id)`,
which returns a list of `{sub_job_id, filename, artifact_uri, content}` dicts.

### Download Persisted Stdout

Reconstruct each sub-job's persisted stdout/stderr chunks into
`<output_dir>/<sub_job_id>/stdout.log`:

```bash
cortex-training download-log JOB_ID --log-type stdout --output-dir /path/to/logs
```

The current working directory is used when `--output-dir` is omitted.

### Download GPU Metrics

Reconstruct each sub-job's GPU metric chunks into
`<output_dir>/<sub_job_id>/gpu.jsonl`:

```bash
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics
```

The command prints the saved path, chunk count, and first/last logical artifact
URIs for each reconstructed file.

### Log TUI

`cortex-training tui` is a read-only terminal UI for tailing a running job's logs
live. It reuses the same connection handling as `cortex-training` — login state,
`--config` /
`CORTEX_TRAINING_CONFIG`, the `CORTEX_TRAINING_*` / `SNOWFLAKE_*` env vars, or explicit
flags. So once you've run `cortex-training login config.json` you can just launch it:

```bash
cortex-training tui                 # opens a job picker
cortex-training tui JOB_ID          # opens that job's logs directly
```

Without login state, pass connection details the same way as the CLI:

```bash
cortex-training tui JOB_ID --config config.json
cortex-training tui JOB_ID --host ACCOUNT.snowflakecomputing.com --pat YOUR_PAT \
  --database CORTEX_TRAINING_DB --schema PUBLIC --endpoint cortex-training
cortex-training tui JOB_ID --base-url http://localhost:8084   # local/mock
```

Keep the PAT out of your shell history by exporting it instead (omit `JOB_ID`
to open the job picker):

```bash
export CORTEX_TRAINING_PAT='YOUR_PROGRAMMATIC_ACCESS_TOKEN'
cortex-training tui \
  --host ACCOUNT.snowflakecomputing.com \
  --database CORTEX_TRAINING_DB --schema PUBLIC
```

Pass `--sub-job-id JOB_ID:training:0` to open one sub-job's log directly instead
of the source list.

The left panel lists the job's sub-jobs; select one to tail its logs (the
zone-manager pod is the Ray head, so a sub-job's worker output is included).
Logs are cached locally so reopening a job replays instantly without
re-fetching from the server — under `~/.cache/cortex-training/` (or
`$XDG_CACHE_HOME`), overridable with `CORTEX_TRAINING_TUI_CACHE_DIR`.

The TUI also writes two files into your home directory: saved logs from the `s`
key (`~/cortex-training-<job8>-<source>.log`, where `<job8>` is the first eight
characters of the job id) and its own error log
(`~/.cortex-training-errors.log`).

Keys in the log view:

| Key | Action |
|-----|--------|
| `/` | Filter the current source |
| `L` | Cycle minimum log level (INFO / WARNING / ERROR) |
| `p` | Pause / resume auto-scroll |
| `s` | Save the current source to `~/cortex-training-<job8>-<source>.log` |
| `y` | Copy the whole log to the clipboard |
| `c` | Copy the current selection |
| `r` | Refresh the sub-job list |
| `[` / `]` | Narrow / widen the sources panel |
| `b` / `esc` | Back |
| `q` | Quit |

In the job picker, `/` filters by id/status/type and `r` refreshes. The
`--poll-interval` flag (default `1.0s`) is the minimum interval between log
polls per source, biasing toward server reliability over freshness.

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
PAT-authenticated clients. `CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY`
(truthy) also emits successful outcomes for essential operations; failures
are emitted by default. See the [Python SDK reference](python-sdk.md#client-metrics).

### Troubleshooting

If you see `provide --base-url for local/mock use, or both --host and --pat`,
the CLI found a `host` but no PAT. Add `"pat": "..."` to `config.json` or set
`CORTEX_TRAINING_PAT`.

If you see `Invalid URL ... No scheme supplied`, the config is using a bare
Snowflake hostname as `base_url`. Use `host` for Snowflake PAT auth, or use a
full local/mock URL such as `http://localhost:8084` for `base_url`.

For server errors, the CLI prints any Snowflake request id and response body
returned by the service. Include those details when debugging a `500`.
