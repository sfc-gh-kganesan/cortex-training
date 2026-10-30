# Python SDK Reference

```python
from cortex_training import CortexTrainingClient, SubJobConfig, JobType
```

`CortexTrainingClient` is the supported entry point. It is a low-level transport
client: every data-plane call returns a `request_id` that you poll, and results
are whatever the backend returns.

Construct it with `from_configs()`, which uses your Snowflake connection
profile:

```python
client = CortexTrainingClient.from_configs()                # pick the profile as below
client = CortexTrainingClient.from_configs(connection="training")
```

`from_configs(connection=None, *, database=None, schema=None, **kwargs)`
picks, first match wins:

- **Profile:** `connection`; then `"profile"` in `config.json`; then the
  Snowflake default profile. `CORTEX_TRAINING_HOST` and `CORTEX_TRAINING_PAT`
  are used only when no Snowflake profile exists, for CI; when a profile is used
  they are ignored and a warning says so.
- **Database:** `database`; then `CORTEX_TRAINING_DATABASE` (logged when used);
  then the profile's database; then `CORTEX_TRAINING_DB`.
- **Schema:** `schema`; then `CORTEX_TRAINING_SCHEMA`; then the profile's
  schema; then `PUBLIC`.

It logs one line naming the target and where each part came from, for example
`profile training (saved in ~/.snowflake/cortex-training/config.json), db TRAIN_DB (from profile training), schema PUBLIC (default)`.
The client shares the profile's live Snowflake session token with REST and
telemetry calls, reconnecting once when the API reports token expiry.

`config.json` holds one key, the name of the Snowflake connection profile to
use:

```json
{"profile": "training"}
```

It lives in the Snowflake folder the profiles come from,
`<Snowflake folder>/cortex-training/config.json`: `$SNOWFLAKE_HOME` if that
folder exists, else `~/.snowflake`, else the platform's config folder. So
`SNOWFLAKE_HOME` moves the profiles and `config.json` together. A missing file
is fine: the Snowflake default profile is used. Any other key is rejected:
credentials and the database belong in the Snowflake profile, and settings such
as `poll_interval` or `verify_ssl` are keyword arguments to `from_configs()`,
which go to the constructor.

These fail with an error that says what's wrong:

- A profile name that doesn't exist (passed in, saved in `config.json`, or the
  Snowflake default from `SNOWFLAKE_DEFAULT_CONNECTION_NAME` or
  `default_connection_name`) names where the name came from and lists the
  profiles that exist.
- A profile that can't sign in names the profile and where it came from.
- `CORTEX_TRAINING_HOST` may include `https://` and a trailing `/`. Anything
  else that isn't an account host, such as `ACCOUNT.snowflakecomputing.com`
  (with an optional `:port`), is rejected.

`from_connection_name()` and `from_pat()` still work but are deprecated: each
emits a `FutureWarning` and will be removed in a future release. Use
`from_configs()` instead.

Tuning knobs on the constructor: `endpoint`, `poll_interval` (0.5s),
`poll_timeout` (1800s), `poll_backoff_multiplier` (1.25), `poll_max_interval`
(6s), `pool_maxsize` (1024), `max_retries` (10), and `request_timeout`
(`(30s connect, 600s read)`). Pass a positive scalar to use the same connect
and read timeout, or a `(connect, read)` pair. `from_configs` also accepts
`telemetry_timeout` (3s) for best-effort client metrics.

The client never creates a database. A request that fails because the
database or schema in its URL can't be found (HTTP 400, code `517602`, message
`Schema DB.SCHEMA is not found or not authorized`) raises `DatabaseAccessError`
(see [Exceptions](#exceptions)), which says how to create it, get access to it,
or use another one. A client from `for_job` raises the plain
`requests.HTTPError` instead. `client.auto_create_database` is deprecated:
it is always False. Setting it to True emits a `FutureWarning`; setting it to False
needs no change.

## Client metrics

Snowflake profile and PAT clients automatically emit one best-effort event
when an essential operation fails. The downloads, `download_stdout_logs`,
`download_metrics`, `download_training_metrics`, `download_execution_logs` and
`fetch_execution_logs`, also emit successful and cancelled outcomes by
default, because they are infrequent and their timing is the point. So does
every operation of a client that `cortex-training tui` builds: its calls are
the waits its user sees, and their timing is the point. While a running job's
Logs tab is shown, its tail sends up to about one event a second while the log
moves, one every 6 seconds once it is quiet, and a few a second while it catches
up. The job list sends one per page of 100 jobs each time it refreshes (every 60
seconds while it is shown), and a live job's preview or details screen one
`get_job` every 60 seconds. A live job's Resources or Training metrics tab
downloads every 30 seconds while shown (after its first download, only the
sub-job it shows; a Training metrics tab with more than one sub-job, all of
them), and so does its Checkpoints tab's list, with one export per
checkpoint whose files it reads (once per opening of the
job, up to 50 at a time) and one more per download. Set
`CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY=1` to emit successful outcomes for
every operation, or `=0` to emit only failures for every operation, the TUI's
included.
Local or mock clients constructed with an explicit `base_url` treat
`emit_metric` as a no-op. Set `CORTEX_TRAINING_DISABLE_TELEMETRY=1` to skip
constructing the emitter.

Every event carries the `client.surface` record attribute, which names the
program that made the call:

| Value | Calls made by |
|---|---|
| `sdk` | Python code using the client (the default) |
| `cli` | A `cortex-training` command |
| `tui` | `cortex-training tui`, including the clients it opens with `for_job` for jobs in other schemas |

Tracked operations:

- Job lifecycle: `create_job`, `wait_for_job`, `get_job`, `list_jobs`,
  `list_models`, `iter_jobs` (one event per page), `list_jobs_page`,
  `cancel_job`, `get_capacity`, `get_capacity_breakdown`, `get_experiment_run`
- Compute: `forward_backward`, `forward`, `generate`, `generate_stream`, `step`
- Checkpoints and logs: `save`, `load`, `list_checkpoints`,
  `export_checkpoint`, `delete_checkpoint`, `tail_logs` (one event per page,
  so per poll of `stream_logs`), `fetch_execution_logs`,
  `download_execution_logs`, `download_stdout_logs`, `download_metrics`,
  `download_training_metrics`
- Multi-sub-job operations: `weight_sync`, `bootstrap_router_replay`,
  `router_replay_discard`, `reset_prefix_cache`
- Async requests: `poll_request`, `get_request_status`, `cancel_request`

The TUI also sends one `download_checkpoint` event for each checkpoint it
downloads, after the `export_checkpoint` call that gets the files' links.

Each emitted event body contains `success`, `duration_ms`, `request_count`,
`attempt_count`, and `retry_count`. Failures also include a bounded
`error_message` with common credential patterns redacted. A cancelled
operation has `cancelled: true`: `DownloadCancelled`, or Ctrl-C
(`KeyboardInterrupt`) during any tracked operation. Cancellations are sent
under the same rule as successes, so by default only for the downloads and
the TUI's operations. A `tail_logs` failure because the job has ended adds
`job_ended: true`; once `stream_logs` has read from the job, it ends on that
failure without raising.
When the client looked the job up to explain a `tail_logs` failure, the event
also has the `job_status` it read, such as `placing` for a job whose logs are
not there yet; a call that succeeded on a retry has none.

Download events add these body fields:

| Field | Meaning |
|---|---|
| `artifact` | `stdout`, `metrics`, `training-metrics`, `execution_logs`, or `checkpoint` |
| `list_ms` / `fetch_ms` | Time listing the run's artifacts, and fetching and writing them |
| `preview_ms` / `first_commit_ms` | When the caller could first show something, from the call's start: `on_preview` returned with the newest chunks, and the first sub-job's file was whole and `on_commit`, if given, returned. Each is sent once it happens (`download_stdout_logs` / `download_metrics` / `download_training_metrics`; only the first two take a preview). On a first download (`resume_mode` `fresh`) these are the TUI's first page or graph: the preview for Logs and Resources, the first file for Training metrics. When the TUI already holds a copy it draws that first, and these then time the newest data |
| `sub_job_count` | Sub-jobs written (`download_stdout_logs` / `download_metrics` / `download_training_metrics`) |
| `chunk_count` / `chunks_skipped` | Chunks this call appended or skipped; skipped ones are also counted in `chunk_count` (`download_stdout_logs` / `download_metrics` / `download_training_metrics`) |
| `bytes_written` | Bytes this call added to the output files |
| `max_workers` | Concurrent GETs used (`download_stdout_logs` / `download_metrics` / `download_training_metrics`) |
| `resume` / `resume_mode` | Whether `resume` was passed, and what happened: `fresh` (no earlier file), `rebuild` (replaced an earlier file), `append`, `partial` (continued an interrupted download), `mixed` across sub-jobs, or `none` (`download_stdout_logs` / `download_metrics` / `download_training_metrics`) |
| `artifact_retry_count` / `transfer_timeout_count` | Retried `LIST`/`GET` attempts, and attempts abandoned after `transfer_timeout_seconds` |
| `cancelled` | The call was stopped by its caller; sent on every download and execution-log event |
| `file_count` / `bytes_read` | Files read and decoded, and their size (`fetch_execution_logs`) |
| `file_count` / `files_skipped` | Files the download covers, and those a `resume=True` call kept because they are unchanged since an earlier download saved them (`download_execution_logs`) |
| `file_count` / `files_written` / `files_skipped` / `total_bytes` | Files the download covers (the export's files that come with a link), those this call downloaded, those already whole on disk and kept, and their size when the export gives every file's (`download_checkpoint`) |

On failure the fields describe the work done before it. A
`download_execution_logs` event has `artifact`, `list_ms` / `fetch_ms`,
`bytes_written`, `artifact_retry_count` / `transfer_timeout_count`,
`cancelled`, and its `file_count` / `files_skipped` row. A `download_checkpoint`
event has `artifact`, `bytes_written`, `cancelled`, and the `file_count` /
`files_written` / `files_skipped` / `total_bytes` row; a file that fails after
the download was stopped counts as stopped. It fetches its files from the export's
links rather than through the REST API, so its `request_count` and
`attempt_count` are 0, and its `error_message` names only the error's type and
HTTP status or OS error: the error's own text can quote a link. Queryable attributes
include `client.surface`, and available `job_id`, sub-job identifiers,
`request_id`, `checkpoint_id`, `error.type`, HTTP status, and server error code.
Two attributes carry Snowflake request IDs:

| Attribute | Meaning | When set |
|---|---|---|
| `error.snowflake_request_id` | Snowflake request ID of the failing response | Failures whose response had the header |
| `snowflake.request_ids` | Recent Snowflake request IDs of the operation (up to 10, including retried attempts), newest last | Whenever any were seen |

`snowflake.request_id` is a deprecated alias of `error.snowflake_request_id`,
emitted for one release. Records use OTLP resource attributes
`service.name = cortex-training` and
`snowflake.account_host = <normalized connection hostname>`.

Applications can emit additional events through the same helper. They carry
`client.surface` too, unless `attributes` gives one:

```python
client.emit_metric(
    "generate",
    {"duration_ms": 42, "success": True},
    attributes={"job_id": job_id},
)
```

Keep caller-supplied custom attributes low-cardinality and never include
prompts, PATs, or stage credentials. Long-lived applications may call
`client.close()` (or use the client as a context manager) to flush briefly and
release telemetry resources. Telemetry authentication, discovery, queue, or
export failures are ignored, so metrics never change the outcome of a client
operation.

## Local training metrics

**The client writes files by default.** When `poll_request` receives a
`/fwd-bwd` or `/step` result, it appends that result's row, in the stage's
`training.jsonl` format
([REST API section 12.5](rest-api.md#125-training-metrics-download)), to a
local file, so the TUI's Training metrics tab shows the step without waiting
for the stage's batched upload. This happens in every process that uses the
client, including notebooks, CI and containers that never open the TUI.

- **Where:** `<cache>/local-training/<job_id>/<sub_job_id>/training.jsonl`,
  where `<cache>` is the TUI cache: `CORTEX_TRAINING_TUI_CACHE_DIR`, else
  `$XDG_CACHE_HOME/cortex-training`, else `~/.cache/cortex-training`.
- **How much:** a file stops growing at 64 MiB. The first write of each client
  prunes the directory: jobs not written to for 30 days, then the least
  recently written jobs until the rest fit in 1 GiB. The TUI never removes
  these rows; it counts them, up to that 1 GiB, in its cache's 5 GiB, so the
  two are one budget rather than two that add up. The stage keeps every job
  whole, so a capped or pruned file only loses the head start, except for a
  job that does not save its training metrics to the stage, whose rows this
  file holds alone.
- **Safety:** writes are best-effort. A read-only or full disk never fails the
  call, and the result returned is unchanged.
- **Turning it off:** set `CORTEX_TRAINING_DISABLE_LOCAL_TRAINING_METRICS=1`.

## Job lifecycle

| Method | Returns | Notes |
|---|---|---|
| `create_job(sub_jobs, job_id=None, experiment_name=None, hardware=None)` | `job_id` | Validates each `SubJobConfig` client-side first. Only `training` and `sampling` sub-jobs can be submitted: a job takes zero or one `training` sub-job and any number of `sampling` sub-jobs. `hardware` is `H200`, `B200`, or `B300` (the `Hardware` enum or its string); omitted means `H200` |
| `create_job_from_body(body)` | response dict | For callers that already hold the REST JSON. Enforces the same one-training-sub-job and no-`log_probability` rules before sending |
| `get_job(job_id)` | job dict | Includes `sub_jobs` with their configs |
| `list_jobs(status=None)` | list of jobs | Returns the inner list, not the envelope |
| `list_models()` | list of models | Active models for the account, from `GET /api/v2/cortex-training/models`; returns the inner list. The configured database and schema are not part of the request |
| `iter_jobs(*, database=None, schema=None, status=None, submitted_by=None, created_after=None, created_before=None, page_size=100)` | iterator of jobs | Jobs across the account that the caller can see, newest first, following page tokens; each also carries `database_name` and `schema_name`. The server applies every filter. Arguments are checked on the call: `page_size` is 1 to 1000, and `schema` needs `database` with it. A 404 means account-level listing is not enabled for the account. See [REST API section 5.3](rest-api.md#across-the-account---get-apiv2endpoint) |
| `list_jobs_page(*, database=None, schema=None, status=None, submitted_by=None, created_after=None, created_before=None, page_size=100, page_token=None, max_retries=None, request_timeout=None)` | `{"jobs": [...], "next_page_token": str \| None}` | One page of what `iter_jobs` walks, with the same filters read the same way. Pass `next_page_token` back as `page_token`, with the same filters, for the next page; `None` means no more. A page can be short or empty and still have a token. `max_retries` (retries after the first attempt) and `request_timeout` (seconds, or `(connect, read)`) bound this page; omitted, the client's own apply. Raises `AccountJobListUnavailable` (`reason`: `not_deployed`, `disabled`, `forbidden`, `rest_disabled`) for 404, 403 and 405 without retrying, and `InvalidPageToken` for 400 `Invalid page_token` (start again from the first page); both subclass `requests.HTTPError`. Bad arguments, or a page body that is not a job list, raise `ValueError`. See [REST API section 5.3](rest-api.md#across-the-account---get-apiv2endpoint) |
| `for_job(job)` | `CortexTrainingClient` | Bound to a job from `iter_jobs` or `list_jobs_page` for per-job calls. Borrows this client's session, auth, and telemetry, its `client.surface` included, so keep this one open; closing the returned client is a no-op. Never creates the job's database |
| `wait_for_job(job_id)` | job dict | Polls until `running`; raises once the job has ended (`failed`, `cancelled`, `terminated`) or on timeout |
| `cancel_job(job_id)` | `None` | Idempotent while cancelling/cancelled |
| `get_capacity(hardware=None)` | capacity dict | `has_reservation`, `max_total_gpus`, `reserved_gpus`, `in_use_gpus`, `pending_gpus`, `available_gpus`, scoped to `hardware` (omitted means the server default, H200). The CLI `capacity` command queries every type unless `--hardware` is set. `max_total_gpus` is the canonical ceiling (`-1` uncapped); `reserved_gpus` is deprecated |
| `get_capacity_breakdown(hardware=None, *, dimensions=("job_type", "submitted_by"))` | capacity dict plus `breakdown` | `get_capacity` split by sub-job type and/or submitter; each split has a `status` and `groups`. Unknown dimensions raise `ValueError`. See [rest-api.md section 5.4](rest-api.md#breakdown---breakdown) |

`JobType.LOG_PROBABILITY` stays a schema type: a `SubJobConfig` carrying it
still constructs, validates, and serializes to `"log_probability"`. Submission
is what is blocked. `create_job()`, `create_job_from_body()`,
`SubJobConfig.sampling_job()`, and `cortex-training submit` (including
`--dry-run`) all raise before sending, with the message
`log_probability sub-jobs are not currently supported` prefixed by the offending
location — for example
`sub_job_configs[1].job_type: log_probability sub-jobs are not currently
supported`. The short alias `log_prob` is rejected the same way.

## Training and sampling

| Method | Returns | Notes |
|---|---|---|
| `forward_backward(job_id, data)` | `request_id` | `data` is a DSSST1 frame; always chunk-wrapped |
| `step(job_id, learning_rate=None)` | `request_id` | Omitting the rate uses the job's optimizer setting |
| `generate(job_id, prompts, sampling_params=None, routing_key=None, strict=None, sub_job_id=None)` | `request_id` | Pre-tokenized prompts are length-checked client-side; `sub_job_id` picks the sampling sub-job when a job has several |
| `generate_stream(...)` | response dict | Same DSSST1 body encoding as `generate`, sent in one POST; read progress with `get_request_status` |
| `weight_sync(job_id, source_sub_job_id, target_sub_job_ids, weight_format=None)` | `request_id` | `weight_format="lora"` syncs adapters only |
| `forward(job_id, payload, ...)` | response dict | Byte frames that exceed the `/operation` JSON ceiling are request-chunked; see [rest-api.md section 7.1](rest-api.md#71-forward) |
| `poll_request(job_id, request_id)` | result dict | Handles backoff, DSSST1 decoding, chunked results, and envelope `metrics` merge (envelope wins on key collision) |
| `get_request_status(job_id, request_id, max_events=None, cursor=None)` | status dict | |
| `cancel_request(job_id, request_id, ...)` | response dict | |

A training sub-job with router replay picks how forward/backward treats rows
whose routing no sampling worker holds with `router_replay["mode"]` in the
`extra_training` argument of `SubJobConfig.training_job(...)`, together with
`"enabled": true`:

- `"strict"` (default) fails the request.
- `"best_effort"` routes those rows with the trainer's own MoE gate and adds
  `router_replay/rows_*` and `router_replay/tokens_*` keys to the polled
  result's `metrics`. With whole-block activation checkpointing it also needs
  `ac_config.router_replay_recompute` left at its default `true`.

The server rejects `"best_effort"` without `"enabled": true`, and any mode
other than `"strict"` on a sampling sub-job. See
[rest-api.md section 8.2](rest-api.md#82-trainingconfig) and
[section 6.1](rest-api.md#61-forwardbackward---post-job_idforward-backward).

The SDK does not size `router_replay.max_cache_bytes`: set it on each sub-job's
`router_replay` object in `extra_training` or `extra_sampling`. The
repository's RL example fills in 5 GiB (training) and 20 GiB (sampling) for MoE
models, and describes an optional planner that sizes a cap from the model's
geometry; see
[Router-replay capacity planning](../../recipes/rl_quickstart/README.md#router-replay-capacity-planning).

## Checkpoints

`save(job_id, checkpoint_id=None, checkpoint_type=None)` →  `request_id`
(`checkpoint_type` is `"resumable"` or `"weights-only"`),
`load(job_id, checkpoint_id, source_job_id=None, target_sub_job_id=None)` →
`request_id`, `list_checkpoints(job_id)`, `export_checkpoint(job_id, checkpoint_id)`,
`delete_checkpoint(job_id, checkpoint_id)`.

## Logs

`tail_logs(job_id, ...)` returns one cursor page of `{"_raw": "<time> <line>"}`
entries (see [REST API section 7.8](rest-api.md#78-tail-logs));
`stream_logs(job_id, follow=True, ...)` yields entries and keeps polling, and
with `follow=False` stops at the first empty page. On a job that has ended,
`tail_logs` raises at once and points at `download_stdout_logs`.
`download_execution_logs(job_id, output_dir, *, resume=False,
sub_job_ids=None, progress=None, transfer_timeout_seconds=300)` saves every
execution log file as `<output_dir>/<sub_job_id>/<filename>` (mode `0600`)
without holding it in memory; with `resume=True` it skips files unchanged since
the last download. `fetch_execution_logs(job_id, *, sub_job_id=None,
filename=None)` returns the same files as
`{sub_job_id, filename, artifact_uri, content}` dicts, in memory, or only
those of one sub-job or with one name (the rest are listed but not fetched).
See
[REST API section 12.2](rest-api.md#122-full-execution-log-download).
`download_stdout_logs(job_id, output_dir, *, resume=False, max_workers=None,
progress=None, on_chunk=None, sub_job_ids=None, strict=False,
stale_after_seconds=600, transfer_timeout_seconds=300)`
reconstructs persisted console chunks as
`<output_dir>/<sub_job_id>/stdout.log`.
`download_metrics(...)`, with the same options plus `priority=()` and
`on_commit=None`, reconstructs GPU metric chunks
as `<output_dir>/<sub_job_id>/gpu.jsonl`.
`download_training_metrics(...)` takes the same options and reconstructs a
training sub-job's training-metric chunks as
`<output_dir>/<sub_job_id>/training.jsonl`; the two are separate streams that
can share an output directory (see
[REST API section 12.5](rest-api.md#125-training-metrics-download)). Chunks are fetched `max_workers` at a
time (default 16) on one Snowflake session and written in order. `resume=True`
appends chunks that are not already recorded in the manifest beside that file
and continues an interrupted download. Progress is reported only through
`progress` and `on_chunk`; retries, skipped chunks, and stalled-upload notices
are logged as warnings on the `cortex_training` logger. `progress` receives status updates without a trailing
newline, then `"\n"` when that sub-job finishes or fails; a status that needs no
updating arrives as one message ending in a newline.
`on_chunk(sub_job_id, done, total, bytes)` reports counts instead, with
`done == total` only after that sub-job's file is committed. For metrics,
`on_commit(entry)` then gets a copy of that sub-job's result entry, so a reader
can open one file while the others download, and sub-jobs named in `priority`
are downloaded first, in that order (the returned list stays in id order).
These callbacks run on the calling thread; an error raised by any of them stops
the download and keeps the chunks already written. Raise `cortex_training.DownloadCancelled` to stop
without the call counting as a failure. `sub_job_ids` limits the download (and,
naming one sub-job, its listing too); an entry with no chunks raises
`cortex_training.NoChunksError`, a `ValueError`. A damaged chunk is fetched
once more and then skipped and listed in the entry's
`skipped_chunks` and `warnings`; `strict=True` raises instead. While the job is
running, a sub-job whose newest chunk is older than `stale_after_seconds` (or,
for metrics, one with stdout but no metrics) gets a warning that uploads may be
stalled. See
[GPU metrics download](rest-api.md#124-gpu-metrics-download) for the JSONL row
schema.
`get_experiment_run(job_id)` resolves the experiment/run names.

## Building payloads

```python
from cortex_training import serialize_forward_backward_args, wire

payload = serialize_forward_backward_args(
    args=(), kwargs={"input_ids": input_ids, "labels": labels}
)
```

Use `wire.dumps(obj, metadata=...)` directly for batches that carry extra
backend keys such as `context` or `processing`.
`build_forward_backward_payload(spec)` builds a frame from readable JSON and is
what the CLI's `fwd-bwd` command uses.

`cortex_training.client.peft.normalize_lora_peft_config(cfg)` validates and normalizes
the LoRA subset supported by weight sync. It requires `peft_type` to be exactly
`"Lora"`, so a config copied from a Hugging Face `adapter_config.json` (which
writes `"LORA"`) must be adjusted.

`cortex_training.client.optimizer.normalize_optimizer_config(cfg)` does the same for
the training optimizer block, returning `(canonical_optimizer,
gradient_clipping)`. It resolves the `type`/`name` and `beta1`/`beta2` aliases,
flattens a DeepSpeed `params` sub-object, hoists a nested `gradient_clipping`
out of the block, and rejects scheduler or unknown keys. `TrainingConfig` runs
it on construction, so calling it directly is only useful for checking a config
before building the typed object; see
[REST API section 8.2](rest-api.md#82-trainingconfig) for the rules.

## Exceptions

`ChunkGroupError` and its subclasses `ChunkGroupRestartError` and
`ChunkGroupConflictError` (all subclasses of `requests.exceptions.HTTPError`)
carry a `.detail` dict for chunk-group failures. `AccountJobListUnavailable` (with a
`.reason`) and `InvalidPageToken`, also `HTTPError` subclasses, come from
`list_jobs_page`. Polling raises `RuntimeError`
on a failed/cancelled request and `TimeoutError` at the deadline.

`DatabaseAccessError` (a subclass of both `requests.exceptions.HTTPError` and
`RuntimeError`) means the configured database or schema can't be found by the
current role: it doesn't exist, or the role can't access it. It carries
`.response` (the HTTP 400), `.kind` (`"schema"`), `.name` (`DATABASE.SCHEMA`),
`.role` (the session's role, or None when it can't be determined, as for PAT
clients), and `.statement`, the SQL that would set it up: `CREATE DATABASE IF NOT
EXISTS ...` and `CREATE SCHEMA IF NOT EXISTS ...`, separated by `; `.

## Details not covered here

For exact wire shapes, request framing, and schemas, see the
[REST API reference](rest-api.md). For commands and configuration, see the
[CLI reference](cli.md).
