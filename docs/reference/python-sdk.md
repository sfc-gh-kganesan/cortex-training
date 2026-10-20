# Python SDK Reference

```python
from cortex_training import CortexTrainingClient, SubJobConfig, JobType
```

`CortexTrainingClient` is the supported entry point. It is a low-level transport
client: every data-plane call returns a `request_id` that you poll, and results
are whatever the backend returns.

Construct it with a named Snowflake connection profile:

```python
client = CortexTrainingClient.from_connection_name("training")
```

Pass no name to use the Connector-configured default profile. The profile's
database and schema are used unless explicitly overridden. The client shares
the profile's live Snowflake session token with REST and telemetry calls,
reconnecting once when the API reports token expiry.

Direct Programmatic Access Token construction remains supported:

```python
client = CortexTrainingClient.from_pat(
    host="ACCOUNT.snowflakecomputing.com",
    pat=PAT,
    database="CORTEX_TRAINING_DB",
    schema="PUBLIC",
)
```

Tuning knobs on the constructor: `endpoint`, `poll_interval` (0.5s),
`poll_timeout` (1800s), `poll_backoff_multiplier` (1.25), `poll_max_interval`
(6s), `pool_maxsize` (1024), `max_retries` (10), and `request_timeout`
(`(30s connect, 600s read)`). Pass a positive scalar to use the same connect
and read timeout, or a `(connect, read)` pair. `from_connection_name` and
`from_pat` also accept `telemetry_timeout` (3s) for best-effort client metrics.

By default a request answered "database or schema not found" makes the client
run `CREATE DATABASE` (and `CREATE SCHEMA`) and retry. Set
`client.auto_create_database = False` for a client that only reads: the error
is then raised as it is. A client from `for_job` starts with it off.

## Client metrics

Snowflake profile and PAT clients automatically emit one best-effort event
when an essential operation fails. `download_stdout_logs`, `download_metrics`,
and `fetch_execution_logs` also emit successful and cancelled outcomes by
default, because they are infrequent and their timing is the point. Set
`CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY=1` to emit successful outcomes for
every operation, or `=0` to emit only failures for every operation.
Local or mock clients constructed with an explicit `base_url` treat
`emit_metric` as a no-op. Set `CORTEX_TRAINING_DISABLE_TELEMETRY=1` to skip
constructing the emitter.

Tracked operations:

- Job lifecycle: `create_job`, `wait_for_job`, `get_job`, `list_jobs`,
  `iter_jobs` (one event per page), `list_jobs_page`, `cancel_job`, `get_capacity`,
  `get_capacity_breakdown`, `get_experiment_run`
- Compute: `forward_backward`, `forward`, `generate`, `generate_stream`, `step`
- Checkpoints and logs: `save`, `load`, `list_checkpoints`,
  `export_checkpoint`, `delete_checkpoint`, `fetch_execution_logs`,
  `download_execution_logs`, `download_stdout_logs`, `download_metrics`
- Multi-sub-job operations: `weight_sync`, `bootstrap_router_replay`,
  `router_replay_discard`, `reset_prefix_cache`
- Async requests: `poll_request`, `get_request_status`, `cancel_request`

Each emitted event body contains `success`, `duration_ms`, `request_count`,
`attempt_count`, and `retry_count`. Failures also include a bounded
`error_message` with common credential patterns redacted. A cancelled
operation has `cancelled: true`: `DownloadCancelled`, or Ctrl-C
(`KeyboardInterrupt`) during any tracked operation. Cancellations are sent
under the same rule as successes, so by default only for the three download
operations.

Download events add these body fields:

| Field | Meaning |
|---|---|
| `artifact` | `stdout`, `metrics`, or `execution_logs` |
| `list_ms` / `fetch_ms` | Time listing the run's artifacts, and fetching and writing them |
| `sub_job_count` | Sub-jobs written (`download_stdout_logs` / `download_metrics`) |
| `chunk_count` / `chunks_skipped` | Chunks this call appended or skipped; skipped ones are also counted in `chunk_count` |
| `bytes_written` | Bytes this call added to the output files |
| `max_workers` | Concurrent GETs used |
| `resume` / `resume_mode` | Whether `resume` was passed, and what happened: `fresh` (no earlier file), `rebuild` (replaced an earlier file), `append`, `partial` (continued an interrupted download), `mixed` across sub-jobs, or `none` |
| `artifact_retry_count` / `transfer_timeout_count` | Retried `LIST`/`GET` attempts, and attempts abandoned after `transfer_timeout_seconds` |
| `cancelled` | The call was stopped by its caller; sent on every download and execution-log event |
| `file_count` / `bytes_read` | Files read and decoded, and their size (`fetch_execution_logs`) |

On failure the fields describe the work done before it. Queryable attributes
include available `job_id`, sub-job identifiers, `request_id`,
`checkpoint_id`, `error.type`, HTTP status, and server error code.
Two attributes carry Snowflake request IDs:

| Attribute | Meaning | When set |
|---|---|---|
| `error.snowflake_request_id` | Snowflake request ID of the failing response | Failures whose response had the header |
| `snowflake.request_ids` | Recent Snowflake request IDs of the operation (up to 10, including retried attempts), newest last | Whenever any were seen |

`snowflake.request_id` is a deprecated alias of `error.snowflake_request_id`,
emitted for one release. Records use OTLP resource attributes
`service.name = cortex-training` and
`snowflake.account_host = <normalized connection hostname>`.

Applications can emit additional events through the same helper:

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

## Job lifecycle

| Method | Returns | Notes |
|---|---|---|
| `create_job(sub_jobs, job_id=None, experiment_name=None, hardware=None)` | `job_id` | Validates each `SubJobConfig` client-side first. Only `training` and `sampling` sub-jobs can be submitted: a job takes zero or one `training` sub-job and any number of `sampling` sub-jobs. `hardware` is `H200`, `B200`, or `B300` (the `Hardware` enum or its string); omitted means `H200` |
| `create_job_from_body(body)` | response dict | For callers that already hold the REST JSON. Enforces the same one-training-sub-job and no-`log_probability` rules before sending |
| `get_job(job_id)` | job dict | Includes `sub_jobs` with their configs |
| `list_jobs(status=None)` | list of jobs | Returns the inner list, not the envelope |
| `iter_jobs(*, database=None, schema=None, status=None, submitted_by=None, created_after=None, created_before=None, page_size=100)` | iterator of jobs | Jobs across the account that the caller can see, newest first, following page tokens; each also carries `database_name` and `schema_name`. The server applies every filter. Arguments are checked on the call: `page_size` is 1 to 1000, and `schema` needs `database` with it. A 404 means account-level listing is not enabled for the account. See [REST API section 5.3](rest-api.md#across-the-account---get-apiv2endpoint) |
| `list_jobs_page(*, database=None, schema=None, status=None, submitted_by=None, created_after=None, created_before=None, page_size=100, page_token=None, max_retries=None, request_timeout=None)` | `{"jobs": [...], "next_page_token": str \| None}` | One page of what `iter_jobs` walks, with the same filters read the same way. Pass `next_page_token` back as `page_token`, with the same filters, for the next page; `None` means no more. A page can be short or empty and still have a token. `max_retries` (retries after the first attempt) and `request_timeout` (seconds, or `(connect, read)`) bound this page; omitted, the client's own apply. Raises `AccountJobListUnavailable` (`reason`: `not_deployed`, `disabled`, `forbidden`, `rest_disabled`) for 404, 403 and 405 without retrying, and `InvalidPageToken` for 400 `Invalid page_token` (start again from the first page); both subclass `requests.HTTPError`. Bad arguments, or a page body that is not a job list, raise `ValueError`. See [REST API section 5.3](rest-api.md#across-the-account---get-apiv2endpoint) |
| `for_job(job)` | `CortexTrainingClient` | Bound to a job from `iter_jobs` or `list_jobs_page` for per-job calls. Borrows this client's session, auth, and telemetry, so keep this one open; closing the returned client is a no-op. Never creates the job's database |
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
| `generate(job_id, prompts, sampling_params=None, routing_key=None, strict=None)` | `request_id` | Pre-tokenized prompts are length-checked client-side |
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

## Router-replay capacity planning

`cortex_training.router_replay_planner` derives the per-worker
`router_replay.max_cache_bytes` caps before you create a job. It only
calculates. It does not call the service, edit a job config, or reserve GPU
memory. Put `CachePlan.max_cache_bytes` on the sampling and training
`router_replay` objects yourself. When bootstrapping with an explicit cap, pass
the training result to `bootstrap_router_replay(..., max_cache_bytes=...)`.

One full-sequence capture occupies:

```text
entry_bytes = seq_len * num_moe_layers * topk * dtype_bytes
```

`moe_geometry_from_config` reads `num_hidden_layers` and `num_experts_per_tok`
from the config root or `text_config`, and rejects conflicting values. Every
recognized expert-count field (`num_local_experts`, `n_routed_experts`,
`num_experts`, `moe_num_experts`) must agree, be positive, and be at least
`topk`. A count of 256 or fewer selects `dtype_bytes=1` because those expert
ids fit in `uint8`. A count from 257 through 65536 selects `dtype_bytes=2`
(`uint16`). A larger count is rejected. `plan_sampling_cache_bytes` and
`plan_training_cache_bytes` use that width unless you pass `dtype_bytes`.

The sampling cap is per sampling replica, and the training cap is per
data-parallel shard:

```text
backlog = max_inflight_rollouts
        + ceil(batch_size * over_generation_multiplier)
        + batch_size * max(async_depth, 1)
per_replica_rollouts = ceil(backlog / replicas * routing_skew)
sampling_cap = ceil(per_replica_rollouts * entry_bytes * safety_factor)

resident_rows = ceil(batch_size / dp_size)
training_cap = ceil(resident_rows * entry_bytes * safety_factor)
```

`max_inflight_rollouts` defaults to `batch_size`. `routing_skew` defaults to
`2.0` and `safety_factor` defaults to `1.2`. Replica and data-parallel counts
come from `derive_num_sampling_workers` and `derive_dp_size`; known GPU counts
must divide exactly by the parallel widths. The skew estimate is not an
overflow-proof bound. Raise it, or set a manual cap, when a worker retains
more replay entries than the average backlog times the skew.

`plan_vllm_gpu_memory_utilization` lowers vLLM's `gpu_memory_utilization` by
the sampling cap divided by the GPU's byte capacity. It raises when that
reservation exceeds half the GPU or would push utilization below `0.05`.

```python
from cortex_training import router_replay_planner as rrp

geometry = rrp.resolve_model_moe_geometry(
    "org/moe-model",
    num_moe_layers_override=40,
    topk_override=8,
    num_experts_override=512,
)
assert geometry.dtype_bytes == 2

backlog = rrp.derive_backlog_rollouts(batch_size=16, async_depth=2)
sampling_workers = rrp.derive_num_sampling_workers(
    sampling_gpus=8,
    sampling_tensor_parallel_size=2,
)
sampling_plan = rrp.plan_sampling_cache_bytes(
    seq_len=8192,
    geometry=geometry,
    backlog_rollouts=backlog,
    num_sampling_workers=sampling_workers,
)
training_plan = rrp.plan_training_cache_bytes(
    seq_len=8192,
    geometry=geometry,
    resident_rows=rrp.derive_resident_rows(
        batch_size=16,
        dp_size=rrp.derive_dp_size(training_gpus=8, sp_size=2),
    ),
)
sampling_cap = sampling_plan.max_cache_bytes
training_cap = training_plan.max_cache_bytes
```

The same sequence length, layer count, and top-k with 512 experts stores each
expert id in two bytes, so each capture is twice as large as one from a model
with 256 or fewer experts. See
[rest-api.md section 8.2](rest-api.md#82-trainingconfig) for the
`max_cache_bytes` field.

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
as `<output_dir>/<sub_job_id>/gpu.jsonl`. Chunks are fetched `max_workers` at a
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
without the call counting as a failure. `sub_job_ids` limits the download; an
entry with no chunks raises `cortex_training.NoChunksError`, a `ValueError`. A
damaged chunk is fetched once more and then skipped and listed in the entry's
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

`cortex_training.peft.normalize_lora_peft_config(cfg)` validates and normalizes
the LoRA subset supported by weight sync. It requires `peft_type` to be exactly
`"Lora"`, so a config copied from a Hugging Face `adapter_config.json` (which
writes `"LORA"`) must be adjusted.

`cortex_training.optimizer.normalize_optimizer_config(cfg)` does the same for
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

## Details not covered here

For exact wire shapes, request framing, and schemas, see the
[REST API reference](rest-api.md). For commands and configuration, see the
[CLI reference](cli.md).
