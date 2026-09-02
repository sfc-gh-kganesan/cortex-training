# Python SDK Reference

```python
from cortex_training import CortexTrainingClient, SubJobConfig, JobType
```

`CortexTrainingClient` is the supported entry point. It is a low-level transport
client: every data-plane call returns a `request_id` that you poll, and results
are whatever the backend returns.

Construct it with a Programmatic Access Token:

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
(6s), `pool_maxsize` (1024), `max_retries` (10). `from_pat` also accepts
`telemetry_timeout` (3s) for best-effort client metrics.

## Client metrics

Clients created with `CortexTrainingClient.from_pat` automatically emit one
best-effort event when an essential operation fails. Set
`CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY=1` to also emit successful outcomes.
Local or mock clients constructed with an explicit `base_url` treat
`emit_metric` as a no-op. Set `CORTEX_TRAINING_DISABLE_TELEMETRY=1` to skip
constructing the emitter.

Tracked operations:

- Job lifecycle: `create_job`, `wait_for_job`, `get_job`, `list_jobs`,
  `cancel_job`, `get_capacity`, `get_experiment_run`
- Compute: `forward_backward`, `forward`, `generate`, `generate_stream`, `step`
- Checkpoints and logs: `save`, `load`, `list_checkpoints`,
  `export_checkpoint`, `delete_checkpoint`, `fetch_execution_logs`
- Multi-sub-job operations: `weight_sync`, `bootstrap_router_replay`,
  `router_replay_discard`, `reset_prefix_cache`
- Async requests: `poll_request`, `get_request_status`, `cancel_request`

Each emitted event body contains `success`, `duration_ms`, `request_count`,
`attempt_count`, and `retry_count`. Failures also include a bounded
`error_message` with common credential patterns redacted. Queryable attributes
include available `job_id`, sub-job identifiers, `request_id`,
`checkpoint_id`, `error.type`, HTTP status, Snowflake request ID, and server
error code. Records use OTLP resource attributes
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
| `create_job(sub_jobs, job_id=None, experiment_name=None)` | `job_id` | Validates each `SubJobConfig` client-side first |
| `create_job_from_body(body)` | response dict | For callers that already hold the REST JSON |
| `get_job(job_id)` | job dict | Includes `sub_jobs` with their configs |
| `list_jobs(status=None)` | list of jobs | Returns the inner list, not the envelope |
| `wait_for_job(job_id)` | job dict | Polls until `running`; raises on `failed`/`done`/`cancelled` or timeout. Does not treat `terminated` as terminal |
| `cancel_job(job_id)` | `None` | Idempotent while cancelling/cancelled |
| `get_capacity()` | capacity dict | `has_reservation`, `reserved_gpus`, `in_use_gpus`, `available_gpus`. The server's `max_total_gpus` ceiling is not surfaced yet |

## Training and sampling

| Method | Returns | Notes |
|---|---|---|
| `forward_backward(job_id, data)` | `request_id` | `data` is a DSSST1 frame; always chunk-wrapped |
| `step(job_id, learning_rate=None)` | `request_id` | Omitting the rate uses the job's optimizer setting |
| `generate(job_id, prompts, sampling_params=None, routing_key=None, strict=None)` | `request_id` | Pre-tokenized prompts are length-checked client-side |
| `generate_stream(...)` | response dict | UTF-8 JSON body; read progress with `get_request_status` |
| `weight_sync(job_id, source_sub_job_id, target_sub_job_ids, weight_format=None)` | `request_id` | `weight_format="lora"` syncs adapters only |
| `forward(job_id, payload, ...)` | response dict | See the known limitation in [rest-api.md section 14](rest-api.md#14-known-limitations) |
| `poll_request(job_id, request_id)` | result dict | Handles backoff, DSSST1 decoding and chunked results |
| `get_request_status(job_id, request_id, max_events=None, cursor=None)` | status dict | |
| `cancel_request(job_id, request_id, ...)` | response dict | |

## Checkpoints

`save(job_id, checkpoint_id=None, checkpoint_type=None)` →  `request_id`
(`checkpoint_type` is `"resumable"` or `"weights-only"`),
`load(job_id, checkpoint_id, source_job_id=None, target_sub_job_id=None)` →
`request_id`, `list_checkpoints(job_id)`, `export_checkpoint(job_id, checkpoint_id)`,
`delete_checkpoint(job_id, checkpoint_id)`.

## Logs

`tail_logs(job_id, ...)` returns one cursor page;
`stream_logs(job_id, follow=True, ...)` yields entries and keeps polling.
`fetch_execution_logs(job_id)` downloads every log file for the job's experiment
run and returns `{sub_job_id, filename, s3_uri, content}` dicts.
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

## Exceptions

`ChunkGroupError` and its subclasses `ChunkGroupRestartError` and
`ChunkGroupConflictError` (all subclasses of `requests.exceptions.HTTPError`)
carry a `.detail` dict for chunk-group failures. Polling raises `RuntimeError`
on a failed/cancelled request and `TimeoutError` at the deadline.

## Details not covered here

For exact wire shapes, request framing, and schemas, see the
[REST API reference](rest-api.md). For commands and configuration, see the
[CLI reference](cli.md).
