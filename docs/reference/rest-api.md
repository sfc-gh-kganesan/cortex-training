# Cortex Training REST API Reference

> **Status.** This page is the customer-facing REST contract: the client in this
> repository codes against it, and it is the reference for anything a caller
> can observe.
>
> **Scope.** This document describes the customer-facing Cortex Training REST
> surface used by this repository's `CortexTrainingClient`. It covers REST paths,
> request framing, asynchronous polling, client-visible schemas, checkpoints,
> generation, operations, and logs.
>
> **Important distinction.** `CortexTrainingClient` is a low-level transport client.
> ArcticTraining may put additional keys such as `context` and `processing`
> inside a forward/backward batch, but those keys are backend contracts, not
> additional REST fields.

---

## 1. Core concepts

### 1.1 Jobs and sub-jobs

A job is the top-level lifecycle object. One job owns one or more typed
sub-jobs:

- `training`: model training, optimizer steps, saves, and runtime loads.
- `sampling`: generation and sampling-side operations.
- `log_probability`: a log-probability worker configuration.

Only `training` and `sampling` sub-jobs can be submitted. Both the Python
client and the CLI reject a `log_probability` sub-job before sending the
request, with the location-prefixed message
`log_probability sub-jobs are not currently supported`; the short alias
`log_prob` is rejected the same way.

The common RL layout is one training sub-job and one sampling sub-job in the
same job.

There is no dedicated public log-probability request method in the current
client. `fwd()` and `fwd_no_grad()` are aliases for the generic `forward`
operation; they are not separate REST endpoints.

### 1.2 Sub-job identifiers

Internal sub-job ids use:

```text
{job_id}:{sub_job_type}:{index}
```

`sub_job_type` is `training`, `sampling`, or `log_prob`, and `index` is the
zero-based occurrence of that type. Examples:

```text
b1fcb345:training:0
b1fcb345:sampling:0
b1fcb345:log_prob:0
```

The create-job `job_type` is `log_probability`, while its internal id segment
is `log_prob`.

### 1.3 Control and data planes

- Control-plane calls return their result synchronously: create, get, list,
  capacity, cancel, experiment metadata, and checkpoint metadata/export.
- Data-plane calls normally return a `request_id`: forward/backward, step,
  save, load, generate, weight sync, and generic operations that schedule work.
  Poll the request until it reaches a terminal state.
- Some generic operations are synchronous and return their result directly.

### 1.4 Supported models

The machine-readable source of truth for model availability, workflow support,
context limits, and recommended configurations is the model catalog under
[`model-catalog/models/`](../../model-catalog/models/): one folder per model,
holding a job file for each supported context, hardware, training or inference
sub-job, and LoRA or full mode. See the
[catalog README](../../model-catalog/README.md).

This API reference does not duplicate the model table. Run
`python model-catalog/validate_catalog.py` after changing the catalog.

---

## 2. Connection, authentication, and headers

### 2.1 Base URL

Paths in this document are relative to:

```text
https://{account_host}/api/v2/databases/{database}/schemas/{schema}/{endpoint}
```

`{account_host}` is the host the client connects to: the Connector's server URL
for a profile, or the PAT host (`--host`, `CORTEX_TRAINING_HOST`,
`from_pat(host=...)`). For a `*.snowflakecomputing.com` or
`*.snowflakecomputing.cn` host whose first label, the account name, contains
`_`, the client sends its REST calls to the hyphen form instead
(`myorg-my_acct.snowflakecomputing.com` becomes
`myorg-my-acct.snowflakecomputing.com`). Snowflake's certificates are wildcards
(for example, `*.snowflakecomputing.com`), and a wildcard can't match a name
containing `_`. Other hosts, such as `ml_team.example.com`, are left as they
are. The Snowflake Connector's own sign-in and artifact downloads keep the host
as given.

`CortexTrainingClient` defaults `endpoint` to `cortex-training`.

```python
client = CortexTrainingClient.from_configs(connection="training")
```

The SQL statements API used by execution-log download is outside this prefix:

```text
https://{account_host}/api/v2/statements
```

So is the account-level job list
([section 5.3](#across-the-account---get-apiv2endpoint)):

```text
https://{account_host}/api/v2/{endpoint}
```

If you point the client at a server that exposes the endpoint under a
different name, pass it explicitly: `endpoint="my-endpoint-name"`.

### 2.2 Authentication

A client built from a Snowflake connection profile
(`CortexTrainingClient.from_configs(...)`) uses the Snowflake Connector's
resolved REST origin and live session token:

```http
Authorization: Snowflake Token="<session-token>"
```

The client reconnects once and replays the request when Snowflake explicitly
reports that the session token expired. Named profiles and the configured
default profile are supported.

When no profile exists, `from_configs()` uses `CORTEX_TRAINING_HOST` and
`CORTEX_TRAINING_PAT` (for CI), and sends:

```http
Authorization: Bearer <PAT>
X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN
```

TLS verification is enabled unless `verify_ssl=False` is passed.

### 2.3 Content types and framing

| Call | Request content type | Body |
|---|---|---|
| Control-plane calls, `step`, `save`, `load`, `operation` | `application/json` | JSON |
| `forward-backward` | `application/octet-stream` | DSSST1 safetensors frame, optionally split into DSSST1 request chunks |
| `generate` | `application/octet-stream` | DSSST1 safetensors frame, optionally split into DSSST1 request chunks |
| `generate-stream` | `application/octet-stream` | DSSST1 safetensors frame, sent in one request and capped at 60 MiB |

Raw `torch.save`/pickle is not the current binary protocol. See
[section 9](#9-dssst1-binary-wire-protocol).

### 2.4 Snowflake request id

Responses may include `x-snowflake-request-id`. Include it when diagnosing an
HTTP failure. When the response carries it, the Python client prefixes
raised HTTP errors with `[snowflake request id: <id>]`.

---

## 3. Status, polling, retries, and errors

### 3.1 Asynchronous request model

A scheduling call usually returns:

```json
{
  "request_id": "request-id",
  "job_id": "job-id"
}
```

Poll:

```text
GET /{job_id}/requests/{request_id}
```

`CortexTrainingClient.poll_request(job_id, request_id)` handles status normalization,
backoff, DSSST1 result decoding, and cursor-addressed result chunks.

### 3.2 Job states

Customer-facing job states are:

| State | Meaning |
|---|---|
| `pending` | Waiting for placement |
| `placing` | Infrastructure is starting (waiting for GPUs) |
| `initializing` | Placed; Cortex Training is warming up (loading the model, starting engines) |
| `running` | Ready for data-plane calls |
| `cancelled` | Cancelled by the caller |
| `terminated` | Platform teardown completed |
| `failed` | Terminal failure; inspect `reason` |

`initializing` is split out from `placing` so callers can distinguish
"waiting for GPUs" from "warming up"; it is customer-visible and not
collapsed by the server. `wait_for_job()` treats it like `placing` — neither
`running` nor terminal — so it keeps polling.

Legacy or internal responses can contain `done`, `unknown`, or enum names such
as `JOB_STATE_RUNNING`. `wait_for_job()` lowercases and removes the
`JOB_STATE_` prefix internally.

`wait_for_job()` raises as soon as the job reaches `failed`, `cancelled`, or
`terminated` (also the legacy `done`, `completed`, `succeeded`, and
`canceled`), instead of polling until its timeout.

### 3.3 Request states

| State | Class |
|---|---|
| `pending` | In flight |
| `running` | In flight |
| `streaming` | In flight on a streaming backend |
| `done`, `completed`, `succeeded` | Success |
| `failed` | Failure |
| `cancelled`, `canceled` | Cancelled |

Raw enum names such as `REQUEST_STATE_DONE` can also appear.
`poll_request()` normalizes them internally.

### 3.4 Poll timing

Defaults:

- Initial interval: `0.5` seconds.
- Backoff multiplier: `1.25`.
- Maximum interval: `6` seconds.
- Overall timeout: `1800` seconds.

All values are configurable on `CortexTrainingClient(...)`.

### 3.5 Retries

Every HTTP request has a default 30-second connect timeout and 600-second read
timeout. Callers can configure a shorter positive scalar or a `(connect, read)`
pair with `CortexTrainingClient(request_timeout=...)`.

The general request path retries connection/time-out failures and HTTP:

```text
307, 308, 404, 409, 429, 500, 502, 503, 504
```

TLS errors count as connection failures and are retried, except certificate
verification failures (an untrusted, expired, or mismatched certificate), which
fail the same way on every attempt and are raised immediately.

Redirects are never followed; a 307 or 308 is retried on the original URL.

A 404 is retried because a job can be briefly invisible right after it is
created. Two 404s are not retried, so they fail at once:

- A 404 whose body has error code `517601` ("Training job ... not found"),
  unless this client created that job in the last 120 seconds. A mistyped job
  id therefore fails immediately instead of after the full backoff.
- A `tail-logs` 404 or 410 for a job that has ended, or for a `sub_job_id` that
  is not one of the job's sub-jobs (see [section 7.8](#78-tail-logs)).

A 404 without a service error code, such as one from a gateway, is still
retried.

Capacity is the exception: it is attempted once with the shorter bounds
documented in [section 5.4](#54-capacity---get-capacity).

The default is ten retries after the first attempt, with exponential jitter.

Create-job uses the narrower connection-establishment retry predicate so an
ambiguous response does not accidentally create a second server-assigned job.

### 3.6 Errors

Non-2xx responses are raised through `requests.Response.raise_for_status()`.
Because that call ignores 3xx, any 3xx response is raised first as an
`Unexpected redirect <status> for url: <url>` HTTP error. Bodies may contain:

```json
{"message": "description", "code": 409}
```

or validation details:

```json
{
  "detail": [
    {"loc": ["body", "field"], "msg": "description", "type": "value_error"}
  ]
}
```

Most data-plane calls and request polls require a `running` job. `tail-logs` is
the exception allowed during `placing`.

---

## 4. Endpoint index

Paths are relative to the prefix in [section 2.1](#21-base-url).

| REST path | HTTP | Client method | Purpose |
|---|---|---|---|
| `/` | `POST` | `create_job`, `create_job_from_body` | Create a job |
| `/` | `GET` | `list_jobs` | List jobs, optionally filtered by status |
| `/api/v2/cortex-training/models` | `GET` | `list_models` | Account-level active models; not under the schema prefix |
| `/capacity` | `GET` | `get_capacity`, `get_capacity_breakdown` | Account reservation and GPU usage, optionally broken down |
| `/{job_id}` | `GET` | `get_job`, `wait_for_job` | Job and sub-job status |
| `/{job_id}:cancel` | `POST` | `cancel_job` | Cancel a job |
| `/{job_id}/experiment-run` | `GET` | `get_experiment_run` | Resolve experiment/run metadata |
| `/{job_id}/checkpoints` | `GET` | `list_checkpoints` | List checkpoints |
| `/{job_id}/checkpoints/{checkpoint_id}:export` | `POST` | `export_checkpoint` | Export checkpoint file links |
| `/{job_id}/checkpoints/{checkpoint_id}` | `DELETE` | `delete_checkpoint` | Delete a checkpoint |
| `/{job_id}/forward-backward` | `POST` | `forward_backward` | Submit forward plus backward |
| `/{job_id}/step` | `POST` | `step` | Submit an optimizer step |
| `/{job_id}/save` | `POST` | `save` | Save a checkpoint |
| `/{job_id}/load` | `POST` | `load` | Load a checkpoint into a running training job |
| `/{job_id}/generate` | `POST` | `generate` | Batch generation |
| `/{job_id}/generate-stream` | `POST` | `generate_stream` | Start polling-based streaming generation |
| `/{job_id}/operation` | `POST` | operation helpers | Generic routed operation |
| `/{job_id}/requests/{request_id}` | `GET` | `get_request_status`, `poll_request` | Poll async work |

The operation endpoint supports the operation types in
[section 7](#7-generic-operation-envelope).

Model deployments add no client-called path: the `*_model_deployment*` methods
use the job paths above, and a deployment serves under
`/inference/v1` ([section 14](#14-model-deployments)).

These calls are outside the Cortex Training prefix:

| REST path | HTTP | Client use | Purpose |
|---|---|---|---|
| `/api/v2/{endpoint}` | `GET` | `iter_jobs` | List jobs across the account ([section 5.3](#across-the-account---get-apiv2endpoint)) |
| `/api/v2/statements` | `POST` | artifact download methods | Resolve the current Snowflake identity |

---

## 5. Control-plane endpoints

### 5.1 Create job - `POST /`

Typed Python call:

```python
job_id = client.create_job(
    sub_jobs=[training_sub_job, sampling_sub_job],
    experiment_name=None,
    hardware="B200",
    idle_timeout_seconds=1800,
    pending_timeout_seconds=86400,
)
```

REST body:

```json
{
  "sub_job_configs": [
    {
      "job_type": "sampling",
      "model_name": "Qwen/Qwen3-1.7B",
      "inference_config": {
        "max_seq_len": 2048,
        "n_gpus": 1
      }
    }
  ],
  "experiment_name": "optional-experiment",
  "hardware": "B200"
  "idle_timeout_seconds": 1800,
  "pending_timeout_seconds": 86400
}
```

`sub_job_configs` must be a non-empty list carrying **zero or one** `training`
sub-job and any number of `sampling` sub-jobs. A second training sub-job is
rejected with `at most one training sub-job is supported per job`; the server
enforces the same rule for every caller. A `log_probability` (or `log_prob`)
sub-job is rejected client-side with
`sub_job_configs[INDEX].job_type: log_probability sub-jobs are not currently
supported`.

The typed path validates each `SubJobConfig`; `create_job_from_body()` checks the
outer body, the non-empty list, the training-sub-job count, and each sub-job's
`job_type` for the unsupported log-probability spellings before forwarding it.
It also normalizes every `peft_config` the body carries, rewriting it in
place and rejecting an invalid one before the request is sent (see
[section 8.2](#82-trainingconfig)), and rejects a training
`ds_config.zero_optimization.stage` of 3 or higher (ZeRO-3 is not supported).
Everything else in a raw body, `optimizer` included, is forwarded exactly as
given. `cortex-training submit` repeats those
checks so `--dry-run` fails the same way without a connection.

#### GPU hardware - `hardware`

Optional. One of `H200`, `B200`, or `B300`; every sub-job in the job runs on
that GPU type. Omitted means `H200`. Any other value is rejected — the typed
`create_job()` path raises `ValueError` before sending, and the server rejects
an unknown value on the raw `create_job_from_body()` path. The typed path also
accepts the `Hardware` enum (`Hardware.B200`) in place of the string.

See [GPU hardware](../concepts/hardware.md) for the job-level model and how
capacity is scoped per type.

#### `idle_timeout_seconds`

Optional. Bounds how long the job may sit idle before the server reclaims it.

| Value | Behavior |
|---|---|
| Omitted | Server default applies, currently 30 minutes. |
| `0` | Disables reclamation for this job. Not a zero-second timeout — the job is not reclaimed for being idle, though it can still be cancelled or terminated. |
| `300` through `604800` | Use that many seconds. |

Negative values, `1`-`299`, values above `604800`, and non-integers are rejected.

#### `pending_timeout_seconds`

Optional. Bounds how long the job may stay `pending` waiting for capacity before
it fails with reason `pending_timeout`.

| Value | Behavior |
|---|---|
| Omitted | Server default applies, currently 24 hours. |
| `300` through `604800` | Use that many seconds. |

There is no disable value: `0` is rejected, along with negatives, `1`-`299`,
values above `604800`, and non-integers.

When either field is set, Get job and List jobs echo it back; when omitted it may
be absent from those responses.

Response:

```json
{"job_id": "job-id"}
```

### 5.2 Get job - `GET /{job_id}`

The response shape consumed by this repository is flat at each sub-job:

```json
{
  "job_id": "job-id",
  "status": "running",
  "reason": "",
  "reason_details": "",
  "created_at": "2026-07-20T18:00:00Z",
  "updated_at": "2026-07-20T18:01:00Z",
  "state_changed_at": "2026-07-20T18:01:00Z",
  "submitted_by": "SOME_USER",
  "owner_role": "SOME_ROLE",
  "sub_jobs": [
    {
      "sub_job_id": "job-id:training:0",
      "job_type": "training",
      "status": "running",
      "model_name": "Qwen/Qwen3-1.7B",
      "training_config": {
        "optimizer": {"name": "AdamW", "lr": 0.0001},
        "max_seq_len": 2048,
        "train_batch_size": 8,
        "n_gpus": 8
      }
    }
  ]
}
```

Each `sub_jobs[]` entry is the sub-job's config plus the `sub_job_id` and
`status` the service adds. On service versions that report it, once the job is
running the config is the sub-job's **effective** config: `job_type`,
`model_name`, `dtype`, `seed` (`null` when unset), any `source_checkpoint_info`,
and the `training_config` or `inference_config` the job runs with, the service's
defaults filled in (`null` for an optional field left unset). A key the service
did not use, such as a misspelled one, is not in it. Before the job runs, on
services that don't report it, and for model deployments, the entry is the
config as submitted, and its whole numbers may come back as floats
(`"n_gpus": 4.0`).

The record also has `hardware` (the default when none was set), `comment`
(`null` when none was set), and `idle_timeout_seconds` /
`pending_timeout_seconds` when they were set (see
[section 5.1](#51-create-job---post-)). It does not name the job's experiment;
see [section 5.7](#57-experiment-run---get-job_idexperiment-run).

#### Launching a job like another

`create_job_body_from_job(job, experiment_name=...)` turns a Get job record
into a CreateJob body for a job configured like it, which
`create_job_from_body()` takes as it is:

```python
from cortex_training import create_job_body_from_job

name = client.get_experiment_run(job_id)["experiment_name"]
body = create_job_body_from_job(client.get_job(job_id), experiment_name=name)
new_job_id = client.create_job_from_body(body)["job_id"]
```

The body has `hardware`, `comment`, `experiment_name`, `idle_timeout_seconds`,
`pending_timeout_seconds` and `sub_job_configs`, each with the value the job
runs with, and each sub-job has the record's fields plus `dtype`, `seed` and
`source_checkpoint_info`. A timeout the record lacks is the service's default
from [section 5.1](#51-create-job---post-) (1800 and 86400 seconds); a
`comment`, `dtype`, `seed` or `source_checkpoint_info` it lacks is `null`, as is
`experiment_name` when none is given. Given the job's own, the new job joins
that experiment as a new run. A model deployment, which can set neither, keeps
`pending_timeout_seconds` and `experiment_name` `null`. Left out are `job_id`
(the service rejects one on create), `sub_job_id`, `status`, `reason`, the
times, `submitted_by`, `owner_role` and `image_tag` (a new job runs the current
image). A sub-job started from external weights has
`source_checkpoint_info: {"external_stage_path": null}` to fill in, as the record
does not say which weights. Whole-number floats become integers. It raises
`ValueError` for a record with no sub-jobs or no `hardware` (the service could
not read the job). `cortex-training get JOB_ID` prints the same body, with the
job's experiment; `get --raw` prints the record.

`wait_for_job()` repeatedly calls this endpoint until `running`.

`state_changed_at` (UTC time the job entered its current `status`) is optional
and may be absent. The client reads it only for the stalled-upload warnings in
[section 12.3](#123-persisted-stdout-download); without it, those warnings skip
the recent-start grace period and still require the sub-job to be `running`.

`submitted_by` (submitting user's name) and `owner_role` (owning role's name)
are best-effort and may be absent, e.g. if the user/role has since been
dropped or the requesting role no longer has access to it. Both fields also
appear on each job entry returned by `list_jobs()`
([section 5.3](#53-list-jobs---get-)).

### 5.3 List jobs - `GET /`

Optional query:

```text
?status=running
```

REST response:

```json
{"jobs": [{"job_id": "job-id", "status": "running", "sub_jobs": []}]}
```

`CortexTrainingClient.list_jobs()` returns only the `jobs` list, not the outer object.
The client forwards the status string without validating an enum.

#### Across the account - `GET /api/v2/{endpoint}`

Lists the jobs the caller can see in every database and schema of the account,
newest first, one page at a time. The path is outside the per-schema prefix:
`https://{account_host}/api/v2/{endpoint}`. It returns 404 where account-level
listing is not enabled for the account. Jobs in dropped databases or schemas
are not listed. Every other job call stays on the per-schema routes.

Optional query parameters:

| Parameter | Value |
|---|---|
| `database` | Only jobs in this database. SQL identifier rules apply: an unquoted name is case-insensitive, a `"quoted"` one exact. A database the caller cannot see returns 400. |
| `schema` | Only jobs in this schema of `database`, by the same rules. Requires `database`: a schema name alone does not name one schema, and sending it without `database` returns 400. A schema the caller cannot see returns 400. |
| `status` | Comma-separated and case-insensitive: `pending`, `placing`, `initializing`, `running`, `failed`, `cancelled`, `terminated`, and the groups `active` (`pending` through `running`) and `done` (`cancelled`, `terminated`, and the `done` that older jobs show). `pending` does not include `placing`. Any other value returns 400. |
| `submitted_by` | Only jobs submitted by this user, by SQL identifier rules. Jobs with no recorded submitter never match, and an unknown user matches no jobs. |
| `created_after` | Inclusive lower bound on creation time: an RFC 3339 timestamp, or `YYYY-MM-DD` for midnight UTC. Must be earlier than `created_before`. |
| `created_before` | Exclusive upper bound on creation time, in the same forms. |
| `limit` | Page size, 1 to 1000. Defaults to 100. |
| `page_token` | The previous page's `next_page_token`. Send it with the same filters; different filters return 400. |

The creation bounds compare with the time the service recorded the job, which
can differ slightly from the job's `created_at`.

REST response:

```json
{
  "jobs": [
    {
      "job_id": "job-id",
      "database_name": "CORTEX_TRAINING_DB",
      "schema_name": "PUBLIC",
      "status": "RUNNING"
    }
  ],
  "next_page_token": "opaque-token"
}
```

Each job carries the per-schema list's fields plus `database_name` and
`schema_name`. A page can hold fewer than `limit` jobs, or none, and still carry
`next_page_token`; keep requesting until it is absent. The token is opaque and
sealed by the service: pass it back unchanged. A token the service can no
longer open, for example after it rotated the key that seals tokens, returns
400 `Invalid page_token`; start again from the first page. A job whose live
status cannot be read while `status` is set is left out, uncounted, so a page's
size is not a count of what matched.

| Answer | Meaning |
|---|---|
| 403 | Cortex Training is not enabled for the account, or its trial has ended; the per-schema routes answer the same. A 403 without the service's JSON error may come from a network policy or a proxy in front of the service. |
| 404 with an empty body | The service does not have this route yet. |
| 404 with JSON `Account-level job listing is not enabled for this account` | The route is there but not enabled for the account. |
| 405 with an empty body | The Cortex Training REST API is turned off for the account. |

`CortexTrainingClient.iter_jobs(...)` takes the same filters, with `page_size`
for `limit`, checks its arguments on the call, and yields jobs across pages as
you iterate. A `schema` without a `database` is refused before any request, as a
`ValueError`, rather than reaching the service's 400. It raises a 404 at once
instead of retrying it, and never creates the configured database. `CortexTrainingClient.list_jobs_page(...)` reads one
page instead: it takes the same filters plus `page_token`, and per-call
`max_retries` and `request_timeout`, and returns
`{"jobs": [...], "next_page_token": str | None}`. It raises 403, 404 and 405
at once as `AccountJobListUnavailable`, whose `reason` is `forbidden`,
`not_deployed`, `disabled` or `rest_disabled` by the table above, and the 400
`Invalid page_token` as `InvalidPageToken`; both subclass `requests.HTTPError`.
`for_job(job)` returns a client bound to a listed job's `database_name` and
`schema_name`; it borrows the parent's session and creates no database either.

The TUI's job list sends `database` (and `schema` with it) when a search names a
whole one, so that location's jobs are read whole rather than filtered out of
the newest jobs the walk reads; see the job list in `docs/reference/cli.md`. A
page token is only sent back with the filters it was made under, as above.

### 5.4 List models - `GET /api/v2/cortex-training/models`

This account-scoped endpoint returns active models configured for new jobs.
It is not under the database and schema prefix. The server resolves the
account from the authenticated session:

```json
{"models": [{"name": "Qwen/Qwen3-8B"}]}
```

`CortexTrainingClient.list_models()` returns only the `models` list. The CLI
`models` command restores the server-shaped envelope. This response is policy
discovery, not a cache or capacity guarantee; a listed model can still wait for
weight synchronization or enough GPUs for a particular job shape.

### 5.5 Capacity - `GET /capacity`

This account-scoped endpoint takes no account id from the caller. The server
resolves the account from the authenticated session. Besides `hardware`
(below), an optional `breakdown` query parameter splits the usage numbers; see
[Breakdown](#breakdown---breakdown).

Optional query:

```text
?hardware=B200
```

`hardware` scopes the numbers to one GPU type and takes the same values as the
create-job field: `H200`, `B200`, or `B300`, defaulting to `H200` when omitted.
`get_capacity()` sends the parameter only when you pass one, and rejects an
unknown value client-side. The CLI `capacity` command, with no `--hardware`
flag, calls this endpoint once per type and wraps the results in
`capacity_by_hardware`.

A known type that the account does not support is rejected with `409` and code
`517604`, for example:

```json
{
  "code": "517604",
  "message": "Cortex training getCapacity failed: unsupported hardware \"B200\": supported hardware is H200, B300"
}
```

The CLI lists such types in `unsupported_hardware` instead of failing, and
fails on any other error.

The SDK attempts each capacity lookup once with at most a 10-second connect
timeout and a 30-second read timeout; a shorter configured request timeout still
wins. This endpoint is a fast control-plane read, so a silent network, proxy, or
service hop fails rather than blocking indefinitely.

```json
{
  "has_reservation": true,
  "max_total_gpus": 64,
  "reserved_gpus": 64,
  "in_use_gpus": 8,
  "pending_gpus": 16,
  "available_gpus": 40
}
```

- `has_reservation`: whether `max_total_gpus` is a *guaranteed* commitment
  rather than a best-effort per-account cap. When false, placement draws on the
  shared pool, so `available_gpus` is an upper bound.
- `max_total_gpus`: the account's GPU ceiling — its commitment when
  `has_reservation`, else its per-account cap. `-1` means no ceiling, `0` a real
  quota of zero, `> 0` the limit.
- `reserved_gpus` **(deprecated)**: use `max_total_gpus` with
  `has_reservation` instead.
- `in_use_gpus`: GPUs the account holds — `placing`, `initializing`, and
  `running` jobs. Queued work is not counted here; see `pending_gpus`.
- `pending_gpus`: GPUs requested by jobs still waiting for capacity
  (`pending`). Claims quota, so it reduces `available_gpus`.
- `available_gpus`: what a new job could start with now — ceiling headroom
  capped by currently schedulable capacity. Zero does not mean blocked: a
  submit within the ceiling is still accepted and queued.

Proto3 JSON may omit false or zero fields, so an account holding nothing under a
zero ceiling responds with literally `{}`. `get_capacity()` fills in the
documented defaults for all six keys.

#### Breakdown - `breakdown`

Optional. A comma-separated list of dimensions that split `in_use_gpus` and
`pending_gpus`:

| Dimension | Splits the usage by |
|---|---|
| `job_type` | Sub-job type |
| `submitted_by` | The user who submitted each job |

Names are case-insensitive; blanks and repeats are ignored. More than 8
names, or a name not matching `[a-z_]{1,32}`, is HTTP `400`. A well-formed
name the server does not know is listed in `unsupported_dimensions`. Without
the parameter the response has no `breakdown` key.

The breakdown is enabled per account. Until it is enabled for yours, each
requested dimension reports `DISABLED`. Example for `?breakdown=job_type,submitted_by`:

```json
{
  "has_reservation": true,
  "max_total_gpus": 64,
  "reserved_gpus": 64,
  "in_use_gpus": 44,
  "pending_gpus": 16,
  "available_gpus": 4,
  "breakdown": {
    "unsupported_dimensions": [],
    "job_count": 6,
    "truncated": false,
    "by_job_type": {
      "status": "OK",
      "groups": [
        {"job_type": "TRAINING", "in_use_gpus": 28, "pending_gpus": 16,
         "sub_job_count": 5},
        {"job_type": "SAMPLING", "in_use_gpus": 16, "pending_gpus": 0,
         "sub_job_count": 2},
        {"job_type": "LOG_PROBABILITY", "in_use_gpus": 0, "pending_gpus": 0,
         "sub_job_count": 0}
      ]
    },
    "by_submitted_by": {
      "status": "OK",
      "groups": [
        {"kind": "USER", "submitted_by": "YOUR_USER", "is_caller": true,
         "in_use_gpus": 24, "pending_gpus": 0, "job_count": 2},
        {"kind": "USER", "submitted_by": "SOME_USER", "is_caller": false,
         "in_use_gpus": 8, "pending_gpus": 16, "job_count": 2},
        {"kind": "OTHER", "in_use_gpus": 8, "pending_gpus": 0,
         "job_count": 1},
        {"kind": "UNATTRIBUTED", "in_use_gpus": 4, "pending_gpus": 0,
         "job_count": 1}
      ]
    }
  }
}
```

- `unsupported_dimensions`: always present, possibly empty.
- `job_count`: jobs behind `in_use_gpus` and `pending_gpus`, including those
  holding no GPUs. Present only when a dimension is `OK`.
- `truncated`: the account had more jobs than the server attributes one by
  one; the rest count as `UNATTRIBUTED`. Present only when a dimension is `OK`.
- `by_job_type` / `by_submitted_by`: present only when requested.

Each dimension has a `status` and `groups`. Every status comes with HTTP
`200` and the six totals unchanged:

| `status` | Meaning |
|---|---|
| `OK` | `groups` adds up to `in_use_gpus` and `pending_gpus`. |
| `DISABLED` | Not enabled for the account. `groups` is empty. |
| `UNAVAILABLE` | Cannot be computed for this request, including when the service does not support it yet. `groups` is empty. |

Read the split from `groups`; a newer service may add statuses.

`by_job_type.groups[]`: `TRAINING`, `SAMPLING` and `LOG_PROBABILITY` first,
then any other type (`UNSPECIFIED` or `UNKNOWN` when the service cannot name
it). A job counts under each of its sub-jobs' types. `log_probability`
sub-jobs and sub-jobs that set `multiplex_job_id` add 0 GPUs but count in
`sub_job_count`.

`by_submitted_by` names the submitter of every job the caller can see. Each
job counts in exactly one group:

| `kind` | Jobs |
|---|---|
| `USER` | Submitted by `submitted_by`, and visible to the caller. `is_caller` marks the caller's row. |
| `OTHER` | Not visible to the caller. |
| `UNATTRIBUTED` | No known submitter: none was recorded, the user no longer exists, or the job is past the `truncated` limit. |

`USER` rows come first, most GPUs (`in_use_gpus + pending_gpus`) first and
then by name, then `OTHER`, then `UNATTRIBUTED`. Find the caller's row by
`is_caller`, not position. The breakdown never contains job ids or database
or schema names.

`get_capacity_breakdown(hardware=None, *, dimensions=("job_type",
"submitted_by"))` sends this parameter. It keeps `groups` whenever the server
sends them and passes `status` through as sent. A missing `status` reads as
`OK` with groups and `UNAVAILABLE` without, and a dimension missing from the
response (including from a server without this parameter) is `UNAVAILABLE`.
It fills missing counts with `0` and always lists the three known job types
when there are groups. The CLI exposes it as
[`capacity --breakdown`](cli.md#capacity-breakdown).

### 5.6 Cancel job - `POST /{job_id}:cancel`

No body. Pending, placing, and running jobs enter cancellation. Repeating a
cancel while a job is cancelling or already cancelled is an idempotent success.
Terminated or failed jobs return a precondition/conflict-style error.

`cancel_job()` returns `None`.

### 5.7 Experiment run - `GET /{job_id}/experiment-run`

```json
{
  "experiment_name": "DB.SCHEMA.EXPERIMENT",
  "experiment_run_name": "RUN_NAME"
}
```

`fetch_execution_logs()` uses these values to locate the run's stage.

### 5.8 List checkpoints - `GET /{job_id}/checkpoints`

```json
{
  "checkpoints": [
    {
      "checkpoint_id": "global_step12",
      "global_steps": 12,
      "avg_loss": 0.83,
      "created_at": "2026-07-20T18:05:00Z",
      "checkpoint_type": "resumable"
    }
  ]
}
```

`list_checkpoints()` returns only the `checkpoints` list. The CLI exposes this
method with a required job id and restores the server-shaped JSON envelope:

```bash
cortex-training checkpoints JOB_ID
```

The command prints `{"checkpoints": [...]}`. As with other commands, compact
output is selected with the global option:

```bash
cortex-training --compact checkpoints JOB_ID
```

### 5.9 Export checkpoint

```text
POST /{job_id}/checkpoints/{checkpoint_id}:export
```

No body.

```json
{
  "checkpoint_id": "global_step12",
  "files": [
    {
      "filename": "model.safetensors",
      "url": "https://presigned.example/...",
      "size_bytes": 1073741824
    }
  ],
  "expires_at": "2026-07-20T19:05:00Z"
}
```

The URLs are short-lived.

The CLI command `cortex-training download-checkpoint JOB_ID CHECKPOINT_ID` with
`--output-dir DIRECTORY` lists checkpoints, calls this export endpoint, and
downloads the returned HTTPS links. It accepts repeated `--include` filename
globs and automatically resumes `.partial` files using HTTP Range requests.
The CLI always displays byte and file progress on stderr without
changing the JSON summary on stdout.
The server-provided `size_bytes` must be a non-negative integer or decimal
string; the CLI verifies the downloaded size before promoting a file. See
[Download a checkpoint](cli.md#download-a-checkpoint) for flags and output.

### 5.10 Delete checkpoint

```text
DELETE /{job_id}/checkpoints/{checkpoint_id}
```

No body, and none returned: `200`/`204` on success, so `delete_checkpoint()`
returns `None`. Unlike `:export`, this is a plain `DELETE` verb on the checkpoint
resource. A missing checkpoint is not treated as success — the server's status
(e.g. `404`) propagates as a raised error.

---

## 6. Data-plane endpoints

### 6.1 Forward/backward - `POST /{job_id}/forward-backward`

The body is a DSSST1 frame. The standard object is:

```python
{
    "args": (),
    "kwargs": {
        "input_ids": input_ids,
        "labels": labels,
    },
}
```

ArcticTraining can add backend-owned `context` and `processing` keys.

Build a basic frame with:

```python
payload = serialize_forward_backward_args(
    args=(),
    kwargs={"input_ids": input_ids, "labels": labels},
)
request_id = client.forward_backward(job_id, payload)
result = client.poll_request(job_id, request_id)
```

Or serialize an extended batch directly:

```python
from cortex_training import wire

payload = wire.dumps(
    batch,
    metadata={
        "response_options": {
            "format": "dssst1",
            "delivery": "chunked",
        }
    },
)
```

`forward_backward()` **always** wraps the frame in a DSSST1 request-chunk
envelope, even when it fits in a single chunk, so every logical operation has a
caller-generated identity for idempotent replay. Frames larger than 60 MiB are
split across several chunks. Each chunk is posted to the same path; only the
last response may carry the `request_id`. See
[section 9.3](#93-request-chunking).

Typical polled result:

```json
{
  "job_id": "job-id",
  "avg_loss": 1.0237,
  "metrics": {},
  "post_process_outputs": {}
}
```

The exact metrics are backend/loss dependent. The current ArcticTraining
forward/backward response assembler intentionally returns
`post_process_outputs` as an empty object; callers must not assume that
`compute_logprobs` appears there.

When the training sub-job sets `router_replay.mode = "best_effort"` (see
[section 8.2](#82-trainingconfig)), `metrics` of a forward/backward request that
carries router-replay ids also reports how the request's rows were routed. Each row is counted once, however rows are sharded across
data-parallel and sequence-parallel ranks. A "fresh" row is one whose routing
no sampling worker held, so the trainer's own MoE gate routed it.

| Key | Meaning |
|---|---|
| `router_replay/rows_sent` | Rows in the request that carried a router-replay id |
| `router_replay/rows_replayed` | Rows routed with the sampler's captured routing |
| `router_replay/rows_fresh` | Rows routed by the trainer's own gate |
| `router_replay/replayed_fraction` | `rows_replayed / rows_sent` (`0.0` for a request with no rows) |
| `router_replay/tokens_replayed` | Valid tokens in replayed rows (attention-mask count; the padded length when the request has no attention mask) |
| `router_replay/tokens_fresh` | Valid tokens in fresh rows (attention-mask count; the padded length when the request has no attention mask) |

In the default `"strict"` mode these keys are absent.

### 6.2 Optimizer step - `POST /{job_id}/step`

```json
{"learning_rate": 0.00002}
```

The field is optional. `step(job_id)` sends `{}`.

Immediate response:

```json
{"request_id": "request-id", "job_id": "job-id"}
```

Typical result:

```json
{"global_steps": 12, "last_lr": 0.00002}
```

`last_lr` may be a scalar or list depending on the backend.

When per-step peak-memory reporting is enabled (see
[section 8.5](#85-memory-diagnostics-settings)), the result also carries a
`peak_memory` object of byte counts:

```json
{
  "global_steps": 12,
  "last_lr": 0.00002,
  "peak_memory": {"gpu_peak": 123456789, "cpu_peak": 987654321}
}
```

`gpu_peak` is the job-wide maximum GPU high-water mark across ranks for the
current `/forward-backward` plus `/step` cycle; `cpu_peak` is rank 0's current
process RSS. The key is absent when the setting is disabled.

### 6.3 Save checkpoint - `POST /{job_id}/save`

```json
{
  "checkpoint_id": "optional-tag",
  "checkpoint_type": "weights-only"
}
```

Both fields are optional in the Python client request. When `checkpoint_type`
is supplied, the client lowercases and validates it as:

- `resumable`: training weights plus optimizer/training state.
- `weights-only`: Hugging Face-style model assets suitable for sampling
  initialization.

The backend default is `resumable`.

Compatibility caveat: `checkpoint_id` is not represented in the server's
`SaveRequest`, so a caller-selected value is not forwarded. Use the
server-assigned id from the result's `stage_path` or the job's checkpoint list
(below), rather than relying on a caller-selected value.

Immediate response:

```json
{"request_id": "request-id", "job_id": "job-id"}
```

On `release_20260903_175601`, the polled `weights-only` save result observed on
2026-09-07 carries no `checkpoint_id` key:

```json
{
  "job_id": "job-id:training:0",
  "stage_path": "s3://<stage>/.../checkpoints/cp_00000000-0000-4000-8000-000000000001/global_step1/",
  "checkpoint_path": "/tmp/dss.ray_dss/ds/job-<job-id>:training:0/weights-only",
  "checkpoint_tag": "global_step1",
  "version": 1.0
}
```

The durable checkpoint id is the `cp_<uuid>` path segment of `stage_path`
(`/checkpoints/(cp_[0-9a-fA-F-]+)/`), and `GET /{job_id}/checkpoints` lists the
same id with its `checkpoint_type` and `created_at`. Resolve the id from either;
`checkpoint_path` is a mount inside the saving job's containers and does not
resolve from any other job.

### 6.4 Runtime load - `POST /{job_id}/load`

```json
{
  "checkpoint_id": "global_step12",
  "source_job_id": "optional-source-job",
  "target_sub_job_id": "optional-job-id:training:0"
}
```

`checkpoint_id` is required. `source_job_id` loads from another job's checkpoint
store. `target_sub_job_id` routes the load to a specific training sub-job of the
session. The route is asynchronous:

```python
request_id = client.load(
    job_id,
    checkpoint_id="global_step12",
    source_job_id=source_job_id,
    target_sub_job_id=f"{job_id}:training:0",  # optional
)
result = client.poll_request(job_id, request_id)
```

When `target_sub_job_id` is omitted, the service routes the load to the
session's training sub-job (the historical behavior). When supplied, the sub-job
must belong to this session — otherwise the request fails with `400` — and must
be a training sub-job; loading into a non-training (e.g. sampling) zone is
rejected with `501`.

#### Discovering sub-job IDs

Sub-job IDs follow the format `{job_id}:{job_type}:{index}`, for example:
`b1fcb345:training:0`. To discover available training sub-jobs:

```python
job = client.get_job(job_id)
training_sub_jobs = [
    sj for sj in job["sub_jobs"] if sj["job_type"] == "training"
]
for sj in training_sub_jobs:
    print(f"ID: {sj['sub_job_id']}, DP size: {sj['training_config']['n_gpus']}")
```

#### When to use target_sub_job_id (load)

A job has at most one training sub-job, so omit `target_sub_job_id` to use the
default. Use it when you want explicit routing control — for example naming the
sub-job in tooling that must not rely on the server's default resolution. It
selects a training sub-job; the sampling-side selector is `target_sub_job_ids` on
weight sync.

#### DP size compatibility

If the target sub-job has a different `n_gpus` than the checkpoint's source
sub-job (changing DP size), the target sub-job **must** have been created with
`load_optimizer_states: false` in its `SubJobConfig`. The optimizer states are
DP-sharded and cannot be resized. This is a **creation-time** setting; it cannot
be changed at runtime. The server will reject incompatible loads.

### 6.5 Create-time checkpoint initialization

Create-time initialization is not `resume_from_checkpoint` in the typed
`cortex-training` API. To initialize from a checkpoint saved by Cortex
Training, use:

```json
{
  "source_checkpoint_info": {
    "checkpoint_id": "checkpoint-id",
    "source_job_id": "source-job-id"
  }
}
```

This object is a field on `SubJobConfig`. The server stamps scoped `stage_info`
credentials; clients should not provide credentials themselves.

To initialize from a complete Hugging Face safetensors directory uploaded to a
Snowflake stage, use the alternative source field:

```json
{
  "source_checkpoint_info": {
    "external_stage_path": "@MY_DB.MY_SCHEMA.MODEL_IMPORT/qwen-sft-run7"
  }
}
```

`external_stage_path` is mutually exclusive with both `checkpoint_id` and
`source_job_id`. The service resolves the stage and credentials under the
submitting role. The source must be an S3-backed stage containing a flat,
uncompressed Hugging Face safetensors model directory. See
[Start a Job From Your Own Weights](../guides/training/external-weights.md)
for the upload workflow and file requirements.

When initializing from a checkpoint saved by Cortex Training, sampling requires
a `weights-only` checkpoint. The source job no longer needs to be running after
the checkpoint has been saved.

For training initialization from a Cortex Training `resumable` checkpoint, set
`training_config.load_optimizer_states=false` (see
[section 8.2](#82-trainingconfig)) to restore weights only with a fresh
optimizer. This is required when the new job changes data-parallel size, since
the DP-sharded optimizer cannot be resized.

### 6.6 Generate - `POST /{job_id}/generate`

Python call:

```python
request_id = client.generate(
    job_id,
    prompts=["Hello", [1, 2, 3]],
    sampling_params={"max_tokens": 64, "temperature": 0.7},
    routing_key="conversation-1",
    strict=False,
    sub_job_id=f"{job_id}:sampling:1",   # optional
)
```

Logical request object inside the DSSST1 frame:

```json
{
  "prompts": ["Hello", [1, 2, 3]],
  "sampling_params": {"max_tokens": 64, "temperature": 0.7},
  "routing_key": "conversation-1",
  "strict": false
}
```

Rules:

- `sub_job_id` is optional and names which sampling sub-job runs the request;
  discover the ids from `get_job()`'s `sub_jobs`. It is sent as a **query
  parameter** (`?sub_job_id=...`), not inside the frame, because the server routes
  the request before it reads the DSSST1 body. If you omit it, the client sends no
  `sub_job_id` and the server chooses the sub-job. Currently it uses the job's only
  sampling sub-job, and rejects the request (listing the valid ids) when there are
  several. Applies to `generate-stream` identically.
- `prompts` is a list of string prompts and/or token-id lists. A flat list of
  integers such as `[1, 2, 3]`, and a bare 1-D tensor, are each read as one
  pre-tokenized prompt rather than as a batch. For a batch of several prompts,
  use the nested form `[[1, 2, 3], [4, 5]]`.
- `sampling_params` may be one object or a list of objects/null values aligned
  with `prompts`.
- `routing_key` may be one string or an aligned list of strings/null values.
- `strict` controls strict routing-key affinity.

Pre-tokenized prompts travel in the frame's tensor section as `int32` tensors,
not as JSON numbers in the frame header. A token id that does not fit `int32`
falls back to `int64` for that prompt, which only doubles its wire size. String
prompts are not converted — the server tokenizes them. Setting
`CORTEX_TRAINING_DISABLE_TENSOR_PROMPTS` to a truthy value puts the token ids
back in the header as JSON lists; the request body is a DSSST1 frame either
way.

For pre-tokenized prompts, the client fetches and caches the sampling sub-job's
`inference_config.max_seq_len`. It rejects a prompt when
`len(prompt) >= max_seq_len`, preserving room for at least one output token.
The check runs before the tensor conversion, so the environment variable above
does not disable it. String prompts are left for the server tokenizer to
validate.

Generate uses the same DSSST1 response options as forward/backward, but unlike
forward/backward it sends the frame unwrapped when it fits, and only splits it
into request chunks above 60 MiB. See [section 9.3](#93-request-chunking).

Typical polled result:

```json
{
  "job_id": "job-id",
  "results": [
    {
      "text": " generated text",
      "token_ids": [101, 102],
      "finish_reason": "stop"
    }
  ]
}
```

The backend can add fields such as log probabilities or action masks. For a
request submitted by the same `CortexTrainingClient` instance, `poll_request()`
converts tensor values under `results` back to Python lists.

#### Per-token log probabilities

On server versions that pass vLLM's `logprobs` and `prompt_logprobs` sampling
options through, two optional `sampling_params` keys add log probabilities to
each result:

| Key | Result field | Contents |
|---|---|---|
| `logprobs: 0` | `logprobs` | One entry per generated token, aligned with `token_ids` |
| `prompt_logprobs: 0` | `prompt_logprobs` | One entry per prompt token; the first entry is `null` because nothing precedes the first token |

Each entry maps a token id to an object that includes `logprob`: the
log probability of that token given everything before it. With `0`, the map
holds only the token that actually appears at that position; a positive value
`k` also includes the `k` most likely alternatives. The token id is a JSON
object key, so it can arrive as a string.

`prompt_logprobs` scores a known token sequence without decoding or
re-tokenizing it: send the sequence as a pre-tokenized prompt with
`max_tokens: 1` and read `prompt_logprobs[i]` as the log probability of prompt
token `i`. The prompt must stay shorter than the sampling sub-job's
`max_seq_len`. The
[on-policy distillation example](../../recipes/distillation/README.md) scores
student rollouts with a teacher this way.

### 6.7 Streaming generate - `POST /{job_id}/generate-stream`

`generate_stream()` accepts the same logical fields as `generate()` and encodes
them into the same DSSST1 frame, including the `int32` tensor encoding of
pre-tokenized prompts described in [section 6.6](#66-generate---post-job_idgenerate).

Unlike `generate`, this path is not request-chunked: the client sends the frame
in a single POST, and the encoded body must not exceed 60 MiB.

**Breaking change.** `POST /{job_id}/generate-stream` no longer accepts a
UTF-8 JSON body. There is no dual encoding and no negotiation — the frame is
the only accepted request form. `CORTEX_TRAINING_DISABLE_TENSOR_PROMPTS`
changes how prompts are encoded *inside* the frame; it does not turn the body
back into JSON. The response is unchanged.

Immediate response:

```json
{
  "request_id": "request-id",
  "job_id": "job-id",
  "count": 2
}
```

Poll the unified request endpoint with `max_events`:

```python
status = client.get_request_status(
    job_id,
    request_id,
    max_events=64,
)
```

Events are opaque backend objects. Common event shapes are:

```json
{"type": "result", "index": 0, "result": {"text": "text", "token_ids": [1, 2]}}
```

```json
{"type": "error", "index": 1, "error": "description"}
```

```json
{"type": "done", "completed": 1, "failed": 1}
```

Streaming event delivery advances a server-side delivery cursor. There is no
client-supplied retry cursor for these stream events, so losing a successful
poll response can lose the events consumed by that response.

### 6.8 Poll request - `GET /{job_id}/requests/{request_id}`

Optional query parameters:

| Parameter | Meaning |
|---|---|
| `max_events` | Event/result-chunk count; server default is 16 and current cap is 512 |
| `cursor` | Retry cursor for cursor-addressed DSSST1 result chunks |

Example:

```json
{
  "request_id": "request-id",
  "status": "done",
  "created_at": "2026-07-20T18:00:00Z",
  "updated_at": "2026-07-20T18:00:01Z",
  "result": {},
  "error": "",
  "events": [],
  "next_cursor": ""
}
```

There are two distinct event-delivery modes:

1. Streaming-generation events are destructively drained by `max_events`.
2. Large DSSST1 results use `result_chunk` events and `next_cursor`.
   `poll_request()` echoes `next_cursor` as `cursor`, validates each chunk hash,
   reassembles the frame, and decodes the result.

`poll_request()` returns `{}` if a successful terminal status has no result. It
raises `RuntimeError` for failed/cancelled status and `TimeoutError` at the
configured deadline.

---

## 7. Generic operation envelope

```text
POST /{job_id}/operation
Content-Type: application/json
```

```json
{
  "operation_type": "weight-sync",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {}
}
```

- `operation_type` is required.
- `sub_job_id` and `sub_job_type` are optional routing hints. If both are
  supplied, they must resolve to the same target.
- `payload` is operation-specific.
- Byte payloads passed to `CortexTrainingClient.forward()` are converted to
  `{"payload_b64": "...", "content_type": "application/octet-stream"}` inside
  the JSON envelope.
- Operation responses are opaque. If a response contains a `request_id`, poll
  it; otherwise consume it inline.

The service accepts these operation types:

| Operation type | Client method | Execution |
|---|---|---|
| `forward` | `forward`, `fwd`, `fwd_no_grad` | Async |
| `weight-sync` | `weight_sync` | Async |
| `bootstrap-router-replay` | `bootstrap_router_replay` | Inline; poll if a deployment returns a request id |
| `router-replay-discard` | `router_replay_discard` | Inline |
| `reset-prefix-cache` | `reset_prefix_cache` | Inline |
| `cancel-request` | `cancel_request` | Inline acknowledgement |
| `tail-logs` | `tail_logs`, `stream_logs` | Inline cursor page |

Only `tail-logs` is allowed while the job is `placing`; all other operations
require `running`.

### 7.1 Forward

```json
{
  "operation_type": "forward",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {
    "payload_b64": "base64-data",
    "content_type": "application/octet-stream"
  }
}
```

`forward()`, `fwd()`, and `fwd_no_grad()` all send exactly
`operation_type="forward"`. The aliases do not add no-gradient semantics.

A byte payload is a DSSST1 frame posted as a JSON `/operation` body. A frame
that fits the JSON ceiling is one POST of the raw frame. A larger frame is
split with `wire.encode_byte_chunks` (`kind="request"`, `operation="forward"`)
and each chunk is its own POST whose `payload_b64` is that request chunk. Only
the last POST returns the pollable `request_id`.

The ceiling is `CORTEX_TRAINING_FORWARD_OPERATION_MAX_JSON_BYTES` (integer
bytes, default 16 MiB, minimum 16 KiB). Dict payloads stay a single plain
`/operation` POST.

Do not treat this route as a portable log-probability API.

### 7.2 Weight sync

```json
{
  "operation_type": "weight-sync",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {
    "source_sub_job_id": "job-id:training:0",
    "target_sub_job_ids": ["job-id:sampling:0"],
    "weight_format": "vllm"
  }
}
```

`weight_sync()` defaults operation routing to the source training sub-job and
returns the `request_id` string.

Optional payload fields:

- `weight_format`: `"vllm"` (default on server) | `"hf"` for full-model
  weights. For adapter-only synchronization, see
  [section 7.3](#73-lora-adapter-sync).

On server versions that include this behavior, the default weight sync, which
pauses generation and keeps in-flight requests, also clears the sampling
sub-job's prefix cache. No request after the sync reads KV computed under the
old weights. In-flight requests are preempted and recomputed with the new
weights, and the next turn of each conversation recomputes its history, so
generation is somewhat slower right after a sync. Syncs that use another mode
are unchanged. A sampling sub-job can opt out with
`vllm_config.clear_cache_on_weight_sync` (see
[section 8.3](#83-inferenceconfig)).

Each sampling target in the result reports `clear_cache`, at
`targets[].recv.recv.clear_cache`. It is the value the sync sent to the sampling
workers' pause, so it shows what was requested and is not proof that the cache
was cleared. It is `false` when no pause was issued. A result without the key
comes from a server version that predates this behavior. If you set the opt-out
and the result still shows `clear_cache: true`, check the key's spelling and
that it is set under `vllm_config`.

### 7.3 LoRA adapter sync

LoRA synchronization uses the same `weight-sync` operation, but it requires
LoRA to be enabled on both sub-jobs when they are created. It is not sufficient
to change only `weight_format` at sync time:

- The training sub-job's `training_config.peft_config` applies PEFT and makes
  the LoRA adapter parameters trainable.
- The sampling sub-job's `inference_config.peft_config` enables the vLLM LoRA
  manager at engine startup.
- The sampling configuration must match the training adapter's `r`,
  `lora_alpha`, and `target_modules`. Reusing the same configuration object for
  both sub-jobs is the simplest way to keep them aligned.

The sync request transfers only the trained adapter tensors:

```json
{
  "operation_type": "weight-sync",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {
    "source_sub_job_id": "job-id:training:0",
    "target_sub_job_ids": ["job-id:sampling:0"],
    "weight_format": "lora"
  }
}
```

Subsequent syncs replace the resident adapter under the same stable adapter
identity.

CLI equivalent:

```bash
cortex-training --job-id JOB_ID weight-sync \
  --weight-format lora
```

See [section 13.5](#135-enable-lora-training-and-adapter-sync) for the complete
create, train, sync, and generate workflow.

### 7.4 Bootstrap router replay

```json
{
  "operation_type": "bootstrap-router-replay",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {
    "source_sub_job_id": "job-id:sampling:0",
    "target_sub_job_id": "job-id:training:0",
    "max_cache_bytes": 4096
  }
}
```

The sampling sub-job is the routing source and the training sub-job is the
replay target/operation receiver. `max_cache_bytes` is optional.
For a mixed training/sampling job, pass `sub_job_id=target_sub_job_id` (or the
matching `sub_job_type`) because the low-level client does not infer the
operation receiver.

### 7.5 Router replay discard

```json
{
  "operation_type": "router-replay-discard",
  "sub_job_id": "job-id:sampling:0",
  "sub_job_type": "sampling",
  "payload": {
    "sample_ids": ["sample-1", "sample-2"]
  }
}
```

`router_replay_discard()` also accepts an `extra` payload for forward-compatible
fields. Explicit `extra["sample_ids"]` takes precedence over the method's
`sample_ids` argument.

### 7.6 Reset prefix cache

```json
{
  "operation_type": "reset-prefix-cache",
  "sub_job_id": "job-id:sampling:0",
  "sub_job_type": "sampling",
  "payload": {
    "drain": true,
    "timeout_s": 60.0,
    "retry_interval_s": 0.1
  }
}
```

The shown payload values are the client defaults. `extra` can supply new
backend fields and override `drain`.

### 7.7 Cancel request

```json
{
  "operation_type": "cancel-request",
  "payload": {
    "request_id": "request-id"
  }
}
```

With a self-describing request id, the server can recover the owning sub-job.
For legacy ids, omitting a sub-job hint causes server-side fan-out across the
job's sub-jobs.

### 7.8 Tail logs

```json
{
  "operation_type": "tail-logs",
  "sub_job_id": "job-id:training:0",
  "payload": {
    "cursor": "cursor-0",
    "max_lines": 50
  }
}
```

Response:

```json
{
  "entries": [
    {"_raw": "2026-01-01T00:00:00.123456789Z first line of output"},
    {"_raw": "2026-01-01T00:00:00.123498765Z {\"ts\": \"...\", \"level\": \"INFO\", \"msg\": \"...\"}"}
  ],
  "next_cursor": "cursor-1",
  "eof": false
}
```

Each entry is `{"_raw": "<time> <line>"}`: the RFC 3339 time (nanosecond
precision, UTC) at which the container runtime received the line, one space,
then the line exactly as the process printed it. A line the process printed as
JSON stays a JSON string inside `_raw`; the saved console from
[section 12.3](#123-persisted-stdout-download) has the same lines without the
leading time. Newer deployments may also add `ts` (that time) and `line` (the
line without it) to each entry; `_raw` is unchanged, and clients should ignore
fields they do not use. A page holds at most 2000 entries or 256 KiB. `cursor` omitted
with `max_lines` set returns the newest `max_lines` lines; `cursor` omitted
without `max_lines` reads from the start.

`eof: true` means the head pod can never log again: it is gone, it has reached
a terminal phase, or it is being deleted and its container has exited.
Deployments from before that change never send it, and report `eof: false` for
as long as the pod exists, so clients must not depend on it. `stream_logs()`
repeatedly calls this operation: with `follow=False` it stops at the first
empty page, once it has caught up with the log; with `follow=True` it keeps
polling. Either way it stops at an empty page that reports `eof: true`, and it
ends without an error when the job ends after the stream has read from it (the
tail then answers 404, 409 or 410).

A finished job's pods are gone, so `tail-logs` answers 404, 409 or 410. The client
then reads the job once and, if it has ended, raises at once with a pointer to
the saved console (`cortex-training download-log JOB_ID --log-type stdout`, or
`download_stdout_logs()`). A `sub_job_id` that is not one of the job's
sub-jobs also fails at once and lists the job's sub-jobs. A job that is still
starting keeps the normal retries.

### 7.9 Unsupported zone-events client helper

`CortexTrainingClient.tail_events()` and `stream_events()` send:

```json
{"operation_type": "zmd-events"}
```

The server does not currently accept this operation type, and the methods are
unit-tested only for request construction. Treat them as unavailable.

---

## 8. Create-job schemas

### 8.1 `SubJobConfig`

Exactly one type-specific config is set.

| Field | Type | Required | Notes |
|---|---|---|---|
| `job_type` | `training`, `sampling`, `log_probability` | yes | Typed `JobType` enum. `log_probability` is a schema value only: the client and CLI reject it at submission |
| `model_name` | string | yes | Must be non-empty |
| `training_config` | object | for training | Produced from `TrainingConfig` |
| `inference_config` | object | for sampling/log probability | Produced from `InferenceConfig` |
| `global_batch_size` | integer | no | Top-level passthrough field |
| `dtype` | string | no | Example: `bfloat16` |
| `seed` | integer | no | |
| `model_post_init` | list of strings | no | Server maps to post-init hooks |
| `source_checkpoint_info` | object | no | Create-time initialization from a saved checkpoint or external weights |

Typed factories:

```python
SubJobConfig.training_job(...)
SubJobConfig.sampling_job(...)
```

The typed client has no `resume_from_checkpoint` field. Use
`source_checkpoint_info`.

### 8.2 `TrainingConfig`

The typed client requires:

| Field | Type | Client validation |
|---|---|---|
| `optimizer` | object | Canonicalized; empty only when `gradient_clipping` is set |
| `max_seq_len` | integer | Greater than zero |
| `train_batch_size` | integer | Greater than zero |
| `n_gpus` | integer | Greater than zero |

#### `optimizer` canonicalization

`optimizer` is no longer an opaque non-empty object. The typed `TrainingConfig`
rewrites it into the canonical flat form on construction and rejects anything it
cannot place. The canonical keys are `name`, `lr`, `weight_decay`, `betas`,
`eps`, and `fused`; the wire form is

```json
{"name": "AdamW", "lr": 1e-5, "weight_decay": 0.0, "betas": [0.9, 0.95], "eps": 1e-8}
```

The rewrites are:

- `type` becomes `name`. Setting both to different values is an error.
- `beta1` / `beta2` become `betas`. Supplying only one fills its partner from
  `[0.9, 0.999]`. `betas` together with a disagreeing `beta1`/`beta2` is an
  error.
- A DeepSpeed-style `params` sub-object is flattened onto the top level. A key
  set in both places with different values is an error.
- `gradient_clipping` inside `optimizer` is hoisted onto the typed
  `gradient_clipping` field, which is where the server reads it. When both are
  set the typed field wins and the nested value is dropped. A block holding only
  `gradient_clipping` therefore canonicalizes to `{}`, which is valid.
- Scheduler names (`lr_scheduler`, `lr_scheduler_type`, `warmup_steps`,
  `warmup_ratio`, `warmup_steps_proportion`, `num_warmup_steps`,
  `training_horizon`) are rejected rather than accepted and ignored. Drive the
  learning rate per step from the client, or set a server-side DeepSpeed
  scheduler under `ds_config`.
- Any other key is rejected, as are non-numeric, non-finite, or out-of-range
  values.

Canonicalization never fills defaults — the server owns those, so an explicitly
submitted value stays distinguishable from a defaulted one. `betas` is the one
exception, since its canonical form is a pair.

This runs on the typed `TrainingConfig` path only. `create_job_from_body()`
forwards `optimizer` exactly as given and leaves the rewrite to the server.

Optional typed fields:

- `gradient_clipping`
- `multiplex_job_id`
- `load_optimizer_states`

`load_optimizer_states` controls checkpoint resume behavior for this training
sub-job. When `false`, resuming a `resumable` checkpoint (via
`source_checkpoint_info` or the runtime `/load` endpoint) restores model weights
only and starts the optimizer fresh. Set it `false` to change data-parallel size
while keeping expert parallelism, because the DP-sharded optimizer cannot be
resized; a `true`/default resume against a different DP/world size fails fast
with an actionable error. `None` (omitted) uses the server default (`true`).

`extra_training` is merged as open passthrough data, without overriding typed
keys. `ds_config.zero_optimization.stage` must be 0, 1, or 2: ZeRO-3 is not
supported and is rejected before the request is sent; for CPU optimizer offload
use stage 2 with `zero_optimization.offload_optimizer`. Examples include
`model_provider`, `ep_size`, `ds_config`,
`activation_checkpointing`, `prime_rl` (deprecated, see
[below](#prime_rl-block-deprecated)), `router_replay`, `peft_config`,
`model_options` (see [below](#model_options)), the memory-diagnostics settings
in [section 8.5](#85-memory-diagnostics-settings), and these newer
long-context/memory knobs.

`model_provider` `"arctic"` is the name the shipped MoE configs and this
document use, but the server accepts that provider only as `"prime_rl"` for
now. The Python client and `cortex-training submit` send a training
`model_provider` of `"arctic"` as `"prime_rl"`; a request sent without the
client must use `"prime_rl"`.

The long-context/memory knobs:

- `sp_size`: Ulysses sequence-parallel degree, sharding each sample's sequence
  across `sp_size` ranks. `1` (default) disables it; the server requires
  `world_size % sp_size == 0`.
- `mb_spec`: `{"max_tokens_per_mb": N}`, the most real tokens whose rows are
  packed into one model call (default `10240`). It counts whole rows before
  sequence parallelism splits them, and a row longer than N runs in a call of
  its own.
- `cuda_allocator_conf`: per-job `PYTORCH_CUDA_ALLOC_CONF` string applied before
  PyTorch is imported, e.g. `"expandable_segments:True"` (or `"backend:native"`
  to disable). Omitted uses the server default.
- `ac_config`: activation-checkpointing config, including CPU activation offload
  via `offload_config.enabled=true`. Offload requires `mode="full"`.
- `router_replay`: `{"enabled", "max_cache_bytes", "mode"}`. `mode` is
  `"strict"` (default) or `"best_effort"`; values other than `"strict"` and
  `"best_effort"` are rejected. It applies to the training sub-job
  only: `"best_effort"` with `enabled` `false`, or any `mode` other than
  `"strict"` on a sampling sub-job, is rejected. `"strict"` fails a forward/backward
  request when any of its rows has no routing held by a sampling worker (for
  example, a row generated before a sampler restart); the request error is
  `router_replay_preflight_failed` with `stage` `"receive"` and `reason`
  `"receiver_error"`. `"best_effort"` routes such rows with the trainer's own MoE
  gate, replays the others, and reports the split in the
  `router_replay/rows_*` and `router_replay/tokens_*` metrics of
  [section 6.1](#61-forwardbackward---post-job_idforward-backward). A model
  whose MoE routers cannot route a row by their own gate, or that has no MoE
  routers, rejects `"best_effort"` when the training sub-job starts. With whole-block activation
  checkpointing, `"best_effort"` also requires `ac_config.router_replay_recompute`
  (default `true`); a training sub-job with it disabled rejects `"best_effort"`
  at startup. If the service does not yet support `"best_effort"`, router-replay
  bootstrap fails with a `router replay bootstrap rejected` error; use `"strict"`
  until the service is upgraded. If only some sampling workers lack support, a
  request with a missing row fails exactly as in `"strict"`.

`max_cache_bytes` is the per-worker cap in bytes. The service stores each
expert id as `uint8` when the model has at most 256 experts and as `uint16`
when it has 257 to 65536 experts. A `uint16` capture uses twice as many bytes
as a `uint8` capture for the same sequence length, MoE layer count, and top-k.
The service and the client do not choose the cap: set
`router_replay.max_cache_bytes` in the job body for each sub-job (on a sampling
sub-job built with the SDK, through `extra_sampling["router_replay"]`). The
repository's RL example fills in `5368709120` (5 GiB) for training and
`21474836480` (20 GiB) for sampling on MoE models, and describes an optional
planner that sizes a cap from the expert count, sequence length, MoE layer
count, and top-k; see
[Router-replay capacity planning](../../recipes/rl_quickstart/README.md#router-replay-capacity-planning).

`extra_training["model_provider"] = "dflash2"` trains a DFlash2
speculative-decoding draft against a frozen target. Its options and rules are in
[section 8.6](#86-dflash2-draft-training-and-serving).

For LoRA training, set `extra_training["peft_config"]` to a PEFT
`LoraConfig`-compatible object. It is validated and rewritten client-side, so it
must specify `peft_type="Lora"` — exactly that spelling, not the `"LORA"` a
Hugging Face `adapter_config.json` writes — plus `target_modules`,
`target_parameters`, or both. `r`, `lora_alpha`, `lora_dropout`, and `bias` are
filled with their defaults (`8`, `8`, `0.0`, `"none"`) by the client when
omitted, although explicit values are recommended when the same configuration is
also used by sampling. Unsupported keys are rejected.

`peft_config` is validated the same way on all three submission paths: typed
training, typed sampling ([section 8.3](#83-inferenceconfig)), and the raw
`create_job_from_body()` body. A config with no `peft_config` stays valid, and
every other key in `extra_training` / `extra_sampling` remains open
passthrough.

The current client also rejects an effective PrimeRL config that combines an
enabled/default `fused_cross_entropy` with `fp32_lm_head=True` or an integer
`fused_lm_head_token_chunk_size`. This check runs on the typed
`TrainingConfig` path and reads only those fields, at the top level and in the
`prime_rl` block, never `model_options`. The client is stricter than the
server, for example: it rejects an integer top-level
`fused_lm_head_token_chunk_size` under the default fused cross-entropy on any
job, where the server accepts it when `ep_size` is 1 or omitted; and the server
accepts `fp32_lm_head=true` with fused cross-entropy `true`, `false` or
`liger`, as it accepts `lm_head.fp32`, while the client rejects the field with
an enabled or default fused cross-entropy. To pass the client's check, set
`fused_cross_entropy` to `false` in the same place (top level or the
`prime_rl` block) as the chunk size or `fp32_lm_head`, or, on server versions
that support `model_options`, use it: the client does not check it. A
top-level `fused_cross_entropy` is read
only by the client. In the `prime_rl` block the server also reads
`fused_cross_entropy`: with `ep_size` > 1, `false` there turns the fused head
off. With `ep_size` > 1 the model reads the fp32 head and the chunk size
only from the `prime_rl` block entries, which `lm_head.fp32` and
`lm_head.token_chunk_size` also write; a top-level `fp32_lm_head` alone has no
effect there. With `ep_size` > 1 a top-level `fused_lm_head_token_chunk_size`
alone does not install the chunked head in the model, but the server still
switches the job to chunked log-probability handling, so do not send it alone
there: use `lm_head.token_chunk_size` with `lm_head.cross_entropy` set to
`false` (on server versions without `model_options`,
`prime_rl.fused_lm_head_token_chunk_size` with `prime_rl.fused_cross_entropy`
set to `false`). With `ep_size` 1 or omitted the top-level field installs the
chunked head by itself.

#### `model_options`

This subsection and the next document the API for server versions that
support `model_options`. Older server versions silently ignore the map: the
job runs with its other settings and the options do not apply. See the end of
[the `prime_rl` block](#prime_rl-block-deprecated) for what to keep sending.

`model_options` names each model option once, by one key, instead of through
the top-level fields, the dict fields and the `prime_rl` block above. Send it
in `extra_training` (and, for a sampling sub-job, in `extra_sampling`, see
[section 8.3](#model_options-on-a-sampling-sub-job)). The server checks the
map, writes each option onto the existing field or fields listed in the last
column below, and, when the sub-job starts, checks the keys against what the
model implementation supports. Apart from the `seq_len` default described
[below](#prime_rl-block-deprecated) and the `numerics.fp32_parity` default
(on; see the notes on single keys), this does not change a sub-job that does
not send `model_options`.

The client does not validate `model_options`, on the typed path or on the raw
`create_job_from_body()` path, so `cortex-training submit --dry-run` does not
catch the errors described below.

For a dense model with `ep_size` omitted:

```python
training = SubJobConfig.training_job(
    model_name="Qwen/Qwen3-8B",
    optimizer={"name": "AdamW", "lr": 1e-5},
    max_seq_len=8192,
    train_batch_size=8,
    n_gpus=8,
    extra_training={
        "model_options": {
            "lm_head": {"fp32": True, "token_chunk_size": 4096},
            "attention": {"backend": "flash_attention_3"},
        },
    },
)
```

With `ep_size` > 1, see the chunked LM head note below.

**Shape.** The map nests by key component: `{"lm_head": {"fp32": true}}` sets
`lm_head.fp32`, and the dotted form `{"lm_head.fp32": true}` names the same key.
Naming one key twice is an error. Nesting stops at a key, so the value of a key
is taken as given. `liger` is a single-word key that takes `true` or `false`. A
group of keys, such as `checkpointing` or `lm_head`, cannot be set to `true`,
`false` or any other value as a whole; set one of its keys instead. An absent
key leaves the existing field at its value or default.

**Keys.** A training sub-job accepts these 20 keys. The Runs in column says
where a key applies: `trainer` keys are accepted only in a training sub-job, and
`both` keys in either sub-job. The server checks the type of each value, and
for `numerics.optimization_dtype` the allowed set; for every other key the
model implementation checks the value when the sub-job starts. Where only some
values work, the Value column says so: an even count, or the names that work.

| Key | Runs in | Value | Sets |
|---|---|---|---|
| `checkpointing.mode` | trainer | non-empty string (`full`, `selective`) | `ac_config.mode` |
| `checkpointing.freq` | trainer | positive integer | `ac_config.freq` and `activation_checkpointing` |
| `checkpointing.targets` | trainer | list of strings (target names the model's layers support, such as `norm`) | `ac_config.targets`; used only with `checkpointing.mode` `selective` |
| `checkpointing.offload` | trainer | `true` or `false` | `ac_config.offload_config.enabled`; `false` sets nothing when no `ac_config` block results (see below) |
| `tiled_mlp.token_chunk_size` | trainer | positive integer | `tiled_mlp_token_chunk_size` |
| `lm_head.fp32` | both | `true` or `false` | `fp32_lm_head`; also `prime_rl.fp32_lm_head` when `ep_size` > 1 |
| `lm_head.token_chunk_size` | trainer | positive integer | `fused_lm_head_token_chunk_size`; also `prime_rl.fused_lm_head_token_chunk_size` when `ep_size` > 1 |
| `lm_head.vocab_chunk_size` | trainer | positive integer; takes effect only with `lm_head.token_chunk_size` | `fused_lm_head_vocab_chunk_size` |
| `lm_head.cross_entropy` | trainer | `true`, `false` or `liger` | `prime_rl.fused_cross_entropy` |
| `liger` | trainer | `true` or `false` | `true` applies Liger kernels on a huggingface model; `false` sets nothing |
| `moe.grouped_mm` | trainer | `true` or `false` | `prime_rl.moe_use_grouped_mm` |
| `moe.comm_backend` | trainer | non-empty string (`deepep`, `uccl`) | `prime_rl.ep_comm_backend` |
| `moe.comm_sms` | trainer | even positive integer | `prime_rl.deepep_num_sms` |
| `moe.comm_token_chunk` | trainer | positive integer | `prime_rl.deepep_token_chunk_size` |
| `attention.backend` | trainer | non-empty string (see below) | `attn_implementation` |
| `attention.sparse_mla` | trainer | not available for any model in the catalog that supports training | `prime_rl.sparse_mla_backend` |
| `compile.fullgraph` | trainer | `true` or `false` | `compile.fullgraph` |
| `numerics.fp32_parity` | trainer | `true` or `false` (default `true`) | `prime_rl.fp32_parity` when `ep_size` > 1; the top-level field `fp32_parity` otherwise |
| `numerics.fp32_residual` | trainer | `true` or `false` (experimental, default `false`) | the top-level field `fp32_residual` |
| `numerics.optimization_dtype` | trainer | `bfloat16` or `float32` | `prime_rl.optimization_dtype` when `ep_size` > 1; nothing otherwise |

An omitted `ep_size` counts as 1 throughout this section. Which keys fail when
the sub-job starts depends on the model and on `ep_size`:

- With `ep_size` 1 or omitted these keys fail at startup:
  `checkpointing.mode`, `checkpointing.targets`, `lm_head.cross_entropy`,
  `moe.grouped_mm`, `moe.comm_backend`, `moe.comm_sms`,
  `moe.comm_token_chunk`, `attention.sparse_mla` and
  `numerics.optimization_dtype`.
- With `ep_size` > 1, on models whose Hugging Face `model_type` is
  `qwen3_5_moe` (or `qwen3_5_moe_text`) (in the
  [model catalog](../../model-catalog/README.md), Qwen3.6-35B-A3B), these fail
  at startup: `compile.fullgraph`, `lm_head.vocab_chunk_size`,
  `attention.sparse_mla` and `numerics.fp32_residual`. `model_options.liger`
  set to `true` applies to huggingface models. `ep_size` above 1 needs
  `model_provider` `"arctic"`, so that combination is rejected before the model loads.
- With `ep_size` > 1 a model that is not a supported MoE model fails.

Other models or `ep_size` combinations may fail the job at startup.

Notes on single keys:

- `checkpointing.offload` takes only `true` or `false`. The other offload
  settings (`keep_last_n` and the rest) stay in `ac_config.offload_config`, and
  the two combine; the `ac_config` rule above, that offload requires
  `mode="full"`, still applies. `false` writes nothing when the job sends no
  `ac_config` or `prime_rl.ac_config` block and sets none of
  `checkpointing.mode`, `.freq` or `.targets` in the same map, because with
  `ep_size` > 1 any `ac_config` block turns activation checkpointing on; when
  one of those keys is set it creates the block, and `false` then writes
  `ac_config.offload_config.enabled: false`. It is still rejected against an
  explicit `ac_config.offload_config.enabled: true`.
- When the job also sends a `prime_rl.ac_config` block, which replaces
  `ac_config` for `ep_size` > 1, the `checkpointing.*` keys are written into
  that block too.
- `compile.fullgraph` set to either value turns per-layer `torch.compile` on,
  because the presence of `compile` enables it; `true` also requires each layer
  to compile as one full graph. `tiled_mlp.token_chunk_size` (or
  `tiled_mlp_token_chunk_size`) still cannot be combined with
  `compile.fullgraph=true`.
- Chunked LM head: with `ep_size` > 1, `lm_head.token_chunk_size` needs fused
  cross-entropy off. Set `lm_head.cross_entropy` to `false` (or send
  `prime_rl.fused_cross_entropy` as `false`). Fused cross-entropy defaults to
  `liger` there, and the server rejects the pair. It also rejects
  `lm_head.cross_entropy` set to `true` or `liger` together with an integer
  block `prime_rl.fused_lm_head_token_chunk_size`. `lm_head.fp32` combines
  with fused cross-entropy `true`, `false` or `liger`.
- Fractional token weights: a plain SFT job whose `loss_mask` has values other
  than 0 and 1 needs non-fused cross-entropy and, on every route, the chunked LM
  head off, because the loss then needs the full logits. Leave
  `lm_head.token_chunk_size` and `fused_lm_head_token_chunk_size` unset (with
  `ep_size` > 1 also `prime_rl.fused_lm_head_token_chunk_size`). To turn fused
  cross-entropy off, set `lm_head.cross_entropy` to `false` with `ep_size` > 1,
  or the block entry `prime_rl.fused_cross_entropy` to `false` with
  `model_provider` `arctic` and `ep_size` 1 or omitted; turn off
  `model_options.liger`.
- On server versions that have it, `numerics.fp32_parity` is on by default
  (`true`), so a job that sends no `model_options` runs it, and its
  log-probabilities and loss differ from those of a server version that predates
  the option. It applies to Qwen3.5-MoE models with `ep_size` > 1 and to dense
  Qwen3.5-family models (Hugging Face `model_type` `qwen3_5` or `qwen3_5_text`,
  for example Qwen3.8-27B) with `ep_size` 1 or omitted. With `true` the residual
  adds, partial rotary embedding and GatedDeltaNet beta gate are evaluated in
  fp32, as vLLM does. With `false` they are evaluated in the input dtype, as
  before the option existed, so the trainer's log-probabilities can differ more
  from the sampler's. Other dense models accept the key and are not changed by
  it, except that with sequence parallelism the GatedDeltaNet beta gate of any
  model follows it (so with the default it is evaluated in fp32 there too). With
  `ep_size` > 1 a model that is not Qwen3.5-MoE rejects the key when the sub-job
  starts. To turn the path off, send `"model_options": {"numerics":
  {"fp32_parity": false}}`. A server version that has the option but not
  `model_options` support ignores that map: a dense job cannot turn the path off
  there, and an `ep_size` > 1 job sends `prime_rl.fp32_parity: false`.
- Memory of `numerics.fp32_parity`: with full activation checkpointing (every
  layer: `activation_checkpointing` with `ep_size` 1 or omitted, an `ac_config`
  block of mode `full` and `freq` 1 with `ep_size` > 1), every decoder layer
  after the first holds its residual at the layer boundary as a second tensor,
  so GPU memory rises. For Qwen3.8-27B with rows of 40000 tokens, `sp_size` 4
  and `mb_spec.max_tokens_per_mb` 131072 (calls of about 120000 tokens), peak
  allocated memory was about 92 GiB per GPU with the option set to `false` and
  104 GiB with it on (the default), on GPUs with 139.8 GiB. The device uses more
  than the allocator reports (12 to 15 GiB more for one 262144-token row at
  `sp_size` 8), so leave headroom. With `sp_size` 4 and a budget of 262144
  (calls of about 240000 tokens, 60000 per GPU) the job with the option on ran
  out of memory; the same job with it off was not measured. Memory follows,
  approximately, the tokens each GPU holds in a call, which is the call's tokens
  divided by `sp_size`. `mb_spec.max_tokens_per_mb` (default `10240`, see the
  `extra_training` knobs above) limits how many rows share a call, but a row
  longer than it runs in a call of its own, so for such rows lowering the budget
  does not help: a larger `sp_size` (at most 8 for Qwen3.8), or turning the
  option off, lowers the memory. On a dense Qwen3.5-family model with `liger`
  on, of the decoder-layer and final norms only the first layer's input norm
  still runs Liger's RMSNorm kernel while this option is on.
- `numerics.fp32_residual` (experimental, default `false`) also keeps the
  residual stream in fp32 between decoder layers, instead of rounding it to the
  dtype of each layer's output. It applies only to dense Qwen3.5-family models
  with `ep_size` 1 or omitted; other dense models accept the key and are not
  changed by it, and with `ep_size` > 1 it fails at startup. `true` is rejected,
  when the sub-job starts, if `numerics.fp32_parity` is `false`. It needs more
  memory again (about 122 GiB per GPU allocated in the example above) and,
  together with `numerics.fp32_parity`, ran forward/backward about 6% slower on
  that model than a run on a different server build, so treat the figure as
  approximate.
- The Sets column names top-level fields for `numerics.fp32_parity` (with
  `ep_size` 1 or omitted) and `numerics.fp32_residual`. They carry the key to
  the dense model: send the key, not the field. A field sent without its key
  skips the startup check, and with `ep_size` > 1 it has no effect.
- `numerics.optimization_dtype` is the dtype the trainer keeps and optimizes
  the weights in. Qwen3.5-MoE models with `ep_size` > 1 support it, with
  default `bfloat16`, and with `ep_size` > 1 the sub-job's `dtype` is not
  used. With `ep_size` 1 the sub-job's `dtype` is used, and the key fails at
  startup.
- `attn_impl` (as in the shipped examples), `attn_implementation` (default
  `flash_attention_3`) and `prime_rl.attn`: all that are sent together with
  `attention.backend` must agree. `attn_impl`, when sent, takes precedence over
  `attn_implementation`, and `prime_rl.attn`, when sent, must equal that
  resolved value (`attn_implementation` defaults to `flash_attention_3`), or
  the job fails when it starts.
- Qwen3.5-MoE models with `ep_size` > 1 take `sdpa` (and `eager`, which they
  run as `sdpa`), `flash_attention_3` and `flash_attention_4` as the attention
  backend.

**Not accepted.** Any key that is not in the table is rejected, including
`peft` (LoRA is a training mode; set `peft_config` instead) and
`numerics.reduce_dtype` (gradient reduction always runs in fp32).

**A field and its key together.** A sub-job may send an existing field and the
`model_options` key that sets it only with the same value. Equality is plain
value equality, so `true` and `1` count as equal. The field
`activation_checkpointing` may also be sent as `activation_checkpointing_freq`,
`gradient_checkpointing_freq` or `gradient_checkpointing`; when several are
sent, the first in the order `activation_checkpointing`,
`activation_checkpointing_freq`, `gradient_checkpointing_freq`,
`gradient_checkpointing` counts. So `gradient_checkpointing=true` with
`checkpointing.freq=1` is accepted. Different values are rejected.
`liger=false` is rejected when the same training config already applies Liger kernels.
With `ep_size` above 1 the `fp32_parity` field is not compared: the key writes
`prime_rl.fp32_parity`, and the key is compared with that entry. The shipped
dense training configs send `model_provider` `huggingface`, and the Qwen3.5 and
Qwen3.8 ones a `fused_lm_head_token_chunk_size` of 8192, so `liger: true`, or a
`lm_head.token_chunk_size` other than 8192, is rejected as a conflict until you
change or remove that field.

**Each sub-job on its own.** The server validates each sub-job's map alone and
never copies a key from one sub-job to another. A key that runs on both sides,
such as `lm_head.fp32`, must be set in every sub-job that needs it: in the
training sub-job for the trainer's head and in the sampling sub-job for the
sampler's.

**Invalid maps.** The rejected inputs include:

- a `model_options` that is not an object;
- a key that is not in the table, including `peft` and
  `numerics.reduce_dtype`;
- a key in the wrong sub-job;
- a group set as a whole;
- a value of the wrong type, such as a `tiled_mlp.token_chunk_size` that is not
  a positive integer;
- a field and its key with different values;
- a field that a key writes into and that holds something other than an
  object, such as `prime_rl` set to a number together with a `moe.*` key;
- the chunked LM head combinations described in the notes above.

The client does not catch these inputs. A rejected input may be refused when
you create the job, or the job may be accepted and later end `failed`
([section 3.2](#32-job-states)); the server's message may not be returned to
you.

**Checks when the sub-job starts.** When a training sub-job starts, it checks
each key in `model_options` against what the model implementation supports,
as listed above. The check looks at which keys are present, not at their
values, with one exception: `liger` set to `false` asks for nothing and is
skipped. So `compile.fullgraph: false` is still checked, because it turns
per-layer compile on, and fails where `compile.fullgraph` is not supported. A
key that is not supported fails the sub-job, as do other startup failures such
as a value the model implementation rejects. The job ends `failed`
([section 3.2](#32-job-states)), and its `reason` need not identify the
option. To find the cause, compare the keys you sent against the table above
and the `ep_size` lists after it.

#### `prime_rl` block (deprecated)

`training_config.prime_rl` is deprecated in favor of `model_options`. It keeps
working: the server still accepts the block and reads it as before, apart from
the `seq_len` default described below and the `numerics.fp32_parity` default
(see the notes on single keys), which applies whether or not a job sends the
block. The table gives the entries that have a
key, and a few that the server sets, ignores or rejects. An entry that it does
not list, such as `trust_remote_code`, has no key and stays in the block.

| Block entry | Replacement |
|---|---|
| `ac_config` | `checkpointing.mode`, `.freq`, `.targets`, `.offload`; `router_replay_recompute` and the other `offload_config` fields have no key and stay in the top-level `ac_config` |
| `ep_comm_backend` | `moe.comm_backend` |
| `deepep_num_sms` | `moe.comm_sms` |
| `deepep_token_chunk_size` | `moe.comm_token_chunk` |
| `moe_use_grouped_mm` | `moe.grouped_mm` |
| `fused_cross_entropy` | `lm_head.cross_entropy` with `ep_size` > 1; no key with `ep_size` 1 or omitted (see below) |
| `fused_lm_head_token_chunk_size` | an integer: `lm_head.token_chunk_size`; `"disabled"` or `"auto"` mean no chunked head: omit the key |
| `fp32_lm_head` | `lm_head.fp32` |
| `fp32_parity` | `numerics.fp32_parity` with `ep_size` > 1. Only Qwen3.5-MoE models on server versions that have the option read this entry; a server version without the option, or another model, fails an `ep_size` > 1 job that carries it when the sub-job starts. With `ep_size` 1 or omitted the entry has no effect, while the key does |
| `sparse_mla_backend` | `attention.sparse_mla`, which no model in the catalog that supports training takes |
| `attn` | `attention.backend` |
| `optimization_dtype` | `numerics.optimization_dtype` with `ep_size` > 1; with `ep_size` 1 or omitted the entry is ignored |
| `seq_len` | on server versions with this behavior, set by the server from `max_seq_len` with `ep_size` > 1 (below); no effect with `ep_size` 1 or omitted |
| `reduce_dtype` | rejected: gradient reduction always runs in fp32 |
| `tiled_mlp_token_chunk_size` | rejected in the block: use `tiled_mlp.token_chunk_size` |

With `ep_size` 1 or omitted, `lm_head.cross_entropy` fails at startup and the
block entry `fused_cross_entropy` is the only spelling: with
`model_provider="arctic"` it decides whether the server refuses
forward/backward requests whose loss needs per-token log-probabilities, and
with `"huggingface"` it has no effect. `model_options.liger` is separate: it
applies Liger kernels on a huggingface model.

With `ep_size` 1 or omitted, the block entries `fp32_lm_head`,
`fused_lm_head_token_chunk_size`, `ac_config` and `fp32_parity` have no effect,
while their keys (`lm_head.fp32`, `lm_head.token_chunk_size`,
`checkpointing.freq`, `checkpointing.offload`, `numerics.fp32_parity`) take
effect, so do not translate such an entry mechanically: for example, a block
`fused_lm_head_token_chunk_size` of 4096 does nothing on such a job, while
`lm_head.token_chunk_size` of 4096 turns the chunked LM head on, and a block
`fp32_parity` of `false` does nothing, while `numerics.fp32_parity` set to
`false` turns the fp32 path off. Conversely, the keys for the block entries
`ep_comm_backend`, `deepep_*`, `moe_use_grouped_mm`, `sparse_mla_backend`,
`fused_cross_entropy` and `optimization_dtype`, and for the `mode` and
`targets` parts of `ac_config`, fail a dense job when it starts, so do not
translate those entries for one.

**`seq_len`.** On server versions with this behavior, with `ep_size` > 1 the
server sets the sequence length the model is built for from the sub-job's
`max_seq_len`, and a `prime_rl.seq_len` that is still sent is used instead.
Older server versions leave that length at its default of 4096 unless the
block sends `seq_len`.

A client that may talk to a server version without `model_options` support
keeps sending, with the same values, the fields that the Sets column of the
key table names (and `vllm_config.fp32_lm_head` for sampling). With
`ep_size` > 1 those include the `prime_rl` block entries, sent as before, and
`prime_rl.seq_len` equal to `max_seq_len`; with `ep_size` 1 or omitted, it
sends the top-level field where the Sets column names one, except
`fused_cross_entropy`, which stays in the block. `numerics.fp32_residual` has
no earlier spelling. `numerics.fp32_parity` has one only with `ep_size` > 1,
the block entry `prime_rl.fp32_parity`; a dense job has none, and the
top-level fields named for the two keys in the Sets column do not exist on
such a server, which ignores them. Send
`prime_rl.fp32_parity` only to a server version that has the option and for a
Qwen3.5-MoE model: any other server or model fails an `ep_size` > 1 job that
carries it when the sub-job starts. A server version that supports
`model_options` but has fewer options than the table above accepts only the
keys it knows and rejects the others, such as the `numerics` keys, as unknown
keys.

### 8.3 `InferenceConfig`

The typed client requires:

| Field | Type | Client validation |
|---|---|---|
| `max_seq_len` | integer | Greater than zero |
| `n_gpus` | integer | Greater than zero |

`multiplex_job_id` is optional. `extra_sampling` is an open passthrough object;
common values include `gpu_memory_utilization`, a nested `vllm_config`, and
`model_options` (see [below](#model_options-on-a-sampling-sub-job)).

Enable speculative decoding on a sampling sub-job with
`vllm_config.speculative_config`. The client forwards that block to the
server. Typical keys are `method`, `model` (the draft model), and
`num_speculative_tokens`.

`speculative_config.model` uses the same convention as `model_name`: pass an
`<org>/<model>` Hub id. The server maps it to the baked model cache.
Explicit local paths are used as-is. Callers should not hard-code cache
locations.

```python
sampling = SubJobConfig.sampling_job(
    model_name="Qwen/Qwen3.8-27B",
    max_seq_len=16384,
    n_gpus=8,
    extra_sampling={
        "vllm_config": {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "speculative_config": {
                "method": "dflash",
                "model": "z-lab/Qwen3.8-27B-DFlash2",
                "num_speculative_tokens": 7,
            },
        },
    },
)
```

To serve a draft trained with the `dflash2` provider, set
`speculative_config.from_source_checkpoint` instead of `model`; see
[section 8.6](#86-dflash2-draft-training-and-serving).

For LoRA sampling, set `extra_sampling["peft_config"]` when creating the
sub-job. This enables the vLLM LoRA manager before the model starts. The
adapter's `r`, `lora_alpha`, and `target_modules` must match the training
configuration used for adapter synchronization. It is validated and rewritten by
the same rules as the training block (see
[section 8.2](#82-trainingconfig)).

`vllm_config.clear_cache_on_weight_sync` controls whether a weight sync clears
the sampling sub-job's prefix cache. It is a boolean, default `true`, and
applies only to the default weight sync (see
[section 7.2](#72-weight-sync)). Set it to `false` to opt out, for example
`extra_sampling={"vllm_config": {"clear_cache_on_weight_sync": False}}`. With
`false`, only a best-effort reset after the sync remains. That reset cannot clear
blocks held by running requests, so requests may read KV computed under the old
weights. The key is not validated: a misspelled key is silently ignored, and the
sync keeps clearing the cache. The `clear_cache` key in each sampling target's
sync result shows whether the opt-out was applied (see
[section 7.2](#72-weight-sync)).

For either config type, the server requires `multiplex_job_id` to be a complete
`{job_id}:{sub_job_type}:{index}` id outside the job being created.

#### `model_options` on a sampling sub-job

On the server versions that [section 8.2](#model_options) describes, a sampling
sub-job takes `model_options` in `extra_sampling`, with the shape, conflict and
error rules of that section. It accepts only the keys that run on the
sampler:

| Key | Runs in | Value | Sets |
|---|---|---|---|
| `lm_head.fp32` | both | `true` or `false` | `vllm_config.fp32_lm_head` |

Every other key in the section 8.2 table runs only in the trainer and is
rejected, as are the keys under "Not accepted" in
[section 8.2](#model_options). A rejected input may be refused when you create
the job, or the job may be accepted and later end `failed` ([section
3.2](#32-job-states)); the server's message may not be returned to you. Nothing
is copied from the training sub-job, so a job that wants the fp32 head on both
sides sets `lm_head.fp32` in both sub-jobs:

```python
sampling = SubJobConfig.sampling_job(
    model_name="Qwen/Qwen3-8B",
    max_seq_len=8192,
    n_gpus=2,
    extra_sampling={"model_options": {"lm_head": {"fp32": True}}},
)
```

### 8.4 `source_checkpoint_info`

```json
{
  "checkpoint_id": "checkpoint-id",
  "source_job_id": "source-job-id"
}
```

Exactly one source is required:

- `checkpoint_id` identifies a checkpoint saved by Cortex Training.
  `source_job_id` identifies the job that owns it and is required for cross-job
  initialization.
- `external_stage_path` identifies a complete Hugging Face safetensors
  directory in a stage the submitting role can read. It cannot be combined with
  `checkpoint_id` or `source_job_id`.

```json
{
  "external_stage_path": "@MY_DB.MY_SCHEMA.MODEL_IMPORT/qwen-sft-run7"
}
```

The typed client forwards this object without validating its fields. The server
resolves stage credentials; clients must not provide `stage_info` themselves.

### 8.5 Memory-diagnostics settings

Training workers expose low-volume, customer-facing memory observability through
`training_config` keys. The typed client has no dedicated fields for these, so
pass them through `extra_training`. Each is an optional per-job override; when
omitted or `null`, the server system default applies.

| Key | Type | Effect |
|---|---|---|
| `step_peak_memory_log` | bool | Adds a `peak_memory` object (`gpu_peak`, `cpu_peak` byte counts) to each `/step` result. See [section 6.2](#62-optimizer-step---post-job_idstep). |
| `training_memory_telemetry` | bool | Emits structured allocator events (`fwd_bwd_end`, `step_start`, `step_end`) to the training event stream. |

```python
training = SubJobConfig.training_job(
    model_name="Qwen/Qwen3-1.7B",
    optimizer={"name": "AdamW", "lr": 1e-4},
    max_seq_len=2048,
    train_batch_size=8,
    n_gpus=8,
    extra_training={"step_peak_memory_log": True},
)
```

`step_peak_memory_log` costs one `all_reduce(MAX)` per step; both settings are
off by default. Failure snapshots on a failed `/forward-backward` or `/step` are
independent of these flags. Per-rank operator diagnostics configured by the
launcher are out of scope for this client-facing surface.

### 8.6 DFlash2 draft training and serving

A DFlash2 draft is a speculative-decoding draft model trained against a frozen
target. It spans the training sub-job ([section 8.2](#82-trainingconfig)), the
checkpoint it saves ([section 8.4](#84-source_checkpoint_info)), and a sampling
sub-job that serves it as the speculator ([section 8.3](#83-inferenceconfig)).

#### Training a draft

`extra_training["model_provider"]` accepts `huggingface`, `arctic` ([section
8.2](#82-trainingconfig) explains its spelling), and `dflash2`. With `dflash2`,
`model_name` is the frozen target, and the draft is configured by
`extra_training["dflash2"]`.

`extra_training["dflash2"]` is required when the provider is `dflash2` and is
rejected for every other provider. Unknown keys are rejected.

| Key | Default | Meaning |
|---|---|---|
| `draft_config` | required | Contents of the draft's `config.json` (`architectures` and `dflash_config`); always required, even with `init_from` |
| `init_from` | none | Optional existing draft directory (with `model.safetensors`) to warm-start the weights from; the architecture still comes from `draft_config` |
| `attention_backend` | `sdpa` | `sdpa`, `flex_attention`, or `eager` |
| `num_anchors` | `512` | Anchor blocks sampled per sequence; greater than zero |
| `loss_decay_gamma` | `7.0` | Per-slot loss decay; greater than zero, or `null` to disable |
| `selector_loss_alpha` | `1.0` | Weight of the selector loss; zero or greater |
| `selector_stop_gradient` | `false` | Detach the selector's inputs so only its own parameters train on its loss |
| `objective_chunk_blocks` | `128` | Blocks per vocabulary-projection chunk; bounds memory |
| `seed` | `0` | Seed for draft initialization and anchor sampling |

A `dflash2` job is rejected when any of these hold:

- `activation_checkpointing` is anything other than `false`. The draft
  recomputes its own forward, so the default of `true` must be overridden.
- `fp32_parity`, `fp32_residual`, or `text_config_overrides` is set.
- `ds_config.bf16.bf16_master_weights_and_grads` or
  `ds_config.bf16.bf16_optimizer_states` is enabled. The service keeps master
  weights and optimizer states in fp32 for this provider, whatever the job
  asks for, so these two flags cannot turn that off. Copying a `ds_config` from
  another job that sets either flag fails.
- A generic model patch, such as Liger kernels, is requested. Patches are
  rejected, not ignored.
- `sp_size` or `ep_size` is greater than one. The provider supports data
  parallelism only, so both must be `1` or omitted.

Requests use `processing.loss_fn="dflash2"`. It reads `loss_mask`, the usual
prediction-aligned supervision mask, and does not use vocabulary `action_masks` or
`compute_logprobs`. Its ratio metrics are returned as additive
`loss_term_<name>__num` and `loss_term_<name>__den` pairs; divide them to get exact
pooled ratios.

```python
training = SubJobConfig.training_job(
    model_name="Qwen/Qwen3-8B",  # the frozen target
    optimizer={"name": "AdamW", "lr": 1e-4},
    max_seq_len=2048,
    train_batch_size=8,
    n_gpus=8,
    extra_training={
        "model_provider": "dflash2",
        "activation_checkpointing": False,
        "dflash2": {"draft_config": draft_config},  # contents of the draft's config.json
    },
)
```

#### Serving a trained draft

To serve a draft trained with the `dflash2` provider, set
`vllm_config.speculative_config.from_source_checkpoint` to `true` instead of
`model`, and name the training job's weights-only checkpoint in
`source_checkpoint_info` (`checkpoint_id` and `source_job_id`). The checkpoint
becomes the draft model, `model_name` stays the target, and the flag is not
forwarded to vLLM. The checkpoint is not treated as a LoRA adapter.

The flag is rejected unless all of these hold:

- The job is a sampling job.
- `source_checkpoint_info` names a checkpoint saved by Cortex Training, not an
  `external_stage_path`.
- `speculative_config.model` is not set. A job names its draft either by `model`
  or by this flag, not both.

---

## 9. DSSST1 binary wire protocol

### 9.1 Why DSSST1

DSSST1 is a safetensors-based frame implemented by `cortex_training.wire`. It
serializes nested dict/list/tuple structures containing tensors and JSON-safe
values without pickle execution.

`wire.loads()` explicitly rejects legacy pickle and `torch.save` signatures.

### 9.2 Frame contents

A frame is one safetensors blob:

```text
u64 header length | safetensors JSON header | tensor bytes
```

Safetensors metadata contains:

- `dss`: the nested structure skeleton and wire version `DSSST1`.
- `op`: optional operation metadata such as response options, router replay,
  and request/result chunk descriptors.

Encode/decode:

```python
from cortex_training import wire

frame = wire.dumps(value, metadata=metadata)
value = wire.loads(frame)
metadata = wire.read_metadata(frame)
```

### 9.3 Request chunking

`forward_backward()`, `generate()`, and byte `forward()` call:

```python
wire.encode_byte_chunks(
    frame,
    kind="request",
    operation="fwd-bwd-or-generate-or-forward",
    max_bytes=...,
)
```

`forward_backward()` always sends a chunk envelope, including when the request
fits in one chunk, so every logical operation has a caller-generated identity.
`generate()` still sends the original frame unchanged when it fits, as
`application/octet-stream`. Byte `forward()` also sends the original frame
unchanged when it fits; it splits only when the encoded JSON body would exceed
`CORTEX_TRAINING_FORWARD_OPERATION_MAX_JSON_BYTES` (default 16 MiB). Those
chunks travel inside the `/operation` JSON envelope (`payload_b64`) rather than
as octet-stream. Each DSSST1 request chunk contains:

- A `uint8` payload tensor.
- `chunk_idx` and `total_chunks`.
- `chunk_group_id`.
- Original frame size and SHA-256.
- Operation name.

For forward-backward, `chunk_group_id` is also the idempotency identity and
`frame_sha256` binds that identity to one exact frame:

- The server retains accepted chunks with the training job rather than the
  HTTP connection.
- An unknown group must start at chunk `0`. A later first chunk returns `409`
  with code `chunk_group_restart_required`; the client may replay once from
  chunk `0` using the same encoded chunks and group id.
- Repeating the same group id and frame hash returns or waits for the original
  execution result. It does not run forward-backward again.
- Reusing a group id with a different frame hash, chunk count, or chunk payload
  returns `409`.

The final chunk schedules work and returns the pollable `request_id`. Retrying a
final chunk after losing its response can return a new `request_id`, but both
request records resolve through the same at-most-once forward-backward
execution.

### 9.4 Encoded results

A non-chunked DSSST1 result can appear inside poll JSON as:

```json
{
  "content_type": "application/octet-stream",
  "encoding": "base64",
  "wire_format": "DSSST1",
  "payload_b64": "base64-frame"
}
```

`poll_request()` base64-decodes and passes the frame to `wire.loads()`. If the
JSON envelope also has a `metrics` object, those keys are shallow-merged into
the decoded result's `metrics`. Envelope keys win on collision. Payload
`metrics` are kept when the envelope has none.

A result with `payload_b64` but no `wire_format` is left encoded unless
`poll_request()` is completing a `forward()` request id minted by this client.
In that case the blob is decoded the same way and envelope `metrics` are merged.

### 9.5 Result chunks

Large results can arrive through poll events:

```json
{
  "type": "result_chunk",
  "payload_b64": "base64-chunk-frame",
  "payload_sha256": "sha256"
}
```

A poll can also include `next_cursor`. `poll_request()` drains all pages,
validates chunk SHA-256 values, uses `wire.decode_result_chunks()`, and returns
the reconstructed object. Terminal-page `result.metrics` are merged into that
object with the same envelope-wins rule as [section 9.4](#94-encoded-results).

---

## 10. Forward/backward batch contract

### 10.1 Transport-level object

The transport accepts a DSSST1 object. The minimal conventional shape is:

```python
batch = {
    "args": (),
    "kwargs": {
        "input_ids": input_ids,
        "labels": labels,
    },
}
```

Common model kwargs are `input_ids`, `attention_mask`, `position_ids`, and
`labels`, generally shaped `[batch, sequence]`.

### 10.2 Readable payload helper

`build_forward_backward_payload(spec)` is a CLI/readability helper. It builds
only `args` and `kwargs`, then calls `serialize_forward_backward_args()`.

Direct tensor-shaped JSON:

```json
{
  "payload": {
    "kwargs": {
      "input_ids": {"data": [[1, 2, 3]], "dtype": "long"},
      "labels": {"data": [[2, 3, -100]], "dtype": "long"}
    }
  }
}
```

It can also tokenize `texts` with a configured Transformers tokenizer.

When building labels, supported strategies are:

- `next_token` / `shifted_input_ids`
- `input_ids` / `self`
- `none`
- explicit tensor data

The default helper strategy is next-token labels. It rolls input ids left,
sets the last target to `ignore_index` (default `-100`), and can mask padding.

### 10.3 ArcticTraining extensions

ArcticTraining serializes a larger object directly with `wire.dumps()`:

```python
batch = {
    "args": (),
    "kwargs": {...},
    "context": {...},
    "processing": {
        "loss_fn": "grpo",
        "config": {...},
        "post": [...],
    },
}
```

`context` and `processing` are open backend contracts. Their accepted keys,
registered loss functions, and post-processors come from the deployed training
backend, not the REST schema. The optional ratio controls of the `grpo` loss and
its variants, and the `grpo` teacher term, both described below, are an
exception: their configuration keys, context tensors, and the metrics named
below, are listed here.

Do not use `build_forward_backward_payload()` for this extended shape: that
helper currently ignores `context` and `processing`.

#### Ratio controls for the CISPO loss

On server versions that include these controls, the `grpo` loss and its variants
`ap_grpo`, `grpo_echo_v1`, and `ap_grpo_echo_v1` accept optional ratio controls
in `processing.config`. All are off by default, and any active control requires
`use_cispo_loss` to be `true`. The masks drop tokens from the CISPO policy term
only; the loss normalizers stay as they were.

| Key | Meaning |
|---|---|
| `ratio_mask_bounds_pos`, `ratio_mask_bounds_neg` | Inclusive importance-ratio bands, chosen by the sign of the token's advantage (see below). A pair `[low, high]` of finite numbers or `null`, with `low < high` and `low >= 0`; a `null` end is unbounded |
| `prob_diff_mask_max_pos`, `prob_diff_mask_max_neg` | Finite, non-negative maximum directional probability gap between the current policy and the sampler |
| `seq_mask_stat` | The sequence statistic that the sequence gate tests: `mean_log_ratio` (default) or `mean_k3` |
| `seq_mask_bounds_pos`, `seq_mask_bounds_neg` | Inclusive bands for that statistic, chosen by advantage sign (see below). A pair of finite numbers or `null`, with `low < high`; a `null` end is unbounded |
| `ratio_m2_threshold` | Finite, positive M2PO filtering threshold for the current-policy versus sampler ratio |
| `log_ratio_sq_coef` | Finite, non-negative coefficient of a squared log-ratio penalty. Unlike the masks, the penalty is computed over the original loss mask whatever the masks drop. A positive value is rejected with `token-mean` aggregation when `sequence_loss_weights` is present |
| `ratio_stats` | Boolean. `true` reports the ratio metrics (histograms, mask counts, and the metrics below) without enabling a mask |
| `ratio_mask_rebalance` | Boolean. `true` rescales the kept advantages of each sign that keeps any mass so that its total returns to its pre-mask value (see below) |

A control is active as soon as its key is present with a valid value, even a
neutral one such as `log_ratio_sq_coef: 0.0`. An explicit `null` is rejected for
every ratio-control key, so omit a key to leave it off; inside a bounds pair, a
`null` end still means unbounded. An explicit `false` of `ratio_stats` or
`ratio_mask_rebalance` does not count: a request whose only ratio-control keys
are one or both of these set to `false` is inert and needs no `use_cispo_loss`.
The mixed losses `grpo_mixed_v1` and `ap_grpo_mixed_v1` reject every
ratio-control key, including an explicit `false`, so that policy-only gates
cannot affect NLL tokens.

The `_pos` and `_neg` gates split tokens by the sign of the token-level
advantage, before any sequence-level averaging and including the teacher term
when it is enabled. An advantage of exactly `0` counts as positive. With
`importance_sampling_level` set to `"sequence"`, the CISPO term uses each
sequence's averaged advantage while the gates keep the token-level sign, so the
two can differ. A sequence is tested against `seq_mask_bounds_pos` when it has
at least one positive-side token and against `seq_mask_bounds_neg` when it has
at least one negative-side token, and a band violation drops only that side's
tokens.

Whenever a ratio control is active, the `metrics` of the forward/backward result
carry the keys below. Keys ending in `_sum` are added up over workers. The
result also carries other ratio-control entries, such as per-drop counts and
further histogram bins, that are not listed here.

| Metric | Meaning |
|---|---|
| `ratio_mask_pos_pre_mass_sum`, `ratio_mask_neg_pre_mass_sum` | The advantage mass before masking: `\|A\|` summed over the policy tokens with advantage `A >= 0` (`pos`) or `A < 0` (`neg`). It goes through the loss's own normalization (for `token-mean`, divided by the number of policy tokens in the whole request), without the data-parallel factor and without importance, clip, `rollout_is_weights`, or `behav_imp_weight_cap` weights. `A` is the advantage the CISPO term uses: with `importance_sampling_level` set to `"sequence"`, each sequence's averaged advantage, so the sequence counts as `pos` or `neg` as a whole by the sign of its mean |
| `ratio_mask_pos_kept_mass_sum`, `ratio_mask_neg_kept_mass_sum` | The same sums over the tokens that survive every active drop (ratio band, probability gap, sequence gate, `ratio_m2_threshold`) |
| `ratio_mask_kept_k3_sum` | The sum over the kept tokens of `exp(d) - 1 - d`, where `d` is the trainer log probability minus the sampler log probability (`old_log_probs_shifted`), clamped to `[-20, 20]`. It is raw, not divided by the loss denominators, and is `0` when the request has no `old_log_probs_shifted` |
| `ratio_trainable_token_count`, `ratio_mask_dropped_token_count` | The policy tokens the ratio controls see, and how many of them some active drop removes |
| `seq_stat_bin_<i>_sequence_count`, `i` from `0` to `6` | The number of sequences whose log ratio (current policy minus sampler log probability), averaged over the sequence's policy tokens, falls in bucket `i` of the edges `-0.2, -0.05, -0.01, 0.01, 0.05, 0.2`. Each bucket includes its upper edge: bucket `0` holds values up to `-0.2` and bucket `6` values above `0.2`. It always bins the mean log ratio, also when `seq_mask_stat` is `mean_k3` |
| `ratio_mask_rebalance` | The flag as `0.0` or `1.0` |
| `ratio_rebalance_pos_post_mass_sum`, `ratio_rebalance_neg_post_mass_sum` | Only with `ratio_mask_rebalance` set to `true`. The kept mass of the sign after scaling: equal to its pre-mask mass, up to rounding, when the sign's kept mass is positive, and `0` when it is not |
| `ratio_rebalance_pos_unrestored_mass_sum`, `ratio_rebalance_neg_unrestored_mass_sum` | Only with `ratio_mask_rebalance` set to `true`. The pre-mask mass of a sign whose kept mass is `0`, and `0` otherwise |

The four mass sums equal their value for the unsplit batch, up to rounding, when
the loss normalization is request-global: `token-mean`; `prompt-mean` with
`prompt_group_ids` (the service derives the request-wide counts it needs) or with
client-supplied `sequence_loss_weights`; and the `seq-mean-token-sum`,
`seq-mean-token-sum-norm`, and `seq-mean-token-mean` values of `loss_agg_mode`
when the client sends a request-global `global_batch_size`. Otherwise each call
normalizes by its own counts, and so do these sums. Sequence-parallel shards
count once, and a worker with no policy token reports `0`. The `post` and
`unrestored` sums below are in the same normalized units and are also added up
over workers. The k3 sum and the two counts are not normalized: they add up across microbatches, workers, and
sequence-parallel ranks in every aggregation mode, apart from the placement
effects below.

`ratio_m2_threshold` ranks the tokens of one model call, so with it the kept
mass, the dropped count, and the k3 sum depend on how rows are placed on
workers, while the pre-mask mass does not. The baseline `m2_threshold` key,
which is not `ratio_m2_threshold`, acts on the tokens a worker holds in one call
(under sequence parallelism, on each rank's own window) and removes tokens
before the ratio controls act. Those tokens are in neither the pre-mask nor the
kept mass, and are excluded from `ratio_trainable_token_count`. With it the
masses, the k3 sum, and the counts depend on how rows are placed in calls and on
the sequence-parallel split, and with several calls per worker on
`max_tokens_per_mb`. With either key the restoration scale below depends on
placement as well.

The clamp of `d` applies to the reported k3 sum only: `seq_mask_stat` set to
`mean_k3` gates on the unclamped k3. The kept-token mean is
`ratio_mask_kept_k3_sum / (ratio_trainable_token_count -
ratio_mask_dropped_token_count)`. When the request carries
`old_log_probs_shifted`, the service also reports
`trainable_logprob_lowvar_all/mean`, the same estimator over every trainable
token before masking, with the same clamp. With no drop, no token removed by the
baseline `m2_threshold`, and finite log probabilities,
`ratio_mask_kept_k3_sum / ratio_trainable_token_count` equals it up to rounding.
The service computes `trainable_logprob_lowvar_all/mean` with `exp(d) - 1 - d` in
float32, which loses precision for `|d|` of about `1e-3` or less, so a gap there
comes from that metric. A trainer log probability that is not finite is replaced
by `0` before the loss, and its token still counts. A sampler log probability
that is not finite at a trained token makes the request fail, so no k3 sum is
returned.

With `ratio_mask_rebalance` set to `true`, once the masks have decided which
tokens to drop, the kept tokens of positive advantage are scaled by `P0/Pk` and
those of negative advantage by `N0/Nk`. `P0` and `N0` are
`ratio_mask_pos_pre_mass_sum` and `ratio_mask_neg_pre_mass_sum`; `Pk` and `Nk`
are `ratio_mask_pos_kept_mass_sum` and `ratio_mask_neg_kept_mass_sum`; all four
are summed over all workers. Every worker applies the same scale, and each sign
returns to its own pre-mask mass, not to the other sign's. Only advantage
magnitudes change: which tokens are dropped, the counts, and the k3 sum do not
depend on the scaling. The scale is computed for one forward/backward request,
so a step split across several requests gets one scale per request, and a sign
that loses all its mass in one part is not restored there. A sign whose kept mass
is `0` has nothing to scale, including a sign whose only surviving tokens have
advantage `0`: its scale is `1` and its pre-mask mass is reported as
unrestored. With nothing dropped, both scales are `1` and the objective is
unchanged, up to rounding. The scale is not capped and is not reported, but it
can be computed from the reported masses, and it is large when little of a
sign's mass survives (`100` when one of 100 equal-magnitude positive tokens is
kept). The CISPO ratio clip (`is_weight_clip_max`, when set) still applies
independently of the scaling, so the restoration does not equalize
ratio-weighted gradients.

`ratio_mask_rebalance` set to `true` and `ratio_m2_threshold` each require one
packed model call per worker. A request that would need more than one call per
worker is rejected before it is dispatched to the workers, with an error that
names the key (both keys when both are set) and the per-worker call counts. The
training sub-job's `training_config.mb_spec.max_tokens_per_mb` budget (default
`10240` tokens when `mb_spec` or the key is omitted) must therefore be large
enough to keep each worker's share in one call. Two rejections come later:
`ratio_m2_threshold` is also rejected under sequence parallelism, and an active
ratio control without `use_cispo_loss` is rejected. A worker raises both when it
resolves the loss, before any forward pass.

#### Teacher term for on-policy distillation

On server versions that include it, the `grpo` and `ap_grpo` losses accept a
teacher term that adds, at every trained token, a clipped teacher-minus-policy
log ratio to that token's advantage:

```text
advantage[i] += teacher_tau * clamp(teacher_log_probs_shifted[i] - log_prob[i],
                                    -teacher_clip_negative, teacher_clip)
```

`log_prob[i]` is the training policy's own log probability, detached. With
`context["advantages"]` all zero, this term is the whole signal and the update
is an on-policy distillation step on reverse KL to the teacher; with task
advantages, the two add.

| Key | Meaning |
|---|---|
| `teacher_tau` | Finite, non-negative weight of the term. `0` (the default) turns it off |
| `teacher_clip` | Finite, positive upper clip on the log ratio. Required when `teacher_tau > 0` |
| `teacher_clip_negative` | Optional finite, non-negative magnitude of the lower clip. Defaults to `teacher_clip` |

When `teacher_tau > 0`, `context` must carry `teacher_log_probs_shifted`: a
floating-point tensor with the same shape as the prediction-aligned
log probabilities, where position `i` holds the teacher's log probability of
`input_ids[:, i + 1]` (see [section 10.4](#104-shifted-log-probability-convention)).
Positions outside `loss_mask` are ignored, as are positions whose teacher value
is not finite. The term also requires the default token-level
`importance_sampling_level`, and with `token-mean` aggregation it rejects
`sequence_loss_weights`. Configuration keys are checked by name, so a misspelled
key fails the request instead of training without the term.

With the term enabled, the `metrics` of the forward/backward result carry:

| Metric | Meaning |
|---|---|
| `teacher_tau` | The weight that was applied |
| `teacher_term_token_count` | Tokens that received the term |
| `teacher_log_ratio_sum` | Sum of the unclipped teacher-minus-policy log ratios over those tokens |
| `teacher_clipped_log_ratio_sum` | The same sum after clipping |

`-teacher_log_ratio_sum / teacher_term_token_count` is a per-token estimate of
the reverse KL from the policy to the teacher on the trained tokens.

### 10.4 Shifted log-probability convention

ArcticTraining's RL adapter documents `_shifted` log-probability tensors as:

```text
tensor[:, i] is the log probability of input_ids[:, i + 1]
```

This is an ArcticTraining processing convention, not a REST-level field
requirement.

---

## 11. Result summary

These are conventional backend results, not closed REST schemas:

| Operation | Common result |
|---|---|
| `forward-backward` | `job_id`, `avg_loss`, `metrics`, `post_process_outputs`; `metrics` carries the ratio-control metrics when those controls are active (see [section 10.3](#103-arctictraining-extensions)) |
| `step` | `global_steps`, `last_lr`, optional `peak_memory` |
| `save` | `job_id`, `stage_path` (carries the `cp_<uuid>` id), `checkpoint_path`, `checkpoint_tag`, `version` — no `checkpoint_id` key in the observed release (see 6.3) |
| `load` | `checkpoint_id` and backend load metadata |
| `generate` | `job_id`, `results[]` |
| `weight-sync` | Completion/transfer metadata; each sampling target's result carries `clear_cache` (see [section 7.2](#72-weight-sync)) |
| `forward` | Backend-specific forward-only result |

Generic-operation responses and fields inside `metrics` or generation results
are intentionally open.

---

## 12. Logs and events

### 12.1 Live logs

Use `tail_logs()` for one cursor page or `stream_logs()` for an iterator. This
uses the `tail-logs` operation described in [section 7.8](#78-tail-logs).

The server reads the selected sub-job's zone-manager/head-pod stdout. It can
serve empty, non-EOF pages during placement while the pod is still appearing.

### 12.2 Full execution-log download

`download_execution_logs(job_id, output_dir, *, resume=False, sub_job_ids=None,
progress=None, transfer_timeout_seconds=300)` saves every file below a
`/_logs/{sub_job_id}/` subtree to `<output_dir>/<sub_job_id>/<filename>`
without holding it in memory. Each file is downloaded into a hidden
`.execution-log-*.tmp` directory beside its destination, made readable only by
its owner (mode `0600`), and then moved into place, so an interrupted download
leaves any earlier copy as it was. A file present under both `_logs/` and
`checkpoints/_logs/` keeps the `_logs/` copy.

A staged execution-log file is replaced as a whole while the job runs, and
`GET` cannot fetch part of one. With `resume=True`, a file whose staged size and
checksum (from `LIST`) match those recorded beside the earlier copy, in
`.<filename>.artifact.json`, is not fetched again; a changed file is
downloaded in full. `transfer_timeout_seconds` bounds each `GET` attempt and is
raised for a large file to its size at 512 KiB/s. `progress` receives one
status line per file, ending in `"\n"`, in the same form as for persisted
stdout. A name in `sub_job_ids` with no execution logs raises
`NoChunksError`, which lists the sub-jobs that have some.

Return:

```python
[
    {
        "sub_job_id": "job-id:training:0",
        "filename": "execution.jsonl",
        "artifact_uri": "snow://experiment/DB.SCHEMA.EXPERIMENT/versions/RUN/_logs/job-id:training:0/execution.jsonl",
        "saved_path": "/path/to/output/job-id:training:0/execution.jsonl",
        "bytes": 309000000,
        "unchanged": False,
    }
]
```

`unchanged` is `True` when `resume` kept the earlier copy.

`fetch_execution_logs(job_id)` returns the same files with their bodies in
memory. An execution log can be hundreds of megabytes, so prefer
`download_execution_logs()` for anything but small jobs. It:

1. Calls `GET /{job_id}/experiment-run`, once per job for the client (see
   [section 12.7](#127-download-sessions)).
2. For profile clients, opens an independent connection from the same
   Snowflake profile.
3. For direct PAT clients, resolves the current user and role through the SQL
   statements API and opens an equivalent Connector session.
4. Uses experiment artifact `LIST` and `GET` to download every object below a
   `/_logs/{sub_job_id}/` subtree.

Return:

```python
[
    {
        "sub_job_id": "job-id:training:0",
        "filename": "execution.jsonl",
        "artifact_uri": "snow://experiment/DB.SCHEMA.EXPERIMENT/versions/RUN/_logs/job-id:training:0/execution.jsonl",
        "content": "...",
    }
]
```

The client returns logical artifact URIs and does not expose physical storage
paths or object-store credentials.

### 12.3 Persisted stdout download

`download_stdout_logs(job_id, output_dir, *, resume=False, max_workers=None,
progress=None, on_chunk=None, sub_job_ids=None, strict=False,
stale_after_seconds=600, transfer_timeout_seconds=300, preview_seconds=None,
on_preview=None)`
downloads gzip chunks under
`_stdout/{sub_job_id}/`, orders them by chunk name (which starts with the
chunk's UTC time), and streams them into `<output_dir>/<sub_job_id>/stdout.log`.
If a run has chunks for one sub-job under both `_stdout/` and
`checkpoints/_stdout/`, the two lists are merged by chunk name and a name found
in both keeps the `_stdout/` copy; the sub-job's result gets a warning.
`preview_seconds` and `on_preview` give the first sub-job named in
`sub_job_ids` its console's newest output first, the same way
`download_metrics` previews metrics (see
[section 12.4](#124-gpu-metrics-download)); that sub-job is written first, and
without `sub_job_ids` there is no preview.

The chunks are found with an experiment artifact `LIST` of each of the two
roots. When `sub_job_ids` names one of the job's sub-jobs, the download lists
only that sub-job's folder in each, `_stdout/{sub_job_id}/` and
`checkpoints/_stdout/{sub_job_id}/`, so the listing returns that sub-job's
chunks rather than the whole run's; naming several, or one whose id has
characters other than letters, digits, `_`, `.`, `:` and `-`, lists the roots
whole. Once
a listing of both roots finds a sub-job's chunks under just one, the client,
and every client `for_job()` makes from it, lists only that one for the
sub-job's later downloads of the same kind of chunk that name it alone, until
that finding is 10 minutes old; it lists the other root too when that one
holds none of them. A sub-job with chunks under both roots has both listed.

Each chunk is one experiment artifact `GET` into a private temporary directory.
Up to `max_workers` (default 16) run at once on one Connector session, each on
its own cursor; fetches may finish in any order, but chunks are appended
strictly in sorted name order. The session, and the listings that come first,
are described in [section 12.7](#127-download-sessions). At most four fetched chunks per worker wait to
be appended, which bounds temporary disk use. `LIST` and `GET` are retried with
bounded exponential backoff on transient Connector, network, and service
errors; a session or token expiry reopens the session. Each `LIST` or `GET`
attempt is also limited to `transfer_timeout_seconds` (default 300; `None`
removes the limit): the Connector's own storage read timeout allows many
minutes per request and retries, so a stalled transfer could otherwise hold a
download indefinitely. An attempt over the limit is abandoned and retried like
any transient error; it only ever writes to its own temporary directory, so a
late finish cannot change the output. Errors another attempt
cannot fix fail immediately: missing privileges, a storage-side access refusal,
an invalid stage or local path, and other statement errors. A chunk that no
longer exists when it is fetched (it was removed after the listing) is skipped
like a damaged chunk, below, and `strict=True` raises instead.

A hidden `.<filename>.manifest.json` beside the file records `version`, the
sorted relative chunk paths, and the committed byte length. While a sub-job is
being rebuilt the output goes to a hidden `.<filename>.partial` file with its
own `.<filename>.partial.manifest.json`; it replaces the destination only once
every chunk is written, so a failed download leaves an earlier destination
untouched. Manifests are checkpointed every 64 chunks or 5 seconds (for long
downloads, at least every tenth of the chunks so far, which keeps manifest
writes proportional to the chunk count) and when a download stops on an error
or interrupt, and only after the data they describe is fsynced, so the recorded
length never exceeds the file. Progress is saved
whether or not `resume` is set. Fetched chunks wait in a private directory under
the system temp directory. It is removed when the download returns, or after a
cancel or error once the `GET`s that were already running finish. A directory
left by a killed process is removed by a later download once it has not been
written for an hour.

`resume=True` first tries to extend the completed file, then a partial one. A
file is extended only when its manifest paths are a prefix of the current
sorted list and the file is at least that many bytes; anything past the
recorded length is truncated first. Otherwise the sub-job is rebuilt from the
first chunk, as it always is with `resume=False`. Leftover temporary files from
a killed download are removed when the next download of that file starts, once
they have not been written for an hour.

A chunk that fails to decompress (bad gzip header, truncated member, corrupt
data) is rolled back from the file, downloaded once more, and then skipped if it
is still damaged. The manifest lists a skipped chunk both in `names` and in an
optional `skipped` array, so resume does not fetch it again; a `resume=False`
download retries it. `strict=True` raises on the first damaged chunk instead.

Each result entry has `sub_job_id`, `filename`, `saved_path`, `chunk_count`
(chunks written), `first_artifact_uri`, and `last_artifact_uri`, plus
`skipped_chunks` (artifact URIs left out of the file) and `warnings`
(customer-facing notices, also logged at WARNING) when either is non-empty.

Stalled-upload warnings never fail the download. When a sub-job's newest chunk
name time (UTC) may be more than `stale_after_seconds` old, the client reads the
job with `GET /{job_id}`. It warns only if the job is `running`, the sub-job's
own `status` (when the response lists it) is `running`, and the job has been in
its current state (`state_changed_at`, when present) for at least
`stale_after_seconds`. Ages are measured with the server's clock from the
response's `Date` header; without one, the local clock is used, so a local
clock that is off can cause or hide a warning. The sub-job's result then gets a
warning such as "newest saved stdout output for SUB_JOB_ID is from
2026-01-01 12:00:00 UTC (14 min ago) while the job is still running; uploads
may be stalled. Contact support if this persists." `None` disables the check.

Progress is reported only through `progress` and `on_chunk`; retries, skipped
chunks, and stalled-upload notices are logged as warnings on the
`cortex_training` logger (and listed in the result's `warnings`). The stall
check reads the job status in the background while chunks are fetched, so it
does not lengthen the download; a status lookup still pending after the chunks
are written is given up to 10 seconds. `progress` receives, for
each sub-job, a status line such as `stdout: [######--------------] 1/3
SUB_JOB_ID` before anything in the output directory changes, an updated line
after each appended chunk (counting only the chunks this call adds), and then
`"\n"` once that sub-job's manifest is written or the sub-job fails. A sub-job
with nothing new gets one message ending in a newline (`stdout: no new chunks
for SUB_JOB_ID`), and a job with no chunks gets `stdout: found 0 chunks`. On a
terminal, the CLI draws one updating line on stderr; a redirected stderr gets
one final line per sub-job.

`on_chunk(sub_job_id, chunks_done, chunks_total, bytes)` reports counts
instead: it is called when a sub-job starts and after each appended chunk, with
totals over the whole sub-job, and the call with `chunks_done == chunks_total`
comes only after that sub-job's file is complete and committed. Both callbacks
run on the calling thread. An error raised by either stops the download as
below. To stop a download on purpose, raise
`cortex_training.DownloadCancelled` from a callback: no new `GET` starts,
workers waiting out a retry backoff stop at once, the chunks already written
are kept for `resume=True`, and the exception is re-raised. Client telemetry
records it as cancelled, not failed. If a download fails or is interrupted
after saving any output, the exception carries
`download_progress_saved = True`. `sub_job_ids` limits the download
to those sub-jobs and raises `cortex_training.NoChunksError` (a `ValueError`
with `sub_job_ids` and `available` attributes) before any `GET` if one has no
chunks. `available` names every other sub-job of the job that has chunks: a
download that listed one sub-job alone then lists both roots whole to find them.
Two downloads of the same output directory at once are unsupported.

### 12.4 GPU metrics download

`download_metrics(job_id, output_dir, *, resume=False, max_workers=None,
progress=None, on_chunk=None, sub_job_ids=None, strict=False,
stale_after_seconds=600, transfer_timeout_seconds=300, priority=(),
on_commit=None, preview_seconds=None, on_preview=None)`
applies the same reconstruction to
`gpu.YYYYMMDD-HHMMSS.UUID.gz` chunks under `_metrics/{sub_job_id}/` and writes
`<output_dir>/<sub_job_id>/gpu.jsonl`, including the options, listing, retries,
and the manifests beside that file. Unless `stale_after_seconds` is `None`, it
also reads the job status. While the job is running it also lists stdout
chunks, limited by `sub_job_ids` as its metrics listing is, and a running
sub-job that has had stdout chunks for at least two minutes (judged from its
first stdout chunk and `state_changed_at`) but no metrics chunks is returned
with `chunk_count` 0,
`saved_path`, `first_artifact_uri`, and `last_artifact_uri` set to `null`, and a
warning that metric uploads may be stalled. Such a sub-job may be named in `sub_job_ids` without an error.

For a job that has ended, each sub-job with stdout chunks but no metrics
chunks gets the same kind of entry, with a warning that no GPU metrics were
saved for it: jobs that ran on images from before GPU metrics were uploaded, or
that ended before their first metrics upload, have none. An empty result is
therefore never silent about a sub-job that has output.

While a job is running, and for 120 seconds after it ends (judged from
`state_changed_at`), its last chunks may still be arriving. A stdout or
metrics download in that window gets a warning per sub-job saying so; download
again with `resume=True` a minute or two later to pick them up. These job
lookups are skipped when `stale_after_seconds` is `None`.

Sub-jobs named in `priority` are written first, in that order; `on_commit` gets
each sub-job's result entry once its file is committed. With `preview_seconds`
and `on_preview` (given together), the first sub-job in `priority` gets its
newest samples before the rest of its history: its chunks from
`preview_seconds` before its newest chunk are fetched first (two minutes more,
since a chunk's name gives the time it starts), and `on_preview` gets
`sub_job_id`, `path`, `chunk_count`, `first_artifact_uri` and
`last_artifact_uri`. The file at `path` holds those chunks' rows in time order
and is removed once `on_preview` returns. The chunks are not fetched again, so
the download makes no more requests than without a preview; there is none when
they are most of what the sub-job has new, or when none of them can be read.
Metrics chunks are uploaded about once a minute per sub-job, so a cold download
of a long job is mostly per-chunk round trips: this is how the TUI draws the
window on screen (the newest 10 minutes, where every job opens, and at most the
newest half hour) before the rest of its history arrives.

The reconstructed file contains the JSONL records emitted by the deployed
training runtime. Version 2 GPU rows have no `kind` field and include:

| Field | Meaning |
|---|---|
| `record_version` | Row schema version; currently `2` |
| `sub_job_id` | Sub-job that produced the sample |
| `sample_ts` | RFC3339 UTC scrape time; fractional seconds may use nanosecond precision |
| `worker_num` | Worker ordinal within the sub-job |
| `local_gpu_index` | GPU index within one worker; it repeats across workers |
| `global_gpu_index` | GPU identity within one sub-job; group GPU series by this field |
| `gpu_uuid` | Device UUID reported by the GPU runtime |
| `gpu_util_pct` | GPU utilization percentage gauge |
| `fb_used_bytes` / `fb_free_bytes` | Used and free framebuffer bytes |
| `power_watts` | Board power gauge |
| `gpu_temp_c` | GPU temperature gauge |
| `sm_active_pct` / `sm_occupancy_pct` | Optional SM activity gauges |
| `tensor_active_pct` / `dram_active_pct` | Optional tensor and DRAM activity gauges |
| `seq` / `seq_epoch` | Sample sequence and its opaque generation |
| `sample_interval_s` | Sampling interval in seconds |

One row represents one GPU at one scrape. Gauge values are used as-is rather
than differenced. JSON `null` means unmeasured and is distinct from `0.0`.
`global_gpu_index` is unique only inside a sub-job, so consumers loading more
than one file also group by `sub_job_id`.

The same file also carries one host row per worker per scrape, with
`"kind": "system"` and `record_version` `3`:

| Field | Meaning |
|---|---|
| `kind` | `"system"` |
| `record_version` | `3` |
| `sub_job_id`, `sample_ts`, `worker_num`, `seq`, `seq_epoch`, `sample_interval_s` | As for GPU rows |
| `cpu_util_pct` | Host CPU utilization percentage gauge |
| `cpu_count` | Logical CPUs on the host |
| `mem_used_bytes` / `mem_total_bytes` | Host memory in use and installed |

### 12.5 Training metrics download

`download_training_metrics(job_id, output_dir, ...)` takes the same options as
`download_metrics` and reconstructs a training sub-job's **training metrics**:
what its `/fwd-bwd` and `/step` calls returned, written by the zone to its own
file and staged beside the GPU chunks as
`training.YYYYMMDD-HHMMSS.UUID.gz` under `_metrics/{sub_job_id}/`. It writes
`<output_dir>/<sub_job_id>/training.jsonl`, with its own manifest, so the two
downloads can share an output directory and neither stream's rows reach the
other's file. `cortex-training download-metrics` runs both and reports them as
`metrics` and `training_metrics`.

Only training sub-jobs have these rows, and only where the deployment saves
them; a job opts out by setting `training_config.save_training_metrics` to
`false`. So, unlike the GPU stream, a sub-job with saved stdout and no training
chunks is not warned about, and a sub-job named in `sub_job_ids` without
training chunks is left out of the result rather than raising `NoChunksError`.

The zone appends one row per successful `/fwd-bwd` call and one per successful
`/step`, with `"kind": "training"` and `record_version` `2`. Each row holds the
numbers its call's result carries, as the client receives them; nothing is
combined or computed:

```json
{"kind": "training", "record_version": 2, "sub_job_id": "JOB_ID:training:0",
 "sample_ts": "2026-10-16T18:02:14Z", "seq": 84, "seq_epoch": "...",
 "op": "fwd_bwd", "step": 3, "call": 1, "request_id": "REQUEST_ID",
 "metrics": {"avg_loss": 2.0419, "rank": 0, "sft_model_calls": 1, "sft_packed_rows": 1}}
{"kind": "training", "record_version": 2, "sub_job_id": "JOB_ID:training:0",
 "sample_ts": "2026-10-16T18:02:15Z", "seq": 86, "seq_epoch": "...",
 "op": "step", "step": 3, "call": null, "request_id": "REQUEST_ID",
 "metrics": {"global_steps": 3.0, "grad_norm": 84.37, "last_lr": 0.0001,
             "update_successful": 1.0, "weight_delta_norm_rank0_sample": 0.1447}}
```

| Field | Meaning |
|---|---|
| `kind` | `"training"` |
| `record_version` | `2` |
| `sub_job_id`, `seq`, `seq_epoch` | As for GPU rows; a gap in `seq` means rows were lost |
| `sample_ts` | When the call completed |
| `op` | `"fwd_bwd"` (one `/fwd-bwd` call) or `"step"` (one `/step`) |
| `step` | On a `step` row, the step's `global_steps`. On a `fwd_bwd` row, the step the call counts toward: the previous `/step`'s `global_steps` + 1, or `null` when the zone doesn't know it yet: before it has seen a `/step` (a job's first calls, or the first after the zone restarted), and after a `load-checkpoint` moves the job to another step. Such a call belongs to the `step` row that follows it. A `/fwd-bwd` retried and answered from the first run's cached result is saved once |
| `call` | On a `fwd_bwd` row, the call's place among its step's calls, from 1; `null` on a `step` row |
| `request_id` | The call's request id |
| `metrics` | Every finite number in the result, by its own name |

A `fwd_bwd` row's `metrics` are the result's top-level numbers (`avg_loss`,
that call's own loss) and every number in its `metrics` (`rank`,
`sft_model_calls`, `sft_packed_rows`, or whatever the loss function reports);
a top-level field wins over a nested metric of the same name. A `step` row's
`metrics` are every number in the `/step` result (`global_steps`, `grad_norm`,
`last_lr`, `update_successful`, `weight_delta_*`, ...). Text (such as
`job_id`), lists, booleans, `null`, NaN and infinities are left out, and a row
keeps at most 256 metrics with names of at most 200 bytes.

A step's loss is not saved as such: whether it is the sum or the mean of its
calls' `avg_loss` depends on how the loss was scaled, so each call's own
`avg_loss` is saved and the choice is left to the reader. A `/step` that fails
saves no row, and calls made before it count toward the same step as the retry's
calls. The metrics a loss function returns vary, so consumers should read
whatever names a row has.

In the GPU file, a row without `kind` is a GPU row; a later GPU row may carry
`kind: "gpu"`. Consumers should select rows by `kind` and ignore kinds they do
not support. The TUI's Resources tab uses GPU and system rows; its Training
metrics tab reads `training.jsonl` and shows it per step and per call.

### 12.6 Zone scheduling events

The client contains `tail_events()` and `stream_events()`, but the server does
not accept their operation type. See
[section 7.9](#79-unsupported-zone-events-client-helper).

### 12.7 Download sessions

Each artifact read in sections 12.2 to 12.5 resolves the job's experiment run
(`GET /{job_id}/experiment-run`) and then lists and fetches on a Connector
session. A client, and every client `for_job()` makes from it, resolves each
job's experiment run once and reuses it, since a job's run never changes: they
share what they read, and remember the newest 256 jobs' runs. A failed read is
not kept, so the next call tries again. A chunked download (stdout, GPU
metrics, training metrics) lists its two roots, `_{name}/` and
`checkpoints/_{name}/` (for one sub-job, its folder in each: see
[section 12.3](#123-persisted-stdout-download)), at the same time, each on its
own cursor; one that lists only the root it knows for the sub-job makes one
listing, then one of the other root if that one holds none of the sub-job's
chunks. If the first listing fails, the download waits up to 10 seconds for
the second, which runs on the same session, before it closes that session; one
still running then finishes on a session that is closed, never kept.

A PAT client opens its sessions with `platform_detection_timeout_seconds=0`
(Connector 3.17.2 and later), which skips the Connector's cloud-metadata
probes at login. The probes only report the platform the client runs on, and
off a cloud VM the Connector waits for each to give up: from a fraction of a
second to over two seconds per login. The login itself is unchanged. A profile
client's sessions are opened as its profile says.

By default `download_execution_logs`, `download_stdout_logs`,
`download_metrics` and `download_training_metrics` close their session when
they return, and `fetch_execution_logs` always does. Setting
`keep_artifact_sessions = True` on the client keeps the session of a download
that succeeded, up to two per database and schema (two downloads can run at
once), for the next download, which then starts with its first `LIST` instead
of a login. A session idle for 30 minutes is closed instead of reused, and one
the Connector reports closed is never handed out. A kept session whose first
call fails before any call on it has worked, because it expired on the service
or its connection dropped while kept, is replaced at once by a fresh login,
without the retry backoff and without using one of the retries; if that fresh
login fails, the usual retries take over. A download that fails or is
cancelled, that abandoned an attempt past `transfer_timeout_seconds`, or that
left a listing running on its session, closes its session. `close()` closes the kept sessions, and so does setting
`keep_artifact_sessions` back to `False` on the client that owns them; a
client from `for_job()` shares them and closes none.

`warm_artifact_session(job_id)` prepares a download before it is asked for: it
resolves the job's experiment run and, while sessions are kept and none is kept
for the client's database and schema, logs one in and keeps it. It raises what
resolving the run or logging in raises, and then keeps nothing.

The TUI keeps sessions while it runs. It calls `warm_artifact_session` once a
job's screen has drawn its first log lines (or 1.5 seconds after the screen
opens, when no lines come), so that the job's first Resources or Training
metrics download starts with its listing. A warm-up that fails shows nothing on
screen, as the download then logs in itself and says what failed; its
traceback goes to the TUI's error log (`~/.cortex-training-errors.log`), once
per job screen. When the TUI quits, it closes the sessions it kept, waiting at
most a second.

---

## 13. End-to-end examples

### 13.1 Create training and sampling sub-jobs

```python
from cortex_training.client import CortexTrainingClient, SubJobConfig

client = CortexTrainingClient.from_configs(database=DATABASE, schema=SCHEMA)

training = SubJobConfig.training_job(
    model_name="Qwen/Qwen3-1.7B",
    optimizer={"name": "AdamW", "lr": 1e-4},
    max_seq_len=2048,
    train_batch_size=8,
    n_gpus=8,
    dtype="bfloat16",
)
sampling = SubJobConfig.sampling_job(
    model_name="Qwen/Qwen3-1.7B",
    max_seq_len=2048,
    n_gpus=1,
    dtype="bfloat16",
)

job_id = client.create_job(sub_jobs=[training, sampling])
client.wait_for_job(job_id)

training_id = f"{job_id}:training:0"
sampling_id = f"{job_id}:sampling:0"
```

### 13.2 Generate, train, step, and sync

```python
request_id = client.generate(
    job_id,
    prompts=["Write a short proof."],
    sampling_params={"max_tokens": 256, "temperature": 0.7},
)
rollouts = client.poll_request(job_id, request_id)["results"]

request_id = client.forward_backward(job_id, fwd_bwd_dssst1_frame)
train_result = client.poll_request(job_id, request_id)

request_id = client.step(job_id, learning_rate=1e-4)
step_result = client.poll_request(job_id, request_id)

request_id = client.weight_sync(
    job_id,
    source_sub_job_id=training_id,
    target_sub_job_ids=[sampling_id],
)
client.poll_request(job_id, request_id)
```

### 13.3 Save and runtime load

```python
from datetime import datetime

request_id = client.save(job_id, checkpoint_type="resumable")
client.poll_request(job_id, request_id)
# Select the latest resumable checkpoint without assuming list order.
checkpoint_id = max(
    (cp for cp in client.list_checkpoints(job_id) if cp["checkpoint_type"] == "resumable"),
    key=lambda cp: datetime.fromisoformat(cp["created_at"].replace("Z", "+00:00")),
)["checkpoint_id"]

request_id = client.load(job_id, checkpoint_id=checkpoint_id)
client.poll_request(job_id, request_id)
```

### 13.4 Start sampling from saved weights

```python
from datetime import datetime

request_id = client.save(training_job_id, checkpoint_type="weights-only")
client.poll_request(training_job_id, request_id)
checkpoint_id = max(
    (cp for cp in client.list_checkpoints(training_job_id) if cp["checkpoint_type"] == "weights-only"),
    key=lambda cp: datetime.fromisoformat(cp["created_at"].replace("Z", "+00:00")),
)["checkpoint_id"]

sampling = SubJobConfig.sampling_job(
    model_name="Qwen/Qwen3-1.7B",
    max_seq_len=2048,
    n_gpus=1,
    source_checkpoint_info={
        "checkpoint_id": checkpoint_id,
        "source_job_id": training_job_id,
    },
)
sampling_job_id = client.create_job(sub_jobs=[sampling])
```

### 13.5 Enable LoRA training and adapter sync

Configure the same adapter on both sub-jobs. The training configuration creates
the trainable PEFT parameters; the sampling configuration enables LoRA support
in vLLM before the first adapter sync.

```python
from cortex_training.client import CortexTrainingClient, SubJobConfig

client = CortexTrainingClient.from_configs(database=DATABASE, schema=SCHEMA)

lora_config = {
    "peft_type": "Lora",
    "r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

training = SubJobConfig.training_job(
    model_name="Qwen/Qwen3-1.7B",
    optimizer={"name": "AdamW", "lr": 1e-4},
    max_seq_len=2048,
    train_batch_size=8,
    n_gpus=8,
    dtype="bfloat16",
    extra_training={"peft_config": lora_config},
)
sampling = SubJobConfig.sampling_job(
    model_name="Qwen/Qwen3-1.7B",
    max_seq_len=2048,
    n_gpus=1,
    dtype="bfloat16",
    extra_sampling={"peft_config": lora_config},
)

job_id = client.create_job(sub_jobs=[training, sampling])
client.wait_for_job(job_id)

training_id = f"{job_id}:training:0"
sampling_id = f"{job_id}:sampling:0"

# Submit a DSSST1 training batch, then update the adapter parameters.
request_id = client.forward_backward(job_id, fwd_bwd_dssst1_frame)
client.poll_request(job_id, request_id)

request_id = client.step(job_id, learning_rate=1e-4)
client.poll_request(job_id, request_id)

# Broadcast only LoRA tensors. Later syncs update the same resident adapter.
request_id = client.weight_sync(
    job_id,
    source_sub_job_id=training_id,
    target_sub_job_ids=[sampling_id],
    weight_format="lora",
)
client.poll_request(job_id, request_id)

# Generation now uses the synchronized resident adapter.
request_id = client.generate(
    job_id,
    prompts=["Write a short proof."],
    sampling_params={"max_tokens": 256, "temperature": 0.7},
)
result = client.poll_request(job_id, request_id)
```

---

## 14. Model deployments

A model deployment is an ephemeral, OpenAI-compatible evaluation endpoint. The
server has no separate deployment resource: a deployment is an ordinary job
([section 5](#5-control-plane-endpoints)) with exactly one `sampling` sub-job
whose `inference_config.deployment_class` is `"EVALUATION"`. The
`CortexTrainingClient.*_model_deployment*` methods and the `cortex-training
model-deployment` CLI build on the job endpoints and never show the backing
job or sub-job IDs.

### 14.1 Deployment IDs

A deployment is identified by the `deployment` output key and the CLI's
`DEPLOYMENT` positional argument. Today that value is always the ID
`sf-ct-<UUID>`, where `<UUID>` is the backing job's canonical lowercase UUID;
names may be accepted later under the same key and argument. Inputs that do
not use exactly this form are rejected before any request is sent
(`deployment must currently be an sf-ct-<UUID> ID`). Use the deployment ID as
the OpenAI `model`; the base URL is `<prefix>/inference/v1` for the
deployment's database and schema (`model_deployment_base_url`).

### 14.2 Deployment spec

`create` accepts one JSON object with exactly these three top-level fields:

```json
{
  "deployment": {"class": "EVALUATION", "min_replicas": 3, "gpus_per_replica": 2, "gpu_type": "H200"},
  "source": {"type": "base_model", "model": "Qwen/Qwen3-1.7B"},
  "inference": {"max_seq_len": 8192}
}
```

`deployment` describes the deployment itself: its class and capacity.

A checkpoint source is `{"type": "checkpoint", "job_id": ..., "checkpoint_id": ...}`
and must name a completed `weights-only` checkpoint
(`recipes/api_quickstart/model-deployment-checkpoint.json`). It names no model: the
server already knows the source job's model and rejects one that does not
match it, so `create` reads the training sub-job's `model_name` from
`GET /{source job_id}` and sends that. `model` is rejected as an unknown key.

The checkpoint can come from a full fine-tune or a LoRA training job. For a
LoRA checkpoint, the server copies the source training sub-job's
`training_config.peft_config` into the deployment, so it serves the base model
with the trained adapter. The spec has no LoRA settings, and `peft_config`
cannot be set on a deployment.

Client-side checks (all others are left to the server):

| Field | Rule |
|---|---|
| every object | no unknown keys; an error names both unknown and missing keys, so a typo such as `min_replica` is reported as such |
| `deployment.class` | exactly `"EVALUATION"`; only `EVALUATION` is supported today |
| `source.type` | `base_model` or `checkpoint`; every source field a non-empty string |
| counts | positive integers (`true`/`false` and fractions rejected) |
| `deployment.gpu_type` | `H200`, `B200`, or `B300`, case-insensitive |
| `gpus_per_replica` | 1, 2, 4, or 8 |
| `min_replicas * gpus_per_replica` | at most 8, or a multiple of 8 |

Which models can be deployed, and the minimum GPUs per replica each needs, is
decided by the server (`Unsupported model for a model deployment: <name>`), as
for ordinary sampling jobs. New models need no client release. `create
--dry-run` runs only these client-side checks and prints the normalized spec;
it does not read a checkpoint's source job.

The spec becomes a `POST /` body with `idle_timeout_seconds: 0`, top-level
`hardware` set to `gpu_type`, and one sampling sub-job with
`inference_config.{deployment_class, num_replicas, n_gpus, max_seq_len}`,
`inference_config.vllm_config.{tensor_parallel_size, max_model_len}`, and
`source_checkpoint_info: {source_job_id, checkpoint_id}` for a checkpoint
source. `n_gpus` is `min_replicas * gpus_per_replica`.

### 14.3 Statuses

Deployment statuses are derived from the job status
([section 3.2](#32-job-states)):

| Job status | Deployment status |
|---|---|
| `pending` | `PENDING` |
| `placing`, `initializing` | `INITIALIZING` |
| `running` | `READY` |
| `failed` | `FAILED` |
| `cancelling` | `DROPPING` |
| `cancelled`, `done` | `DROPPED` |
| `terminated` | `TERMINATED` |
| anything else | the job status upper-cased (e.g. `UNKNOWN`) |

`wait_for_model_deployment` returns on `READY`. It raises
`ModelDeploymentWaitError` on `FAILED` or `TERMINATED` (naming the failure
code and pointing to `download-log`) or `DROPPED`, and
`ModelDeploymentWaitTimeout` (a `TimeoutError`) when the poll timeout
(`--timeout`, default `--poll-timeout`) passes. The timeout message says the
deployment is still running and holding GPUs. Both carry the last observed
metadata as `deployment`; `create --wait` prints that instead of the initial
`PENDING`. Ctrl+C during `wait` or `create --wait` prints
`interrupted; sf-ct-<UUID> is still running ...` and exits with code 130. No
deployment is dropped automatically.

The server reports why a job is in its status as `reason`, a short token such
as `sub_job_failed`, and `reason_details`, a readable sentence that explains it.
For `FAILED` and `TERMINATED`, these are shown as `failure: {code, message}`
(`code` is `reason`, `message` is `reason_details`), and `wait` includes both
in its error. In any other status, they are shown as `status_reason` and
`status_details` (e.g. `quota_filled` and `needs 8 GPUs; 64 currently
active`), except that the server's `cancelled_by_client` is not repeated for
`DROPPING`/`DROPPED`, where the status already says it; its `reason_details`
is omitted with it. Backing job and sub-job IDs in `reason_details` are
replaced by the deployment ID. `base_url` is shown only while the deployment
can serve: it is omitted for `FAILED`, `DROPPING`, `DROPPED`, and
`TERMINATED`.

### 14.4 Reading deployments

In the CLI, `create`, `create --wait`, and `wait` print the same summary:
`deployment`, `status`, `base_url`, and `failure` when present. `list` prints
only `deployment` and `status` per deployment, oldest first, and `describe`
prints the full record. The client methods return the full metadata in every
case.

`list_model_deployments` (`GET /`, keeping only deployments) and
`describe_model_deployment` (`GET /{job_id}`) report what the server stored and
do not re-check admission rules, so a deployment using a model or GPU shape
this client does not know about is still listed. A record is rejected only if
it is malformed: non-positive counts,
`n_gpus != num_replicas * tensor_parallel_size`, or
`max_seq_len != max_model_len`.

`list --status STATUS` (repeatable, case-insensitive) keeps only deployments in
the given statuses; an unknown status is rejected.

`describe` groups the deployment's configuration under `spec` for readability,
in the same `deployment` / `source` / `inference` shape. It is a read view, not
a `create` input: it may gain read-only fields, and it is not guaranteed to be
accepted by `create`.

Missing metadata is reported, not guessed:

- `gpu_type` (and `describe`'s `spec.deployment.gpu_type`) is `null` when the
  server returns no `hardware`, e.g. for a job read while the server cannot
  reach its scheduler.
- The source is `checkpoint` only when the sub-job carries
  `source_checkpoint_info` (see [section 15](#15-known-limitations), item 5).

### 14.5 Dropping

`drop_model_deployment` looks the job up with `GET /{job_id}` and refuses a job
that is not a deployment. If the deployment is `PENDING`, `INITIALIZING`, or
`READY` (or in a status this client does not know), it calls
`POST /{job_id}:cancel`, and the status moves through `DROPPING` to `DROPPED`.
The other statuses are handled without a cancel, as described below.

Dropping is idempotent. The server's job-not-found error counts as already
dropped; any other 404 is raised, since the deployment may still be running. A
`DROPPING` or `DROPPED` deployment is reported as dropped without a second
cancel.

A `FAILED` or `TERMINATED` deployment holds no GPUs and cannot be dropped: the
server rejects cancelling a failed or terminated job
([section 5.5](#55-cancel-job---post-job_idcancel)). `drop` does not call
cancel for it and returns `dropped: false` with its status. This is not an
error, and the CLI still exits `0`:

```json
{"deployment": "sf-ct-<UUID>", "dropped": false, "status": "FAILED"}
```

Every deployment lookup (`describe`, `wait`, `drop`, and the downloads) treats
the server's job-not-found error (code `517601`) as final on the first attempt,
even for a deployment this client created in the last 120 seconds, which other
lookups keep retrying ([section 3.5](#35-retries)).

### 14.6 Logs and metrics

`download_model_deployment_logs` (execution logs by default, or
`log_type="stdout"`) and `download_model_deployment_metrics` use the downloads
in [section 12](#12-logs-and-events), then write every file directly into the
output directory under its server filename. Sub-job directories are flattened
away; a deployment has exactly one sub-job, so its filenames do not collide. A
file from an earlier download into the same directory is replaced.

Results list each `filename` and `saved_path`. The stdout and metrics results
also include `chunk_count`. Every saved file has mode `0600`.

File contents are written as the server produced them, so a log line may
mention the backing job or sub-job ID.

### 14.7 Errors

Every HTTP error from a deployment method, including `create` and the requests
made by the log and metrics downloads, is replaced by

```text
model deployment sf-ct-<UUID> request failed with HTTP <status> (<code>): <message> (snowflake request id: <id>)
```

`create` has no deployment ID yet, so its errors start with
`model deployment request failed`. The server's error code and message are
kept, so "feature disabled", "not authorized", and "unsupported model" stay
distinguishable. The backing job ID is replaced by the deployment ID in the
message, and the request URL and raw body are never shown. A job-not-found
error becomes `model deployment sf-ct-<UUID> not found in <DB>.<SCHEMA>`.

Request debug log lines written while a deployment method runs also name the
deployment ID instead of the backing job ID. Local filesystem errors are
raised unchanged.

### 14.8 Calling a deployment

`CortexTrainingClient` does not call the deployment itself; any
OpenAI-compatible client does. This is the behavior a caller sees. The
`test_tc6_*` cases in `tests/test_model_deployment_e2e.py` check the path,
auth, `model`, and length-limit rows against a `READY` deployment; the error
rows are not exercised by a live test. The streaming example
(`recipes/model_deployment/streaming.py`) codes against it.

| Item | Behavior |
|---|---|
| Path | `POST {base_url}/chat/completions`, with `base_url` = `<prefix>/inference/v1` ([section 14.1](#141-deployment-ids)) |
| Auth | `Authorization: Bearer <PAT>` and `X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN` |
| `model` | The deployment ID, `sf-ct-<UUID>`; responses echo it as `model` |
| Length limit | `max_completion_tokens`. `max_tokens` is rejected with HTTP 400, `{"error": {"type": "invalid_request_error", "param": "max_tokens", "code": "unsupported_parameter"}}` |
| Not ready | HTTP 503 `endpoint_failed` while the deployment is `PENDING` or `INITIALIZING`, and possibly briefly after it is dropped. Check its status with `get_model_deployment` |
| Unavailable | HTTP 503 on a `READY` deployment: `endpoint_failed` when it is temporarily unavailable, or `frontend_overloaded` when the service is busy. Retry shortly |
| Not found | HTTP 404 `model_not_found` for an unknown or malformed ID, a deployment that has failed or been dropped, or one the caller does not own or that lives in another schema. The last two are deliberate, so a caller cannot tell whether someone else's deployment exists |
| Not permitted | HTTP 403 `feature_disabled` when Cortex Training or model deployments are not enabled for the account, or its trial has expired or has no expiration configured; the message says which. HTTP 403 `deployment_class_not_served` when the deployment exists but this endpoint does not serve its class |
| Routes off | HTTP 405 with an empty body when the inference routes are turned off for the account, or for a wrong HTTP method |

Without `stream`, the response is one `chat.completion` object with
`choices[0].message.content`, `finish_reason`, and `usage`.

With `"stream": true`, the response is `Content-Type: text/event-stream` with
one `data: <chat.completion.chunk JSON>` line per chunk and a final
`data: [DONE]`:

1. The first chunk's delta is `{"role": "assistant", "content": ""}`.
2. Each following chunk's `choices[0].delta.content` is the next text.
3. The last chunk before `[DONE]` has an empty delta and a non-null
   `finish_reason` (`stop`, or `length` at `max_completion_tokens`).
4. With `"stream_options": {"include_usage": true}`, every chunk carries
   `"usage": null` and one more chunk, with `"choices": []`, carries the token
   counts.

The status is already 200 once streaming starts, so a later failure cannot
change it. The example treats a chunk with an `error` key, a `data` line that
is not JSON, or a stream that ends without `[DONE]` as a failed call.

`GET {base_url}/models` is meant to list the `READY` deployments the caller
owns in the schema, but see [section 15](#15-known-limitations), item 6.

---

## 15. Known limitations

These are current gaps, not supported API behavior:

1. `save(checkpoint_id=...)` sends a field that is absent from the server's
   `SaveRequest`, so a caller-selected id is not honored. Use the server-assigned
   id from `stage_path` or the job's checkpoint list (section 6.3).
2. `tail_events()` and `stream_events()` send an operation type the server does
   not accept (see [section 7.9](#79-unsupported-zone-events-client-helper)).
3. `wait_for_job()` does not treat `terminated` as terminal, so a torn-down job
   polls until `poll_timeout` rather than failing immediately.
4. Generate prompt validation resolves `max_seq_len` from the first sub-job
   carrying an `inference_config` rather than matching `job_type="sampling"`. A
   `log_probability` sub-job listed first therefore supplies the wrong window.
5. The server does not yet echo `source_checkpoint_info` when a job is read
   back (it is accepted on create), so a checkpoint model deployment is
   reported with a `base_model` source by `list_model_deployments`,
   `describe_model_deployment`, and `wait_for_model_deployment`
   ([section 14.4](#144-reading-deployments)).
6. `GET {base_url}/models` has been seen to return HTTP 200 with an empty
   `data` list while a `READY` deployment in the same schema served chat
   completions ([section 14.8](#148-calling-a-deployment)). Use
   `list_model_deployments` or `cortex-training model-deployment list` to find
   deployments.
