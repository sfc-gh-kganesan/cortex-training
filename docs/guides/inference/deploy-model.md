# Deploy a Model for Streaming Inference

A model deployment serves a base model or a training checkpoint as an
OpenAI-compatible chat endpoint, for real-time (streaming) inference: each
request gets its reply token by token as it is generated. Deployments are
managed with `cortex-training model-deployment`, and you call them with any
OpenAI-compatible client, `requests`, or `curl`.

For a runnable version of this page, see the
[model deployment example](../../../recipes/model_deployment/README.md). To
generate through the Cortex Training generate API instead, see
[Serve a Training Checkpoint](serve-checkpoint.md).

Only `EVALUATION` deployments are supported today. They are ephemeral: a
deployment holds its GPUs until you drop it.

## 1. Write a spec

From a base model:

```json
{
  "deployment": {"class": "EVALUATION", "min_replicas": 1, "gpus_per_replica": 1, "gpu_type": "H200"},
  "source": {"type": "base_model", "model": "Qwen/Qwen3-1.7B"},
  "inference": {"max_seq_len": 8192}
}
```

From a training run, point `source` at a completed `weights-only` checkpoint.
It names no model; the client reads it from the training job. LoRA checkpoints
work the same way: the deployment serves the base model with your adapter,
using the LoRA settings of the training job.

```json
"source": {"type": "checkpoint", "job_id": "TRAINING_JOB_UUID", "checkpoint_id": "CHECKPOINT_ID"}
```

`gpus_per_replica` (1, 2, 4, or 8) is each replica's tensor-parallel size, and
`min_replicas * gpus_per_replica` must be at most 8 or a multiple of 8. Which
models can be deployed is decided by the server. Check a spec without creating
anything:

```bash
cortex-training model-deployment create --spec deployment.json --dry-run
```

## 2. Create the deployment

```bash
cortex-training model-deployment create --spec deployment.json --wait
```

`--wait` returns once the deployment is `READY`, which can take several
minutes on a cold start. It prints the deployment ID and the endpoint:

```json
{
  "base_url": "https://ACCOUNT.snowflakecomputing.com/api/v2/databases/CORTEX_TRAINING_DB/schemas/PUBLIC/cortex-training/inference/v1",
  "deployment": "sf-ct-DEPLOYMENT_UUID",
  "status": "READY"
}
```

Without `--wait`, `create` returns at `PENDING`; run
`cortex-training model-deployment wait sf-ct-DEPLOYMENT_UUID` later. If waiting
fails, times out, or is interrupted, the deployment keeps running until you
drop it.

From Python:

```python
from cortex_training import CortexTrainingClient

client = CortexTrainingClient.from_configs(database="CORTEX_TRAINING_DB", schema="PUBLIC")
deployment_id = client.create_model_deployment(spec)["deployment"]
ready = client.wait_for_model_deployment(deployment_id)
base_url = ready["base_url"]
```

## 3. Stream a chat completion

Authenticate with a Snowflake programmatic access token (PAT), the documented
method for calling a deployment, and pass the deployment ID as `model`. The
examples below read it from `PAT`:

```python
import os

PAT = os.environ["PAT"]
```

Call the deployment once it is `READY`: before that, requests return HTTP 503
(`endpoint_failed`). A `READY` deployment can also return 503 while it is
temporarily unavailable or the service is busy; retry shortly. An unknown,
failed, or dropped deployment, or one you do not own, returns HTTP 404.
Inference is not yet available on all accounts; until it is enabled for yours,
requests return HTTP 403 (`feature_disabled`). See
[REST API reference section 14.8](../../reference/rest-api.md#148-calling-a-deployment)
for every case.

With the `openai` package:

```python
from openai import OpenAI

openai = OpenAI(
    base_url=base_url,
    api_key=PAT,  # sent as "Authorization: Bearer <PAT>"
    default_headers={"X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN"},
)
stream = openai.chat.completions.create(
    model="sf-ct-DEPLOYMENT_UUID",
    messages=[{"role": "user", "content": "Write a haiku about snow."}],
    max_completion_tokens=256,
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

With `curl` (`-N` turns off buffering so events print as they arrive):

```bash
curl -N "$BASE_URL/chat/completions" \
  -H "Authorization: Bearer $PAT" \
  -H "X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model": "sf-ct-DEPLOYMENT_UUID", "stream": true, "max_completion_tokens": 256,
       "messages": [{"role": "user", "content": "Write a haiku about snow."}]}'
```

The streamed response is server-sent events: one `data: {...}` line per
`chat.completion.chunk`, whose `choices[0].delta.content` holds the next piece
of text, and a final `data: [DONE]`. The first chunk's delta carries only
`"role": "assistant"`, and the last chunk before `[DONE]` has an empty delta
and the `finish_reason`. Add `"stream_options": {"include_usage": true}` to get
one more chunk, with empty `choices`, that carries the token counts. Leave out
`"stream": true` to get one `chat.completion` object instead.

Limit the reply length with `max_completion_tokens`. The endpoint rejects
`max_tokens` with HTTP 400 (`"param": "max_tokens"`,
`"code": "unsupported_parameter"`).

## 4. Inspect

```bash
cortex-training model-deployment list --status READY
cortex-training model-deployment describe sf-ct-DEPLOYMENT_UUID
cortex-training model-deployment download-log sf-ct-DEPLOYMENT_UUID --output-dir ./logs
cortex-training model-deployment download-metrics sf-ct-DEPLOYMENT_UUID --output-dir ./metrics
```

A `FAILED` deployment reports `failure: {code, message}`; its execution logs
say why.

## 5. Drop it

```bash
cortex-training model-deployment drop sf-ct-DEPLOYMENT_UUID
```

`drop` is safe to retry. A deployment that already failed or terminated holds
no GPUs and reports `"dropped": false` with its status.

## Reference

- [CLI reference: Model Deployments](../../reference/cli.md#model-deployments)
- [REST API reference section 14](../../reference/rest-api.md#14-model-deployments):
  spec rules, statuses, drop, downloads, and errors, and
  [section 14.8](../../reference/rest-api.md#148-calling-a-deployment) for the
  chat completions request, streaming format, and errors
