# Deploy a Model and Stream Chat Completions

Deploy a base model or a weights-only training checkpoint as an
OpenAI-compatible endpoint, then stream a chat completion from it token by
token. The example uses `cortex-training model-deployment` through the Python
client and is the runnable counterpart of the
[model deployment guide](../../docs/guides/inference/deploy-model.md).

One run:

1. creates a model deployment from a spec,
2. waits until it is `READY`,
3. sends one chat completion with `stream: true` and writes each delta to
   stdout as it arrives,
4. logs the time to first token, and
5. drops the deployment, so it stops holding GPUs.

Unlike the [inference endpoint example](../inference/README.md), which drives a
sampling job through the Cortex Training generate API, a model deployment is
called with any OpenAI-compatible client and its backing job stays hidden.

## Prerequisites

Install the client from the repository root (`chz` runs the entry point):

```bash
uv pip install -e .
uv pip install chz
```

This example needs a PAT, not only a profile: the chat call sends it as a
Bearer token. Pass `config=` with a JSON file from
`docs/getting-started/connection.json.template` (`host`, `pat`, `database`,
`schema`). With `connection=` or the default profile, the example stops before
it creates a deployment, because profile connections send a session token and
the endpoint's documented authentication is the PAT.
Inference is not yet available on all accounts: until it is enabled for yours,
chat calls return HTTP 403.

## Hardware

The default spec, `configs/base-model.json`, deploys `Qwen/Qwen3-1.7B` on one
H200 (`min_replicas: 1`, `gpus_per_replica: 1`). Check capacity first:

```bash
cortex-training capacity
```

## Run

From a base model:

```bash
python -m recipes.model_deployment.deploy \
  config=/path/to/config.json
```

From a weights-only checkpoint saved by a training example, copy
`configs/checkpoint.json`, fill in the training `job_id` and `checkpoint_id`,
and pass it as `spec=`:

```bash
python -m recipes.model_deployment.deploy \
  config=/path/to/config.json \
  spec=/path/to/checkpoint.json
```

The checkpoint must come from a completed `save` whose type is `weights-only`.
A checkpoint spec names no model; the client reads it from the training job.

## Expected output

Status lines are logged to stderr, and the reply streams to stdout:

```text
INFO Created sf-ct-DEPLOYMENT_UUID; waiting for READY (this pulls the model and starts vLLM)
INFO sf-ct-DEPLOYMENT_UUID is READY at https://ACCOUNT.snowflakecomputing.com/api/v2/databases/CORTEX_TRAINING_DB/schemas/PUBLIC/cortex-training/inference/v1
Streaming shows each token as soon as it is generated, ...
INFO First token after 0.45s; 64 completion tokens in 0.57s (finish_reason=stop)
INFO Dropped sf-ct-DEPLOYMENT_UUID
```

A cold start pulls the model and starts vLLM, so `READY` can take several
minutes; the timings above are only illustrative.

## Options

| Option | Default | Meaning |
|---|---|---|
| `spec` | `configs/base-model.json` | Deployment spec JSON. A relative path is also looked up in this directory |
| `deployment` | none | Call a running `sf-ct-<UUID>` deployment instead of creating one. It is never dropped |
| `keep_deployment` | `False` | Leave the created deployment running for more calls |
| `timeout` | connection `poll_timeout` | Seconds to wait for `READY` |
| `prompt` | a short question | User message |
| `system` | none | Optional system message |
| `max_completion_tokens` | `256` | Completion token limit, sent as `max_completion_tokens`; the endpoint rejects `max_tokens` |
| `temperature` | `0.7` | Sampling temperature |
| `stream` | `True` | `False` sends one non-streaming request and prints the whole reply |
| `request_timeout` | `300` | Seconds for the chat request |

Keep one deployment and call it repeatedly:

```bash
python -m recipes.model_deployment.deploy config=/path/to/config.json keep_deployment=True
python -m recipes.model_deployment.deploy config=/path/to/config.json \
  deployment=sf-ct-DEPLOYMENT_UUID prompt="Write a haiku about snow."
cortex-training model-deployment drop sf-ct-DEPLOYMENT_UUID
```

## Deployment spec

```json
{
  "deployment": {"class": "EVALUATION", "min_replicas": 1, "gpus_per_replica": 1, "gpu_type": "H200"},
  "source": {"type": "base_model", "model": "Qwen/Qwen3-1.7B"},
  "inference": {"max_seq_len": 8192}
}
```

Only `EVALUATION` deployments are supported today. `gpus_per_replica` is 1, 2,
4, or 8, and `min_replicas * gpus_per_replica` must be at most 8 or a multiple
of 8. Check a spec without creating anything:

```bash
cortex-training model-deployment create --spec recipes/model_deployment/configs/base-model.json --dry-run
```

See [REST API reference section 14](../../docs/reference/rest-api.md#14-model-deployments)
for every rule.

## Troubleshooting

- **HTTP 503 from chat completions.** The deployment is not `READY` yet, or
  is temporarily unavailable or busy. Check it with
  `cortex-training model-deployment describe sf-ct-DEPLOYMENT_UUID`, and retry
  shortly if it is `READY`.
- **HTTP 404 from chat completions.** The deployment ID is unknown, the
  deployment failed or was dropped, or it belongs to another owner or schema.
- **HTTP 403 from chat completions.** The account may not call deployments:
  inference is not enabled yet, or its trial has expired. The error message
  says which.
- **HTTP 405 from chat completions.** The inference routes are turned off for
  the account.
- **`Unsupported model for a model deployment`.** The server does not deploy
  that model; pick another.
- **Waiting fails with `FAILED`.** Read the deployment's logs with
  `cortex-training model-deployment download-log sf-ct-DEPLOYMENT_UUID`.
- **Waiting times out or is interrupted.** The example still drops a deployment
  it created. With `keep_deployment=True`, or after the process is killed, run
  `cortex-training model-deployment list` and `drop` what is left.
- **A checkpoint source is shown as `base_model`.** The server does not echo
  the checkpoint source yet
  ([known limitation 5](../../docs/reference/rest-api.md#15-known-limitations)).

`last_validated` in `example.yaml` is the date of the last end-to-end run
against a live account, with the default spec. The live checks of the endpoint
are `test_tc6_*` in `tests/test_model_deployment_e2e.py`.
