# API Request Examples

Small standalone request bodies for trying one API call at a time. They are
fixed examples, not tuned configurations. For a job you plan to run, build one
from the model catalog instead; see
[Create a job config from the command line](../README.md#create-a-job-config-from-the-command-line).

## Create-job bodies

| File | Creates |
|---|---|
| [`training.json`](training.json) | A training sub-job for `Qwen/Qwen3.6-35B-A3B` (`arctic` provider) |
| [`sampling.json`](sampling.json) | A sampling sub-job for `Qwen/Qwen3.6-35B-A3B` |
| [`rl.json`](rl.json) | Colocated training and sampling sub-jobs |
| [`external-weights-training.json`](external-weights-training.json) | A training sub-job from a Hugging Face model directory on a stage |
| [`glm-sampling.json`](glm-sampling.json) | A 16-GPU `zai-org/GLM-5.2-FP8` sampling sub-job with a 1M-token context and an FP8 MLA KV cache |

[`model-deployment-checkpoint.json`](model-deployment-checkpoint.json) is not a
create-job body: it is a model-deployment spec for
`cortex-training model-deployment create --spec` (see
[Deploy a Model for Streaming Inference](../../docs/guides/inference/deploy-model.md)).

```bash
ct submit --dry-run recipes/api_quickstart/training.json   # validate offline, print the body
ct submit recipes/api_quickstart/training.json --wait
```

From Python:

```python
import json
from pathlib import Path

from cortex_training import CortexTrainingClient

client = CortexTrainingClient.from_configs(connection="training")
body = json.loads(Path("recipes/api_quickstart/rl.json").read_text())  # training + sampling
job_id = client.create_job_from_body(body)["job_id"]
client.wait_for_job(job_id)
```

`create_job_from_body` checks the body the same way `submit --dry-run` does,
and rewrites each `peft_config` in place into its canonical form (the dict you
pass changes). See
[REST API section 5.1](../../docs/reference/rest-api.md#51-create-job---post-)
for every field.

## Operation bodies

These are readable specs that need a running job: `fwd-bwd.json` needs a
training sub-job and `generate.json` a sampling sub-job, so the Python code
above creates the job from `rl.json`. They are not raw wire bodies: the CLI
tokenizes `fwd-bwd.json` (this needs `transformers`) into a DSSST1 frame.

| File | Sends |
|---|---|
| [`fwd-bwd.json`](fwd-bwd.json) | A forward/backward batch: text, tokenizer and label settings |
| [`generate.json`](generate.json) | Prompts and sampling parameters |

```bash
ct --job JOB_ID fwd-bwd recipes/api_quickstart/fwd-bwd.json
ct --job JOB_ID step
ct --job JOB_ID generate recipes/api_quickstart/generate.json
```

From Python:

```python
from cortex_training import build_forward_backward_payload

spec = json.loads(Path("recipes/api_quickstart/fwd-bwd.json").read_text())
request_id = client.forward_backward(job_id, build_forward_backward_payload(spec))
loss = client.poll_request(job_id, request_id)["avg_loss"]

generate = json.loads(Path("recipes/api_quickstart/generate.json").read_text())
request_id = client.generate(job_id, prompts=generate["prompts"], sampling_params=generate["sampling_params"])
texts = [result["text"] for result in client.poll_request(job_id, request_id)["results"]]
```

See the [CLI reference](../../docs/reference/cli.md#run-a-forward-backward-smoke-test)
and [REST API section 6](../../docs/reference/rest-api.md#6-data-plane-endpoints).
