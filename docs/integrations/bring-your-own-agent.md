# Bring your own agent

Train a model inside the agent you already run. The agent keeps its own loop,
tools, sandbox, and context handling. Training changes one thing: where the
agent sends its model calls. Each rollout gets its own OpenAI-compatible URL
and API key, served by the policy that is being trained. The model learns in
the same agent it will be deployed in.

[OpenHands](openhands.md) is the worked example. The code is in the
[Arctic Platform](https://github.com/Snowflake-AI-Research/Arctic-Platform)
repository: the endpoint and generator in `arctic_platform.integrations.harness`,
the example in `recipes/rl/openhands/code_localization`.

## How it works

Your agent connects in two places.

1. **The model endpoint.** For every rollout, training hands the agent a
   `base_url` and an `api_key`. Point the agent's OpenAI client at them. To the
   agent, a training rollout looks like ordinary inference.
2. **The rollout lifecycle.** A small adapter lets the trainer start a rollout,
   watch it, and fetch what it produced:

   ```text
   submit(request)          -> rollout_id
   get_status(rollout_id)   -> "running" | "completed" | "errored"
   collect_output(rollout_id) -> artifacts
   ```

A **grader** turns the artifacts into a reward. The agent never sees the
reward, so you can change how rollouts are scored without touching the agent.

The endpoint records every model call with the exact token ids the policy
sampled. When the next call replays the conversation plus the reply it got,
the endpoint reuses those sampled tokens instead of re-tokenizing the text, so
training sees what the model actually generated.

One training step:

1. SkyRL picks a batch of dataset rows. Each row is rolled out several times
   (`generator.n_samples_per_prompt`).
2. `submit` receives each rollout's row with its own URL and key. The rollouts
   in a batch run concurrently.
3. The trainer polls `get_status` until the rollout finishes, then calls
   `collect_output`.
4. The grader scores the artifacts. It also gets the recorded model calls.
5. The sampled tokens and rewards go to Cortex for a GRPO update, and the new
   weights go to the sampling GPUs.

## What the endpoint accepts

- `POST {base_url}/chat/completions` in OpenAI format, with
  `Authorization: Bearer {api_key}`. `base_url` already ends in `/v1`, so it
  goes straight into an OpenAI client.
- One completion per call (`n=1`), no streaming. Any `model` name is accepted.
- `tools`. Tool calls in the reply come back as OpenAI `tool_calls`. Hermes
  JSON and Qwen3.5 XML tool-call formats are parsed.
- `temperature`, `top_p`, `top_k`, `stop`, `seed`, and `max_tokens` or
  `max_completion_tokens`. `chat_template_kwargs` is passed to the chat
  template, for example `{"enable_thinking": false}`.

A call that does not continue the previous conversation starts a new context
in the same rollout. Context compaction, a subagent, or a new system prompt
all do this. Set `generator.step_wise_trajectories=true` to train on every
context. Without it, a rollout with more than one context fails the batch.

## Write the adapter

`submit` gets a `RolloutRequest`:

| Field | Meaning |
| --- | --- |
| `env_extra` | The dataset row |
| `base_url`, `api_key` | The model endpoint for this rollout |
| `routing_key` | A unique id for this rollout |
| `phase`, `global_step` | `train` or `eval`, and the step number |

This adapter runs an agent as a command and reads its answer from a file:

```python
import os
import subprocess
import tempfile
import threading

from arctic_platform.integrations.harness.generator import Grade
from arctic_platform.integrations.harness.generator import RolloutRequest


class CommandAdapter:
    """Runs the agent as a subprocess pointed at the rollout's model endpoint."""

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.runs = {}
        self.lock = threading.Lock()

    def submit(self, request: RolloutRequest) -> str:
        workdir = tempfile.mkdtemp(prefix="rollout-")
        env = dict(os.environ, OPENAI_BASE_URL=request.base_url, OPENAI_API_KEY=request.api_key)
        task = request.env_extra["problem_statement"]
        process = subprocess.Popen([*self.command, task], cwd=workdir, env=env)
        with self.lock:
            self.runs[request.routing_key] = (process, workdir)
        return request.routing_key

    def get_status(self, rollout_id: str) -> str:
        with self.lock:
            process, _ = self.runs[rollout_id]
        if process.poll() is None:
            return "running"
        return "completed" if process.returncode == 0 else "errored"

    def collect_output(self, rollout_id: str) -> dict:
        with self.lock:
            _, workdir = self.runs.pop(rollout_id)
        answer = os.path.join(workdir, "answer.txt")
        if not os.path.exists(answer):
            return {}
        with open(answer) as f:
            return {"answer": f.read().strip()}


def grade(episodes, artifacts, request) -> Grade:
    correct = artifacts.get("answer") == request.env_extra["expected_answer"]
    return Grade(reward=float(correct), reward_detail={"correct": float(correct)})
```

- `submit` must return without waiting for the agent. The trainer calls the
  three methods from worker threads, one rollout per thread.
- `collect_output` returns any JSON-serializable dict. Token ids are not part
  of it; the endpoint already has them.
- If a method raises, the rollout is recorded as an error and the grader gets
  `artifacts["error"]` with the traceback.

## Write the grader

The grader is `grade(episodes, artifacts, request) -> Grade`:

- `episodes` are the recorded model calls, with the sampled token ids.
- `artifacts` is what `collect_output` returned.
- `request` is the `RolloutRequest`, so `request.env_extra` has the dataset row.

`Grade(reward, reward_detail, exhausted=False)`. `reward` is the scalar for
GRPO. Each numeric key in `reward_detail` is logged as `reward/<key>`.
`exhausted=True` drops the rollout from the loss, for example when the agent
ran out of turns without an answer.

## Wire it in

Start from a copy of the OpenHands recipe:

1. Copy `recipes/rl/openhands/code_localization` and rename the
   `openhands_localization` package.
2. In `entrypoint.py`, `get_generator` builds the adapter and the grader.
   Replace those two with yours. Keep the endpoint (`ChatCompletionsProxy`)
   and `HarnessGenerator` as they are.
3. In the run script, point `trainer.override_entrypoint` at your package,
   replace the OpenHands `--with` packages with your agent's dependencies, and
   point `DATA_DIR` at your data.
4. Write the data as `train.parquet` and `validation.parquet`. Each row needs a
   `prompt` column of chat messages. Your adapter and grader read the other
   columns from `request.env_extra`.

Each rollout is saved as JSON under `$CKPT_DIR/trajectories/step_<n>/<phase>/`
with the reward and the artifacts. Return the conversation as
`artifacts["messages"]` to have it saved too. A rollout that failed is saved
as `.error` with the traceback.

| Piece | In the OpenHands example |
| --- | --- |
| Adapter | `OpenHandsAdapter`: one OpenHands conversation per rollout, in a thread |
| Sandbox | A fresh git checkout of the task repository |
| Grader | `grade_localization`: file + module + function F1 |
| Data | `prepare_data.py`: SWE-smith tasks from Hugging Face |
