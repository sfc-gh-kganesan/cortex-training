# Cortex Training Client

Python SDK, command-line tools, runnable recipes, and documentation for the
Cortex Training REST API.

## Install

Requires Python 3.10 or later and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Create an environment and install the client:

```bash
uv venv
source .venv/bin/activate
uv pip install git+https://github.com/snowflakedb/cortex-training.git
```

The package includes the CLI, log TUI, and Python SDK. `pip install` also works
in place of `uv pip install` in an active Python environment.

## Log In

Create `~/cortex-training-config.json` using the
[connection template](examples/config/connection.json.template). Set `host` to
your Snowflake account hostname, `pat` to your programmatic access token, and
`database` and `schema` to the location of your Cortex Training endpoint.
Keep this file outside the repository and do not commit credentials.

Login validates the config and remembers its path for future CLI commands:

```bash
cortex-training login ~/cortex-training-config.json
```

The equivalent `cortex-training login --config ~/cortex-training-config.json`
form is also supported.

`ct` is an alias for `cortex-training`: every CLI example also works with `ct`.
See [connection setup](docs/getting-started/setup.md) for more detail.

## Try the CLI

```bash
cortex-training capacity           # Check available GPU capacity
cortex-training list               # List jobs
cortex-training get JOB_ID         # Inspect a job from the list
cortex-training tui                # Pick a job and view its logs
```

See the [CLI quick reference](docs/reference/cli.md#quick-reference) for
submission, training, generation, checkpoints, and log downloads.

## Run a Recipe

Start with [Run a Quick SFT Job](docs/getting-started/first-sft-run.md), a short
end-to-end supervised fine-tuning walkthrough.

Recipes require a repository checkout; they are not included in the installed
package:

```bash
git clone https://github.com/snowflakedb/cortex-training.git
cd cortex-training
```

Install the [recipe dependencies](recipes/README.md#prerequisites), then run
the chosen recipe's commands from the repository root. Recipes take an explicit
`config=/path/to/config.json` argument; use the same connection file you logged
in with.

| Task | Recipe |
|---|---|
| Fine-tune a chat model with LoRA or full-parameter training | [Conversational SFT](recipes/sft/conversational/README.md) |
| Train math reasoning with reinforcement learning | [Math GRPO](recipes/rl/math_grpo/README.md) |
| Serve a model or checkpoint, generate responses, and evaluate | [Inference endpoint](recipes/inference/README.md) |

Check `cortex-training capacity` and the recipe's GPU requirements before
starting a run. Browse the [recipe index](recipes/README.md) for all workflows.

## More Documentation

- [Getting started and prerequisites](docs/getting-started/README.md)
- [CLI commands and configuration](docs/reference/cli.md)
- [Python SDK](docs/reference/python-sdk.md)
- [REST API](docs/reference/rest-api.md)
- [Model and training-method compatibility](docs/reference/model-compatibility.md)
- [Job management and cluster status](docs/guides/operations/manage-jobs.md)
- [RL framework integrations](docs/integrations/README.md)

## Development

### Editable Install

From a repository checkout, with your Python environment active:

```bash
uv pip install -e ".[dev]"
```

### Build a Wheel

With `uv` installed, run:

```bash
./scripts/build_wheel.sh
```

The script builds the package using `pyproject.toml` in an isolated build
environment and writes the wheel to `dist/`. It can be invoked from any working
directory and does not install the package's runtime dependencies.

### Repository Map

| Path | Purpose |
|---|---|
| `model-catalog/` | Supported models, context limits, and recommended job profiles |
| `docs/` | Getting started material, concepts, guides, and reference |
| `recipes/` | End-to-end training, sampling, and evaluation workflows |
| `examples/api/` | Small JSON examples for individual API operations |
| `examples/config/` | Connection configuration templates |
| `src/cortex_training/` | Installable Python client |
| `tests/` | Client and CLI tests |
| `cluster-status.py` | Optional watch view of running jobs and GPU usage |
