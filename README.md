# Cortex Training Client

Python SDK, command-line tools, runnable recipes, and documentation for the
Cortex Training REST API.

## Start Here

- [Follow the getting-started path](docs/getting-started/README.md)
- [Run the first supervised fine-tuning job](docs/getting-started/first-sft-run.md)
- [Browse runnable recipes](recipes/README.md)
- [Check model and training-method compatibility](docs/reference/model-compatibility.md)
- [Use the CLI and Python client](docs/reference/cli.md)
- [Read the REST API reference](docs/reference/rest-api.md)

## Install

Requires Python 3.10 or later.

This project uses [uv](https://docs.astral.sh/uv/). Install it first if you
have not:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
uv pip install git+https://github.com/snowflakedb/cortex-training.git
```

For local development:

```bash
git clone https://github.com/snowflakedb/cortex-training.git
cd cortex-training
uv pip install -e .
```

(`pip` works in place of `uv pip` throughout if you prefer.)

The package installs:

- `cortex-training`, for submitting and managing jobs
- `cortex-training tui`, for viewing job logs
- `cortex_training`, the Python SDK

Verify the command entry points:

```bash
cortex-training --help
cortex-training tui --help
```

## Usage

```bash
cortex-training list
cortex-training submit examples/api/training.json
cortex-training tui JOB_ID
```

```python
from cortex_training import CortexTrainingClient, SubJobConfig
```

Connection settings use `CORTEX_TRAINING_*` and `SNOWFLAKE_*` environment
variables. Login state is stored under `~/.config/cortex-training/`, and TUI
cache state is stored under `~/.cache/cortex-training/`, unless their existing
override variables are used.

See the [CLI reference](docs/reference/cli.md) for commands and configuration.

## Repository Map

| Path | Purpose |
|---|---|
| `docs/` | Getting started material, concepts, guides, and reference |
| `recipes/` | End-to-end training, sampling, and evaluation workflows |
| `examples/api/` | Small JSON examples for individual API operations |
| `examples/config/` | Connection configuration templates |
| `src/cortex_training/` | Installable Python client |
| `tests/` | Client and CLI tests |
| `cluster-status.py` | Optional watch view of running jobs and GPU usage |

`cluster-status.py` is a standalone helper that shells out to `cortex-training
list`; it needs a working connection but is not imported by the package. It
refreshes every 5 seconds by default:

```bash
python cluster-status.py            # live view
python cluster-status.py --once     # print one summary and exit
```
