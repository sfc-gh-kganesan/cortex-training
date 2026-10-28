<h3 align="center">
  <img src="assets/cortex_training_logo.svg" width=400px><br>
</h3>

<div align="center">

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cortex-training)](https://pypi.org/project/cortex-training/)

</div>

Cortex Training is Snowflake's serverless platform for post-training open-weight LLMs on managed GPU clusters. This repository contains the Python SDK, CLI, and ready-to-run examples, powered by the open-source [Arctic Platform](https://github.com/Snowflake-AI-Research/Arctic-Platform) engine. Run reinforcement learning, supervised fine-tuning, and inference serving without managing GPU infrastructure.

## Getting Started

1. **Clone and install** — this gives you the SDK, CLI, and all examples:
   ```bash
   git clone https://github.com/snowflakedb/cortex-training.git
   cd cortex-training
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```
   Alternatively, `uv pip install cortex-training` installs the SDK and CLI
   without the examples.
2. **Get access** — you need a Snowflake account with Cortex Training enabled.
   See [account setup](docs/getting-started/account_setup.md) for the full
   walkthrough and [authentication](docs/getting-started/authentication.md) for
   PAT generation and connection configuration.
3. **Connect** — configure a standard
   [Snowflake connection profile](docs/getting-started/setup.md), then:
   ```bash
   cortex-training --connection training capacity
   # With a configured default profile: cortex-training capacity
   ```

## Quick Example

### **Run an example** (recommended)

Fine-tune a chat model in one command:

```bash
uv pip install 'tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
python -m recipes.sft_quickstart.conversational.train \
  config=~/cortex-training-config.json \
  max_steps=50
```

`config=` is the JSON connection file described in
[setup](docs/getting-started/setup.md#json-connection-file). This
fine-tunes Qwen3-8B (full-parameter, four H200 GPUs) on a small bundled
dataset. See the [Conversational SFT example](recipes/sft_quickstart/README.md) for all
options.

### **CLI Mode**

The CLI runs the same step-level operations the examples call, one at a time:

```bash
cortex-training models                                           # list active configured models
cortex-training submit recipes/api_quickstart/rl.json --wait               # training + sampling job
cortex-training --job JOB_ID fwd-bwd recipes/api_quickstart/fwd-bwd.json   # forward-backward pass
cortex-training --job JOB_ID step --lr 1e-4                      # optimizer step
cortex-training --job JOB_ID generate recipes/api_quickstart/generate.json # sample from the model
cortex-training cancel JOB_ID                                    # release GPUs
```

See the [CLI reference](docs/reference/cli.md) for the full command set. To use
an existing RL framework like SkyRL, see
[framework integrations](recipes/README.md).

## Examples

Examples are end-to-end post-training and inference workflows you can run as
is or customize. Each builds its job from the
[model catalog](model-catalog/README.md), which holds a tuned job file for each
supported model, context and GPU type.

- **[Conversational SFT](recipes/sft_quickstart/README.md)**: supervised fine-tuning on chat datasets with LoRA or full-parameter training.
- **[Math GRPO](recipes/rl_quickstart/README.md)**: reinforcement learning for mathematical reasoning with verifiable rewards.
- **[Inference](recipes/inference/README.md)**: serve a model or checkpoint, generate responses, and evaluate.

The [example index](recipes/README.md) lists every example and explains how to
customize the job.

## Documentation

- [Getting started](docs/getting-started/README.md): set up your environment, configure credentials, and run your first job
- [Concepts](docs/concepts/): jobs and sub-jobs, the training lifecycle, checkpoints, GPU hardware, and sizing
- [CLI reference](docs/reference/cli.md): all commands, flags, and usage examples
- [Python SDK reference](docs/reference/python-sdk.md): CortexTrainingClient API for programmatic workflows
- [REST API reference](docs/reference/rest-api.md): HTTP endpoints and wire format
- [Model compatibility](docs/reference/model-compatibility.md): supported models, methods, and hardware configurations
- [Framework integrations](recipes/README.md): run SkyRL, tinker-cookbook, or your own agent on Cortex Training

## Development

- **Contributing to the client or examples** — see the [contributing guides](docs/contributing/) for documentation guidelines and the example template. Run the tests with `python -m pytest tests/ -q` after `uv pip install -e ".[dev]"`.
- **Building a distributable package** — see [build instructions](build_wheel.sh).
