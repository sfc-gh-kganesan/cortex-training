# Cortex Training Recipes

Runnable, end-to-end workflows and framework integrations. Each catalog-backed recipe builds its job from the
[model catalog](../model-catalog/README.md): you pick the model, context and GPU
type on the command line instead of writing a job file.

| Recipe | Method | Dataset / Environment |
|---|---|---|
| [Conversational SFT](sft_quickstart/README.md) | LoRA or full-parameter SFT | Hugging Face chat datasets, GSM8K, identity |
| [DARE-Bench agent SFT](sft_quickstart/dare/README.md) | Full-parameter multi-turn tool SFT + agent eval | DARE-Bench |
| [Math GRPO](rl_quickstart/README.md) | Reinforcement learning | Hendrycks MATH, MATH-500 |
| [On-policy distillation](distillation/README.md) | Distillation from a frozen teacher | GSM8K |
| [Inference endpoint](inference/README.md) | Serve, generate, evaluate open weights or a checkpoint | MATH-500, GSM8K, identity |
| [Model deployment streaming](model_deployment/README.md) | OpenAI-compatible model deployment, streamed chat | Open weights, checkpoints |
| [DFlash2 draft](dflash2/README.md) | Speculative-decoding draft training, then serving with the target | Target-written chats (UltraChat prompts) |
| [SkyRL GRPO](skyrl/gsm8k_grpo/README.md) | SkyRL trainer on CPU driver + Cortex training/sampling sub-jobs | GSM8K |
| [Tinker Cookbook](tinker/README.md) | Run `tinker-cookbook` recipes or `ServiceClient` on Cortex | GSM8K, MATH, etc. |
| [Harbor RL](harbor/README.md) | Go from `harbor run` evals to RL on the same tasks, sandbox and verifier | Any Harbor tasks (`Terminal-Bench 2.0`, SWE-smith, etc.) |
| [OpenHands RL](openhands/README.md) | Train an OpenHands agent with online GSPO | SWE-smith code localization |
| [Bring your own agent](bring-your-own-agent.md) | Connect any custom agent harness and grader to Cortex RL | Custom agent tasks |

```text
recipes/
├── utils.py                 # make_config, build_job and shared helpers
├── logging.py               # metrics logging
├── sft_quickstart/          # runner.py (shared loop), overrides.json, conversational/, dare/
├── rl_quickstart/           # runner.py (shared loop), overrides.json, math_grpo/, configs/ (generated)
├── distillation/            # on_policy/, overrides.json, configs/ (generated)
├── inference/               # serve.py, generate.py, evaluate.py, overrides.json
├── dflash2/                 # generate_data.py, train.py, evaluate.py, overrides.json, drafts/<model>.json
├── model_deployment/        # deploy.py, streaming.py, configs/
├── api_quickstart/          # raw REST API request bodies
├── skyrl/                   # SkyRL GRPO recipe
├── tinker/                  # tinker-cookbook integration
├── harbor/                  # RL post-training on Harbor tasks, from harbor run evals
├── openhands/               # OpenHands agent RL post-training
└── bring-your-own-agent.md  # custom agent harness RL guide
```

## Prerequisites

From the repository root:

```bash
uv pip install -e .
uv pip install 'tinker-cookbook[math-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

Set up a Snowflake connection profile (see
[Set up the client](../docs/getting-started/setup.md)). A
[JSON connection file](../docs/getting-started/setup.md#json-connection-file)
(from [`connection.json.template`](../docs/getting-started/connection.json.template),
with account host and PAT) also works, through `config=`.
To log metrics to Weights & Biases or Snowflake, see
[experiment tracking](../docs/guides/operations/wandb.md).

## Run a recipe

```bash
# Add connection=<profile> or config=<file> to choose a connection; with neither,
# the default connection is used.
python -m recipes.sft_quickstart.conversational.train config=/path/to/config.json
python -m recipes.rl_quickstart.math_grpo.train config=/path/to/config.json
python -m recipes.distillation.on_policy.train config=/path/to/config.json
python -m recipes.inference.serve config=/path/to/config.json
python -m recipes.model_deployment.deploy config=/path/to/config.json
```

Every recipe resolves its connection the same way the CLI does:

- `connection=<name>` uses a profile from `~/.snowflake/connections.toml`
  (`$SNOWFLAKE_HOME/connections.toml` if `SNOWFLAKE_HOME` is set).
- `config=<file>` reads a legacy JSON file with `host`, `pat`, `database`, and
  `schema`, exactly as before: only the file is used, and environment variables,
  the saved login and profiles are ignored.
- With neither, the sources are tried in order: `CORTEX_TRAINING_CONNECTION`,
  `CORTEX_TRAINING_CONFIG`, complete credentials in the environment, the saved
  `cortex-training login`, then the Snowflake default connection profile.

Each run logs the target it connected to and the source, never the token. The
`model_deployment` recipe needs a PAT (see its README).

`config=` and `connection=` only choose the connection, never a job body. The SFT, RL and
inference recipes pick their job with these arguments (distillation uses
`student=` and `teacher=` in place of `model=`):

| Argument | Values | Default |
|---|---|---|
| `model=` | Exact model ID, e.g. `Qwen/Qwen3.6-35B-A3B` | `Qwen/Qwen3-8B` |
| `context=` | `standard` (the model's smaller window) or `long` | `standard` |
| `hardware=` | `h200`, `b200`, `b300` | `h200` |
| `lora=` | `true` or `false` | per recipe |
| `job_config=` | Your own job file, used instead of the catalog | none |

[Model compatibility](../docs/reference/model-compatibility.md) lists the
supported combinations. If you ask for one that doesn't exist, the error lists
the files the model has. Add `--help` to any recipe command to list all of its
arguments and defaults.

## How a recipe builds its job

The SFT, RL and inference recipes build their job in three layers, each
overriding the one before:

1. **Catalog file:** the model's job file for the selected context, hardware,
   kind and mode (see the [catalog layout](../model-catalog/README.md#layout)).
   An RL job combines the training and inference files.
2. **The recipe's `overrides.json`:** applies to every catalog-built run of
   that recipe. An SFT task folder can add its own, applied after
   `sft_quickstart/overrides.json`: `sft_quickstart/dare/overrides.json` trains DARE on 8 GPUs.
3. **Command line:** `lr=`, `train_batch_size=`, `max_seq_len=`. When you leave
   them out, the values from the layers below stay.

`job_config=FILE` replaces layers 1 and 2; the command-line layer still
applies on top of it. Without `job_config=`, an RL job on an MoE model also gets
[router replay](rl_quickstart/README.md#router-replay) filled in after the merge. The
final job is logged before it is submitted.

Distillation builds a student and a teacher job instead; see
[How the jobs are built](distillation/README.md#how-the-jobs-are-built). Its
`job_config=` and `teacher_job_config=` files are used as is, without its
sizing arguments.

Merge rules:

- Dictionaries merge key by key, and `null` removes a key.
- `sub_job_configs` entries are matched by `job_type`, so an override can
  change one sub-job without restating the others.
- Other lists are replaced whole.
- A `_comment` key, at any level, is a free-text note. It is stripped before
  merging.
- If a merged training sub-job's `sp_size` or `ep_size` no longer divides its
  `n_gpus`, or its `ds_config` batch fields no longer fit that layout, the build
  stops with an error instead of submitting the job.

For example, this override changes only the training learning rate:

```json
{"sub_job_configs": [{"job_type": "training", "training_config": {"optimizer": {"lr": 2e-5}}}]}
```

## Create a job config from the command line

`python -m recipes.utils` builds the same jobs and writes them as JSON:

```bash
# Training job: Qwen3-8B LoRA, standard context
python -m recipes.utils --model Qwen/Qwen3-8B --context standard --lora -o train.json

# The matching inference job
python -m recipes.utils --model Qwen/Qwen3-8B --context standard --lora --inference -o infer.json

# RL job, with an override file (repeat --override to apply several, in order)
python -m recipes.utils --model Qwen/Qwen3.6-35B-A3B --context long --lora --rl \
  --override my-override.json -o rl.json

# RL job from your own training and inference files
python -m recipes.utils --training-config train.json --inference-config infer.json --rl -o rl.json
```

`--hardware` defaults to `h200`. `-o -` writes to stdout. `--distillation`
builds the student and teacher jobs of the
[distillation recipe](distillation/README.md#how-the-jobs-are-built). Submit the result, or
pass it to a recipe with `job_config=`:

```bash
ct submit --dry-run rl.json
ct submit rl.json
```

## Where to make a change

| You want to | Change |
|---|---|
| Change one run | Command-line arguments, or `job_config=` with an edited file |
| Change a default for every run of a recipe | That recipe's `overrides.json` |
| Support a new model or GPU type | `model-catalog/models/<model>/`. No recipe code changes |
| Add an RL task | Copy `rl_quickstart/math_grpo/` and change `task.py` and `example.yaml` |
| Add an SFT dataset | `sft_quickstart/conversational/task.py` (see the [SFT README](sft_quickstart/README.md#datasets)) |

To add a recipe, see the [recipe template](../docs/contributing/example-template.md).
