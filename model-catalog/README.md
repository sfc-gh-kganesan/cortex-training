# Cortex Training Model Catalog

The supported models, and a tuned job file for each model, context, GPU type and
mode. The examples and `python -m recipes.utils` build their jobs from these
files unless you pass your own job file; see
[How an example builds its job](../recipes/README.md#how-an-example-builds-its-job).

## Layout

```text
model-catalog/
├── models/<slug>/                 # one folder per model
│   ├── model.json                 # model ID, license, architecture, validation dates
│   ├── standard/<hw>-<kind>-<mode>.json   # the model's smaller max_seq_len tier
│   └── long/<hw>-<kind>-<mode>.json       # the model's larger max_seq_len tier
├── schema.json                    # schema for model.json
├── validate_catalog.py
└── smoke_test_catalog.py
```

- `<slug>` is the last part of the model ID, in lowercase:
  `Qwen/Qwen3.6-35B-A3B` → `qwen3.6-35b-a3b`.
- `<hw>` is `h200`, `b200` or `b300`, and matches the file's `hardware` field.
  All files are H200 today.
- `<kind>` is `training` (one training sub-job) or `inference` (one sampling
  sub-job).
- `<mode>` is `lora` (the file has a `peft_config`) or `full`.

Each file is a complete create-job body (see the
[create-job schema](../docs/reference/rest-api.md#8-create-job-schemas)), and its
existence is the claim of support. The catalog has no RL files: an RL job
combines a model's training and inference files of the same context, hardware
and mode.

`model.json` holds:

- `model_id`
- `license` (a name and URL)
- `architecture`: `dense` or `moe`. MoE models also record
  `moe: {num_moe_layers, topk, num_experts}`.
- `last_validated`: the date of the last successful live run for each job
  file, or `null`

## Updating the catalog

1. Add or change job files in the model's folder. For a new model, add a
   folder with a `model.json`. No example code changes.
2. In `model.json`, set the file's entry in the `last_validated` map to `null`
   until a live smoke test (below) passes, then to the test date.
3. Keep [`docs/_data/models.yaml`](../docs/_data/models.yaml) (its header
   explains each field) and the table in
   [model compatibility](../docs/reference/model-compatibility.md) in sync.
4. Validate:

   ```bash
   python model-catalog/validate_catalog.py
   python -m pytest tests/test_model_catalog.py -q
   ```

The validator checks:

- file names and `hardware`, `model_name` and LoRA mode are consistent with
  `model.json`;
- each file passes `cortex-training submit --dry-run`;
- GPU counts are 1, 2, 4 or multiples of 8, and `sp_size` and
  `vllm_config.tensor_parallel_size` divide them;
- long-context files have a larger `max_seq_len` than standard ones;
- every training and inference pair builds a valid RL job, and the default
  distillation student and teacher jobs build.

## Smoke testing

Print the smoke-test plan for a catalog file without submitting it: a JSON
list with one `{"job", "request"}` entry per `--job`, where `request` is the
create-job body:

```bash
python model-catalog/smoke_test_catalog.py --model-id Qwen/Qwen3.8-27B --context long --job inference-full
```

`--job` is `<kind>-<mode>` (`training-lora`, `inference-full`, ...), or
`rl-lora` / `rl-full` for the composed RL job. You can repeat it. `--context`
defaults to `standard`, and `--hardware` to `h200`.

To run a job live, add `--submit`. The job runs a minimal probe once it's
running, and is cancelled afterwards, even if a probe fails. The probe is one
generate for inference, one forward-backward for training, and both for RL. The
PAT is read only from the environment, so it stays out of shell history:

```bash
export SNOWFLAKE_HOST='ACCOUNT.snowflakecomputing.com'
export SNOWFLAKE_DATABASE='CORTEX_TRAINING_DB'
export SNOWFLAKE_SCHEMA='PUBLIC'
read -rs SNOWFLAKE_PAT; export SNOWFLAKE_PAT

python model-catalog/smoke_test_catalog.py --model-id Qwen/Qwen3.8-27B \
  --context long --job inference-full --full-context-prefill --submit

unset SNOWFLAKE_PAT
```

The default probes check only startup and basic execution. Validating a
context-length claim takes the full-context flags:

| Flag | Effect |
|---|---|
| `--full-context-prefill` | Inference: prefill `max_seq_len - 1` tokens, then generate one |
| `--full-context-training` | Training: forward-backward at `max_seq_len` for the full batch |
| `--training-step` | Also run an optimizer step |
| `--checkpoint-round-trip` | Step, save a resumable checkpoint, load it back, then delete it |
| `--memory-telemetry` | Report peak GPU and CPU memory per step |
| `--training-sp-size N` | Try a sequence-parallel size before changing a file. Adjusts `train_batch_size` to `n_gpus / N` |
| `--debug-image-tag TAG` | Use a specific backend build |
