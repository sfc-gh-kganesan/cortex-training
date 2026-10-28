# Example Template

Each runnable example has its own folder next to the shared loop it uses. An RL
task, for example, goes in `recipes/rl_quickstart/<task>/` and reuses `recipes/rl_quickstart/runner.py`:

```text
recipes/rl_quickstart/
├── runner.py               # shared loop
├── overrides.json          # config changes for every RL run
└── <task>/
    ├── __init__.py
    ├── train.py            # entry point: the chz Config and main()
    ├── task.py             # dataset, prompt, reward, evaluator
    ├── example.yaml        # metadata
    ├── data/               # optional
    └── *.ipynb             # optional
```

## Code

- Build the job with `build_job` (or `make_distillation_config` for a student
  and a teacher) from `recipes/utils.py`, and ship no job files. See
  [How an example builds its job](../../recipes/README.md#how-an-example-builds-its-job).
- Give `Config` the same job arguments as the existing entry point of its
  kind, so commands stay uniform: copy them from
  `recipes/sft_quickstart/conversational/train.py` (training), `recipes/inference/serve.py`
  (inference endpoint), or `recipes/distillation/on_policy/train.py` (a
  student and a teacher model). A job-field override such as `lr=` defaults to
  `None`, which keeps the job's value, unless the example needs its own
  default, as `recipes/sft_quickstart/dare/train.py` does.
- Keep task code in `task.py`. Import shared helpers from `runner.py` and
  `recipes/utils.py`. A helper that another module uses is public (no leading
  underscore).
- Add tests for local validation logic where practical.

## README

Cover these, briefly, and link to `recipes/README.md` and `docs/` for anything
shared rather than restating it:

1. What it does, and the method
2. Default model, dataset and hardware
3. The run command
4. An options table
5. Expected results and metrics
6. Checkpoint and evaluation
7. Troubleshooting
8. Provenance, when adapted from another project

## example.yaml

Records `id`, `title`, `status`, `task`, `methods`, `datasets`, `precision`,
`provenance` (when adapted), and `last_validated`: the date of the last verified
run, or `null` if the example has changed since. Leave out `models` and
`default_training_gpus`, since the model catalog owns those. Distillation is
the exception: it lists its student and teacher models.
