# Troubleshooting

For authentication, URL, and server error messages, see the
[CLI troubleshooting section](cli.md#troubleshooting).

For recipe failures, first capture:

- The recipe command with secrets removed
- Job and sub-job IDs
- The installed client version, from `python -c "import cortex_training; print(cortex_training.__version__)"`
- Model, precision, GPU count, and sequence length
- The Snowflake request ID, which the CLI prints on server errors
- Downloaded execution logs, from `cortex-training download-log JOB_ID`

Recipe-specific failure modes are documented beside the recipe rather than
accumulated on this page.

## Cookbook install (git SSH timeout)

`uv pip install 'tinker-cookbook @ git+https://github.com/...'` uses git. If your
environment rewrites `https://github.com` to SSH on a blocked port, the fetch
hangs or exits 128. Clone the `nightly` branch over HTTPS with
`GIT_CONFIG_GLOBAL=/dev/null` and install from that checkout (see
[Recipes](../../recipes/README.md)).

## HTTP 429 and capacity after cancel

`POST /cortex-training` can return **429 Too Many Requests** while jobs are
draining. `cortex-training capacity` may still show `in_use_gpus` at the account
ceiling after `cancel`, even when `cortex-training list --status running` is
empty. Wait and re-check `available_gpus` before the next recipe.

Default Math GRPO needs **8** GPUs (4 train + 4 sample). Conversational SFT
Qwen3-8B needs **4**. Inference `qwen3_8b_*` sampling configs request **2**.
