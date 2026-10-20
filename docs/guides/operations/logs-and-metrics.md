# Logs and Metrics

Download complete execution logs:

```bash
cortex-training download-log JOB_ID --output-dir /path/to/logs
```

Download persisted stdout/stderr reconstructed as one file per sub-job:

```bash
cortex-training download-log JOB_ID --log-type stdout --output-dir /path/to/logs
cortex-training download-log JOB_ID --log-type stdout --output-dir /path/to/logs --resume
```

`--resume` continues a stdout download already in that directory, including
one that was interrupted, and fetches only what is missing. It does not
apply to execution logs. On a terminal, stdout and GPU metrics progress is
one stderr line that updates in place. A redirected stderr gets one final line
per sub-job. The JSON result stays on stdout.

Download reconstructed GPU metrics:

```bash
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics
cortex-training download-metrics JOB_ID --output-dir /path/to/metrics --resume
```

Metrics are written to `<output_dir>/<sub_job_id>/gpu.jsonl`.

Open a job dashboard in the terminal:

```bash
cortex-training tui JOB_ID
```

The **Logs** tab tails a running job or loads its saved console after it
finishes. The **Resources** tab charts the selected sub-job's GPU use and lists
each GPU; `w` cycles 10 minutes (the default), 30 minutes, 1 hour, 6 hours, or
the whole job, and `r` refreshes. Run `cortex-training tui` without a job id
to browse and search your jobs first. See [Job TUI](../../reference/cli.md#job-tui).

Recipe-level metrics are also written under each recipe's `log_path`.

Snowflake profile and PAT clients also emit best-effort client-side operation
metrics over Snowflake's OTLP endpoint. Failures of essential SDK methods, and
the outcome and timing of every download, are emitted automatically; other
successful outcomes require `CORTEX_TRAINING_ENABLE_SUCCESS_TELEMETRY=1`. See
[Client metrics](../../reference/python-sdk.md#client-metrics).
