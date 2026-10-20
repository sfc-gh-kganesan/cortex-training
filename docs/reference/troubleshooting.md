# Troubleshooting

For authentication, URL, and server error messages, see the
[CLI troubleshooting section](cli.md#troubleshooting).

For recipe failures, first capture:

- The recipe command with secrets removed
- Job and sub-job IDs
- The installed client version, from `python -c "import cortex_training; print(cortex_training.__version__)"`
- Model, precision, GPU count, and sequence length
- The Snowflake request ID. The CLI prints it on server errors. In the Python
  SDK, HTTP errors start with `[snowflake request id: <id>]` when the response
  carried one. A server-side failure with no response of its own (a request
  or job that ended `failed`, a poll that timed out, an undecodable response,
  or a connection error) logs a `WARNING` such as
  `[last snowflake request id: <id>] poll_request failed: ...`; that is the
  operation's most recent request, which did not necessarily fail. Other
  errors, such as client-side validation or a job that reached `done` or
  `cancelled`, log nothing.
- Downloaded execution logs, from `cortex-training download-log JOB_ID`

Recipe-specific failure modes are documented beside the recipe rather than
accumulated on this page.
