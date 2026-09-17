# Jobs and Sub-Jobs

A job is the top-level lifecycle resource. It contains one or more sub-jobs:

- `training` handles forward/backward requests, optimizer steps, and training
  checkpoints.
- `sampling` handles generation.
- `log_probability` configures log-probability workers.

A job supports **zero or one** `training` sub-job, and any number of `sampling`
and `log_probability` sub-jobs. A create request carrying a second training
sub-job is rejected — client-side before it is sent, and by the server for any
other caller.

Internal sub-job identifiers use the form:

```text
{job_id}:{sub_job_type}:{index}
```

For example, a colocated RL job commonly has `{job_id}:training:0` and
`{job_id}:sampling:0`.

For complete request and routing details, see the
[REST API reference](../reference/rest-api.md). GPU type is a job-level field;
see [GPU hardware](hardware.md).
