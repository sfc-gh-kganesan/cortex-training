# GPU Hardware

A job selects one GPU type. The field is top-level `hardware` on the create-job
body, not on individual sub-jobs, so training and sampling in the same job
always share a pool.

## Supported values

| Value | When to use |
|---|---|
| `H200` | Default if you omit `hardware`. |
| `B200` | Request the B200 pool. |
| `B300` | Request the B300 pool. |

The typed SDK accepts the `Hardware` enum (`Hardware.B200`) or the same
strings. Any other value is rejected client-side on `create_job()` and
`get_capacity()`, and by the server on a raw JSON body.

Reservation and usage are counted separately per type. An account with H200
capacity is not automatically given B200 capacity.

## Check capacity

```bash
cortex-training capacity
cortex-training capacity --hardware B200
```

Omitting `--hardware` queries `H200`, `B200`, and `B300` independently and
prints a `capacity_by_hardware` map. Pass `--hardware` for a single type in
the same shape as `get_capacity()`. Use the same value you will pass on
create-job.

The Python SDK stays per-request: `client.get_capacity()` omits the query
parameter (the server defaults to H200). Pass `hardware=` to scope it.

## Set it on create

JSON body (also how recipes submit — add the field next to `sub_job_configs`):

```json
{
  "hardware": "B200",
  "sub_job_configs": []
}
```

Python:

```python
from cortex_training import CortexTrainingClient, Hardware, SubJobConfig

job_id = client.create_job(sub_jobs=[sub_job], hardware=Hardware.B200)
```

See [REST API reference section 5.1](../reference/rest-api.md#51-create-job---post-)
and [section 5.4](../reference/rest-api.md#54-capacity---get-capacity) for the
wire shapes.
