# Manage Jobs

Common lifecycle commands:

```bash
cortex-training list
cortex-training list --status running
cortex-training get JOB_ID
cortex-training checkpoints JOB_ID
cortex-training wait JOB_ID
cortex-training cancel JOB_ID
```

Use `cortex-training capacity` before starting a recipe, and
`cortex-training capacity --hardware B200` (or `B300`) when you want a single
type. See [GPU hardware](../../concepts/hardware.md). Resume and retry guidance
is tracked separately because support depends on checkpoint type and failure
state.

## Watch Cluster Status

From a repository checkout, use `cluster-status.py` for a summary of running
jobs and GPU usage. It shells out to `cortex-training list`, so configure
[CLI login](../../reference/cli.md#login) first.

```bash
python cluster-status.py            # Live view, refreshed every 5 seconds
python cluster-status.py --once     # Print one summary and exit
```

This standalone helper is not part of the installed package. Run it from the
repository root.
