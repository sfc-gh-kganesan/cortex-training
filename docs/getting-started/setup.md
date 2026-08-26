# Set Up the Client

Install the client:

```bash
uv pip install git+https://github.com/Snowflake-AI-Research/cortex-training.git
```

If you cloned the repository (needed to run the recipes, which are not part of
the installed package), install it in editable mode instead:

```bash
uv pip install -e .
```

Create a connection file. From a clone, copy the template:

```bash
cp examples/config/connection.json.template ~/cortex-training-config.json
```

Otherwise create it by hand, outside the repository:

```json
{
  "host": "ACCOUNT.snowflakecomputing.com",
  "pat": "YOUR_PROGRAMMATIC_ACCESS_TOKEN",
  "database": "CORTEX_TRAINING_DB",
  "schema": "PUBLIC"
}
```

Fill in the account host, programmatic access token, database, and schema. Keep
the file outside the repository and do not commit it.

Validate and store the config path:

```bash
cortex-training login --config ~/cortex-training-config.json
cortex-training capacity
```

See the [CLI reference](../reference/cli.md) for environment variables,
alternative server targets, and one-command overrides.
