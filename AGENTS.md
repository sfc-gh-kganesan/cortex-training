# Agent Instructions for `cortex-training`

Always run the tests before and after changing any code in `src/cortex_training/`,
from the root of this repository:

```bash
uv pip install -e ".[dev]"
python3 -m pytest tests/ -q
```

All tests must pass.

## Context

`src/cortex_training/client.py` is an HTTP client for the Cortex Training REST
API. This repository is the client only; the service implementation lives
elsewhere and is not public.

The reference documents in this repository are the contract to code against:

- `docs/reference/rest-api.md` — REST paths, request framing, polling, schemas.
  Treat it as the source of truth for anything a customer can observe, and
  update it in the same change as the client whenever the wire behavior moves.
- `docs/reference/cli.md` — commands, flags, configuration, environment vars.

## House rules

- **Never commit account identifiers, hostnames, PATs, request IDs, or job
  UUIDs**, including inside docstrings and example output. Use placeholders such
  as `ACCOUNT.snowflakecomputing.com` and `CORTEX_TRAINING_DB`.
- **Never reference non-public repositories, file paths, or internal codenames**
  in code comments, docstrings, or docs. A reader only has this repository.
- **No `print()` in library code.** The CLI writes machine-readable JSON to
  stdout; use `logger` so piping to `jq` keeps working.
- Binary payloads use the DSSST1 wire protocol in `src/cortex_training/wire.py`.
  Never use `torch.save`/pickle for a request body — the server rejects those
  frames.
- Assertions on serialized payloads should decode them (`wire.loads`) rather
  than only checking `isinstance(payload, bytes)`, which any encoding satisfies.
- When you change a default, a flag, or a field, update the docs that quote it in
  the same change: `docs/reference/cli.md`, `docs/reference/rest-api.md`, and the
  recipe READMEs plus `recipe.yaml`.
