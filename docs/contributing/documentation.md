# Documentation Guidelines

Place content according to its purpose:

- `getting-started/` for the shortest path to first success
- `concepts/` for mental models and architecture
- `guides/` for task-oriented procedures
- `reference/` for exact commands, schemas, and supported values
- `recipes/` for runnable workflows and their local documentation

Document only what exists in the branch. Do not add placeholder pages for
unimplemented features -- if a workflow is partly implemented, say so inline on
the page that covers it and name what is missing. Do not present unvalidated
model support, hardware requirements, or expected metrics as guarantees.

Use relative links. When you document a default, a flag, or a field, check it
against the code in the same change -- the recipe `Config` classes for
`name=value` overrides, `src/cortex_training/` for client behavior.
