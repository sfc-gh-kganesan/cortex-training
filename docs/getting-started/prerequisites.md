# Prerequisites

Before running a training recipe, you need:

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) (or `pip`) to install the client
- Access to a Snowflake account with the Cortex Training endpoint enabled
- A programmatic access token
- A database and schema containing the endpoint
- Sufficient reserved GPU capacity for the selected recipe

Check current capacity after installing and authenticating:

```bash
cortex-training capacity
```

Shipped defaults (check `capacity` before you submit):

- Conversational SFT LoRA/full Qwen3-8B: **4** training GPUs
- Math GRPO Qwen3-8B: **4** training + **4** sampling GPUs (8 total)
- Qwen3.6-35B-A3B recipes: **8** training GPUs (GRPO adds sampling GPUs on top)

Model-specific GPU requirements are not yet fully validated. Treat recipe
defaults as starting configurations, not guaranteed minimums.
