# Tinker Cookbook

The [conversational SFT](../../recipes/sft/conversational/README.md) and
[Math GRPO](../../recipes/rl/math_grpo/README.md) recipes are adapted from
[Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
workflows, and run on Cortex Training for execution.

The recipes depend on the cookbook at run time for chat rendering, tokenizer
lookup and metric logging, so it is a required install for them:

```bash
uv pip install 'tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

Each adapted recipe names the cookbook file it was ported from in its `train.py`
module docstring. Conversational SFT also records it under `provenance` in its
`recipe.yaml`.
