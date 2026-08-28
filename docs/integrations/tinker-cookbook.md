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

The conversational SFT recipe records its upstream source under `provenance` in
`recipes/sft/conversational/recipe.yaml`. Math GRPO does not have a `recipe.yaml`
yet.
