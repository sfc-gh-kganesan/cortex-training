# Integrations

External projects that work with Cortex Training, split by which side owns the
training loop:

| Integration | Who drives training | Where the code lives |
|---|---|---|
| [Tinker Cookbook](tinker-cookbook.md) | Cortex recipes, using the cookbook as a library | `recipes/` in this repository |
| [SkyRL](skyrl.md) | SkyRL's own GRPO trainer, dispatching to Cortex | Arctic Platform repository |

**[Tinker Cookbook](tinker-cookbook.md)** is a library, not a driver. The
[conversational SFT](../../recipes/sft/conversational/README.md) and
[Math GRPO](../../recipes/rl/math_grpo/README.md) recipes are ports of cookbook
workflows that import it at run time for chat rendering, tokenizer lookup and
metric logging; Math GRPO also takes dataset loading and answer grading from
`recipes.math_rl`. The recipe contract stays this repository's, and each
`train.py` names the cookbook file it was ported from.

**[SkyRL](skyrl.md)** is the reverse: SkyRL's trainer and entry point drive the
run, Cortex supplies the training and sampling sub-jobs, and the driver stays on
CPU. Nothing under `recipes/` here runs it, and it is not `recipes.rl.math_grpo`.

For framework-driven RL, see the
[reinforcement learning guide](../guides/training/reinforcement-learning.md).
