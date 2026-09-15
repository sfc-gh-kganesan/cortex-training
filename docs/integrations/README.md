# Integrations

External projects that work with Cortex Training. They differ in which side owns
the training loop:

| Integration | Who drives training | Where the code lives |
|---|---|---|
| [Tinker Cookbook](tinker-cookbook.md) | Cortex recipes, using the cookbook as a library | `recipes/` in this repository |
| [SkyRL](skyrl.md) | SkyRL's own GRPO trainer, dispatching to Cortex | Arctic Platform repository |

**Tinker Cookbook** is a runtime dependency rather than a driver. The
[conversational SFT](../../recipes/sft/conversational/README.md) and
[Math GRPO](../../recipes/rl/math_grpo/README.md) recipes are ports of cookbook
workflows and import from it at run time for chat rendering, tokenizer lookup
and metric logging; Math GRPO also takes its dataset loading and answer grading
from `recipes.math_rl`. The recipe contract is this repository's. Each `train.py`
names the cookbook file it was ported from in its module docstring. See the
[Tinker Cookbook page](tinker-cookbook.md) for the install.

**SkyRL** is the other direction: SkyRL's trainer and entry point drive the run,
and Cortex Training provides training and sampling sub-jobs underneath. The
driver stays on CPU. Nothing under this repository's `recipes/` runs it, and it
is not `recipes.rl.math_grpo`.

For the in-repository RL path, use the
[Math GRPO recipe](../../recipes/rl/math_grpo/README.md). For framework-driven
RL, see the [reinforcement learning guide](../guides/training/reinforcement-learning.md).
