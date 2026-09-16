# Integrations

External projects that drive training on Cortex Training.

**[SkyRL](skyrl.md)** — SkyRL's GRPO trainer and entry point drive the run,
Cortex supplies the training and sampling sub-jobs, and the driver stays on CPU.
The recipe lives in the Arctic Platform repository; nothing under `recipes/`
here runs it.

For framework-driven RL more generally, see the
[reinforcement learning guide](../guides/training/reinforcement-learning.md).
