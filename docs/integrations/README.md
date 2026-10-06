# Integrations

RL frameworks integrated with Cortex Training.

- [SkyRL](skyrl.md): runs GRPO with its trainer on a CPU driver and training
  and sampling in Cortex sub-jobs. The GSM8K example lives in the Arctic
  Platform repository.
- [Tinker](tinker.md): serves Tinker's HTTP API on a CPU driver. Training and
  sampling run as Cortex sub-jobs through the published `arctic-platform`
  package.

For RL using this repository's own recipes, see the
[reinforcement learning guide](../guides/training/reinforcement-learning.md).
