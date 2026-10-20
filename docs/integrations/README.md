# Integrations

RL frameworks integrated with Cortex Training.

- [SkyRL](skyrl.md): runs GRPO with its trainer on a CPU driver and training
  and sampling in Cortex sub-jobs. The GSM8K example lives in the Arctic
  Platform repository.
- [Bring your own agent](bring-your-own-agent.md): train a model inside the
  agent you already run. The agent sends its model calls to a per-rollout
  OpenAI-compatible endpoint, and a grader you write scores each rollout.
  - [OpenHands](openhands.md): the worked example. It trains an OpenHands
    agent to localize bugs.

For RL using this repository's own recipes, see the
[reinforcement learning guide](../guides/training/reinforcement-learning.md).
