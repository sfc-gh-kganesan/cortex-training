# Reinforcement Learning

The [Math GRPO example](../../../recipes/rl_quickstart/README.md) is the current
end-to-end RL example. It demonstrates:

- Colocated training and sampling sub-jobs
- Grouped rollout generation and reward centering
- GRPO loss configuration
- Training-to-sampling weight synchronization
- Held-out MATH-500 evaluation

## Integrated RL frameworks

[SkyRL](../../../recipes/skyrl/gsm8k_grpo/README.md) can also run GRPO against Cortex Training,
with its trainer on a CPU driver and training and sampling in Cortex sub-jobs.
Its GSM8K example lives in the Arctic Platform repository and has its own
install.

To train a model inside an agent you already have, see
[Bring your own agent](../../../recipes/bring-your-own-agent.md). The agent
keeps its own loop and sends model calls to a per-rollout endpoint; SkyRL and
Cortex do the training. [OpenHands](../../../recipes/openhands/README.md) is the
worked example.
