# Reinforcement Learning

The [Math GRPO recipe](../../../recipes/rl/math_grpo/README.md) is the current
end-to-end RL example. It demonstrates:

- Colocated training and sampling sub-jobs
- Grouped rollout generation and reward centering
- GRPO loss configuration
- Training-to-sampling weight synchronization
- Held-out MATH-500 evaluation

## External frameworks

[SkyRL](../../integrations/skyrl.md) can run GRPO against Cortex Training, with
its trainer on a CPU driver and training and sampling in Cortex sub-jobs. The
[GSM8K example](https://github.com/Snowflake-AI-Research/Arctic-Platform/blob/main/recipes/rl/skyrl/simple_gsm8k_cortex/README.md)
lives in the Arctic Platform repository and is driven by SkyRL, so it is not
`recipes.rl.math_grpo` and has its own install -- see the
[integration page](../../integrations/skyrl.md) before running it.

Additional code, tool-use, and multi-agent RL recipes are planned.
