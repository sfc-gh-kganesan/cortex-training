# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""GSM8K and MATH on Cortex, using the tinker-cookbook recipe unchanged.

The client lives in Arctic Platform (``arctic_platform.tinker``). This file
only selects that client and the cookbook math recipe. GPU counts are
arguments, as on the Cortex client CLI. ``env=gsm8k`` or ``env=math`` picks
the dataset.

    python -m recipes.tinker.math_rl --training-gpus 1 --sampling-gpus 1 -- \\
        env=gsm8k model_name=Qwen/Qwen3.5-4B renderer_name=qwen3_5_disable_thinking \\
        lora_rank=32 group_size=8 groups_per_batch=16 learning_rate=1e-4 max_tokens=256
"""

from __future__ import annotations

import sys

_FLAGS = {
    "--training-gpus",
    "--sampling-gpus",
    "--max-prompt-length",
    "--max-response-length",
}
_MODULE = "tinker_cookbook.recipes.math_rl.train"


def _insert_module(argv: list[str]) -> list[str]:
    index = 0
    while index < len(argv) and argv[index] in _FLAGS:
        index += 2
    rest = argv[index:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest or "=" in rest[0]:
        rest = [_MODULE, *rest]
    return [*argv[:index], *rest]


def main() -> None:
    sys.argv = [sys.argv[0], *_insert_module(sys.argv[1:])]
    from arctic_platform.tinker.run import main as run_main

    run_main()


if __name__ == "__main__":
    main()
