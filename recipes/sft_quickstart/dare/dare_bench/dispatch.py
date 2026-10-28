# Excerpt of DARE-Bench's scripts/llm_serving/remote_llm.py (Apache-2.0):
# OpenAIChatCompletionClient.dispatch_tool_call, changed into a module-level
# function. The changes are dropping ``self`` and the ``Session`` annotation.
import json
from typing import Dict, Set, Tuple

from .tools.base import BaseTool


def dispatch_tool_call(
    call,
    session,
    dispatch_map: Dict[str, BaseTool],
    heavy_init: Set[str],
) -> Tuple[str, bool]:
    """
    Try to invoke the tool, return (output, success_flag).
    """
    name = call.function.name

    if name not in dispatch_map:
        return f"\nError: Unsupported tool call: {name}. Please check the tool name carefully.\n", False

    tool = dispatch_map[name]

    # Heavy initialize once
    if name not in heavy_init:
        try:
            tool.initialize()
        except Exception as e:
            return f"Initialization failed for {name}: {e}", False
        heavy_init.add(name)

    try:
        tool_args = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        return f"Json decode error for tool call: {call.arguments}", False
    except Exception as e:
        return f"Unable to process tool call: {call.arguments}\n Error: {e}", False
    
    # Also input seq as inputs
    merged_args = {**tool_args, "session": session}
    try:
        result = tool(**merged_args)
        return result

    except TypeError as e:
        return f"Function call failed: {e}", False
    except Exception as e:
        return f"Tool runtime error in {name}: {e}", False
