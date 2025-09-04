"""
AetherFlow Agent Tools System

This module provides a comprehensive tool system for ReAct agents, featuring:
- @tool decorator for converting functions into agent-callable tools
- ToolRegistry for dynamic tool discovery and registration
- Concurrent execution capabilities using AetherFlow's fan_out_to mechanism
- Type-safe tool interfaces with parameter validation
- Standardized tool result handling and error isolation

Core Components:
- ToolMetadata: Tool description and parameter information
- ToolCall: Tool invocation request structure
- ToolResult: Standardized tool execution result
- ToolRegistry: Global tool registration and discovery
- @tool: Decorator to mark functions as agent tools

Architecture:
The tool system integrates seamlessly with AetherFlow's @node decoration system,
providing async/sync compatibility, retry mechanisms, and dependency injection.
Tools are automatically converted to @node functions for optimal performance
and integration with the ReAct execution engine.

Usage Example:
    from agents.tools import tool, ToolRegistry

    @tool("calculator", "Perform basic arithmetic operations")
    async def calculator(operation: str, a: float, b: float) -> float:
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # Tool is automatically registered and available to ReAct agents
    available_tools = ToolRegistry.get_available_tools()
"""

# Core tool system components
# Built-in example tools
from .builtin import *
from .decorators import tool

# Concurrent execution components
from .executor import ToolExecutor, execute_tools_concurrently
from .models import ToolCall, ToolMetadata, ToolResult
from .registry import ToolRegistry

# Re-export main interfaces for convenient imports
__all__ = [
    # Core interfaces
    "tool",
    "ToolCall",
    "ToolMetadata",
    "ToolResult",
    "ToolRegistry",
    # Execution components
    "ToolExecutor",
    "execute_tools_concurrently",
    # Built-in tools are exported via * import
]
