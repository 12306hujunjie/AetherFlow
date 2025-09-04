"""
Tool Decorator System

Implements the @tool decorator that converts regular Python functions
into ReAct agent-compatible tools. The decorator automatically:

1. Extracts function signature and type information
2. Creates ToolMetadata for registration and validation
3. Wraps functions with @node for AetherFlow integration
4. Provides error handling, timeout, and retry capabilities
5. Registers tools in the global ToolRegistry

The decorator ensures seamless integration with both sync and async functions,
maintaining type safety and providing consistent tool execution semantics.
"""

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, get_type_hints

from .models import (
    ParameterInfo,
    ToolCategory,
    ToolExecutionStatus,
    ToolMetadata,
    ToolResult,
)
from .registry import ToolRegistry

logger = logging.getLogger("aetherflow.agents.tools")

F = TypeVar("F", bound=Callable[..., Any])


def tool(
    name: str | None = None,
    description: str | None = None,
    category: ToolCategory | str = ToolCategory.GENERAL,
    timeout: float = 30.0,
    supports_concurrent: bool = True,
    requires_context: bool = False,
    enable_retry: bool = True,
    retry_count: int = 2,
    retry_delay: float = 0.5,
) -> Callable[[F], F]:
    """
    Decorator to convert a function into a ReAct agent tool.

    This decorator performs automatic tool registration and wrapping,
    converting regular Python functions into @node-decorated tools
    that can be called by ReAct agents.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        category: Tool category for organization
        timeout: Execution timeout in seconds
        supports_concurrent: Whether tool can run concurrently
        requires_context: Whether tool needs ReAct context
        enable_retry: Enable retry mechanism for failures
        retry_count: Number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Decorated function as an AetherFlow @node with tool capabilities

    Example:
        @tool("calculator", "Perform arithmetic operations")
        def add(a: float, b: float) -> float:
            return a + b

        @tool("search", "Search the web", category=ToolCategory.SEARCH)
        async def web_search(query: str, max_results: int = 10) -> List[str]:
            # Implementation here
            return results
    """

    def decorator(func: F) -> F:
        # Extract metadata from function
        func_name = name or func.__name__
        func_description = description or (
            func.__doc__.strip() if func.__doc__ else f"Execute {func_name}"
        )

        # Validate category
        if isinstance(category, str):
            try:
                tool_category = ToolCategory(category)
            except ValueError:
                logger.warning(f"Unknown category '{category}', using GENERAL")
                tool_category = ToolCategory.GENERAL
        else:
            tool_category = category

        # Extract function signature and type information
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Build parameter information
        parameters = []
        for param_name, param in signature.parameters.items():
            # Skip context parameters and internal parameters
            if requires_context and param_name in ["context", "react_context"]:
                continue
            if param_name.startswith("__"):  # Skip internal parameters like __call_id
                continue

            param_info = ParameterInfo(
                name=param_name,
                type_hint=_format_type_hint(type_hints.get(param_name, Any)),
                description=_extract_param_description(func, param_name),
                required=(param.default is inspect.Parameter.empty),
                default_value=param.default
                if param.default is not inspect.Parameter.empty
                else None,
            )
            parameters.append(param_info)

        # Determine if function is async
        is_async = inspect.iscoroutinefunction(func)

        # Get return type
        return_type_hint = type_hints.get("return", Any)
        return_type = _format_type_hint(return_type_hint)

        # Create tool metadata
        metadata = ToolMetadata(
            name=func_name,
            description=func_description,
            parameters=parameters,
            return_type=return_type,
            category=tool_category,
            is_async=is_async,
            timeout_seconds=timeout,
            supports_concurrent=supports_concurrent,
            requires_context=requires_context,
            module_path=func.__module__,
        )

        # Create the tool wrapper function with retry logic
        @wraps(func)
        async def tool_wrapper(*args, **kwargs) -> ToolResult:
            """
            Wrapped tool function with standardized result format and error handling.
            """
            last_exception = None

            for attempt in range(retry_count + 1):
                start_time = time.time()

                # Create result object for this attempt
                result = ToolResult(
                    tool_name=func_name,
                    call_id=None,
                    status=ToolExecutionStatus.RUNNING,
                    success=False,
                    start_time=start_time,
                    retry_count=attempt,
                )

                try:
                    logger.debug(
                        f"Executing tool '{func_name}' (attempt {attempt + 1}/{retry_count + 1}) with args={args}, kwargs={kwargs}"
                    )

                    # Execute the tool function
                    if is_async:
                        # Execute async function with timeout
                        try:
                            raw_result = await asyncio.wait_for(
                                func(*args, **kwargs), timeout=timeout
                            )
                        except asyncio.TimeoutError:
                            result.mark_timeout()
                            logger.warning(
                                f"Tool '{func_name}' timed out after {timeout}s"
                            )
                            if not enable_retry or attempt == retry_count:
                                return result
                            last_exception = asyncio.TimeoutError("Function timeout")
                            await asyncio.sleep(retry_delay)
                            continue
                    else:
                        # Execute sync function in thread pool for non-blocking execution
                        loop = asyncio.get_event_loop()
                        try:
                            # Create a function that properly calls the function
                            def executor_func():
                                return func(*args, **kwargs)

                            raw_result = await asyncio.wait_for(
                                loop.run_in_executor(None, executor_func),
                                timeout=timeout,
                            )
                        except asyncio.TimeoutError:
                            result.mark_timeout()
                            logger.warning(
                                f"Tool '{func_name}' timed out after {timeout}s"
                            )
                            if not enable_retry or attempt == retry_count:
                                return result
                            last_exception = asyncio.TimeoutError("Function timeout")
                            await asyncio.sleep(retry_delay)
                            continue

                    # Mark as successful
                    result.mark_completed(result=raw_result)
                    logger.debug(
                        f"Tool '{func_name}' completed successfully in {result.execution_time_ms:.1f}ms"
                    )

                    return result

                except Exception as e:
                    # Handle any execution error
                    last_exception = e
                    result.mark_completed(error=e)
                    logger.error(
                        f"Tool '{func_name}' failed (attempt {attempt + 1}/{retry_count + 1}): {e}"
                    )

                    if not enable_retry or attempt == retry_count:
                        return result

                    # Wait before retry
                    await asyncio.sleep(retry_delay)

            # Should not reach here, but handle just in case
            final_result = ToolResult(
                tool_name=func_name,
                call_id=None,
                status=ToolExecutionStatus.FAILED,
                success=False,
                error_message=f"All retry attempts failed. Last error: {str(last_exception)}",
                error_type=type(last_exception).__name__
                if last_exception
                else "UnknownError",
                retry_count=retry_count,
            )
            return final_result

        # Store original function reference for introspection
        tool_wrapper.__wrapped__ = func
        tool_wrapper.__tool_metadata__ = metadata

        # Register the tool in the global registry
        try:
            ToolRegistry.register(metadata, tool_wrapper)
            logger.info(
                f"Registered tool '{func_name}' in category '{tool_category.value}'"
            )
        except Exception as e:
            logger.error(f"Failed to register tool '{func_name}': {e}")
            raise

        return tool_wrapper

    return decorator


def _format_type_hint(type_hint: Any) -> str:
    """
    Format type hints as readable strings for metadata.

    Args:
        type_hint: Type hint object from get_type_hints()

    Returns:
        String representation of the type
    """
    if hasattr(type_hint, "__name__"):
        return type_hint.__name__
    elif hasattr(type_hint, "_name"):  # Generic types like List, Dict
        return str(type_hint)
    elif str(type_hint).startswith("typing."):
        # Clean up typing module references
        return str(type_hint).replace("typing.", "")
    else:
        return str(type_hint)


def _extract_param_description(func: Callable, param_name: str) -> str:
    """
    Extract parameter description from function docstring.

    This is a simple implementation that looks for parameter
    descriptions in docstrings. A more sophisticated version
    could parse Google-style or Sphinx-style docstrings.

    Args:
        func: Function to extract description from
        param_name: Parameter name to find description for

    Returns:
        Parameter description or empty string
    """
    if not func.__doc__:
        return ""

    # Simple extraction - look for "param_name:" in docstring
    lines = func.__doc__.split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith(f"{param_name}:"):
            # Found parameter description
            return line[len(f"{param_name}:") :].strip()
        elif line.lower().startswith("Args:"):
            # Look in Args section
            continue
        elif f"{param_name}" in line and ":" in line:
            # Try to extract from Args-style documentation
            if param_name in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()

    return ""


def get_tool_metadata(tool_func: Callable) -> ToolMetadata | None:
    """
    Extract tool metadata from a decorated function.

    Args:
        tool_func: Function decorated with @tool

    Returns:
        ToolMetadata if function is a tool, None otherwise
    """
    return getattr(tool_func, "__tool_metadata__", None)


def is_tool_function(func: Callable) -> bool:
    """
    Check if a function is decorated with @tool.

    Args:
        func: Function to check

    Returns:
        True if function is a tool, False otherwise
    """
    return hasattr(func, "__tool_metadata__")


def list_tools_in_module(module) -> dict[str, ToolMetadata]:
    """
    Discover all tools in a Python module.

    Args:
        module: Module to scan for tools

    Returns:
        Dictionary mapping tool names to metadata
    """
    tools = {}

    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and is_tool_function(obj):
            metadata = get_tool_metadata(obj)
            if metadata:
                tools[metadata.name] = metadata

    return tools
