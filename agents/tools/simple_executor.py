"""
Simple Tool Execution System

A simplified tool execution system that focuses on functionality
over integration with AetherFlow's @node system. This provides
concurrent execution capabilities while maintaining simplicity.
"""

import asyncio
import logging
import time

from .models import (
    ConcurrentExecutionResult,
    ToolCall,
    ToolExecutionStatus,
    ToolResult,
)
from .registry import ToolRegistry

logger = logging.getLogger("aetherflow.agents.tools.simple")


class SimpleToolExecutor:
    """
    Simplified tool executor with basic concurrent execution capabilities.

    This executor avoids complex AetherFlow integration and focuses on
    providing reliable concurrent tool execution with error isolation.
    """

    def __init__(self, max_workers: int = 5, default_timeout: float = 30.0):
        """
        Initialize simple tool executor.

        Args:
            max_workers: Maximum concurrent workers
            default_timeout: Default execution timeout
        """
        self.max_workers = max_workers
        self.default_timeout = default_timeout

    async def execute_single_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a single tool call.

        Args:
            tool_call: Tool to execute

        Returns:
            Tool execution result
        """
        tool_func = ToolRegistry.get_tool(tool_call.tool_name)
        if not tool_func:
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                status=ToolExecutionStatus.FAILED,
                success=False,
                error_message=f"Tool '{tool_call.tool_name}' not found in registry",
                error_type="ToolNotFoundError",
            )

        # Validate tool call
        is_valid, error_msg = ToolRegistry.validate_tool_call(tool_call)
        if not is_valid:
            return ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                status=ToolExecutionStatus.FAILED,
                success=False,
                error_message=error_msg,
                error_type="ValidationError",
            )

        try:
            # Execute tool function directly (it returns ToolResult)
            result = await tool_func(**tool_call.parameters)

            # Update result with call_id if it wasn't set
            if result.call_id is None:
                result.call_id = tool_call.call_id

            # Record execution statistics
            ToolRegistry.record_execution_stats(
                tool_call.tool_name,
                result.execution_time_ms,
                result.success,
                result.error_type if not result.success else None,
            )

            return result

        except Exception as e:
            logger.error(
                f"Unexpected error executing tool '{tool_call.tool_name}': {e}"
            )
            error_result = ToolResult(
                tool_name=tool_call.tool_name,
                call_id=tool_call.call_id,
                status=ToolExecutionStatus.FAILED,
                success=False,
                error_message=f"Unexpected execution error: {str(e)}",
                error_type=type(e).__name__,
            )

            # Record failed execution
            ToolRegistry.record_execution_stats(
                tool_call.tool_name,
                0.0,
                False,
                type(e).__name__,
            )

            return error_result

    async def execute_tools_concurrently(
        self, tool_calls: list[ToolCall], max_workers: int | None = None
    ) -> ConcurrentExecutionResult:
        """
        Execute multiple tools concurrently using asyncio.

        Args:
            tool_calls: List of tools to execute
            max_workers: Override default max workers

        Returns:
            Aggregated execution results
        """
        start_time = time.time()

        if not tool_calls:
            return ConcurrentExecutionResult(
                tool_results=[],
                batch_success=True,
                total_execution_time_ms=0.0,
                total_tools=0,
                max_workers_used=1,
                executor_strategy_used="none",
            )

        effective_max_workers = min(max_workers or self.max_workers, len(tool_calls))

        try:
            # Use asyncio semaphore to limit concurrent execution
            semaphore = asyncio.Semaphore(effective_max_workers)

            async def bounded_execute(tool_call: ToolCall) -> ToolResult:
                async with semaphore:
                    return await self.execute_single_tool(tool_call)

            # Execute all tools concurrently
            tasks = [bounded_execute(call) for call in tool_calls]
            results = await asyncio.gather(*tasks, return_exceptions=False)

            end_time = time.time()
            total_time_ms = (end_time - start_time) * 1000

            # Create result
            return ConcurrentExecutionResult(
                tool_results=results,
                batch_success=all(r.success for r in results),
                total_execution_time_ms=total_time_ms,
                total_tools=len(results),
                successful_tools=sum(1 for r in results if r.success),
                failed_tools=sum(1 for r in results if not r.success),
                max_workers_used=effective_max_workers,
                executor_strategy_used="asyncio",
            )

        except Exception as e:
            logger.error(f"Concurrent execution failed: {e}")

            # Return error results for all tools
            error_results = [
                ToolResult(
                    tool_name=call.tool_name,
                    call_id=call.call_id,
                    status=ToolExecutionStatus.FAILED,
                    success=False,
                    error_message=f"Batch execution failed: {str(e)}",
                    error_type="ConcurrentExecutionError",
                )
                for call in tool_calls
            ]

            end_time = time.time()
            total_time_ms = (end_time - start_time) * 1000

            return ConcurrentExecutionResult(
                tool_results=error_results,
                batch_success=False,
                total_execution_time_ms=total_time_ms,
                total_tools=len(tool_calls),
                successful_tools=0,
                failed_tools=len(tool_calls),
                max_workers_used=effective_max_workers,
                executor_strategy_used="asyncio",
            )
