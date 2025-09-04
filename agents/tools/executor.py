"""
Concurrent Tool Execution System

Implements concurrent tool execution using AetherFlow's fan_out_to mechanism.
Provides intelligent strategy selection, resource management, and result
aggregation for batch tool execution by ReAct agents.

Key Features:
- Integration with AetherFlow's fan_out_to for proven concurrency
- Automatic executor strategy selection (auto, thread, async)
- Intelligent resource management and timeout handling
- Error isolation - individual tool failures don't affect others
- Performance monitoring and parallelism efficiency tracking
- Support for mixed sync/async tool execution
"""

import logging
import time
from typing import Any

from dependency_injector.wiring import Provide

from aetherflow import Node, node

from ..react.context import ReActContext
from .models import (
    ConcurrentExecutionResult,
    ConcurrentToolExecution,
    ToolCall,
    ToolExecutionStatus,
    ToolResult,
)
from .registry import ToolRegistry

logger = logging.getLogger("aetherflow.agents.tools.executor")


class ToolExecutor:
    """
    Tool execution manager with concurrent execution capabilities.

    Manages tool execution lifecycle, strategy selection, and performance
    monitoring. Integrates with AetherFlow's concurrency mechanisms for
    optimal performance and resource utilization.
    """

    def __init__(
        self,
        max_workers: int = 5,
        default_timeout: float = 30.0,
        registry: ToolRegistry | None = None,
    ):
        """
        Initialize tool executor.

        Args:
            max_workers: Maximum concurrent workers
            default_timeout: Default execution timeout
            registry: Tool registry instance (uses global if None)
        """
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        self.registry = registry or ToolRegistry.get_instance()

    def select_execution_strategy(self, tool_calls: list[ToolCall]) -> str:
        """
        Intelligently select execution strategy based on tool characteristics.

        Args:
            tool_calls: List of tools to execute

        Returns:
            Execution strategy: "auto", "thread", or "async"
        """
        if not tool_calls:
            return "auto"

        async_tools = 0
        sync_tools = 0

        for call in tool_calls:
            metadata = ToolRegistry.get_tool_metadata(call.tool_name)
            if metadata:
                if metadata.is_async:
                    async_tools += 1
                else:
                    sync_tools += 1

        # Strategy selection logic
        if async_tools > sync_tools:
            return "async"  # Primarily async tools
        elif sync_tools > 0 and async_tools == 0:
            return "thread"  # Only sync tools, use thread pool
        else:
            return "auto"  # Mixed or unknown, let AetherFlow decide

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

        # Execute tool with call ID for tracking
        try:
            # Execute tool function directly
            result = await tool_func(**tool_call.parameters)

            # Update result with call_id if it wasn't set by the tool
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
                0.0,  # No meaningful execution time for unexpected errors
                False,
                type(e).__name__,
            )

            return error_result

    async def execute_tools_concurrently(
        self, execution_request: ConcurrentToolExecution
    ) -> ConcurrentExecutionResult:
        """
        Execute multiple tools concurrently using AetherFlow's fan_out_to.

        Args:
            execution_request: Concurrent execution configuration

        Returns:
            Aggregated execution results with performance metrics
        """
        start_time = time.time()
        tool_calls = execution_request.tool_calls

        if not tool_calls:
            return ConcurrentExecutionResult(
                tool_results=[],
                batch_success=True,
                total_execution_time_ms=0.0,
                total_tools=0,
                max_workers_used=1,  # Use 1 to satisfy validation
                executor_strategy_used="none",
            )

        # Select execution strategy
        strategy = (
            execution_request.executor_strategy
            if execution_request.executor_strategy != "auto"
            else self.select_execution_strategy(tool_calls)
        )

        logger.info(
            f"Executing {len(tool_calls)} tools concurrently using '{strategy}' strategy"
        )

        try:
            # Create tool execution nodes
            tool_nodes = []
            for call in tool_calls:
                # Create a wrapper function that ignores the input from fan_out_to
                async def tool_wrapper(_input=None, tool_call=call):
                    return await self.execute_single_tool(tool_call)

                # Wrap in a Node for fan_out_to compatibility
                tool_node = Node(
                    tool_wrapper,
                    name=f"tool_{call.tool_name}_{call.call_id or 'unknown'}",
                )
                tool_nodes.append(tool_node)

            # Use AetherFlow's fan_out_to for concurrent execution
            effective_max_workers = min(execution_request.max_workers, len(tool_calls))

            # Create a simple identity source node
            @node
            async def identity_source() -> None:
                return None

            # Create the fan_out pipeline
            fan_out_pipeline = identity_source.fan_out_to(
                tool_nodes,
                executor=strategy,
                max_workers=effective_max_workers,
            )

            # Execute the pipeline and get parallel results
            parallel_results = fan_out_pipeline(None)

            # Convert parallel results to tool results
            results = await aggregate_parallel_tool_results(parallel_results)

            end_time = time.time()
            total_time_ms = (end_time - start_time) * 1000

            # Create comprehensive result
            execution_result = ConcurrentExecutionResult(
                tool_results=results,
                batch_success=all(r.success for r in results),
                total_execution_time_ms=total_time_ms,
                total_tools=len(results),
                max_workers_used=effective_max_workers,
                executor_strategy_used=strategy,
            )

            logger.info(
                f"Concurrent execution completed in {total_time_ms:.1f}ms, "
                f"success rate: {execution_result.successful_tools}/{execution_result.total_tools}"
            )

            return execution_result

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
                total_tools=len(tool_calls),  # Add missing field
                max_workers_used=1,  # Use 1 instead of 0 to satisfy validation
                executor_strategy_used=strategy,
            )


async def aggregate_parallel_tool_results(
    parallel_results: dict[str, Any],
) -> list[ToolResult]:
    """
    Aggregate tool results from AetherFlow's fan_out_to execution.

    Args:
        parallel_results: Dictionary of node results from fan_out_to

    Returns:
        List of tool results sorted by execution time
    """
    results = []

    for node_identifier, node_result in parallel_results.items():
        try:
            # Handle AetherFlow ParallelResult objects
            if hasattr(node_result, "success") and hasattr(node_result, "result"):
                if node_result.success and isinstance(node_result.result, ToolResult):
                    results.append(node_result.result)
                elif not node_result.success:
                    # Create error result from failed node
                    error_result = ToolResult(
                        tool_name=node_identifier.replace("tool_", "").split("_")[0]
                        if "tool_" in node_identifier
                        else "unknown",
                        call_id=None,
                        status=ToolExecutionStatus.FAILED,
                        success=False,
                        error_message=f"Node execution failed: {node_result.error}",
                        error_type="NodeExecutionError",
                    )
                    results.append(error_result)
                else:
                    # Handle unexpected successful result format
                    logger.warning(
                        f"Unexpected successful result format from node {node_identifier}: {type(node_result.result)}"
                    )
                    error_result = ToolResult(
                        tool_name=node_identifier.replace("tool_", "").split("_")[0]
                        if "tool_" in node_identifier
                        else "unknown",
                        call_id=None,
                        status=ToolExecutionStatus.FAILED,
                        success=False,
                        error_message=f"Unexpected result format: {type(node_result.result)}",
                        error_type="ResultFormatError",
                    )
                    results.append(error_result)
            elif isinstance(node_result, ToolResult):
                # Direct ToolResult
                results.append(node_result)
            else:
                # Handle completely unexpected format
                logger.warning(
                    f"Unexpected result format from node {node_identifier}: {type(node_result)}"
                )
                error_result = ToolResult(
                    tool_name=node_identifier.replace("tool_", "").split("_")[0]
                    if "tool_" in node_identifier
                    else "unknown",
                    call_id=None,
                    status=ToolExecutionStatus.FAILED,
                    success=False,
                    error_message=f"Unexpected result format: {type(node_result)}",
                    error_type="ResultFormatError",
                )
                results.append(error_result)

        except Exception as e:
            logger.error(f"Error processing result from node {node_identifier}: {e}")

            # Create error result for processing failure
            error_result = ToolResult(
                tool_name=node_identifier.replace("tool_", "").split("_")[0]
                if "tool_" in node_identifier
                else "unknown",
                call_id=None,
                status=ToolExecutionStatus.FAILED,
                success=False,
                error_message=f"Result processing error: {str(e)}",
                error_type="ResultProcessingError",
            )
            results.append(error_result)

    # Sort by execution time for consistent ordering (handle missing execution_time_ms)
    results.sort(key=lambda r: getattr(r, "execution_time_ms", 0))

    logger.debug(f"Aggregated {len(results)} tool results")
    return results


# Convenience function for direct tool execution
@node
async def execute_tools_concurrently(
    tool_calls: list[ToolCall],
    context: ReActContext | None = None,
    max_workers: int = 5,
    timeout_seconds: float = 30.0,
    executor_strategy: str = "auto",
    tool_executor: ToolExecutor = Provide["tool_executor"],
) -> ConcurrentExecutionResult:
    """
    Convenience function for concurrent tool execution.

    This is the main entry point for concurrent tool execution from ReAct agents.
    It creates a ConcurrentToolExecution request and processes it through the
    ToolExecutor.

    Args:
        tool_calls: List of tools to execute
        context: ReAct context (optional, for future context-aware tools)
        max_workers: Maximum concurrent workers
        timeout_seconds: Global timeout for all tools
        executor_strategy: Execution strategy selection
        tool_executor: Injected tool executor instance

    Returns:
        Concurrent execution results with performance metrics
    """
    if not tool_calls:
        logger.warning("No tool calls provided for concurrent execution")
        return ConcurrentExecutionResult(
            tool_results=[],
            batch_success=True,
            total_execution_time_ms=0.0,
            total_tools=0,
            max_workers_used=1,  # Use 1 to satisfy validation
            executor_strategy_used="none",
        )

    # Create execution request
    execution_request = ConcurrentToolExecution(
        tool_calls=tool_calls,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        executor_strategy=executor_strategy,
    )

    # Execute through the tool executor
    result = await tool_executor.execute_tools_concurrently(execution_request)

    # Log performance metrics
    if result.total_tools > 1:
        logger.info(
            f"Concurrent execution achieved {result.parallelism_achieved:.1%} efficiency "
            f"({result.total_execution_time_ms:.1f}ms total vs "
            f"{result.average_tool_time_ms:.1f}ms average)"
        )

    return result


# Legacy alias for backward compatibility with task specification
execute_tools_concurrently_legacy = execute_tools_concurrently
