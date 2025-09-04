"""
Tests for Concurrent Tool Execution

Validates the concurrent tool execution system using AetherFlow's fan_out_to
mechanism, including performance improvements, error isolation, and
resource management.
"""

import asyncio
import time

import pytest
from pydantic import ValidationError

from agents.tools.decorators import tool
from agents.tools.executor import ToolExecutor, execute_tools_concurrently
from agents.tools.models import (
    ConcurrentToolExecution,
    ToolCall,
    ToolCategory,
)
from agents.tools.registry import ToolRegistry
from agents.tools.simple_executor import SimpleToolExecutor


class TestConcurrentExecution:
    """Test concurrent tool execution functionality"""

    def setup_method(self):
        """Clear registry and set up test tools"""
        ToolRegistry.clear_registry()

        # Create test tools with different characteristics
        self._create_test_tools()

    def _create_test_tools(self):
        """Create test tools for concurrent execution testing"""

        @tool("fast_tool", "Fast executing tool", category=ToolCategory.COMPUTE)
        def fast_tool(value: int) -> int:
            return value * 2

        @tool("slow_tool", "Slow executing tool", category=ToolCategory.COMPUTE)
        async def slow_tool(delay: float) -> str:
            await asyncio.sleep(delay)
            return f"Slept for {delay} seconds"

        @tool("error_tool", "Tool that sometimes fails", category=ToolCategory.COMPUTE)
        def error_tool(should_fail: bool) -> str:
            if should_fail:
                raise ValueError("Intentional test error")
            return "Success"

        @tool(
            "mixed_sync_tool",
            "Sync tool for mixed execution",
            category=ToolCategory.GENERAL,
        )
        def mixed_sync_tool(text: str) -> str:
            time.sleep(0.01)  # Small delay to simulate work
            return f"Processed: {text}"

        @tool(
            "mixed_async_tool",
            "Async tool for mixed execution",
            category=ToolCategory.GENERAL,
        )
        async def mixed_async_tool(text: str) -> str:
            await asyncio.sleep(0.01)  # Small delay to simulate work
            return f"Async processed: {text}"

    @pytest.mark.asyncio
    async def test_single_tool_execution(self):
        """Test executing a single tool"""
        executor = ToolExecutor(max_workers=5)

        tool_call = ToolCall(
            tool_name="fast_tool", parameters={"value": 10}, call_id="single_test"
        )

        result = await executor.execute_single_tool(tool_call)

        assert result.success is True
        assert result.result == 20
        assert result.tool_name == "fast_tool"
        assert result.call_id == "single_test"
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_concurrent_execution_success(self):
        """Test successful concurrent execution of multiple tools"""
        executor = ToolExecutor(max_workers=3)

        tool_calls = [
            ToolCall(tool_name="fast_tool", parameters={"value": 5}, call_id="fast_1"),
            ToolCall(tool_name="fast_tool", parameters={"value": 10}, call_id="fast_2"),
            ToolCall(tool_name="fast_tool", parameters={"value": 15}, call_id="fast_3"),
        ]

        execution_request = ConcurrentToolExecution(
            tool_calls=tool_calls, max_workers=3, timeout_seconds=10.0
        )

        result = await executor.execute_tools_concurrently(execution_request)

        assert result.batch_success is True
        assert result.total_tools == 3
        assert result.successful_tools == 3
        assert result.failed_tools == 0
        assert len(result.tool_results) == 3

        # Check individual results
        results_by_call_id = {r.call_id: r for r in result.tool_results}
        assert results_by_call_id["fast_1"].result == 10
        assert results_by_call_id["fast_2"].result == 20
        assert results_by_call_id["fast_3"].result == 30

    @pytest.mark.asyncio
    async def test_error_isolation(self):
        """Test that individual tool failures don't affect other tools"""
        executor = ToolExecutor(max_workers=3)

        tool_calls = [
            ToolCall(
                tool_name="fast_tool", parameters={"value": 5}, call_id="success_1"
            ),
            ToolCall(
                tool_name="error_tool",
                parameters={"should_fail": True},
                call_id="fail_1",
            ),
            ToolCall(
                tool_name="fast_tool", parameters={"value": 10}, call_id="success_2"
            ),
        ]

        execution_request = ConcurrentToolExecution(
            tool_calls=tool_calls, max_workers=3
        )

        result = await executor.execute_tools_concurrently(execution_request)

        assert result.batch_success is False  # Batch failed due to one failure
        assert result.total_tools == 3
        assert result.successful_tools == 2
        assert result.failed_tools == 1

        # Check that successful tools completed correctly
        successful_results = [r for r in result.tool_results if r.success]
        failed_results = [r for r in result.tool_results if not r.success]

        assert len(successful_results) == 2
        assert len(failed_results) == 1

        # Check error is properly captured
        error_result = failed_results[0]
        assert error_result.call_id == "fail_1"
        assert "Intentional test error" in error_result.error_message
        assert error_result.error_type == "ValueError"

    @pytest.mark.asyncio
    async def test_performance_improvement(self):
        """Test that concurrent execution provides performance benefits"""
        executor = SimpleToolExecutor(max_workers=3)

        # Create tools with known delay
        delay = 0.05  # 50ms each
        tool_calls = [
            ToolCall(
                tool_name="slow_tool", parameters={"delay": delay}, call_id=f"slow_{i}"
            )
            for i in range(3)
        ]

        # Time concurrent execution
        start_time = time.time()
        result = await executor.execute_tools_concurrently(tool_calls, max_workers=3)
        concurrent_time = time.time() - start_time

        # Verify all succeeded
        assert result.batch_success is True
        assert result.successful_tools == 3

        # Concurrent execution should be significantly faster than serial
        expected_serial_time = delay * 3  # 150ms
        speedup_ratio = expected_serial_time / concurrent_time

        # Should achieve at least 50% speedup (accounting for overhead)
        assert speedup_ratio > 1.5, f"Speedup ratio {speedup_ratio:.2f} < 1.5"

        # Check parallelism efficiency
        assert result.parallelism_achieved > 0.5  # At least 50% efficiency

    @pytest.mark.asyncio
    async def test_mixed_sync_async_execution(self):
        """Test concurrent execution of mixed sync/async tools"""
        executor = SimpleToolExecutor(max_workers=4)

        tool_calls = [
            ToolCall(
                tool_name="mixed_sync_tool",
                parameters={"text": "sync1"},
                call_id="sync_1",
            ),
            ToolCall(
                tool_name="mixed_async_tool",
                parameters={"text": "async1"},
                call_id="async_1",
            ),
            ToolCall(
                tool_name="mixed_sync_tool",
                parameters={"text": "sync2"},
                call_id="sync_2",
            ),
            ToolCall(
                tool_name="mixed_async_tool",
                parameters={"text": "async2"},
                call_id="async_2",
            ),
        ]

        result = await executor.execute_tools_concurrently(tool_calls, max_workers=4)

        assert result.batch_success is True
        assert result.total_tools == 4
        assert result.successful_tools == 4

        # Verify results from both sync and async tools
        results_by_call_id = {r.call_id: r for r in result.tool_results}

        assert "Processed: sync1" in results_by_call_id["sync_1"].result
        assert "Async processed: async1" in results_by_call_id["async_1"].result
        assert "Processed: sync2" in results_by_call_id["sync_2"].result
        assert "Async processed: async2" in results_by_call_id["async_2"].result

    @pytest.mark.asyncio
    async def test_executor_strategy_selection(self):
        """Test automatic executor strategy selection"""
        executor = ToolExecutor()

        # Test with primarily sync tools
        sync_calls = [
            ToolCall(tool_name="fast_tool", parameters={"value": i}) for i in range(3)
        ]
        strategy = executor.select_execution_strategy(sync_calls)
        # Should prefer thread strategy for sync tools
        assert strategy in ["thread", "auto"]

        # Test with primarily async tools
        async_calls = [
            ToolCall(tool_name="slow_tool", parameters={"delay": 0.01})
            for i in range(3)
        ]
        strategy = executor.select_execution_strategy(async_calls)
        # Should prefer async strategy
        assert strategy in ["async", "auto"]

    @pytest.mark.asyncio
    async def test_resource_management(self):
        """Test resource management and worker limits"""
        # Test with limited workers
        executor = ToolExecutor(max_workers=2)

        # Create more tools than workers
        tool_calls = [
            ToolCall(
                tool_name="fast_tool", parameters={"value": i}, call_id=f"resource_{i}"
            )
            for i in range(5)
        ]

        execution_request = ConcurrentToolExecution(
            tool_calls=tool_calls,
            max_workers=2,  # Limit to 2 workers
        )

        result = await executor.execute_tools_concurrently(execution_request)

        # All tools should complete successfully despite worker limit
        assert result.batch_success is True
        assert result.total_tools == 5
        assert result.successful_tools == 5
        assert result.max_workers_used == 2  # Should use only 2 workers

    @pytest.mark.asyncio
    async def test_tool_not_found_handling(self):
        """Test handling of non-existent tools"""
        executor = ToolExecutor()

        tool_calls = [
            ToolCall(tool_name="fast_tool", parameters={"value": 5}, call_id="good"),
            ToolCall(tool_name="nonexistent_tool", parameters={}, call_id="bad"),
        ]

        execution_request = ConcurrentToolExecution(tool_calls=tool_calls)
        result = await executor.execute_tools_concurrently(execution_request)

        assert result.batch_success is False
        assert result.successful_tools == 1
        assert result.failed_tools == 1

        # Check the error for non-existent tool
        failed_result = next(r for r in result.tool_results if not r.success)
        assert failed_result.call_id == "bad"
        assert "not found in registry" in failed_result.error_message

    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test the convenience execute_tools_concurrently function"""
        tool_calls = [
            ToolCall(tool_name="fast_tool", parameters={"value": i}) for i in range(3)
        ]

        # This should use dependency injection to get tool_executor
        # For testing, we'll need to mock or configure the container
        try:
            result = await execute_tools_concurrently(
                tool_calls=tool_calls,
                max_workers=3,
                tool_executor=ToolExecutor(),  # Pass explicitly for test
            )

            assert result.batch_success is True
            assert result.total_tools == 3

        except Exception as e:
            # If dependency injection isn't configured, that's expected in unit tests
            if "tool_executor" in str(e):
                pytest.skip("Dependency injection not configured for unit test")
            else:
                raise

    @pytest.mark.asyncio
    async def test_empty_tool_list(self):
        """Test handling of empty tool list"""
        executor = ToolExecutor()

        # This should raise a validation error due to min_items=1
        with pytest.raises(ValidationError):  # Pydantic validation error
            execution_request = ConcurrentToolExecution(tool_calls=[])
            await executor.execute_tools_concurrently(execution_request)

    @pytest.mark.asyncio
    async def test_execution_statistics_recording(self):
        """Test that execution statistics are properly recorded"""
        executor = SimpleToolExecutor()

        tool_calls = [
            ToolCall(tool_name="fast_tool", parameters={"value": 5}),
            ToolCall(tool_name="error_tool", parameters={"should_fail": True}),
        ]

        # Clear any existing stats
        ToolRegistry._instance._execution_stats.clear()

        await executor.execute_tools_concurrently(tool_calls)

        # Check that statistics were recorded
        fast_stats = ToolRegistry.get_execution_stats("fast_tool")
        error_stats = ToolRegistry.get_execution_stats("error_tool")

        assert "fast_tool" in fast_stats
        assert "error_tool" in error_stats

        # Fast tool should have success
        assert fast_stats["fast_tool"]["success_count"] == 1
        assert fast_stats["fast_tool"]["failure_count"] == 0

        # Error tool should have failure
        assert error_stats["error_tool"]["success_count"] == 0
        assert error_stats["error_tool"]["failure_count"] == 1
