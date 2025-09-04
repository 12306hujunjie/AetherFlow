"""
Enhanced ReAct Nodes with Tool Integration

Extends the core ReAct nodes to support real tool execution through
the AetherFlow tool system. Provides backward compatibility while
adding concurrent tool execution capabilities.
"""

import logging
import time
from typing import Any

from dependency_injector.wiring import Provide

from aetherflow import node

from ..tools.executor import ToolExecutor
from ..tools.models import ConcurrentToolExecution, ToolCall
from .context import ReActContext
from .models import (
    ActionResult,
    ActionType,
    ObservationResult,
    ReasoningResult,
)

logger = logging.getLogger("aetherflow.react.enhanced")


@node(enable_retry=True, retry_count=2, retry_delay=0.5)
async def enhanced_action_step(
    reasoning: ReasoningResult,
    context: ReActContext = Provide[ReActContext],
    tool_executor: ToolExecutor = Provide["tool_executor"],
    enable_concurrent_tools: bool = True,
    max_concurrent_tools: int = 3,
) -> ActionResult:
    """
    Enhanced action step with real tool execution capabilities.

    Extends the basic action_step to support:
    - Real tool registry integration
    - Concurrent tool execution via AetherFlow fan_out_to
    - Comprehensive error handling and isolation
    - Performance monitoring and statistics

    Args:
        reasoning: Reasoning result from previous step
        context: ReAct execution context
        tool_executor: Tool execution service (injected)
        enable_concurrent_tools: Whether to enable concurrent tool execution
        max_concurrent_tools: Maximum tools to execute concurrently

    Returns:
        Enhanced action result with tool execution data
    """
    start_time = time.time()

    try:
        logger.info(f"Enhanced action step: {reasoning.next_action_type}")

        if reasoning.next_action_type == ActionType.TOOL_CALL:
            # Extract tool calls from reasoning
            tool_calls = _extract_tool_calls_from_reasoning(reasoning, context)

            if not tool_calls:
                # No valid tools extracted, fall back to mock behavior
                logger.warning("No valid tool calls extracted from reasoning")
                return _create_no_tools_result(reasoning, start_time)

            # Execute tools (concurrent or sequential based on configuration)
            if enable_concurrent_tools and len(tool_calls) > 1:
                execution_result = await _execute_tools_concurrently(
                    tool_calls, tool_executor, max_concurrent_tools
                )
            else:
                execution_result = await _execute_tools_sequentially(
                    tool_calls, tool_executor
                )

            # Create action result from tool execution
            action_result = ActionResult(
                action_type=ActionType.TOOL_CALL,
                content=_format_tool_execution_summary(execution_result),
                tool_call=tool_calls[0]
                if tool_calls
                else None,  # For backward compatibility
                execution_time_ms=(time.time() - start_time) * 1000,
            )

            # Store detailed results in context for observation step
            context.add_tool_execution_result(execution_result)

            return action_result

        elif reasoning.next_action_type == ActionType.FINAL_ANSWER:
            # Generate final answer (same as original implementation)
            return _generate_enhanced_final_answer(reasoning, context, start_time)

        elif reasoning.next_action_type == ActionType.CONTINUE_THINKING:
            # Continue thinking (same as original implementation)
            return ActionResult(
                action_type=ActionType.CONTINUE_THINKING,
                content="Continuing analysis to gather more information before deciding on action.",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        else:
            raise ValueError(f"Unknown action type: {reasoning.next_action_type}")

    except Exception as e:
        logger.error(f"Error in enhanced action step: {e}", exc_info=True)

        # Return error action result
        return ActionResult(
            action_type=reasoning.next_action_type,
            content=f"Action execution failed: {str(e)}",
            execution_time_ms=(time.time() - start_time) * 1000,
        )


def _extract_tool_calls_from_reasoning(
    reasoning: ReasoningResult, context: ReActContext
) -> list[ToolCall]:
    """
    Extract tool calls from reasoning content.

    This is a simplified implementation that demonstrates the concept.
    A real implementation would parse LLM responses for tool call instructions.

    Args:
        reasoning: Reasoning result to parse
        context: Execution context for additional clues

    Returns:
        List of extracted tool calls
    """
    tool_calls = []

    # Simple keyword-based extraction (in reality, this would parse structured LLM output)
    thought = reasoning.thought.lower()
    analysis = reasoning.analysis.lower()
    combined_text = f"{thought} {analysis}"

    # Example extraction logic
    if any(word in combined_text for word in ["计算", "算", "数学", "math"]):
        tool_calls.append(
            ToolCall(
                tool_name="calculator",
                parameters={"operation": "add", "a": 10, "b": 5},
                call_id=f"calc_{int(time.time())}",
            )
        )

    if any(word in combined_text for word in ["搜索", "查找", "search", "find"]):
        # Extract search query from reasoning
        query = _extract_search_query(combined_text) or "sample query"
        tool_calls.append(
            ToolCall(
                tool_name="mock_search",
                parameters={"query": query, "max_results": 3},
                call_id=f"search_{int(time.time())}",
            )
        )

    if any(word in combined_text for word in ["网址", "url", "网站", "http"]):
        tool_calls.append(
            ToolCall(
                tool_name="url_analyzer",
                parameters={"url": "https://example.com/test"},
                call_id=f"url_{int(time.time())}",
            )
        )

    return tool_calls


def _extract_search_query(text: str) -> str | None:
    """Extract search query from text (simplified implementation)"""
    # This is a very basic implementation
    # Real implementation would use NLP or structured parsing
    words = text.split()
    search_indicators = ["搜索", "查找", "search", "find"]

    for i, word in enumerate(words):
        if word in search_indicators and i + 1 < len(words):
            # Return next few words as query
            return " ".join(words[i + 1 : i + 4])

    return "information query"


async def _execute_tools_concurrently(
    tool_calls: list[ToolCall],
    tool_executor: ToolExecutor,
    max_workers: int,
) -> Any:
    """Execute tools concurrently using the tool system"""
    execution_request = ConcurrentToolExecution(
        tool_calls=tool_calls,
        max_workers=min(max_workers, len(tool_calls)),
        timeout_seconds=30.0,
        executor_strategy="auto",
    )

    return await tool_executor.execute_tools_concurrently(execution_request)


async def _execute_tools_sequentially(
    tool_calls: list[ToolCall],
    tool_executor: ToolExecutor,
) -> Any:
    """Execute tools sequentially"""
    results = []
    for tool_call in tool_calls:
        result = await tool_executor.execute_single_tool(tool_call)
        results.append(result)

    # Create a simple result structure similar to concurrent execution
    from ..tools.models import ConcurrentExecutionResult

    return ConcurrentExecutionResult(
        tool_results=results,
        batch_success=all(r.success for r in results),
        total_execution_time_ms=sum(r.execution_time_ms for r in results),
        total_tools=len(results),
        max_workers_used=1,
        executor_strategy_used="sequential",
    )


def _format_tool_execution_summary(execution_result) -> str:
    """Format tool execution results for display"""
    if hasattr(execution_result, "tool_results"):
        successful = sum(1 for r in execution_result.tool_results if r.success)
        total = len(execution_result.tool_results)

        if total == 1:
            result = execution_result.tool_results[0]
            return f"Executed tool '{result.tool_name}': {'Success' if result.success else 'Failed'}"
        else:
            return f"Executed {total} tools concurrently: {successful} succeeded, {total - successful} failed"
    else:
        return "Tool execution completed"


def _create_no_tools_result(
    reasoning: ReasoningResult, start_time: float
) -> ActionResult:
    """Create action result when no tools are available"""
    return ActionResult(
        action_type=ActionType.CONTINUE_THINKING,
        content="No applicable tools found for this reasoning. Continuing with analysis.",
        execution_time_ms=(time.time() - start_time) * 1000,
    )


def _generate_enhanced_final_answer(
    reasoning: ReasoningResult, context: ReActContext, start_time: float
) -> ActionResult:
    """Generate enhanced final answer with tool execution history"""

    # Get execution summary including tool results
    summary = context.get_execution_summary()
    tool_history = getattr(context, "_tool_execution_history", [])

    # Build final answer incorporating tool results
    answer_parts = [
        "基于分析和推理，我的回答如下：",
        f"思考过程：{reasoning.thought}",
        f"分析结果：{reasoning.analysis}",
    ]

    if tool_history:
        answer_parts.append("工具执行结果:")
        for result in tool_history[-3:]:  # Show last 3 tool executions
            if hasattr(result, "tool_results"):
                for tool_result in result.tool_results[:2]:  # Show first 2 results
                    if tool_result.success:
                        answer_parts.append(f"- {tool_result.tool_name}: 执行成功")
                    else:
                        answer_parts.append(f"- {tool_result.tool_name}: 执行失败")

    answer_parts.append(
        f"经过 {summary['current_step']} 步推理和工具调用，我提供以上回答。"
    )

    final_answer = "\n".join(answer_parts)

    # Update context with final answer
    react_state = context.get_react_state(context)
    react_state.final_answer = final_answer

    return ActionResult(
        action_type=ActionType.FINAL_ANSWER,
        content="Generated enhanced final answer with tool execution history",
        final_answer=final_answer,
        execution_time_ms=(time.time() - start_time) * 1000,
    )


@node(enable_retry=False)
async def enhanced_observation_step(
    action: ActionResult,
    context: ReActContext = Provide[ReActContext],
) -> ObservationResult:
    """
    Enhanced observation step with tool result processing.

    Processes action results including tool execution outcomes,
    updating context with tool results and performance metrics.

    Args:
        action: Action result to observe
        context: ReAct execution context

    Returns:
        Enhanced observation result
    """
    start_time = time.time()

    try:
        logger.info(f"Enhanced observation of action type: {action.action_type}")

        success = True
        error_message = None
        updated_context = {}

        if action.action_type == ActionType.FINAL_ANSWER:
            # Final answer processing (same as original)
            observation = f"Generated final answer: {action.final_answer[:100]}..."
            should_continue = False

            context.add_conversation_entry("assistant", action.final_answer)

        elif action.action_type == ActionType.TOOL_CALL:
            # Enhanced tool call processing
            tool_execution_history = getattr(context, "_tool_execution_history", [])

            if tool_execution_history:
                latest_execution = tool_execution_history[-1]
                observation = f"Tool execution completed: {latest_execution.successful_tools}/{latest_execution.total_tools} succeeded"
                should_continue = True

                # Add successful tool results to conversation history
                for tool_result in latest_execution.tool_results:
                    if tool_result.success:
                        context.add_conversation_entry(
                            "tool",
                            f"Tool {tool_result.tool_name} result: {str(tool_result.result)[:200]}...",
                            {
                                "tool_name": tool_result.tool_name,
                                "call_id": tool_result.call_id,
                                "execution_time_ms": tool_result.execution_time_ms,
                            },
                        )

                updated_context["latest_tool_execution"] = latest_execution
            else:
                observation = (
                    "Tool call action completed but no execution results found"
                )
                should_continue = True

        elif action.action_type == ActionType.CONTINUE_THINKING:
            # Continue thinking processing (same as original)
            observation = "Continuing analysis mode, preparing for deeper investigation"
            should_continue = True

        else:
            # Unknown action type
            observation = f"Unknown action type processed: {action.action_type}"
            should_continue = False
            success = False
            error_message = f"Unsupported action type: {action.action_type}"

        observation_result = ObservationResult(
            observation=observation,
            tool_output=updated_context.get("latest_tool_execution"),
            success=success,
            error_message=error_message,
            should_continue=should_continue,
            updated_context=updated_context,
        )

        # Update context
        context.add_observation(observation_result)

        execution_time = time.time() - start_time
        logger.info(
            f"Enhanced observation completed in {execution_time * 1000:.2f}ms, continue: {should_continue}"
        )

        return observation_result

    except Exception as e:
        logger.error(f"Error in enhanced observation step: {e}", exc_info=True)
        raise


# Add helper method to ReActContext for tool execution history
def _extend_react_context():
    """Extend ReActContext with tool execution tracking"""

    def add_tool_execution_result(self, execution_result):
        """Add tool execution result to context history"""
        if not hasattr(self, "_tool_execution_history"):
            self._tool_execution_history = []
        self._tool_execution_history.append(execution_result)

        # Limit history size
        if len(self._tool_execution_history) > 10:
            self._tool_execution_history = self._tool_execution_history[-10:]

    # Add method to ReActContext class
    ReActContext.add_tool_execution_result = add_tool_execution_result


# Initialize context extension
_extend_react_context()
