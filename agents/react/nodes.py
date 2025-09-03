"""ReAct Agent Core Nodes

实现ReAct循环的三个核心节点：reasoning_step、action_step、observation_step。
使用AetherFlow的@node装饰器提供类型安全、依赖注入和异步兼容性。
"""

import asyncio
import logging
import time
from typing import Any

from dependency_injector.wiring import Provide

from aetherflow import node

from .context import ReActContext
from .models import (
    ActionResult,
    ActionType,
    ObservationResult,
    ReasoningResult,
    ToolCall,
)

logger = logging.getLogger("aetherflow.react")


@node(enable_retry=True, retry_count=2, retry_delay=0.5)
async def reasoning_step(
    query: str,
    context: ReActContext = Provide[ReActContext],
    llm_client: Any | None = None,
) -> ReasoningResult:
    """推理阶段：分析当前状态，决定下一步行动

    这是ReAct循环的第一阶段，代理分析当前情况，回顾会话历史，
    决定下一步应该采取什么行动。

    Args:
        query: 用户查询或当前需要处理的问题
        context: ReAct执行上下文
        llm_client: LLM客户端（可选，为后续集成预留）

    Returns:
        ReasoningResult: 推理结果，包含思考过程和下一步行动计划
    """
    start_time = time.time()

    try:
        # 获取当前状态
        current_step = context.get_current_step()
        conversation_history = context.get_conversation_context()

        logger.info(
            f"Starting reasoning step {current_step + 1} for query: {query[:100]}..."
        )

        # 模拟推理过程（在实际实现中，这里会调用LLM）
        if current_step == 0:
            # 首次推理：理解问题并计划行动
            thought = f"用户询问: {query}. 我需要分析这个问题并决定如何回应。"
            analysis = "这是一个新的查询，我需要确定是否需要使用工具来获取信息，还是可以直接回答。"
            next_action_type = (
                ActionType.TOOL_CALL
                if _requires_tool_call(query)
                else ActionType.FINAL_ANSWER
            )
        else:
            # 后续推理：基于历史信息继续分析
            thought = "基于之前的观察结果，我需要继续分析当前情况。"
            analysis = (
                "根据之前的行动结果，我需要决定下一步是继续收集信息还是提供最终答案。"
            )

            # 简单逻辑：如果步骤数过多，给出最终答案
            if current_step >= 3:
                next_action_type = ActionType.FINAL_ANSWER
            else:
                next_action_type = ActionType.CONTINUE_THINKING

        # 计算推理时间和模拟token消耗
        reasoning_time = time.time() - start_time
        reasoning_tokens = (
            len(thought.split()) + len(analysis.split()) + 50
        )  # 模拟token计算

        reasoning_result = ReasoningResult(
            thought=thought,
            analysis=analysis,
            next_action_type=next_action_type,
            confidence=0.8,  # 固定置信度，实际实现中由LLM提供
            reasoning_tokens=reasoning_tokens,
        )

        # 更新上下文
        context.add_reasoning(reasoning_result)

        logger.info(
            f"Reasoning completed in {reasoning_time * 1000:.2f}ms, next action: {next_action_type}"
        )

        return reasoning_result

    except Exception as e:
        logger.error(f"Error in reasoning step: {e}")
        raise


def _requires_tool_call(query: str) -> bool:
    """判断查询是否需要工具调用（简化版本）

    Args:
        query: 用户查询

    Returns:
        是否需要工具调用
    """
    # 简单的关键词检测
    tool_keywords = ["搜索", "查找", "计算", "获取", "查询", "数据", "信息"]
    return any(keyword in query for keyword in tool_keywords)


@node(enable_retry=True, retry_count=2, retry_delay=0.5)
async def action_step(
    reasoning: ReasoningResult,
    context: ReActContext = Provide[ReActContext],
    tool_registry: dict[str, Any] | None = None,
) -> ActionResult:
    """行动阶段：执行工具调用或生成最终回复

    基于推理结果执行具体的行动，可能是调用工具获取信息，
    或者直接生成最终答案。

    Args:
        reasoning: 推理阶段的结果
        context: ReAct执行上下文
        tool_registry: 工具注册表（可选，为后续集成预留）

    Returns:
        ActionResult: 行动执行结果
    """
    start_time = time.time()

    try:
        logger.info(f"Executing action of type: {reasoning.next_action_type}")

        if reasoning.next_action_type == ActionType.TOOL_CALL:
            # 执行工具调用
            action_result = await _execute_tool_call(reasoning, tool_registry)

        elif reasoning.next_action_type == ActionType.FINAL_ANSWER:
            # 生成最终答案
            action_result = _generate_final_answer(reasoning, context)

        elif reasoning.next_action_type == ActionType.CONTINUE_THINKING:
            # 继续思考（无具体行动）
            action_result = ActionResult(
                action_type=ActionType.CONTINUE_THINKING,
                content="继续分析问题，收集更多信息后再做决定。",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        else:
            raise ValueError(f"Unknown action type: {reasoning.next_action_type}")

        # 更新上下文
        context.add_action(action_result)

        logger.info(f"Action completed in {action_result.execution_time_ms:.2f}ms")

        return action_result

    except Exception as e:
        logger.error(f"Error in action step: {e}")
        raise


async def _execute_tool_call(
    reasoning: ReasoningResult, tool_registry: dict[str, Any] | None
) -> ActionResult:
    """执行工具调用（模拟实现）

    Args:
        reasoning: 推理结果
        tool_registry: 工具注册表

    Returns:
        工具调用的行动结果
    """
    start_time = time.time()

    # 模拟工具调用（实际实现中会调用真实工具）
    tool_call = ToolCall(
        name="search_tool",
        parameters={"query": "模拟搜索查询"},
        call_id=f"call_{int(time.time())}",
    )

    # 模拟异步工具执行
    await asyncio.sleep(0.1)  # 模拟I/O延迟

    return ActionResult(
        action_type=ActionType.TOOL_CALL,
        content=f"使用工具 {tool_call.name} 执行查询",
        tool_call=tool_call,
        execution_time_ms=(time.time() - start_time) * 1000,
    )


def _generate_final_answer(
    reasoning: ReasoningResult, context: ReActContext
) -> ActionResult:
    """生成最终答案

    Args:
        reasoning: 推理结果
        context: 执行上下文

    Returns:
        包含最终答案的行动结果
    """
    start_time = time.time()

    # 获取执行摘要
    summary = context.get_execution_summary()

    # 生成基于推理历史的最终答案
    final_answer = (
        f"基于分析，我的回答如下：\n"
        f"思考过程：{reasoning.thought}\n"
        f"分析结果：{reasoning.analysis}\n"
        f"经过 {summary['current_step']} 步推理，我提供以上回答。"
    )

    # 更新上下文的最终答案
    react_state = context.get_react_state(context)
    react_state.final_answer = final_answer

    return ActionResult(
        action_type=ActionType.FINAL_ANSWER,
        content="生成最终答案",
        final_answer=final_answer,
        execution_time_ms=(time.time() - start_time) * 1000,
    )


@node(enable_retry=False)  # 观察步骤通常不需要重试
async def observation_step(
    action: ActionResult,
    context: ReActContext = Provide[ReActContext],
) -> ObservationResult:
    """观察阶段：处理行动结果，更新上下文状态

    分析行动的执行结果，决定是否继续ReAct循环，
    并更新上下文状态。

    Args:
        action: 行动阶段的结果
        context: ReAct执行上下文

    Returns:
        ObservationResult: 观察结果，包含是否继续循环的决策
    """
    start_time = time.time()

    try:
        logger.info(f"Observing action result of type: {action.action_type}")

        success = True
        error_message = None
        updated_context = {}

        # 根据行动类型处理结果
        if action.action_type == ActionType.FINAL_ANSWER:
            # 最终答案：完成执行
            observation = f"已生成最终答案：{action.final_answer[:100]}..."
            should_continue = False

            # 添加到会话历史
            context.add_conversation_entry("assistant", action.final_answer)

        elif action.action_type == ActionType.TOOL_CALL:
            # 工具调用：处理工具输出
            tool_output = _simulate_tool_output(action.tool_call)
            observation = f"工具 {action.tool_call.name} 执行成功，获得结果"
            should_continue = True  # 需要基于工具结果继续推理

            updated_context["last_tool_output"] = tool_output

            # 添加工具结果到会话历史
            context.add_conversation_entry(
                "tool",
                str(tool_output),
                {
                    "tool_name": action.tool_call.name,
                    "call_id": action.tool_call.call_id,
                },
            )

        elif action.action_type == ActionType.CONTINUE_THINKING:
            # 继续思考：准备下一轮推理
            observation = "继续思考模式，准备进行更深入的分析"
            should_continue = True

        else:
            # 未知行动类型
            observation = f"未知的行动类型: {action.action_type}"
            should_continue = False
            success = False
            error_message = f"Unsupported action type: {action.action_type}"

        observation_result = ObservationResult(
            observation=observation,
            tool_output=updated_context.get("last_tool_output"),
            success=success,
            error_message=error_message,
            should_continue=should_continue,
            updated_context=updated_context,
        )

        # 更新上下文（这会自动更新步骤计数和完成状态）
        context.add_observation(observation_result)

        execution_time = time.time() - start_time
        logger.info(
            f"Observation completed in {execution_time * 1000:.2f}ms, continue: {should_continue}"
        )

        return observation_result

    except Exception as e:
        logger.error(f"Error in observation step: {e}")
        raise


def _simulate_tool_output(tool_call: ToolCall) -> dict[str, Any]:
    """模拟工具输出（用于测试）

    Args:
        tool_call: 工具调用信息

    Returns:
        模拟的工具输出
    """
    return {
        "result": f"模拟的 {tool_call.name} 执行结果",
        "data": {"status": "success", "items": ["item1", "item2", "item3"]},
        "metadata": {
            "execution_time": "100ms",
            "call_id": tool_call.call_id,
            "timestamp": time.time(),
        },
    }
