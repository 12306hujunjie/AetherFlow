"""ReAct Agent Engine

实现ReAct循环的核心引擎，提供fluent interface组合和循环控制。
基于AetherFlow的Node组合能力构建智能代理执行流程。
"""

import logging
from functools import partial
from typing import Any

from aetherflow import Node

from .context import ReActContext
from .models import ReActExecutionResult
from .nodes import action_step, observation_step, reasoning_step

logger = logging.getLogger("aetherflow.react")


def create_react_agent(
    context: ReActContext,
    max_steps: int = 10,
    stop_on_error: bool = False,
) -> Node:
    """创建完整的ReAct代理执行流程

    使用AetherFlow的fluent interface构建ReAct循环，支持：
    - 推理 → 行动 → 观察的循环流程
    - 基于条件的动态终止
    - 异步/同步混合执行
    - 完整的错误处理和重试机制

    Args:
        context: ReAct执行上下文，管理代理状态
        max_steps: 最大执行步骤数，防止无限循环
        stop_on_error: 遇到错误时是否立即停止

    Returns:
        配置好的Node，可以通过调用执行ReAct循环

    Example:
        ```python
        context = ReActContext()
        agent = create_react_agent(context, max_steps=5)
        result = await agent("用户的查询问题")
        ```
    """
    # 初始化执行环境
    context.initialize_execution(max_steps=max_steps)

    logger.info(f"Creating ReAct agent with max_steps={max_steps}")

    # 创建单次ReAct循环：推理 → 行动 → 观察
    def create_single_react_cycle(query: str) -> Node:
        """创建单次ReAct循环"""
        # 使用partial绑定查询参数到推理节点
        reasoning_node = Node(
            func=partial(_reasoning_with_query, query=query),
            name="reasoning_step",
            is_start_node=True,
        )

        # 构建链式流程：推理 → 行动 → 观察
        return reasoning_node.then(action_step).then(observation_step)

    # 创建带循环控制的ReAct引擎
    def react_engine(query: str) -> ReActExecutionResult:
        """ReAct引擎主执行函数"""
        import time

        start_time = time.time()

        try:
            logger.info(f"Starting ReAct execution for query: {query[:100]}...")

            # 添加用户查询到会话历史
            context.add_conversation_entry("user", query)

            # 创建单次循环节点
            single_cycle = create_single_react_cycle(query)

            # 执行ReAct循环，直到满足终止条件
            last_observation = None
            step_count = 0

            while step_count < max_steps and context.should_continue():
                try:
                    step_count += 1
                    logger.info(f"Executing ReAct cycle {step_count}/{max_steps}")

                    # 执行单次循环
                    last_observation = single_cycle()

                    # 检查是否应该继续
                    if not last_observation.should_continue:
                        logger.info(f"ReAct cycle completed after {step_count} steps")
                        break

                except Exception as e:
                    logger.error(f"Error in ReAct cycle {step_count}: {e}")
                    if stop_on_error:
                        raise
                    # 继续下一次循环，但记录错误
                    context.add_conversation_entry(
                        "system", f"循环 {step_count} 发生错误: {str(e)}"
                    )

            # 生成执行结果
            execution_time = (time.time() - start_time) * 1000
            summary = context.get_execution_summary()

            result = ReActExecutionResult(
                success=summary["is_complete"],
                final_answer=summary["final_answer"],
                total_steps=summary["current_step"],
                execution_time_ms=execution_time,
                termination_reason=summary["termination_reason"] or "max_steps_reached",
                error_message=None if summary["is_complete"] else "未完成执行",
                total_reasoning_tokens=summary["total_reasoning_tokens"],
            )

            logger.info(f"ReAct execution completed: {result.termination_reason}")
            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"ReAct execution failed: {e}")

            return ReActExecutionResult(
                success=False,
                final_answer=None,
                total_steps=context.get_current_step(),
                execution_time_ms=execution_time,
                termination_reason="error",
                error_message=str(e),
                total_reasoning_tokens=context.get_execution_summary()[
                    "total_reasoning_tokens"
                ],
            )

    # 返回包装后的引擎节点
    return Node(func=react_engine, name="react_agent_engine", is_start_node=True)


async def _reasoning_with_query(query: str, context: ReActContext) -> Any:
    """推理步骤的查询绑定版本

    Args:
        query: 用户查询
        context: ReAct上下文

    Returns:
        推理结果
    """
    return await reasoning_step(query, context)


def create_react_single_step(
    query: str,
    context: ReActContext,
) -> Node:
    """创建单步ReAct执行（用于测试或调试）

    Args:
        query: 用户查询
        context: ReAct执行上下文

    Returns:
        单步执行的Node
    """
    reasoning_with_query = Node(
        func=partial(_reasoning_with_query, query=query), name="single_reasoning_step"
    )

    return reasoning_with_query.then(action_step).then(observation_step)


def create_react_reasoning_only(
    query: str,
    context: ReActContext,
) -> Node:
    """创建仅推理的Node（用于测试推理功能）

    Args:
        query: 用户查询
        context: ReAct执行上下文

    Returns:
        仅推理的Node
    """
    return Node(func=partial(_reasoning_with_query, query=query), name="reasoning_only")


class ReActEngineBuilder:
    """ReAct引擎构建器，提供流式配置接口"""

    def __init__(self):
        self._context: ReActContext | None = None
        self._max_steps: int = 10
        self._stop_on_error: bool = False
        self._custom_tools: dict[str, Any] = {}
        self._custom_prompts: dict[str, str] = {}

    def with_context(self, context: ReActContext) -> "ReActEngineBuilder":
        """设置执行上下文"""
        self._context = context
        return self

    def max_steps(self, steps: int) -> "ReActEngineBuilder":
        """设置最大执行步骤数"""
        if steps <= 0:
            raise ValueError("max_steps must be positive")
        self._max_steps = steps
        return self

    def stop_on_error(self, stop: bool = True) -> "ReActEngineBuilder":
        """设置错误停止策略"""
        self._stop_on_error = stop
        return self

    def with_tools(self, tools: dict[str, Any]) -> "ReActEngineBuilder":
        """添加自定义工具（为后续扩展预留）"""
        self._custom_tools.update(tools)
        return self

    def with_prompts(self, prompts: dict[str, str]) -> "ReActEngineBuilder":
        """添加自定义提示词（为后续扩展预留）"""
        self._custom_prompts.update(prompts)
        return self

    def build(self) -> Node:
        """构建ReAct引擎"""
        if self._context is None:
            raise ValueError("Context is required. Use with_context() to set it.")

        return create_react_agent(
            context=self._context,
            max_steps=self._max_steps,
            stop_on_error=self._stop_on_error,
        )


def create_react_engine_builder() -> ReActEngineBuilder:
    """创建ReAct引擎构建器的便捷函数

    Returns:
        新的ReActEngineBuilder实例

    Example:
        ```python
        context = ReActContext()
        agent = (create_react_engine_builder()
                .with_context(context)
                .max_steps(5)
                .stop_on_error(True)
                .build())

        result = await agent("用户查询")
        ```
    """
    return ReActEngineBuilder()
