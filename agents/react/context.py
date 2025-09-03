"""ReAct Agent Context

扩展AetherFlow的BaseFlowContext，提供ReAct代理专用的状态管理和会话历史功能。
"""

import time
from typing import Any

from dependency_injector import containers, providers
from pydantic import BaseModel, Field

from aetherflow import ContextVarProvider

from .models import ActionResult, ConversationEntry, ObservationResult, ReasoningResult


class ReActState(BaseModel):
    """ReAct代理状态模型

    管理ReAct循环的执行状态和会话历史。
    """

    # 会话管理
    conversation_history: list[ConversationEntry] = Field(
        default_factory=list, description="完整的会话历史记录"
    )

    # 执行状态
    current_step: int = Field(default=0, ge=0, description="当前步骤编号")
    max_steps: int = Field(default=10, ge=1, description="最大执行步骤数")
    is_complete: bool = Field(default=False, description="是否已完成执行")

    # 结果管理
    final_answer: str | None = Field(None, description="最终答案")
    termination_reason: str = Field(default="", description="终止原因")

    # 执行历史
    reasoning_history: list[ReasoningResult] = Field(
        default_factory=list, description="推理步骤历史"
    )
    action_history: list[ActionResult] = Field(
        default_factory=list, description="行动步骤历史"
    )
    observation_history: list[ObservationResult] = Field(
        default_factory=list, description="观察步骤历史"
    )

    # 性能统计
    start_time: float | None = Field(None, description="开始时间戳")
    total_reasoning_tokens: int = Field(default=0, ge=0, description="总推理token消耗")

    class Config:
        # 允许任意类型，支持复杂的状态对象
        arbitrary_types_allowed = True


class ReActContextContainer(containers.DeclarativeContainer):
    """ReAct代理上下文容器定义

    定义ReAct代理专用的依赖注入提供者。
    """

    # 继承BaseFlowContext的基础提供者
    state: providers.Provider = providers.ThreadLocalSingleton(dict)
    context: providers.Provider = providers.ThreadLocalSingleton(dict)
    shared_data: providers.Provider = providers.Singleton(dict)

    # ReAct特定的状态提供者
    react_state: providers.Provider = providers.ThreadLocalSingleton(ReActState)

    # 异步环境下的状态提供者（使用ContextVar）
    async_react_state: providers.Provider = ContextVarProvider(ReActState)


class ReActContext:
    """ReAct代理执行上下文

    提供ReAct代理专用的状态管理功能，集成依赖注入容器，
    支持线程安全的状态隔离。
    """

    def __init__(self):
        """初始化ReAct上下文"""
        self.container = ReActContextContainer()

    def get_react_state(self) -> ReActState:
        """获取当前线程/协程的ReAct状态

        Returns:
            当前的ReActState实例
        """
        # 暂时只使用ThreadLocalSingleton，确保稳定性
        # TODO: 修复ContextVarProvider对自定义类的支持
        return self.container.react_state()

    def add_conversation_entry(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """添加会话条目

        Args:
            role: 角色 (user, assistant, system, tool)
            content: 消息内容
            metadata: 可选的元数据
        """
        state = self.get_react_state()
        entry = ConversationEntry(
            role=role, content=content, timestamp=time.time(), metadata=metadata or {}
        )
        state.conversation_history.append(entry)

    def add_reasoning(self, reasoning_result: ReasoningResult) -> None:
        """添加推理步骤记录

        Args:
            reasoning_result: 推理结果
        """
        state = self.get_react_state()
        state.reasoning_history.append(reasoning_result)
        state.total_reasoning_tokens += reasoning_result.reasoning_tokens

    def add_action(self, action_result: ActionResult) -> None:
        """添加行动记录

        Args:
            action_result: 行动结果
        """
        state = self.get_react_state()
        state.action_history.append(action_result)

    def add_observation(self, observation_result: ObservationResult) -> None:
        """添加观察结果

        Args:
            observation_result: 观察结果
        """
        state = self.get_react_state()
        state.observation_history.append(observation_result)

        # 更新执行状态
        state.current_step += 1
        state.is_complete = not observation_result.should_continue

        # 如果有最终答案，更新终止原因
        if not observation_result.should_continue:
            if state.final_answer:
                state.termination_reason = "final_answer_provided"
            elif state.current_step >= state.max_steps:
                state.termination_reason = "max_steps_reached"
            else:
                state.termination_reason = "observation_indicated_completion"

    def should_continue(self) -> bool:
        """判断是否应该继续ReAct循环

        Returns:
            是否应该继续执行
        """
        state = self.get_react_state()

        # 检查完成状态
        if state.is_complete:
            return False

        # 检查最大步骤数
        if state.current_step >= state.max_steps:
            state.is_complete = True
            state.termination_reason = "max_steps_reached"
            return False

        # 检查是否有最终答案
        if state.final_answer:
            state.is_complete = True
            state.termination_reason = "final_answer_provided"
            return False

        return True

    def get_current_step(self) -> int:
        """获取当前步骤编号"""
        state = self.get_react_state()
        return state.current_step

    def get_conversation_context(self, max_entries: int = 10) -> list[dict[str, str]]:
        """获取会话上下文，用于LLM调用

        Args:
            max_entries: 最大条目数

        Returns:
            格式化的会话上下文
        """
        state = self.get_react_state()
        recent_entries = (
            state.conversation_history[-max_entries:]
            if max_entries > 0
            else state.conversation_history
        )

        return [
            {"role": entry.role, "content": entry.content, "timestamp": entry.timestamp}
            for entry in recent_entries
        ]

    def initialize_execution(self, max_steps: int = 10) -> None:
        """初始化执行环境

        Args:
            max_steps: 最大执行步骤数
        """
        state = self.get_react_state()
        state.max_steps = max_steps
        state.current_step = 0
        state.is_complete = False
        state.final_answer = None
        state.termination_reason = ""
        state.start_time = time.time()

        # 清空历史记录
        state.reasoning_history.clear()
        state.action_history.clear()
        state.observation_history.clear()
        state.total_reasoning_tokens = 0

    def get_execution_summary(self) -> dict[str, Any]:
        """获取执行摘要统计信息

        Returns:
            执行摘要字典
        """
        state = self.get_react_state()

        current_time = time.time()
        execution_time = (current_time - state.start_time) if state.start_time else 0.0

        return {
            "current_step": state.current_step,
            "max_steps": state.max_steps,
            "is_complete": state.is_complete,
            "final_answer": state.final_answer,
            "termination_reason": state.termination_reason,
            "execution_time_ms": execution_time * 1000,
            "total_reasoning_tokens": state.total_reasoning_tokens,
            "conversation_entries": len(state.conversation_history),
            "reasoning_steps": len(state.reasoning_history),
            "action_steps": len(state.action_history),
            "observation_steps": len(state.observation_history),
        }
