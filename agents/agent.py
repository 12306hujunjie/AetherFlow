"""
ReAct Agent Main Class

This module provides the main ReActAgent class, which serves as the primary
interface for creating and managing ReAct (Reasoning-Acting-Observing) agents.

The ReActAgent class integrates all core components:
- ReAct execution engine
- LLM interface layer
- Tool system
- Memory management
- Session management

It provides both simple factory methods and fluent interface configuration
for maximum flexibility and ease of use.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from typing import Any

from agents.container import AgentContainer, ContainerFactory
from agents.exceptions import (
    SessionError,
)
from agents.memory import SessionState
from agents.models import (
    AgentResponse,
    AgentStreamChunk,
    Session,
    ToolInfo,
)
from agents.react import ReActContext, create_react_agent
from agents.tools import ToolRegistry

logger = logging.getLogger(__name__)


class ReActAgent:
    """
    ReAct代理主类，提供完整的智能代理能力。

    该类作为门面(Facade)模式的实现，隐藏内部复杂性，
    提供简洁直观的API接口，支持fluent interface链式配置。

    主要功能：
    - 创建和配置ReAct代理
    - 执行推理-行动-观察循环
    - 管理多会话状态
    - 工具注册和调用
    - 流式和非流式执行

    Example:
        # 快速创建
        agent = ReActAgent(model="gpt-4")

        # 链式配置
        agent = (ReActAgent()
                .with_model("gpt-4")
                .with_tools(weather_tool, calculator)
                .with_max_steps(5))

        # 执行查询
        response = await agent.run("What's the weather in Beijing?")
    """

    def __init__(
        self,
        name: str = "ReActAgent",
        model: str = "gpt-4",
        api_key: str | None = None,
        max_steps: int = 10,
        tools: list[Callable] | None = None,
        **kwargs,
    ):
        """
        初始化ReAct代理。

        Args:
            name: 代理名称，用于日志和状态管理
            model: LLM模型名称 (例如: gpt-4, gpt-3.5-turbo)
            api_key: OpenAI API密钥，默认从环境变量获取
            max_steps: 最大ReAct循环步数
            tools: 可用工具函数列表
            **kwargs: 额外的配置参数
        """
        self.name = name
        self.model = model
        self.max_steps = max_steps

        # 创建配置和容器
        self._config = self._create_config(
            model=model, api_key=api_key, max_steps=max_steps, **kwargs
        )
        self._container = self._create_container()

        # 初始化组件
        self._initialize_components()

        # 注册工具
        if tools:
            for tool in tools:
                self.register_tool(tool)

        # 会话管理
        self._sessions: dict[str, SessionState] = {}
        self._session_lock = asyncio.Lock()

        logger.info(f"ReActAgent '{name}' initialized with model '{model}'")

    def _create_config(self, **kwargs) -> dict[str, Any]:
        """创建代理配置。"""
        config = {
            "llm_model": kwargs.get("model", "gpt-4"),
            "llm_temperature": kwargs.get("temperature", 0.1),
            "llm_max_tokens": kwargs.get("max_tokens", 2000),
            "react_max_steps": kwargs.get("max_steps", 10),
            "tools_concurrency_limit": kwargs.get("concurrency_limit", 5),
            "memory_max_messages": kwargs.get("memory_max_messages", 100),
            "memory_max_tokens": kwargs.get("memory_max_tokens", 8000),
        }

        # 添加API密钥如果提供
        if kwargs.get("api_key"):
            config["llm_api_key"] = kwargs["api_key"]

        return config

    def _create_container(self) -> AgentContainer:
        """创建配置好的依赖注入容器。"""
        return ContainerFactory.create_container(**self._config)

    def _initialize_components(self) -> None:
        """初始化所有核心组件。"""
        # 连接容器
        self._container.wire(modules=[__name__])

        # 获取核心服务 (目前使用占位符)
        self._llm_client = self._container.llm_client()

        # 创建记忆管理服务
        from agents.memory import MemoryIntegrationService, ThreadSafeMemoryManager

        thread_safe_manager = ThreadSafeMemoryManager(self._container)
        self._memory_service = MemoryIntegrationService(thread_safe_manager)

        # 获取工具注册表
        self._tool_registry = self._container.tool_registry()

        logger.debug(f"Components initialized for agent '{self.name}'")

    # ==================== Fluent Interface Methods ====================

    def with_model(self, model: str) -> "ReActAgent":
        """
        配置LLM模型。

        Args:
            model: 模型名称 (例如: gpt-4, gpt-3.5-turbo)

        Returns:
            新的ReActAgent实例
        """
        new_config = self._config.copy()
        new_config["llm_model"] = model

        return self._create_copy(**new_config)

    def with_tools(self, *tools: Callable) -> "ReActAgent":
        """
        添加工具函数。

        Args:
            *tools: 工具函数列表

        Returns:
            新的ReActAgent实例
        """
        new_agent = self._create_copy()
        for tool in tools:
            new_agent.register_tool(tool)
        return new_agent

    def with_max_steps(self, max_steps: int) -> "ReActAgent":
        """
        配置最大步数。

        Args:
            max_steps: 最大ReAct循环步数

        Returns:
            新的ReActAgent实例
        """
        new_config = self._config.copy()
        new_config["react_max_steps"] = max_steps

        return self._create_copy(**new_config)

    def with_temperature(self, temperature: float) -> "ReActAgent":
        """
        配置LLM温度参数。

        Args:
            temperature: 温度值 (0.0-2.0)

        Returns:
            新的ReActAgent实例
        """
        new_config = self._config.copy()
        new_config["llm_temperature"] = temperature

        return self._create_copy(**new_config)

    def _create_copy(self, **overrides) -> "ReActAgent":
        """创建带有配置覆盖的新实例。"""
        config = self._config.copy()
        config.update(overrides)

        # 提取构造函数参数，避免重复传递
        constructor_args = {
            "name": self.name,
            "model": config.get("llm_model", self.model),
            "max_steps": config.get("react_max_steps", self.max_steps),
        }

        # 提取其他配置参数，排除构造函数参数
        extra_config = {
            k: v for k, v in config.items() if k not in ["llm_model", "react_max_steps"]
        }

        return ReActAgent(**constructor_args, **extra_config)

    # ==================== Core Execution Methods ====================

    async def run(
        self,
        query: str,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """
        执行ReAct循环，处理用户查询。

        Args:
            query: 用户查询
            session_id: 会话ID，用于状态隔离，默认生成新会话
            context: 额外的上下文信息

        Returns:
            AgentResponse: 包含答案、推理过程、使用的工具等信息
        """
        start_time = time.time()

        # 生成或使用现有会话ID
        if session_id is None:
            session_id = str(uuid.uuid4())

        try:
            # 获取或创建会话状态
            session_state = await self._get_or_create_session(session_id)

            # 创建ReAct上下文
            react_context = self._create_react_context(session_state, context)

            # 设置初始查询
            react_context.current_query = query
            react_context.max_steps = self.max_steps

            # 执行ReAct循环
            agent_flow = create_react_agent(
                context=react_context, max_steps=self.max_steps, stop_on_error=True
            )
            final_result = agent_flow(react_context)

            # 构建响应
            execution_time = time.time() - start_time
            response = self._build_response(
                query=query,
                session_id=session_id,
                result=final_result,
                execution_time=execution_time,
                react_context=react_context,
            )

            # 更新会话
            await self._update_session(session_id, query, response.answer)

            logger.info(
                f"Query completed in {execution_time:.2f}s with {response.total_steps} steps"
            )
            return response

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed after {execution_time:.2f}s: {e}")

            return AgentResponse(
                query=query,
                answer=f"抱歉，执行过程中发生错误：{str(e)}",
                session_id=session_id,
                execution_time=execution_time,
                success=False,
                error_message=str(e),
            )

    async def run_stream(
        self,
        query: str,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[AgentStreamChunk, None]:
        """
        流式执行ReAct循环，实时返回中间结果。

        Args:
            query: 用户查询
            session_id: 会话ID
            context: 额外上下文

        Yields:
            AgentStreamChunk: 流式响应块
        """
        # TODO: 实现流式执行，这里先提供基础版本
        response = await self.run(query, session_id, context)

        # 将响应转换为流式块
        for i, step in enumerate(response.steps):
            if step.reasoning:
                yield AgentStreamChunk(
                    type="reasoning",
                    content=step.reasoning,
                    step_number=step.step_number,
                )

            if step.action:
                yield AgentStreamChunk(
                    type="action",
                    content=f"正在执行: {step.action.function_name}",
                    step_number=step.step_number,
                )

            if step.observation:
                yield AgentStreamChunk(
                    type="observation",
                    content=step.observation,
                    step_number=step.step_number,
                )

        yield AgentStreamChunk(
            type="final", content=response.answer, step_number=response.total_steps
        )

    def _create_react_context(
        self, session_state: SessionState, context: dict[str, Any] | None = None
    ) -> ReActContext:
        """创建ReAct执行上下文。"""
        react_context = ReActContext()

        # 设置会话状态和容器引用
        react_context.session_state = session_state
        react_context.agent_container = self._container

        # 设置额外上下文
        if context:
            for key, value in context.items():
                setattr(react_context, key, value)

        return react_context

    def _build_response(
        self,
        query: str,
        session_id: str,
        result: Any,
        execution_time: float,
        react_context: ReActContext,
    ) -> AgentResponse:
        """构建AgentResponse对象。"""
        # 从ReAct上下文提取步骤信息
        steps = []
        tools_called = []
        total_tokens = 0

        # TODO: 从react_context提取实际的执行步骤
        # 这里需要与ReAct引擎的具体实现集成

        # 获取最终答案
        answer = getattr(result, "final_answer", str(result))
        if not answer:
            answer = "任务已完成。"

        return AgentResponse(
            query=query,
            answer=answer,
            session_id=session_id,
            steps=steps,
            total_steps=len(steps),
            execution_time=execution_time,
            tokens_used=total_tokens,
            tools_called=tools_called,
            success=True,
        )

    # ==================== Session Management ====================

    async def create_session(self, session_id: str) -> Session:
        """
        创建新会话。

        Args:
            session_id: 会话标识符

        Returns:
            Session: 会话信息
        """
        async with self._session_lock:
            if session_id in self._sessions:
                raise SessionError(f"会话 {session_id} 已存在")

            session_state = SessionState(
                session_id=session_id, created_at=datetime.now(), metadata={}
            )

            self._sessions[session_id] = session_state

            return Session(
                session_id=session_id,
                created_at=session_state.created_at,
                updated_at=session_state.created_at,
                status="active",
            )

    async def get_session(self, session_id: str) -> Session | None:
        """
        获取现有会话。

        Args:
            session_id: 会话标识符

        Returns:
            Session或None: 会话信息
        """
        async with self._session_lock:
            session_state = self._sessions.get(session_id)
            if not session_state:
                return None

            return Session(
                session_id=session_id,
                created_at=session_state.created_at,
                updated_at=session_state.last_accessed,
                message_count=len(session_state.conversation_thread.messages)
                if session_state.conversation_thread
                else 0,
                status="active",
            )

    async def delete_session(self, session_id: str) -> bool:
        """
        删除会话及其状态。

        Args:
            session_id: 会话标识符

        Returns:
            bool: 是否删除成功
        """
        async with self._session_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Session {session_id} deleted")
                return True
            return False

    async def _get_or_create_session(self, session_id: str) -> SessionState:
        """获取或创建会话状态。"""
        async with self._session_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(
                    session_id=session_id, created_at=datetime.now(), metadata={}
                )
            return self._sessions[session_id]

    async def _update_session(self, session_id: str, query: str, answer: str) -> None:
        """更新会话状态。"""
        async with self._session_lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()  # 更新last_accessed时间
                # TODO: 添加对话历史到会话
                logger.debug(f"Session {session_id} updated")

    # ==================== Tool Management ====================

    def register_tool(self, tool: Callable) -> "ReActAgent":
        """
        注册新工具。

        Args:
            tool: 工具函数（应该用@tool装饰器装饰）

        Returns:
            self: 支持链式调用

        Note:
            如果工具函数已经用@tool装饰器装饰，则会自动注册到全局注册表中。
            此方法主要用于确认工具已被正确注册和链式调用支持。
        """
        # 检查工具是否已经被@tool装饰器处理
        if hasattr(tool, "__tool_metadata__"):
            logger.info(
                f"Tool '{tool.__name__}' already registered via @tool decorator"
            )
        else:
            logger.warning(
                f"Tool '{tool.__name__}' not decorated with @tool, registration may fail"
            )
            # 尝试从ToolRegistry注册（如果需要的话）
            # 但通常工具应该通过@tool装饰器自动注册

        return self

    def unregister_tool(self, tool_name: str) -> "ReActAgent":
        """
        注销工具。

        Args:
            tool_name: 工具名称

        Returns:
            self: 支持链式调用
        """
        if ToolRegistry.unregister(tool_name):
            logger.info(f"Tool '{tool_name}' unregistered")
        else:
            logger.warning(f"Tool '{tool_name}' not found")
        return self

    def list_tools(self) -> list[ToolInfo]:
        """
        列出所有可用工具。

        Returns:
            List[ToolInfo]: 工具信息列表
        """
        tools = ToolRegistry.get_available_tools()
        return [
            ToolInfo(
                name=metadata.name,
                description=metadata.description,
                parameters={
                    param.name: {
                        "type": param.type_hint,
                        "required": param.required,
                        "description": param.description or "",
                        "default": param.default_value,
                    }
                    for param in metadata.parameters
                },
                category=getattr(metadata, "category", None),
                enabled=True,
            )
            for metadata in tools
        ]

    # ==================== Utility Methods ====================

    def __repr__(self) -> str:
        """字符串表示。"""
        return (
            f"ReActAgent(name='{self.name}', model='{self.model}', "
            f"max_steps={self.max_steps})"
        )

    def get_status(self) -> dict[str, Any]:
        """获取代理状态信息。"""
        return {
            "name": self.name,
            "model": self.model,
            "max_steps": self.max_steps,
            "sessions_count": len(self._sessions),
            "tools_count": len(ToolRegistry.get_available_tools()),
            "memory_service_status": "active" if self._memory_service else "inactive",
            "llm_client_status": "active" if self._llm_client else "inactive",
        }


# ==================== Builder Pattern ====================


class ReActAgentBuilder:
    """
    ReActAgent构建器，提供更灵活的配置选项。

    使用构建器模式可以更方便地配置复杂的代理实例，
    特别适合需要多种配置组合的场景。

    Example:
        agent = (ReActAgentBuilder()
                .model("gpt-4")
                .tools(weather_tool, calculator)
                .memory(max_messages=500, max_tokens=6000)
                .max_steps(8)
                .temperature(0.2)
                .build())
    """

    def __init__(self):
        """初始化构建器。"""
        self._config = {}

    def model(self, model: str) -> "ReActAgentBuilder":
        """设置LLM模型。"""
        self._config["model"] = model
        return self

    def api_key(self, api_key: str) -> "ReActAgentBuilder":
        """设置API密钥。"""
        self._config["api_key"] = api_key
        return self

    def tools(self, *tools: Callable) -> "ReActAgentBuilder":
        """添加工具。"""
        self._config["tools"] = list(tools)
        return self

    def max_steps(self, max_steps: int) -> "ReActAgentBuilder":
        """设置最大步数。"""
        self._config["max_steps"] = max_steps
        return self

    def temperature(self, temperature: float) -> "ReActAgentBuilder":
        """设置温度参数。"""
        self._config["temperature"] = temperature
        return self

    def memory(
        self, max_messages: int = 100, max_tokens: int = 8000
    ) -> "ReActAgentBuilder":
        """配置记忆管理。"""
        self._config["memory_max_messages"] = max_messages
        self._config["memory_max_tokens"] = max_tokens
        return self

    def name(self, name: str) -> "ReActAgentBuilder":
        """设置代理名称。"""
        self._config["name"] = name
        return self

    def build(self) -> ReActAgent:
        """构建ReActAgent实例。"""
        return ReActAgent(**self._config)


# ==================== Factory Functions ====================


def create_agent(
    model: str = "gpt-4", tools: list[Callable] | None = None, **kwargs
) -> ReActAgent:
    """
    快速创建ReActAgent实例。

    Args:
        model: LLM模型名称
        tools: 工具函数列表
        **kwargs: 额外配置参数

    Returns:
        ReActAgent: 配置好的代理实例

    Example:
        agent = create_agent(
            model="gpt-4",
            tools=[weather_tool, calculator],
            max_steps=5,
            temperature=0.1
        )
    """
    return ReActAgent(model=model, tools=tools or [], **kwargs)


def create_agent_builder() -> ReActAgentBuilder:
    """
    创建ReActAgent构建器。

    Returns:
        ReActAgentBuilder: 构建器实例

    Example:
        agent = (create_agent_builder()
                .model("gpt-4")
                .tools(tool1, tool2)
                .build())
    """
    return ReActAgentBuilder()
