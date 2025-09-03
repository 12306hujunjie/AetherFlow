"""LLM（大语言模型）集成模块。

提供与各种LLM提供商的集成，包括：
- OpenAI GPT系列模型
- 通用的LLM接口抽象
- 针对智能代理优化的提示模板
- 智能重试和错误处理机制

主要组件：
- OpenAIClient: OpenAI API客户端，支持同步/异步调用
- OpenAIConfig: OpenAI配置管理，支持环境变量
- ReActPromptTemplate: ReAct推理-行动提示模板系统
- AgentContainer: 依赖注入容器，集成AetherFlow DI系统
- RetryHandler: 智能重试处理器，支持指数退避
- CircuitBreaker: 断路器模式，防止级联故障
"""

from .client import OpenAIClient
from .config import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    OpenAIConfig,
    create_config_from_env,
)
from .container import (
    AgentContainer,
    create_agent_container,
    create_simple_agent_container,
)
from .prompts import (
    ReActEntry,
    ReActPromptTemplate,
    ReActResponse,
    ReActStep,
    get_react_template,
)
from .retry import (
    AGGRESSIVE_RETRY_CONFIG,
    CONSERVATIVE_RETRY_CONFIG,
    DEFAULT_RETRY_CONFIG,
    CircuitBreaker,
    RetryConfig,
    RetryHandler,
    retry_on_failure,
)

__all__ = [
    # 核心客户端
    "OpenAIClient",
    # 配置管理
    "OpenAIConfig",
    "ChatMessage",
    "CompletionRequest",
    "ChatCompletionRequest",
    "CompletionResponse",
    "create_config_from_env",
    # ReAct提示模板
    "ReActPromptTemplate",
    "ReActResponse",
    "ReActEntry",
    "ReActStep",
    "get_react_template",
    # 依赖注入
    "AgentContainer",
    "create_agent_container",
    "create_simple_agent_container",
    # 重试机制
    "RetryConfig",
    "RetryHandler",
    "CircuitBreaker",
    "retry_on_failure",
    "DEFAULT_RETRY_CONFIG",
    "AGGRESSIVE_RETRY_CONFIG",
    "CONSERVATIVE_RETRY_CONFIG",
]
