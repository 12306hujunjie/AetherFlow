"""智能代理相关异常类定义。"""

from typing import Any


class AgentException(Exception):
    """智能代理基础异常类。"""

    def __init__(
        self, message: str, agent_type: str | None = None, **kwargs: Any
    ) -> None:
        self.agent_type = agent_type
        self.context = kwargs
        super().__init__(message)


class OpenAIException(AgentException):
    """OpenAI API相关异常基类。"""

    retryable = False  # 默认不重试

    def __init__(
        self, message: str, error_code: str | None = None, **kwargs: Any
    ) -> None:
        self.error_code = error_code
        super().__init__(message, agent_type="openai", **kwargs)


class OpenAIAuthenticationException(OpenAIException):
    """OpenAI认证异常。"""

    retryable = False  # 认证异常不应重试

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_code="authentication_failed", **kwargs)


class OpenAIRateLimitException(OpenAIException):
    """OpenAI速率限制异常。"""

    retryable = True  # 速率限制可以重试

    def __init__(
        self, message: str, retry_after: int | None = None, **kwargs: Any
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, error_code="rate_limit_exceeded", **kwargs)


class OpenAITimeoutException(OpenAIException):
    """OpenAI请求超时异常。"""

    retryable = True  # 超时可以重试

    def __init__(
        self, message: str, timeout_duration: float | None = None, **kwargs: Any
    ) -> None:
        self.timeout_duration = timeout_duration
        super().__init__(message, error_code="request_timeout", **kwargs)


class OpenAIModelException(OpenAIException):
    """OpenAI模型相关异常。"""

    retryable = False  # 模型错误通常不应重试

    def __init__(
        self, message: str, model_name: str | None = None, **kwargs: Any
    ) -> None:
        self.model_name = model_name
        super().__init__(message, error_code="model_error", **kwargs)


class ReActException(AgentException):
    """ReAct推理异常。"""

    def __init__(
        self, message: str, reasoning_step: str | None = None, **kwargs: Any
    ) -> None:
        self.reasoning_step = reasoning_step
        super().__init__(message, agent_type="react", **kwargs)


class PromptTemplateException(AgentException):
    """提示模板异常。"""

    def __init__(
        self, message: str, template_name: str | None = None, **kwargs: Any
    ) -> None:
        self.template_name = template_name
        super().__init__(message, agent_type="prompt_template", **kwargs)


class ToolExecutionException(AgentException):
    """工具执行异常。"""

    def __init__(
        self, message: str, tool_name: str | None = None, **kwargs: Any
    ) -> None:
        self.tool_name = tool_name
        super().__init__(message, agent_type="tool", **kwargs)


# ReActAgent specific exceptions
class AgentError(AgentException):
    """ReActAgent基础异常类。"""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, agent_type="react_agent", **kwargs)


class AgentConfigurationError(AgentError):
    """ReActAgent配置错误。"""

    pass


class AgentExecutionError(AgentError):
    """ReActAgent执行错误。"""

    pass


class AgentToolError(AgentError):
    """ReActAgent工具相关错误。"""

    def __init__(
        self, message: str, tool_name: str | None = None, **kwargs: Any
    ) -> None:
        self.tool_name = tool_name
        super().__init__(message, **kwargs)


class SessionError(AgentError):
    """会话相关错误。"""

    def __init__(
        self, message: str, session_id: str | None = None, **kwargs: Any
    ) -> None:
        self.session_id = session_id
        super().__init__(message, **kwargs)
