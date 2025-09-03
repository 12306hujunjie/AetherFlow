"""OpenAI配置类和相关配置管理。"""

import os
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator


@dataclass
class OpenAIConfig:
    """OpenAI API配置类，支持环境变量和灵活配置。

    该类负责管理OpenAI API的各项配置，包括API密钥、模型选择、
    连接参数等。支持从环境变量自动加载配置。
    """

    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    """OpenAI API密钥，默认从环境变量OPENAI_API_KEY获取"""

    model: str = "gpt-3.5-turbo"
    """默认使用的模型"""

    base_url: str | None = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
    """API基础URL，支持自定义OpenAI兼容服务"""

    timeout: int = 30
    """请求超时时间（秒）"""

    max_retries: int = 3
    """最大重试次数"""

    temperature: float = 0.7
    """默认温度参数，控制响应的随机性"""

    max_tokens: int | None = None
    """最大token数量限制"""

    def __post_init__(self) -> None:
        """初始化后验证配置。"""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or provide api_key parameter."
            )

        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")

        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")

        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative")


class ChatMessage(BaseModel):
    """聊天消息模型。"""

    role: str = Field(..., description="消息角色：system, user, assistant")
    content: str = Field(..., description="消息内容")

    @validator("role")
    def validate_role(cls, v: str) -> str:
        """验证角色有效性。"""
        valid_roles = {"system", "user", "assistant", "developer"}
        if v not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}")
        return v


class CompletionRequest(BaseModel):
    """完成请求模型。"""

    prompt: str = Field(..., description="提示文本")
    temperature: float = Field(0.7, ge=0, le=2, description="温度参数")
    max_tokens: int | None = Field(None, gt=0, description="最大token数")
    model: str = Field("gpt-3.5-turbo", description="使用的模型")


class ChatCompletionRequest(BaseModel):
    """聊天完成请求模型。"""

    messages: list[ChatMessage] = Field(..., description="聊天消息列表")
    temperature: float = Field(0.7, ge=0, le=2, description="温度参数")
    max_tokens: int | None = Field(None, gt=0, description="最大token数")
    model: str = Field("gpt-3.5-turbo", description="使用的模型")
    stream: bool = Field(False, description="是否启用流式响应")


class CompletionResponse(BaseModel):
    """完成响应模型。"""

    content: str = Field(..., description="生成的内容")
    model: str = Field(..., description="使用的模型")
    usage_prompt_tokens: int = Field(..., description="提示token数")
    usage_completion_tokens: int = Field(..., description="完成token数")
    usage_total_tokens: int = Field(..., description="总token数")
    finish_reason: str = Field(..., description="完成原因")


def create_config_from_env() -> OpenAIConfig:
    """从环境变量创建配置。

    Returns:
        OpenAIConfig: 从环境变量创建的配置实例

    Raises:
        ValueError: 如果必需的环境变量未设置
    """
    return OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        timeout=int(os.getenv("OPENAI_TIMEOUT", "30")),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS"))
        if os.getenv("OPENAI_MAX_TOKENS")
        else None,
    )
