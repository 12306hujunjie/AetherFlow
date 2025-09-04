"""智能代理依赖注入容器，扩展AetherFlow的DI系统。"""

import os

from dependency_injector import providers

from aetherflow import BaseFlowContext

from .client import OpenAIClient
from .config import OpenAIConfig, create_config_from_env
from .prompts import ReActPromptTemplate, get_react_template


class AgentContainer(BaseFlowContext):
    """扩展容器，添加LLM客户端和相关服务。

    该容器扩展了AetherFlow的BaseFlowContext，添加了：
    - OpenAI客户端配置和实例
    - ReAct提示模板管理
    - 智能代理相关服务
    """

    # 配置提供者
    config = providers.Configuration()

    # OpenAI相关配置 - 使用Singleton来保持单例行为
    openai_config = providers.Singleton(
        OpenAIConfig,
        api_key=config.openai.api_key,
        model=config.openai.model,
        base_url=config.openai.base_url,
        timeout=config.openai.timeout,
        max_retries=config.openai.max_retries,
        temperature=config.openai.temperature,
        max_tokens=config.openai.max_tokens,
    )

    # 从环境变量创建配置的备用方案
    openai_config_from_env = providers.Singleton(create_config_from_env)

    # OpenAI客户端
    llm_client = providers.Singleton(
        OpenAIClient,
        config=openai_config,
    )

    # ReAct提示模板
    react_template_zh = providers.Singleton(
        ReActPromptTemplate,
        language="zh",
    )

    react_template_en = providers.Singleton(
        ReActPromptTemplate,
        language="en",
    )

    # 默认模板（中文）
    react_template = providers.Singleton(lambda: get_react_template("zh"))


def create_agent_container(
    openai_api_key: str | None = None,
    openai_model: str | None = None,
    openai_base_url: str | None = None,
    **kwargs,
) -> AgentContainer:
    """创建预配置的智能代理容器。

    Args:
        openai_api_key: OpenAI API密钥，默认从环境变量获取
        openai_model: 使用的模型，默认gpt-3.5-turbo
        openai_base_url: API基础URL，可选
        **kwargs: 其他OpenAI配置参数

    Returns:
        配置好的AgentContainer实例

    Example:
        >>> container = create_agent_container(
        ...     openai_api_key="sk-xxx",
        ...     openai_model="gpt-4",
        ... )
        >>> llm_client = container.llm_client()
        >>> template = container.react_template()
    """
    container = AgentContainer()

    # 配置OpenAI相关设置
    openai_config = {
        "api_key": openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        "model": openai_model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    }

    if openai_base_url:
        openai_config["base_url"] = openai_base_url

    # 添加其他配置参数
    openai_config.update(kwargs)

    # 设置配置
    container.config.openai.override(openai_config)

    return container


def create_simple_agent_container() -> AgentContainer:
    """创建简单的智能代理容器，从环境变量自动加载配置。

    Returns:
        从环境变量配置的AgentContainer实例

    Example:
        >>> # 设置环境变量
        >>> os.environ["OPENAI_API_KEY"] = "sk-xxx"
        >>> container = create_simple_agent_container()
        >>> llm_client = container.llm_client()
    """
    container = AgentContainer()

    # 使用环境变量配置
    openai_config = {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "timeout": int(os.getenv("OPENAI_TIMEOUT", "30")),
        "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "3")),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
    }

    max_tokens_env = os.getenv("OPENAI_MAX_TOKENS")
    if max_tokens_env:
        openai_config["max_tokens"] = int(max_tokens_env)

    container.config.openai.override(openai_config)

    return container
