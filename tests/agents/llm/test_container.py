"""测试智能代理依赖注入容器。"""

import os
from unittest.mock import patch

import pytest

from agents.llm.client import OpenAIClient
from agents.llm.config import OpenAIConfig
from agents.llm.container import (
    AgentContainer,
    create_agent_container,
    create_simple_agent_container,
)
from agents.llm.prompts import ReActPromptTemplate


class TestAgentContainer:
    """测试智能代理容器。"""

    def test_container_initialization(self):
        """测试容器初始化。"""
        container = AgentContainer()

        # 检查容器是否正确继承了BaseFlowContext
        assert hasattr(container, "state")
        assert hasattr(container, "context")
        assert hasattr(container, "shared_data")

        # 检查新增的提供者
        assert hasattr(container, "openai_config")
        assert hasattr(container, "llm_client")
        assert hasattr(container, "react_template")

    def test_openai_config_provider(self):
        """测试OpenAI配置提供者。"""
        container = AgentContainer()

        # 设置测试配置
        test_config = {
            "api_key": "test-key",
            "model": "gpt-4",
            "timeout": 45,
            "max_retries": 5,
        }
        container.config.openai.override(test_config)

        # 获取配置实例
        config = container.openai_config()

        assert isinstance(config, OpenAIConfig)
        assert config.api_key == "test-key"
        assert config.model == "gpt-4"
        assert config.timeout == 45
        assert config.max_retries == 5

    def test_llm_client_provider(self):
        """测试LLM客户端提供者。"""
        container = AgentContainer()

        # 设置测试配置
        test_config = {
            "api_key": "test-key",
            "model": "gpt-3.5-turbo",
        }
        container.config.openai.override(test_config)

        # 获取客户端实例
        client = container.llm_client()

        assert isinstance(client, OpenAIClient)
        assert client.config.api_key == "test-key"
        assert client.config.model == "gpt-3.5-turbo"

    def test_react_template_providers(self):
        """测试ReAct模板提供者。"""
        container = AgentContainer()

        # 测试中文模板
        zh_template = container.react_template_zh()
        assert isinstance(zh_template, ReActPromptTemplate)
        assert zh_template.language == "zh"

        # 测试英文模板
        en_template = container.react_template_en()
        assert isinstance(en_template, ReActPromptTemplate)
        assert en_template.language == "en"

        # 测试默认模板
        default_template = container.react_template()
        assert isinstance(default_template, ReActPromptTemplate)
        assert default_template.language == "zh"  # 默认中文

    def test_singleton_behavior(self):
        """测试单例行为。"""
        container = AgentContainer()

        # 设置配置
        test_config = {"api_key": "test-key"}
        container.config.openai.override(test_config)

        # 多次获取应该返回同一个实例
        config1 = container.openai_config()
        config2 = container.openai_config()
        assert config1 is config2

        client1 = container.llm_client()
        client2 = container.llm_client()
        assert client1 is client2

        template1 = container.react_template()
        template2 = container.react_template()
        assert template1 is template2


class TestCreateAgentContainer:
    """测试创建代理容器函数。"""

    def test_create_with_parameters(self):
        """测试使用参数创建容器。"""
        container = create_agent_container(
            openai_api_key="custom-key",
            openai_model="gpt-4",
            openai_base_url="https://custom.api.com",
            temperature=0.5,
            max_tokens=2000,
        )

        assert isinstance(container, AgentContainer)

        # 获取配置并验证
        config = container.openai_config()
        assert config.api_key == "custom-key"
        assert config.model == "gpt-4"
        assert config.base_url == "https://custom.api.com"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000

    def test_create_with_minimal_parameters(self):
        """测试使用最少参数创建容器。"""
        container = create_agent_container(openai_api_key="test-key")

        config = container.openai_config()
        assert config.api_key == "test-key"
        assert config.model == "gpt-3.5-turbo"  # 默认值
        assert config.base_url is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})
    def test_create_from_env_fallback(self):
        """测试从环境变量回退。"""
        container = create_agent_container()

        config = container.openai_config()
        assert config.api_key == "env-key"

    @patch.dict(os.environ, {"OPENAI_MODEL": "env-model"})
    def test_create_parameter_override_env(self):
        """测试参数覆盖环境变量。"""
        container = create_agent_container(
            openai_api_key="param-key", openai_model="param-model"
        )

        config = container.openai_config()
        assert config.api_key == "param-key"
        assert config.model == "param-model"

    def test_create_with_base_url(self):
        """测试创建带有base_url的容器。"""
        container = create_agent_container(
            openai_api_key="test-key", openai_base_url="https://custom.example.com"
        )

        config = container.openai_config()
        assert config.base_url == "https://custom.example.com"

    def test_create_container_integration(self):
        """测试容器集成功能。"""
        container = create_agent_container(openai_api_key="test-key")

        # 测试所有组件是否正常工作
        config = container.openai_config()
        client = container.llm_client()
        template = container.react_template()

        assert isinstance(config, OpenAIConfig)
        assert isinstance(client, OpenAIClient)
        assert isinstance(template, ReActPromptTemplate)

        # 验证客户端使用了正确的配置
        assert client.config is config


class TestCreateSimpleAgentContainer:
    """测试创建简单代理容器函数。"""

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "env-key",
            "OPENAI_MODEL": "gpt-4",
            "OPENAI_BASE_URL": "https://env.api.com",
            "OPENAI_TIMEOUT": "60",
            "OPENAI_MAX_RETRIES": "5",
            "OPENAI_TEMPERATURE": "0.8",
            "OPENAI_MAX_TOKENS": "2000",
        },
    )
    def test_create_simple_from_env(self):
        """测试从环境变量创建简单容器。"""
        container = create_simple_agent_container()

        config = container.openai_config()
        assert config.api_key == "env-key"
        assert config.model == "gpt-4"
        assert config.base_url == "https://env.api.com"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.temperature == 0.8
        assert config.max_tokens == 2000

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True)
    def test_create_simple_with_defaults(self):
        """测试使用默认值创建简单容器。"""
        container = create_simple_agent_container()

        config = container.openai_config()
        assert config.api_key == "env-key"
        assert config.model == "gpt-3.5-turbo"
        assert config.base_url is None
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.temperature == 0.7
        assert config.max_tokens is None

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "env-key",
            "OPENAI_MAX_TOKENS": "1500",
        },
    )
    def test_create_simple_with_optional_int(self):
        """测试处理可选整数类型环境变量。"""
        container = create_simple_agent_container()

        config = container.openai_config()
        assert config.max_tokens == 1500

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})
    def test_create_simple_without_optional_int(self):
        """测试不设置可选整数类型环境变量。"""
        # 确保OPENAI_MAX_TOKENS环境变量不存在
        if "OPENAI_MAX_TOKENS" in os.environ:
            del os.environ["OPENAI_MAX_TOKENS"]

        container = create_simple_agent_container()

        config = container.openai_config()
        assert config.max_tokens is None

    def test_simple_container_components(self):
        """测试简单容器的所有组件。"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            container = create_simple_agent_container()

        # 验证所有核心组件都能正常获取
        config = container.openai_config()
        client = container.llm_client()
        zh_template = container.react_template_zh()
        en_template = container.react_template_en()
        default_template = container.react_template()

        assert isinstance(config, OpenAIConfig)
        assert isinstance(client, OpenAIClient)
        assert isinstance(zh_template, ReActPromptTemplate)
        assert isinstance(en_template, ReActPromptTemplate)
        assert isinstance(default_template, ReActPromptTemplate)

        # 验证模板语言设置
        assert zh_template.language == "zh"
        assert en_template.language == "en"
        assert default_template.language == "zh"

    @patch.dict(os.environ, {}, clear=True)
    def test_create_simple_missing_api_key(self):
        """测试缺少API密钥时的行为。"""
        # 应该能创建容器，但在获取配置时会失败
        container = create_simple_agent_container()

        with pytest.raises(ValueError, match="OpenAI API key is required"):
            container.openai_config()


class TestContainerIntegration:
    """测试容器集成功能。"""

    def test_full_workflow_integration(self):
        """测试完整工作流程集成。"""
        # 创建容器
        container = create_agent_container(
            openai_api_key="test-key",
            openai_model="gpt-3.5-turbo",
        )

        # 获取所有核心组件
        config = container.openai_config()
        client = container.llm_client()
        template = container.react_template()

        # 验证组件间的关联
        assert client.config is config
        assert isinstance(template, ReActPromptTemplate)

        # 测试模板功能
        prompt = template.format_reasoning_prompt(
            query="测试查询", history=[], available_tools=["search", "calculator"]
        )
        assert "测试查询" in prompt
        assert "search" in prompt

        # 验证配置正确传递
        assert config.api_key == "test-key"
        assert config.model == "gpt-3.5-turbo"

    def test_container_inheritance(self):
        """测试容器继承。"""
        container = AgentContainer()

        # 验证继承自BaseFlowContext
        from aetherflow import BaseFlowContext

        assert isinstance(container, BaseFlowContext)

        # 验证基础容器功能
        assert hasattr(container, "state")
        assert hasattr(container, "context")
        assert hasattr(container, "shared_data")

    def test_container_wiring(self):
        """测试容器连接功能。"""
        container = create_agent_container(openai_api_key="test-key")

        # 测试是否能正确连接和获取实例
        try:
            config = container.openai_config()
            client = container.llm_client()
            template = container.react_template()

            # 所有实例都应该成功创建
            assert config is not None
            assert client is not None
            assert template is not None

        except Exception as e:
            pytest.fail(f"容器连接失败: {e}")
