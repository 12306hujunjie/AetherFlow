"""测试OpenAI配置类。"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agents.llm.config import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    OpenAIConfig,
    create_config_from_env,
)


class TestOpenAIConfig:
    """测试OpenAIConfig配置类。"""

    def test_default_config_creation(self):
        """测试默认配置创建。"""
        config = OpenAIConfig(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.model == "gpt-3.5-turbo"
        assert config.base_url is None
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.temperature == 0.7
        assert config.max_tokens is None

    def test_custom_config_creation(self):
        """测试自定义配置创建。"""
        config = OpenAIConfig(
            api_key="custom-key",
            model="gpt-4",
            base_url="https://api.custom.com",
            timeout=60,
            max_retries=5,
            temperature=0.5,
            max_tokens=2000,
        )

        assert config.api_key == "custom-key"
        assert config.model == "gpt-4"
        assert config.base_url == "https://api.custom.com"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.temperature == 0.5
        assert config.max_tokens == 2000

    def test_api_key_validation(self):
        """测试API密钥验证。"""
        # 空API密钥应该抛出异常
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIConfig(api_key="")

    def test_temperature_validation(self):
        """测试温度参数验证。"""
        # 温度过低
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            OpenAIConfig(api_key="test", temperature=-0.1)

        # 温度过高
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            OpenAIConfig(api_key="test", temperature=2.1)

        # 边界值应该正常
        config1 = OpenAIConfig(api_key="test", temperature=0.0)
        config2 = OpenAIConfig(api_key="test", temperature=2.0)
        assert config1.temperature == 0.0
        assert config2.temperature == 2.0

    def test_timeout_validation(self):
        """测试超时参数验证。"""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            OpenAIConfig(api_key="test", timeout=0)

        with pytest.raises(ValueError, match="Timeout must be positive"):
            OpenAIConfig(api_key="test", timeout=-1)

    def test_max_retries_validation(self):
        """测试最大重试次数验证。"""
        with pytest.raises(ValueError, match="Max retries must be non-negative"):
            OpenAIConfig(api_key="test", max_retries=-1)

        # 0次重试应该正常
        config = OpenAIConfig(api_key="test", max_retries=0)
        assert config.max_retries == 0

    @patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "env-key",
            "OPENAI_MODEL": "gpt-4",
            "OPENAI_BASE_URL": "https://env.api.com",
            "OPENAI_TIMEOUT": "45",
            "OPENAI_MAX_RETRIES": "2",
            "OPENAI_TEMPERATURE": "0.8",
            "OPENAI_MAX_TOKENS": "1500",
        },
    )
    def test_config_from_env(self):
        """测试从环境变量创建配置。"""
        config = create_config_from_env()

        assert config.api_key == "env-key"
        assert config.model == "gpt-4"
        assert config.base_url == "https://env.api.com"
        assert config.timeout == 45
        assert config.max_retries == 2
        assert config.temperature == 0.8
        assert config.max_tokens == 1500

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True)
    def test_config_from_env_defaults(self):
        """测试从环境变量创建配置时使用默认值。"""
        config = create_config_from_env()

        assert config.api_key == "env-key"
        assert config.model == "gpt-3.5-turbo"  # 默认值
        assert config.base_url is None
        assert config.timeout == 30  # 默认值
        assert config.max_retries == 3  # 默认值
        assert config.temperature == 0.7  # 默认值
        assert config.max_tokens is None


class TestChatMessage:
    """测试ChatMessage模型。"""

    def test_valid_message_creation(self):
        """测试有效消息创建。"""
        message = ChatMessage(role="user", content="Hello, world!")

        assert message.role == "user"
        assert message.content == "Hello, world!"

    def test_valid_roles(self):
        """测试所有有效角色。"""
        valid_roles = ["system", "user", "assistant", "developer"]

        for role in valid_roles:
            message = ChatMessage(role=role, content="test")
            assert message.role == role

    def test_invalid_role_validation(self):
        """测试无效角色验证。"""
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="test")

    def test_empty_content_allowed(self):
        """测试允许空内容。"""
        message = ChatMessage(role="user", content="")
        assert message.content == ""


class TestCompletionRequest:
    """测试CompletionRequest模型。"""

    def test_valid_request_creation(self):
        """测试有效请求创建。"""
        request = CompletionRequest(
            prompt="What is AI?",
            temperature=0.5,
            max_tokens=100,
            model="gpt-4",
        )

        assert request.prompt == "What is AI?"
        assert request.temperature == 0.5
        assert request.max_tokens == 100
        assert request.model == "gpt-4"

    def test_default_values(self):
        """测试默认值。"""
        request = CompletionRequest(prompt="test")

        assert request.temperature == 0.7
        assert request.max_tokens is None
        assert request.model == "gpt-3.5-turbo"

    def test_temperature_validation(self):
        """测试温度验证。"""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", temperature=-0.1)

        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", temperature=2.1)

    def test_max_tokens_validation(self):
        """测试最大token验证。"""
        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", max_tokens=0)

        with pytest.raises(ValidationError):
            CompletionRequest(prompt="test", max_tokens=-1)


class TestChatCompletionRequest:
    """测试ChatCompletionRequest模型。"""

    def test_valid_request_creation(self):
        """测试有效请求创建。"""
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello!"),
        ]

        request = ChatCompletionRequest(
            messages=messages,
            temperature=0.8,
            max_tokens=150,
            model="gpt-4",
            stream=True,
        )

        assert len(request.messages) == 2
        assert request.messages[0].role == "system"
        assert request.messages[1].role == "user"
        assert request.temperature == 0.8
        assert request.max_tokens == 150
        assert request.model == "gpt-4"
        assert request.stream is True

    def test_default_values(self):
        """测试默认值。"""
        messages = [ChatMessage(role="user", content="test")]
        request = ChatCompletionRequest(messages=messages)

        assert request.temperature == 0.7
        assert request.max_tokens is None
        assert request.model == "gpt-3.5-turbo"
        assert request.stream is False


class TestCompletionResponse:
    """测试CompletionResponse模型。"""

    def test_response_creation(self):
        """测试响应创建。"""
        response = CompletionResponse(
            content="AI stands for Artificial Intelligence.",
            model="gpt-4",
            usage_prompt_tokens=10,
            usage_completion_tokens=8,
            usage_total_tokens=18,
            finish_reason="stop",
        )

        assert response.content == "AI stands for Artificial Intelligence."
        assert response.model == "gpt-4"
        assert response.usage_prompt_tokens == 10
        assert response.usage_completion_tokens == 8
        assert response.usage_total_tokens == 18
        assert response.finish_reason == "stop"
