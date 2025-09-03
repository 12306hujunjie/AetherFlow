"""测试OpenAI客户端，使用Mock避免真实API调用。"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from agents.exceptions import (
    OpenAIAuthenticationException,
    OpenAIException,
    OpenAIRateLimitException,
    OpenAITimeoutException,
)
from agents.llm.client import OpenAIClient
from agents.llm.config import ChatMessage, OpenAIConfig


class TestOpenAIClient:
    """测试OpenAI客户端类。"""

    @pytest.fixture
    def config(self) -> OpenAIConfig:
        """测试配置fixture。"""
        return OpenAIConfig(
            api_key="test-key",
            model="gpt-3.5-turbo",
            timeout=30,
            max_retries=3,
        )

    @pytest.fixture
    def client(self, config: OpenAIConfig) -> OpenAIClient:
        """测试客户端fixture。"""
        return OpenAIClient(config)

    def test_client_initialization(self, config: OpenAIConfig):
        """测试客户端初始化。"""
        client = OpenAIClient(config)

        assert client.config == config
        assert client._sync_client is None
        assert client._async_client is None
        assert client._circuit_breaker is not None

    @patch("agents.llm.client.OpenAI")
    def test_sync_client_property(self, mock_openai, client: OpenAIClient):
        """测试同步客户端属性。"""
        mock_instance = Mock()
        mock_openai.return_value = mock_instance

        # 第一次访问应该创建客户端
        sync_client = client.sync_client
        assert sync_client is mock_instance
        mock_openai.assert_called_once_with(
            api_key="test-key",
            timeout=30,
            max_retries=3,
        )

        # 第二次访问应该返回同一个实例
        sync_client2 = client.sync_client
        assert sync_client2 is mock_instance
        assert mock_openai.call_count == 1  # 只调用一次

    @patch("agents.llm.client.AsyncOpenAI")
    def test_async_client_property(self, mock_async_openai, client: OpenAIClient):
        """测试异步客户端属性。"""
        mock_instance = AsyncMock()
        mock_async_openai.return_value = mock_instance

        # 第一次访问应该创建客户端
        async_client = client.async_client
        assert async_client is mock_instance
        mock_async_openai.assert_called_once_with(
            api_key="test-key",
            timeout=30,
            max_retries=3,
        )

        # 第二次访问应该返回同一个实例
        async_client2 = client.async_client
        assert async_client2 is mock_instance
        assert mock_async_openai.call_count == 1  # 只调用一次

    @patch("agents.llm.client.OpenAI")
    def test_complete_success(self, mock_openai, client: OpenAIClient):
        """测试同步完成成功。"""
        # 模拟OpenAI响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI is artificial intelligence."
        mock_response.usage.total_tokens = 25

        mock_instance = Mock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_instance

        # 调用完成方法
        result = client.complete("What is AI?")

        assert result == "AI is artificial intelligence."
        mock_instance.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What is AI?"}],
            temperature=0.7,
            max_tokens=None,
        )

    @patch("agents.llm.client.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_complete_async_success(
        self, mock_async_openai, client: OpenAIClient
    ):
        """测试异步完成成功。"""
        # 模拟异步OpenAI响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI is artificial intelligence."
        mock_response.usage.total_tokens = 25

        mock_instance = AsyncMock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_async_openai.return_value = mock_instance

        # 调用异步完成方法
        result = await client.complete_async("What is AI?")

        assert result == "AI is artificial intelligence."
        mock_instance.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What is AI?"}],
            temperature=0.7,
            max_tokens=None,
        )

    @patch("agents.llm.client.OpenAI")
    def test_chat_complete_success(self, mock_openai, client: OpenAIClient):
        """测试聊天完成成功。"""
        # 模拟OpenAI响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        mock_response.usage.total_tokens = 20

        mock_instance = Mock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_instance

        # 准备消息
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello!"),
        ]

        # 调用聊天完成方法
        result = client.chat_complete(messages)

        assert result == "Hello! How can I help you?"

        expected_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        mock_instance.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=expected_messages,
            temperature=0.7,
            max_tokens=None,
        )

    @patch("agents.llm.client.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_chat_complete_async_no_stream(
        self, mock_async_openai, client: OpenAIClient
    ):
        """测试异步聊天完成（非流式）。"""
        # 模拟异步OpenAI响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        mock_response.usage.total_tokens = 20

        mock_instance = AsyncMock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_async_openai.return_value = mock_instance

        # 准备消息
        messages = [ChatMessage(role="user", content="Hello!")]

        # 调用异步聊天完成方法
        result = await client.chat_complete_async(messages, stream=False)

        assert result == "Hello! How can I help you?"

        expected_messages = [{"role": "user", "content": "Hello!"}]
        mock_instance.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=expected_messages,
            temperature=0.7,
            max_tokens=None,
        )

    @patch("agents.llm.client.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_chat_complete_async_stream(
        self, mock_async_openai, client: OpenAIClient
    ):
        """测试异步聊天完成（流式）。"""
        # 模拟流式响应
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" there"))]),
            Mock(choices=[Mock(delta=Mock(content="!"))]),
            Mock(choices=[Mock(delta=Mock(content=None))]),  # 结束
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        mock_instance = AsyncMock()
        mock_instance.chat.completions.create.return_value = mock_stream()
        mock_async_openai.return_value = mock_instance

        # 准备消息
        messages = [ChatMessage(role="user", content="Hello!")]

        # 调用流式聊天完成方法
        stream = await client.chat_complete_async(messages, stream=True)

        # 收集流式输出
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert chunks == ["Hello", " there", "!"]

        expected_messages = [{"role": "user", "content": "Hello!"}]
        mock_instance.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=expected_messages,
            temperature=0.7,
            max_tokens=None,
            stream=True,
        )

    @patch("agents.llm.client.OpenAI")
    def test_complete_with_custom_params(self, mock_openai, client: OpenAIClient):
        """测试使用自定义参数的完成。"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Custom response"
        mock_response.usage = None

        mock_instance = Mock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_instance

        # 使用自定义参数调用
        result = client.complete(
            "What is AI?", temperature=0.5, max_tokens=100, model="gpt-4"
        )

        assert result == "Custom response"
        mock_instance.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is AI?"}],
            temperature=0.5,
            max_tokens=100,
        )

    @patch("agents.llm.client.OpenAI")
    def test_complete_empty_response(self, mock_openai, client: OpenAIClient):
        """测试空响应处理。"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None  # 空内容
        mock_response.usage.total_tokens = 10

        mock_instance = Mock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_instance

        result = client.complete("Test prompt")

        assert result == ""  # 应该返回空字符串

    def test_handle_exception_authentication(self, client: OpenAIClient):
        """测试认证异常处理。"""
        error = Exception("authentication failed")

        result = client._handle_exception(error)

        assert isinstance(result, OpenAIAuthenticationException)
        assert "认证失败" in str(result)

    def test_handle_exception_rate_limit(self, client: OpenAIClient):
        """测试速率限制异常处理。"""
        error = Exception("rate_limit exceeded")

        result = client._handle_exception(error)

        assert isinstance(result, OpenAIRateLimitException)
        assert "速率限制" in str(result)

    def test_handle_exception_timeout(self, client: OpenAIClient):
        """测试超时异常处理。"""
        error = Exception("request timeout")

        result = client._handle_exception(error)

        assert isinstance(result, OpenAITimeoutException)
        assert "请求超时" in str(result)

    def test_handle_exception_generic(self, client: OpenAIClient):
        """测试通用异常处理。"""
        error = Exception("Some unknown error")

        result = client._handle_exception(error)

        assert isinstance(result, OpenAIException)
        assert "OpenAI API错误" in str(result)

    @patch("agents.llm.client.OpenAI")
    def test_complete_with_exception(self, mock_openai, client: OpenAIClient):
        """测试完成方法异常处理。"""
        mock_instance = Mock()
        mock_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_instance

        with pytest.raises(OpenAIException):
            client.complete("Test prompt")

    @patch("agents.llm.client.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_complete_async_with_exception(
        self, mock_async_openai, client: OpenAIClient
    ):
        """测试异步完成方法异常处理。"""
        mock_instance = AsyncMock()
        mock_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_async_openai.return_value = mock_instance

        with pytest.raises(OpenAIException):
            await client.complete_async("Test prompt")

    def test_close_sync_client(self, client: OpenAIClient):
        """测试关闭同步客户端。"""
        # 先创建一个模拟客户端
        mock_client = Mock()
        client._sync_client = mock_client

        # 关闭客户端
        client.close()

        mock_client.close.assert_called_once()
        assert client._sync_client is None

    @pytest.mark.asyncio
    async def test_close_async_client(self, client: OpenAIClient):
        """测试关闭异步客户端。"""
        # 先创建一个模拟异步客户端
        mock_client = AsyncMock()
        client._async_client = mock_client

        # 关闭客户端
        await client.close_async()

        mock_client.close.assert_called_once()
        assert client._async_client is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self, client: OpenAIClient):
        """测试异步上下文管理器。"""
        mock_client = AsyncMock()
        client._async_client = mock_client

        async with client as ctx_client:
            assert ctx_client is client

        # 退出上下文时应该关闭客户端
        mock_client.close.assert_called_once()

    @patch("agents.llm.client.OpenAI")
    def test_sync_client_with_base_url(self, mock_openai):
        """测试带有base_url的同步客户端创建。"""
        config = OpenAIConfig(
            api_key="test-key",
            base_url="https://custom.api.com",
        )
        client = OpenAIClient(config)

        # 访问客户端属性以触发创建
        _ = client.sync_client

        mock_openai.assert_called_once_with(
            api_key="test-key",
            timeout=30,
            max_retries=3,
            base_url="https://custom.api.com",
        )

    @patch("agents.llm.client.AsyncOpenAI")
    def test_async_client_with_base_url(self, mock_async_openai):
        """测试带有base_url的异步客户端创建。"""
        config = OpenAIConfig(
            api_key="test-key",
            base_url="https://custom.api.com",
        )
        client = OpenAIClient(config)

        # 访问客户端属性以触发创建
        _ = client.async_client

        mock_async_openai.assert_called_once_with(
            api_key="test-key",
            timeout=30,
            max_retries=3,
            base_url="https://custom.api.com",
        )
