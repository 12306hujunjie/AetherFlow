"""OpenAI API客户端抽象层。"""

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI, OpenAI

from ..exceptions import (
    OpenAIAuthenticationException,
    OpenAIException,
    OpenAIRateLimitException,
    OpenAITimeoutException,
)
from .config import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    OpenAIConfig,
)
from .retry import DEFAULT_RETRY_CONFIG, CircuitBreaker, retry_on_failure

logger = logging.getLogger("aetherflow.agents.llm")


class OpenAIClient:
    """OpenAI API客户端，支持同步和异步调用，集成依赖注入。

    该客户端提供了对OpenAI API的抽象，支持：
    - 异步和同步调用
    - 智能重试机制
    - 错误分类和处理
    - 流式响应支持
    - 连接池管理
    """

    def __init__(self, config: OpenAIConfig):
        """初始化OpenAI客户端。

        Args:
            config: OpenAI配置实例
        """
        self.config = config
        self._sync_client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

        # 初始化断路器
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=OpenAIException,
        )

        logger.info(
            f"OpenAI客户端初始化完成，模型: {config.model}, "
            f"超时: {config.timeout}s, 重试: {config.max_retries}次"
        )

    @property
    def sync_client(self) -> OpenAI:
        """获取同步客户端实例。"""
        if self._sync_client is None:
            client_kwargs = {
                "api_key": self.config.api_key,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url

            self._sync_client = OpenAI(**client_kwargs)
        return self._sync_client

    @property
    def async_client(self) -> AsyncOpenAI:
        """获取异步客户端实例。"""
        if self._async_client is None:
            client_kwargs = {
                "api_key": self.config.api_key,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
            }
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url

            self._async_client = AsyncOpenAI(**client_kwargs)
        return self._async_client

    @retry_on_failure(config=DEFAULT_RETRY_CONFIG)
    def complete(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        """同步文本补全。

        Args:
            prompt: 提示文本
            temperature: 温度参数，默认使用配置中的值
            max_tokens: 最大token数，默认使用配置中的值
            model: 模型名称，默认使用配置中的值

        Returns:
            生成的文本内容

        Raises:
            OpenAIException: OpenAI API调用异常
        """
        request = CompletionRequest(
            prompt=prompt,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            model=model or self.config.model,
        )

        try:
            start_time = time.time()

            # 使用chat completions API，因为completions API已被弃用
            response = self.sync_client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            elapsed_time = time.time() - start_time
            content = response.choices[0].message.content or ""

            logger.info(
                f"同步completion完成，耗时: {elapsed_time:.2f}s, "
                f"tokens: {response.usage.total_tokens if response.usage else 'N/A'}"
            )

            return content

        except Exception as e:
            logger.error(f"同步completion失败: {e}")
            raise self._handle_exception(e)

    @retry_on_failure(config=DEFAULT_RETRY_CONFIG)
    async def complete_async(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        """异步文本补全。

        Args:
            prompt: 提示文本
            temperature: 温度参数，默认使用配置中的值
            max_tokens: 最大token数，默认使用配置中的值
            model: 模型名称，默认使用配置中的值

        Returns:
            生成的文本内容

        Raises:
            OpenAIException: OpenAI API调用异常
        """
        request = CompletionRequest(
            prompt=prompt,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            model=model or self.config.model,
        )

        try:
            start_time = time.time()

            response = await self.async_client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            elapsed_time = time.time() - start_time
            content = response.choices[0].message.content or ""

            logger.info(
                f"异步completion完成，耗时: {elapsed_time:.2f}s, "
                f"tokens: {response.usage.total_tokens if response.usage else 'N/A'}"
            )

            return content

        except Exception as e:
            logger.error(f"异步completion失败: {e}")
            raise self._handle_exception(e)

    @retry_on_failure(config=DEFAULT_RETRY_CONFIG)
    def chat_complete(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        """同步聊天补全。

        Args:
            messages: 聊天消息列表
            temperature: 温度参数，默认使用配置中的值
            max_tokens: 最大token数，默认使用配置中的值
            model: 模型名称，默认使用配置中的值

        Returns:
            生成的回复内容

        Raises:
            OpenAIException: OpenAI API调用异常
        """
        request = ChatCompletionRequest(
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            model=model or self.config.model,
        )

        try:
            start_time = time.time()

            # 转换消息格式
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ]

            response = self.sync_client.chat.completions.create(
                model=request.model,
                messages=openai_messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            elapsed_time = time.time() - start_time
            content = response.choices[0].message.content or ""

            logger.info(
                f"同步chat completion完成，耗时: {elapsed_time:.2f}s, "
                f"tokens: {response.usage.total_tokens if response.usage else 'N/A'}"
            )

            return content

        except Exception as e:
            logger.error(f"同步chat completion失败: {e}")
            raise self._handle_exception(e)

    @retry_on_failure(config=DEFAULT_RETRY_CONFIG)
    async def chat_complete_async(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """异步聊天补全。

        Args:
            messages: 聊天消息列表
            temperature: 温度参数，默认使用配置中的值
            max_tokens: 最大token数，默认使用配置中的值
            model: 模型名称，默认使用配置中的值
            stream: 是否启用流式响应

        Returns:
            生成的回复内容或流式内容迭代器

        Raises:
            OpenAIException: OpenAI API调用异常
        """
        request = ChatCompletionRequest(
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            model=model or self.config.model,
            stream=stream,
        )

        try:
            start_time = time.time()

            # 转换消息格式
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ]

            if stream:
                return self._create_stream_response(
                    request.model,
                    openai_messages,
                    request.temperature,
                    request.max_tokens,
                )
            else:
                response = await self.async_client.chat.completions.create(
                    model=request.model,
                    messages=openai_messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )

                elapsed_time = time.time() - start_time
                content = response.choices[0].message.content or ""

                logger.info(
                    f"异步chat completion完成，耗时: {elapsed_time:.2f}s, "
                    f"tokens: {response.usage.total_tokens if response.usage else 'N/A'}"
                )

                return content

        except Exception as e:
            logger.error(f"异步chat completion失败: {e}")
            raise self._handle_exception(e)

    async def _create_stream_response(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
    ) -> AsyncIterator[str]:
        """创建流式响应迭代器。

        Args:
            model: 模型名称
            messages: OpenAI格式的消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Yields:
            生成的内容片段
        """
        try:
            stream = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"流式响应异常: {e}")
            raise self._handle_exception(e)

    def _handle_exception(self, error: Exception) -> OpenAIException:
        """处理和转换异常类型。

        Args:
            error: 原始异常

        Returns:
            转换后的OpenAI异常
        """
        error_msg = str(error)

        # 分类异常类型
        if "authentication" in error_msg.lower() or "401" in error_msg:
            return OpenAIAuthenticationException(f"认证失败: {error_msg}")
        elif "rate_limit" in error_msg.lower() or "429" in error_msg:
            return OpenAIRateLimitException(f"速率限制: {error_msg}")
        elif "timeout" in error_msg.lower():
            return OpenAITimeoutException(f"请求超时: {error_msg}")
        else:
            return OpenAIException(f"OpenAI API错误: {error_msg}")

    async def close_async(self) -> None:
        """关闭异步客户端连接。"""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None
            logger.info("异步OpenAI客户端连接已关闭")

    def close(self) -> None:
        """关闭同步客户端连接。"""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
            logger.info("同步OpenAI客户端连接已关闭")

    async def __aenter__(self) -> "OpenAIClient":
        """异步上下文管理器入口。"""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器退出。"""
        await self.close_async()
