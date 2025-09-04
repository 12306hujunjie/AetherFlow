"""测试重试机制。"""

import time
from unittest.mock import AsyncMock, patch

import pytest

from agents.exceptions import (
    OpenAIAuthenticationException,
    OpenAIException,
    OpenAIRateLimitException,
    OpenAITimeoutException,
)
from agents.llm.retry import (
    AGGRESSIVE_RETRY_CONFIG,
    CONSERVATIVE_RETRY_CONFIG,
    DEFAULT_RETRY_CONFIG,
    CircuitBreaker,
    RetryConfig,
    RetryHandler,
    retry_on_failure,
)


class TestRetryConfig:
    """测试重试配置类。"""

    def test_default_config(self):
        """测试默认配置。"""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_factor == 2.0
        assert config.jitter is True
        assert OpenAIRateLimitException in config.retry_on_exceptions
        assert OpenAITimeoutException in config.retry_on_exceptions
        assert OpenAIAuthenticationException in config.permanent_exceptions

    def test_custom_config(self):
        """测试自定义配置。"""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            backoff_factor=3.0,
            jitter=False,
        )

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_factor == 3.0
        assert config.jitter is False

    def test_predefined_configs(self):
        """测试预定义配置。"""
        # 默认配置
        assert DEFAULT_RETRY_CONFIG.max_attempts == 3

        # 激进配置
        assert AGGRESSIVE_RETRY_CONFIG.max_attempts == 5
        assert AGGRESSIVE_RETRY_CONFIG.base_delay == 2.0
        assert AGGRESSIVE_RETRY_CONFIG.max_delay == 120.0

        # 保守配置
        assert CONSERVATIVE_RETRY_CONFIG.max_attempts == 2
        assert CONSERVATIVE_RETRY_CONFIG.base_delay == 0.5
        assert CONSERVATIVE_RETRY_CONFIG.max_delay == 10.0


class TestRetryHandler:
    """测试重试处理器。"""

    @pytest.fixture
    def handler(self) -> RetryHandler:
        """重试处理器fixture。"""
        config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0,
            backoff_factor=2.0,
            jitter=False,  # 关闭抖动以便测试
        )
        return RetryHandler(config)

    def test_calculate_delay_no_jitter(self, handler: RetryHandler):
        """测试计算延迟（无抖动）。"""
        # 第一次重试
        delay = handler.calculate_delay(0)
        assert delay == 1.0

        # 第二次重试
        delay = handler.calculate_delay(1)
        assert delay == 2.0

        # 第三次重试
        delay = handler.calculate_delay(2)
        assert delay == 4.0

        # 超过最大延迟
        delay = handler.calculate_delay(5)
        assert delay == 10.0  # 应该被限制为max_delay

    def test_calculate_delay_with_jitter(self):
        """测试计算延迟（有抖动）。"""
        config = RetryConfig(base_delay=1.0, jitter=True)
        handler = RetryHandler(config)

        # 多次计算应该得到不同的值（由于抖动）
        delays = [handler.calculate_delay(0) for _ in range(10)]

        # 所有延迟应该在合理范围内
        for delay in delays:
            assert 0.9 <= delay <= 1.1  # 允许10%的抖动

        # 应该有一些变化（不是所有值都相同）
        assert len(set(delays)) > 1

    def test_should_retry_max_attempts(self, handler: RetryHandler):
        """测试最大重试次数限制。"""
        exception = OpenAIRateLimitException("Rate limit")

        # 前3次应该重试
        assert handler.should_retry(exception, 0) is True
        assert handler.should_retry(exception, 1) is True
        assert handler.should_retry(exception, 2) is True

        # 超过最大次数不应该重试
        assert handler.should_retry(exception, 3) is False

    def test_should_retry_permanent_exception(self, handler: RetryHandler):
        """测试永久性异常不重试。"""
        exception = OpenAIAuthenticationException("Auth failed")

        # 永久性异常不应该重试
        assert handler.should_retry(exception, 0) is False
        assert handler.should_retry(exception, 1) is False

    def test_should_retry_retryable_exception(self, handler: RetryHandler):
        """测试可重试异常。"""
        rate_limit_ex = OpenAIRateLimitException("Rate limit")
        timeout_ex = OpenAITimeoutException("Timeout")

        assert handler.should_retry(rate_limit_ex, 0) is True
        assert handler.should_retry(timeout_ex, 0) is True

    def test_should_retry_custom_retryable_attribute(self, handler: RetryHandler):
        """测试自定义retryable属性。"""

        # 创建一个有retryable属性的异常
        class CustomException(Exception):
            retryable = True

        exception = CustomException("Custom error")
        assert handler.should_retry(exception, 0) is True

        # 设置为不可重试
        exception.retryable = False
        assert handler.should_retry(exception, 0) is False

    def test_should_retry_unknown_exception(self, handler: RetryHandler):
        """测试未知异常。"""
        exception = ValueError("Unknown error")

        # 未知异常默认不重试
        assert handler.should_retry(exception, 0) is False

    def test_get_retry_delay_default(self, handler: RetryHandler):
        """测试获取重试延迟。"""
        exception = OpenAITimeoutException("Timeout")

        delay = handler.get_retry_delay(exception, 1)
        assert delay == 2.0  # base_delay * backoff_factor ^ 1

    def test_get_retry_delay_rate_limit_with_retry_after(self, handler: RetryHandler):
        """测试速率限制异常的重试延迟。"""
        exception = OpenAIRateLimitException("Rate limit", retry_after=5)

        delay = handler.get_retry_delay(exception, 0)
        assert delay == 5.0  # 应该使用retry_after


class TestRetryDecorator:
    """测试重试装饰器。"""

    def test_sync_function_success(self):
        """测试同步函数成功执行。"""

        @retry_on_failure()
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_sync_function_retry_success(self):
        """测试同步函数重试后成功。"""
        call_count = 0

        @retry_on_failure()
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OpenAIRateLimitException("Rate limit")
            return "success after retries"

        with patch("time.sleep"):  # 模拟sleep以加快测试
            result = flaky_function()

        assert result == "success after retries"
        assert call_count == 3

    def test_sync_function_permanent_failure(self):
        """测试同步函数永久失败。"""

        @retry_on_failure()
        def permanent_failure():
            raise OpenAIAuthenticationException("Auth failed")

        with pytest.raises(OpenAIAuthenticationException):
            permanent_failure()

    def test_sync_function_max_retries_exceeded(self):
        """测试同步函数超过最大重试次数。"""
        call_count = 0

        @retry_on_failure(config=RetryConfig(max_attempts=2))
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise OpenAIRateLimitException("Rate limit")

        with patch("time.sleep"):
            with pytest.raises(OpenAIRateLimitException):
                always_fails()

        assert call_count == 2  # 应该调用2次

    @pytest.mark.asyncio
    async def test_async_function_success(self):
        """测试异步函数成功执行。"""

        @retry_on_failure()
        async def async_successful_function():
            return "async success"

        result = await async_successful_function()
        assert result == "async success"

    @pytest.mark.asyncio
    async def test_async_function_retry_success(self):
        """测试异步函数重试后成功。"""
        call_count = 0

        @retry_on_failure()
        async def async_flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OpenAITimeoutException("Timeout")
            return "async success after retries"

        with patch("asyncio.sleep", new_callable=lambda: AsyncMock()):
            result = await async_flaky_function()

        assert result == "async success after retries"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_function_permanent_failure(self):
        """测试异步函数永久失败。"""

        @retry_on_failure()
        async def async_permanent_failure():
            raise OpenAIAuthenticationException("Auth failed")

        with pytest.raises(OpenAIAuthenticationException):
            await async_permanent_failure()


class TestCircuitBreaker:
    """测试断路器。"""

    def test_circuit_breaker_initialization(self):
        """测试断路器初始化。"""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=OpenAIException,
        )

        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30.0
        assert breaker.expected_exception == OpenAIException
        assert breaker.state == "CLOSED"

    def test_circuit_breaker_closed_state(self):
        """测试断路器关闭状态。"""
        breaker = CircuitBreaker(failure_threshold=2)

        @breaker
        def test_function():
            return "success"

        # 在关闭状态下应该正常执行
        result = test_function()
        assert result == "success"
        assert breaker.state == "CLOSED"

    def test_circuit_breaker_open_after_failures(self):
        """测试断路器在失败后开启。"""
        breaker = CircuitBreaker(failure_threshold=2)
        call_count = 0

        @breaker
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise OpenAIException("Test error")

        # 前两次调用应该执行并记录失败
        with pytest.raises(OpenAIException):
            failing_function()
        assert breaker.state == "CLOSED"

        with pytest.raises(OpenAIException):
            failing_function()
        assert breaker.state == "OPEN"

        # 第三次调用应该直接被断路器阻止
        with pytest.raises(OpenAIException, match="断路器打开"):
            failing_function()

        assert call_count == 2  # 只执行了2次

    def test_circuit_breaker_half_open_recovery(self):
        """测试断路器半开恢复。"""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        @breaker
        def test_function(should_fail=True):
            if should_fail:
                raise OpenAIException("Test error")
            return "success"

        # 触发断路器开启
        with pytest.raises(OpenAIException):
            test_function()
        assert breaker.state == "OPEN"

        # 等待恢复超时
        time.sleep(0.15)

        # 下一次调用应该进入半开状态并成功
        result = test_function(should_fail=False)
        assert result == "success"
        assert breaker.state == "CLOSED"

    def test_circuit_breaker_non_expected_exception(self):
        """测试非预期异常不影响断路器。"""
        breaker = CircuitBreaker(
            failure_threshold=1, expected_exception=OpenAIException
        )

        @breaker
        def test_function():
            raise ValueError("Different exception")

        # 非预期异常应该正常抛出，不影响断路器状态
        with pytest.raises(ValueError):
            test_function()
        assert breaker.state == "CLOSED"

    @pytest.mark.asyncio
    async def test_circuit_breaker_async_function(self):
        """测试断路器异步函数支持。"""
        breaker = CircuitBreaker(failure_threshold=1)

        @breaker
        async def async_test_function():
            return "async success"

        result = await async_test_function()
        assert result == "async success"
        assert breaker.state == "CLOSED"

    def test_circuit_breaker_success_resets_failure_count(self):
        """测试成功调用重置失败计数。"""
        breaker = CircuitBreaker(failure_threshold=3)

        @breaker
        def test_function(should_fail=True):
            if should_fail:
                raise OpenAIException("Test error")
            return "success"

        # 两次失败
        with pytest.raises(OpenAIException):
            test_function()
        with pytest.raises(OpenAIException):
            test_function()

        assert breaker._failure_count == 2
        assert breaker.state == "CLOSED"

        # 一次成功应该重置计数
        result = test_function(should_fail=False)
        assert result == "success"
        assert breaker._failure_count == 0
        assert breaker.state == "CLOSED"
