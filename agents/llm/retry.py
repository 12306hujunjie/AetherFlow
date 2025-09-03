"""智能重试机制，支持指数退避和异常分类。"""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps

from ..exceptions import (
    OpenAIAuthenticationException,
    OpenAIException,
    OpenAIRateLimitException,
    OpenAITimeoutException,
)

logger = logging.getLogger("aetherflow.agents.llm.retry")


@dataclass
class RetryConfig:
    """重试配置。"""

    max_attempts: int = 3
    """最大重试次数"""

    base_delay: float = 1.0
    """基础延迟时间（秒）"""

    max_delay: float = 60.0
    """最大延迟时间（秒）"""

    backoff_factor: float = 2.0
    """退避因子"""

    jitter: bool = True
    """是否添加随机抖动"""

    retry_on_exceptions: tuple = (
        OpenAIRateLimitException,
        OpenAITimeoutException,
    )
    """可重试的异常类型"""

    permanent_exceptions: tuple = (OpenAIAuthenticationException,)
    """永久性异常，不应重试"""


class RetryHandler:
    """智能重试处理器。"""

    def __init__(self, config: RetryConfig | None = None):
        """初始化重试处理器。

        Args:
            config: 重试配置，如果为None则使用默认配置
        """
        self.config = config or RetryConfig()

    def calculate_delay(self, attempt: int, base_delay: float | None = None) -> float:
        """计算重试延迟时间。

        Args:
            attempt: 当前重试次数（从0开始）
            base_delay: 基础延迟时间，如果为None则使用配置中的值

        Returns:
            计算出的延迟时间（秒）
        """
        base = base_delay or self.config.base_delay

        # 指数退避
        delay = base * (self.config.backoff_factor**attempt)

        # 限制最大延迟
        delay = min(delay, self.config.max_delay)

        # 添加随机抖动
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10%的抖动
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """判断是否应该重试。

        Args:
            exception: 捕获的异常
            attempt: 当前重试次数（从0开始）

        Returns:
            是否应该重试
        """
        # 超过最大重试次数
        if attempt >= self.config.max_attempts:
            return False

        # 永久性异常不重试
        if isinstance(exception, self.config.permanent_exceptions):
            logger.info(f"永久性异常，不重试: {exception}")
            return False

        # 检查是否是可重试的异常
        if isinstance(exception, self.config.retry_on_exceptions):
            logger.info(f"检测到可重试异常，准备重试: {exception}")
            return True

        # 检查异常是否有retryable属性
        if hasattr(exception, "retryable") and exception.retryable:
            logger.info(f"异常标记为可重试: {exception}")
            return True

        return False

    def get_retry_delay(self, exception: Exception, attempt: int) -> float:
        """获取重试延迟时间。

        Args:
            exception: 捕获的异常
            attempt: 当前重试次数（从0开始）

        Returns:
            延迟时间（秒）
        """
        base_delay = self.config.base_delay

        # 对于速率限制异常，使用更长的延迟
        if isinstance(exception, OpenAIRateLimitException):
            if hasattr(exception, "retry_after") and exception.retry_after:
                base_delay = max(base_delay, exception.retry_after)

        return self.calculate_delay(attempt, base_delay)


def retry_on_failure(
    config: RetryConfig | None = None,
    logger_name: str | None = None,
) -> Callable:
    """重试装饰器工厂。

    Args:
        config: 重试配置
        logger_name: 日志记录器名称

    Returns:
        重试装饰器
    """
    retry_config = config or RetryConfig()
    retry_handler = RetryHandler(retry_config)
    retry_logger = logging.getLogger(logger_name or "aetherflow.agents.llm.retry")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            """同步函数包装器。"""
            last_exception = None

            for attempt in range(retry_config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not retry_handler.should_retry(e, attempt):
                        retry_logger.error(
                            f"函数 {func.__name__} 执行失败，不可重试: {e}"
                        )
                        raise

                    delay = retry_handler.get_retry_delay(e, attempt)
                    retry_logger.warning(
                        f"函数 {func.__name__} 执行失败 (尝试 {attempt + 1}/{retry_config.max_attempts})，"
                        f"{delay:.2f}秒后重试: {e}"
                    )

                    time.sleep(delay)

            # 所有重试都失败了
            retry_logger.error(f"函数 {func.__name__} 所有重试均失败")
            raise last_exception

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            """异步函数包装器。"""
            last_exception = None

            for attempt in range(retry_config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not retry_handler.should_retry(e, attempt):
                        retry_logger.error(
                            f"异步函数 {func.__name__} 执行失败，不可重试: {e}"
                        )
                        raise

                    delay = retry_handler.get_retry_delay(e, attempt)
                    retry_logger.warning(
                        f"异步函数 {func.__name__} 执行失败 (尝试 {attempt + 1}/{retry_config.max_attempts})，"
                        f"{delay:.2f}秒后重试: {e}"
                    )

                    await asyncio.sleep(delay)

            # 所有重试都失败了
            retry_logger.error(f"异步函数 {func.__name__} 所有重试均失败")
            raise last_exception

        # 根据函数类型返回相应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class CircuitBreaker:
    """断路器模式，防止级联故障。"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = OpenAIException,
    ):
        """初始化断路器。

        Args:
            failure_threshold: 故障阈值，连续失败次数
            recovery_timeout: 恢复超时时间（秒）
            expected_exception: 预期的异常类型
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    @property
    def state(self) -> str:
        """获取当前状态。"""
        return self._state

    def _should_allow_request(self) -> bool:
        """判断是否应该允许请求。"""
        if self._state == "CLOSED":
            return True
        elif self._state == "OPEN":
            # 检查是否到了恢复时间
            if (
                self._last_failure_time
                and time.time() - self._last_failure_time >= self.recovery_timeout
            ):
                self._state = "HALF_OPEN"
                logger.info("断路器状态变更为 HALF_OPEN")
                return True
            return False
        elif self._state == "HALF_OPEN":
            return True

        return False

    def _on_success(self) -> None:
        """记录成功调用。"""
        self._failure_count = 0
        if self._state != "CLOSED":
            self._state = "CLOSED"
            logger.info("断路器状态变更为 CLOSED")

    def _on_failure(self, exception: Exception) -> None:
        """记录失败调用。"""
        if isinstance(exception, self.expected_exception):
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                if self._state != "OPEN":
                    self._state = "OPEN"
                    logger.warning(
                        f"断路器状态变更为 OPEN，失败次数: {self._failure_count}"
                    )

    def __call__(self, func: Callable) -> Callable:
        """断路器装饰器。"""

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not self._should_allow_request():
                raise OpenAIException("断路器打开，拒绝请求")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure(e)
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not self._should_allow_request():
                raise OpenAIException("断路器打开，拒绝请求")

            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure(e)
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


# 预定义的重试配置
DEFAULT_RETRY_CONFIG = RetryConfig()

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=2.0,
    max_delay=120.0,
    backoff_factor=2.5,
)

CONSERVATIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    base_delay=0.5,
    max_delay=10.0,
    backoff_factor=1.5,
)
