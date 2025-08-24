#!/usr/bin/env python3
"""
test_retry_decorator.py - 测试重试装饰器功能
包含：基本重试、异常类型过滤、指数退避、Node集成、@node装饰器集成等测试
"""

import time
from unittest.mock import patch

import pytest

from src.aetherflow import (
    NodeExecutionException,
    NodeRetryExhaustedException,
    RetryConfig,
    node,
)


class TestRetryDecorator:
    """重试装饰器基础功能测试"""

    def test_retry_config_basic(self):
        """测试RetryConfig基本功能"""
        config = RetryConfig(
            retry_count=3,
            retry_delay=0.1,
            exception_types=(ValueError, TypeError),
            backoff_factor=2.0,
            max_delay=5.0,
        )

        # 测试异常类型判断
        assert config.should_retry(ValueError("test"))
        assert config.should_retry(TypeError("test"))
        assert not config.should_retry(RuntimeError("test"))

        # 测试延迟计算
        assert config.get_delay(0) == 0.1  # 基础延迟
        assert config.get_delay(1) == 0.2  # 0.1 * 2^1
        assert config.get_delay(2) == 0.4  # 0.1 * 2^2
        assert config.get_delay(10) == 5.0  # 受max_delay限制

    def test_retry_decorator_success_first_try(self):
        """测试第一次尝试就成功的情况"""
        call_count = 0

        @node(retry_count=3, retry_delay=0.01)
        def success_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result = success_function(5)
        assert result == 10
        assert call_count == 1  # 只调用一次

    def test_retry_decorator_success_after_retries(self):
        """测试重试后成功的情况"""
        call_count = 0

        @node(retry_count=3, retry_delay=0.01, exception_types=(ValueError,))
        def flaky_function(x):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # 前两次失败
                raise ValueError(f"Attempt {call_count} failed")
            return x * 2

        result = flaky_function(5)
        assert result == 10
        assert call_count == 3  # 调用3次

    def test_retry_decorator_exhausted(self):
        """测试重试次数耗尽的情况"""
        call_count = 0

        @node(
            retry_count=2,
            retry_delay=0.01,
            name="test_node",
            exception_types=(ValueError,),
        )
        def always_fail_function(x):
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Attempt {call_count} failed")

        with pytest.raises(NodeRetryExhaustedException) as exc_info:
            always_fail_function(5)

        assert call_count == 3  # 初始尝试 + 2次重试
        assert "test_node" in str(exc_info.value)
        assert "重试次数耗尽" in str(exc_info.value)
        assert exc_info.value.retry_count == 2
        assert isinstance(exc_info.value.last_exception, ValueError)

    def test_retry_decorator_non_retryable_exception(self):
        """测试不可重试异常类型"""
        call_count = 0

        @node(
            retry_count=3,
            retry_delay=0.01,
            exception_types=(ValueError,),  # 只重试ValueError
            name="test_node",
        )
        def mixed_exception_function(x):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Non-retryable error")  # 不可重试
            raise ValueError("Retryable error")

        with pytest.raises(NodeExecutionException) as exc_info:
            mixed_exception_function(5)

        assert call_count == 1  # 只调用一次
        assert "不支持重试" in str(exc_info.value)
        assert isinstance(exc_info.value.original_exception, RuntimeError)

    def test_retry_decorator_backoff(self):
        """测试指数退避功能"""
        call_times = []

        @node(
            retry_count=2,
            retry_delay=0.1,
            backoff_factor=2.0,
            name="backoff_test",
            exception_types=(ValueError,),
        )
        def backoff_function():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Still failing")
            return "success"

        start_time = time.time()
        result = backoff_function()

        assert result == "success"
        assert len(call_times) == 3

        # 验证延迟时间（允许一定误差）
        delay1 = call_times[1] - call_times[0]  # 第一次重试延迟
        delay2 = call_times[2] - call_times[1]  # 第二次重试延迟

        assert 0.08 <= delay1 <= 0.15  # 约0.1秒
        assert 0.18 <= delay2 <= 0.25  # 约0.2秒（指数退避）


class TestNodeIntegration:
    """Node类重试集成测试"""

    def test_node_default_retry_enabled(self):
        """测试Node默认启用重试"""
        call_count = 0

        @node
        def flaky_func(x):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("First attempt failed")
            return x * 2

        result = flaky_func(5)

        assert result == 10
        assert call_count == 2

    def test_node_custom_retry_config(self):
        """测试Node自定义重试配置"""
        call_count = 0

        @node(retry_count=5, retry_delay=0.01, exception_types=(ValueError,))
        def flaky_func(x):
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ValueError(f"Attempt {call_count} failed")
            return x * 2

        result = flaky_func(5)
        assert result == 10
        assert call_count == 4

    def test_node_retry_disabled(self):
        """测试Node禁用重试"""
        call_count = 0

        @node(enable_retry=False)
        def always_fail_func(x):
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fail_func(5)

        assert call_count == 1  # 只调用一次

    def test_node_chain_with_retry(self):
        """测试带重试的节点链式调用"""
        call_count_1 = 0
        call_count_2 = 0

        @node(retry_delay=0.01)
        def flaky_func_1(x):
            nonlocal call_count_1
            call_count_1 += 1
            if call_count_1 < 2:
                raise ConnectionError("Node1 first attempt failed")
            return x * 2

        @node(retry_delay=0.01)
        def flaky_func_2(x):
            nonlocal call_count_2
            call_count_2 += 1
            if call_count_2 < 3:
                raise ConnectionError("Node2 attempts failed")
            return x + 10

        chain = flaky_func_1.then(flaky_func_2)
        result = chain(5)  # 5 * 2 + 10 = 20

        assert result == 20
        assert call_count_1 == 2  # node1重试1次
        assert call_count_2 == 3  # node2重试2次


class TestNodeDecoratorIntegration:
    """@node装饰器重试集成测试"""

    def test_node_decorator_default_retry(self):
        """测试@node装饰器默认重试"""
        call_count = 0

        @node
        def flaky_decorated_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Decorated function first attempt failed")
            return x * 3

        result = flaky_decorated_func(4)
        assert result == 12
        assert call_count == 2

    def test_node_decorator_custom_retry(self):
        """测试@node装饰器自定义重试配置"""
        call_count = 0

        @node(
            retry_count=5,
            retry_delay=0.01,
            exception_types=(ConnectionError, TypeError),
        )
        def custom_retry_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ConnectionError(f"Custom retry attempt {call_count} failed")
            return x * 4

        result = custom_retry_func(3)
        assert result == 12
        assert call_count == 4

    def test_node_decorator_no_retry(self):
        """测试@node装饰器禁用重试"""
        call_count = 0

        @node(enable_retry=False)
        def no_retry_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            raise ValueError("No retry function always fails")

        with pytest.raises(ValueError):
            no_retry_func(5)

        assert call_count == 1

    def test_node_decorator_chain_with_different_retry_configs(self):
        """测试不同重试配置的@node装饰器链式调用"""
        call_count_a = 0
        call_count_b = 0

        @node(retry_count=2, retry_delay=0.01)
        def func_a(x: int) -> int:
            nonlocal call_count_a
            call_count_a += 1
            if call_count_a < 2:
                raise ConnectionError("Func A failed")
            return x * 2

        @node(retry_count=4, retry_delay=0.01, exception_types=(ConnectionError,))
        def func_b(x: int) -> int:
            nonlocal call_count_b
            call_count_b += 1
            if call_count_b < 3:
                raise ConnectionError("Func B failed")
            return x + 5

        chain = func_a.then(func_b)
        result = chain(3)  # 3 * 2 + 5 = 11

        assert result == 11
        assert call_count_a == 2
        assert call_count_b == 3


class TestRetryLogging:
    """重试日志测试"""

    @patch("src.aetherflow.logger")
    def test_retry_logging(self, mock_logger):
        """测试重试过程的日志记录"""
        call_count = 0

        @node(
            retry_count=2,
            retry_delay=0.01,
            name="log_test_node",
            exception_types=(ValueError,),
        )
        def logging_func(x):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return x * 2

        result = logging_func(5)

        assert result == 10
        assert call_count == 3

        # 验证日志调用
        assert mock_logger.debug.call_count >= 3  # 每次尝试都有debug日志
        assert mock_logger.warning.call_count == 2  # 两次重试警告
        assert mock_logger.info.call_count == 1  # 最终成功信息

        # 验证日志内容
        warning_calls = [call.args[0] for call in mock_logger.warning.call_args_list]
        assert any("log_test_node" in msg and "重试" in msg for msg in warning_calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
