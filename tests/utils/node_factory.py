"""标准化的测试节点定义

使用@node装饰器直接定义节点函数，避免pickle序列化问题。
所有节点函数都定义在模块级别，支持进程池序列化。"""

import random
import time
from collections.abc import Callable
from typing import Any

from src.aetherflow import Node, ParallelResult, node

from ..shared.data_models import (
    FinalResult,
    ProcessedUser,
    StandardTestData,
    TestNodeConfig,
    UserInput,
)


# 通用基础节点函数 - 支持pickle序列化
# 基础数学运算节点
@node
def multiply_by_2_node(x: int) -> int:
    """乘以2节点"""
    result = x * 2
    print(f"Multiply: {x} -> {result}")
    return result


@node
def double_node(x: int) -> int:
    """双倍节点"""
    return x * 2


@node
def square_node(x: int) -> int:
    """平方节点"""
    return x * x


@node
def add_10_node(x: int) -> int:
    """加10节点"""
    result = x + 10
    print(f"Add 10: {x} -> {result}")
    return result


@node
def add_1_node(x: int) -> int:
    """加1节点"""
    return x + 1


@node
def multiply_by_3_node(x: int) -> int:
    """乘以3节点"""
    return x * 3


@node
def format_result_node(x: int) -> str:
    """格式化结果节点"""
    result = f"结果: {x}"
    print(f"Format: {x} -> {result}")
    return result


@node
def format_final_node(x: int) -> str:
    """格式化最终结果节点"""
    return f"Final: {x}"


@node
def add_prefix_node(s: str) -> str:
    """添加前缀节点"""
    return f"Result: {s}"


# 类型转换节点
@node
def to_string_node(x: int) -> str:
    """转字符串节点"""
    return str(x)


@node
def stringify_node(x: int) -> str:
    """字符串化节点"""
    return f"Number: {x}"


@node
def string_length_node(s: str) -> int:
    """字符串长度节点"""
    return len(s)


# 业务逻辑节点
@node
def process_user_input_node(user: UserInput) -> ProcessedUser:
    """处理用户输入节点"""
    print(f"Processing: {user.name}, age: {user.age}")
    return ProcessedUser(formatted_name=user.name.title(), is_adult=user.age >= 18)


@node
def generate_final_result_node(processed: ProcessedUser) -> FinalResult:
    """生成最终结果节点"""
    print(f"Generating result for: {processed.formatted_name}")
    user_type = "成人" if processed.is_adult else "未成年"
    return FinalResult(message=f"欢迎 {processed.formatted_name}", user_type=user_type)


# 数据处理节点
@node
def process_numbers_node(nums: list[int]) -> dict[str, float]:
    """处理数字列表节点"""
    total = sum(nums)
    avg = total / len(nums) if nums else 0
    return {"sum": total, "average": avg, "count": len(nums)}


@node
def format_stats_node(stats: dict[str, float]) -> str:
    """格式化统计信息节点"""
    return (
        f"统计: 总和={stats['sum']}, 平均={stats['average']:.2f}, 数量={stats['count']}"
    )


# 类型验证节点
@node
def strict_int_processor_node(x: int) -> str:
    """严格整数处理节点"""
    return f"processed_{x}"


@node
def strict_str_processor_node(s: str) -> int:
    """严格字符串处理节点"""
    return len(s)


@node
def strict_final_processor_node(length: int) -> dict[str, Any]:
    """严格最终处理节点"""
    return {"length": length, "valid": length > 5}


# 步骤节点
@node
def step_a_node(x: int) -> int:
    """步骤A节点"""
    return x + 1


@node
def step_b_node(x: int) -> int:
    """步骤B节点"""
    return x * 2


@node
def step_c_node(x: int) -> str:
    """步骤C节点"""
    return f"final_{x}"


# 模块级节点函数定义 - 使用@node装饰器，支持pickle序列化
# 标准处理器节点 - 使用配置参数
@node
def standard_processor_node(
    data: StandardTestData, config: TestNodeConfig
) -> StandardTestData:
    """标准处理器节点 - 支持pickle序列化"""
    if config.should_fail and random.random() < config.failure_rate:
        raise ValueError(f"节点 {config.name} 模拟失败")

    if config.delay_seconds > 0:
        time.sleep(config.delay_seconds)

    processed_value = data.value * config.multiplier
    return StandardTestData(
        value=processed_value,
        name=f"{config.name}_processed_{data.name}",
        metadata={
            "processor": config.name,
            "multiplier": config.multiplier,
            "original_value": data.value,
        },
    )


# 简单处理器节点 - 固定倍数
@node
def simple_processor_node(data: StandardTestData) -> StandardTestData:
    """简单处理器节点"""
    processed_value = data.value * 2
    return StandardTestData(
        value=processed_value,
        name=f"simple_processed_{data.name}",
        metadata={"multiplier": 2, "original_value": data.value},
    )


# 可序列化的纯函数 - 无装饰器修饰，可被pickle序列化
def success_processor_1_func(data: StandardTestData) -> StandardTestData:
    """成功处理器1 - 纯函数"""
    return StandardTestData(
        value=data.value * 2,
        name=f"success_1_processed_{data.name}",
        metadata={
            "processor": "success_1",
            "multiplier": 2,
            "original_value": data.value,
        },
    )


def success_processor_2_func(data: StandardTestData) -> StandardTestData:
    """成功处理器2 - 纯函数"""
    return StandardTestData(
        value=data.value * 3,
        name=f"success_2_processed_{data.name}",
        metadata={
            "processor": "success_2",
            "multiplier": 3,
            "original_value": data.value,
        },
    )


def failing_processor_1_func(data: StandardTestData) -> StandardTestData:
    """失败处理器1 - 纯函数，100%失败"""
    raise ValueError("节点 failure_1 模拟失败")


def failing_processor_2_func(data: StandardTestData) -> StandardTestData:
    """失败处理器2 - 纯函数，100%失败"""
    raise ValueError("节点 failure_2 模拟失败")


def slow_processor_1_func(data: StandardTestData) -> StandardTestData:
    """慢处理器1 - 纯函数"""
    time.sleep(0.1)
    return StandardTestData(
        value=data.value * 2,
        name=f"slow_1_processed_{data.name}",
        metadata={"processor": "slow_1", "multiplier": 2, "original_value": data.value},
    )


# Node包装器 - 用于非多进程场景
success_processor_1 = Node(success_processor_1_func, name="success_processor_1")
success_processor_2 = Node(success_processor_2_func, name="success_processor_2")
failing_processor_1 = Node(failing_processor_1_func, name="failing_processor_1")
failing_processor_2 = Node(failing_processor_2_func, name="failing_processor_2")
slow_processor_1 = Node(slow_processor_1_func, name="slow_processor_1")


# 聚合节点
@node
def aggregator_node(parallel_results: dict[str, ParallelResult]) -> dict[str, Any]:
    """标准聚合节点"""
    successful_results = []
    failed_results = []
    execution_times = []

    for key, result in parallel_results.items():
        if result.success:
            successful_results.append(
                {
                    "key": key,
                    "result": result.result,
                    "execution_time": result.execution_time,
                }
            )
        else:
            failed_results.append(
                {
                    "key": key,
                    "error": result.error,
                    "execution_time": result.execution_time,
                }
            )

        if result.execution_time:
            execution_times.append(result.execution_time)

    return {
        "total_results": len(parallel_results),
        "successful_count": len(successful_results),
        "failed_count": len(failed_results),
        "result_keys": list(parallel_results.keys()),
        "successful_results": successful_results,
        "failed_results": failed_results,
        "avg_execution_time": sum(execution_times) / len(execution_times)
        if execution_times
        else 0,
    }


# 求和聚合节点
@node
def sum_aggregator_node(
    parallel_results: dict[str, ParallelResult],
) -> StandardTestData:
    """求和聚合节点"""
    total = 0
    processed_count = 0
    error_count = 0

    for key, result in parallel_results.items():
        if result.success and result.result:
            total += result.result.value
            processed_count += 1
        else:
            error_count += 1

    return StandardTestData(
        value=total,
        name=f"aggregated_sum_{processed_count}_success_{error_count}_errors",
        metadata={
            "aggregated_from": list(parallel_results.keys()),
            "success_count": processed_count,
            "error_count": error_count,
        },
    )


# 便捷函数 - 用于创建配置化的节点实例
def create_processor_node(config: TestNodeConfig) -> Node:
    """根据配置创建标准处理节点"""
    from functools import partial

    # 使用partial创建配置化的节点函数
    configured_func = partial(standard_processor_node.func, config=config)
    configured_node = node(configured_func, name=config.name)
    return configured_node


def create_simple_processor_node(name: str, multiplier: int = 2) -> Node:
    """创建简单处理器节点"""
    from functools import partial

    configured_func = partial(simple_processor_node.func, multiplier=multiplier)
    configured_node = node(configured_func, name=name)
    return configured_node


def create_aggregator_node(name: str = "aggregator") -> Node:
    """创建聚合节点"""
    configured_node = node(aggregator_node.func, name=name)
    return configured_node


def create_sum_aggregator_node(name: str = "sum_aggregator") -> Node:
    """创建求和聚合节点"""
    configured_node = node(sum_aggregator_node.func, name=name)
    return configured_node


def create_test_node(
    func: Callable, name: str | None = None, use_decorator: bool = True
) -> Node:
    """
    创建测试节点的工厂函数

    Args:
        func: 节点函数
        name: 节点名称
        use_decorator: 是否使用@node装饰器（推荐为True）

    Returns:
        创建的节点

    Note:
        强烈推荐使用@node装饰器而不是手动创建Node！
        如果func使用依赖注入，必须设置use_decorator=True
    """
    if use_decorator:
        decorated_func = node(func)
        if name:
            decorated_func.name = name
        return decorated_func
    else:
        return Node(func, name=name or func.__name__)


def create_simple_processor(name: str, multiplier: int = 2) -> Node:
    """创建简单处理器节点的便捷函数"""
    config = TestNodeConfig(name=name, multiplier=multiplier)
    return create_processor_node(config)


def create_failing_processor(
    name: str, failure_rate: float = 1.0, multiplier: int = 2
) -> Node:
    """创建会失败的处理器节点的便捷函数"""
    config = TestNodeConfig(
        name=name, multiplier=multiplier, should_fail=True, failure_rate=failure_rate
    )
    return create_processor_node(config)


def create_slow_processor(
    name: str, delay_seconds: float = 0.1, multiplier: int = 2
) -> Node:
    """创建执行缓慢的处理器节点的便捷函数"""
    config = TestNodeConfig(
        name=name, multiplier=multiplier, delay_seconds=delay_seconds
    )
    return create_processor_node(config)


# 为了兼容现有测试，添加特定的节点创建函数
def create_simple_processor_node(name: str, multiplier: int = 2) -> Node:
    """创建简单处理器节点 - 兼容现有测试"""
    return create_simple_processor(name, multiplier)


def create_failing_node(name: str, failure_rate: float = 1.0) -> Node:
    """创建失败节点 - 兼容现有测试"""
    return create_failing_processor(name, failure_rate)


def create_slow_node(name: str, delay_seconds: float = 0.1) -> Node:
    """创建慢节点 - 兼容现有测试"""
    return create_slow_processor(name, delay_seconds)


# 兼容性工厂类
class StandardNodeFactory:
    """标准节点工厂 - 兼容现有测试"""

    @staticmethod
    def create_simple_processor_node(name: str, multiplier: int = 2) -> Node:
        """创建简单处理器节点"""
        return create_simple_processor_node(name, multiplier)

    @staticmethod
    def create_failing_node(name: str, failure_rate: float = 1.0) -> Node:
        """创建失败节点"""
        return create_failing_node(name, failure_rate)

    @staticmethod
    def create_slow_node(name: str, delay_seconds: float = 0.1) -> Node:
        """创建慢节点"""
        return create_slow_node(name, delay_seconds)


# 模块级的处理器类（支持pickle序列化，兼容现有测试）
class SimpleProcessor:
    """简单的数据处理器 - 兼容现有测试"""

    def __init__(self, name: str, multiplier: int = 2):
        self.name = name
        self.multiplier = multiplier

    def __call__(self, data: StandardTestData) -> StandardTestData:
        processed_value = data.value * self.multiplier
        return StandardTestData(
            value=processed_value,
            name=f"{self.name}_processed_{data.name}",
            metadata={"processor": self.name, "multiplier": self.multiplier},
        )


# 同步节点 - 供异步/同步混合测试复用
@node
def sync_subtract_5_node(value: int) -> int:
    """同步节点：减5"""
    return value - 5


@node
def sync_final_processing_node(value: int) -> str:
    """同步节点：最终处理"""
    return f"Final result: {value}"


# 异步节点 - 供异步/同步混合测试复用
@node
async def async_add_10_node(value: int) -> int:
    """异步节点：加10"""
    import asyncio

    await asyncio.sleep(0.01)  # 模拟异步操作
    return value + 10


@node
async def async_multiply_2_node(value: int) -> int:
    """异步节点：乘以2"""
    import asyncio

    await asyncio.sleep(0.01)  # 模拟异步操作
    return value * 2


@node
async def async_subtract_5_node(value: int) -> int:
    """异步节点：减5"""
    import asyncio

    await asyncio.sleep(0.01)  # 模拟异步操作
    return value - 5


@node
async def async_divide_3_node(value: int) -> int:
    """异步节点：除以3"""
    import asyncio

    await asyncio.sleep(0.01)  # 模拟异步操作
    return value // 3


@node
async def async_final_processing_node(value: int) -> str:
    """异步节点：最终处理"""
    import asyncio

    await asyncio.sleep(0.01)  # 模拟异步操作
    return f"Async final result: {value}"


@node
async def async_aggregator_node(parallel_results: dict) -> str:
    """异步聚合节点"""
    import asyncio

    await asyncio.sleep(0.01)  # 模拟异步操作
    from src.aetherflow import ParallelResult

    total = sum(
        result.result
        for result in parallel_results.values()
        if isinstance(result, ParallelResult) and result.success
    )
    return f"Async aggregated: {total}"


# 异步重试测试节点 - 复用于重试功能测试

# 全局计数器，用于测试重试次数
_async_retry_call_count = 0


@node(retry_count=2, retry_delay=0.01)
async def async_retry_success_node(value: int) -> int:
    """异步重试节点：前2次失败，第3次成功"""
    global _async_retry_call_count
    _async_retry_call_count += 1
    import asyncio

    await asyncio.sleep(0.01)
    if _async_retry_call_count < 3:
        raise ValueError(f"Async attempt {_async_retry_call_count} failed")
    result = value * 2
    _async_retry_call_count = 0  # 重置计数器
    return result


@node(retry_count=2, retry_delay=0.01)
async def async_always_fail_node(value: int) -> int:
    """异步重试节点：总是失败"""
    import asyncio

    await asyncio.sleep(0.01)
    raise ValueError(f"Async always fails with value: {value}")


# composition测试专用节点（最小新增集合）
@node
def simple_error_node(x: int) -> int:
    """简单错误节点：总是抛出异常"""
    raise ValueError(f"Simple error with input: {x}")


@node
def intermittent_error_node(x: int) -> int:
    """间歇性错误节点：当x能被3整除时抛出异常"""
    if x % 3 == 0:
        raise ValueError(f"Intermittent error at value: {x}")
    return x * 2


# 条件节点已移动到各自的测试文件中，使用依赖注入模式


# 导出列表
__all__ = [
    # 数学运算节点
    "multiply_by_2_node",
    "double_node",
    "square_node",
    "add_10_node",
    "add_1_node",
    "multiply_by_3_node",
    # 格式化节点
    "format_result_node",
    "format_final_node",
    "add_prefix_node",
    # 类型转换节点
    "to_string_node",
    "stringify_node",
    "string_length_node",
    # 业务逻辑节点
    "process_user_input_node",
    "generate_final_result_node",
    # 数据处理节点
    "process_numbers_node",
    "format_stats_node",
    # 类型验证节点
    "strict_int_processor_node",
    "strict_str_processor_node",
    "strict_final_processor_node",
    # 步骤节点
    "step_a_node",
    "step_b_node",
    "step_c_node",
    # 标准处理器节点
    "standard_processor_node",
    "simple_processor_node",
    "aggregator_node",
    "sum_aggregator_node",
    # 可序列化纯函数
    "success_processor_1_func",
    "success_processor_2_func",
    "failing_processor_1_func",
    "failing_processor_2_func",
    "slow_processor_1_func",
    # Node包装器
    "success_processor_1",
    "success_processor_2",
    "failing_processor_1",
    "failing_processor_2",
    "slow_processor_1",
    # 工厂函数
    "create_test_node",
    "create_simple_processor",
    "create_failing_processor",
    "create_slow_processor",
    "create_processor_node",
    "create_simple_processor_node",
    "create_aggregator_node",
    "create_sum_aggregator_node",
    # 兼容性函数
    "create_failing_node",
    "create_slow_node",
    "SimpleProcessor",
    # 同步节点 - 供异步/同步混合测试复用
    "sync_subtract_5_node",
    "sync_final_processing_node",
    # 异步节点 - 供异步/同步混合测试复用
    "async_add_10_node",
    "async_multiply_2_node",
    "async_subtract_5_node",
    "async_divide_3_node",
    "async_final_processing_node",
    "async_aggregator_node",
    # composition测试专用节点（仅repeat测试使用）
    "simple_error_node",
    "intermittent_error_node",
]
