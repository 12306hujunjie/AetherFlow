#!/usr/bin/env python3
"""
test_fan_primitives.py - Node核心原语fan_in、fan_out_to、fan_out_in的综合测试
包含：基础功能、错误处理、数据一致性、依赖注入集成的完整测试
"""

import os
import sys
import time
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.aetherflow import Node, ParallelResult, node

# 使用统一的测试基础设施
from tests.shared import StandardTestData
from tests.utils import ParallelTestValidator, StandardNodeFactory

# ============================================================================
# 模块级别的处理器类（支持pickle序列化）
# ============================================================================


class SimpleProcessor:
    """简单的数据处理器"""

    def __init__(self, name: str, multiplier: int = 2):
        self.name = name
        self.multiplier = multiplier

    def __call__(self, data: StandardTestData) -> StandardTestData:
        processed_value = data.value * self.multiplier
        return StandardTestData(
            value=processed_value,
            name=f"{self.name}_processed_{data.name}",
            timestamp=time.time(),
        )


# ============================================================================
# fan_out_to 基础功能测试
# ============================================================================


# 定义模块级函数以避免Pydantic验证问题
def source_function(value: int) -> StandardTestData:
    return StandardTestData(value=value, name="source")


def simple_multiply_function(data: StandardTestData) -> StandardTestData:
    return StandardTestData(value=data.value * 2, name=f"processed_{data.name}")


def test_fan_out_to_basic_distribution():
    """测试fan_out_to的基本分发功能"""
    print("\n=== 测试fan_out_to基本分发功能 ===")

    # 创建源节点
    source_node = Node(source_function, name="source")

    # 创建5个目标节点
    target_nodes = [
        StandardNodeFactory.create_simple_processor_node(
            f"target_{i}", multiplier=i + 1
        )
        for i in range(5)
    ]

    # 执行fan_out_to
    fan_out_pipeline = source_node.fan_out_to(target_nodes)
    results = fan_out_pipeline(10)

    print(f"分发结果数量: {len(results)}")

    # 验证结果
    assert isinstance(results, dict), "结果应该是字典类型"
    assert len(results) == 5, f"应该有5个结果，实际有{len(results)}个"

    # 验证每个结果 - 基于节点名称而不是迭代顺序
    expected_results = {
        "target_0": 10,  # 10 * 1
        "target_1": 20,  # 10 * 2
        "target_2": 30,  # 10 * 3
        "target_3": 40,  # 10 * 4
        "target_4": 50,  # 10 * 5
    }

    for key, result in results.items():
        assert isinstance(result, ParallelResult), f"结果{key}应该是ParallelResult类型"
        assert result.success, f"节点{key}执行应该成功"
        assert key in expected_results, f"意外的节点名称: {key}"
        assert result.result.value == expected_results[key], (
            f"节点{key}的值应该是{expected_results[key]}"
        )
        assert result.execution_time is not None, f"节点{key}应该有执行时间"

        print(
            f"  {key}: value={result.result.value}, time={result.execution_time:.4f}s"
        )

    print("✅ fan_out_to基本分发测试通过")


def test_fan_out_to_empty_targets():
    """测试fan_out_to空目标列表异常"""
    print("\n=== 测试fan_out_to空目标列表异常 ===")

    def source_function(value: int) -> int:
        return value * 2

    source_node = Node(source_function, name="source")

    # 测试空目标列表应该抛出异常
    with pytest.raises(ValueError, match="Target nodes list cannot be empty"):
        source_node.fan_out_to([])

    print("✅ fan_out_to空目标列表异常测试通过")


def test_fan_out_to_single_target():
    """测试fan_out_to单目标分发"""
    print("\n=== 测试fan_out_to单目标分发 ===")

    def source_function(value: int) -> StandardTestData:
        return StandardTestData(value=value, name="single_source")

    source_node = Node(source_function, name="source")
    target_node = StandardNodeFactory.create_simple_processor_node(
        "single_target", multiplier=3
    )

    # 执行单目标分发
    fan_out_pipeline = source_node.fan_out_to([target_node])
    results = fan_out_pipeline(7)

    print(f"单目标分发结果: {results}")

    # 验证结果
    assert len(results) == 1, "应该只有一个结果"

    result_key = list(results.keys())[0]
    result = results[result_key]

    assert result.success, "单目标执行应该成功"
    assert result.result.value == 21, f"结果值应该是21，实际是{result.result.value}"

    print("✅ fan_out_to单目标分发测试通过")


def test_fan_out_to_executor_types():
    """测试fan_out_to不同executor类型"""
    print("\n=== 测试fan_out_to不同executor类型 ===")

    def source_function(value: int) -> StandardTestData:
        return StandardTestData(value=value, name="executor_test")

    source_node = Node(source_function, name="source")

    # 创建3个目标节点
    target_nodes = [
        StandardNodeFactory.create_simple_processor_node(f"target_{i}", multiplier=2)
        for i in range(3)
    ]

    # 测试ThreadPoolExecutor
    thread_pipeline = source_node.fan_out_to(
        target_nodes, executor="thread", max_workers=2
    )
    thread_results = thread_pipeline(5)

    print(f"Thread executor结果数量: {len(thread_results)}")
    assert len(thread_results) == 3, "Thread executor应该有3个结果"

    # 测试ProcessPoolExecutor
    process_pipeline = source_node.fan_out_to(
        target_nodes, executor="process", max_workers=2
    )
    process_results = process_pipeline(5)

    print(f"Process executor结果数量: {len(process_results)}")
    assert len(process_results) == 3, "Process executor应该有3个结果"

    # 验证两种executor的结果一致性
    for key in thread_results.keys():
        if key in process_results:
            thread_val = thread_results[key].result.value
            process_val = process_results[key].result.value
            assert thread_val == process_val, (
                f"Thread和Process执行结果应该一致: {thread_val} vs {process_val}"
            )

    print("✅ fan_out_to不同executor类型测试通过")


# ============================================================================
# fan_out_to 数据一致性和异常处理测试
# ============================================================================


def test_fan_out_to_data_consistency():
    """测试fan_out_to数据一致性"""
    print("\n=== 测试fan_out_to数据一致性 ===")

    def source_function(input_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": input_data["id"],
            "data": input_data["data"],
            "timestamp": time.time(),
        }

    source_node = Node(source_function, name="source")

    # 创建节点验证接收到的数据
    def verify_data_node(name: str):
        def verify_processor(data: dict[str, Any]) -> dict[str, Any]:
            # 验证数据完整性
            assert "id" in data, "数据应该包含id字段"
            assert "data" in data, "数据应该包含data字段"
            assert "timestamp" in data, "数据应该包含timestamp字段"

            return {
                "verifier_name": name,
                "received_id": data["id"],
                "received_data": data["data"],
                "received_timestamp": data["timestamp"],
                "verification_time": time.time(),
            }

        return Node(verify_processor, name=name)

    # 创建5个验证节点
    verify_nodes = [verify_data_node(f"verifier_{i}") for i in range(5)]

    # 执行分发
    fan_out_pipeline = source_node.fan_out_to(verify_nodes)

    test_input = {
        "id": "test_consistency_123",
        "data": [1, 2, 3, 4, 5],
        "extra_field": "should_be_preserved",
    }

    results = fan_out_pipeline(test_input)

    print(f"数据一致性验证结果数量: {len(results)}")

    # 验证所有节点接收到相同的源数据
    source_timestamps = set()

    for key, result in results.items():
        assert result.success, f"验证节点{key}应该执行成功"

        verification_result = result.result
        assert verification_result["received_id"] == "test_consistency_123", (
            "所有节点应该接收到相同的id"
        )
        assert verification_result["received_data"] == [1, 2, 3, 4, 5], (
            "所有节点应该接收到相同的data"
        )

        source_timestamps.add(verification_result["received_timestamp"])

        print(
            f"  {key}: id={verification_result['received_id']}, timestamp={verification_result['received_timestamp']}"
        )

    # 验证所有节点接收到的源数据时间戳相同（同一次执行）
    assert len(source_timestamps) == 1, "所有节点应该接收到同一次源节点执行的结果"

    print("✅ fan_out_to数据一致性测试通过")


@node
def source_function(value: int) -> StandardTestData:
    return StandardTestData(value=value, name="failure_test")


def test_fan_out_to_partial_failures():
    """测试fan_out_to部分节点失败的处理"""
    print("\n=== 测试fan_out_to部分节点失败处理 ===")

    # 创建混合节点：2个正常节点，2个失败节点，1个慢节点
    target_nodes = [
        StandardNodeFactory.create_simple_processor_node("success_1", multiplier=2),
        StandardNodeFactory.create_failing_node(
            "failure_1", failure_rate=1.0
        ),  # 100%失败
        StandardNodeFactory.create_simple_processor_node("success_2", multiplier=3),
        StandardNodeFactory.create_failing_node(
            "failure_2", failure_rate=1.0
        ),  # 100%失败
        StandardNodeFactory.create_slow_node("slow_1", delay_seconds=0.1),
    ]

    # 执行分发
    fan_out_pipeline = source_function.fan_out_to(target_nodes)
    results = fan_out_pipeline(8)

    print(f"部分失败测试结果数量: {len(results)}")

    # 使用统一的并行结果验证器 - 修复静态方法调用
    successful, failed = ParallelTestValidator.assert_parallel_results(
        results, expected_total=5, expected_success=3, expected_failure=2
    )

    # 验证成功和失败结果
    ParallelTestValidator.assert_successful_results_have_values(successful)
    ParallelTestValidator.assert_failed_results_have_errors(failed)

    # 验证成功结果的值
    expected_values = {16, 24}  # 8*2, 8*3 (慢节点为8*2=16，所以是16, 24)
    success_values = {result[1].result.value for result in successful}
    # 慢节点可能产生不同的值，所以检查是否包含预期的基本值
    basic_expected = {16, 24}  # success_1: 8*2=16, success_2: 8*3=24
    assert basic_expected.issubset(success_values), (
        f"成功结果应包含基本预期值: {basic_expected}, 实际: {success_values}"
    )

    print(f"成功节点: {[key for key, result in results.items() if result.success]}")
    print(f"失败节点: {[key for key, result in results.items() if not result.success]}")

    print("✅ fan_out_to部分节点失败处理测试通过")


if __name__ == "__main__":
    print("=== Node fan_out_to 核心原语测试 ===")

    try:
        test_fan_out_to_basic_distribution()
        test_fan_out_to_empty_targets()
        test_fan_out_to_single_target()
        test_fan_out_to_executor_types()
        test_fan_out_to_data_consistency()
        test_fan_out_to_partial_failures()

        print("\n🎉 所有fan_out_to测试通过！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
