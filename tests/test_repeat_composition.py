#!/usr/bin/env python3
"""
test_repeat_composition.py - repeat_composition重复执行功能专项测试

专门测试repeat_composition的核心功能，包括：
1. 基本重复执行功能
2. 不同次数的重复行为
3. 边界条件处理
4. 错误处理策略(stop_on_error=True/False)
5. 数据流在迭代间的传递
6. 性能和稳定性验证

测试原则：
- 最大化复用现有测试节点
- 专注于repeat_composition核心逻辑
- 全面覆盖正向、边界、异常场景
"""

import time
from typing import Any

import pytest

from src.aetherflow import RepeatStopException, node, repeat_composition

# 使用统一的测试基础设施
from .shared.test_constants import ASSERTION_MESSAGES

# 复用现有测试节点
from .utils.node_factory import (
    add_1_node,
    intermittent_error_node,
    multiply_by_2_node,
    simple_error_node,
)


class TestRepeatComposition:
    """repeat_composition专项功能测试"""

    def test_basic_repeat_functionality(self):
        """测试基本重复执行功能"""
        print("\n=== 测试基本重复执行功能 ===")

        # 使用add_1_node进行递增测试
        repeat_node = repeat_composition(add_1_node, times=3)
        result = repeat_node(5)  # 5 -> 6 -> 7 -> 8

        assert result == 8, ASSERTION_MESSAGES["value_mismatch"].format(
            expected=8, actual=result
        )
        assert repeat_node.name == f"({add_1_node.name} * 3)"
        print(f"✅ 重复3次递增: 5 -> {result}")

    def test_repeat_times_variations(self):
        """测试不同重复次数的行为"""
        print("\n=== 测试不同重复次数 ===")

        test_cases = [
            (1, 6),  # 5 + 1 = 6 (执行1次)
            (2, 7),  # 5 -> 6 -> 7 (执行2次)
            (4, 9),  # 5 -> 6 -> 7 -> 8 -> 9 (执行4次)
            (10, 15),  # 执行10次递增
        ]

        for times, expected in test_cases:
            repeat_node = repeat_composition(add_1_node, times=times)
            result = repeat_node(5)
            assert result == expected, ASSERTION_MESSAGES["value_mismatch"].format(
                expected=expected, actual=result
            )
            print(f"✅ 重复{times}次: 5 -> {result}")

    def test_boundary_conditions(self):
        """测试边界条件"""
        print("\n=== 测试边界条件 ===")

        # 测试最小有效次数 times=1
        repeat_node = repeat_composition(multiply_by_2_node, times=1)
        result = repeat_node(10)
        assert result == 20, "times=1结果错误"
        print(f"✅ times=1: 10 -> {result}")

        # 测试无效次数 times<=0
        with pytest.raises(ValueError, match="Repeat times must be greater than 0"):
            invalid_node = repeat_composition(add_1_node, times=0)
            invalid_node(10)

        with pytest.raises(ValueError, match="Repeat times must be greater than 0"):
            invalid_node = repeat_composition(add_1_node, times=-1)
            invalid_node(10)

        print("✅ 无效times参数正确抛出异常")

    def test_data_flow_between_iterations(self):
        """测试数据在迭代间的正确传递"""
        print("\n=== 测试迭代间数据流 ===")

        # 使用倍增节点测试数据累积效应
        repeat_node = repeat_composition(multiply_by_2_node, times=3)
        result = repeat_node(2)  # 2 -> 4 -> 8 -> 16

        assert result == 16, f"数据流测试失败: 期望16，实际{result}"
        print(f"✅ 数据累积: 2 -> 4 -> 8 -> 16 = {result}")

        # 测试类型转换的数据流
        @node
        def append_x_node(s: str) -> str:
            """追加字符x的节点"""
            return s + "x"

        string_repeat = repeat_composition(append_x_node, times=4)
        string_result = string_repeat(
            "start"
        )  # "start" -> "startx" -> "startxx" -> "startxxx" -> "startxxxx"

        assert string_result == "startxxxx", f"字符串累积错误: {string_result}"
        print(f"✅ 字符串累积: start -> {string_result}")

    def test_error_handling_stop_on_error_true(self):
        """测试stop_on_error=True的错误处理"""
        print("\n=== 测试stop_on_error=True ===")

        # 使用总是失败的节点
        repeat_node = repeat_composition(simple_error_node, times=3, stop_on_error=True)

        with pytest.raises(RepeatStopException, match="Execution stopped due to error"):
            repeat_node(5)
        print("✅ 总是失败节点正确抛出RepeatStopException")

        # 使用间歇性失败节点
        repeat_intermittent = repeat_composition(
            intermittent_error_node, times=5, stop_on_error=True
        )

        # 输入3会在第一次迭代就失败(3能被3整除)
        with pytest.raises(RepeatStopException, match="Execution stopped due to error"):
            repeat_intermittent(3)

        # 输入2正常执行: 2*2=4, 4*2=8, 8*2=16, 16*2=32, 32*2=64
        result_success = repeat_intermittent(2)
        assert result_success == 64, (
            f"间歇性错误正常执行失败: 期望64，实际{result_success}"
        )
        print(f"✅ 间歇性错误处理: 2 -> {result_success}")

    def test_error_handling_stop_on_error_false(self):
        """测试stop_on_error=False的错误处理"""
        print("\n=== 测试stop_on_error=False ===")

        # 总是失败的情况
        repeat_node = repeat_composition(
            simple_error_node, times=3, stop_on_error=False
        )
        result = repeat_node(5)

        # 因为所有迭代都失败，应该返回None（没有成功的结果）
        assert result is None, "连续失败应返回None"
        print("✅ 连续失败返回None")

        # 混合成功失败的情况
        @node
        def conditional_error_node(x: int) -> int:
            """条件错误节点：x大于10时失败"""
            if x > 10:
                raise ValueError(f"Value too large: {x}")
            return x + 2

        mixed_repeat = repeat_composition(
            conditional_error_node, times=4, stop_on_error=False
        )
        result_mixed = mixed_repeat(5)  # 5->7->9->11(失败)->继续用11

        # 最后一次成功的结果应该是11，即使第4次失败了，但11是第3次的结果
        assert result_mixed == 11, f"混合场景结果错误: 期望11，实际{result_mixed}"
        print(f"✅ 混合成功失败场景: 5 -> {result_mixed}")

    def test_complex_data_type_handling(self):
        """测试复杂数据类型的重复处理"""
        print("\n=== 测试复杂数据类型处理 ===")

        @node
        def dict_accumulator_node(data: dict[str, Any]) -> dict[str, Any]:
            """字典累加器节点"""
            return {
                "value": data.get("value", 0) + 1,
                "count": data.get("count", 0) + 1,
                "history": data.get("history", []) + [data.get("value", 0)],
            }

        repeat_dict = repeat_composition(dict_accumulator_node, times=3)
        initial_data = {"value": 5, "count": 0, "history": []}
        result = repeat_dict(initial_data)

        # 验证字典累积效果
        assert result["value"] == 8, f"字典value累积错误: {result['value']}"
        assert result["count"] == 3, f"字典count累积错误: {result['count']}"
        assert result["history"] == [5, 6, 7], f"字典history错误: {result['history']}"
        print(f"✅ 字典累积: {initial_data} -> {result}")

    def test_performance_large_iterations(self):
        """测试大次数重复的性能表现"""
        print("\n=== 测试大次数重复性能 ===")

        # 测试1000次重复的性能
        large_repeat = repeat_composition(add_1_node, times=1000)

        start_time = time.time()
        result = large_repeat(0)
        execution_time = time.time() - start_time

        assert result == 1000, f"大次数重复结果错误: 期望1000，实际{result}"
        assert execution_time < 2.0, f"大次数重复耗时过长: {execution_time:.3f}秒"
        print(f"✅ 1000次重复: 0 -> {result}, 耗时{execution_time:.3f}秒")

        # 测试更大次数但简单操作
        very_large_repeat = repeat_composition(add_1_node, times=5000)
        start_time = time.time()
        result_large = very_large_repeat(0)
        execution_time_large = time.time() - start_time

        assert result_large == 5000, f"超大次数结果错误: {result_large}"
        assert execution_time_large < 5.0, (
            f"超大次数耗时过长: {execution_time_large:.3f}秒"
        )
        print(f"✅ 5000次重复: 0 -> {result_large}, 耗时{execution_time_large:.3f}秒")

    def test_node_name_generation(self):
        """测试重复节点名称生成"""
        print("\n=== 测试节点名称生成 ===")

        repeat_node = repeat_composition(add_1_node, times=5)
        expected_name = f"({add_1_node.name} * 5)"

        assert repeat_node.name == expected_name, f"节点名称错误: {repeat_node.name}"
        print(f"✅ 节点名称: {repeat_node.name}")

    def test_single_iteration_equivalence(self):
        """测试单次重复与原节点的等价性"""
        print("\n=== 测试单次重复等价性 ===")

        # times=1应该等价于直接调用原节点
        repeat_once = repeat_composition(multiply_by_2_node, times=1)
        original_result = multiply_by_2_node(15)
        repeat_result = repeat_once(15)

        assert original_result == repeat_result, (
            f"单次重复不等价: 原始{original_result} vs 重复{repeat_result}"
        )
        print(f"✅ 单次重复等价性: {original_result} == {repeat_result}")

    def test_error_accumulation_logging(self):
        """测试错误累积和日志记录"""
        print("\n=== 测试错误累积 ===")

        # 这个测试主要验证stop_on_error=False时错误被正确收集
        # 通过检查执行是否能够完成来间接验证
        repeat_node = repeat_composition(
            simple_error_node, times=2, stop_on_error=False
        )

        # 应该能正常执行完成（尽管所有迭代都失败）
        try:
            result = repeat_node(10)
            assert result is None, "错误累积测试：应返回None"
            print("✅ 错误累积正确处理")
        except Exception as e:
            pytest.fail(f"错误累积处理失败: {e}")


if __name__ == "__main__":
    print("=== repeat_composition专项功能测试 ===")

    try:
        test_instance = TestRepeatComposition()

        # 执行所有测试方法
        test_instance.test_basic_repeat_functionality()
        test_instance.test_repeat_times_variations()
        test_instance.test_boundary_conditions()
        test_instance.test_data_flow_between_iterations()
        test_instance.test_error_handling_stop_on_error_true()
        test_instance.test_error_handling_stop_on_error_false()
        test_instance.test_complex_data_type_handling()
        test_instance.test_performance_large_iterations()
        test_instance.test_node_name_generation()
        test_instance.test_single_iteration_equivalence()
        test_instance.test_error_accumulation_logging()

        print("\n🎉 所有repeat_composition测试通过！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
