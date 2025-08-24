#!/usr/bin/env python3
"""
test_conditional_composition.py - conditional_composition条件分支功能专项测试

使用正确的链式API模式测试conditional_composition：
- 使用 condition_node.branch_on(branches) 链式调用
- 依赖注入状态管理的数据传递
- 专注于conditional_composition核心逻辑验证
"""

from typing import Any

import pytest
from dependency_injector.wiring import Provide

from src.aetherflow import (
    BaseFlowContext,
    NodeExecutionException,
    node,
)

from .shared.test_constants import ASSERTION_MESSAGES

# 复用错误节点


# 专用于conditional_composition的无参数错误节点
@node
def parameter_free_error_node(state: dict = Provide[BaseFlowContext.state]) -> int:
    """无参数错误节点：用于分支测试"""
    input_val = state.get("current_input", 0)
    raise ValueError(f"Branch error with input: {input_val}")


# 依赖注入模式的测试节点定义
@node
def boolean_condition_node(
    x: int, state: dict = Provide[BaseFlowContext.state]
) -> bool:
    """布尔条件节点：存储输入，返回偶数/奇数判断"""
    state["current_input"] = x
    return x % 2 == 0


@node
def string_condition_node(x: int, state: dict = Provide[BaseFlowContext.state]) -> str:
    """字符串条件节点：存储输入，返回范围分类"""
    state["current_input"] = x
    if x < 0:
        return "negative"
    elif x == 0:
        return "zero"
    elif x < 10:
        return "small"
    else:
        return "large"


@node
def multiply_branch_node(state: dict = Provide[BaseFlowContext.state]) -> int:
    """乘法分支：从状态读取数据并乘以2"""
    input_val = state.get("current_input", 0)
    return input_val * 2


@node
def add_branch_node(state: dict = Provide[BaseFlowContext.state]) -> int:
    """加法分支：从状态读取数据并加1"""
    input_val = state.get("current_input", 0)
    return input_val + 1


@node
def format_negative_node(state: dict = Provide[BaseFlowContext.state]) -> str:
    """格式化负数分支"""
    input_val = state.get("current_input", 0)
    return f"negative: {abs(input_val)}"


@node
def format_zero_node(state: dict = Provide[BaseFlowContext.state]) -> str:
    """格式化零值分支"""
    return "zero value"


@node
def format_small_node(state: dict = Provide[BaseFlowContext.state]) -> str:
    """格式化小数分支"""
    input_val = state.get("current_input", 0)
    return f"small: {input_val}"


@node
def format_large_node(state: dict = Provide[BaseFlowContext.state]) -> str:
    """格式化大数分支"""
    input_val = state.get("current_input", 0)
    return f"large: {input_val}"


@node
def error_condition_node(x: int, state: dict = Provide[BaseFlowContext.state]) -> str:
    """错误条件节点：总是失败"""
    state["current_input"] = x
    raise ValueError(f"Condition error with input: {x}")


# None条件处理节点
@node
def none_condition_node(x: int, state: dict = Provide[BaseFlowContext.state]) -> Any:
    """返回None或其他值的条件节点"""
    state["current_input"] = x
    if x == 0:
        return None
    elif x > 0:
        return "positive"
    else:
        return "negative"


@node
def none_handler_node(state: dict = Provide[BaseFlowContext.state]) -> str:
    """处理None条件"""
    input_val = state.get("current_input", 0)
    return f"none_case: {input_val}"


@node
def positive_handler_node(state: dict = Provide[BaseFlowContext.state]) -> str:
    """处理positive条件"""
    input_val = state.get("current_input", 0)
    return f"positive_case: {input_val}"


@node
def negative_handler_node(state: dict = Provide[BaseFlowContext.state]) -> str:
    """处理negative条件"""
    input_val = state.get("current_input", 0)
    return f"negative_case: {input_val}"


# 链式条件流节点
@node
def result_condition_node(x: Any, state: dict = Provide[BaseFlowContext.state]) -> str:
    """根据第一层结果进行分类"""
    state["current_input"] = x
    if isinstance(x, str):
        return "string_result"
    else:
        return "numeric_result"


@node
def string_final_node(state: dict = Provide[BaseFlowContext.state]) -> str:
    """处理字符串结果"""
    input_val = state.get("current_input", "")
    return f"final: {input_val}"


@node
def numeric_final_node(state: dict = Provide[BaseFlowContext.state]) -> str:
    """处理数值结果"""
    input_val = state.get("current_input", 0)
    return f"final_num: {input_val}"


class TestConditionalComposition:
    """conditional_composition链式API测试"""

    @pytest.fixture(autouse=True)
    def setup_injection(self, wired_container):
        """自动设置依赖注入"""
        self.container = wired_container(__name__)
        yield
        self.container.unwire()

    def test_basic_boolean_branching(self):
        """测试基本布尔条件分支"""
        print("\n=== 测试基本布尔条件分支 ===")

        branches = {
            True: multiply_branch_node,  # 偶数 -> 乘以2
            False: add_branch_node,  # 奇数 -> 加1
        }

        # 使用链式API
        branching_flow = boolean_condition_node.branch_on(branches)

        # 测试偶数路径
        result_even = branching_flow(4)  # 4(偶数) -> True -> 4*2 = 8
        assert result_even == 8, ASSERTION_MESSAGES["value_mismatch"].format(
            expected=8, actual=result_even
        )

        # 测试奇数路径
        result_odd = branching_flow(3)  # 3(奇数) -> False -> 3+1 = 4
        assert result_odd == 4, ASSERTION_MESSAGES["value_mismatch"].format(
            expected=4, actual=result_odd
        )

        print(f"✅ 布尔分支: 4(偶) -> {result_even}, 3(奇) -> {result_odd}")

    def test_string_condition_branching(self):
        """测试字符串条件分支"""
        print("\n=== 测试字符串条件分支 ===")

        branches = {
            "negative": format_negative_node,
            "zero": format_zero_node,
            "small": format_small_node,
            "large": format_large_node,
        }

        # 使用链式API
        branching_flow = string_condition_node.branch_on(branches)

        test_cases = [
            (-5, "negative: 5"),
            (0, "zero value"),
            (5, "small: 5"),
            (15, "large: 15"),
        ]

        for input_val, expected in test_cases:
            result = branching_flow(input_val)
            assert result == expected, ASSERTION_MESSAGES["value_mismatch"].format(
                expected=expected, actual=result
            )
            print(f"✅ 字符串条件 {input_val} -> {result}")

    def test_condition_not_found_error(self):
        """测试条件未匹配时的错误"""
        print("\n=== 测试条件未匹配错误 ===")

        # 只定义部分分支
        partial_branches = {
            "small": format_small_node,
            "large": format_large_node,
            # 故意缺少 "negative" 和 "zero"
        }

        branching_flow = string_condition_node.branch_on(partial_branches)

        # 存在的分支应该正常工作
        result_small = branching_flow(5)  # "small"
        assert result_small == "small: 5"

        # 不存在的分支应该抛出ValueError（composition函数直接抛出）
        with pytest.raises(
            ValueError, match="No branch defined for condition result: negative"
        ):
            branching_flow(-5)  # "negative" 分支不存在

        with pytest.raises(
            ValueError, match="No branch defined for condition result: zero"
        ):
            branching_flow(0)  # "zero" 分支不存在

        print("✅ 条件未匹配时正确抛出异常")

    def test_condition_node_failure(self):
        """测试条件节点执行失败"""
        print("\n=== 测试条件节点失败 ===")

        branches = {"any": format_small_node}

        # 使用会失败的条件节点
        failing_flow = error_condition_node.branch_on(branches)

        # 条件节点失败应该传播异常
        with pytest.raises(
            NodeExecutionException,
            match="节点执行失败，异常类型不支持重试: ValueError",
        ):
            failing_flow(10)

        print("✅ 条件节点失败正确传播异常")

    def test_branch_node_failure(self):
        """测试分支节点执行失败"""
        print("\n=== 测试分支节点失败 ===")

        branches = {
            True: parameter_free_error_node,  # True分支会失败
            False: add_branch_node,  # False分支正常
        }

        branching_flow = boolean_condition_node.branch_on(branches)

        # 偶数 -> True分支 -> parameter_free_error_node失败
        with pytest.raises(
            NodeExecutionException,
            match="节点执行失败，异常类型不支持重试: ValueError",
        ):
            branching_flow(4)

        # 奇数 -> False分支 -> add_branch_node正常
        result = branching_flow(3)
        assert result == 4

        print("✅ 分支节点失败正确处理")

    def test_none_condition_handling(self):
        """测试None条件值处理"""
        print("\n=== 测试None条件处理 ===")

        branches = {
            None: none_handler_node,
            "positive": positive_handler_node,
            "negative": negative_handler_node,
        }

        branching_flow = none_condition_node.branch_on(branches)

        # 测试各种条件值
        result_none = branching_flow(0)  # None条件
        result_positive = branching_flow(5)  # "positive"条件
        result_negative = branching_flow(-3)  # "negative"条件

        assert result_none == "none_case: 0"
        assert result_positive == "positive_case: 5"
        assert result_negative == "negative_case: -3"

        print(
            f"✅ None条件处理: 0->{result_none}, 5->{result_positive}, -3->{result_negative}"
        )

    def test_chained_conditional_flows(self):
        """测试链式条件流组合"""
        print("\n=== 测试链式条件流 ===")

        # 第一层分支：数值范围分类
        first_branches = {
            "small": format_small_node,  # 返回 "small: x"
            "large": multiply_branch_node,  # 返回 x*2
        }

        # 第二层分支：处理第一层的结果

        second_branches = {
            "string_result": string_final_node,
            "numeric_result": numeric_final_node,
        }

        # 构建链式条件流
        chained_flow = (
            string_condition_node.branch_on(first_branches)
            .then(result_condition_node)
            .branch_on(second_branches)
        )

        # 测试小数路径: 5 -> "small" -> "small: 5" -> "string_result" -> "final: small: 5"
        result_small = chained_flow(5)
        assert result_small == "final: small: 5"

        # 测试大数路径: 15 -> "large" -> 30 -> "numeric_result" -> "final_num: 30"
        result_large = chained_flow(15)
        assert result_large == "final_num: 30"

        print(f"✅ 链式条件流: 5 -> {result_small}, 15 -> {result_large}")


if __name__ == "__main__":
    print("=== conditional_composition链式API测试 ===")

    # 配置依赖注入容器
    container = BaseFlowContext()
    container.wire(modules=[__name__])

    try:
        test_instance = TestConditionalComposition()

        test_instance.test_basic_boolean_branching()
        test_instance.test_string_condition_branching()
        test_instance.test_condition_not_found_error()
        test_instance.test_condition_node_failure()
        test_instance.test_branch_node_failure()
        test_instance.test_none_condition_handling()
        test_instance.test_chained_conditional_flows()

        print("\n🎉 所有conditional_composition链式API测试通过！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()

    finally:
        container.unwire()
