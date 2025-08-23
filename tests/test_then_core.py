#!/usr/bin/env python3
"""
test_then_core.py - Node的then方法核心功能测试
专注于测试then原语的基本功能：基本链式调用、类型验证、多级链式组合
使用@node装饰器定义模块级节点函数，支持pickle序列化

重构后使用@node装饰器模式
"""

# 使用统一的测试基础设施
from .shared.data_models import FinalResult, UserInput
from .utils.node_factory import (
    add_10_node,
    double_node,
    format_result_node,
    format_stats_node,
    generate_final_result_node,
    multiply_by_2_node,
    process_numbers_node,
    process_user_input_node,
    square_node,
    step_a_node,
    step_b_node,
    step_c_node,
    strict_final_processor_node,
    strict_int_processor_node,
    strict_str_processor_node,
    stringify_node,
)


def test_basic_then():
    """测试基本的then功能和链式调用"""
    print("\n=== 测试基本then功能 ===")

    # 测试业务逻辑链式调用
    user_pipeline = process_user_input_node.then(generate_final_result_node)
    user_input = UserInput(name="Alice", age=25)
    result = user_pipeline(user_input)

    assert isinstance(result, FinalResult)
    assert result.message == "欢迎 Alice"
    assert result.user_type == "成人"
    print(f"业务逻辑链: {user_input.name} -> {result.message}")

    # 测试数学运算链式调用 (合并原test_chain_then)
    math_pipeline = multiply_by_2_node.then(add_10_node).then(format_result_node)
    math_result = math_pipeline(5)  # 5 -> 10 -> 20 -> "结果: 20"
    assert math_result == "结果: 20"
    print(f"数学运算链: 5 -> {math_result}")

    print("✅ 基本then测试通过")


def test_chain_combinations():
    """测试多种链式组合方式 (精简版)"""
    print("\n=== 测试链式组合 ===")

    # 测试核心组合: 双倍->平方->字符串化
    pipeline = double_node.then(square_node).then(stringify_node)
    result = pipeline(5)  # 5 -> 10 -> 100 -> "Number: 100"
    assert result == "Number: 100"
    print(f"组合链式: 5 -> {result}")

    # 测试四级链式调用
    extended_pipeline = step_a_node.then(step_b_node).then(step_c_node)
    extended_result = extended_pipeline(3)  # 3 -> 4 -> 8 -> "final_8"
    assert extended_result == "final_8"
    print(f"步骤链式: 3 -> {extended_result}")

    print("✅ 链式组合测试通过")


def test_advanced_chains():
    """测试高级链式调用功能 (合并类型验证、数据流和属性测试)"""
    print("\n=== 测试高级链式功能 ===")

    # 1. 类型验证链式调用
    type_pipeline = strict_int_processor_node.then(strict_str_processor_node).then(
        strict_final_processor_node
    )
    type_result = type_pipeline(42)
    assert type_result["length"] == 12  # len("processed_42") = 12
    assert type_result["valid"] == True  # 12 > 5
    print(f"类型验证链: 42 -> {type_result}")

    # 2. 数据流处理链式调用
    data_pipeline = process_numbers_node.then(format_stats_node)
    test_numbers = [1, 2, 3, 4, 5]
    data_result = data_pipeline(test_numbers)
    assert "总和=15.0" in data_result
    assert "平均=3.00" in data_result
    assert "数量=5" in data_result
    print(f"数据流链: {test_numbers} -> {data_result}")

    # 3. 链式调用组合性测试
    pipeline1 = step_a_node.then(step_b_node).then(step_c_node)
    pipeline2 = step_a_node.then(step_b_node.then(step_c_node))

    test_input = 10
    result1 = pipeline1(test_input)
    result2 = pipeline2(test_input)
    assert result1 == result2 == "final_22"  # (10+1)*2 = 22
    print(f"组合性验证: {test_input} -> {result1}")

    print("✅ 高级链式功能测试通过")


if __name__ == "__main__":
    print("=== Node.then() 核心功能测试 ===")

    try:
        test_basic_then()
        test_chain_combinations()
        test_advanced_chains()
        print("\n🎉 所有then核心功能测试通过！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
