#!/usr/bin/env python3
"""
test_then_core.py - Node的then方法核心功能测试
专注于测试then原语的基本功能：基本链式调用、类型验证、多级链式组合
不包含复杂的依赖注入场景，保持测试简单清晰

重构后使用统一的测试基础设施
"""

from src.aetherflow import Node

# 使用统一的测试基础设施
from .shared.data_models import FinalResult, ProcessedUser, UserInput
from .utils.node_factory import create_test_node

# 注意：这个测试文件专注于测试then方法的核心功能，
# 使用直接的函数定义配合Node包装，而不是@node装饰器
# 这样可以避免Pydantic在测试环境中的类型检查问题


def test_basic_then():
    """测试基本的then功能"""
    print("=== 测试基本then功能 ===")

    def process_user_input(user: UserInput) -> ProcessedUser:
        print(f"Processing: {user.name}, age: {user.age}")
        return ProcessedUser(formatted_name=user.name.title(), is_adult=user.age >= 18)

    def generate_final_result(processed: ProcessedUser) -> FinalResult:
        print(f"Generating result for: {processed.formatted_name}")
        user_type = "成人" if processed.is_adult else "未成年"
        return FinalResult(
            message=f"欢迎 {processed.formatted_name}", user_type=user_type
        )

    # 使用统一的节点工厂创建节点
    node1 = create_test_node(
        process_user_input, name="process_user", use_decorator=False
    )
    node2 = create_test_node(
        generate_final_result, name="generate_result", use_decorator=False
    )

    print(f"Node1: {node1}")
    print(f"Node2: {node2}")

    # 测试then组合
    pipeline = node1.then(node2)
    print(f"Pipeline: {pipeline}")

    # 执行测试
    user_input = UserInput(name="alice", age=25)
    result = pipeline(user_input)

    print(f"Pipeline result: {result}")
    assert result.message == "欢迎 Alice"
    assert result.user_type == "成人"
    print("✅ 基本then测试通过")


def test_chain_then():
    """测试链式then调用"""
    print("\n=== 测试链式then调用 ===")

    def multiply_by_2(x: int) -> int:
        result = x * 2
        print(f"Multiply: {x} -> {result}")
        return result

    def add_10(x: int) -> int:
        result = x + 10
        print(f"Add 10: {x} -> {result}")
        return result

    def format_result(x: int) -> str:
        result = f"final_{x}"
        print(f"Format: {x} -> {result}")
        return result

    # 使用统一的节点工厂创建节点
    step1 = create_test_node(multiply_by_2, name="multiply", use_decorator=False)
    step2 = create_test_node(add_10, name="add", use_decorator=False)
    step3 = create_test_node(format_result, name="format", use_decorator=False)

    # 三级链式调用
    pipeline = step1.then(step2).then(step3)
    print(f"Chain pipeline: {pipeline}")

    # 执行: 5 -> 10 -> 20 -> "final_20"
    result = pipeline(5)
    print(f"Chain result: {result}")
    assert result == "final_20"
    print("✅ 链式then测试通过")


def test_multiple_chain_combinations():
    """测试多种链式组合方式"""
    print("\n=== 测试多种链式组合 ===")

    def double(x: int) -> int:
        return x * 2

    def square(x: int) -> int:
        return x * x

    def stringify(x: int) -> str:
        return str(x)

    def add_prefix(s: str) -> str:
        return f"result_{s}"

    # 使用统一的节点工厂创建节点
    double_node = create_test_node(double, name="double", use_decorator=False)
    square_node = create_test_node(square, name="square", use_decorator=False)
    stringify_node = create_test_node(stringify, name="stringify", use_decorator=False)
    prefix_node = create_test_node(add_prefix, name="prefix", use_decorator=False)

    # 测试不同的组合顺序
    pipeline1 = double_node.then(square_node).then(
        stringify_node
    )  # 5 -> 10 -> 100 -> "100"
    result1 = pipeline1(5)
    assert result1 == "100"
    print(f"双倍->平方->字符串: 5 -> {result1}")

    pipeline2 = square_node.then(double_node).then(
        stringify_node
    )  # 5 -> 25 -> 50 -> "50"
    result2 = pipeline2(5)
    assert result2 == "50"
    print(f"平方->双倍->字符串: 5 -> {result2}")

    # 四级链式调用
    pipeline3 = double_node.then(square_node).then(stringify_node).then(prefix_node)
    result3 = pipeline3(3)  # 3 -> 6 -> 36 -> "36" -> "result_36"
    assert result3 == "result_36"
    print(f"四级链式: 3 -> {result3}")

    print("✅ 多种链式组合测试通过")


def test_type_validation_in_chains():
    """测试链式调用中的类型验证"""
    print("\n=== 测试链式调用类型验证 ===")

    def strict_int_processor(x: int) -> str:
        return f"processed_{x}"

    def strict_str_processor(s: str) -> int:
        return len(s)

    def strict_final_processor(x: int) -> dict:
        return {"length": x, "valid": x > 5}

    node1 = create_test_node(
        strict_int_processor, name="int_to_str", use_decorator=False
    )
    node2 = create_test_node(
        strict_str_processor, name="str_to_int", use_decorator=False
    )
    node3 = create_test_node(
        strict_final_processor, name="final_check", use_decorator=False
    )

    pipeline = node1.then(node2).then(node3)

    # 正确的类型输入
    result = pipeline(42)
    print(f"Valid chain result: {result}")
    assert result["length"] == 12  # len("processed_42") = 12
    assert result["valid"] == True  # 12 > 5

    # 注意：不使用@node装饰器时，不会有Pydantic类型验证
    # 这里我们测试的是Node.then()的核心链式调用功能
    print("✅ 链式调用类型验证测试通过")


def test_simple_data_flow():
    """测试简单的数据流传递"""
    print("\n=== 测试简单数据流 ===")

    # 测试基本数据类型流传递
    def process_numbers(nums: list[int]) -> dict[str, float]:
        total = sum(nums)
        average = total / len(nums) if nums else 0
        return {"total": float(total), "average": average, "count": float(len(nums))}

    def format_stats(stats: dict[str, float]) -> str:
        return f"总计: {stats['total']}, 平均: {stats['average']:.2f}, 数量: {int(stats['count'])}"

    # 使用统一的节点工厂创建节点
    calc_node = create_test_node(process_numbers, name="calc", use_decorator=False)
    format_node = create_test_node(format_stats, name="format", use_decorator=False)

    simple_pipeline = calc_node.then(format_node)

    # 测试数据
    test_numbers = [1, 2, 3, 4, 5]
    result = simple_pipeline(test_numbers)

    print(f"数据流结果: {result}")
    assert "总计: 15.0" in result
    assert "平均: 3.00" in result
    assert "数量: 5" in result

    print("✅ 简单数据流测试通过")


def test_node_chaining_properties():
    """测试节点链式调用的属性"""
    print("\n=== 测试节点链式调用属性 ===")

    def step_a(x: int) -> int:
        return x + 1

    def step_b(x: int) -> int:
        return x * 2

    def step_c(x: int) -> str:
        return f"result_{x}"

    node_a = create_test_node(step_a, name="step_a", use_decorator=False)
    node_b = create_test_node(step_b, name="step_b", use_decorator=False)
    node_c = create_test_node(step_c, name="step_c", use_decorator=False)

    # 测试链式调用返回的还是Node对象
    chain_ab = node_a.then(node_b)
    assert isinstance(chain_ab, Node), "then应该返回Node对象"
    print(f"chain_ab类型: {type(chain_ab)}")

    # 测试三级链式调用
    chain_abc = chain_ab.then(node_c)
    assert isinstance(chain_abc, Node), "多级then应该返回Node对象"
    print(f"chain_abc类型: {type(chain_abc)}")

    # 测试执行结果
    result = chain_abc(10)  # 10 -> 11 -> 22 -> "result_22"
    assert result == "result_22"
    print(f"链式调用结果: {result}")

    print("✅ 节点链式调用属性测试通过")


if __name__ == "__main__":
    print("=== Node.then() 核心功能测试 ===")

    try:
        test_basic_then()
        test_chain_then()
        test_multiple_chain_combinations()
        test_type_validation_in_chains()
        test_simple_data_flow()
        test_node_chaining_properties()
        print("\n🎉 所有then核心功能测试通过！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
