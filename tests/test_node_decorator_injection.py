#!/usr/bin/env python3
"""
test_node_decorator_injection.py - 专门测试@node装饰器的依赖注入功能
包含：@node装饰器基本功能、依赖注入链式调用、state/context传递、多线程隔离等完整测试
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dependency_injector.wiring import Provide

from src.aetherflow import BaseFlowContext, Node, node
from tests.fixtures.injection_helpers import setup_test_container

# 使用统一的测试基础设施
from tests.shared import ProcessedTestData, TestUserData


# 模块级别的@node装饰器函数，用于测试依赖注入
def process_step1(data: dict) -> dict:
    """第一步处理：简单的数据变换，不使用依赖注入"""
    result = data["value"] * 2
    print(f"Step1: {data['value']} -> {result}")
    return {"processed": result, "step": "step1"}


def process_step2(data: dict) -> dict:
    """第二步处理：累加处理，不使用依赖注入"""
    result = data["processed"] + 10
    print(f"Step2: {data['processed']} -> {result}")
    return {"final": result, "step": "step2"}


def simple_transform(x: int) -> int:
    """简单的数值变换函数"""
    return x * 3


def test_node_decorator_basic_chain():
    """测试@node装饰器的基本链式调用"""
    print("\n=== 测试@node装饰器基本链式调用 ===")

    # 使用@node装饰器创建节点
    @node
    def step1(data: dict) -> dict:
        result = data["input"] * 2
        print(f"Node step1: {data['input']} -> {result}")
        return {"output": result}

    @node
    def step2(data: dict) -> dict:
        result = data["output"] + 5
        print(f"Node step2: {data['output']} -> {result}")
        return {"final": result}

    # 创建then链
    chain = step1.then(step2)

    # 测试执行
    result = chain({"input": 10})  # 10 * 2 + 5 = 25
    print(f"链式调用结果: {result}")
    assert result["final"] == 25
    print("✅ @node装饰器基本链式调用测试通过")


def test_node_decorator_vs_manual_node():
    """测试@node装饰器与手动Node创建的兼容性"""
    print("\n=== 测试@node装饰器与手动Node兼容性 ===")

    # @node装饰器创建的节点
    @node
    def decorated_node(x: int) -> int:
        result = x * 2
        print(f"装饰器节点: {x} -> {result}")
        return result

    # 手动创建的节点（不使用依赖注入）
    manual_node = Node(simple_transform, name="manual")

    # 混合链式调用：@node -> Manual Node
    chain1 = decorated_node.then(manual_node)
    result1 = chain1(5)  # 5 * 2 * 3 = 30
    print(f"@node -> Manual: {result1}")
    assert result1 == 30

    # 混合链式调用：Manual Node -> @node
    chain2 = manual_node.then(decorated_node)
    result2 = chain2(4)  # 4 * 3 * 2 = 24
    print(f"Manual -> @node: {result2}")
    assert result2 == 24

    print("✅ @node装饰器与手动Node兼容性测试通过")


def test_node_decorator_multiple_chains():
    """测试@node装饰器的多级链式调用"""
    print("\n=== 测试@node装饰器多级链式调用 ===")

    @node
    def multiply_by_2(x: int) -> int:
        result = x * 2
        print(f"multiply_by_2: {x} -> {result}")
        return result

    @node
    def add_10(x: int) -> int:
        result = x + 10
        print(f"add_10: {x} -> {result}")
        return result

    @node
    def divide_by_3(x: int) -> int:
        result = x // 3
        print(f"divide_by_3: {x} -> {result}")
        return result

    # 创建多级链式调用
    chain = multiply_by_2.then(add_10).then(divide_by_3)

    # 测试执行：6 * 2 + 10 / 3 = 22 / 3 = 7
    result = chain(6)
    print(f"多级链式结果: {result}")
    assert result == 7

    print("✅ @node装饰器多级链式调用测试通过")


def test_node_decorator_error_handling():
    """测试@node装饰器的错误处理"""
    print("\n=== 测试@node装饰器错误处理 ===")

    # 禁用重试以测试原始异常处理
    @node(enable_retry=False)
    def safe_divide(data: dict) -> dict:
        if data["denominator"] == 0:
            raise ValueError("Division by zero")
        result = data["numerator"] / data["denominator"]
        return {"result": result}

    @node(enable_retry=False)
    def format_result(data: dict) -> str:
        return f"结果: {data['result']:.2f}"

    # 创建链式调用
    chain = safe_divide.then(format_result)

    # 测试正常情况
    result1 = chain({"numerator": 10, "denominator": 2})
    print(f"正常计算: {result1}")
    assert result1 == "结果: 5.00"

    # 测试错误情况
    try:
        result2 = chain({"numerator": 10, "denominator": 0})
        assert False, "应该抛出除零错误"
    except ValueError as e:
        print(f"✅ 正确捕获错误: {e}")

    print("✅ @node装饰器错误处理测试通过")


def test_node_decorator_type_validation():
    """测试@node装饰器的类型验证"""
    print("\n=== 测试@node装饰器类型验证 ===")

    @node
    def strict_int_input(x: int) -> str:
        return f"整数: {x}"

    @node
    def strict_str_input(s: str) -> int:
        return len(s)

    # 测试正确类型
    result1 = strict_int_input(42)
    print(f"正确整数输入: {result1}")
    assert result1 == "整数: 42"

    # 测试类型链式传递
    chain = strict_int_input.then(strict_str_input)
    result2 = chain(123)  # 123 -> "整数: 123" -> 7
    print(f"类型链式传递: {result2}")
    assert result2 == 7  # len("整数: 123") = 7

    print("✅ @node装饰器类型验证测试通过")


# 模块级依赖注入函数 - 用于复杂场景测试
@node
def process_input_with_injection(
    data: dict, state: dict = Provide[BaseFlowContext.state]
) -> dict:
    processed_value = data["input"] * 2
    state["step1_processed"] = processed_value
    state["step1_timestamp"] = "2024-01-01"
    print(f"Step1: {data['input']} -> {processed_value} (stored in state)")
    return {"processed": processed_value, "original": data["input"]}


@node
def enhance_data_with_injection(
    data: dict, context: BaseFlowContext = Provide[BaseFlowContext]
) -> dict:
    state = context.state()
    # 读取前一步的state
    step1_value = state.get("step1_processed", 0)
    enhanced = step1_value + data["processed"] + 10
    state["step2_enhanced"] = enhanced
    state["step2_source"] = f"from_step1_{step1_value}"
    print(f"Step2: enhanced {enhanced} using state value {step1_value}")
    return {"enhanced": enhanced, "chain_data": data}


@node
def finalize_result_with_injection(
    data: dict, context: BaseFlowContext = Provide[BaseFlowContext]
) -> dict:
    state = context.state()
    final_value = data["enhanced"] * 1.5
    state["final_result"] = final_value
    state["processing_chain"] = [
        state.get("step1_processed"),
        state.get("step2_enhanced"),
        final_value,
    ]
    print(f"Step3: final result {final_value}")
    return {
        "final": final_value,
        "chain_history": state["processing_chain"],
        "metadata": {
            "step1_time": state.get("step1_timestamp"),
            "step2_source": state.get("step2_source"),
        },
    }


@node
def state_tracker_with_injection(
    data: dict, context: BaseFlowContext = Provide[BaseFlowContext]
) -> dict:
    state = context.state()
    shared = context.shared_data()

    # 更新状态
    current_step = state.get("step_count", 0) + 1
    state["step_count"] = current_step
    state[f"step_{current_step}_data"] = data

    # 更新共享数据
    if "global_sum" not in shared:
        shared["global_sum"] = 0
    shared["global_sum"] += data.get("value", 0)

    print(f"状态跟踪 - 步骤 {current_step}, 全局和: {shared['global_sum']}")

    return {
        "step": current_step,
        "local_data": data,
        "global_sum": shared["global_sum"],
    }


def test_then_with_dependency_injection(wired_container):
    """测试then链式调用中的依赖注入 - 使用@node装饰器"""
    print("\n=== 测试then + 依赖注入 ===")

    wired_container(__name__)
    test_container = setup_test_container(__name__)
    container = test_container.container

    # 构建链式调用
    pipeline = process_input_with_injection.then(enhance_data_with_injection).then(
        finalize_result_with_injection
    )
    print(f"Injection pipeline: {pipeline}")

    # 执行链式调用
    result = pipeline({"input": 5})

    print(f"链式调用结果: {result}")
    print(f"最终state: {container.state()}")

    # 验证链式调用结果 - ((5*2) + (5*2) + 10) * 1.5 = 30 * 1.5 = 45
    assert result["final"] == 45.0
    assert result["chain_history"] == [10, 30, 45.0]
    assert result["metadata"]["step1_time"] == "2024-01-01"
    assert result["metadata"]["step2_source"] == "from_step1_10"

    # 验证state中的中间结果
    final_state = container.state()
    assert final_state["step1_processed"] == 10
    assert final_state["step2_enhanced"] == 30
    assert final_state["final_result"] == 45.0
    print("✅ then + 依赖注入测试通过")


@node
def extract_number(
    text: str, context: BaseFlowContext = Provide[BaseFlowContext]
) -> int:
    state = context.state()
    # 提取文本中的数字
    numbers = "".join(filter(str.isdigit, text))
    extracted = int(numbers) if numbers else 0
    state["extracted_number"] = extracted
    state["original_text"] = text
    print(f"Extracted {extracted} from '{text}'")
    return extracted


@node
def calculate_square(
    num: int, context: BaseFlowContext = Provide[BaseFlowContext]
) -> int:
    state = context.state()
    squared = num * num
    state["squared_result"] = squared
    print(f"Squared {num} to get {squared}")
    return squared


@node
def format_output(
    num: int, context: BaseFlowContext = Provide[BaseFlowContext]
) -> dict:
    state = context.state()
    original_text = state.get("original_text", "unknown")
    extracted = state.get("extracted_number", 0)

    result = {
        "final_result": num,
        "calculation_chain": f"{original_text} -> {extracted} -> {num}",
        "operation": "extract_and_square",
    }
    state["final_output"] = result
    print(f"Formatted final output: {result}")
    return result


def test_decorator_with_then_chain(wired_container):
    """测试@node装饰器与then链式调用的结合"""
    print("\n=== 测试@node装饰器 + then链式调用 ===")

    wired_container(__name__)
    test_container = setup_test_container(__name__)
    container = test_container.container

    # 构建装饰器节点的链式调用
    pipeline = extract_number.then(calculate_square).then(format_output)

    # 执行: "hello123world" -> 123 -> 15129 -> formatted result
    result = pipeline("hello123world")

    print(f"装饰器链式结果: {result}")
    print(f"装饰器链式state: {container.state()}")

    assert result["final_result"] == 15129  # 123^2 = 15129
    assert result["calculation_chain"] == "hello123world -> 123 -> 15129"
    assert container.state()["extracted_number"] == 123
    assert container.state()["squared_result"] == 15129
    print("✅ @node装饰器 + then链式调用测试通过")


# 依赖注入节点
@node
def state_tracker(x: int, context: BaseFlowContext = Provide[BaseFlowContext]) -> int:
    state = context.state()
    state["current_value"] = x
    state["operations"] = state.get("operations", []) + ["tracked"]
    print(f"State tracker: {x} (operations: {state['operations']})")
    return x + 5


# 简单节点（无注入）
def simple_double(x: int) -> int:
    result = x * 2
    print(f"Simple double: {x} -> {result}")
    return result


# 另一个简单节点
def simple_format(x: int) -> str:
    result = f"result_{x}"
    print(f"Simple format: {x} -> {result}")
    return result


def test_mixed_injection_and_simple_nodes(wired_container):
    """测试混合使用有注入和无注入的节点"""
    print("\n=== 测试混合节点类型 ===")

    wired_container(__name__)
    test_container = setup_test_container(__name__)
    container = test_container.container

    # 创建节点
    double_node = Node(simple_double, name="double")
    format_node = Node(simple_format, name="format")

    # 混合链式调用：simple -> injection -> simple
    mixed_pipeline = double_node.then(state_tracker).then(format_node)

    # 执行: 10 -> 20 -> 25 -> "result_25"
    result = mixed_pipeline(10)

    print(f"混合节点结果: {result}")
    print(f"混合节点state: {container.state()}")

    assert result == "result_25"
    assert container.state()["current_value"] == 20  # double后的值
    assert container.state()["operations"] == ["tracked"]
    print("✅ 混合节点类型测试通过")


# 节点1: 输入验证和初步处理 - 使用统一的测试数据模型
@node
def validate_and_process_user(
    user_data: dict, context: BaseFlowContext = Provide[BaseFlowContext]
) -> TestUserData:
    state = context.state()
    print(f"验证用户数据: {user_data}")
    # 使用统一的TestUserData模型
    user = TestUserData(**user_data)
    state["validated_user"] = user
    print(f"验证通过的用户: {user}")
    return user


# 节点2: 用户模型转换
@node
def transform_user_model(
    user: TestUserData, context: BaseFlowContext = Provide[BaseFlowContext]
) -> ProcessedTestData:
    state = context.state()
    print(f"转换用户模型: {user.name}, {user.age}")
    processed = ProcessedTestData(
        name=user.name.title(),  # 格式化用户名
        metadata={
            "is_adult": user.age >= 18,
            "email_domain": user.email.split("@")[1] if user.email else "unknown",
            "created_at": "2024-01-01T00:00:00Z",
        },
    )
    state["processed_user"] = processed.model_dump()
    print(f"转换后的用户: {processed}")
    return processed


def test_pydantic_model_with_injection(wired_container):
    """测试Pydantic模型与依赖注入的结合"""
    print("\n=== 测试Pydantic模型+依赖注入 ===")
    wired_container(__name__)
    test_container = setup_test_container(__name__)
    container = test_container.container

    # 构建链式调用
    pipeline = validate_and_process_user.then(transform_user_model)

    # 测试正确的用户数据
    user_data = {"name": "alice smith", "age": 25, "email": "alice@example.com"}

    result = pipeline(user_data)
    print(f"Pydantic+注入结果: {result}")

    assert isinstance(result, ProcessedTestData)
    assert result.name == "Alice Smith"
    assert result.metadata["is_adult"] == True
    assert result.metadata["email_domain"] == "example.com"

    # 验证state中保存了中间结果
    final_state = container.state()
    assert "validated_user" in final_state
    assert "processed_user" in final_state

    print("✅ Pydantic模型+依赖注入测试通过")


@node
def thread_local_state_node(
    thread_id: int, value: int, context: BaseFlowContext = Provide[BaseFlowContext]
) -> dict:
    """测试线程本地state的节点"""
    state = context.state()
    shared_data = context.shared_data()

    # 在state中存储线程特定的数据
    state["thread_id"] = thread_id
    state["local_value"] = value
    state["processing_time"] = time.time()
    state["operations"] = state.get("operations", []) + [f"op_{thread_id}"]

    # 在shared_data中累积全局数据（所有线程共享）
    total_processed = shared_data.get("total_processed", 0) + 1
    shared_data["total_processed"] = total_processed
    shared_data[f"thread_{thread_id}_value"] = value

    # 模拟一些处理时间
    time.sleep(random.uniform(0.001, 0.01))

    result = {
        "thread_id": thread_id,
        "local_state_value": state["local_value"],
        "total_operations": len(state["operations"]),
        "shared_total": shared_data["total_processed"],
    }

    print(
        f"线程 {thread_id}: state={dict(state)}, shared_total={shared_data['total_processed']}"
    )
    return result


@node
def thread_processor_node(
    data: dict, context: BaseFlowContext = Provide[BaseFlowContext]
) -> dict:
    """处理线程数据的第二个节点"""
    state = context.state()
    shared_data = context.shared_data()

    thread_id = data["thread_id"]

    # 验证state中的线程本地数据仍然存在
    assert state["thread_id"] == thread_id, (
        f"State污染检测：期望thread_id={thread_id}, 实际={state.get('thread_id')}"
    )

    # 更新线程本地状态
    state["final_result"] = data["local_state_value"] * 10
    state["chain_completed"] = True

    # 更新共享数据
    if "completed_threads" not in shared_data:
        shared_data["completed_threads"] = []
    shared_data["completed_threads"].append(thread_id)

    final_result = {
        "thread_id": thread_id,
        "final_value": state["final_result"],
        "state_preserved": state["thread_id"] == thread_id,
        "total_completed": len(shared_data["completed_threads"]),
        "shared_data_keys": list(shared_data.keys()),
    }

    print(
        f"线程 {thread_id} 完成: final_value={state['final_result']}, 完成总数={len(shared_data['completed_threads'])}"
    )
    return final_result


def test_multithreading_state_isolation(wired_container):
    """测试多线程环境下Node的state和context隔离性"""
    print("\n=== 测试多线程State隔离性 ===")

    # 用于收集线程执行结果的字典
    thread_results = {}
    thread_states = {}

    wired_container(__name__)
    test_container = setup_test_container(__name__)
    container = test_container.container

    # 创建流水线
    pipeline = thread_local_state_node.then(thread_processor_node)

    def run_thread_pipeline(thread_id: int, input_value: int):
        """在单独线程中运行pipeline"""
        try:
            result = pipeline(thread_id, input_value)
            thread_results[thread_id] = result

            # 获取线程结束时的state快照
            state_snapshot = dict(container.state())
            thread_states[thread_id] = state_snapshot

        except Exception as e:
            print(f"线程 {thread_id} 执行失败: {e}")
            thread_results[thread_id] = {"error": str(e)}

    # 启动多个线程
    num_threads = 5
    test_values = [10, 20, 30, 40, 50]

    print(f"启动 {num_threads} 个并发线程...")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(run_thread_pipeline, i, test_values[i])
            for i in range(num_threads)
        ]

        # 等待所有线程完成
        for future in as_completed(futures):
            future.result()  # 获取结果，如果有异常会在这里抛出

    # 获取最终的shared_data状态
    final_shared_data = dict(container.shared_data())

    print("\n=== 线程执行结果分析 ===")
    print(f"最终shared_data: {final_shared_data}")

    # 验证结果
    assert len(thread_results) == num_threads, (
        f"期望 {num_threads} 个线程结果，实际得到 {len(thread_results)}"
    )

    # 验证State隔离性
    for thread_id in range(num_threads):
        result = thread_results[thread_id]
        assert "error" not in result, (
            f"线程 {thread_id} 执行出错: {result.get('error')}"
        )

        # 验证线程本地数据正确
        assert result["thread_id"] == thread_id, (
            f"线程ID不匹配: {result['thread_id']} != {thread_id}"
        )
        assert result["final_value"] == test_values[thread_id] * 10, (
            f"计算结果错误: {result['final_value']}"
        )
        assert result["state_preserved"], f"线程 {thread_id} 的state数据被污染"

        print(
            f"✅ 线程 {thread_id}: state隔离正常, final_value={result['final_value']}"
        )

    # 验证Shared_data共享性
    assert final_shared_data["total_processed"] == num_threads, (
        f"共享计数器错误: {final_shared_data['total_processed']}"
    )
    assert len(final_shared_data["completed_threads"]) == num_threads, "完成线程数错误"

    # 验证每个线程的数据都在shared_data中
    for thread_id in range(num_threads):
        thread_key = f"thread_{thread_id}_value"
        assert thread_key in final_shared_data, (
            f"线程 {thread_id} 数据未在shared_data中找到"
        )
        assert final_shared_data[thread_key] == test_values[thread_id], (
            f"线程 {thread_id} 共享数据值错误"
        )

    print(
        f"✅ Shared_data共享性验证通过: total_processed={final_shared_data['total_processed']}"
    )
    print("✅ ThreadLocalSingleton隔离效果验证通过")
    print("✅ 多线程State隔离性测试通过")


if __name__ == "__main__":
    print("=== @node装饰器依赖注入完整测试 ===")

    try:
        # 基础功能测试
        test_node_decorator_basic_chain()
        test_node_decorator_vs_manual_node()
        test_node_decorator_multiple_chains()
        test_node_decorator_error_handling()
        test_node_decorator_type_validation()

        # 复杂依赖注入测试
        test_then_with_dependency_injection()
        test_decorator_with_then_chain()
        test_mixed_injection_and_simple_nodes()
        test_pydantic_model_with_injection()
        test_multithreading_state_isolation()

        print("\n🎉 所有@node装饰器依赖注入测试通过！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
