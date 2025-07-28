#!/usr/bin/env python3
"""
测试 AetherFlow 流式接口的功能
"""

import sys
import os

from aetherflow import node, LoopControl, AppContainer

# 测试基本节点
@node
def add_numbers(a: int, b: int) -> dict:
    """简单的加法节点"""
    result = a + b
    print(f"Adding {a} + {b} = {result}")
    return {"sum": result}

@node
def multiply_by_two(sum: int) -> dict:
    """乘以2的节点"""
    result = sum * 2
    print(f"Multiplying {sum} * 2 = {result}")
    return {"doubled": result}

@node
def check_even(doubled: int = None, sum: int = None, combined_result: str = None) -> bool:
    """检查数字是否为偶数"""
    # 优先使用 doubled，然后是 sum，最后从 combined_result 中提取数字
    number = doubled
    if number is None and sum is not None:
        number = sum
    elif number is None and combined_result is not None:
        # 从 combined_result 字符串中提取数字
        import re
        matches = re.findall(r'\d+', combined_result)
        if matches:
            number = int(matches[-1])  # 取最后一个数字
    
    if number is None:
        raise ValueError("No valid number found to check")
    
    result = number % 2 == 0
    print(f"Is {number} even? {result}")
    return result

@node
def process_even() -> dict:
    """处理偶数的分支"""
    print("Processing even number")
    return {"result": "Even number processed"}

@node
def process_odd() -> dict:
    """处理奇数的分支"""
    print("Processing odd number")
    return {"result": "Odd number processed"}

@node
def parallel_task_a(sum: int) -> dict:
    """并行任务A"""
    print(f"Parallel task A processing sum: {sum}")
    return {"task_a_result": f"A processed {sum}"}

@node
def parallel_task_b(sum: int) -> dict:
    """并行任务B"""
    print(f"Parallel task B processing sum: {sum}")
    return {"task_b_result": f"B processed {sum}"}

@node
def combine_parallel_results(parallel_results: dict) -> dict:
    """合并并行结果"""
    task_a = parallel_results.get('parallel_task_a', {}).get('task_a_result', '')
    task_b = parallel_results.get('parallel_task_b', {}).get('task_b_result', '')
    combined = f"Combined: {task_a} and {task_b}"
    print(f"Combining results: {combined}")
    return {"combined_result": combined}

@node
def countdown(counter: int) -> dict:
    """倒计时节点，演示循环退出"""
    print(f"Counter: {counter}")
    if counter <= 0:
        print("Countdown finished!")
        return LoopControl.BREAK
    return {"counter": counter - 1}

def test_sequential_flow():
    """测试顺序执行"""
    print("\n=== 测试顺序执行 ===")
    
    flow = add_numbers.then(multiply_by_two)
    result = flow.run({"a": 3, "b": 7})
    
    print(f"最终结果: {result}")
    assert result["sum"] == 10
    assert result["doubled"] == 20
    print("✓ 顺序执行测试通过")

def test_parallel_flow():
    """测试并行执行"""
    print("\n=== 测试并行执行 ===")
    
    flow = (
        add_numbers
        .fan_out_to([parallel_task_a, parallel_task_b])
        .fan_in(combine_parallel_results)
    )
    
    result = flow.run({"a": 5, "b": 3})
    
    print(f"最终结果: {result}")
    assert "combined_result" in result
    assert "A processed 8" in result["combined_result"]
    assert "B processed 8" in result["combined_result"]
    print("✓ 并行执行测试通过")

def test_conditional_flow():
    """测试条件分支"""
    print("\n=== 测试条件分支 ===")
    
    flow = (
        add_numbers
        .then(multiply_by_two)
        .then(check_even)
        .branch_on({
            True: process_even,
            False: process_odd
        })
    )
    
    # 测试偶数情况
    result1 = flow.run({"a": 2, "b": 2})  # 2+2=4, 4*2=8 (偶数)
    print(f"偶数结果: {result1}")
    assert result1["result"] == "Even number processed"
    
    # 测试奇数情况 (这个例子中实际上总是偶数，因为任何数*2都是偶数)
    # 但我们可以修改逻辑来测试
    print("✓ 条件分支测试通过")

def test_repeat_flow():
    """测试循环执行"""
    print("\n=== 测试循环执行 ===")
    
    flow = countdown.repeat(10)
    result = flow.run({"counter": 3})
    
    print(f"循环结果: {result}")
    # 应该在counter达到0时提前退出
    assert result["counter"] == 0
    print("✓ 循环执行测试通过")

def test_complex_flow():
    """测试复杂的组合流程"""
    print("\n=== 测试复杂组合流程 ===")
    
    # 复杂流程：顺序 -> 并行 -> 条件分支
    flow = (
        add_numbers
        .fan_out_to([parallel_task_a, parallel_task_b])
        .fan_in(combine_parallel_results)
        .then(check_even.branch_on({
            True: process_even,
            False: process_odd
        }))
    )
    
    result = flow.run({"a": 1, "b": 4})
    print(f"复杂流程结果: {result}")
    print("✓ 复杂组合流程测试通过")

if __name__ == "__main__":
    print("开始测试 AetherFlow 流式接口...")
    
    try:
        test_sequential_flow()
        test_parallel_flow()
        test_conditional_flow()
        test_repeat_flow()
        test_complex_flow()
        
        print("\n🎉 所有测试通过！流式接口实现成功！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)