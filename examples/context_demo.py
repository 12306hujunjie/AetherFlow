#!/usr/bin/env python3
"""
演示 AetherFlow 的简化上下文机制

这个示例展示了如何使用 thread-local 上下文进行依赖注入，
以及在多线程环境中如何收集并行执行的结果。
"""

import threading
from aetherflow import node, BaseFlowContext, get_current_context
from dependency_injector import containers, providers


class SharedDataService:
    """共享数据服务，用于演示上下文注入"""
    def __init__(self):
        self.counter = 0
        self.lock = threading.Lock()
        self.results = []
    
    def increment_counter(self):
        with self.lock:
            self.counter += 1
            return self.counter
    
    def add_result(self, result):
        with self.lock:
            self.results.append(result)


class DemoContext(BaseFlowContext):
    """演示用的上下文容器"""
    shared_data = providers.Singleton(SharedDataService)


@node
def task_with_shared_data(shared_data: SharedDataService):
    """使用共享数据服务的任务节点"""
    task_id = threading.current_thread().ident
    counter_value = shared_data.increment_counter()
    
    print(f"Task {task_id}: Counter = {counter_value}")
    
    return {
        'task_id': task_id,
        'counter_value': counter_value,
        'thread_id': task_id
    }


@node
def collect_results(parallel_results: dict, shared_data: SharedDataService):
    """收集并行执行的结果"""
    print(f"Collecting results from parallel execution: {list(parallel_results.keys())}")
    
    # 统计任务数量 - 新的键名格式：相同节点名会使用 node[index] 格式
    total_tasks = sum(1 for key in parallel_results.keys() 
                     if key == 'task_with_shared_data' or key.startswith('task_with_shared_data['))
    
    print(f"Total tasks executed: {total_tasks}")
    print(f"Final counter value: {shared_data.counter}")
    
    # 验证结果
    assert total_tasks == 3, f"应该有3个任务，但实际有{total_tasks}个"
    assert shared_data.counter == 4, f"计数器应该是4，但实际是{shared_data.counter}"
    
    return {
        "total_tasks": total_tasks,
        "final_counter": shared_data.counter
    }


def main():
    """主函数：演示简化的上下文机制"""
    print("=== AetherFlow 简化上下文机制演示 ===")
    
    # 创建上下文
    context = DemoContext()
    
    # 构建并行流程：三个任务并行执行，然后收集结果
    flow = (
        task_with_shared_data
        .fan_out_to([
            task_with_shared_data,
            task_with_shared_data,
            task_with_shared_data
        ])
        .fan_in(collect_results)
    )
    
    # 执行流程
    initial_state = {'message': 'Hello from simplified context!'}
    result = flow.run(initial_state, context)
    
    print("\n=== 执行结果 ===")
    print(f"最终状态: {result}")
    
    # 验证上下文中的共享数据
    shared_service = context.shared_data()
    print(f"共享服务最终计数器值: {shared_service.counter}")
    
    print("\n✅ 演示完成！新的简化架构工作正常。")


if __name__ == '__main__':
    main()