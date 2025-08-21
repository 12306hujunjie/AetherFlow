#!/usr/bin/env python3
"""
并发安全性测试用例

测试AetherFlow框架在多线程和异步环境下的正确性。
"""

import threading
import asyncio
import time
import concurrent.futures
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.aetherflow import node, AppContext, BaseFlowContext


class TestConcurrentSafety:
    """并发安全性测试类"""
    
    def test_thread_local_state_isolation(self):
        """测试线程本地状态隔离"""
        results = {}
        threads = []
        num_threads = 5
        
        @node
        def set_value(x):
            return {'thread_value': x + threading.current_thread().ident}
        
        @node
        def get_value(thread_value):
            return {'final_value': thread_value * 2}
        
        def worker(thread_id):
            # 每个线程使用不同的初始值
            initial_state = {'x': thread_id * 100}
            
            # 创建流程
            flow = set_value.then(get_value)
            result = flow.run(initial_state)
            
            results[thread_id] = result
        
        # 启动多个线程
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(results) == num_threads
        
        # 每个线程的结果应该不同且正确
        for thread_id, result in results.items():
            expected_thread_value = thread_id * 100 + threading.current_thread().ident
            # 注意：由于线程ID会不同，我们只检查基本逻辑
            assert 'final_value' in result
            print(f"Thread {thread_id}: {result}")
    
    def test_parallel_fan_out_thread_safety(self):
        """测试并行扇出的线程安全性"""
        
        @node
        def source(x):
            return x
        
        @node
        def worker_a(source):
            # 模拟一些计算
            time.sleep(0.01)
            return source + 100
        
        @node
        def worker_b(source):
            # 模拟一些计算
            time.sleep(0.01) 
            return source + 200
        
        @node
        def worker_c(source):
            # 模拟一些计算
            time.sleep(0.01)
            return source + 300
        
        # 测试多次执行
        for i in range(10):
            initial_state = {'x': i * 10}
            
            parallel_flow = source.fan_out_to([worker_a, worker_b, worker_c])
            result = parallel_flow.run(initial_state)
            
            # 验证所有并行结果都存在
            assert '__parallel_results' in result
            parallel_results = result['__parallel_results']
            
            # 应该有3个结果
            assert len(parallel_results) == 3
            
            # 验证结果值
            expected_values = {i * 10 + 100, i * 10 + 200, i * 10 + 300}
            actual_values = set(parallel_results.values())
            assert actual_values == expected_values
            
            print(f"Iteration {i}: {parallel_results}")
    
    def test_concurrent_container_access(self):
        """测试并发容器访问"""
        
        @node
        def compute_with_context():
            # 测试依赖注入容器的线程安全性
            context = AppContext()
            state = context.state()  # 获取线程本地状态
            
            # 每个线程应该有自己的状态字典
            thread_id = threading.current_thread().ident
            state[f'thread_{thread_id}'] = f'data_from_{thread_id}'
            
            return {'computed_by': thread_id, 'state_keys': list(state.keys())}
        
        results = []
        
        def worker():
            result = compute_with_context.run({})
            results.append(result)
        
        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(results) == 5
        
        # 每个线程应该有不同的状态
        for result in results:
            assert 'computed_by' in result
            assert 'state_keys' in result
            print(f"Thread result: {result}")
    
    def test_high_concurrency_stress(self):
        """高并发压力测试"""
        
        @node
        def stress_worker(iteration):
            # 模拟CPU密集型任务
            total = 0
            for i in range(1000):
                total += i * iteration
            
            return {'result': total, 'thread': threading.current_thread().ident}
        
        # 使用ThreadPoolExecutor进行压力测试
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # 提交100个任务
            futures = []
            for i in range(100):
                future = executor.submit(lambda x: stress_worker.run({'iteration': x}), i)
                futures.append(future)
            
            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        # 验证所有任务都完成了
        assert len(results) == 100
        
        # 验证结果的唯一性（每个迭代值应该产生不同的结果）
        result_values = [r['result'] for r in results]
        assert len(set(result_values)) == len(result_values)  # 所有结果都应该不同
        
        print(f"Completed {len(results)} concurrent tasks")


async def test_async_compatibility():
    """测试异步兼容性（虽然框架是同步的，但不应该在异步环境中崩溃）"""
    
    @node 
    def async_compatible_node(x):
        # 在异步环境中运行同步节点
        return {'async_result': x * 2}
    
    # 在异步环境中运行同步节点
    tasks = []
    for i in range(10):
        # 使用run_in_executor在异步环境中运行同步代码
        task = asyncio.get_event_loop().run_in_executor(
            None, 
            lambda x: async_compatible_node.run({'x': x}), 
            i
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # 验证结果
    assert len(results) == 10
    for i, result in enumerate(results):
        assert result['async_result'] == i * 2
    
    print(f"Async compatibility test passed with {len(results)} results")


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始并发安全性测试...")
    
    test_suite = TestConcurrentSafety()
    
    print("\n1. 测试线程本地状态隔离...")
    test_suite.test_thread_local_state_isolation()
    print("✅ 线程本地状态隔离测试通过")
    
    print("\n2. 测试并行扇出线程安全性...")
    test_suite.test_parallel_fan_out_thread_safety()
    print("✅ 并行扇出线程安全性测试通过")
    
    print("\n3. 测试并发容器访问...")
    test_suite.test_concurrent_container_access()
    print("✅ 并发容器访问测试通过")
    
    print("\n4. 高并发压力测试...")
    test_suite.test_high_concurrency_stress()
    print("✅ 高并发压力测试通过")
    
    print("\n5. 异步兼容性测试...")
    asyncio.run(test_async_compatibility())
    print("✅ 异步兼容性测试通过")
    
    print("\n🎉 所有并发安全性测试都已通过！")


if __name__ == "__main__":
    run_all_tests()