#!/usr/bin/env python3
"""
并发安全示例

展示AetherFlow框架在多线程环境下的线程安全特性。
这个示例证明了使用ThreadLocalSingleton后，每个线程都有独立的状态空间。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import threading
import time
import concurrent.futures
from src.aetherflow import node, AppContext


# 定义一些示例节点
@node
def init_counter(start_value):
    """初始化计数器"""
    return {'counter': start_value}


@node  
def increment(counter):
    """递增计数器"""
    time.sleep(0.01)  # 模拟一些处理时间
    return {'counter': counter + 1}


@node
def multiply(counter, factor):
    """将计数器乘以因子"""
    return {'counter': counter * factor}


@node
def get_thread_info(counter):
    """获取线程信息和最终结果"""
    thread_id = threading.current_thread().ident
    return {
        'final_counter': counter,
        'thread_id': thread_id,
        'thread_name': threading.current_thread().name
    }


def concurrent_demo():
    """并发执行演示"""
    print("🚀 AetherFlow 并发安全演示")
    print("=" * 50)
    
    # 创建一个复合流程
    workflow = (init_counter
                .then(increment)
                .then(increment) 
                .then(multiply)
                .then(get_thread_info))
    
    results = {}
    
    def worker(worker_id):
        """工作线程函数"""
        print(f"⚡ 工作线程 {worker_id} 开始执行...")
        
        # 每个线程使用不同的初始值和因子
        initial_state = {
            'start_value': worker_id * 10,
            'factor': worker_id + 1
        }
        
        # 执行工作流程
        result = workflow.run(initial_state)
        
        results[worker_id] = result
        print(f"✅ 工作线程 {worker_id} 完成: 最终计数器 = {result['final_counter']}")
    
    # 启动多个工作线程
    threads = []
    num_workers = 5
    
    print(f"\n启动 {num_workers} 个并发工作线程...")
    
    for i in range(num_workers):
        thread = threading.Thread(target=worker, args=(i,), name=f"Worker-{i}")
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 显示结果
    print(f"\n📊 最终结果汇总:")
    print("-" * 30)
    
    for worker_id in sorted(results.keys()):
        result = results[worker_id]
        expected = ((worker_id * 10 + 2) * (worker_id + 1))
        actual = result['final_counter']
        
        print(f"工作线程 {worker_id}:")
        print(f"  初始值: {worker_id * 10}")
        print(f"  递增2次后: {worker_id * 10 + 2}")
        print(f"  乘以因子 {worker_id + 1}: 期望={expected}, 实际={actual}")
        print(f"  线程ID: {result['thread_id']}")
        print(f"  状态正确: {'✅' if actual == expected else '❌'}")
        print()
    
    # 验证所有结果都是正确的
    all_correct = all(
        result['final_counter'] == ((worker_id * 10 + 2) * (worker_id + 1))
        for worker_id, result in results.items()
    )
    
    print("🎉 并发安全性验证:", "✅ 通过" if all_correct else "❌ 失败")
    
    return all_correct


def parallel_processing_demo():
    """并行处理演示"""
    print(f"\n🔄 并行处理演示")
    print("=" * 50)
    
    @node
    def data_source(base_number):
        """数据源"""
        return base_number * 2
    
    @node
    def process_a(data_source):
        """处理器A: 加法运算"""
        time.sleep(0.05)  # 模拟处理时间
        return data_source + 100
    
    @node
    def process_b(data_source):  
        """处理器B: 乘法运算"""
        time.sleep(0.05)  # 模拟处理时间
        return data_source * 3
    
    @node
    def process_c(data_source):
        """处理器C: 幂运算"""
        time.sleep(0.05)  # 模拟处理时间
        return data_source ** 1.5
    
    # 创建扇出流程
    parallel_workflow = data_source.fan_out_to([process_a, process_b, process_c])
    
    # 测试多个输入值
    test_values = [1, 2, 3, 4, 5]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # 并发执行多个工作流程实例
        futures = []
        
        for i, value in enumerate(test_values):
            future = executor.submit(
                parallel_workflow.run,
                {'base_number': value}
            )
            futures.append((i, value, future))
        
        # 收集结果
        print("处理结果:")
        for i, value, future in futures:
            result = future.result()
            parallel_results = result['__parallel_results']
            
            print(f"输入 {value}:")
            print(f"  数据源处理后: {value * 2}")
            print(f"  处理器A (加法): {parallel_results.get('process_a', 'N/A')}")
            print(f"  处理器B (乘法): {parallel_results.get('process_b', 'N/A')}")
            print(f"  处理器C (幂运算): {parallel_results.get('process_c', 'N/A')}")
            print()
    
    print("✅ 并行处理演示完成")


if __name__ == "__main__":
    try:
        # 运行并发演示
        success = concurrent_demo()
        
        # 运行并行处理演示
        parallel_processing_demo()
        
        if success:
            print("\n🎊 所有演示都成功完成！AetherFlow框架在并发环境下工作正常。")
        else:
            print("\n⚠️ 发现问题，请检查实现。")
            
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        raise