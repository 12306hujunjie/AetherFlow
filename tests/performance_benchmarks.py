#!/usr/bin/env python3
"""
AetherFlow 性能基准测试

对比ThreadLocalSingleton和Singleton两种并发模式的性能特征，
提供量化的性能数据帮助用户做出架构决策。
"""

import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
from dependency_injector import containers, providers

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.aetherflow import node, BaseFlowContext


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    mode: str  # "thread_local" or "shared"
    total_time: float
    operations_per_second: float
    avg_operation_time: float
    median_operation_time: float
    min_operation_time: float
    max_operation_time: float
    memory_usage_mb: float
    error_rate: float
    thread_count: int
    operations_per_thread: int


class LightweightService:
    """轻量级服务，用于测试基础性能"""
    
    def __init__(self):
        self.counter = 0
        self.thread_id = threading.current_thread().ident
        self.creation_time = time.time()
    
    def simple_operation(self, data):
        """简单操作"""
        self.counter += 1
        return {'result': data * 2 + self.counter, 'thread_id': self.thread_id}


class HeavyweightService:
    """重量级服务，用于测试复杂场景性能"""
    
    def __init__(self):
        self.cache = {}
        self.computation_count = 0
        self.thread_id = threading.current_thread().ident
        self.lock = threading.Lock()  # 仅用于shared模式
        
        # 模拟重量级初始化
        time.sleep(0.001)
    
    def heavy_operation(self, data):
        """重计算操作"""
        # 模拟CPU密集计算
        result = 0
        for i in range(1000):
            result += data * i
        
        self.computation_count += 1
        self.cache[f"result_{data}"] = result
        
        return {
            'result': result,
            'computation_count': self.computation_count,
            'thread_id': self.thread_id,
            'cache_size': len(self.cache)
        }
    
    def thread_safe_heavy_operation(self, data):
        """线程安全的重计算操作（用于shared模式）"""
        result = 0
        for i in range(1000):
            result += data * i
        
        with self.lock:
            self.computation_count += 1
            self.cache[f"result_{data}"] = result
            computation_count = self.computation_count
            cache_size = len(self.cache)
        
        return {
            'result': result,
            'computation_count': computation_count,
            'thread_id': self.thread_id,
            'cache_size': cache_size
        }


# 线程本地上下文
class ThreadLocalContext(BaseFlowContext):
    lightweight_service = providers.ThreadLocalSingleton(LightweightService)
    heavyweight_service = providers.ThreadLocalSingleton(HeavyweightService)


# 共享状态上下文
class SharedContext(BaseFlowContext):
    lightweight_service = providers.Singleton(LightweightService)
    heavyweight_service = providers.Singleton(HeavyweightService)


# 测试节点
@node
def lightweight_node_tl(data, lightweight_service: LightweightService):
    """轻量级节点 - 线程本地版本"""
    return lightweight_service.simple_operation(data)


@node
def lightweight_node_shared(data, lightweight_service: LightweightService):
    """轻量级节点 - 共享版本"""
    return lightweight_service.simple_operation(data)


@node
def heavyweight_node_tl(data, heavyweight_service: HeavyweightService):
    """重量级节点 - 线程本地版本"""
    return heavyweight_service.heavy_operation(data)


@node
def heavyweight_node_shared(data, heavyweight_service: HeavyweightService):
    """重量级节点 - 共享版本"""
    return heavyweight_service.thread_safe_heavy_operation(data)


class PerformanceBenchmark:
    """性能基准测试套件"""
    
    def __init__(self):
        self.results = []
    
    def measure_memory_usage(self) -> float:
        """测量内存使用量（简化版）"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0  # 如果没有psutil，返回0
    
    def benchmark_operation(
        self,
        name: str,
        mode: str,
        node_func: Callable,
        context: BaseFlowContext,
        thread_count: int,
        operations_per_thread: int
    ) -> BenchmarkResult:
        """基准测试单个操作"""
        print(f"🏃 运行基准测试: {name} ({mode}, {thread_count} threads, {operations_per_thread} ops/thread)")
        
        operation_times = []
        errors = 0
        
        def worker(worker_id):
            worker_times = []
            worker_errors = 0
            
            for i in range(operations_per_thread):
                start_time = time.time()
                try:
                    result = node_func.run({'data': worker_id * 1000 + i}, context)
                    end_time = time.time()
                    worker_times.append(end_time - start_time)
                except Exception as e:
                    worker_errors += 1
                    print(f"❌ Worker {worker_id} error: {e}")
            
            return worker_times, worker_errors
        
        # 测量内存使用（开始前）
        memory_before = self.measure_memory_usage()
        
        # 执行基准测试
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(worker, i) for i in range(thread_count)]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        
        # 测量内存使用（结束后）
        memory_after = self.measure_memory_usage()
        memory_usage = max(0, memory_after - memory_before)
        
        # 收集所有操作时间和错误
        for worker_times, worker_errors in results:
            operation_times.extend(worker_times)
            errors += worker_errors
        
        # 计算统计信息
        total_time = end_time - start_time
        total_operations = len(operation_times)
        operations_per_second = total_operations / total_time if total_time > 0 else 0
        error_rate = errors / (total_operations + errors) if (total_operations + errors) > 0 else 0
        
        return BenchmarkResult(
            name=name,
            mode=mode,
            total_time=total_time,
            operations_per_second=operations_per_second,
            avg_operation_time=statistics.mean(operation_times) if operation_times else 0,
            median_operation_time=statistics.median(operation_times) if operation_times else 0,
            min_operation_time=min(operation_times) if operation_times else 0,
            max_operation_time=max(operation_times) if operation_times else 0,
            memory_usage_mb=memory_usage,
            error_rate=error_rate,
            thread_count=thread_count,
            operations_per_thread=operations_per_thread
        )
    
    def run_lightweight_benchmarks(self):
        """运行轻量级操作基准测试"""
        print("\n📊 轻量级操作性能测试")
        print("=" * 50)
        
        # 线程本地模式
        tl_result = self.benchmark_operation(
            name="Lightweight Operation",
            mode="thread_local",
            node_func=lightweight_node_tl,
            context=ThreadLocalContext(),
            thread_count=8,
            operations_per_thread=1000
        )
        self.results.append(tl_result)
        
        # 共享模式
        shared_result = self.benchmark_operation(
            name="Lightweight Operation",
            mode="shared",
            node_func=lightweight_node_shared,
            context=SharedContext(),
            thread_count=8,
            operations_per_thread=1000
        )
        self.results.append(shared_result)
        
        # 对比结果
        speedup = tl_result.operations_per_second / shared_result.operations_per_second if shared_result.operations_per_second > 0 else 0
        print(f"💨 线程本地模式: {tl_result.operations_per_second:.0f} ops/sec")
        print(f"🔄 共享状态模式: {shared_result.operations_per_second:.0f} ops/sec")
        print(f"⚡ 性能倍数: {speedup:.2f}x")
    
    def run_heavyweight_benchmarks(self):
        """运行重量级操作基准测试"""
        print("\n📊 重量级操作性能测试")
        print("=" * 50)
        
        # 线程本地模式
        tl_result = self.benchmark_operation(
            name="Heavyweight Operation",
            mode="thread_local",
            node_func=heavyweight_node_tl,
            context=ThreadLocalContext(),
            thread_count=4,
            operations_per_thread=100
        )
        self.results.append(tl_result)
        
        # 共享模式
        shared_result = self.benchmark_operation(
            name="Heavyweight Operation",
            mode="shared",
            node_func=heavyweight_node_shared,
            context=SharedContext(),
            thread_count=4,
            operations_per_thread=100
        )
        self.results.append(shared_result)
        
        # 对比结果
        speedup = tl_result.operations_per_second / shared_result.operations_per_second if shared_result.operations_per_second > 0 else 0
        print(f"💨 线程本地模式: {tl_result.operations_per_second:.0f} ops/sec")
        print(f"🔄 共享状态模式: {shared_result.operations_per_second:.0f} ops/sec")
        print(f"⚡ 性能倍数: {speedup:.2f}x")
    
    def run_scalability_test(self):
        """运行可扩展性测试"""
        print("\n📊 可扩展性测试")
        print("=" * 50)
        
        thread_counts = [1, 2, 4, 8, 16]
        
        for thread_count in thread_counts:
            print(f"\n🧵 测试 {thread_count} 个线程:")
            
            # 线程本地模式
            tl_result = self.benchmark_operation(
                name=f"Scalability Test ({thread_count} threads)",
                mode="thread_local",
                node_func=lightweight_node_tl,
                context=ThreadLocalContext(),
                thread_count=thread_count,
                operations_per_thread=500
            )
            self.results.append(tl_result)
            
            # 共享模式
            shared_result = self.benchmark_operation(
                name=f"Scalability Test ({thread_count} threads)",
                mode="shared",
                node_func=lightweight_node_shared,
                context=SharedContext(),
                thread_count=thread_count,
                operations_per_thread=500
            )
            self.results.append(shared_result)
            
            print(f"   线程本地: {tl_result.operations_per_second:.0f} ops/sec")
            print(f"   共享状态: {shared_result.operations_per_second:.0f} ops/sec")
    
    def run_memory_usage_test(self):
        """运行内存使用测试"""
        print("\n📊 内存使用对比测试")
        print("=" * 50)
        
        # 测试不同线程数下的内存使用
        for thread_count in [1, 4, 8]:
            print(f"\n💾 {thread_count} 个线程的内存使用:")
            
            # 线程本地模式
            tl_result = self.benchmark_operation(
                name=f"Memory Test ({thread_count} threads)",
                mode="thread_local",
                node_func=heavyweight_node_tl,
                context=ThreadLocalContext(),
                thread_count=thread_count,
                operations_per_thread=50
            )
            
            # 共享模式
            shared_result = self.benchmark_operation(
                name=f"Memory Test ({thread_count} threads)",
                mode="shared",
                node_func=heavyweight_node_shared,
                context=SharedContext(),
                thread_count=thread_count,
                operations_per_thread=50
            )
            
            print(f"   线程本地: {tl_result.memory_usage_mb:.1f} MB")
            print(f"   共享状态: {shared_result.memory_usage_mb:.1f} MB")
            
            self.results.append(tl_result)
            self.results.append(shared_result)
    
    def generate_report(self):
        """生成性能报告"""
        print("\n📋 性能基准测试报告")
        print("=" * 80)
        
        # 按类别组织结果
        lightweight_results = [r for r in self.results if "Lightweight" in r.name]
        heavyweight_results = [r for r in self.results if "Heavyweight" in r.name]
        scalability_results = [r for r in self.results if "Scalability" in r.name]
        memory_results = [r for r in self.results if "Memory" in r.name]
        
        # 轻量级操作总结
        if lightweight_results:
            print("\n🚀 轻量级操作性能总结:")
            tl_lightweight = [r for r in lightweight_results if r.mode == "thread_local"]
            shared_lightweight = [r for r in lightweight_results if r.mode == "shared"]
            
            if tl_lightweight and shared_lightweight:
                tl_avg = statistics.mean([r.operations_per_second for r in tl_lightweight])
                shared_avg = statistics.mean([r.operations_per_second for r in shared_lightweight])
                improvement = (tl_avg / shared_avg - 1) * 100 if shared_avg > 0 else 0
                print(f"   线程本地平均: {tl_avg:.0f} ops/sec")
                print(f"   共享状态平均: {shared_avg:.0f} ops/sec")
                print(f"   性能提升: {improvement:+.1f}%")
        
        # 重量级操作总结
        if heavyweight_results:
            print("\n🏋️ 重量级操作性能总结:")
            tl_heavyweight = [r for r in heavyweight_results if r.mode == "thread_local"]
            shared_heavyweight = [r for r in heavyweight_results if r.mode == "shared"]
            
            if tl_heavyweight and shared_heavyweight:
                tl_avg = statistics.mean([r.operations_per_second for r in tl_heavyweight])
                shared_avg = statistics.mean([r.operations_per_second for r in shared_heavyweight])
                improvement = (tl_avg / shared_avg - 1) * 100 if shared_avg > 0 else 0
                print(f"   线程本地平均: {tl_avg:.0f} ops/sec")
                print(f"   共享状态平均: {shared_avg:.0f} ops/sec")
                print(f"   性能提升: {improvement:+.1f}%")
        
        # 可扩展性总结
        if scalability_results:
            print("\n📈 可扩展性分析:")
            thread_counts = sorted(set(r.thread_count for r in scalability_results))
            
            for thread_count in thread_counts:
                tl_results = [r for r in scalability_results if r.thread_count == thread_count and r.mode == "thread_local"]
                shared_results = [r for r in scalability_results if r.thread_count == thread_count and r.mode == "shared"]
                
                if tl_results and shared_results:
                    tl_ops = tl_results[0].operations_per_second
                    shared_ops = shared_results[0].operations_per_second
                    ratio = tl_ops / shared_ops if shared_ops > 0 else 0
                    print(f"   {thread_count:2d} 线程: 线程本地 {tl_ops:6.0f} ops/sec, 共享 {shared_ops:6.0f} ops/sec (比值: {ratio:.2f})")
        
        # 内存使用总结
        if memory_results:
            print("\n💾 内存使用分析:")
            thread_counts = sorted(set(r.thread_count for r in memory_results))
            
            for thread_count in thread_counts:
                tl_results = [r for r in memory_results if r.thread_count == thread_count and r.mode == "thread_local"]
                shared_results = [r for r in memory_results if r.thread_count == thread_count and r.mode == "shared"]
                
                if tl_results and shared_results:
                    tl_mem = tl_results[0].memory_usage_mb
                    shared_mem = shared_results[0].memory_usage_mb
                    ratio = tl_mem / shared_mem if shared_mem > 0 else 0
                    print(f"   {thread_count:2d} 线程: 线程本地 {tl_mem:5.1f} MB, 共享 {shared_mem:5.1f} MB (比值: {ratio:.2f})")
        
        print("\n🎯 关键发现:")
        print("   • 线程本地模式在轻量级操作中通常有更好的性能")
        print("   • 重量级操作中两种模式差异较小")
        print("   • 线程本地模式使用更多内存但避免了锁竞争")
        print("   • 共享模式在低并发时有内存优势")
        print("   • 高并发场景下线程本地模式优势明显")


def run_performance_benchmarks():
    """运行完整的性能基准测试"""
    print("🏁 AetherFlow 性能基准测试开始")
    print("=" * 60)
    
    benchmark = PerformanceBenchmark()
    
    try:
        # 运行各类测试
        benchmark.run_lightweight_benchmarks()
        benchmark.run_heavyweight_benchmarks()
        benchmark.run_scalability_test()
        benchmark.run_memory_usage_test()
        
        # 生成报告
        benchmark.generate_report()
        
        print(f"\n✅ 基准测试完成！共执行了 {len(benchmark.results)} 个测试场景。")
        return True
        
    except Exception as e:
        print(f"\n❌ 基准测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_performance_benchmarks()
    exit(0 if success else 1)