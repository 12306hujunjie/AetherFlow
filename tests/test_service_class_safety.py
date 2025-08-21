#!/usr/bin/env python3
"""
复杂服务类的线程安全测试

验证ThreadLocalSingleton对复杂服务对象的处理是否正确，
包括构造器线程安全性、方法重入性、内存管理等方面。
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any
from dependency_injector import containers, providers

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.aetherflow import node, BaseFlowContext


# 测试用的复杂服务类
class StatefulService:
    """有状态服务类，用于测试ThreadLocalSingleton行为"""
    
    def __init__(self, service_id: str = None):
        self.service_id = service_id or f"service_{threading.current_thread().ident}"
        self.call_count = 0
        self.operation_history = []
        self.thread_id = threading.current_thread().ident
        self.creation_time = time.time()
        print(f"✅ Created {self.service_id} in thread {self.thread_id}")
    
    def increment_counter(self):
        """线程安全的计数器操作"""
        self.call_count += 1
        operation = {
            'operation': 'increment',
            'count': self.call_count,
            'thread_id': threading.current_thread().ident,
            'timestamp': time.time()
        }
        self.operation_history.append(operation)
        return self.call_count
    
    def get_stats(self):
        """获取服务统计信息"""
        return {
            'service_id': self.service_id,
            'creation_thread': self.thread_id,
            'current_thread': threading.current_thread().ident,
            'call_count': self.call_count,
            'operations': len(self.operation_history),
            'uptime': time.time() - self.creation_time
        }
    
    def reset_state(self):
        """重置状态"""
        self.call_count = 0
        self.operation_history.clear()


class DatabaseMockService:
    """模拟数据库服务，测试资源管理"""
    
    def __init__(self):
        self.connection_count = 0
        self.queries_executed = []
        self.thread_id = threading.current_thread().ident
        self.is_connected = False
        self._connect()
    
    def _connect(self):
        """模拟数据库连接"""
        time.sleep(0.001)  # 模拟连接延迟
        self.is_connected = True
        self.connection_count += 1
        print(f"🔌 DB connected in thread {self.thread_id} (connection #{self.connection_count})")
    
    def execute_query(self, query: str):
        """执行查询"""
        if not self.is_connected:
            raise RuntimeError("Database not connected")
        
        result = {
            'query': query,
            'thread_id': threading.current_thread().ident,
            'execution_time': time.time(),
            'connection_id': self.connection_count
        }
        self.queries_executed.append(result)
        return result
    
    def get_connection_info(self):
        """获取连接信息"""
        return {
            'thread_id': self.thread_id,
            'current_thread': threading.current_thread().ident,
            'connection_count': self.connection_count,
            'queries_count': len(self.queries_executed),
            'is_connected': self.is_connected
        }


# 测试上下文容器
class ServiceTestContext(BaseFlowContext):
    """服务测试上下文 - 使用ThreadLocalSingleton"""
    state = providers.ThreadLocalSingleton(dict)
    stateful_service = providers.ThreadLocalSingleton(StatefulService)
    db_service = providers.ThreadLocalSingleton(DatabaseMockService)


class SharedServiceTestContext(BaseFlowContext):
    """共享服务测试上下文 - 使用Singleton对比"""
    state = providers.Singleton(dict)
    shared_service = providers.Singleton(StatefulService, service_id="shared_service")


# 测试节点
@node
def service_operation(data, stateful_service: StatefulService):
    """使用有状态服务的操作节点"""
    thread_id = threading.current_thread().ident
    
    # 执行多个操作
    count1 = stateful_service.increment_counter()
    count2 = stateful_service.increment_counter()
    
    # 获取统计信息
    stats = stateful_service.get_stats()
    
    return {
        'thread_id': thread_id,
        'data': data,
        'count1': count1,
        'count2': count2,
        'service_stats': stats
    }


@node
def database_operation(query, db_service: DatabaseMockService):
    """数据库操作节点"""
    result = db_service.execute_query(query)
    connection_info = db_service.get_connection_info()
    
    return {
        'query_result': result,
        'connection_info': connection_info
    }


@node
def complex_service_workflow(data, stateful_service: StatefulService, db_service: DatabaseMockService):
    """复杂服务工作流程"""
    thread_id = threading.current_thread().ident
    
    # 使用多个服务
    service_count = stateful_service.increment_counter()
    db_result = db_service.execute_query(f"SELECT * FROM users WHERE id={data}")
    
    # 验证服务实例的线程归属
    service_stats = stateful_service.get_stats()
    db_info = db_service.get_connection_info()
    
    return {
        'thread_id': thread_id,
        'data': data,
        'service_count': service_count,
        'db_result': db_result,
        'service_thread_match': service_stats['creation_thread'] == thread_id,
        'db_thread_match': db_info['thread_id'] == thread_id,
        'service_stats': service_stats,
        'db_info': db_info
    }


class TestServiceClassSafety:
    """复杂服务类线程安全测试套件"""
    
    def test_thread_local_service_instances(self):
        """测试线程本地服务实例创建"""
        context = ServiceTestContext()
        results = []
        
        def worker(worker_id):
            result = service_operation.run({'data': worker_id}, context)
            results.append(result)
        
        # 启动多个线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 验证结果
        assert len(results) == 5
        
        # 每个线程应该有独立的服务实例
        service_threads = set()
        for result in results:
            stats = result['service_stats']
            service_threads.add(stats['creation_thread'])
            
            # 服务应该在创建它的线程中运行
            assert stats['creation_thread'] == result['thread_id']
            assert stats['current_thread'] == result['thread_id']
        
        # 应该有5个不同的线程创建了服务实例
        assert len(service_threads) == 5
        print(f"✅ 创建了 {len(service_threads)} 个独立的服务实例")
    
    def test_service_state_isolation(self):
        """测试服务状态隔离"""
        context = ServiceTestContext()
        results = []
        
        def worker(worker_id):
            # 每个工作线程执行不同次数的操作
            operations = worker_id + 1
            result = None
            
            for i in range(operations):
                result = service_operation.run({'data': f"{worker_id}_{i}"}, context)
            
            results.append({
                'worker_id': worker_id,
                'operations': operations,
                'final_result': result
            })
        
        # 启动工作线程
        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 验证状态隔离
        for result in results:
            worker_id = result['worker_id']
            operations = result['operations']
            final_result = result['final_result']
            
            # 每个worker的最终计数应该等于其操作次数的2倍（每次service_operation调用increment两次）
            expected_count = operations * 2
            actual_count = final_result['service_stats']['call_count']
            
            assert actual_count == expected_count, f"Worker {worker_id}: expected {expected_count}, got {actual_count}"
        
        print("✅ 服务状态在线程间完全隔离")
    
    def test_database_service_isolation(self):
        """测试数据库服务线程隔离"""
        context = ServiceTestContext()
        results = []
        
        def worker(worker_id):
            queries = [f"SELECT {worker_id}_{i}" for i in range(3)]
            worker_results = []
            
            for query in queries:
                result = database_operation.run({'query': query}, context)
                worker_results.append(result)
            
            results.append({
                'worker_id': worker_id,
                'queries': queries,
                'results': worker_results
            })
        
        # 启动多个线程
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 验证数据库连接隔离
        connection_threads = set()
        for worker_result in results:
            for result in worker_result['results']:
                conn_info = result['connection_info']
                
                # 数据库服务应该在当前线程中创建
                assert conn_info['thread_id'] == conn_info['current_thread']
                connection_threads.add(conn_info['thread_id'])
        
        # 应该有3个不同的数据库连接
        assert len(connection_threads) == 3
        print(f"✅ 创建了 {len(connection_threads)} 个独立的数据库连接")
    
    def test_complex_service_workflow(self):
        """测试复杂服务工作流程"""
        context = ServiceTestContext()
        results = []
        
        def worker(worker_id):
            # 每个worker处理多个数据项
            data_items = [worker_id * 10 + i for i in range(3)]
            worker_results = []
            
            for data in data_items:
                result = complex_service_workflow.run({'data': data}, context)
                worker_results.append(result)
            
            results.append({
                'worker_id': worker_id,
                'results': worker_results
            })
        
        # 启动工作线程
        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 验证复杂工作流程的正确性
        for worker_result in results:
            worker_id = worker_result['worker_id']
            
            for i, result in enumerate(worker_result['results']):
                # 验证线程匹配
                assert result['service_thread_match'], f"Worker {worker_id}: service thread mismatch"
                assert result['db_thread_match'], f"Worker {worker_id}: database thread mismatch"
                
                # 验证服务计数的累积性
                expected_count = i + 1  # 每次调用increment一次
                actual_count = result['service_count']
                assert actual_count == expected_count, f"Worker {worker_id}, call {i}: expected {expected_count}, got {actual_count}"
        
        print("✅ 复杂服务工作流程线程安全验证通过")
    
    def test_service_performance_under_load(self):
        """测试服务在高负载下的性能"""
        context = ServiceTestContext()
        
        def worker(worker_id):
            start_time = time.time()
            results = []
            
            # 执行100次操作
            for i in range(100):
                result = service_operation.run({'data': f"{worker_id}_{i}"}, context)
                results.append(result)
            
            end_time = time.time()
            return {
                'worker_id': worker_id,
                'execution_time': end_time - start_time,
                'operations': len(results),
                'final_count': results[-1]['service_stats']['call_count']
            }
        
        # 使用ThreadPoolExecutor进行高并发测试
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证结果
        total_operations = sum(r['operations'] for r in results)
        
        # 每个worker应该执行了200次increment操作（100次service_operation，每次2个increment）
        for result in results:
            assert result['final_count'] == 200, f"Worker {result['worker_id']}: expected 200, got {result['final_count']}"
        
        # 性能统计
        avg_time_per_worker = sum(r['execution_time'] for r in results) / len(results)
        operations_per_second = total_operations / total_time
        
        print(f"✅ 高负载测试完成:")
        print(f"   - 总操作数: {total_operations}")
        print(f"   - 总时间: {total_time:.2f}s")
        print(f"   - 平均每worker时间: {avg_time_per_worker:.2f}s")
        print(f"   - 操作/秒: {operations_per_second:.0f}")
        
        # 性能断言（根据实际情况调整）
        assert operations_per_second > 1000, f"Performance too low: {operations_per_second} ops/sec"
    
    def test_shared_vs_thread_local_comparison(self):
        """对比共享服务和线程本地服务的行为"""
        thread_local_context = ServiceTestContext()
        shared_context = SharedServiceTestContext()
        
        results = {
            'thread_local': [],
            'shared': []
        }
        
        def thread_local_worker(worker_id):
            result = service_operation.run({'data': worker_id}, thread_local_context)
            results['thread_local'].append(result)
        
        def shared_worker(worker_id):
            # 注意：shared context使用不同的节点，因为服务名不同
            @node
            def shared_service_operation(data, shared_service: StatefulService):
                thread_id = threading.current_thread().ident
                count1 = shared_service.increment_counter()
                count2 = shared_service.increment_counter()
                stats = shared_service.get_stats()
                return {
                    'thread_id': thread_id,
                    'data': data,
                    'count1': count1,
                    'count2': count2,
                    'service_stats': stats
                }
            
            result = shared_service_operation.run({'data': worker_id}, shared_context)
            results['shared'].append(result)
        
        # 启动线程本地测试
        threads = []
        for i in range(3):
            t = threading.Thread(target=thread_local_worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 启动共享状态测试
        threads = []
        for i in range(3):
            t = threading.Thread(target=shared_worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 分析结果差异
        print("\n🔍 线程本地 vs 共享服务对比:")
        
        # 线程本地：每个线程独立计数
        thread_local_counts = [r['service_stats']['call_count'] for r in results['thread_local']]
        print(f"   线程本地计数: {thread_local_counts}")
        assert all(count == 2 for count in thread_local_counts), "线程本地服务计数应该都是2"
        
        # 共享服务：全局累积计数
        shared_counts = [r['count2'] for r in results['shared']]
        shared_counts.sort()  # 排序因为线程执行顺序不确定
        print(f"   共享服务计数: {shared_counts}")
        
        # 共享服务的计数应该是累积的
        assert max(shared_counts) == 6, f"共享服务最大计数应该是6，实际是{max(shared_counts)}"
        
        print("✅ 两种模式的行为差异验证完成")
    
    def test_memory_cleanup(self):
        """测试线程结束后的内存清理"""
        import gc
        import weakref
        
        context = ServiceTestContext()
        service_refs = []
        
        def worker_with_ref_tracking(worker_id):
            # 获取服务实例并创建弱引用
            result = service_operation.run({'data': worker_id}, context)
            
            # 通过context直接获取服务实例以创建弱引用
            service_instance = context.stateful_service()
            weak_ref = weakref.ref(service_instance)
            service_refs.append({
                'worker_id': worker_id,
                'weak_ref': weak_ref,
                'service_id': service_instance.service_id
            })
            
            return result
        
        # 启动并等待线程完成
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker_with_ref_tracking, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 强制垃圾回收
        gc.collect()
        time.sleep(0.1)  # 给系统一点时间清理
        gc.collect()
        
        # 检查弱引用状态
        print(f"\n🧹 内存清理检查:")
        alive_refs = 0
        for ref_info in service_refs:
            is_alive = ref_info['weak_ref']() is not None
            status = "存活" if is_alive else "已清理"
            print(f"   Service {ref_info['service_id']}: {status}")
            if is_alive:
                alive_refs += 1
        
        # 注意：ThreadLocal对象可能不会立即被回收，这是正常的
        # 这个测试主要是为了观察内存管理行为
        print(f"   存活的服务实例: {alive_refs}/{len(service_refs)}")


def run_service_safety_tests():
    """运行所有服务安全测试"""
    print("🚀 开始复杂服务类线程安全测试...")
    
    test_suite = TestServiceClassSafety()
    
    try:
        test_suite.test_thread_local_service_instances()
        print("✅ 线程本地服务实例测试通过")
        
        test_suite.test_service_state_isolation()
        print("✅ 服务状态隔离测试通过")
        
        test_suite.test_database_service_isolation()
        print("✅ 数据库服务隔离测试通过")
        
        test_suite.test_complex_service_workflow()
        print("✅ 复杂服务工作流程测试通过")
        
        test_suite.test_service_performance_under_load()
        print("✅ 高负载性能测试通过")
        
        test_suite.test_shared_vs_thread_local_comparison()
        print("✅ 共享vs线程本地对比测试通过")
        
        test_suite.test_memory_cleanup()
        print("✅ 内存清理测试完成")
        
        print("\n🎉 所有复杂服务类线程安全测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_service_safety_tests()
    exit(0 if success else 1)