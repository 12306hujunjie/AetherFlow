"""
标准化的测试节点工厂

提供统一的节点创建接口，消除各测试文件中重复的节点创建代码。
"""

import time
import random
from typing import Callable, Optional, Dict, Any
from src.aetherflow import Node, node, ParallelResult
from ..shared.data_models import StandardTestData, TestNodeConfig


# 模块级处理器函数 - 支持pickle序列化和避免Pydantic兼容性问题
def _processor_func(data: StandardTestData, config: TestNodeConfig) -> StandardTestData:
    """模块级处理器函数，支持进程池序列化"""
    if config.should_fail and random.random() < config.failure_rate:
        raise ValueError(f"Node {config.name} intentionally failed for testing")
        
    if config.delay_seconds > 0:
        time.sleep(config.delay_seconds)
        
    processed_value = data.value * config.multiplier
    return StandardTestData(
        value=processed_value,
        name=f"{config.name}_processed_{data.name}",
        metadata={
            "processed_by": config.name,
            "original_value": data.value,
            "multiplier": config.multiplier
        }
    )


class StandardNodeFactory:
    """标准化的测试节点工厂 - 消除重复的节点创建代码"""
    
    @staticmethod
    def create_processor_node(config: TestNodeConfig) -> Node:
        """根据配置创建标准处理节点 - 支持进程池序列化"""
        # 使用functools.partial创建可序列化的处理器函数
        from functools import partial
        processor_func = partial(_processor_func, config=config)
        
        # 直接使用Node构造器，避免@node装饰器的Pydantic兼容性问题
        return Node(processor_func, name=config.name)
    
    @staticmethod
    def create_simple_node(name: str, func: Callable, use_decorator: bool = True) -> Node:
        """通用节点创建工具"""
        if use_decorator:
            decorated_func = node(func)
            decorated_func.name = name
            return decorated_func
        else:
            return Node(func, name=name)
    
    @staticmethod
    def create_aggregator_node(name: str = "aggregator") -> Node:
        """创建标准聚合节点"""
        def aggregator_func(parallel_results: Dict[str, ParallelResult]) -> Dict[str, Any]:
            successful_results = []
            failed_results = []
            execution_times = []
            
            for key, result in parallel_results.items():
                if result.success:
                    successful_results.append({
                        'key': key,
                        'result': result.result,
                        'execution_time': result.execution_time
                    })
                else:
                    failed_results.append({
                        'key': key,
                        'error': result.error,
                        'execution_time': result.execution_time
                    })
                
                if result.execution_time:
                    execution_times.append(result.execution_time)
            
            return {
                'total_results': len(parallel_results),
                'successful_count': len(successful_results),
                'failed_count': len(failed_results),
                'result_keys': list(parallel_results.keys()),
                'successful_results': successful_results,
                'failed_results': failed_results,
                'avg_execution_time': sum(execution_times) / len(execution_times) if execution_times else 0
            }
        
        # 直接使用Node构造器，避免@node装饰器的Pydantic兼容性问题
        return Node(aggregator_func, name=name)
    
    @staticmethod
    def create_sum_aggregator_node(name: str = "sum_aggregator") -> Node:
        """创建求和聚合节点"""
        def sum_aggregator_func(parallel_results: Dict[str, ParallelResult]) -> StandardTestData:
            total = 0
            processed_count = 0
            error_count = 0
            
            for key, result in parallel_results.items():
                if result.success and result.result:
                    total += result.result.value
                    processed_count += 1
                else:
                    error_count += 1
            
            return StandardTestData(
                value=total,
                name=f"aggregated_sum_{processed_count}_success_{error_count}_errors",
                metadata={
                    "aggregated_from": list(parallel_results.keys()),
                    "success_count": processed_count,
                    "error_count": error_count
                }
            )
        
        # 直接使用Node构造器，避免@node装饰器的Pydantic兼容性问题
        return Node(sum_aggregator_func, name=name)


# 便捷函数
def create_test_node(func: Callable, name: Optional[str] = None, use_decorator: bool = True) -> Node:
    """
    创建测试节点的工厂函数
    
    Args:
        func: 节点函数
        name: 节点名称
        use_decorator: 是否使用@node装饰器（推荐为True）
        
    Returns:
        创建的节点
        
    Note:
        强烈推荐使用@node装饰器而不是手动创建Node！
        如果func使用依赖注入，必须设置use_decorator=True
    """
    return StandardNodeFactory.create_simple_node(name or func.__name__, func, use_decorator)


def create_simple_processor(name: str, multiplier: int = 2) -> Node:
    """创建简单处理器节点的便捷函数"""
    config = TestNodeConfig(name=name, multiplier=multiplier)
    return StandardNodeFactory.create_processor_node(config)


def create_failing_processor(name: str, failure_rate: float = 1.0, multiplier: int = 2) -> Node:
    """创建会失败的处理器节点的便捷函数"""
    config = TestNodeConfig(
        name=name,
        multiplier=multiplier,
        should_fail=True,
        failure_rate=failure_rate
    )
    return StandardNodeFactory.create_processor_node(config)


def create_slow_processor(name: str, delay_seconds: float = 0.1, multiplier: int = 2) -> Node:
    """创建执行缓慢的处理器节点的便捷函数"""
    config = TestNodeConfig(
        name=name,
        multiplier=multiplier,
        delay_seconds=delay_seconds
    )
    return StandardNodeFactory.create_processor_node(config)


# 为StandardNodeFactory类添加便捷方法
class StandardNodeFactoryExtended(StandardNodeFactory):
    """扩展的StandardNodeFactory，添加便捷方法"""
    
    @staticmethod
    def create_simple_processor_node(name: str, multiplier: int = 2) -> Node:
        """创建简单处理器节点"""
        config = TestNodeConfig(name=name, multiplier=multiplier)
        return StandardNodeFactory.create_processor_node(config)
    
    @staticmethod
    def create_failing_node(name: str, failure_rate: float = 1.0) -> Node:
        """创建失败节点"""
        config = TestNodeConfig(name=name, multiplier=2, should_fail=True, failure_rate=failure_rate)
        return StandardNodeFactory.create_processor_node(config)
    
    @staticmethod
    def create_slow_node(name: str, delay_seconds: float = 0.1) -> Node:
        """创建慢节点"""
        config = TestNodeConfig(name=name, multiplier=2, delay_seconds=delay_seconds)
        return StandardNodeFactory.create_processor_node(config)

# 使用扩展类替换原始类
StandardNodeFactory = StandardNodeFactoryExtended


# 模块级的处理器类（支持pickle序列化，兼容现有测试）
class SimpleProcessor:
    """简单的数据处理器 - 兼容现有测试"""
    def __init__(self, name: str, multiplier: int = 2):
        self.name = name
        self.multiplier = multiplier
    
    def __call__(self, data: StandardTestData) -> StandardTestData:
        processed_value = data.value * self.multiplier
        return StandardTestData(
            value=processed_value,
            name=f"{self.name}_processed_{data.name}",
            metadata={"processor": self.name, "multiplier": self.multiplier}
        )


# 导出列表
__all__ = [
    'StandardNodeFactory',
    'create_test_node',
    'create_simple_processor',
    'create_failing_processor',
    'create_slow_processor',
    'SimpleProcessor',
]