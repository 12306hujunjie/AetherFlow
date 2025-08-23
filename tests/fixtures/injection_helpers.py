#!/usr/bin/env python3
"""
injection_helpers.py - 依赖注入测试基础设施

提供标准化的容器创建和wire操作。
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Callable, Optional
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.aetherflow import BaseFlowContext, node
from dependency_injector.wiring import Provide

# 导入统一的测试数据结构
from ..shared.data_models import StandardTestData, TestUserData, ProcessedTestData, create_test_data


@dataclass
class TestConfig:
    """测试配置数据结构"""
    container_name: str = "test_container"
    auto_wire: bool = True
    state_isolation: bool = True


class BaseTestContainer:
    """标准化的测试容器基类"""

    def __init__(self, config: Optional[TestConfig] = None):
        self.config = config or TestConfig()
        self.container = BaseFlowContext()
        self._wired_modules = []

    def wire_module(self, module):
        """Wire模块到容器"""
        if module not in self._wired_modules:
            self.container.wire(modules=[module])
            self._wired_modules.append(module)
        return self

    def get_state(self) -> dict:
        """获取当前状态"""
        return self.container.state()
    
    def get_shared_data(self) -> dict:
        """获取共享数据"""
        return self.container.shared_data()
    
    def reset_state(self):
        """重置状态（仅用于测试）"""
        # 注意：这会破坏线程安全，仅在测试中使用
        state = self.container.state()
        state.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 清理资源
        pass


def setup_test_container(module=None, config: Optional[TestConfig] = None) -> BaseTestContainer:
    """
    设置标准化测试容器

    Args:
        module: 需要wire的模块，通常传入 __name__
        config: 测试配置

    Returns:
        配置好的测试容器

    Example:
        container = setup_test_container(__name__)
        # 现在可以使用依赖注入的节点了
    """
    container = BaseTestContainer(config)
    if module:
        container.wire_module(module)
    return container


@contextmanager
def isolated_injection_context(module):
    """
    隔离的依赖注入上下文管理器
    
    Example:
        with isolated_injection_context(__name__) as container:
            # 使用依赖注入的节点
            result = my_injection_node(test_data)
    """
    container = BaseFlowContext()
    container.wire(modules=[module])
    try:
        yield container
    finally:
        # 清理资源
        pass

# 导出的接口
__all__ = [
    'BaseTestContainer',
    'setup_test_container',
    'TestConfig'
]