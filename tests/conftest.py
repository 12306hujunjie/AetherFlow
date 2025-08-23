"""
pytest全局配置文件

提供测试会话级别的fixtures和配置。
"""

import pytest

from src.aetherflow import BaseFlowContext
from .fixtures.injection_helpers import TestConfig


@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """全局测试配置"""
    return TestConfig(
        container_name="pytest_container",
        auto_wire=True,
        state_isolation=True,
    )


@pytest.fixture(scope="function")
def injection_container(test_config) -> BaseFlowContext:
    """函数级别的依赖注入容器fixture"""
    container = BaseFlowContext()
    return container

@pytest.fixture(scope="function")
def wired_container(injection_container):
    """预配置依赖注入的容器工厂"""
    def _wire_container(module):
        injection_container.wire(modules=[module])
        return injection_container
    return _wire_container