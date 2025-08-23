# tests/fixtures/__init__.py
"""
测试fixtures模块 - 提供依赖注入测试基础设施
"""

from .injection_helpers import BaseTestContainer, setup_test_container

__all__ = [
    'BaseTestContainer',
    'setup_test_container'
]