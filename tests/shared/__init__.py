"""
测试共享模块

提供统一的测试数据结构、常量和模式类。
"""

from .data_models import *
from .test_constants import *

__all__ = [
    # 数据模型
    'StandardTestData',
    'TestUserData',
    'TestNodeConfig',
    'ProcessedTestData',
    'UserInput',
    'ProcessedUser', 
    'FinalResult',
    'create_test_data',
    
    # 测试常量
    'ASSERTION_MESSAGES',
]