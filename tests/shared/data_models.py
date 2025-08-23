"""
统一的测试数据结构

替代各个测试文件中重复定义的数据类，提供标准化的测试数据模型。
"""

import time
from typing import Any

from pydantic import BaseModel, Field, model_validator


class StandardTestData(BaseModel):
    """标准化的测试数据结构 - 替代各文件中重复的TestData"""

    value: int
    name: str
    timestamp: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_defaults(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        return self


class TestUserData(BaseModel):
    """标准化的用户测试数据"""

    name: str
    age: int
    email: str | None = None


class TestNodeConfig(BaseModel):
    """标准化的节点配置"""

    name: str
    node_type: str = "processor"
    multiplier: int = 2
    should_fail: bool = False
    delay_seconds: float = 0.0
    failure_rate: float = 1.0  # 仅当should_fail=True时有效


class ProcessedTestData(BaseModel):
    """处理后的测试数据结构"""

    name: str | None = None
    original: StandardTestData | None = None
    processed_value: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    processing_time: float | None = None
    processor_name: str | None = None

    @model_validator(mode="after")
    def set_processing_time(self):
        if self.processing_time is None:
            self.processing_time = time.time()
        return self


class UserInput(BaseModel):
    """用户输入数据结构 - 兼容现有测试"""

    name: str
    age: int


class ProcessedUser(BaseModel):
    """处理后的用户数据 - 兼容现有测试"""

    formatted_name: str
    is_adult: bool


class FinalResult(BaseModel):
    """最终结果数据结构 - 兼容现有测试"""

    message: str
    user_type: str


# 辅助函数
def create_test_data(value: int, name: str = None) -> StandardTestData:
    """创建标准测试数据"""
    if name is None:
        name = f"test_data_{value}"
    return StandardTestData(value=value, name=name)


# 导出列表
__all__ = [
    "StandardTestData",
    "TestUserData",
    "TestNodeConfig",
    "ProcessedTestData",
    "UserInput",
    "ProcessedUser",
    "FinalResult",
    "create_test_data",
]
