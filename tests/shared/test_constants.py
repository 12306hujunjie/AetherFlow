"""
测试常量定义

集中管理测试中使用的常量、配置和模板数据。
"""

from typing import Dict, List, Any


# 断言消息模板 - 唯一被使用的常量组

# 常用断言消息模板
ASSERTION_MESSAGES = {
    "type_mismatch": "期望类型 {expected}，实际类型 {actual}",
    "value_mismatch": "期望值 {expected}，实际值 {actual}",
    "count_mismatch": "期望数量 {expected}，实际数量 {actual}",
    "key_missing": "缺少必需的键: {key}",
    "execution_failed": "节点 {node_name} 执行失败: {error}",
    "timeout_exceeded": "操作超时，期望在 {timeout}s 内完成",
    "state_corruption": "状态数据被污染，期望 {expected}，实际 {actual}",
}

# 导出列表
__all__ = [
    'ASSERTION_MESSAGES',
]