"""
自定义断言函数

提供专用的断言函数，增强测试的可读性和错误信息。
"""

from typing import Any

from src.aetherflow import BaseFlowContext

from ..shared.data_models import StandardTestData, TestUserData
from ..shared.test_constants import ASSERTION_MESSAGES


def assert_node_state(
    context: BaseFlowContext,
    expected_keys: list[str],
    expected_values: dict[str, Any] = None,
):
    """断言节点状态包含预期的键和值"""
    state = context.state()

    # 检查必需的键是否存在
    for key in expected_keys:
        assert key in state, ASSERTION_MESSAGES["key_missing"].format(key=key)

    # 检查特定的值
    if expected_values:
        for key, expected_value in expected_values.items():
            actual_value = state.get(key)
            assert actual_value == expected_value, ASSERTION_MESSAGES[
                "value_mismatch"
            ].format(expected=expected_value, actual=actual_value)


def assert_execution_time(
    execution_time: float,
    min_time: float = 0.0,
    max_time: float = 5.0,
    operation_name: str = "操作",
):
    """断言执行时间在合理范围内"""
    assert execution_time is not None, f"{operation_name}应该有执行时间"
    assert isinstance(execution_time, int | float), (
        f"{operation_name}执行时间应该是数值类型，实际为{type(execution_time)}"
    )
    assert min_time <= execution_time <= max_time, (
        f"{operation_name}执行时间{execution_time:.4f}s不在合理范围[{min_time}, {max_time}]内"
    )


def assert_test_data_equal(
    actual: StandardTestData,
    expected: StandardTestData,
    check_timestamp: bool = False,
    check_metadata: bool = True,
):
    """断言StandardTestData对象相等"""
    assert isinstance(actual, StandardTestData), ASSERTION_MESSAGES[
        "type_mismatch"
    ].format(expected="StandardTestData", actual=type(actual).__name__)

    assert actual.value == expected.value, ASSERTION_MESSAGES["value_mismatch"].format(
        expected=f"value={expected.value}", actual=f"value={actual.value}"
    )

    assert actual.name == expected.name, ASSERTION_MESSAGES["value_mismatch"].format(
        expected=f"name='{expected.name}'", actual=f"name='{actual.name}'"
    )

    if check_timestamp:
        assert actual.timestamp == expected.timestamp, ASSERTION_MESSAGES[
            "value_mismatch"
        ].format(
            expected=f"timestamp={expected.timestamp}",
            actual=f"timestamp={actual.timestamp}",
        )

    if check_metadata and expected.metadata:
        for key, expected_value in expected.metadata.items():
            assert key in actual.metadata, ASSERTION_MESSAGES["key_missing"].format(
                key=f"metadata.{key}"
            )
            assert actual.metadata[key] == expected_value, ASSERTION_MESSAGES[
                "value_mismatch"
            ].format(
                expected=f"metadata.{key}={expected_value}",
                actual=f"metadata.{key}={actual.metadata[key]}",
            )


def assert_user_data_equal(actual: TestUserData, expected: TestUserData):
    """断言TestUserData对象相等"""
    assert isinstance(actual, TestUserData), ASSERTION_MESSAGES["type_mismatch"].format(
        expected="TestUserData", actual=type(actual).__name__
    )

    assert actual.name == expected.name, ASSERTION_MESSAGES["value_mismatch"].format(
        expected=f"name='{expected.name}'", actual=f"name='{actual.name}'"
    )

    assert actual.age == expected.age, ASSERTION_MESSAGES["value_mismatch"].format(
        expected=f"age={expected.age}", actual=f"age={actual.age}"
    )

    assert actual.email == expected.email, ASSERTION_MESSAGES["value_mismatch"].format(
        expected=f"email='{expected.email}'", actual=f"email='{actual.email}'"
    )


def assert_container_state_isolation(
    container: BaseFlowContext,
    thread_id: int,
    expected_local_keys: list[str],
    expected_shared_keys: list[str] = None,
):
    """断言容器状态隔离性"""
    state = container.state()
    shared_data = container.shared_data()

    # 检查线程本地状态
    for key in expected_local_keys:
        assert key in state, f"线程{thread_id}状态缺少本地键: {key}"

    # 检查共享数据（如果提供）
    if expected_shared_keys:
        for key in expected_shared_keys:
            assert key in shared_data, f"共享数据缺少键: {key}"


def assert_error_contains_message(
    error: Exception, expected_message: str, case_sensitive: bool = False
):
    """断言错误包含预期的消息"""
    error_str = str(error)
    if not case_sensitive:
        error_str = error_str.lower()
        expected_message = expected_message.lower()

    assert expected_message in error_str, (
        f"错误消息应包含'{expected_message}'，实际错误: '{error_str}'"
    )


def assert_values_in_range(
    values: list[float], min_value: float, max_value: float, value_name: str = "值"
):
    """断言所有值都在指定范围内"""
    for i, value in enumerate(values):
        assert min_value <= value <= max_value, (
            f"{value_name}[{i}]={value}不在范围[{min_value}, {max_value}]内"
        )


def assert_list_contains_types(items: list[Any], expected_types: tuple):
    """断言列表中的所有元素都是预期类型"""
    for i, item in enumerate(items):
        assert isinstance(item, expected_types), ASSERTION_MESSAGES[
            "type_mismatch"
        ].format(
            expected=f"{expected_types}",
            actual=f"items[{i}]的类型为{type(item).__name__}",
        )


def assert_dict_structure(
    data: dict[str, Any], required_keys: list[str], optional_keys: list[str] = None
):
    """断言字典包含必需的键结构"""
    # 检查必需键
    for key in required_keys:
        assert key in data, ASSERTION_MESSAGES["key_missing"].format(key=key)

    # 检查是否有多余的键
    all_expected_keys = set(required_keys)
    if optional_keys:
        all_expected_keys.update(optional_keys)

    unexpected_keys = set(data.keys()) - all_expected_keys
    if unexpected_keys:
        print(f"警告: 发现意外的键: {unexpected_keys}")


def assert_processing_chain_valid(
    chain_history: list[Any], expected_length: int, monotonic_increasing: bool = True
):
    """断言处理链历史的有效性"""
    assert len(chain_history) == expected_length, ASSERTION_MESSAGES[
        "count_mismatch"
    ].format(expected=expected_length, actual=len(chain_history))

    # 检查单调性（如果需要）
    if monotonic_increasing and len(chain_history) > 1:
        for i in range(1, len(chain_history)):
            if isinstance(chain_history[i], int | float) and isinstance(
                chain_history[i - 1], int | float
            ):
                assert chain_history[i] >= chain_history[i - 1], (
                    f"处理链在位置{i}不是单调递增: {chain_history[i - 1]} -> {chain_history[i]}"
                )


# 导出列表
__all__ = [
    "assert_node_state",
    "assert_execution_time",
    "assert_test_data_equal",
    "assert_user_data_equal",
    "assert_container_state_isolation",
    "assert_error_contains_message",
    "assert_values_in_range",
    "assert_list_contains_types",
    "assert_dict_structure",
    "assert_processing_chain_valid",
]
