"""
并行测试验证工具

提供标准化的并行执行结果验证工具，消除重复的验证逻辑。
"""

from typing import Any

from src.aetherflow import ParallelResult

from ..shared.test_constants import ASSERTION_MESSAGES


class ParallelTestValidator:
    """并行执行结果验证工具 - 消除重复的验证逻辑"""

    @staticmethod
    def assert_parallel_results(
        results: dict[str, ParallelResult],
        expected_total: int,
        expected_success: int = None,
        expected_failure: int = None,
    ):
        """标准化的并行结果断言"""
        # 基本类型检查
        assert isinstance(results, dict), ASSERTION_MESSAGES["type_mismatch"].format(
            expected="dict", actual=type(results).__name__
        )

        # 结果数量检查
        assert len(results) == expected_total, ASSERTION_MESSAGES[
            "count_mismatch"
        ].format(expected=expected_total, actual=len(results))

        # 分离成功和失败结果
        successful_results = []
        failed_results = []

        for key, result in results.items():
            assert isinstance(result, ParallelResult), (
                f"结果{key}应为ParallelResult类型，实际为{type(result).__name__}"
            )
            assert result.node_name is not None, f"结果{key}应包含节点名称"
            assert result.execution_time is not None, f"结果{key}应包含执行时间"

            if result.success:
                successful_results.append((key, result))
            else:
                failed_results.append((key, result))

        # 成功数量检查
        if expected_success is not None:
            assert len(successful_results) == expected_success, ASSERTION_MESSAGES[
                "count_mismatch"
            ].format(
                expected=f"{expected_success}个成功结果",
                actual=f"{len(successful_results)}个成功结果",
            )

        # 失败数量检查
        if expected_failure is not None:
            assert len(failed_results) == expected_failure, ASSERTION_MESSAGES[
                "count_mismatch"
            ].format(
                expected=f"{expected_failure}个失败结果",
                actual=f"{len(failed_results)}个失败结果",
            )

        return successful_results, failed_results

    @staticmethod
    def assert_result_keys_unique(
        results: dict[str, ParallelResult], base_names: list[str] = None
    ):
        """验证结果键的唯一性"""
        keys = list(results.keys())
        unique_keys = set(keys)

        assert len(keys) == len(unique_keys), f"结果键应唯一，发现重复: {keys}"

        if base_names:
            # 验证键是基于预期的base_names生成的
            expected_patterns = set()
            for base_name in base_names:
                count = sum(1 for key in keys if key.startswith(base_name))
                if count == 1:
                    expected_patterns.add(base_name)
                else:
                    for i in range(count):
                        if i == 0:
                            expected_patterns.add(base_name)
                        else:
                            expected_patterns.add(f"{base_name}[{i}]")

            assert set(keys) == expected_patterns, (
                f"键格式不正确，期望: {expected_patterns}, 实际: {set(keys)}"
            )

    @staticmethod
    def assert_execution_times_reasonable(
        results: dict[str, ParallelResult],
        min_time: float = 0.0,
        max_time: float = 10.0,
    ):
        """验证执行时间在合理范围内"""
        for key, result in results.items():
            exec_time = result.execution_time
            assert exec_time is not None, f"结果{key}缺少执行时间"
            assert min_time <= exec_time <= max_time, (
                f"结果{key}执行时间{exec_time:.4f}s不在合理范围[{min_time}, {max_time}]内"
            )

    @staticmethod
    def assert_successful_results_have_values(successful_results: list[tuple]):
        """验证成功结果包含有效值"""
        for key, result in successful_results:
            assert result.result is not None, f"成功结果{key}应包含返回值"
            assert result.error is None, f"成功结果{key}不应包含错误信息"

    @staticmethod
    def assert_failed_results_have_errors(failed_results: list[tuple]):
        """验证失败结果包含错误信息"""
        for key, result in failed_results:
            assert result.error is not None, f"失败结果{key}应包含错误信息"
            assert (
                "intentionally failed" in result.error
                or "failed" in result.error
                or "模拟失败" in result.error
                or "重试次数耗尽" in result.error
            ), f"失败结果{key}的错误信息格式不正确: {result.error}"


# 便捷断言函数
def assert_parallel_results(
    results: dict[str, ParallelResult],
    expected_total: int,
    expected_success: int = None,
    expected_failure: int = None,
):
    """并行结果断言的便捷函数"""
    return ParallelTestValidator.assert_parallel_results(
        results, expected_total, expected_success, expected_failure
    )


def assert_result_keys_unique(
    results: dict[str, ParallelResult], base_names: list[str] = None
):
    """结果键唯一性断言的便捷函数"""
    return ParallelTestValidator.assert_result_keys_unique(results, base_names)


def assert_all_results_successful(results: dict[str, ParallelResult]):
    """断言所有结果都成功"""
    successful, failed = ParallelTestValidator.assert_parallel_results(
        results, len(results), expected_success=len(results), expected_failure=0
    )
    ParallelTestValidator.assert_successful_results_have_values(successful)
    return successful


def assert_mixed_results(
    results: dict[str, ParallelResult], expected_success: int, expected_failure: int
):
    """断言混合成功/失败结果"""
    successful, failed = ParallelTestValidator.assert_parallel_results(
        results, expected_success + expected_failure, expected_success, expected_failure
    )

    ParallelTestValidator.assert_successful_results_have_values(successful)
    ParallelTestValidator.assert_failed_results_have_errors(failed)

    return successful, failed


def extract_result_values(
    results: dict[str, ParallelResult], only_successful: bool = True
) -> list[Any]:
    """提取并行结果中的值"""
    values = []
    for key, result in results.items():
        if result.success and result.result is not None:
            values.append(result.result)
        elif not only_successful:
            values.append(None)
    return values


def calculate_success_rate(results: dict[str, ParallelResult]) -> float:
    """计算成功率"""
    if not results:
        return 0.0

    successful_count = sum(1 for result in results.values() if result.success)
    return successful_count / len(results)


# 导出列表
__all__ = [
    "ParallelTestValidator",
    "assert_parallel_results",
    "assert_result_keys_unique",
    "assert_all_results_successful",
    "assert_mixed_results",
    "extract_result_values",
    "calculate_success_rate",
]
