"""
Test smart async/sync mixing functionality.

Tests the intelligent detection and execution of mixed async/sync node flows.
"""

import asyncio

import pytest

from aetherflow import BaseFlowContext, node

# 创建测试容器
container = BaseFlowContext()
container.wire(modules=[__name__])


# ==================== 测试节点定义 ====================


@node
def sync_add_10(value: int) -> int:
    """同步节点：加10"""
    return value + 10


@node
async def async_multiply_2(value: int) -> int:
    """异步节点：乘以2"""
    await asyncio.sleep(0.01)  # 模拟异步操作
    return value * 2


@node
def sync_subtract_5(value: int) -> int:
    """同步节点：减5"""
    return value - 5


@node
async def async_divide_3(value: int) -> int:
    """异步节点：除以3"""
    await asyncio.sleep(0.01)  # 模拟异步操作
    return value // 3


@node
def sync_final_processing(value: int) -> str:
    """同步节点：最终处理"""
    return f"Final result: {value}"


@node
async def async_final_processing(value: int) -> str:
    """异步节点：最终处理"""
    await asyncio.sleep(0.01)  # 模拟异步操作
    return f"Async final result: {value}"


# ==================== 基础异步检测测试 ====================


def test_async_detection_basic():
    """测试基本的异步函数检测"""
    from aetherflow import _is_async_callable

    # 同步函数检测
    assert not _is_async_callable(sync_add_10.func)

    # 异步函数检测
    assert _is_async_callable(async_multiply_2.func)


def test_async_detection_complex():
    """测试复杂场景下的异步函数检测"""
    from aetherflow import _is_async_callable

    # 直接的协程函数
    async def direct_async():
        pass

    assert _is_async_callable(direct_async)

    # 普通同步函数
    def direct_sync():
        pass

    assert not _is_async_callable(direct_sync)


def test_node_pattern_analysis():
    """测试节点模式分析功能"""
    from aetherflow import _analyze_nodes_async_pattern

    # 纯同步节点
    sync_nodes = [sync_add_10, sync_subtract_5]
    analysis = _analyze_nodes_async_pattern(sync_nodes)
    assert analysis["async_count"] == 0
    assert analysis["sync_count"] == 2
    assert analysis["recommended_executor"] == "thread"
    assert not analysis["mixed_mode"]

    # 纯异步节点
    async_nodes = [async_multiply_2, async_divide_3]
    analysis = _analyze_nodes_async_pattern(async_nodes)
    assert analysis["async_count"] == 2
    assert analysis["sync_count"] == 0
    assert analysis["recommended_executor"] == "async"
    assert not analysis["mixed_mode"]

    # 混合节点
    mixed_nodes = [sync_add_10, async_multiply_2, sync_subtract_5]
    analysis = _analyze_nodes_async_pattern(mixed_nodes)
    assert analysis["async_count"] == 1
    assert analysis["sync_count"] == 2
    assert analysis["recommended_executor"] == "async"  # 混合模式推荐async
    assert analysis["mixed_mode"]


# ==================== 智能混合执行测试 ====================


def test_sync_to_async_sequential():
    """测试同步到异步的顺序执行"""
    # 创建混合链：sync -> async -> sync
    flow = sync_add_10.then(async_multiply_2).then(sync_subtract_5)

    # 同步调用应该正常工作
    result = flow(5)  # (5+10)*2-5 = 25
    assert result == 25


def test_async_to_sync_sequential():
    """测试异步到同步的顺序执行"""
    # 创建混合链：async -> sync -> async
    flow = async_multiply_2.then(sync_add_10).then(async_divide_3)

    # 同步调用应该正常工作
    result = flow(6)  # (6*2+10)//3 = 22//3 = 7
    assert result == 7


def test_complex_mixed_chain():
    """测试复杂的混合链"""
    # 创建复杂混合链
    flow = (
        sync_add_10.then(async_multiply_2)
        .then(sync_subtract_5)
        .then(async_divide_3)
        .then(sync_final_processing)
    )

    result = flow(10)  # ((10+10)*2-5)//3 = (40-5)//3 = 35//3 = 11
    assert result == "Final result: 11"


def test_pure_sync_chain():
    """测试纯同步链（确保向后兼容性）"""
    flow = sync_add_10.then(sync_subtract_5).then(sync_final_processing)

    result = flow(20)  # (20+10-5) = 25
    assert result == "Final result: 25"


def test_pure_async_chain():
    """测试纯异步链"""
    flow = async_multiply_2.then(async_divide_3).then(async_final_processing)

    result = flow(12)  # (12*2)//3 = 24//3 = 8
    assert result == "Async final result: 8"


# ==================== 并行执行测试 ====================


def test_fan_out_mixed_execution():
    """测试扇出混合执行"""
    # 创建混合扇出
    source = sync_add_10
    targets = [async_multiply_2, sync_subtract_5, async_divide_3]

    # 使用thread executor（应该能处理混合节点）
    fan_out_node = source.fan_out_to(targets, executor="thread")
    results = fan_out_node(10)  # 源节点输出 20

    # 检查结果
    assert len(results) == 3
    expected_results = {
        "async_multiply_2": 40,  # 20 * 2
        "sync_subtract_5": 15,  # 20 - 5
        "async_divide_3": 6,  # 20 // 3
    }

    for node_name, expected in expected_results.items():
        assert node_name in results
        assert results[node_name].success
        assert results[node_name].result == expected


def test_fan_out_async_executor_mixed():
    """测试使用async executor的混合执行"""
    source = sync_add_10
    targets = [async_multiply_2, sync_subtract_5]

    # 使用async executor
    fan_out_node = source.fan_out_to(targets, executor="async")

    # 当使用async executor时，需要在异步上下文中调用
    async def run_test():
        return await fan_out_node.func(8)  # 源节点输出 18

    # 在事件循环中运行
    results = asyncio.run(run_test())

    # 检查结果
    assert len(results) == 2
    assert results["async_multiply_2"].success
    assert results["async_multiply_2"].result == 36  # 18 * 2
    assert results["sync_subtract_5"].success
    assert results["sync_subtract_5"].result == 13  # 18 - 5


def test_fan_out_auto_executor_selection():
    """测试auto executor自动选择"""
    source = sync_add_10

    # 纯同步目标节点 - 应该选择thread
    sync_targets = [sync_subtract_5, sync_final_processing]
    fan_out_sync = source.fan_out_to(sync_targets, executor="auto")
    results_sync = fan_out_sync(15)  # 源节点输出 25

    assert len(results_sync) == 2
    assert results_sync["sync_subtract_5"].success
    assert results_sync["sync_subtract_5"].result == 20  # 25 - 5

    # 混合目标节点 - 应该选择async
    mixed_targets = [async_multiply_2, sync_subtract_5]
    fan_out_mixed = source.fan_out_to(mixed_targets, executor="auto")
    results_mixed = fan_out_mixed(5)  # 源节点输出 15

    assert len(results_mixed) == 2
    assert results_mixed["async_multiply_2"].success
    assert results_mixed["async_multiply_2"].result == 30  # 15 * 2
    assert results_mixed["sync_subtract_5"].success
    assert results_mixed["sync_subtract_5"].result == 10  # 15 - 5


# ==================== Fan-out-in测试 ====================


def test_fan_out_in_mixed():
    """测试混合节点的完整fan-out-in"""

    @node
    def aggregator(parallel_results: dict) -> str:
        """聚合器节点"""
        total = sum(
            result.result for result in parallel_results.values() if result.success
        )
        return f"Aggregated total: {total}"

    source = sync_add_10
    targets = [async_multiply_2, sync_subtract_5]

    # 完整的fan-out-in
    flow = source.fan_out_in(targets, aggregator, executor="auto")
    result = flow(10)  # 源节点输出 20, 目标节点输出 40 和 15

    assert result == "Aggregated total: 55"


def test_fan_out_in_auto_selection():
    """测试fan-out-in的auto executor选择"""

    @node
    async def async_aggregator(parallel_results: dict) -> str:
        """异步聚合器节点"""
        await asyncio.sleep(0.01)
        total = sum(
            result.result for result in parallel_results.values() if result.success
        )
        return f"Async aggregated: {total}"

    source = sync_add_10
    targets = [sync_subtract_5]

    # 包含异步聚合器，应该自动选择async
    flow = source.fan_out_in(targets, async_aggregator, executor="auto")
    result = flow(20)  # 源节点输出 30, 目标节点输出 25

    assert result == "Async aggregated: 25"


# ==================== 错误处理测试 ====================


def test_invalid_executor_types():
    """测试无效的executor类型"""
    source = sync_add_10
    targets = [sync_subtract_5]

    # 测试无效的executor类型
    with pytest.raises(
        ValueError, match="Only 'thread', 'async', and 'auto' executors are supported"
    ):
        source.fan_out_to(targets, executor="invalid")

    with pytest.raises(
        ValueError, match="Only 'thread', 'async', and 'auto' executors are supported"
    ):
        source.fan_out_in(targets, sync_final_processing, executor="process")


def test_empty_node_analysis():
    """测试空节点列表分析"""
    from aetherflow import _analyze_nodes_async_pattern

    analysis = _analyze_nodes_async_pattern([])
    assert analysis["async_count"] == 0
    assert analysis["sync_count"] == 0
    assert analysis["total_count"] == 0
    assert analysis["async_ratio"] == 0.0
    assert analysis["recommended_executor"] == "thread"
    assert not analysis["mixed_mode"]


# ==================== 性能对比测试 ====================


def test_execution_performance_comparison():
    """比较不同执行器的性能（基础测试）"""
    import time

    source = sync_add_10
    targets = [sync_subtract_5, sync_final_processing]

    # Thread executor
    start_time = time.time()
    thread_flow = source.fan_out_to(targets, executor="thread")
    thread_results = thread_flow(50)
    thread_time = time.time() - start_time

    # Auto executor (应该选择thread)
    start_time = time.time()
    auto_flow = source.fan_out_to(targets, executor="auto")
    auto_results = auto_flow(50)
    auto_time = time.time() - start_time

    # 结果应该相同
    assert len(thread_results) == len(auto_results) == 2

    # 性能应该接近（auto应该选择了thread）
    assert abs(thread_time - auto_time) < 0.1  # 允许100ms差异


# ==================== 向后兼容性测试 ====================


def test_backward_compatibility():
    """测试向后兼容性 - 现有代码应该继续工作"""

    # 传统的sequential composition
    old_style_flow = sync_add_10.then(sync_subtract_5)
    result = old_style_flow(100)
    assert result == 105  # (100+10-5)

    # 传统的parallel fan-out
    old_fan_out = sync_add_10.fan_out_to([sync_subtract_5, sync_final_processing])
    results = old_fan_out(30)
    assert len(results) == 2
    assert results["sync_subtract_5"].success
    assert results["sync_subtract_5"].result == 35  # (30+10-5)


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])
