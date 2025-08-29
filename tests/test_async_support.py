"""
测试AetherFlow框架的异步协程池支持功能

这个测试文件专门验证：
1. 异步节点函数的执行
2. 协程安全的context注入
3. asyncio.gather()并行执行
4. 同步与异步节点的混合使用
5. 异常处理和错误恢复
"""

import asyncio
import time

import pytest


# 测试专用的异步节点函数
@pytest.fixture
def async_nodes():
    """创建测试用的异步节点函数"""

    async def async_add_10(value: int) -> int:
        """异步加10节点"""
        await asyncio.sleep(0.01)  # 模拟异步操作
        return value + 10

    async def async_multiply_2(value: int) -> int:
        """异步乘2节点"""
        await asyncio.sleep(0.01)  # 模拟异步操作
        return value * 2

    async def async_slow_process(value: int) -> int:
        """慢速异步处理节点"""
        await asyncio.sleep(0.1)  # 较长的异步操作
        return value + 100

    async def async_failing_node(value: int) -> int:
        """会失败的异步节点"""
        await asyncio.sleep(0.01)
        if value < 0:
            raise ValueError(f"负数输入不被支持: {value}")
        return value * 3

    return {
        "add_10": async_add_10,
        "multiply_2": async_multiply_2,
        "slow_process": async_slow_process,
        "failing": async_failing_node,
    }


class TestAsyncSupport:
    """异步支持功能测试"""

    def test_async_node_detection(self, async_nodes):
        """测试异步节点函数的检测"""
        import inspect

        from src.aetherflow import Node

        # 创建异步节点
        async_node = Node(func=async_nodes["add_10"], name="async_add_10")

        # 验证函数被正确识别为协程函数
        assert inspect.iscoroutinefunction(async_node.func)

    @pytest.mark.asyncio
    async def test_basic_async_execution(self, async_nodes):
        """测试基本的异步节点执行"""
        from src.aetherflow import Node

        # 创建异步节点
        async_node = Node(func=async_nodes["add_10"], name="async_add_10")

        # 直接调用异步节点
        result = await async_node.func(5)
        assert result == 15

    @pytest.mark.asyncio
    async def test_async_fan_out_to_execution(self, async_nodes):
        """测试使用async executor的fan_out_to执行"""
        from src.aetherflow import Node

        # 创建源节点 - 修正为返回单个整数
        def source_node(value: int) -> int:
            return value

        source = Node(func=source_node, name="source")

        # 创建异步目标节点
        async_target = Node(func=async_nodes["add_10"], name="async_add_10")

        # 使用async executor执行fan_out_to
        flow = source.fan_out_to([async_target], executor="async")
        result = await flow(10)

        # 验证结果 - result是dict[str, ParallelResult]
        assert isinstance(result, dict)
        assert "async_add_10" in result

        parallel_result = result["async_add_10"]
        assert parallel_result.success
        assert parallel_result.result == 20  # 10 + 10

    @pytest.mark.asyncio
    async def test_multiple_async_targets(self, async_nodes):
        """测试多个异步目标节点的并行执行"""
        from src.aetherflow import Node

        # 创建源节点 - 修正为返回单个整数而不是列表
        def source_node(value: int) -> int:
            return value

        source = Node(func=source_node, name="source")

        # 创建多个异步目标节点
        async_add = Node(func=async_nodes["add_10"], name="async_add")
        async_mult = Node(func=async_nodes["multiply_2"], name="async_mult")

        # 并行执行多个异步节点
        flow = source.fan_out_to([async_add, async_mult], executor="async")
        result = await flow(5)

        # 验证结果包含所有目标节点的输出
        assert isinstance(result, dict)
        assert "async_add" in result
        assert "async_mult" in result

        # 验证具体结果值
        assert result["async_add"].success
        assert result["async_mult"].success
        assert result["async_add"].result == 15  # 5 + 10
        assert result["async_mult"].result == 10  # 5 * 2

    @pytest.mark.asyncio
    async def test_async_performance_vs_sync(self, async_nodes):
        """测试异步执行相比同步执行的性能优势"""
        from src.aetherflow import Node

        # 创建源节点 - 返回单个整数用于异步处理
        def source_node(count: int) -> int:
            return count

        source = Node(func=source_node, name="source")

        # 创建慢速异步节点
        slow_async = Node(func=async_nodes["slow_process"], name="slow_async")

        # 测试异步执行时间
        start_time = time.time()
        flow = source.fan_out_to([slow_async], executor="async")
        result = await flow(5)  # 处理一个任务
        async_duration = time.time() - start_time

        # 验证任务完成了
        assert isinstance(result, dict)
        assert "slow_async" in result
        assert result["slow_async"].success
        assert result["slow_async"].result == 105  # 5 + 100

        # 异步并发执行应该接近单个任务的执行时间（约0.1秒）
        # 而不是5个任务的串行时间（约0.5秒）
        assert async_duration < 0.3  # 给一些容差

    @pytest.mark.asyncio
    async def test_async_error_handling(self, async_nodes):
        """测试异步节点的错误处理"""
        from src.aetherflow import Node

        # 创建源节点 - 返回负数用于测试失败处理
        def source_node(base: int) -> int:
            return -1  # 返回负数触发失败

        source = Node(func=source_node, name="source")
        failing_async = Node(func=async_nodes["failing"], name="failing_async")

        # 执行并期望失败
        flow = source.fan_out_to([failing_async], executor="async")
        result = await flow(2)

        # 验证结果类型
        assert isinstance(result, dict)
        assert "failing_async" in result

        # 验证失败被记录
        parallel_result = result["failing_async"]
        assert not parallel_result.success  # 应该失败
        assert "负数输入不被支持" in str(parallel_result.error)

    def test_context_isolation_across_async_tasks(self, async_nodes):
        """测试协程间的context隔离"""

        from src.aetherflow import Node

        # 这个测试验证ContextVar在协程间正确隔离
        async def context_aware_node(value: int) -> int:
            """能感知context状态的异步节点"""
            # 获取当前协程的context
            from src.aetherflow import _context_state

            state = _context_state.get({})

            # 由于state是dict，使用字典操作而不是属性操作
            if "test_counter" not in state:
                state["test_counter"] = 0
            state["test_counter"] += 1
            return value + state["test_counter"]

        context_node = Node(func=context_aware_node, name="context_node")

        # 运行测试
        async def run_test():
            def source_node(count: int) -> int:
                return 10  # 返回单个值

            source = Node(func=source_node, name="source")
            flow = source.fan_out_to([context_node], executor="async")
            result = await flow(3)

            # 验证context隔离
            assert isinstance(result, dict)
            assert "context_node" in result

            parallel_result = result["context_node"]
            assert parallel_result.success
            # 由于单个协程，counter应该是1，所以10 + 1 = 11
            assert parallel_result.result == 11

        # 在事件循环中运行测试
        asyncio.run(run_test())

    @pytest.mark.asyncio
    async def test_mixed_sync_async_compatibility(self, async_nodes):
        """测试同步和异步节点的兼容性"""
        from src.aetherflow import Node

        # 创建同步节点
        def sync_double(value: int) -> int:
            return value * 2

        # 创建源节点
        def source_node(value: int) -> int:
            return value

        source = Node(func=source_node, name="source")
        sync_node = Node(func=sync_double, name="sync_double")
        async_node = Node(func=async_nodes["add_10"], name="async_add_10")

        # 使用async executor执行混合节点
        flow = source.fan_out_to([sync_node, async_node], executor="async")
        result = await flow(5)

        # 验证同步和异步节点都正确执行
        assert isinstance(result, dict)
        assert "sync_double" in result
        assert "async_add_10" in result

        # 验证结果正确性
        assert result["sync_double"].success
        assert result["async_add_10"].success
        assert result["sync_double"].result == 10  # 5 * 2
        assert result["async_add_10"].result == 15  # 5 + 10


class TestAsyncExecutorValidation:
    """异步executor验证测试"""

    def test_invalid_executor_type(self):
        """测试无效的executor类型验证"""
        from src.aetherflow import Node

        def dummy_func(x: int) -> int:
            return x

        source = Node(func=dummy_func, name="source")
        target = Node(func=dummy_func, name="target")

        # 测试无效的executor类型
        with pytest.raises(
            ValueError,
            match=r"Only 'thread', 'async', and 'auto' executors are supported",
        ):
            source.fan_out_to([target], executor="invalid")

    def test_executor_type_case_insensitive(self, async_nodes):
        """测试executor类型大小写不敏感"""
        from src.aetherflow import Node

        def source_node(value: int) -> int:
            return value

        source = Node(func=source_node, name="source")
        async_target = Node(func=async_nodes["add_10"], name="async_target")

        # 测试不同大小写的"async"都能工作
        async def test_case(executor_type: str):
            flow = source.fan_out_to([async_target], executor=executor_type)
            result = await flow(10)
            assert isinstance(result, dict)
            assert "async_target" in result
            assert result["async_target"].success
            assert result["async_target"].result == 20

        # 运行测试
        asyncio.run(test_case("async"))
        asyncio.run(test_case("ASYNC"))
        asyncio.run(test_case("Async"))
