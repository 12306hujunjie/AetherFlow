import inspect
import logging
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from dependency_injector import containers, providers
from dependency_injector.wiring import inject
from pydantic import ConfigDict, validate_call

logger = logging.getLogger("aetherflow")


@dataclass
class ParallelResult:
    """Pydantic model for recording parallel execution results and exception stacks."""

    node_name: str
    success: bool
    result: Any = None
    error: str | None = None
    error_traceback: str | None = None
    execution_time: float | None = None


# ==================== 异常类型体系 ====================


class AetherFlowException(Exception):
    """AetherFlow框架基础异常类"""

    def __init__(self, message: str, node_name: str = None, **kwargs):
        self.node_name = node_name
        self.context = kwargs
        super().__init__(message)


class NodeExecutionException(AetherFlowException):
    """节点执行异常"""

    def __init__(
        self,
        message: str,
        node_name: str = None,
        original_exception: Exception = None,
        **kwargs,
    ):
        self.original_exception = original_exception
        super().__init__(message, node_name, **kwargs)


class NodeTimeoutException(NodeExecutionException):
    """节点执行超时异常"""

    def __init__(
        self,
        message: str,
        node_name: str = None,
        timeout_seconds: float = None,
        **kwargs,
    ):
        self.timeout_seconds = timeout_seconds
        super().__init__(message, node_name, **kwargs)


class NodeRetryExhaustedException(NodeExecutionException):
    """节点重试次数耗尽异常"""

    def __init__(
        self,
        message: str,
        node_name: str = None,
        retry_count: int = None,
        last_exception: Exception = None,
        **kwargs,
    ):
        self.retry_count = retry_count
        self.last_exception = last_exception
        super().__init__(message, node_name, last_exception, **kwargs)


class DependencyInjectionException(AetherFlowException):
    """依赖注入异常"""

    pass


class LoopControlException(AetherFlowException):
    """循环控制异常基类"""

    pass


class RepeatStopException(LoopControlException):
    """重复执行停止异常"""

    pass


# ==================== 重试装饰器 ====================


class RetryConfig:
    """重试配置类"""

    def __init__(
        self,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        exception_types: tuple = (Exception,),
        backoff_factor: float = 1.0,
        max_delay: float = 60.0,
    ):
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.exception_types = exception_types
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay

    def should_retry(self, exception: Exception) -> bool:
        """判断是否应该重试"""
        return isinstance(exception, self.exception_types)

    def get_delay(self, attempt: int) -> float:
        """计算重试延迟时间（支持指数退避）"""
        delay = self.retry_delay * (self.backoff_factor**attempt)
        return min(delay, self.max_delay)


def _get_func_name(func, fallback_name: str = None) -> str:
    """安全获取函数名称"""
    if hasattr(func, "__name__"):
        return func.__name__
    elif hasattr(func, "func") and hasattr(func.func, "__name__"):  # partial对象
        return func.func.__name__
    elif hasattr(func, "name"):  # Node对象
        return func.name
    elif fallback_name:
        return fallback_name
    else:
        return "unknown_function"


def retry_decorator(
    retry_count: int = 3,
    retry_delay: float = 1.0,
    exception_types: tuple = (Exception,),
    backoff_factor: float = 1.0,
    max_delay: float = 60.0,
    node_name: str = None,
):
    """重试装饰器

    Args:
        retry_count: 最大重试次数
        retry_delay: 基础重试间隔时间（秒）
        exception_types: 需要捕获并重试的异常类型元组
        backoff_factor: 退避因子，用于指数退避
        max_delay: 最大延迟时间
        node_name: 节点名称，用于异常信息
    """
    config = RetryConfig(
        retry_count, retry_delay, exception_types, backoff_factor, max_delay
    )

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            func_name = node_name or _get_func_name(func)

            for attempt in range(config.retry_count + 1):  # +1 因为包含初始尝试
                try:
                    logger.debug(
                        f"执行节点 {func_name}，尝试 {attempt + 1}/{config.retry_count + 1}"
                    )
                    result = func(*args, **kwargs)

                    if attempt > 0:
                        logger.info(f"节点 {func_name} 在第 {attempt + 1} 次尝试后成功")

                    return result

                except Exception as e:
                    last_exception = e

                    # 检查是否应该重试
                    if not config.should_retry(e):
                        logger.error(f"节点 {func_name} 遇到不可重试异常: {e}")
                        raise NodeExecutionException(
                            f"节点执行失败，异常类型不支持重试: {type(e).__name__}",
                            node_name=func_name,
                            original_exception=e,
                        ) from e

                    # 如果是最后一次尝试，抛出重试耗尽异常
                    if attempt == config.retry_count:
                        logger.error(
                            f"节点 {func_name} 重试 {config.retry_count} 次后仍然失败: {e}"
                        )
                        raise NodeRetryExhaustedException(
                            f"节点 {func_name} 重试次数耗尽，最后异常: {type(e).__name__}: {e}",
                            node_name=func_name,
                            retry_count=config.retry_count,
                            last_exception=e,
                        ) from e

                    # 计算延迟时间并等待
                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"节点 {func_name} 第 {attempt + 1} 次尝试失败: {e}，"
                        f"{delay:.2f}秒后重试"
                    )
                    time.sleep(delay)

            # 理论上不会到达这里，但为了安全起见
            raise NodeRetryExhaustedException(
                "节点重试逻辑异常",
                node_name=node_name or _get_func_name(func, "unknown_node"),
                retry_count=config.retry_count,
                last_exception=last_exception,
            )

        return wrapper

    return decorator


class BaseFlowContext(containers.DeclarativeContainer):
    """Base container for flow context with thread-safe dependency injection support."""

    # Use ThreadLocalSingleton for thread-local state isolation
    # Each thread gets its own state dictionary
    state = providers.ThreadLocalSingleton(dict)
    context = providers.ThreadLocalSingleton(dict)
    shared_data = providers.Singleton(dict)


class Node:
    """A node in the execution graph that supports fluent interface methods."""

    def __init__(
        self,
        func: Callable,
        name: str = None,
        is_start_node: bool = True,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        exception_types: tuple = (Exception,),
        backoff_factor: float = 1.0,
        max_delay: float = 60.0,
        enable_retry: bool = True,
    ):
        # 配置Pydantic支持任意类型（包括dependency injection的类型）
        self.func = func
        self.name = name or _get_func_name(func, "unnamed_node")
        self.is_start_node = is_start_node
        self.enable_retry = enable_retry

        # 重试配置
        self.retry_config = RetryConfig(
            retry_count=retry_count,
            retry_delay=retry_delay,
            exception_types=exception_types,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
        )

        # 如果启用重试，应用重试装饰器
        if self.enable_retry:
            self._wrapped_func = retry_decorator(
                retry_count=retry_count,
                retry_delay=retry_delay,
                exception_types=exception_types,
                backoff_factor=backoff_factor,
                max_delay=max_delay,
                node_name=self.name,
            )(func)
        else:
            self._wrapped_func = func

    def __call__(self, *args, **kwargs):
        return self._wrapped_func(*args, **kwargs)

    def __getstate__(self):
        """支持pickle序列化"""
        state = self.__dict__.copy()
        # 序列化时保留所有状态，包括装饰函数
        # retry_decorator已经通过设置__module__和__qualname__支持序列化
        return state

    def __setstate__(self, state):
        """支持pickle反序列化"""
        self.__dict__.update(state)
        # 反序列化后直接恢复状态，无需重新创建装饰函数

    @property
    def input_signature(self):
        return inspect.signature(self.func)

    def then(self, next_node: "Node") -> "Node":
        """Chain this node with another node for sequential execution."""
        return sequential_composition(self, next_node)

    def fan_out_to(
        self,
        nodes: list["Node"],
        executor: str = "thread",
        max_workers: int | None = None,
    ) -> "Node":
        """Fan out to multiple nodes for parallel execution."""
        return parallel_fan_out(self, nodes, executor, max_workers)

    def fan_in(self, aggregator: "Node") -> "Node":
        """Aggregate results using the specified aggregator node."""
        return parallel_fan_in(self, aggregator)

    def fan_out_in(
        self,
        targets: list["Node"],
        aggregator: "Node",
        executor: str = "thread",
        max_workers: int | None = None,
    ) -> "Node":
        """Complete fan-out and fan-in operation in one step."""
        return parallel_fan_out_in(self, targets, aggregator, executor, max_workers)

    def branch_on(self, conditions: dict[bool, "Node"]) -> "Node":
        """Branch execution based on the boolean output of this node."""
        return conditional_composition(self, conditions)

    def repeat(self, times: int, stop_on_error: bool = False) -> "Node":
        """重复执行此节点。

        Args:
            times: 重复次数
            stop_on_error: 遇到错误时是否立即停止
        """
        return repeat_composition(self, times, stop_on_error)

    def __repr__(self) -> str:
        return f"Node(name='{self.name}')"


def sequential_composition(left: Node, right: Node) -> Node:
    """Sequential execution that combines two nodes with type safety and intelligent retry configuration inheritance."""

    def run(*args, **kwargs):
        # 输入传给左节点
        left_result = left(*args, **kwargs)
        # 左节点输出传给右节点
        right_result = right(left_result)
        return right_result

    # 最严格配置优先原则：如果任一子节点禁用重试，组合节点也禁用重试
    enable_retry = left.enable_retry and right.enable_retry

    # 如果两个节点都启用重试，则合并重试配置（采用最保守策略）
    if enable_retry:
        # 重试次数取最小值（最保守）
        retry_count = min(left.retry_config.retry_count, right.retry_config.retry_count)

        # 重试延迟取最大值（更保守的等待时间）
        retry_delay = max(left.retry_config.retry_delay, right.retry_config.retry_delay)

        # 退避因子取最大值（更保守的退避策略）
        backoff_factor = max(
            left.retry_config.backoff_factor, right.retry_config.backoff_factor
        )

        # 最大延迟取最小值（更保守的上限）
        max_delay = min(left.retry_config.max_delay, right.retry_config.max_delay)

        # 异常类型取交集（只有两个节点都能处理的异常类型才重试）
        exception_types = tuple(
            set(left.retry_config.exception_types)
            & set(right.retry_config.exception_types)
        )

        # 如果没有共同的异常类型，则禁用重试
        if not exception_types:
            enable_retry = False
            retry_count = 0
            retry_delay = 0
            backoff_factor = 1.0
            max_delay = 60.0
            exception_types = (Exception,)
    else:
        # 如果禁用重试，使用默认配置（不会实际使用）
        retry_count = 0
        retry_delay = 0
        backoff_factor = 1.0
        max_delay = 60.0
        exception_types = (Exception,)

    # 创建描述性名称
    composition_name = f"({left.name} -> {right.name})"

    return Node(
        func=run,
        name=composition_name,
        is_start_node=False,
        enable_retry=enable_retry,
        retry_count=retry_count,
        retry_delay=retry_delay,
        exception_types=exception_types,
        backoff_factor=backoff_factor,
        max_delay=max_delay,
    )


def _generate_unique_result_key(base_name: str, existing_results: dict) -> str:
    """生成唯一的结果键，避免重复覆盖

    Args:
        base_name: 基础键名（通常是节点名称）
        existing_results: 现有的结果字典

    Returns:
        唯一的键名，如果base_name无冲突则返回原名，否则返回带数字后缀的名称
    """
    if base_name not in existing_results:
        return base_name

    counter = 1
    unique_key = f"{base_name}[{counter}]"
    while unique_key in existing_results:
        counter += 1
        unique_key = f"{base_name}[{counter}]"

    return unique_key


# 定义并行任务执行函数
def execute_target_node(node: Node, input_data):
    """Execute a single target node with the provided input."""
    import traceback

    start_time = time.time()

    try:
        result = node(input_data)
        execution_time = time.time() - start_time

        return ParallelResult(
            node_name=node.name,
            success=True,
            result=result,
            execution_time=execution_time,
        )
    except Exception as e:
        execution_time = time.time() - start_time
        error_traceback = traceback.format_exc()
        logger.error(f"Node '{node.name}' failed: {e}")

        return ParallelResult(
            node_name=node.name,
            success=False,
            error=str(e),
            error_traceback=error_traceback,
            execution_time=execution_time,
        )


def parallel_fan_out(
    source: Node,
    targets: list[Node],
    executor: str = "thread",
    max_workers: int | None = None,
) -> Node:
    """
    Simplified parallel fan-out execution with direct parameter passing.

    Args:
        source: Source node to execute first
        targets: List of target nodes for parallel execution
        executor: 'thread' or 'process' executor type
        max_workers: Maximum worker threads/processes

    Returns:
        Node that performs parallel fan-out execution
    """
    if not targets:
        raise ValueError("Target nodes list cannot be empty")

    executor_map = {"thread": ThreadPoolExecutor, "process": ProcessPoolExecutor}

    def run(*args, **kwargs):
        target_names = [t.name for t in targets]
        composition_name = f"({source.name} -> [{', '.join(target_names)}])"
        logger.info(f"Executing Parallel Fan-Out: {composition_name}")

        # 执行源节点
        source_result = source(*args, **kwargs)

        # 执行并行任务
        parallel_results = {}

        with executor_map[executor](max_workers=max_workers) as executor_instance:
            # 提交所有并行任务
            future_to_node = {
                executor_instance.submit(execute_target_node, node, source_result): node
                for node in targets
            }

            # 收集结果
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    parallel_result = future.result()

                    # 生成唯一的结果键
                    result_key = _generate_unique_result_key(
                        parallel_result.node_name, parallel_results
                    )

                    parallel_results[result_key] = parallel_result
                    logger.debug(
                        f"Collected result from '{result_key}': success={parallel_result.success}"
                    )

                except Exception as e:
                    import traceback

                    logger.error(f"Failed to get result from '{node.name}': {e}")

                    # 生成唯一的结果键以避免异常情况下的键覆盖
                    error_result_key = _generate_unique_result_key(
                        node.name, parallel_results
                    )
                    parallel_results[error_result_key] = ParallelResult(
                        node_name=node.name,
                        success=False,
                        error=str(e),
                        error_traceback=traceback.format_exc(),
                    )

        # 返回并行结果
        logger.info(f"Parallel fan-out completed with {len(parallel_results)} results")
        return parallel_results

    # 创建新的Node
    target_names = [t.name for t in targets]
    composition_name = f"({source.name} -> [{', '.join(target_names)}])"

    return Node(func=run, name=composition_name)


def parallel_fan_in(fan_out_node: Node, aggregator: Node) -> Node:
    """
    Simplified parallel fan-in aggregation with direct parameter passing.

    Args:
        fan_out_node: The fan-out node to execute first
        aggregator: The aggregator node that combines results

    Returns:
        Node that performs fan-in aggregation
    """

    def run(*args, **kwargs):
        composition_name = f"({fan_out_node.name} -> {aggregator.name})"
        logger.info(f"Executing Parallel Fan-In: {composition_name}")

        # 执行fan-out节点，获取并行结果
        parallel_results = fan_out_node(*args, **kwargs)

        # 将并行结果作为参数传递给聚合器
        aggregator_result = aggregator(parallel_results)

        logger.info("Fan-in aggregation completed successfully")
        return aggregator_result

    return Node(func=run, name=f"({fan_out_node.name} -> {aggregator.name})")


def parallel_fan_out_in(
    source: Node,
    targets: list[Node],
    aggregator: Node,
    executor: str = "thread",
    max_workers: int | None = None,
) -> Node:
    """
    Simplified convenience function that combines fan-out and fan-in into a single operation.

    Args:
        source: Source node to execute first
        targets: List of target nodes for parallel execution
        aggregator: Aggregator node that combines parallel results
        executor: 'thread' or 'process' executor type
        max_workers: Maximum worker threads/processes

    Returns:
        Node that performs complete fan-out-in operation
    """
    # 创建fan-out节点
    fan_out_node = parallel_fan_out(
        source=source, targets=targets, executor=executor, max_workers=max_workers
    )

    # 创建fan-in节点
    return parallel_fan_in(fan_out_node, aggregator)


def conditional_composition(condition_node: Node, branches: dict[Any, Node]) -> Node:
    """Conditional branching based on boolean output."""

    @inject
    def run(*args, **kwargs):
        branch_names = {k: v.name for k, v in branches.items()}
        composition_name = f"({condition_node.name} ? {branch_names})"
        logger.info(f"--- Executing Conditional Branch: {composition_name} ---")

        # Execute condition node
        condition_result = condition_node(*args, **kwargs)

        # Execute the appropriate branch
        if condition_result in branches:
            selected_branch = branches[condition_result]
            logger.info(
                f"Condition is {condition_result}, executing branch: {selected_branch.name}"
            )
            return selected_branch()
        else:
            msg = f"No branch defined for condition result: {condition_result}"
            logger.error(msg)
            raise ValueError(msg)

    # Create a new Node with the conditional execution
    branch_names = {k: v.name for k, v in branches.items()}
    return Node(func=run, name=f"({condition_node.name} ? {branch_names})")


def repeat_composition(node: Node, times: int, stop_on_error: bool = False) -> Node:
    """重复执行节点的简化版本。

    Args:
        node: 要重复执行的节点
        times: 重复次数
        stop_on_error: 遇到错误时是否立即停止

    Returns:
        包装后的重复执行节点
    """

    def run(*args, **kwargs):
        composition_name = f"({node.name} * {times})"
        logger.info(f"--- Executing Repeat Composition: {composition_name} ---")

        # 参数验证
        if times <= 0:
            raise ValueError("Repeat times must be greater than 0")

        last_result = None
        errors = []

        for i in range(times):
            logger.info(f"  - Iteration {i + 1}/{times}")

            try:
                # 正常执行
                if i == 0:
                    result = node(*args, **kwargs)
                else:
                    result = node(last_result)

                # 成功执行
                last_result = result
                logger.debug(f"Iteration {i + 1} completed successfully")

            except Exception as e:
                errors.append(e)
                logger.error(f"Iteration {i + 1} failed: {e}")

                # 检查是否应该停止
                if stop_on_error:
                    logger.error("Stopping due to stop_on_error=True")
                    raise RepeatStopException(
                        f"Execution stopped due to error at iteration {i + 1}: {e}"
                    ) from e

                # 继续执行，但使用上一次的成功结果
                logger.info(
                    f"Continuing with last successful result from iteration {i}"
                )

        # 正常完成
        logger.info(
            f"Repeat composition completed. Iterations: {times}, Errors: {len(errors)}"
        )

        return last_result

    return Node(func=run, name=f"({node.name} * {times})")


def node(
    func: Callable = None,
    *,
    retry_count: int = 3,
    name: str = None,
    retry_delay: float = 1.0,
    exception_types: tuple = (Exception,),
    backoff_factor: float = 1.0,
    max_delay: float = 60.0,
    enable_retry: bool = True,
) -> Node | Callable:
    """
    装饰器：从函数创建Node，支持依赖注入、类型验证和重试机制。

    **这是使用依赖注入的标准和唯一推荐方式！**

    使用方式：
    ```python
    # 基本用法
    @node
    def my_processing_function(data: dict, state: dict = Provide[BaseFlowContext.state]) -> dict:
        # 你的处理逻辑...


        result = process_data(data)

        state['last_result'] = result
        return result

    # 自定义重试配置
    @node(retry_count=5, retry_delay=2.0, exception_types=(ValueError, TypeError))
    def critical_function(data: dict) -> dict:
        # 关键处理逻辑...
        return process_critical_data(data)

    # 禁用重试
    @node(enable_retry=False)
    def no_retry_function(data: dict) -> dict:
        # 不需要重试的处理逻辑...
        return data

    # 使用then链式调用
    flow = my_node1.then(my_node2).then(my_node3)

    result = flow(input_data)

    ```



    Args:

        func: 要装饰的函数
        retry_count: 最大重试次数（默认3次）
        retry_delay: 基础重试间隔时间（默认1.0秒）
        exception_types: 需要捕获并重试的异常类型元组（默认所有异常）
        backoff_factor: 退避因子，用于指数退避（默认1.0，无退避）
        max_delay: 最大延迟时间（默认60秒）
        enable_retry: 是否启用重试机制（默认True）

    Returns:
        Node实例或装饰器函数

    注意事项：
    - 如果函数使用了BaseFlowContext依赖注入，必须使用@node装饰器
    - 手动创建Node(func)不支持依赖注入，仅用于简单函数
    - 在使用依赖注入前需要配置容器: container.wire(modules=[__name__])
    - 重试机制默认启用，可通过enable_retry=False禁用
    - 重试仅对指定的异常类型生效，其他异常会立即抛出
    """

    def decorator(f: Callable) -> Node:
        # 尝试应用完整的验证和注入
        validated_func = validate_call(
            validate_return=True, config=ConfigDict(arbitrary_types_allowed=True)
        )(inject(f))

        return Node(
            func=validated_func,
            name=name or _get_func_name(f, "unnamed_node"),
            retry_count=retry_count,
            retry_delay=retry_delay,
            exception_types=exception_types,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
            enable_retry=enable_retry,
        )

    # 支持两种调用方式：@node 和 @node(...)
    if func is None:
        # @node(...) 带参数调用
        return decorator
    else:
        # @node 直接调用
        return decorator(func)


# Export list - 只暴露用户需要的公共接口
__all__ = [
    # 装饰器：创建节点
    "node",
    # 上下文：自定义依赖注入
    "BaseFlowContext",
    # 并行执行结果模型
    "ParallelResult",
    # 异常类
    "AetherFlowException",
    "NodeExecutionException",
    "NodeTimeoutException",
    "NodeRetryExhaustedException",
    "DependencyInjectionException",
    "LoopControlException",
    "RepeatStopException",
    # 重试相关
    "RetryConfig",
    "retry_decorator",
]
