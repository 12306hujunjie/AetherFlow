import asyncio
import functools
import inspect
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from dependency_injector import containers, providers
from dependency_injector.wiring import inject
from pydantic import ConfigDict, TypeAdapter, ValidationError, validate_call

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

    retryable = False  # 默认框架异常不重试

    def __init__(
        self, message: str, node_name: str | None = None, **kwargs: Any
    ) -> None:
        self.node_name = node_name
        self.context = kwargs
        super().__init__(message)


class ValidationInputException(AetherFlowException):
    """参数验证异常 - validate_call前置校验失败"""

    retryable = False  # 参数验证失败不应该重试

    def __init__(
        self,
        message: str,
        validation_error: Any = None,
        node_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.validation_error = validation_error
        super().__init__(message, node_name, **kwargs)


class ValidationOutputException(AetherFlowException):
    """返回值验证异常 - validate_call返回值校验失败"""

    retryable = False  # 返回值验证失败不应该重试

    def __init__(
        self,
        message: str,
        validation_error: Any = None,
        node_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.validation_error = validation_error
        super().__init__(message, node_name, **kwargs)


class UserBusinessException(AetherFlowException):
    """用户业务异常基类 - 用户可自定义重试策略"""

    retryable = True  # 默认用户业务异常可重试

    def __init__(
        self,
        message: str,
        retryable: bool | None = None,
        node_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # 允许用户在实例化时覆盖重试策略
        if retryable is not None:
            self.retryable = retryable
        super().__init__(message, node_name, **kwargs)


class NodeExecutionException(AetherFlowException):
    """节点执行异常"""

    def __init__(
        self,
        message: str,
        node_name: str | None = None,
        original_exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        self.original_exception = original_exception
        super().__init__(message, node_name, **kwargs)


class NodeTimeoutException(NodeExecutionException):
    """节点执行超时异常"""

    def __init__(
        self,
        message: str,
        node_name: str | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(message, node_name, **kwargs)


class NodeRetryExhaustedException(NodeExecutionException):
    """节点重试次数耗尽异常"""

    def __init__(
        self,
        message: str,
        node_name: str | None = None,
        retry_count: int | None = None,
        last_exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        self.retry_count = retry_count
        self.last_exception = last_exception
        super().__init__(
            message, node_name, original_exception=last_exception, **kwargs
        )


class LoopControlException(AetherFlowException):
    """循环控制异常基类"""

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
        """判断是否应该重试 - 优先检查retryable属性，否则使用isinstance检查继承关系"""
        # 如果异常有retryable属性，优先使用
        if hasattr(exception, "retryable"):
            return bool(exception.retryable)

        # 否则使用isinstance检查异常是否属于指定类型（包括继承关系）
        return isinstance(exception, self.exception_types)

    def get_delay(self, attempt: int) -> float:
        """计算重试延迟时间（支持指数退避）"""
        delay = self.retry_delay * (self.backoff_factor**attempt)
        return min(delay, self.max_delay)


def custom_validate_call(
    validate_return: bool = True,
    config: ConfigDict | None = None,
    node_name: str | None = None,
) -> Callable[[Callable], Callable]:
    """
    自定义validate_call包装器，支持异步函数并使用Pydantic最佳实践区分输入验证和输出验证异常

    Args:
        validate_return: 是否验证返回值
        config: Pydantic配置
        node_name: 节点名称用于异常信息

    Returns:
        装饰器函数
    """

    def decorator(func: Callable) -> Callable:
        # 获取函数签名
        sig = inspect.signature(func)

        # 检测原始函数是否为异步函数
        is_async_func = inspect.iscoroutinefunction(func)

        # 创建输入验证器 - 只验证参数，不验证返回值
        input_validator = validate_call(
            validate_return=False,
            config=config or ConfigDict(arbitrary_types_allowed=True),
        )(func)

        # 创建返回值验证器（如果需要且有返回值类型注解）
        return_type_adapter = None
        if validate_return and sig.return_annotation != inspect.Signature.empty:
            return_type_adapter = TypeAdapter(sig.return_annotation)

        if is_async_func:
            # 异步函数版本
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    # 第一步：验证输入参数并执行函数，获取协程对象
                    coro = input_validator(*args, **kwargs)
                    # await 协程对象获取实际结果
                    result = await coro
                except ValidationError as e:
                    # 输入参数验证失败
                    raise ValidationInputException(
                        f"输入参数验证失败: {e}",
                        validation_error=e,
                        node_name=node_name or _get_func_name(func),
                    ) from e

                # 第二步：验证返回值（如果需要）
                if return_type_adapter:
                    try:
                        return_type_adapter.validate_python(result)
                    except ValidationError as e:
                        # 返回值验证失败
                        raise ValidationOutputException(
                            f"返回值验证失败: {e}",
                            validation_error=e,
                            node_name=node_name or _get_func_name(func),
                        ) from e

                return result

            return async_wrapper
        else:
            # 同步函数版本（保持原有逻辑）
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    # 第一步：验证输入参数并执行函数
                    result = input_validator(*args, **kwargs)
                except ValidationError as e:
                    # 输入参数验证失败
                    raise ValidationInputException(
                        f"输入参数验证失败: {e}",
                        validation_error=e,
                        node_name=node_name or _get_func_name(func),
                    ) from e

                # 第二步：验证返回值（如果需要）
                if return_type_adapter:
                    try:
                        return_type_adapter.validate_python(result)
                    except ValidationError as e:
                        # 返回值验证失败
                        raise ValidationOutputException(
                            f"返回值验证失败: {e}",
                            validation_error=e,
                            node_name=node_name or _get_func_name(func),
                        ) from e

                return result

            return wrapper

    return decorator


def _get_func_name(func: Any, fallback_name: str | None = None) -> str:
    """安全获取函数名称"""
    if hasattr(func, "__name__"):
        return str(func.__name__)
    elif hasattr(func, "func") and hasattr(func.func, "__name__"):  # partial对象
        return str(func.func.__name__)
    elif hasattr(func, "name"):  # Node对象
        return str(func.name)
    elif fallback_name:
        return fallback_name
    else:
        return "unknown_function"


def retry_decorator(
    config: RetryConfig,
    node_name: str = None,
):
    """重试装饰器

    Args:
        config: RetryConfig 配置模型
        node_name: 节点名称，用于异常信息
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
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

                except (KeyboardInterrupt, SystemExit) as e:
                    # 系统级异常（调试器中断或系统退出），直接重新抛出
                    logger.debug(f"节点 {func_name} 收到系统级异常: {type(e).__name__}")
                    raise
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


# Context variables for asyncio coroutine safety
_context_state: ContextVar[dict | None] = ContextVar("aetherflow_state", default=None)
_context_context: ContextVar[dict | None] = ContextVar(
    "aetherflow_context", default=None
)


# ==================== 智能异步检测系统 ====================


def _is_async_callable(func: Callable) -> bool:
    """智能检测函数是否为异步函数，支持多种场景。

    Args:
        func: 要检测的函数或可调用对象

    Returns:
        bool: True if the function is async, False otherwise
    """
    # 检测直接的协程函数
    if inspect.iscoroutinefunction(func):
        return True

    # 检测 Node 对象中包装的协程函数
    if hasattr(func, "func") and inspect.iscoroutinefunction(func.func):
        return True

    # 检测 partial 对象包装的协程函数
    if hasattr(func, "func") and hasattr(func.func, "__wrapped__"):
        return inspect.iscoroutinefunction(func.func.__wrapped__)

    # 检测装饰器包装的协程函数 - 递归检查多层装饰器
    current_func = func
    while hasattr(current_func, "__wrapped__"):
        current_func = current_func.__wrapped__
        if inspect.iscoroutinefunction(current_func):
            return True

    # 检测装饰器链中的原始函数
    if hasattr(func, "__wrapped__"):
        return _is_async_callable(func.__wrapped__)

    return False


def _analyze_nodes_async_pattern(nodes: list["Node"]) -> dict[str, Any]:
    """分析节点列表的异步模式，提供执行策略建议。

    Args:
        nodes: 节点列表

    Returns:
        dict: 包含异步分析结果和执行建议
    """
    if not nodes:
        return {
            "async_count": 0,
            "sync_count": 0,
            "total_count": 0,
            "async_ratio": 0.0,
            "recommended_executor": "thread",
            "mixed_mode": False,
            "async_nodes": [],
            "sync_nodes": [],
        }

    async_nodes = []
    sync_nodes = []

    for node in nodes:
        if _is_async_callable(node.func):
            async_nodes.append(node.name)
        else:
            sync_nodes.append(node.name)

    total_count = len(nodes)
    async_count = len(async_nodes)
    sync_count = len(sync_nodes)
    async_ratio = async_count / total_count if total_count > 0 else 0.0

    # 执行器推荐策略
    if async_count == 0:
        recommended_executor = "thread"
    elif sync_count == 0:
        recommended_executor = "async"
    else:
        # 混合模式：优先推荐async executor，因为它可以同时处理sync和async
        recommended_executor = "async"

    return {
        "async_count": async_count,
        "sync_count": sync_count,
        "total_count": total_count,
        "async_ratio": async_ratio,
        "recommended_executor": recommended_executor,
        "mixed_mode": async_count > 0 and sync_count > 0,
        "async_nodes": async_nodes,
        "sync_nodes": sync_nodes,
    }


def _get_original_func(func: Callable) -> Callable:
    """递归获取装饰器链中的原始函数。"""
    current = func
    while hasattr(current, "__wrapped__"):
        current = current.__wrapped__
    return current


def _smart_execute_with_fallback(node: "Node", *args, **kwargs):
    """智能执行节点，自动处理同步/异步调用。

    Args:
        node: 要执行的节点
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        执行结果

    Note:
        这是一个同步函数，用于在同步上下文中智能调用异步节点
    """
    if _is_async_callable(node.func):
        # 异步节点需要在事件循环中执行
        try:
            # 尝试获取当前事件循环
            try:
                loop = asyncio.get_running_loop()
                # 如果有运行中的事件循环，在新线程中运行以避免嵌套事件循环问题
                logger.debug(
                    f"Running async node '{node.name}' in sync context with running loop"
                )
                import concurrent.futures

                def run_async():
                    # 创建新事件循环运行异步节点
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        # 通过完整装饰器链调用异步节点
                        coro = node.func(*args, **kwargs)
                        return new_loop.run_until_complete(coro)
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    return future.result()

            except RuntimeError:
                # 没有运行中的事件循环，可以直接使用asyncio.run
                coro = node.func(*args, **kwargs)
                return asyncio.run(coro)

        except Exception as e:
            logger.error(f"Failed to execute async node '{node.name}': {e}")
            raise
    else:
        # 同步节点直接调用
        return node(*args, **kwargs)


async def _smart_execute_async_with_fallback(node: "Node", *args, **kwargs):
    """智能异步执行节点，自动处理同步/异步调用。

    Args:
        node: 要执行的节点
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        执行结果

    Note:
        这是一个异步函数，用于在异步上下文中调用任何类型的节点
    """
    if _is_async_callable(node.func):
        # 异步节点直接await
        return await node.func(*args, **kwargs)
    else:
        # 同步节点直接调用（在异步上下文中）
        return node(*args, **kwargs)


def _get_context_safe_state():
    """Get context-safe state that works in both thread and coroutine environments."""
    try:
        # Try to get from ContextVar first (asyncio coroutines)
        state = _context_state.get()
        return state if state is not None else {}
    except LookupError:
        # Fallback to thread-local if ContextVar not available
        # This happens in thread-based execution
        return {}


def _get_context_safe_context():
    """Get context-safe context that works in both thread and coroutine environments."""
    try:
        # Try to get from ContextVar first (asyncio coroutines)
        context = _context_context.get()
        return context if context is not None else {}
    except LookupError:
        # Fallback to thread-local if ContextVar not available
        return {}


class BaseFlowContext(containers.DeclarativeContainer):
    """Base container for flow context with thread-safe and coroutine-safe dependency injection support."""

    # Use ThreadLocalSingleton for thread-local state isolation
    # Each thread gets its own state dictionary
    state: providers.Provider = providers.ThreadLocalSingleton(dict)
    context: providers.Provider = providers.ThreadLocalSingleton(dict)
    shared_data: providers.Provider = providers.Singleton(dict)

    # Context-safe providers for asyncio coroutines
    async_state: providers.Provider = providers.Factory(_get_context_safe_state)
    async_context: providers.Provider = providers.Factory(_get_context_safe_context)


class Node:
    """A node in the execution graph that supports fluent interface methods."""

    def __init__(
        self,
        func: Callable,
        name: str,
        is_start_node: bool = True,
    ):
        # 配置Pydantic支持任意类型（包括dependency injection的类型）
        self.func = func
        self.name = name
        self.is_start_node = is_start_node

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

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
        # Normalize executor type to lowercase for case-insensitive comparison
        executor_lower = executor.lower()
        if executor_lower not in ["thread", "async", "auto"]:
            raise ValueError(
                "Only 'thread', 'async', and 'auto' executors are supported. ProcessPoolExecutor has been removed to resolve pickle serialization issues."
            )
        return parallel_fan_out(self, nodes, executor_lower, max_workers)

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
        # Normalize executor type to lowercase for case-insensitive comparison
        executor_lower = executor.lower()
        if executor_lower not in ["thread", "async", "auto"]:
            raise ValueError(
                "Only 'thread', 'async', and 'auto' executors are supported. ProcessPoolExecutor has been removed to resolve pickle serialization issues."
            )
        return parallel_fan_out_in(
            self, targets, aggregator, executor_lower, max_workers
        )

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
    """Sequential execution that combines two nodes with intelligent async/sync mixing."""

    def run(*args: Any, **kwargs: Any) -> Any:
        # 智能执行左节点
        left_result = _smart_execute_with_fallback(left, *args, **kwargs)
        # 智能执行右节点，传入左节点结果
        right_result = _smart_execute_with_fallback(right, left_result)
        return right_result

    async def run_async(*args: Any, **kwargs: Any) -> Any:
        # 智能异步执行左节点
        left_result = await _smart_execute_async_with_fallback(left, *args, **kwargs)
        # 智能异步执行右节点，传入左节点结果
        right_result = await _smart_execute_async_with_fallback(right, left_result)
        return right_result

    # 分析节点异步模式，选择最优执行函数
    analysis = _analyze_nodes_async_pattern([left, right])

    # 创建描述性名称
    composition_name = f"({left.name} -> {right.name})"

    # 智能选择执行函数：如果有异步节点但在同步上下文调用，使用同步版本（会内部处理异步）
    # 只有在确定是异步上下文调用时才使用async版本
    return Node(func=run, name=composition_name, is_start_node=False)


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
def execute_target_node(node: Node, input_data: Any) -> ParallelResult:
    """Execute a single target node with the provided input using intelligent async/sync handling."""
    import traceback

    start_time = time.time()

    try:
        # 使用智能执行器处理同步和异步节点
        result = _smart_execute_with_fallback(node, input_data)
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


async def execute_target_node_async(node: Node, input_data: Any) -> ParallelResult:
    """Execute a single target node asynchronously with intelligent async/sync handling."""
    import traceback

    start_time = time.time()

    try:
        # Set context variables for coroutine safety
        current_state = (_context_state.get() or {}).copy()
        current_context = (_context_context.get() or {}).copy()

        # Execute in context
        token_state = _context_state.set(current_state)
        token_context = _context_context.set(current_context)

        try:
            # 使用智能异步执行器处理任何类型的节点
            result = await _smart_execute_async_with_fallback(node, input_data)

            execution_time = time.time() - start_time

            return ParallelResult(
                node_name=node.name,
                success=True,
                result=result,
                execution_time=execution_time,
            )
        finally:
            # Reset context variables
            _context_state.reset(token_state)
            _context_context.reset(token_context)

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
    Parallel fan-out execution with support for both ThreadPoolExecutor and asyncio.

    Args:
        source: Source node to execute first
        targets: List of target nodes for parallel execution
        executor: 'thread', 'async', or 'auto' for automatic selection
        max_workers: Maximum worker threads (ignored for async)

    Returns:
        Node that performs parallel fan-out execution
    """
    if not targets:
        raise ValueError("Target nodes list cannot be empty")

    # Normalize executor type to lowercase for case-insensitive comparison
    executor = executor.lower()

    # Handle 'auto' executor selection
    if executor == "auto":
        # Analyze all nodes (source + targets) for optimal executor choice
        all_nodes = [source] + targets
        analysis = _analyze_nodes_async_pattern(all_nodes)
        recommended_executor = analysis["recommended_executor"]
        logger.info(
            f"Auto-selected executor '{recommended_executor}' based on node analysis: "
            f"{analysis['async_count']} async, {analysis['sync_count']} sync nodes"
        )
        # 只有auto模式才强制使用thread，显式指定async时需要保留
        executor = "thread"

    if executor not in ["thread", "async"]:
        raise ValueError(
            "Only 'thread', 'async', and 'auto' executors are supported. ProcessPoolExecutor has been removed to resolve pickle serialization issues."
        )

    # 定义executor映射
    executor_map = {"thread": ThreadPoolExecutor}

    async def run_async(*args: Any, **kwargs: Any) -> dict[str, ParallelResult]:
        target_names = [t.name for t in targets]
        composition_name = f"({source.name} -> [{', '.join(target_names)}])"
        logger.info(f"Executing Async Parallel Fan-Out: {composition_name}")

        # 执行源节点（使用智能异步执行器）
        source_result = await _smart_execute_async_with_fallback(
            source, *args, **kwargs
        )

        # 设置当前协程的context
        current_state = (_context_state.get() or {}).copy()
        current_context = (_context_context.get() or {}).copy()

        # 为每个任务创建协程
        tasks = []
        for node in targets:
            # 在每个协程中设置独立的context
            task = asyncio.create_task(execute_target_node_async(node, source_result))
            tasks.append(task)

        # 并行执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 收集结果
        parallel_results: dict[str, ParallelResult] = {}
        for i, result in enumerate(results):
            node = targets[i]

            if isinstance(result, Exception):
                import traceback

                error_result_key = _generate_unique_result_key(
                    node.name, parallel_results
                )
                parallel_results[error_result_key] = ParallelResult(
                    node_name=node.name,
                    success=False,
                    error=str(result),
                    error_traceback=traceback.format_exception(
                        type(result), result, result.__traceback__
                    ),
                )
            else:
                result_key = _generate_unique_result_key(
                    result.node_name, parallel_results
                )
                parallel_results[result_key] = result
                logger.debug(
                    f"Collected async result from '{result_key}': success={result.success}"
                )

        logger.info(
            f"Async parallel fan-out completed with {len(parallel_results)} results"
        )
        return parallel_results

    def run_thread(*args: Any, **kwargs: Any) -> dict[str, ParallelResult]:
        target_names = [t.name for t in targets]
        composition_name = f"({source.name} -> [{', '.join(target_names)}])"
        logger.info(f"Executing Thread Parallel Fan-Out: {composition_name}")

        # 执行源节点（使用智能执行器）
        source_result = _smart_execute_with_fallback(source, *args, **kwargs)

        # 执行并行任务
        parallel_results: dict[str, ParallelResult] = {}

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
        logger.info(
            f"Thread parallel fan-out completed with {len(parallel_results)} results"
        )
        return parallel_results

    # 创建新的Node - 默认使用同步版本，内部智能处理异步节点
    target_names = [t.name for t in targets]
    composition_name = f"({source.name} -> [{', '.join(target_names)}])"

    # 根据最终executor类型选择执行函数
    if executor == "async":
        return Node(func=run_async, name=composition_name)
    else:
        return Node(func=run_thread, name=composition_name)


def parallel_fan_in(fan_out_node: Node, aggregator: Node) -> Node:
    """
    Simplified parallel fan-in aggregation with intelligent async/sync handling.

    Args:
        fan_out_node: The fan-out node to execute first
        aggregator: The aggregator node that combines results

    Returns:
        Node that performs fan-in aggregation
    """

    def run(*args: Any, **kwargs: Any) -> Any:
        composition_name = f"({fan_out_node.name} -> {aggregator.name})"
        logger.info(f"Executing Parallel Fan-In: {composition_name}")

        # 执行fan-out节点，获取并行结果（使用智能执行器）
        parallel_results = _smart_execute_with_fallback(fan_out_node, *args, **kwargs)

        # 将并行结果作为参数传递给聚合器（使用智能执行器）
        aggregator_result = _smart_execute_with_fallback(aggregator, parallel_results)

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
    Convenience function that combines fan-out and fan-in into a single operation.

    Args:
        source: Source node to execute first
        targets: List of target nodes for parallel execution
        aggregator: Aggregator node that combines parallel results
        executor: 'thread', 'async', or 'auto' for automatic selection
        max_workers: Maximum worker threads (ignored for async)

    Returns:
        Node that performs complete fan-out-in operation
    """
    # Normalize executor type to lowercase for case-insensitive comparison
    executor = executor.lower()

    # Handle 'auto' executor selection
    if executor == "auto":
        # Analyze all nodes (source + targets + aggregator) for optimal executor choice
        all_nodes = [source] + targets + [aggregator]
        analysis = _analyze_nodes_async_pattern(all_nodes)
        recommended_executor = analysis["recommended_executor"]
        logger.info(
            f"Auto-selected executor '{recommended_executor}' for fan-out-in based on node analysis: "
            f"{analysis['async_count']} async, {analysis['sync_count']} sync nodes"
        )
        # 对于智能混合调用系统，统一使用thread executor但内部智能处理异步
        executor = "thread"

    if executor not in ["thread", "async"]:
        raise ValueError(
            "Only 'thread', 'async', and 'auto' executors are supported. ProcessPoolExecutor has been removed to resolve pickle serialization issues."
        )
    # 创建fan-out节点
    fan_out_node = parallel_fan_out(
        source=source, targets=targets, executor=executor, max_workers=max_workers
    )

    # 创建fan-in节点
    return parallel_fan_in(fan_out_node, aggregator)


def conditional_composition(condition_node: Node, branches: dict[Any, Node]) -> Node:
    """Conditional branching based on boolean output."""

    @inject
    def run(*args: Any, **kwargs: Any) -> Any:
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
    return Node(
        func=run,
        name=f"({condition_node.name} ? {branch_names})",
    )


def repeat_composition(node: Node, times: int, stop_on_error: bool = False) -> Node:
    """重复执行节点的简化版本。

    Args:
        node: 要重复执行的节点
        times: 重复次数
        stop_on_error: 遇到错误时是否立即停止

    Returns:
        包装后的重复执行节点
    """

    # 参数前置验证：立即检查参数，fail-fast原则
    if times <= 0:
        raise ValueError("Repeat times must be greater than 0")
    if not isinstance(node, Node):
        raise TypeError("node must be a Node instance")

    # 创建组合节点的名称
    composition_name = f"({node.name} * {times})"

    # 定义执行函数
    def run(*args: Any, **kwargs: Any) -> Any:
        logger.info(f"--- Executing Repeat Composition: {composition_name} ---")

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

                # 检查是否应该立即停止
                if stop_on_error:
                    logger.error("Stopping immediately due to stop_on_error=True")
                    # 抛出RepeatStopException，让其被重试机制处理
                    raise LoopControlException(
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

    # 使用@node装饰器创建节点，禁用重试避免重复处理
    return Node(func=run, name=composition_name)


def node(
    func: Callable | None = None,
    *,
    retry_count: int = 3,
    name: str | None = None,
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

    # 自定义重试配置（指定可重试的异常类型）
    @node(retry_count=5, retry_delay=2.0, exception_types=(ConnectionError, TimeoutError))
    def api_call_function(data: dict) -> dict:
        # API调用逻辑...
        return call_external_api(data)

    # 用户自定义业务异常（可重试）
    class MyBusinessError(UserBusinessException):
        pass

    @node
    def business_logic_function(data: dict) -> dict:
        if not data:
            raise MyBusinessError("数据为空，可以重试", retryable=True)
        return process_business_data(data)

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
        name: 标识节点名称
        retry_count: 最大重试次数（默认3次）
        retry_delay: 基础重试间隔时间（默认1.0秒）
        exception_types: 需要捕获并重试的异常类型元组（默认空，依赖异常retryable属性）
        backoff_factor: 退避因子，用于指数退避（默认1.0，无退避）
        max_delay: 最大延迟时间（默认60秒）
        enable_retry: 是否启用重试机制（默认True）

    Returns:
        Node实例或装饰器函数

    注意事项：
    - 如果函数使用了BaseFlowContext依赖注入，必须使用@node装饰器
    - 不支持直接使用Node(func)
    - 在使用依赖注入前需要配置容器: container.wire(modules=[__name__])
    - 重试机制默认启用，可通过enable_retry=False禁用
    - 重试仅对指定的异常类型生效，其他异常会立即抛出
    """
    config = RetryConfig(
        retry_count, retry_delay, exception_types, backoff_factor, max_delay
    )

    @functools.wraps(Node)
    def decorator(f: Callable) -> Node:
        node_name = name or _get_func_name(f, "unnamed_node")

        # 使用functools.reduce应用装饰器链
        decorators = [
            inject,
            custom_validate_call(
                validate_return=True,
                config=ConfigDict(arbitrary_types_allowed=True),
                node_name=node_name,
            ),
        ]
        if enable_retry:
            decorators.append(retry_decorator(config=config, node_name=node_name))

        decorated_func = functools.reduce(lambda func, deco: deco(func), decorators, f)

        return Node(func=decorated_func, name=node_name)

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
    # 验证装饰器：自定义验证异常处理
    "custom_validate_call",
    # 上下文：自定义依赖注入
    "BaseFlowContext",
    # 并行执行结果模型
    "ParallelResult",
    # 异常类
    "AetherFlowException",
    "ValidationInputException",
    "ValidationOutputException",
    "UserBusinessException",
    "NodeExecutionException",
    "NodeTimeoutException",
    "NodeRetryExhaustedException",
    "LoopControlException",
    # 重试相关
    "RetryConfig",
]
