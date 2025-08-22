import inspect
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Any, List, Dict, Optional

from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide
from pydantic import validate_call, ConfigDict

logger = logging.getLogger("aetherflow")



@dataclass
class ParallelResult:
    """Pydantic model for recording parallel execution results and exception stacks."""
    node_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    execution_time: Optional[float] = None


class LoopControlException(Exception):
    pass


class BaseFlowContext(containers.DeclarativeContainer):
    """Base container for flow context with thread-safe dependency injection support."""
    # Use ThreadLocalSingleton for thread-local state isolation
    # Each thread gets its own state dictionary
    state = providers.ThreadLocalSingleton(dict)
    context = providers.ThreadLocalSingleton(dict)
    shared_data = providers.Singleton(dict)


class Node:
    """A node in the execution graph that supports fluent interface methods."""

    def __init__(self, func: Callable, name: str = None, is_start_node: bool = True):
        # 配置Pydantic支持任意类型（包括dependency injection的类型）
        self.func = func
        self.name = name or func.__name__
        self.is_start_node = is_start_node

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
    @property  
    def input_signature(self):
        return inspect.signature(self.func)

    def then(self, next_node: 'Node') -> 'Node':
        """Chain this node with another node for sequential execution."""
        return sequential_composition(self, next_node)

    def fan_out_to(self, nodes: List['Node'], executor: str = 'thread',
                   max_workers: Optional[int] = None) -> 'Node':
        """Fan out to multiple nodes for parallel execution."""
        return parallel_fan_out(self, nodes, executor, max_workers)
    
    def fan_in(self, aggregator: 'Node') -> 'Node':
        """Aggregate results using the specified aggregator node."""
        return parallel_fan_in(self, aggregator)
    
    def fan_out_in(self, targets: List['Node'], aggregator: 'Node',
                   executor: str = 'thread',
                   max_workers: Optional[int] = None) -> 'Node':
        """Complete fan-out and fan-in operation in one step."""
        return parallel_fan_out_in(self, targets, aggregator, executor, max_workers)

    def branch_on(self, conditions: Dict[bool, 'Node']) -> 'Node':
        """Branch execution based on the boolean output of this node."""
        return conditional_composition(self, conditions)

    def repeat(self, times: int) -> 'Node':
        """Repeat this node for a fixed number of times."""
        return repeat_composition(self, times)

    def __repr__(self) -> str:
        return f"Node(name='{self.name}')"


def sequential_composition(left: Node, right: Node) -> Node:
    """Sequential execution that combines two nodes with type safety."""

    def run(*args, **kwargs):
        # 输入传给左节点
        left_result = left(*args, **kwargs)
        # 左节点输出传给右节点
        right_result = right(left_result)
        return right_result

    return Node(run, is_start_node=False)


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
            execution_time=execution_time
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
            execution_time=execution_time
        )


def parallel_fan_out(source: Node, targets: List[Node],
                    executor: str = 'thread',
                    max_workers: Optional[int] = None) -> Node:
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
    
    executor_map = {'thread': ThreadPoolExecutor, 'process': ProcessPoolExecutor}

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
                    result_key = parallel_result.node_name

                    counter = 1
                    while result_key in parallel_results:
                        result_key = f"{parallel_result.node_name}[{counter}]"
                        counter += 1
                    
                    parallel_results[result_key] = parallel_result
                    logger.debug(f"Collected result from '{result_key}': success={parallel_result.success}")
                    
                except Exception as e:
                    import traceback
                    logger.error(f"Failed to get result from '{node.name}': {e}")

                    parallel_results[node.name] = ParallelResult(
                        node_name=node.name,
                        success=False,
                        error=str(e),
                        error_traceback=traceback.format_exc()
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
        
        logger.info(f"Fan-in aggregation completed successfully")
        return aggregator_result
    
    return Node(func=run, name=f"({fan_out_node.name} -> {aggregator.name})")


def parallel_fan_out_in(source: Node, targets: List[Node], aggregator: Node,
                       executor: str = 'thread',
                       max_workers: Optional[int] = None) -> Node:
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
        source=source,
        targets=targets,
        executor=executor,
        max_workers=max_workers
    )
    
    # 创建fan-in节点
    return parallel_fan_in(fan_out_node, aggregator)


def conditional_composition(condition_node: Node, branches: Dict[bool, Node]) -> Node:
    """Conditional branching based on boolean output."""

    @inject
    def _execute(state: dict, context: BaseFlowContext = Provide[BaseFlowContext]):
        branch_names = {k: v.name for k, v in branches.items()}
        composition_name = f"({condition_node.name} ? {branch_names})"
        print(f"--- Executing Conditional Branch: {composition_name} ---")

        # Execute condition node
        result = condition_node.run(state, context)

        # Get boolean result - either from the direct return value or from state
        condition_result = None
        if isinstance(result, bool):
            condition_result = result
        elif isinstance(condition_node, Node):
            # For simple nodes, check if they returned a boolean directly or stored it in state
            condition_result = state.get(condition_node.name)

        if not isinstance(condition_result, bool):
            raise ValueError(
                f"Condition node '{condition_node.name}' must return a boolean value, got {type(condition_result)}")

        # Execute the appropriate branch
        if condition_result in branches:
            selected_branch = branches[condition_result]
            print(f"  - Condition is {condition_result}, executing branch: {selected_branch.name}")
            return selected_branch.run(state, context)
        else:
            print(f"  - No branch defined for condition result: {condition_result}")
            return None

    # Create a new Node with the conditional execution
    branch_names = {k: v.name for k, v in branches.items()}
    return Node(func=_execute, name=f"({condition_node.name} ? {branch_names})")


def repeat_composition(node: Node, times: int) -> Node:
    """Repeat a node for a fixed number of times with early exit support."""

    @inject
    def _execute(state: dict, context: BaseFlowContext = Provide[BaseFlowContext]):
        composition_name = f"({node.name} * {times})"
        print(f"--- Executing Repeat Composition: {composition_name} ---")

        last_result = None
        for i in range(times):
            print(f"  - Iteration {i + 1}/{times}")
            result = node.run(state, context)

            # Store the last valid result (not a control signal)
            last_result = result

        # Return the last valid result or None if no valid results were produced
        return last_result

    return Node(func=_execute, name=f"({node.name} * {times})")


def node(func: Callable) -> Node:
    """Decorator to create a Node from a function."""
    func = validate_call(
        validate_return=True,
        config=ConfigDict(arbitrary_types_allowed=True)
    )(inject(func))
    return Node(func=func, name=func.__name__)


# Export list - 只暴露用户需要的公共接口
__all__ = [
    # 装饰器：创建节点
    'node',

    # 上下文：自定义依赖注入
    'BaseFlowContext',

    # 核心类：节点和链接
    'Node',
    
    # 并行执行结果模型
    'ParallelResult',
    
    # 并行组合函数
    'parallel_fan_out',
    'parallel_fan_in',
    'parallel_fan_out_in',
    
    # 其他组合函数
    'sequential_composition',
    'conditional_composition',
    'repeat_composition'
]
