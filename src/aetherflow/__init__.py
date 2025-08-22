import inspect
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, Any, List, Dict

from dependency_injector import containers, providers
from dependency_injector.wiring import inject, Provide
from pydantic import validate_call, ConfigDict


logger = logging.getLogger("aetherflow")

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
        self.func = validate_call(
            validate_return=True,
            config=ConfigDict(arbitrary_types_allowed=True)
        )(inject(func))
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
                   max_workers: int = None) -> 'Node':
        """Fan out to multiple nodes for parallel execution."""
        return parallel_fan_out(self, nodes, executor, max_workers)

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


def parallel_fan_out(source: Node, targets: List[Node], executor: str = 'thread', max_workers: int = None) -> Node:
    """Parallel fan-out execution that distributes to multiple nodes."""
    executor_map = {'thread': ThreadPoolExecutor, 'process': ProcessPoolExecutor}

    def run(state: dict, context: BaseFlowContext = Provide[BaseFlowContext]):
        target_names = [t.name for t in targets]
        composition_name = f"({source.name} -> [{', '.join(target_names)}])"
        print(f"--- Executing Parallel Fan-Out: {composition_name} ---")

        # Execute source first
        source.run(state, context)

        # Store pre-fan-out state for fan-in
        original_state = dict(state)
        state['__pre_fan_out_state'] = original_state

        print(f"  - Fanning out with state keys: {list(original_state.keys())} using '{executor}' executor.")

        # Generate better keys for parallel results
        def generate_result_key(node: Node, index: int, all_nodes: List[Node]) -> str:
            # Count occurrences of this node name
            node_count = sum(1 for n in all_nodes if n.name == node.name)
            if node_count == 1:
                # If node name is unique, use it directly
                return node.name
            else:
                # If there are duplicates, use a cleaner format
                return f"{node.name}[{index}]"

        def run_isolated_node(node: Node, node_index: int, initial_state: dict, node_context: BaseFlowContext):
            # Create isolated state for this thread
            thread_state = dict(initial_state)
            result = node.run(thread_state, node_context)

            # Generate a clean result key
            result_key = generate_result_key(node, node_index, targets)

            # Return the result key and its result
            return result_key, thread_state.get(node.name, result)

        with executor_map[executor](max_workers=max_workers) as exec_instance:
            futures = [exec_instance.submit(run_isolated_node, node, i, original_state, context)
                       for i, node in enumerate(targets)]
            parallel_results = {}

            for future in as_completed(futures):
                node_key, node_result = future.result()
                parallel_results[node_key] = node_result
                print(f"  - Collected result from '{node_key}': {node_result}")

        # Store parallel results in state for potential fan-in
        state['__parallel_results'] = parallel_results
        print(f"  - Stored parallel results: {parallel_results}")

        # Return the parallel results
        return parallel_results

    # Create a new Node with the fan-out execution
    target_names = [t.name for t in targets]
    return Node(func=run, name=f"({source.name} -> [{', '.join(target_names)}])")


def parallel_fan_in(fan_out_node: Node, aggregator: Node) -> Node:
    """Parallel fan-in aggregation that combines results from fan-out."""

    @inject
    def _execute(state: dict, context: BaseFlowContext = Provide[BaseFlowContext]):
        composition_name = f"({fan_out_node.name} -> {aggregator.name})"
        print(f"--- Executing Parallel Fan-In: {composition_name} ---")

        # Execute the fan-out first
        fan_out_node.run(state, context)

        # Execute aggregator
        aggregator_result = aggregator.run(state, context)

        # Update main state with aggregator results
        if isinstance(aggregator_result, dict):
            state.update(aggregator_result)
        elif aggregator_result is not None:
            state[aggregator.name] = aggregator_result
        return aggregator_result

    return Node(func=_execute, name=f"({fan_out_node.name} -> {aggregator.name})")


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
    return Node(func=func, name=func.__name__)


# Export list - 只暴露用户需要的公共接口
__all__ = [
    # 装饰器：创建节点
    'node',

    # 上下文：自定义依赖注入
    'BaseFlowContext',

    # 核心类：节点和链接
    'Node'
]
