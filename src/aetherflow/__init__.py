import inspect
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from enum import Enum
from typing import Callable, Any, List, Dict, Union

from dependency_injector import containers, providers


# --- 1. Loop Control Enum ---

class LoopControl(Enum):
    """Special signals for controlling loop execution."""
    BREAK = "BREAK"


# --- 2. Containers for Dependency Injection ---

class BaseFlowContext(containers.DeclarativeContainer):
    """Base container for flow context with thread-safe dependency injection support."""

    # Use ThreadLocalSingleton for thread-local state isolation
    # Each thread gets its own state dictionary
    state = providers.ThreadLocalSingleton(dict)
    context = providers.ThreadLocalSingleton(dict)
    shared_data = providers.Singleton(dict)


class AppContext(BaseFlowContext):
    """Main application context with thread-safe dependency injection."""
    pass


# --- 4. Node Implementation ---

class Node:
    """A node in the execution graph that supports fluent interface methods."""

    def __init__(self, func: Callable = None, name: str = None):
        self.func = func
        self.name = name or (func.__name__ if func else "unnamed")
        self.params = inspect.signature(func).parameters if func else {}

    def run(self, state: Dict[str, Any], context: BaseFlowContext = None) -> Dict[str, Any]:
        """Run the node with dependency injection for context."""
        injected_context = context or AppContext()
        result = self._execute(state, injected_context)

        # Handle LoopControl signals
        if isinstance(result, LoopControl):
            return state

        return state

    def _execute(self, state: dict, context: BaseFlowContext):
        """Internal execution method with dependency injection."""
        print(f"--- Executing Node: {self.name} ---")

        kwargs = {}
        for name, param in self.params.items():
            if hasattr(context, name):
                provider = getattr(context, name)
                injected_obj = provider()
                kwargs[name] = injected_obj

                if param.annotation is not inspect.Parameter.empty and not isinstance(injected_obj, param.annotation):
                    print(f"  - Injected '{name}' from context. WARNING: Type mismatch. "
                          f"Expected {param.annotation.__name__}, got {type(injected_obj).__name__}.")
                else:
                    print(f"  - Injected '{name}' from context provider.")
            elif name in state:
                kwargs[name] = state.get(name)

        print(f"  - Final injected arguments: {list(kwargs.keys())}")
        result = self.func(**kwargs)

        # Handle LoopControl signals
        if isinstance(result, LoopControl):
            return result

        if isinstance(result, dict):
            state.update(result)
            print(f"  - State updated with keys: {list(result.keys())}")
        elif result is not None:
            state[self.name] = result
            print(f"  - State updated with key: '{self.name}'")

        return result

    def then(self, next_node: 'Node') -> 'SequentialComposition':
        """Chain this node with another node for sequential execution."""
        return SequentialComposition(self, next_node)

    def fan_out_to(self, nodes: List['Node'], executor: str = 'thread',
                   max_workers: int = None) -> 'ParallelFanOutComposition':
        """Fan out to multiple nodes for parallel execution."""
        return ParallelFanOutComposition(self, nodes, executor, max_workers)

    def branch_on(self, conditions: Dict[bool, 'Node']) -> 'ConditionalComposition':
        """Branch execution based on the boolean output of this node."""
        return ConditionalComposition(self, conditions)

    def repeat(self, times: int) -> 'RepeatComposition':
        """Repeat this node for a fixed number of times."""
        return RepeatComposition(self, times)

    def __repr__(self) -> str:
        return f"Node(name='{self.name}')"


# --- 5. Composition Classes ---

class Composition:
    """Base class for all composition types."""

    def __init__(self, name: str):
        self.name = name

    def run(self, state: Dict[str, Any], context: BaseFlowContext = None) -> Dict[str, Any]:
        """Run the composition with dependency injection for context."""
        injected_context = context or AppContext()
        self._execute(state, injected_context)
        return state

    def _execute(self, state: dict, context: BaseFlowContext):
        """Internal execution method to be implemented by subclasses."""
        _ = state, context  # Suppress unused parameter warnings
        raise NotImplementedError

    def then(self, next_node: Node) -> 'SequentialComposition':
        """Chain this composition with another node."""
        return SequentialComposition(self, next_node)

    def fan_out_to(self, nodes: List[Node], executor: str = 'thread',
                   max_workers: int = None) -> 'ParallelFanOutComposition':
        """Fan out from this composition to multiple nodes."""
        return ParallelFanOutComposition(self, nodes, executor, max_workers)

    def branch_on(self, conditions: Dict[bool, Node]) -> 'ConditionalComposition':
        """Branch execution based on the output of this composition."""
        return ConditionalComposition(self, conditions)

    def repeat(self, times: int) -> 'RepeatComposition':
        """Repeat this composition for a fixed number of times."""
        return RepeatComposition(self, times)


class SequentialComposition(Composition):
    """Sequential execution of two components."""

    def __init__(self, left: Union[Node, Composition], right: Union[Node, Composition]):
        super().__init__(f"({left.name} -> {right.name})")
        self.left = left
        self.right = right

    def _execute(self, state: dict, context: BaseFlowContext):
        print(f"--- Executing Sequential Composition: {self.name} ---")
        self.left._execute(state, context)
        result = self.right._execute(state, context)
        return result


class ParallelFanOutComposition(Composition):
    """Parallel fan-out execution followed by fan-in."""

    def __init__(self, source: Union[Node, Composition], targets: List[Node],
                 executor: str = 'thread', max_workers: int = None):
        target_names = [t.name for t in targets]
        super().__init__(f"({source.name} -> [{', '.join(target_names)}])")
        self.source = source
        self.targets = targets
        self.executor_str = executor
        self.max_workers = max_workers
        self.executor_map = {'thread': ThreadPoolExecutor, 'process': ProcessPoolExecutor}

    def _execute(self, state: dict, context: BaseFlowContext):
        print(f"--- Executing Parallel Fan-Out: {self.name} ---")

        # Execute source first
        self.source._execute(state, context)

        # Store pre-fan-out state for fan-in
        original_state = dict(state)
        state['__pre_fan_out_state'] = original_state

        print(f"  - Fanning out with state keys: {list(original_state.keys())} using '{self.executor_str}' executor.")

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
            result = node._execute(thread_state, node_context)

            # Generate a clean result key
            result_key = generate_result_key(node, node_index, self.targets)

            # Handle LoopControl signals
            if isinstance(result, LoopControl):
                return result_key, None

            # Return the result key and its result
            return result_key, thread_state.get(node.name, result)

        with self.executor_map[self.executor_str](max_workers=self.max_workers) as executor:
            futures = [executor.submit(run_isolated_node, node, i, original_state, context)
                       for i, node in enumerate(self.targets)]
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

    def fan_in(self, aggregator: Node) -> 'ParallelFanInComposition':
        """Complete the fan-out/fan-in pattern with an aggregator node."""
        return ParallelFanInComposition(self, aggregator)


class ParallelFanInComposition(Composition):
    """Parallel fan-in aggregation."""

    def __init__(self, fan_out: ParallelFanOutComposition, aggregator: Node):
        super().__init__(f"({fan_out.name} -> {aggregator.name})")
        self.fan_out = fan_out
        self.aggregator = aggregator

    def _execute(self, state: dict, context: BaseFlowContext):
        print(f"--- Executing Parallel Fan-In: {self.name} ---")

        # Execute the fan-out first
        self.fan_out._execute(state, context)

        # Get the results from fan-out
        original_state = state.get('__pre_fan_out_state', {})
        parallel_results = state.get('__parallel_results', {})

        print(f"  - Original state keys: {list(original_state.keys())}")
        print(f"  - Parallel results: {parallel_results}")

        # Create aggregator state with parallel results
        aggregator_state = dict(original_state)
        aggregator_state['parallel_results'] = parallel_results

        print(f"  - Fanning in to aggregator: {self.aggregator.name}")

        # Execute aggregator
        aggregator_result = self.aggregator._execute(aggregator_state, context)

        # Update main state with aggregator results
        if isinstance(aggregator_result, dict):
            state.update(aggregator_result)
        elif aggregator_result is not None:
            state[self.aggregator.name] = aggregator_result

        # Clean up internal keys
        state.pop('__pre_fan_out_state', None)
        state.pop('__parallel_results', None)

        print(f"  - Final state keys after fan-in: {list(state.keys())}")

        # Return the aggregator result
        return aggregator_result


class ConditionalComposition(Composition):
    """Conditional branching based on boolean output."""

    def __init__(self, condition_node: Union[Node, Composition], branches: Dict[bool, Node]):
        branch_names = {k: v.name for k, v in branches.items()}
        super().__init__(f"({condition_node.name} ? {branch_names})")
        self.condition_node = condition_node
        self.branches = branches

    def _execute(self, state: dict, context: BaseFlowContext):
        print(f"--- Executing Conditional Branch: {self.name} ---")

        # Execute condition node
        result = self.condition_node._execute(state, context)

        # Get boolean result - either from the direct return value or from state
        condition_result = None
        if isinstance(result, bool):
            condition_result = result
        elif isinstance(self.condition_node, Node):
            # For simple nodes, check if they returned a boolean directly or stored it in state
            condition_result = state.get(self.condition_node.name)

        if not isinstance(condition_result, bool):
            raise ValueError(
                f"Condition node '{self.condition_node.name}' must return a boolean value, got {type(condition_result)}")

        # Execute the appropriate branch
        if condition_result in self.branches:
            selected_branch = self.branches[condition_result]
            print(f"  - Condition is {condition_result}, executing branch: {selected_branch.name}")
            return selected_branch._execute(state, context)
        else:
            print(f"  - No branch defined for condition result: {condition_result}")
            return None


class RepeatComposition(Composition):
    """Repeat a node for a fixed number of times with early exit support."""

    def __init__(self, node: Union[Node, Composition], times: int):
        super().__init__(f"({node.name} * {times})")
        self.node = node
        self.times = times

    def _execute(self, state: dict, context: BaseFlowContext):
        print(f"--- Executing Repeat Composition: {self.name} ---")

        last_result = None
        for i in range(self.times):
            print(f"  - Iteration {i + 1}/{self.times}")
            result = self.node._execute(state, context)

            # Check for early exit signal
            if isinstance(result, LoopControl) and result == LoopControl.BREAK:
                print(f"  - Loop terminated early at iteration {i + 1} due to BREAK signal")
                break

            # Store the last valid result (not a control signal)
            if not isinstance(result, LoopControl):
                last_result = result

        # Return the last valid result or None if no valid results were produced
        return last_result


# --- 6. Decorator ---

def node(func: Callable) -> Node:
    """Decorator to create a Node from a function."""
    return Node(func=func, name=func.__name__)


# Export list - 只暴露用户需要的公共接口
__all__ = [
    # 核心装饰器和枚举
    'node',
    'LoopControl',

    # 上下文相关（用户可能需要自定义）
    'BaseFlowContext',
    'AppContext',

    # 核心节点类（用户可能需要直接使用）
    'Node'
]
