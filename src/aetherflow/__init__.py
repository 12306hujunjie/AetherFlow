import functools
import inspect
import threading
import json
import hashlib
from typing import Callable, Any, List, Dict, Union, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from enum import Enum

from dependency_injector import containers, providers

# --- 1. Loop Control Enum ---

class LoopControl(Enum):
    """Special signals for controlling loop execution."""
    BREAK = "BREAK"

# --- 2. Containers for Dependency Injection ---

class BaseFlowContainer(containers.DeclarativeContainer):
    """Base container for flow-related dependencies. Can be extended."""
    pass

class PersistenceService:
    """A mock service that mimics a Redis-like key-value store."""
    def __init__(self, storage: dict, lock: threading.Lock):
        self._storage = storage
        self._lock = lock

    def hset(self, name: str, key: str, value: str):
        with self._lock:
            print(f"  - (Redis) HSET on '{name}' with key '{key}'")
            self._storage[f"{name}:{key}"] = value

    def hget(self, name: str, key: str) -> Union[str, None]:
        with self._lock:
            return self._storage.get(f"{name}:{key}")

class PersistenceContainer(containers.DeclarativeContainer):
    """A container for persistence-related services."""
    storage = providers.Singleton(dict)
    lock = providers.Singleton(threading.Lock)
    service = providers.Singleton(PersistenceService, storage=storage, lock=lock)

class AppContainer(BaseFlowContainer):
    """Main application container wiring all services together."""
    persistence = providers.Container(PersistenceContainer).service

# --- 3. Graph State ---

class GraphState(dict):
    """Enhanced dictionary that can be frozen for parallel execution."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen = False
    
    def freeze(self):
        """Make this state read-only."""
        self._frozen = True
        return self
    
    def unfreeze(self):
        """Make this state writable again."""
        self._frozen = False
        return self
    
    def __setitem__(self, key, value):
        if self._frozen:
            raise RuntimeError(f"Cannot modify frozen GraphState. Attempted to set '{key}' = {value}")
        super().__setitem__(key, value)
    
    def update(self, *args, **kwargs):
        if self._frozen:
            raise RuntimeError("Cannot modify frozen GraphState with update()")
        super().update(*args, **kwargs)
    
    def copy(self):
        """Create an unfrozen copy of this state."""
        new_state = GraphState(super().copy())
        return new_state

# --- 4. Node Implementation ---

class Node:
    """A node in the execution graph that supports fluent interface methods."""
    
    def __init__(self, func: Callable = None, name: str = None):
        self.func = func
        self.name = name or (func.__name__ if func else "unnamed")
        self.params = inspect.signature(func).parameters if func else {}
    
    def run(self, initial_state: Dict[str, Any], container: BaseFlowContainer = None) -> Dict[str, Any]:
        """Execute this node with the given initial state."""
        if container is None:
            container = AppContainer()
        
        state = GraphState(initial_state)
        result = self._execute(state, container)
        
        # Handle LoopControl signals
        if isinstance(result, LoopControl):
            return dict(state)  # Return current state, ignore the signal
        
        # Clean up internal keys
        state.pop('__pre_fan_out_state', None)
        state.pop('__parallel_results', None)
        return dict(state)
    
    def _execute(self, state: GraphState, container: BaseFlowContainer):
        """Internal execution method."""
        print(f"--- Executing Node: {self.name} ---")
        
        kwargs = {}
        for name, param in self.params.items():
            if hasattr(container, name):
                provider = getattr(container, name)
                injected_obj = provider()
                kwargs[name] = injected_obj
                
                if param.annotation is not inspect.Parameter.empty and not isinstance(injected_obj, param.annotation):
                    print(f"  - Injected '{name}' from container. WARNING: Type mismatch. "
                          f"Expected {param.annotation.__name__}, got {type(injected_obj).__name__}.")
                else:
                    print(f"  - Injected '{name}' from container provider.")
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
    
    def fan_out_to(self, nodes: List['Node'], executor: str = 'thread', max_workers: int = None) -> 'ParallelFanOutComposition':
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
    
    def run(self, initial_state: Dict[str, Any], container: BaseFlowContainer = None) -> Dict[str, Any]:
        """Execute this composition with the given initial state."""
        if container is None:
            container = AppContainer()
        
        state = GraphState(initial_state)
        self._execute(state, container)
        
        # Clean up internal keys
        state.pop('__pre_fan_out_state', None)
        state.pop('__parallel_results', None)
        return dict(state)
    
    def _execute(self, state: GraphState, container: BaseFlowContainer):
        """Internal execution method to be implemented by subclasses."""
        raise NotImplementedError
    
    def then(self, next_node: Node) -> 'SequentialComposition':
        """Chain this composition with another node."""
        return SequentialComposition(self, next_node)
    
    def fan_out_to(self, nodes: List[Node], executor: str = 'thread', max_workers: int = None) -> 'ParallelFanOutComposition':
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
    
    def _execute(self, state: GraphState, container: BaseFlowContainer):
        print(f"--- Executing Sequential Composition: {self.name} ---")
        self.left._execute(state, container)
        result = self.right._execute(state, container)
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
    
    def _execute(self, state: GraphState, container: BaseFlowContainer):
        print(f"--- Executing Parallel Fan-Out: {self.name} ---")
        
        # Execute source first
        self.source._execute(state, container)
        
        # Fan out with frozen state
        frozen_state = state.copy().freeze()
        print(f"  - Fanning out with frozen state keys: {list(frozen_state.keys())} using '{self.executor_str}' executor.")
        
        def run_isolated_node(node: Node, initial_state: GraphState, node_container: BaseFlowContainer):
            thread_local_state = initial_state.copy()
            result = node._execute(thread_local_state, node_container)
            
            # Handle LoopControl signals
            if isinstance(result, LoopControl):
                return node.name, {}
            
            # Extract new keys as isolated result
            new_keys = thread_local_state.keys() - initial_state.keys()
            isolated_result = {k: thread_local_state[k] for k in new_keys}
            if not isolated_result and node.name in thread_local_state:
                isolated_result = {node.name: thread_local_state[node.name]}
            return node.name, isolated_result
        
        parallel_results = {}
        with self.executor_map[self.executor_str](max_workers=self.max_workers) as executor:
            future_to_node = {
                executor.submit(run_isolated_node, node, frozen_state, container): node
                for node in self.targets
            }
            for future in as_completed(future_to_node):
                node_name, isolated_result = future.result()
                parallel_results[node_name] = isolated_result
        
        # Store results for potential fan-in
        state["__pre_fan_out_state"] = dict(frozen_state)
        state["__parallel_results"] = parallel_results
    
    def fan_in(self, aggregator: Node) -> 'ParallelFanInComposition':
        """Complete the fan-out/fan-in pattern with an aggregator node."""
        return ParallelFanInComposition(self, aggregator)

class ParallelFanInComposition(Composition):
    """Parallel fan-in aggregation."""
    
    def __init__(self, fan_out: ParallelFanOutComposition, aggregator: Node):
        super().__init__(f"({fan_out.name} -> {aggregator.name})")
        self.fan_out = fan_out
        self.aggregator = aggregator
    
    def _execute(self, state: GraphState, container: BaseFlowContainer):
        print(f"--- Executing Parallel Fan-In: {self.name} ---")
        
        # Execute fan-out first
        self.fan_out._execute(state, container)
        
        # Check for required state
        if "__pre_fan_out_state" not in state:
            raise ValueError("Fan-in requires a preceding fan-out operation.")
        
        original_state = state.get("__pre_fan_out_state")
        p_results = state.get("__parallel_results")
        
        # Create aggregator state
        aggregator_state = GraphState({**original_state, "parallel_results": p_results})
        
        print(f"  - Fanning in to aggregator: {self.aggregator}")
        self.aggregator._execute(aggregator_state, container)
        
        # Merge results from the aggregator back into the main state
        new_keys = aggregator_state.keys() - original_state.keys() - {"parallel_results"}
        for key in new_keys:
            state[key] = aggregator_state[key]

class ConditionalComposition(Composition):
    """Conditional branching based on boolean output."""
    
    def __init__(self, condition_node: Union[Node, Composition], branches: Dict[bool, Node]):
        branch_names = {k: v.name for k, v in branches.items()}
        super().__init__(f"({condition_node.name} ? {branch_names})")
        self.condition_node = condition_node
        self.branches = branches
    
    def _execute(self, state: GraphState, container: BaseFlowContainer):
        print(f"--- Executing Conditional Branch: {self.name} ---")
        
        # Execute condition node
        result = self.condition_node._execute(state, container)
        
        # Get boolean result - either from the direct return value or from state
        condition_result = None
        if isinstance(result, bool):
            condition_result = result
        elif isinstance(self.condition_node, Node):
            # For simple nodes, check if they returned a boolean directly or stored it in state
            condition_result = state.get(self.condition_node.name)
        
        if not isinstance(condition_result, bool):
            raise ValueError(f"Condition node '{self.condition_node.name}' must return a boolean value, got {type(condition_result)}")
        
        # Execute the appropriate branch
        if condition_result in self.branches:
            selected_branch = self.branches[condition_result]
            print(f"  - Condition is {condition_result}, executing branch: {selected_branch.name}")
            selected_branch._execute(state, container)
        else:
            print(f"  - No branch defined for condition result: {condition_result}")

class RepeatComposition(Composition):
    """Repeat a node for a fixed number of times with early exit support."""
    
    def __init__(self, node: Union[Node, Composition], times: int):
        super().__init__(f"({node.name} * {times})")
        self.node = node
        self.times = times
    
    def _execute(self, state: GraphState, container: BaseFlowContainer):
        print(f"--- Executing Repeat Composition: {self.name} ---")
        
        for i in range(self.times):
            print(f"  - Iteration {i + 1}/{self.times}")
            result = self.node._execute(state, container)
            
            # Check for early exit signal
            if isinstance(result, LoopControl) and result == LoopControl.BREAK:
                print(f"  - Loop terminated early at iteration {i + 1} due to BREAK signal")
                break

# --- 6. Decorator ---

def node(func: Callable) -> Node:
    """Decorator to create a Node from a function."""
    return Node(func=func, name=func.__name__)
