"""
Tool Registry System

Provides centralized registration and discovery of tools for ReAct agents.
The registry maintains a global catalog of available tools, supports dynamic
discovery, and provides querying capabilities for tool selection.

Key Features:
- Thread-safe global tool registration
- Dynamic tool discovery from modules
- Category-based tool filtering
- Tool validation and conflict resolution
- Execution statistics and performance monitoring
- Integration with dependency injection system
"""

import importlib
import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from .models import ToolCall, ToolCategory, ToolMetadata

logger = logging.getLogger("aetherflow.agents.tools.registry")


class ToolRegistrationError(Exception):
    """Raised when tool registration fails"""

    pass


class ToolNotFoundError(Exception):
    """Raised when a requested tool is not found"""

    pass


class ToolExecutionStats:
    """Statistics tracking for tool execution performance"""

    def __init__(self):
        self.call_count = 0
        self.total_execution_time_ms = 0.0
        self.success_count = 0
        self.failure_count = 0
        self.average_execution_time_ms = 0.0
        self.last_execution_time = None
        self.error_types = defaultdict(int)

    def record_execution(
        self, execution_time_ms: float, success: bool, error_type: str | None = None
    ) -> None:
        """Record a tool execution for statistics"""
        self.call_count += 1
        self.total_execution_time_ms += execution_time_ms

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            if error_type:
                self.error_types[error_type] += 1

        self.average_execution_time_ms = self.total_execution_time_ms / self.call_count
        self.last_execution_time = execution_time_ms

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.call_count == 0:
            return 0.0
        return (self.success_count / self.call_count) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for serialization"""
        return {
            "call_count": self.call_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "average_execution_time_ms": self.average_execution_time_ms,
            "total_execution_time_ms": self.total_execution_time_ms,
            "error_types": dict(self.error_types),
        }


class ToolRegistry:
    """
    Global registry for ReAct agent tools.

    Provides thread-safe registration, discovery, and management of tools.
    Supports dynamic discovery from modules and packages, category-based
    organization, and performance statistics tracking.
    """

    _instance: Optional["ToolRegistry"] = None
    _lock = threading.RLock()

    def __init__(self):
        """Initialize empty registry"""
        self._tools: dict[str, tuple[ToolMetadata, Callable]] = {}
        self._categories: dict[ToolCategory, set[str]] = defaultdict(set)
        self._execution_stats: dict[str, ToolExecutionStats] = {}
        self._module_tools: dict[str, set[str]] = defaultdict(set)
        self._registry_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """Get or create the singleton registry instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info("Created ToolRegistry singleton instance")
        return cls._instance

    @classmethod
    def register(cls, metadata: ToolMetadata, tool_func: Callable) -> None:
        """
        Register a tool in the global registry.

        Args:
            metadata: Tool metadata and configuration
            tool_func: Callable tool function (should be @node decorated)

        Raises:
            ToolRegistrationError: If registration fails or tool already exists
        """
        registry = cls.get_instance()
        with registry._registry_lock:
            tool_name = metadata.name

            # Check for existing tool
            if tool_name in registry._tools:
                existing_metadata, _ = registry._tools[tool_name]
                if existing_metadata.module_path != metadata.module_path:
                    # Different modules trying to register same tool name
                    raise ToolRegistrationError(
                        f"Tool '{tool_name}' already registered from module "
                        f"'{existing_metadata.module_path}', cannot register from "
                        f"'{metadata.module_path}'"
                    )
                else:
                    # Same module re-registering (e.g., during reload)
                    logger.warning(
                        f"Re-registering tool '{tool_name}' from same module"
                    )

            # Register the tool
            registry._tools[tool_name] = (metadata, tool_func)
            registry._categories[metadata.category].add(tool_name)

            # Track module association
            if metadata.module_path:
                registry._module_tools[metadata.module_path].add(tool_name)

            # Initialize execution stats
            if tool_name not in registry._execution_stats:
                registry._execution_stats[tool_name] = ToolExecutionStats()

            logger.debug(
                f"Registered tool '{tool_name}' in category '{metadata.category.value}'"
            )

    @classmethod
    def get_tool(cls, tool_name: str) -> Callable | None:
        """
        Get a registered tool function by name.

        Args:
            tool_name: Name of tool to retrieve

        Returns:
            Tool function if found, None otherwise
        """
        registry = cls.get_instance()
        with registry._registry_lock:
            if tool_name in registry._tools:
                _, tool_func = registry._tools[tool_name]
                return tool_func
            return None

    @classmethod
    def get_tool_metadata(cls, tool_name: str) -> ToolMetadata | None:
        """
        Get tool metadata by name.

        Args:
            tool_name: Name of tool

        Returns:
            Tool metadata if found, None otherwise
        """
        registry = cls.get_instance()
        with registry._registry_lock:
            if tool_name in registry._tools:
                metadata, _ = registry._tools[tool_name]
                return metadata
            return None

    @classmethod
    def get_available_tools(
        cls,
        category: ToolCategory | None = None,
        module_path: str | None = None,
        supports_concurrent: bool | None = None,
    ) -> list[ToolMetadata]:
        """
        Get list of available tools with optional filtering.

        Args:
            category: Filter by tool category
            module_path: Filter by module path
            supports_concurrent: Filter by concurrency support

        Returns:
            List of matching tool metadata
        """
        registry = cls.get_instance()
        with registry._registry_lock:
            tools = []

            for tool_name, (metadata, _) in registry._tools.items():
                # Apply filters
                if category and metadata.category != category:
                    continue
                if module_path and metadata.module_path != module_path:
                    continue
                if (
                    supports_concurrent is not None
                    and metadata.supports_concurrent != supports_concurrent
                ):
                    continue

                tools.append(metadata)

            # Sort by category and name for consistent ordering
            tools.sort(key=lambda m: (m.category.value, m.name))
            return tools

    @classmethod
    def get_tools_by_category(cls) -> dict[ToolCategory, list[ToolMetadata]]:
        """
        Get tools organized by category.

        Returns:
            Dictionary mapping categories to tool lists
        """
        registry = cls.get_instance()
        with registry._registry_lock:
            categorized = defaultdict(list)

            for metadata, _ in registry._tools.values():
                categorized[metadata.category].append(metadata)

            # Sort tools within each category
            for category in categorized:
                categorized[category].sort(key=lambda m: m.name)

            return dict(categorized)

    @classmethod
    def validate_tool_call(cls, tool_call: ToolCall) -> tuple[bool, str | None]:
        """
        Validate a tool call against registered tool metadata.

        Args:
            tool_call: Tool call to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        registry = cls.get_instance()

        # Check if tool exists
        metadata = cls.get_tool_metadata(tool_call.tool_name)
        if not metadata:
            return False, f"Tool '{tool_call.tool_name}' not found in registry"

        # Validate required parameters
        required_params = {p.name for p in metadata.parameters if p.required}
        provided_params = set(tool_call.parameters.keys())

        missing_params = required_params - provided_params
        if missing_params:
            return False, f"Missing required parameters: {', '.join(missing_params)}"

        # Check for extra parameters (warn but don't fail)
        expected_params = {p.name for p in metadata.parameters}
        extra_params = provided_params - expected_params
        if extra_params:
            logger.warning(
                f"Tool '{tool_call.tool_name}' received unexpected parameters: {', '.join(extra_params)}"
            )

        return True, None

    @classmethod
    def record_execution_stats(
        cls,
        tool_name: str,
        execution_time_ms: float,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        """
        Record execution statistics for a tool.

        Args:
            tool_name: Name of executed tool
            execution_time_ms: Execution time in milliseconds
            success: Whether execution was successful
            error_type: Type of error if failed
        """
        registry = cls.get_instance()
        with registry._registry_lock:
            if tool_name not in registry._execution_stats:
                registry._execution_stats[tool_name] = ToolExecutionStats()

            registry._execution_stats[tool_name].record_execution(
                execution_time_ms, success, error_type
            )

    @classmethod
    def get_execution_stats(
        cls, tool_name: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """
        Get execution statistics for tools.

        Args:
            tool_name: Specific tool name, or None for all tools

        Returns:
            Dictionary of tool names to statistics
        """
        registry = cls.get_instance()
        with registry._registry_lock:
            if tool_name:
                if tool_name in registry._execution_stats:
                    return {tool_name: registry._execution_stats[tool_name].to_dict()}
                else:
                    return {}
            else:
                return {
                    name: stats.to_dict()
                    for name, stats in registry._execution_stats.items()
                }

    @classmethod
    def discover_tools_in_module(cls, module_path: str, recursive: bool = True) -> int:
        """
        Discover and register tools from a Python module.

        Args:
            module_path: Python module path (e.g., 'mypackage.tools')
            recursive: Whether to recursively scan submodules

        Returns:
            Number of tools discovered and registered

        Raises:
            ImportError: If module cannot be imported
        """
        registry = cls.get_instance()
        tools_found = 0

        try:
            # Import the module
            module = importlib.import_module(module_path)

            # Scan for tool functions
            from .decorators import get_tool_metadata, is_tool_function

            for name in dir(module):
                obj = getattr(module, name)
                if callable(obj) and is_tool_function(obj):
                    metadata = get_tool_metadata(obj)
                    if metadata:
                        try:
                            cls.register(metadata, obj)
                            tools_found += 1
                        except ToolRegistrationError as e:
                            logger.warning(
                                f"Failed to register tool '{metadata.name}': {e}"
                            )

            # Recursive discovery if requested
            if recursive:
                # This is a simplified version - a full implementation would
                # use pkgutil.walk_packages to discover submodules
                logger.debug(
                    f"Recursive discovery not fully implemented for {module_path}"
                )

            logger.info(f"Discovered {tools_found} tools in module '{module_path}'")
            return tools_found

        except ImportError as e:
            logger.error(f"Failed to import module '{module_path}': {e}")
            raise

    @classmethod
    def discover_tools_in_package(cls, package_path: str) -> int:
        """
        Discover tools in a Python package directory.

        Args:
            package_path: Path to package directory

        Returns:
            Number of tools discovered
        """
        tools_found = 0
        package_dir = Path(package_path)

        if not package_dir.is_dir():
            logger.warning(f"Package path does not exist: {package_path}")
            return 0

        # Find all Python files in package
        python_files = list(package_dir.rglob("*.py"))

        for python_file in python_files:
            if python_file.name.startswith("__"):
                continue  # Skip __init__.py, __main__.py, etc.

            # Convert file path to module path
            relative_path = python_file.relative_to(package_dir.parent)
            module_path = str(relative_path.with_suffix("")).replace("/", ".")

            try:
                found = cls.discover_tools_in_module(module_path, recursive=False)
                tools_found += found
            except ImportError as e:
                logger.debug(f"Could not import {module_path}: {e}")
                continue

        logger.info(f"Discovered {tools_found} tools in package '{package_path}'")
        return tools_found

    @classmethod
    def unregister_tool(cls, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.

        Args:
            tool_name: Name of tool to unregister

        Returns:
            True if tool was unregistered, False if not found
        """
        registry = cls.get_instance()
        with registry._registry_lock:
            if tool_name not in registry._tools:
                return False

            # Get metadata for cleanup
            metadata, _ = registry._tools[tool_name]

            # Remove from main registry
            del registry._tools[tool_name]

            # Remove from category index
            registry._categories[metadata.category].discard(tool_name)

            # Remove from module tracking
            if metadata.module_path:
                registry._module_tools[metadata.module_path].discard(tool_name)

            # Keep execution stats for historical purposes

            logger.info(f"Unregistered tool '{tool_name}'")
            return True

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered tools (primarily for testing)"""
        registry = cls.get_instance()
        with registry._registry_lock:
            registry._tools.clear()
            registry._categories.clear()
            registry._module_tools.clear()
            # Keep execution stats for historical purposes
            logger.info("Cleared tool registry")

    @classmethod
    def get_registry_info(cls) -> dict[str, Any]:
        """
        Get comprehensive information about the registry state.

        Returns:
            Dictionary with registry statistics and information
        """
        registry = cls.get_instance()
        with registry._registry_lock:
            category_counts = {
                category.value: len(tools)
                for category, tools in registry._categories.items()
            }

            return {
                "total_tools": len(registry._tools),
                "categories": category_counts,
                "modules": list(registry._module_tools.keys()),
                "tools_with_stats": len(registry._execution_stats),
                "registry_id": id(registry),
            }
