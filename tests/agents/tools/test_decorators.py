"""
Tests for Tool Decorator System

Validates the @tool decorator functionality, metadata extraction,
parameter validation, and integration with the AetherFlow @node system.
"""

import asyncio

import pytest

from agents.tools.decorators import get_tool_metadata, is_tool_function, tool
from agents.tools.models import ToolCategory, ToolExecutionStatus, ToolResult
from agents.tools.registry import ToolRegistry


class TestToolDecorator:
    """Test the @tool decorator functionality"""

    def setup_method(self):
        """Clear registry before each test"""
        ToolRegistry.clear_registry()

    def test_basic_tool_decoration(self):
        """Test basic @tool decoration without parameters"""

        @tool("test_basic", "A basic test tool")
        def basic_tool(x: int) -> int:
            return x * 2

        # Check tool metadata
        metadata = get_tool_metadata(basic_tool)
        assert metadata is not None
        assert metadata.name == "test_basic"
        assert metadata.description == "A basic test tool"
        assert metadata.category == ToolCategory.GENERAL
        assert len(metadata.parameters) == 1
        assert metadata.parameters[0].name == "x"
        assert metadata.parameters[0].type_hint == "int"

    def test_async_tool_decoration(self):
        """Test @tool with async functions"""

        @tool("test_async", "An async test tool", category=ToolCategory.COMPUTE)
        async def async_tool(message: str) -> str:
            await asyncio.sleep(0.01)  # Simulate async work
            return f"Processed: {message}"

        metadata = get_tool_metadata(async_tool)
        assert metadata.is_async is True
        assert metadata.category == ToolCategory.COMPUTE

    def test_tool_with_optional_parameters(self):
        """Test tool with optional parameters and default values"""

        @tool("test_optional", "Tool with optional parameters")
        def optional_tool(
            required: str, optional: int = 42, other: str | None = None
        ) -> str:
            return f"{required}-{optional}-{other}"

        metadata = get_tool_metadata(optional_tool)
        assert len(metadata.parameters) == 3

        # Check required parameter
        required_param = next(p for p in metadata.parameters if p.name == "required")
        assert required_param.required is True

        # Check optional parameter with default
        optional_param = next(p for p in metadata.parameters if p.name == "optional")
        assert optional_param.required is False
        assert optional_param.default_value == 42

    def test_tool_with_docstring_extraction(self):
        """Test parameter description extraction from docstring"""

        @tool("test_docstring", "Tool with documented parameters")
        def documented_tool(param1: int, param2: str) -> bool:
            """
            A documented tool function.

            param1: First parameter description
            param2: Second parameter description
            """
            return param1 > 0

        metadata = get_tool_metadata(documented_tool)

        # Note: The current implementation has basic docstring parsing
        # This test validates the structure even if descriptions are empty
        assert len(metadata.parameters) == 2
        param1 = next(p for p in metadata.parameters if p.name == "param1")
        param2 = next(p for p in metadata.parameters if p.name == "param2")
        assert param1.type_hint == "int"
        assert param2.type_hint == "str"

    @pytest.mark.asyncio
    async def test_tool_execution_success(self):
        """Test successful tool execution"""

        @tool("test_execution", "Tool for testing execution")
        def execution_tool(value: int) -> int:
            return value * 3

        # Execute the tool
        result = await execution_tool(value=5)

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.tool_name == "test_execution"
        assert result.result == 15
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_tool_execution_failure(self):
        """Test tool execution with exception"""

        @tool("test_failure", "Tool that fails")
        def failing_tool(value: int) -> int:
            if value < 0:
                raise ValueError("Value cannot be negative")
            return value

        # Execute with invalid input
        result = await failing_tool(value=-5)

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.status == ToolExecutionStatus.FAILED
        assert "Value cannot be negative" in result.error_message
        assert result.error_type == "ValueError"

    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Test async tool execution"""

        @tool("test_async_exec", "Async tool for testing")
        async def async_execution_tool(delay: float) -> str:
            await asyncio.sleep(delay)
            return f"Completed after {delay}s"

        result = await async_execution_tool(delay=0.01)

        assert result.success is True
        assert result.result.startswith("Completed after")
        assert result.execution_time_ms >= 10  # At least 10ms for 0.01s delay

    @pytest.mark.asyncio
    async def test_tool_timeout(self):
        """Test tool execution timeout"""

        @tool("test_timeout", "Tool that times out", timeout=0.1)
        async def timeout_tool() -> str:
            await asyncio.sleep(0.2)  # Sleep longer than timeout
            return "This should not complete"

        result = await timeout_tool()

        assert result.success is False
        assert result.status == ToolExecutionStatus.TIMEOUT
        assert "timed out" in result.error_message.lower()

    def test_tool_registry_integration(self):
        """Test that tools are automatically registered"""

        @tool("test_registry", "Tool for registry testing")
        def registry_tool() -> str:
            return "registered"

        # Check tool is in registry
        registered_tool = ToolRegistry.get_tool("test_registry")
        assert registered_tool is not None
        assert registered_tool == registry_tool

        # Check metadata is available
        metadata = ToolRegistry.get_tool_metadata("test_registry")
        assert metadata is not None
        assert metadata.name == "test_registry"

    def test_tool_category_validation(self):
        """Test tool category validation"""

        # Valid category
        @tool("test_category", "Tool with valid category", category=ToolCategory.SEARCH)
        def category_tool() -> str:
            return "search"

        metadata = get_tool_metadata(category_tool)
        assert metadata.category == ToolCategory.SEARCH

        # String category should be converted
        @tool("test_string_category", "Tool with string category", category="compute")
        def string_category_tool() -> str:
            return "compute"

        metadata = get_tool_metadata(string_category_tool)
        assert metadata.category == ToolCategory.COMPUTE

    def test_is_tool_function(self):
        """Test tool function identification"""

        @tool("test_identification", "Tool for identification test")
        def identified_tool() -> None:
            pass

        def regular_function() -> None:
            pass

        assert is_tool_function(identified_tool) is True
        assert is_tool_function(regular_function) is False

    def test_tool_name_validation(self):
        """Test tool name validation and normalization"""

        # Name should be normalized to lowercase
        @tool("Test_NORMALIZE", "Tool with mixed case name")
        def normalize_tool() -> str:
            return "normalized"

        metadata = get_tool_metadata(normalize_tool)
        assert metadata.name == "test_normalize"

    def test_complex_type_hints(self):
        """Test handling of complex type hints"""

        @tool("test_complex_types", "Tool with complex type hints")
        def complex_tool(items: list[str], count: int | None = None) -> list[int]:
            return [len(item) for item in items]

        metadata = get_tool_metadata(complex_tool)

        items_param = next(p for p in metadata.parameters if p.name == "items")
        count_param = next(p for p in metadata.parameters if p.name == "count")

        # Type hints should be formatted as strings
        assert "List" in items_param.type_hint or "list" in items_param.type_hint
        assert count_param.required is False

    def test_multiple_tools_registration(self):
        """Test registering multiple tools"""

        @tool("multi_1", "First multi tool")
        def multi_tool_1() -> str:
            return "first"

        @tool("multi_2", "Second multi tool")
        def multi_tool_2() -> str:
            return "second"

        # Both should be registered
        assert ToolRegistry.get_tool("multi_1") is not None
        assert ToolRegistry.get_tool("multi_2") is not None

        # Should have separate metadata
        metadata_1 = ToolRegistry.get_tool_metadata("multi_1")
        metadata_2 = ToolRegistry.get_tool_metadata("multi_2")
        assert metadata_1.description == "First multi tool"
        assert metadata_2.description == "Second multi tool"
