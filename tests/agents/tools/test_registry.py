"""
Tests for Tool Registry System

Validates tool registration, discovery, validation, and statistics tracking
functionality of the ToolRegistry.
"""

import pytest

from agents.tools.decorators import tool
from agents.tools.models import ToolCall, ToolCategory
from agents.tools.registry import ToolRegistry


class TestToolRegistry:
    """Test the ToolRegistry functionality"""

    def setup_method(self):
        """Clear registry before each test"""
        ToolRegistry.clear_registry()

    def test_singleton_instance(self):
        """Test that registry is a singleton"""
        registry1 = ToolRegistry.get_instance()
        registry2 = ToolRegistry.get_instance()
        assert registry1 is registry2

    def test_tool_registration(self):
        """Test basic tool registration"""

        @tool("test_registration", "Tool for registration test")
        def test_tool() -> str:
            return "registered"

        # Tool should be automatically registered
        registered_tool = ToolRegistry.get_tool("test_registration")
        assert registered_tool is test_tool

        # Metadata should be available
        metadata = ToolRegistry.get_tool_metadata("test_registration")
        assert metadata is not None
        assert metadata.name == "test_registration"

    def test_get_available_tools(self):
        """Test getting list of available tools"""

        @tool("available_1", "First available tool", category=ToolCategory.COMPUTE)
        def tool1() -> int:
            return 1

        @tool("available_2", "Second available tool", category=ToolCategory.SEARCH)
        def tool2() -> int:
            return 2

        # Get all tools
        all_tools = ToolRegistry.get_available_tools()
        assert len(all_tools) == 2
        tool_names = [t.name for t in all_tools]
        assert "available_1" in tool_names
        assert "available_2" in tool_names

        # Filter by category
        compute_tools = ToolRegistry.get_available_tools(category=ToolCategory.COMPUTE)
        assert len(compute_tools) == 1
        assert compute_tools[0].name == "available_1"

    def test_tools_by_category(self):
        """Test getting tools organized by category"""

        @tool("compute_tool", "Compute tool", category=ToolCategory.COMPUTE)
        def compute() -> int:
            return 42

        @tool("search_tool", "Search tool", category=ToolCategory.SEARCH)
        def search() -> str:
            return "found"

        @tool("general_tool", "General tool")  # Default category
        def general() -> str:
            return "general"

        categorized = ToolRegistry.get_tools_by_category()

        assert ToolCategory.COMPUTE in categorized
        assert ToolCategory.SEARCH in categorized
        assert ToolCategory.GENERAL in categorized

        assert len(categorized[ToolCategory.COMPUTE]) == 1
        assert categorized[ToolCategory.COMPUTE][0].name == "compute_tool"

    def test_tool_call_validation(self):
        """Test tool call validation"""

        @tool("validation_tool", "Tool for validation testing")
        def validation_tool(required: str, optional: int = 10) -> str:
            return f"{required}-{optional}"

        # Valid call
        valid_call = ToolCall(
            tool_name="validation_tool", parameters={"required": "test", "optional": 20}
        )
        is_valid, error = ToolRegistry.validate_tool_call(valid_call)
        assert is_valid is True
        assert error is None

        # Missing required parameter
        invalid_call = ToolCall(
            tool_name="validation_tool", parameters={"optional": 20}
        )
        is_valid, error = ToolRegistry.validate_tool_call(invalid_call)
        assert is_valid is False
        assert "Missing required parameters" in error

        # Non-existent tool
        nonexistent_call = ToolCall(tool_name="nonexistent_tool", parameters={})
        is_valid, error = ToolRegistry.validate_tool_call(nonexistent_call)
        assert is_valid is False
        assert "not found in registry" in error

    def test_execution_statistics(self):
        """Test execution statistics recording and retrieval"""

        @tool("stats_tool", "Tool for statistics testing")
        def stats_tool() -> str:
            return "stats"

        # Record some executions
        ToolRegistry.record_execution_stats("stats_tool", 100.0, True)
        ToolRegistry.record_execution_stats("stats_tool", 150.0, True)
        ToolRegistry.record_execution_stats("stats_tool", 200.0, False, "TestError")

        # Get statistics
        stats = ToolRegistry.get_execution_stats("stats_tool")
        assert "stats_tool" in stats

        tool_stats = stats["stats_tool"]
        assert tool_stats["call_count"] == 3
        assert tool_stats["success_count"] == 2
        assert tool_stats["failure_count"] == 1
        assert tool_stats["success_rate"] == pytest.approx(66.67, rel=0.1)
        assert tool_stats["average_execution_time_ms"] == pytest.approx(150.0)

    def test_registry_info(self):
        """Test getting comprehensive registry information"""

        @tool("info_tool_1", "First info tool", category=ToolCategory.COMPUTE)
        def tool1() -> int:
            return 1

        @tool("info_tool_2", "Second info tool", category=ToolCategory.WEB)
        def tool2() -> str:
            return "web"

        info = ToolRegistry.get_registry_info()

        assert info["total_tools"] == 2
        assert "categories" in info
        assert info["categories"]["compute"] == 1
        assert info["categories"]["web"] == 1

    def test_tool_unregistration(self):
        """Test tool unregistration"""

        @tool("unregister_tool", "Tool to be unregistered")
        def unregister_tool() -> str:
            return "will be removed"

        # Tool should be registered
        assert ToolRegistry.get_tool("unregister_tool") is not None

        # Unregister tool
        success = ToolRegistry.unregister_tool("unregister_tool")
        assert success is True

        # Tool should no longer be available
        assert ToolRegistry.get_tool("unregister_tool") is None

        # Unregistering non-existent tool should return False
        success = ToolRegistry.unregister_tool("non_existent")
        assert success is False

    def test_duplicate_registration_handling(self):
        """Test handling of duplicate tool registrations"""

        # First registration
        @tool("duplicate_tool", "First registration")
        def first_tool() -> str:
            return "first"

        # Get the registered function
        registered_first = ToolRegistry.get_tool("duplicate_tool")
        assert registered_first is first_tool

        # Re-registering from same module should work (with warning)
        @tool("duplicate_tool", "Re-registration")
        def second_tool() -> str:
            return "second"

        # Should now reference the second tool
        registered_second = ToolRegistry.get_tool("duplicate_tool")
        assert registered_second is second_tool

    def test_filter_by_concurrency_support(self):
        """Test filtering tools by concurrency support"""

        @tool("concurrent_tool", "Supports concurrency", supports_concurrent=True)
        def concurrent_tool() -> str:
            return "concurrent"

        @tool("sequential_tool", "Sequential only", supports_concurrent=False)
        def sequential_tool() -> str:
            return "sequential"

        # Filter for concurrent tools
        concurrent_tools = ToolRegistry.get_available_tools(supports_concurrent=True)
        assert len(concurrent_tools) == 1
        assert concurrent_tools[0].name == "concurrent_tool"

        # Filter for sequential tools
        sequential_tools = ToolRegistry.get_available_tools(supports_concurrent=False)
        assert len(sequential_tools) == 1
        assert sequential_tools[0].name == "sequential_tool"

    def test_clear_registry(self):
        """Test clearing the registry"""

        @tool("clear_test", "Tool for clear test")
        def clear_tool() -> str:
            return "clear"

        # Tool should be registered
        assert ToolRegistry.get_tool("clear_test") is not None

        # Clear registry
        ToolRegistry.clear_registry()

        # Tool should no longer be available
        assert ToolRegistry.get_tool("clear_test") is None
        assert len(ToolRegistry.get_available_tools()) == 0

    def test_thread_safety(self):
        """Test basic thread safety of registry operations"""
        import threading
        import time

        results = []
        errors = []

        def register_tools(thread_id: int):
            try:
                for i in range(5):

                    @tool(
                        f"thread_{thread_id}_tool_{i}",
                        f"Tool {i} from thread {thread_id}",
                    )
                    def thread_tool(i=i) -> str:
                        return f"thread_{thread_id}_result_{i}"

                    results.append(f"thread_{thread_id}_tool_{i}")
                    time.sleep(0.001)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(str(e))

        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=register_tools, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 15  # 3 threads × 5 tools each

        # Verify all tools are registered
        all_tools = ToolRegistry.get_available_tools()
        registered_names = [t.name for t in all_tools]

        for expected_name in results:
            assert expected_name in registered_names
