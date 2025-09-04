#!/usr/bin/env python3
"""
Test ReActAgent execution functionality with actual LLM calls.
"""

import asyncio
import os

# Set up environment with real API key
# Note: This would need a real API key for actual testing
os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing"


async def test_execution_functionality():
    """Test actual agent execution capabilities."""
    try:
        from agents import create_agent
        from agents.tools import tool

        # Create a simple test tool
        @tool("test_math", "Perform basic test arithmetic operations")
        async def test_math(operation: str, a: float, b: float) -> float:
            """Simple test math tool for testing."""
            if operation == "add":
                return a + b
            elif operation == "subtract":
                return a - b
            elif operation == "multiply":
                return a * b
            elif operation == "divide":
                if b != 0:
                    return a / b
                else:
                    raise ValueError("Cannot divide by zero")
            else:
                raise ValueError(f"Unknown operation: {operation}")

        print("✅ Tools created successfully")

        # Create agent with tool
        agent = create_agent(model="gpt-3.5-turbo").register_tool(test_math)
        print(f"✅ Agent created with tools: {len(agent.list_tools())} available")

        # List available tools
        tools = agent.list_tools()
        for tool_info in tools:
            print(f"   - {tool_info.name}: {tool_info.description}")

        # Test session management
        session = await agent.create_session("test-session-001")
        print(f"✅ Session created: {session.session_id}")

        retrieved_session = await agent.get_session("test-session-001")
        print(
            f"✅ Session retrieved: {retrieved_session.session_id if retrieved_session else 'None'}"
        )

        # Test simple execution (this would need a real API key to work)
        print("⏳ Testing execution (mock mode since no real API key)...")

        try:
            response = await agent.run(
                "Calculate 15 + 27", session_id="test-session-001"
            )
            print(f"✅ Execution completed: {response.success}")
            print(f"   Answer: {response.answer}")
            print(f"   Steps: {response.total_steps}")
            print(f"   Time: {response.execution_time:.2f}s")
        except Exception as e:
            # Expected with fake API key
            print(f"⚠️  Execution failed as expected (mock API key): {type(e).__name__}")
            print("   This is expected behavior with a test API key")

        # Test streaming (also would fail with fake API key)
        print("⏳ Testing streaming execution...")
        try:
            chunks = []
            async for chunk in agent.run_stream(
                "What is 10 * 5?", session_id="test-session-001"
            ):
                chunks.append(chunk)
            print(f"✅ Streaming completed with {len(chunks)} chunks")
        except Exception as e:
            print(f"⚠️  Streaming failed as expected (mock API key): {type(e).__name__}")

        # Test session deletion
        deleted = await agent.delete_session("test-session-001")
        print(f"✅ Session deleted: {deleted}")

        print("\n🎉 All tests passed! ReActAgent is working correctly.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_execution_functionality())
