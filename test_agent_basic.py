#!/usr/bin/env python3
"""
Basic test for ReActAgent to check if imports work and basic functionality.
"""

import asyncio
import os

# Set up environment
os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing"


async def test_basic_imports():
    """Test basic imports and initialization."""
    try:
        from agents import ReActAgent, create_agent

        print("✅ Import successful")

        # Test basic creation
        agent = create_agent(model="gpt-3.5-turbo")
        print(f"✅ Agent created: {agent}")

        # Test fluent interface
        agent2 = ReActAgent().with_model("gpt-4").with_max_steps(5)
        print(f"✅ Fluent interface works: {agent2}")

        # Test builder pattern
        from agents import create_agent_builder

        agent3 = (
            create_agent_builder().model("gpt-4").max_steps(8).temperature(0.2).build()
        )
        print(f"✅ Builder pattern works: {agent3}")

        # Test status
        status = agent.get_status()
        print(f"✅ Status check works: {status}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_basic_imports())
