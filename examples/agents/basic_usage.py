#!/usr/bin/env python3
"""
AetherFlow Agent Configuration System - Basic Usage Examples

This script demonstrates how to use the AetherFlow agent configuration system
with different configuration sources and patterns.
"""

import os
from pathlib import Path

# Import agent system components
from agents import (
    AgentConfig,
    ContainerFactory,
    create_agent_container,
    with_config,
)


def example_1_basic_usage():
    """Example 1: Basic container creation with default configuration."""
    print("=== Example 1: Basic Usage ===")

    # Set required environment variable
    os.environ["OPENAI_API_KEY"] = "sk-your-api-key-here"

    # Create container with defaults
    container = create_agent_container()

    # Access configuration
    config = container.config()
    print(f"LLM Model: {config['llm']['model']}")
    print(f"Max Workers: {config['tools']['max_workers']}")
    print(f"Max Messages: {config['memory']['max_messages']}")

    # Perform health check
    health = container.health_check()
    print(f"Container Health: {health['container']}")

    return container


def example_2_environment_variables():
    """Example 2: Configuration via environment variables."""
    print("\n=== Example 2: Environment Variables ===")

    # Set environment variables for configuration
    env_vars = {
        "AETHERFLOW_AGENT_LLM_MODEL": "gpt-3.5-turbo",
        "AETHERFLOW_AGENT_LLM_TEMPERATURE": "0.3",
        "AETHERFLOW_AGENT_TOOLS_MAX_WORKERS": "8",
        "AETHERFLOW_AGENT_MEMORY_MAX_MESSAGES": "500",
        "AETHERFLOW_AGENT_REACT_MAX_STEPS": "15",
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    try:
        # Create container - will automatically load from environment
        container = create_agent_container()
        config = container.config()

        print(f"Model (from env): {config['llm']['model']}")
        print(f"Temperature (from env): {config['llm']['temperature']}")
        print(f"Max Workers (from env): {config['tools']['max_workers']}")
        print(f"Max Messages (from env): {config['memory']['max_messages']}")
        print(f"Max Steps (from env): {config['react']['max_steps']}")

        return container

    finally:
        # Clean up environment variables
        for key in env_vars:
            os.environ.pop(key, None)


def example_3_config_file():
    """Example 3: Configuration via YAML file."""
    print("\n=== Example 3: Configuration File ===")

    # Create a temporary configuration file
    config_content = """
llm:
  api_key: "sk-file-config-key"
  model: "gpt-4"
  temperature: 0.2
  max_tokens: 1500

tools:
  max_workers: 6
  timeout: 45

memory:
  max_messages: 750
  enable_compression: true

react:
  max_steps: 12
  enable_streaming: false

logging:
  level: "DEBUG"
  handlers: ["console"]
"""

    config_file = Path("temp_config.yaml")
    config_file.write_text(config_content)

    try:
        # Create container using configuration file
        container = create_agent_container(config_file="temp_config.yaml")
        config = container.config()

        print(f"Model (from file): {config['llm']['model']}")
        print(f"Temperature (from file): {config['llm']['temperature']}")
        print(f"Max Tokens (from file): {config['llm']['max_tokens']}")
        print(f"Max Workers (from file): {config['tools']['max_workers']}")
        print(f"Tool Timeout (from file): {config['tools']['timeout']}")
        print(
            f"Compression Enabled (from file): {config['memory']['enable_compression']}"
        )

        return container

    finally:
        # Clean up temporary file
        if config_file.exists():
            config_file.unlink()


def example_4_programmatic_config():
    """Example 4: Programmatic configuration with overrides."""
    print("\n=== Example 4: Programmatic Configuration ===")

    # Create container with inline configuration overrides
    container = create_agent_container(
        llm_api_key="sk-programmatic-key",
        llm_model="gpt-4",
        llm_temperature=0.5,
        llm_max_tokens=3000,
        tools_max_workers=12,
        memory_max_messages=2000,
        react_max_steps=20,
        logging_level="WARNING",
    )

    config = container.config()
    print(f"Model (programmatic): {config['llm']['model']}")
    print(f"Temperature (programmatic): {config['llm']['temperature']}")
    print(f"Max Tokens (programmatic): {config['llm']['max_tokens']}")
    print(f"Max Workers (programmatic): {config['tools']['max_workers']}")
    print(f"Max Messages (programmatic): {config['memory']['max_messages']}")
    print(f"Max Steps (programmatic): {config['react']['max_steps']}")

    return container


def example_5_factory_methods():
    """Example 5: Using ContainerFactory for advanced scenarios."""
    print("\n=== Example 5: Factory Methods ===")

    # Method 1: Create test container
    test_container = ContainerFactory.create_test_container(
        llm_model="gpt-3.5-turbo", tools_max_workers=2
    )

    test_config = test_container.config()
    print("Test Container Configuration:")
    print(f"  API Key: {test_config['llm']['api_key']}")  # Should be test key
    print(f"  Model: {test_config['llm']['model']}")
    print(f"  Auto Save: {test_config['storage']['auto_save']}")  # Should be False
    print(f"  Monitoring: {test_config['monitoring']['enabled']}")  # Should be False

    # Method 2: Use default container singleton
    os.environ["OPENAI_API_KEY"] = "sk-default-key"

    try:
        # Reset any existing default
        ContainerFactory.reset_default_container()

        # Get default container (creates if needed)
        default1 = ContainerFactory.get_default_container()
        default2 = ContainerFactory.get_default_container()

        print(f"\nDefault containers are same instance: {default1 is default2}")

        # Method 3: Create container with pre-built config
        custom_config = AgentConfig(
            llm={"api_key": "sk-custom-key", "model": "gpt-4", "temperature": 0.8},
            tools={"max_workers": 16},
            memory={"max_messages": 3000},
        )

        custom_container = ContainerFactory.create_container(config=custom_config)
        custom_config_dict = custom_container.config()

        print(f"\nCustom container model: {custom_config_dict['llm']['model']}")
        print(
            f"Custom container temperature: {custom_config_dict['llm']['temperature']}"
        )

    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        ContainerFactory.reset_default_container()


def example_6_configuration_context():
    """Example 6: Temporary configuration overrides using context manager."""
    print("\n=== Example 6: Configuration Context ===")

    # Create base container
    container = ContainerFactory.create_test_container()
    original_config = container.config()

    print("Original Configuration:")
    print(f"  Model: {original_config['llm']['model']}")
    print(f"  Temperature: {original_config['llm']['temperature']}")
    print(f"  Max Steps: {original_config['react']['max_steps']}")

    # Use context manager for temporary overrides
    from agents.container import ConfigContext

    overrides = {
        "llm": {"model": "gpt-4", "temperature": 0.9},
        "react": {"max_steps": 25},
    }

    with ConfigContext(container, overrides) as ctx_container:
        temp_config = ctx_container.config()
        print("\nTemporary Configuration (within context):")
        print(f"  Model: {temp_config['llm']['model']}")
        print(f"  Temperature: {temp_config['llm']['temperature']}")
        print(f"  Max Steps: {temp_config['react']['max_steps']}")

    # Configuration should be restored
    restored_config = container.config()
    print("\nRestored Configuration (after context):")
    print(f"  Model: {restored_config['llm']['model']}")
    print(f"  Temperature: {restored_config['llm']['temperature']}")
    print(f"  Max Steps: {restored_config['react']['max_steps']}")


def example_7_decorator_usage():
    """Example 7: Using the @with_config decorator."""
    print("\n=== Example 7: Configuration Decorator ===")

    # Set up a default container
    os.environ["OPENAI_API_KEY"] = "sk-decorator-test"

    try:
        default_container = ContainerFactory.get_default_container()
        original_model = default_container.config()["llm"]["model"]

        print(f"Original model: {original_model}")

        @with_config(llm_model="decorator-model", llm_temperature=0.7)
        def creative_task():
            """A function that uses creative LLM settings."""
            current_container = ContainerFactory.get_default_container()
            config = current_container.config()

            print(f"  Inside function - Model: {config['llm']['model']}")
            print(f"  Inside function - Temperature: {config['llm']['temperature']}")

            return f"Processed with {config['llm']['model']}"

        # Call the decorated function
        result = creative_task()
        print(f"Function result: {result}")

        # Check that configuration was restored
        final_config = default_container.config()
        print(f"Model after function: {final_config['llm']['model']}")

    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        ContainerFactory.reset_default_container()


def example_8_configuration_priority():
    """Example 8: Demonstrate configuration priority system."""
    print("\n=== Example 8: Configuration Priority ===")

    # Create a config file (lowest priority)
    config_content = """
llm:
  model: "file-model"
  temperature: 0.1
  max_tokens: 1000

tools:
  max_workers: 4
"""

    config_file = Path("priority_test.yaml")
    config_file.write_text(config_content)

    # Set environment variables (highest priority)
    os.environ["AETHERFLOW_AGENT_LLM_MODEL"] = "env-model"
    os.environ["AETHERFLOW_AGENT_TOOLS_MAX_WORKERS"] = "8"
    os.environ["OPENAI_API_KEY"] = "sk-priority-test"

    try:
        # Create container with all three sources
        container = ContainerFactory.create_container(
            config_file="priority_test.yaml",
            llm_temperature=0.5,  # Programmatic (medium priority)
            memory_max_messages=1500,
        )

        config = container.config()

        print("Configuration Priority Results:")
        print(
            f"  Model: {config['llm']['model']} (should be 'env-model' from environment)"
        )
        print(
            f"  Temperature: {config['llm']['temperature']} (should be 0.5 from programmatic)"
        )
        print(f"  Max Tokens: {config['llm']['max_tokens']} (should be 1000 from file)")
        print(
            f"  Max Workers: {config['tools']['max_workers']} (should be 8 from environment)"
        )
        print(
            f"  Max Messages: {config['memory']['max_messages']} (should be 1500 from programmatic)"
        )

        print(
            "\nPriority Order: Environment Variables > Programmatic > Configuration File > Defaults"
        )

    finally:
        # Clean up
        os.environ.pop("AETHERFLOW_AGENT_LLM_MODEL", None)
        os.environ.pop("AETHERFLOW_AGENT_TOOLS_MAX_WORKERS", None)
        os.environ.pop("OPENAI_API_KEY", None)
        if config_file.exists():
            config_file.unlink()


def example_9_error_handling():
    """Example 9: Configuration error handling and validation."""
    print("\n=== Example 9: Error Handling ===")

    try:
        # Try to create container with invalid API key
        print("Testing invalid API key...")
        container = ContainerFactory.create_container(llm_api_key="invalid-key")
    except Exception as e:
        print(f"✓ Caught expected error: {e}")

    try:
        # Try to create container with missing required file
        print("\nTesting missing required config file...")
        container = ContainerFactory.create_container(
            config_file="/nonexistent/config.yaml"
        )
    except Exception as e:
        print(f"✓ Caught expected error: {e}")

    # Valid configuration with validation
    print("\nTesting valid configuration...")
    container = ContainerFactory.create_test_container()
    health = container.health_check()
    print(f"Health check result: {health['container']}")

    if health["errors"]:
        print("Health check errors:")
        for error in health["errors"]:
            print(f"  - {error}")


def main():
    """Run all configuration examples."""
    print("AetherFlow Agent Configuration System - Usage Examples")
    print("=" * 60)

    # Run all examples
    example_1_basic_usage()
    example_2_environment_variables()
    example_3_config_file()
    example_4_programmatic_config()
    example_5_factory_methods()
    example_6_configuration_context()
    example_7_decorator_usage()
    example_8_configuration_priority()
    example_9_error_handling()

    print("\n" + "=" * 60)
    print("All examples completed!")


if __name__ == "__main__":
    main()
