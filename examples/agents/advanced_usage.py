#!/usr/bin/env python3
"""
AetherFlow Agent Configuration System - Advanced Usage Examples

This script demonstrates advanced patterns for using the AetherFlow agent
configuration system including custom configuration sources, configuration
validation, and integration with different environments.
"""

import os
from contextlib import contextmanager
from typing import Any

# Import agent system components
from agents import (
    AgentConfig,
    ConfigContext,
    ConfigLoader,
    ContainerFactory,
    with_config,
)


class DatabaseConfigSource:
    """Custom configuration source that loads from a simulated database."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # Simulate database configuration data
        self._db_config = {
            "llm": {"model": "gpt-4-database", "temperature": 0.15},
            "tools": {"max_workers": 10},
            "monitoring": {
                "enabled": True,
                "backend": "http",
                "endpoint": "http://monitoring.example.com/metrics",
            },
        }

    def load(self) -> dict[str, Any]:
        """Simulate loading configuration from database."""
        print(f"Loading configuration from database: {self.connection_string}")
        return self._db_config.copy()


class RemoteConfigSource:
    """Custom configuration source that loads from a remote API."""

    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        # Simulate remote configuration data
        self._remote_config = {
            "llm": {"model": "gpt-4-remote", "max_tokens": 3000},
            "memory": {"max_messages": 1500, "trimming_strategy": "importance_based"},
        }

    def load(self) -> dict[str, Any]:
        """Simulate loading configuration from remote API."""
        print(f"Loading configuration from remote API: {self.api_endpoint}")
        # In a real implementation, this would make an HTTP request
        return self._remote_config.copy()


def example_1_custom_config_sources():
    """Example 1: Using custom configuration sources."""
    print("=== Example 1: Custom Configuration Sources ===")

    # Create a custom config loader with multiple sources
    loader = ConfigLoader()

    # Add default configuration (lowest priority)
    defaults = {
        "llm": {"api_key": "sk-default-key", "model": "gpt-3.5-turbo"},
        "tools": {"max_workers": 4},
    }
    loader.add_dict_source(defaults)

    # Add database configuration (medium priority)
    db_source = DatabaseConfigSource("postgresql://localhost/config")
    loader.config_sources.append(db_source)

    # Add remote configuration (higher priority)
    remote_source = RemoteConfigSource("https://config.example.com/api", "api-key-123")
    loader.config_sources.append(remote_source)

    # Add environment variables (highest priority)
    os.environ["AETHERFLOW_AGENT_LLM_TEMPERATURE"] = "0.3"
    os.environ["AETHERFLOW_AGENT_REACT_MAX_STEPS"] = "15"

    try:
        loader.add_env_source()

        # Load merged configuration
        config = loader.load()

        print("Merged configuration from custom sources:")
        print(f"  Model: {config.llm.model} (from remote)")
        print(f"  Temperature: {config.llm.temperature} (from environment)")
        print(f"  Max Workers: {config.tools.max_workers} (from database)")
        print(f"  Max Tokens: {config.llm.max_tokens} (from remote)")
        print(f"  Max Steps: {config.react.max_steps} (from environment)")

        return config

    finally:
        os.environ.pop("AETHERFLOW_AGENT_LLM_TEMPERATURE", None)
        os.environ.pop("AETHERFLOW_AGENT_REACT_MAX_STEPS", None)


def example_2_environment_specific_configs():
    """Example 2: Environment-specific configuration management."""
    print("\n=== Example 2: Environment-Specific Configuration ===")

    # Define configuration for different environments
    environments = {
        "development": {
            "llm": {"model": "gpt-3.5-turbo", "temperature": 0.0, "max_tokens": 500},
            "tools": {"max_workers": 2},
            "memory": {"max_messages": 50},
            "monitoring": {"enabled": False},
            "logging": {"level": "DEBUG"},
        },
        "staging": {
            "llm": {"model": "gpt-4", "temperature": 0.1, "max_tokens": 1500},
            "tools": {"max_workers": 4},
            "memory": {"max_messages": 500},
            "monitoring": {"enabled": True, "backend": "file"},
            "logging": {"level": "INFO"},
        },
        "production": {
            "llm": {"model": "gpt-4", "temperature": 0.05, "max_tokens": 2000},
            "tools": {"max_workers": 8},
            "memory": {"max_messages": 1000},
            "monitoring": {"enabled": True, "backend": "http"},
            "logging": {"level": "WARNING"},
        },
    }

    def create_env_container(env: str) -> AgentConfig:
        """Create container for specific environment."""
        base_config = {"llm": {"api_key": f"sk-{env}-key"}}

        env_config = environments.get(env, {})

        # Merge base config with environment-specific config
        merged_config = ContainerFactory._deep_merge(base_config, env_config)

        return ContainerFactory.create_container(**merged_config)

    # Test each environment
    for env_name in ["development", "staging", "production"]:
        print(f"\n{env_name.title()} Environment:")
        container = create_env_container(env_name)
        config = container.config()

        print(f"  Model: {config['llm']['model']}")
        print(f"  Temperature: {config['llm']['temperature']}")
        print(f"  Max Tokens: {config['llm']['max_tokens']}")
        print(f"  Max Workers: {config['tools']['max_workers']}")
        print(f"  Monitoring: {config['monitoring']['enabled']}")
        print(f"  Log Level: {config['logging']['level']}")


@contextmanager
def temporary_environment(env_vars: dict[str, str]):
    """Context manager for temporarily setting environment variables."""
    original_values = {}

    # Set new values and save originals
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        yield
    finally:
        # Restore original values
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def example_3_configuration_validation():
    """Example 3: Custom configuration validation and error handling."""
    print("\n=== Example 3: Configuration Validation ===")

    class ValidatedAgentConfig(AgentConfig):
        """Extended configuration with custom validation."""

        def validate_complete(self) -> None:
            """Extended validation with custom business rules."""
            # Call parent validation first
            super().validate_complete()

            # Custom validation rules
            if self.llm.temperature > 0.8 and self.react.max_steps > 15:
                raise ValueError(
                    "High temperature (>0.8) with many steps (>15) may produce "
                    "inconsistent results"
                )

            if self.tools.max_workers > 16:
                print("WARNING: High worker count may cause resource contention")

            if self.memory.max_tokens < self.memory.reserve_tokens * 2:
                print(
                    "WARNING: Low max_tokens to reserve_tokens ratio may cause "
                    "frequent memory trimming"
                )

            # Environment-specific validation
            if os.environ.get("ENVIRONMENT") == "production":
                if self.llm.temperature > 0.2:
                    print("WARNING: High temperature in production environment")

                if not self.monitoring.enabled:
                    print("WARNING: Monitoring disabled in production environment")

    # Test valid configuration
    print("Testing valid configuration...")
    try:
        config = ValidatedAgentConfig(
            llm={"api_key": "sk-valid-key", "model": "gpt-4", "temperature": 0.1},
            react={"max_steps": 10},
        )
        config.validate_complete()
        print("✓ Configuration validation passed")
    except Exception as e:
        print(f"✗ Validation failed: {e}")

    # Test invalid configuration
    print("\nTesting invalid configuration...")
    try:
        config = ValidatedAgentConfig(
            llm={"api_key": "sk-valid-key", "model": "gpt-4", "temperature": 0.9},
            react={"max_steps": 20},
        )
        config.validate_complete()
        print("✓ Configuration validation passed")
    except Exception as e:
        print(f"✗ Expected validation failure: {e}")

    # Test production environment validation
    print("\nTesting production environment validation...")
    with temporary_environment({"ENVIRONMENT": "production"}):
        config = ValidatedAgentConfig(
            llm={"api_key": "sk-valid-key", "model": "gpt-4", "temperature": 0.3},
            monitoring={"enabled": False},
        )
        config.validate_complete()  # Should show warnings


def example_4_dynamic_configuration():
    """Example 4: Dynamic configuration updates and hot reloading."""
    print("\n=== Example 4: Dynamic Configuration Updates ===")

    # Create initial container
    container = ContainerFactory.create_test_container(
        llm_model="gpt-3.5-turbo", llm_temperature=0.1
    )

    print("Initial configuration:")
    config = container.config()
    print(f"  Model: {config['llm']['model']}")
    print(f"  Temperature: {config['llm']['temperature']}")
    print(f"  Max Steps: {config['react']['max_steps']}")

    # Simulate configuration updates
    updates = [
        {
            "name": "Performance optimization",
            "changes": {"llm": {"temperature": 0.0}, "tools": {"max_workers": 8}},
        },
        {
            "name": "Creative mode",
            "changes": {"llm": {"temperature": 0.8}, "react": {"max_steps": 20}},
        },
        {
            "name": "Conservative mode",
            "changes": {"llm": {"temperature": 0.05}, "react": {"max_steps": 5}},
        },
    ]

    for update in updates:
        print(f"\nApplying update: {update['name']}")

        with ConfigContext(container, update["changes"]) as ctx_container:
            updated_config = ctx_container.config()
            print(f"  Model: {updated_config['llm']['model']}")
            print(f"  Temperature: {updated_config['llm']['temperature']}")
            print(f"  Max Workers: {updated_config['tools']['max_workers']}")
            print(f"  Max Steps: {updated_config['react']['max_steps']}")

            # Simulate using the updated configuration
            print(f"  → Running with {update['name']} configuration")

        print("  Configuration restored to original")

    # Final state
    final_config = container.config()
    print("\nFinal configuration:")
    print(f"  Temperature: {final_config['llm']['temperature']} (should be 0.1)")


def example_5_configuration_profiles():
    """Example 5: Configuration profiles for different use cases."""
    print("\n=== Example 5: Configuration Profiles ===")

    # Define different configuration profiles
    profiles = {
        "quick_responses": {
            "name": "Quick Responses",
            "description": "Optimized for fast, short responses",
            "config": {
                "llm": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.0,
                    "max_tokens": 500,
                },
                "react": {"max_steps": 5},
                "memory": {"max_messages": 20},
            },
        },
        "detailed_analysis": {
            "name": "Detailed Analysis",
            "description": "Optimized for thorough, detailed analysis",
            "config": {
                "llm": {"model": "gpt-4", "temperature": 0.1, "max_tokens": 3000},
                "react": {"max_steps": 25},
                "memory": {"max_messages": 100},
            },
        },
        "creative_tasks": {
            "name": "Creative Tasks",
            "description": "Optimized for creative and diverse outputs",
            "config": {
                "llm": {"model": "gpt-4", "temperature": 0.8, "max_tokens": 2000},
                "react": {"max_steps": 15},
                "memory": {"max_messages": 50},
            },
        },
        "batch_processing": {
            "name": "Batch Processing",
            "description": "Optimized for processing multiple tasks efficiently",
            "config": {
                "llm": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.1,
                    "max_tokens": 1000,
                },
                "tools": {"max_workers": 16},
                "react": {"max_steps": 10},
            },
        },
    }

    @with_config()
    def simulate_task(task_name: str):
        """Simulate running a task with current configuration."""
        container = ContainerFactory.get_default_container()
        config = container.config()

        print(f"    Executing '{task_name}':")
        print(f"      Model: {config['llm']['model']}")
        print(f"      Temperature: {config['llm']['temperature']}")
        print(f"      Max Tokens: {config['llm']['max_tokens']}")
        print(f"      Max Steps: {config['react']['max_steps']}")

    # Set up base container
    os.environ["OPENAI_API_KEY"] = "sk-profile-test"

    try:
        base_container = ContainerFactory.get_default_container()

        for profile_id, profile in profiles.items():
            print(f"\n{profile['name']}:")
            print(f"  Description: {profile['description']}")

            # Apply profile configuration
            with ConfigContext(base_container, profile["config"]):
                if profile_id == "quick_responses":
                    simulate_task("Status check")
                elif profile_id == "detailed_analysis":
                    simulate_task("Research analysis")
                elif profile_id == "creative_tasks":
                    simulate_task("Story generation")
                elif profile_id == "batch_processing":
                    simulate_task("Data processing batch")

    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        ContainerFactory.reset_default_container()


def example_6_configuration_testing():
    """Example 6: Testing with different configurations."""
    print("\n=== Example 6: Configuration Testing ===")

    # Test configurations for different scenarios
    test_scenarios = [
        {
            "name": "Minimal Configuration",
            "config": {
                "llm_model": "gpt-3.5-turbo",
                "llm_max_tokens": 100,
                "tools_max_workers": 1,
                "memory_max_messages": 5,
            },
            "expected_behavior": "Fast, minimal resource usage",
        },
        {
            "name": "Maximum Configuration",
            "config": {
                "llm_model": "gpt-4",
                "llm_max_tokens": 4000,
                "tools_max_workers": 16,
                "memory_max_messages": 5000,
            },
            "expected_behavior": "Comprehensive, resource-intensive",
        },
        {
            "name": "Balanced Configuration",
            "config": {
                "llm_model": "gpt-4",
                "llm_max_tokens": 2000,
                "tools_max_workers": 8,
                "memory_max_messages": 1000,
            },
            "expected_behavior": "Good balance of capability and efficiency",
        },
    ]

    def run_test_scenario(scenario):
        """Run a test scenario with specific configuration."""
        print(f"\nTesting: {scenario['name']}")
        print(f"Expected: {scenario['expected_behavior']}")

        container = ContainerFactory.create_test_container(**scenario["config"])
        config = container.config()
        health = container.health_check()

        print("Results:")
        print(f"  Health: {health['container']}")
        print(f"  Model: {config['llm']['model']}")
        print(f"  Max Tokens: {config['llm']['max_tokens']}")
        print(f"  Workers: {config['tools']['max_workers']}")
        print(f"  Memory: {config['memory']['max_messages']}")

        # Simulate performance metrics
        estimated_cost = {
            "gpt-3.5-turbo": config["llm"]["max_tokens"] * 0.001,
            "gpt-4": config["llm"]["max_tokens"] * 0.006,
        }.get(config["llm"]["model"], 0)

        estimated_speed = max(1, 10 - config["llm"]["max_tokens"] // 500)

        print(f"  Estimated cost per call: ${estimated_cost:.4f}")
        print(f"  Estimated speed (1-10): {estimated_speed}")

        return {
            "health": health["container"],
            "cost": estimated_cost,
            "speed": estimated_speed,
        }

    # Run all test scenarios
    results = []
    for scenario in test_scenarios:
        result = run_test_scenario(scenario)
        result["name"] = scenario["name"]
        results.append(result)

    # Summary
    print("\nTest Summary:")
    for result in results:
        print(
            f"  {result['name']}: "
            f"Health={result['health']}, "
            f"Cost=${result['cost']:.4f}, "
            f"Speed={result['speed']}/10"
        )


def main():
    """Run all advanced configuration examples."""
    print("AetherFlow Agent Configuration System - Advanced Usage Examples")
    print("=" * 70)

    # Run all examples
    example_1_custom_config_sources()
    example_2_environment_specific_configs()
    example_3_configuration_validation()
    example_4_dynamic_configuration()
    example_5_configuration_profiles()
    example_6_configuration_testing()

    print("\n" + "=" * 70)
    print("All advanced examples completed!")


if __name__ == "__main__":
    main()
