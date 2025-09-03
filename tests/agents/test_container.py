"""
Tests for Agent Container System

Tests dependency injection container functionality, factory methods,
configuration context management, and service lifecycle.
"""

import os
import tempfile
from pathlib import Path

import pytest

from agents.config import AgentConfig, ConfigurationError
from agents.container import (
    AgentContainer,
    ConfigContext,
    ContainerFactory,
    with_config,
)


class TestAgentContainer:
    """Test AgentContainer functionality."""

    def test_container_inherits_base_flow_context(self):
        """Test that AgentContainer properly inherits from BaseFlowContext."""
        container = AgentContainer()

        # Verify inheritance from BaseFlowContext
        assert hasattr(container, "state")
        assert hasattr(container, "context")
        assert hasattr(container, "shared_data")
        assert hasattr(container, "async_state")
        assert hasattr(container, "async_context")

        # Verify agent-specific providers exist
        assert hasattr(container, "config")
        assert hasattr(container, "llm_client")
        assert hasattr(container, "tool_registry")
        assert hasattr(container, "conversation_memory")
        assert hasattr(container, "react_engine")

    def test_container_configuration(self):
        """Test container configuration setup."""
        # Create test configuration
        test_config = AgentConfig(
            llm={"api_key": "sk-test-key", "model": "gpt-3.5-turbo"},
            tools={"max_workers": 8},
            memory={"max_messages": 500},
        )

        container = AgentContainer()
        container.config.from_dict(test_config.dict_for_container())

        # Verify configuration is accessible
        config_dict = container.config()
        assert config_dict["llm"]["model"] == "gpt-3.5-turbo"
        assert config_dict["tools"]["max_workers"] == 8
        assert config_dict["memory"]["max_messages"] == 500

    def test_container_logging_configuration(self):
        """Test container logging configuration."""
        test_config = AgentConfig(
            llm={"api_key": "sk-test-key"},
            logging={
                "level": "DEBUG",
                "format": "%(name)s - %(message)s",
                "handlers": ["console"],
            },
        )

        container = ContainerFactory.create_container(config=test_config)

        # Configure logging should not raise errors
        container.configure_logging()

        # Verify logger is created
        logger = container.agent_logger()
        assert logger is not None
        assert logger.name == "aetherflow.agent"

    def test_container_health_check(self):
        """Test container health check functionality."""
        # Valid configuration
        test_config = AgentConfig(llm={"api_key": "sk-test-key"})

        container = ContainerFactory.create_container(config=test_config)

        health = container.health_check()

        assert health["container"] in ["healthy", "degraded"]
        assert "services" in health
        assert "config" in health["services"]
        assert "logging" in health["services"]
        assert "errors" in health

        # Test with invalid configuration by creating a new container
        try:
            invalid_config = AgentConfig(llm={"api_key": ""})
            ContainerFactory.create_container(config=invalid_config)
            assert False, "Should have failed with invalid config"
        except Exception:
            pass  # Expected to fail


class TestContainerFactory:
    """Test ContainerFactory functionality."""

    def test_create_container_with_defaults(self):
        """Test creating container with default configuration."""
        # Set required API key
        os.environ["OPENAI_API_KEY"] = "sk-test-key"

        try:
            container = ContainerFactory.create_container()

            # Due to dependency-injector design, we get a DynamicContainer
            assert hasattr(container, "config")
            assert hasattr(container, "configure_logging")
            assert hasattr(container, "health_check")

            # Verify container is configured
            config_dict = container.config()
            assert config_dict is not None
            assert config_dict["llm"]["api_key"] == "sk-test-key"

        finally:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_create_container_with_config_overrides(self):
        """Test creating container with configuration overrides."""
        container = ContainerFactory.create_container(
            llm_api_key="sk-override-key",
            llm_model="gpt-4",
            tools_max_workers=12,
            memory_max_messages=2000,
        )

        config_dict = container.config()
        assert config_dict["llm"]["api_key"] == "sk-override-key"
        assert config_dict["llm"]["model"] == "gpt-4"
        assert config_dict["tools"]["max_workers"] == 12
        assert config_dict["memory"]["max_messages"] == 2000

    def test_create_container_with_config_file(self):
        """Test creating container with configuration file."""
        # Create temporary config file
        config_content = """
llm:
  api_key: "sk-file-key"
  model: "gpt-3.5-turbo"
  temperature: 0.3

tools:
  max_workers: 6
  timeout: 45

memory:
  max_messages: 750
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            container = ContainerFactory.create_container(config_file=config_path)

            config_dict = container.config()
            assert config_dict["llm"]["api_key"] == "sk-file-key"
            assert config_dict["llm"]["model"] == "gpt-3.5-turbo"
            assert config_dict["llm"]["temperature"] == 0.3
            assert config_dict["tools"]["max_workers"] == 6
            assert config_dict["memory"]["max_messages"] == 750

        finally:
            Path(config_path).unlink()

    def test_create_container_with_pre_built_config(self):
        """Test creating container with pre-built AgentConfig."""
        config = AgentConfig(
            llm={"api_key": "sk-prebuilt-key", "model": "gpt-4"},
            react={"max_steps": 25},
        )

        container = ContainerFactory.create_container(config=config)

        config_dict = container.config()
        assert config_dict["llm"]["api_key"] == "sk-prebuilt-key"
        assert config_dict["llm"]["model"] == "gpt-4"
        assert config_dict["react"]["max_steps"] == 25

    def test_get_default_container(self):
        """Test default container singleton behavior."""
        # Set required environment
        os.environ["OPENAI_API_KEY"] = "sk-test-key"

        try:
            # Reset any existing default container
            ContainerFactory.reset_default_container()

            # First call should create container
            container1 = ContainerFactory.get_default_container()
            # Due to dependency-injector design, we get a DynamicContainer
            assert hasattr(container1, "config")
            assert hasattr(container1, "configure_logging")

            # Second call should return same instance
            container2 = ContainerFactory.get_default_container()
            assert container1 is container2

        finally:
            os.environ.pop("OPENAI_API_KEY", None)
            ContainerFactory.reset_default_container()

    def test_reset_default_container(self):
        """Test resetting default container."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"

        try:
            # Create default container
            container1 = ContainerFactory.get_default_container()

            # Reset and create again
            ContainerFactory.reset_default_container()
            container2 = ContainerFactory.get_default_container()

            # Should be different instances
            assert container1 is not container2

        finally:
            os.environ.pop("OPENAI_API_KEY", None)
            ContainerFactory.reset_default_container()

    def test_create_test_container(self):
        """Test creating container configured for testing."""
        container = ContainerFactory.create_test_container()

        config_dict = container.config()

        # Verify test-specific defaults
        assert config_dict["llm"]["api_key"] == "sk-test-key-for-testing"
        assert config_dict["llm"]["model"] == "gpt-3.5-turbo"
        assert config_dict["llm"]["temperature"] == 0.0
        assert config_dict["memory"]["max_messages"] == 10
        assert config_dict["storage"]["auto_save"] is False
        assert config_dict["monitoring"]["enabled"] is False

    def test_create_test_container_with_overrides(self):
        """Test creating test container with configuration overrides."""
        container = ContainerFactory.create_test_container(
            llm_model="gpt-4", tools_max_workers=2
        )

        config_dict = container.config()

        # Overrides should be applied
        assert config_dict["llm"]["model"] == "gpt-4"
        assert config_dict["tools"]["max_workers"] == 2

        # Test defaults should still be present for non-overridden values
        assert config_dict["llm"]["api_key"] == "sk-test-key-for-testing"
        assert config_dict["llm"]["temperature"] == 0.0

    def test_create_container_invalid_config(self):
        """Test error handling for invalid configuration."""
        with pytest.raises(ConfigurationError):
            ContainerFactory.create_container(
                llm_api_key=""  # Invalid empty API key
            )


class TestConfigContext:
    """Test configuration context manager."""

    def test_config_context_override(self):
        """Test temporary configuration override."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"

        try:
            container = ContainerFactory.create_container()
            original_model = container.config()["llm"]["model"]

            # Use context manager to temporarily override configuration
            overrides = {"llm": {"model": "gpt-3.5-turbo-override"}}

            with ConfigContext(container, overrides) as ctx_container:
                # Configuration should be overridden within context
                config_dict = ctx_container.config()
                assert config_dict["llm"]["model"] == "gpt-3.5-turbo-override"

                # Should be same container instance
                assert ctx_container is container

            # Configuration should be restored after context
            restored_config = container.config()
            assert restored_config["llm"]["model"] == original_model

        finally:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_config_context_nested_override(self):
        """Test nested configuration override."""
        container = ContainerFactory.create_test_container()

        original_config = container.config()
        original_model = original_config["llm"]["model"]
        original_temp = original_config["llm"]["temperature"]

        # First level override
        with ConfigContext(container, {"llm": {"model": "override-1"}}) as ctx1:
            assert ctx1.config()["llm"]["model"] == "override-1"
            assert ctx1.config()["llm"]["temperature"] == original_temp

            # Nested override
            with ConfigContext(container, {"llm": {"temperature": 0.8}}) as ctx2:
                # Should have both overrides
                config = ctx2.config()
                assert config["llm"]["model"] == "override-1"  # From outer context
                assert config["llm"]["temperature"] == 0.8  # From inner context

            # Back to first level
            config = ctx1.config()
            assert config["llm"]["model"] == "override-1"
            assert config["llm"]["temperature"] == original_temp

        # Back to original
        final_config = container.config()
        assert final_config["llm"]["model"] == original_model
        assert final_config["llm"]["temperature"] == original_temp


class TestWithConfigDecorator:
    """Test the with_config decorator."""

    def test_with_config_decorator(self):
        """Test configuration decorator functionality."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"

        try:
            # Create default container
            container = ContainerFactory.get_default_container()
            original_model = container.config()["llm"]["model"]

            @with_config(llm_model="decorator-model", llm_temperature=0.9)
            def test_function():
                current_container = ContainerFactory.get_default_container()
                config = current_container.config()
                return config["llm"]["model"], config["llm"]["temperature"]

            # Call decorated function
            model, temperature = test_function()

            # Should have overridden values during function execution
            assert model == "decorator-model"
            assert temperature == 0.9

            # Configuration should be restored after function
            final_config = container.config()
            assert final_config["llm"]["model"] == original_model

        finally:
            os.environ.pop("OPENAI_API_KEY", None)
            ContainerFactory.reset_default_container()

    def test_with_config_decorator_with_exception(self):
        """Test that configuration is restored even when decorated function raises exception."""
        container = ContainerFactory.create_test_container()
        original_model = container.config()["llm"]["model"]

        @with_config(llm_model="exception-model")
        def failing_function():
            # Verify override is applied
            current_config = ContainerFactory.get_default_container().config()
            assert current_config["llm"]["model"] == "exception-model"
            raise ValueError("Test exception")

        # Set this as default for the decorator to use
        ContainerFactory._default_container = container

        try:
            with pytest.raises(ValueError, match="Test exception"):
                failing_function()

            # Configuration should still be restored
            final_config = container.config()
            assert final_config["llm"]["model"] == original_model

        finally:
            ContainerFactory.reset_default_container()


class TestContainerIntegration:
    """Integration tests for complete container functionality."""

    def test_container_lifecycle(self):
        """Test complete container lifecycle from creation to cleanup."""
        # Create container with file configuration
        config_content = """
llm:
  api_key: "sk-integration-key"
  model: "gpt-4"
  temperature: 0.2

tools:
  max_workers: 8

memory:
  max_messages: 1000
  enable_compression: true

logging:
  level: "INFO"

monitoring:
  enabled: true
  interval: 60
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            # Create container
            container = ContainerFactory.create_container(config_file=config_path)

            # Verify configuration
            config_dict = container.config()
            assert config_dict["llm"]["api_key"] == "sk-integration-key"
            assert config_dict["llm"]["model"] == "gpt-4"

            # Configure logging
            container.configure_logging()

            # Perform health check
            health = container.health_check()
            assert health["container"] in ["healthy", "degraded"]

            # Test configuration override
            with ConfigContext(container, {"llm": {"temperature": 0.8}}):
                temp_config = container.config()
                assert temp_config["llm"]["temperature"] == 0.8

            # Verify configuration restored
            final_config = container.config()
            assert final_config["llm"]["temperature"] == 0.2

        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
