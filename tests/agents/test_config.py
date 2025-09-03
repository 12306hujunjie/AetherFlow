"""
Tests for Agent Configuration System

Tests configuration loading, validation, environment variable parsing,
file loading, and configuration merging functionality.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from agents.config import (
    AgentConfig,
    ConfigLoader,
    ConfigurationError,
    DictConfigSource,
    EnvConfigSource,
    FileConfigSource,
    LLMConfig,
    MemoryConfig,
    RetryConfig,
    _deep_merge_dict,
    _parse_env_value,
)


class TestConfigModels:
    """Test configuration model validation and defaults."""

    def test_agent_config_defaults(self):
        """Test that AgentConfig creates valid defaults."""
        # Set required environment variable
        os.environ["OPENAI_API_KEY"] = "sk-test-key"

        try:
            config = AgentConfig()

            # Verify all subsystem configs are created
            assert config.llm is not None
            assert config.tools is not None
            assert config.memory is not None
            assert config.storage is not None
            assert config.react is not None
            assert config.prompts is not None
            assert config.logging is not None
            assert config.monitoring is not None

            # Verify default values
            assert config.llm.model == "gpt-4"
            assert config.llm.temperature == 0.1
            assert config.tools.max_workers == 4
            assert config.memory.max_messages == 1000
            assert config.react.max_steps == 10

        finally:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_llm_config_validation(self):
        """Test LLM configuration validation."""
        # Valid API key
        config = LLMConfig(api_key="sk-valid-key")
        assert config.api_key == "sk-valid-key"

        # Invalid API key format
        with pytest.raises(ValueError, match="Invalid OpenAI API key format"):
            LLMConfig(api_key="invalid-key")

        # Empty API key
        with pytest.raises(ValueError, match="API key is required"):
            LLMConfig(api_key="")

    def test_memory_config_validation(self):
        """Test memory configuration validation."""
        # Valid configuration
        config = MemoryConfig(max_tokens=4000, reserve_tokens=1000)
        assert config.max_tokens == 4000
        assert config.reserve_tokens == 1000

        # Invalid: reserve_tokens >= max_tokens
        with pytest.raises(
            ValueError, match="reserve_tokens must be less than max_tokens"
        ):
            MemoryConfig(max_tokens=1000, reserve_tokens=1000)

    def test_retry_config_validation(self):
        """Test retry configuration validation."""
        # Valid configuration
        config = RetryConfig(max_retries=5, initial_delay=0.5, max_delay=30.0)
        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0

        # Invalid: max_delay < initial_delay
        with pytest.raises(
            ValueError, match="max_delay must be greater than or equal to initial_delay"
        ):
            RetryConfig(initial_delay=10.0, max_delay=5.0)


class TestConfigSources:
    """Test different configuration sources."""

    def test_env_config_source(self):
        """Test environment variable configuration source."""
        # Set up environment variables
        env_vars = {
            "AETHERFLOW_AGENT_LLM_MODEL": "gpt-3.5-turbo",
            "AETHERFLOW_AGENT_LLM_TEMPERATURE": "0.5",
            "AETHERFLOW_AGENT_TOOLS_MAX_WORKERS": "8",
            "AETHERFLOW_AGENT_MEMORY_MAX_MESSAGES": "500",
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        try:
            source = EnvConfigSource("AETHERFLOW_AGENT_")
            config_data = source.load()

            # Verify nested structure is created correctly
            assert config_data["llm"]["model"] == "gpt-3.5-turbo"
            assert config_data["llm"]["temperature"] == 0.5  # Parsed as float
            assert config_data["tools"]["max_workers"] == 8  # Parsed as int
            assert config_data["memory"]["max_messages"] == 500

        finally:
            for key in env_vars:
                os.environ.pop(key, None)

    def test_file_config_source_yaml(self):
        """Test YAML file configuration source."""
        yaml_config = """
llm:
  model: "gpt-4"
  temperature: 0.2
  max_tokens: 1500

tools:
  max_workers: 6
  timeout: 45

memory:
  max_messages: 800
  enable_compression: false
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_config)
            temp_path = f.name

        try:
            source = FileConfigSource(temp_path)
            config_data = source.load()

            assert config_data["llm"]["model"] == "gpt-4"
            assert config_data["llm"]["temperature"] == 0.2
            assert config_data["tools"]["max_workers"] == 6
            assert config_data["memory"]["max_messages"] == 800
            assert config_data["memory"]["enable_compression"] is False

        finally:
            Path(temp_path).unlink()

    def test_file_config_source_json(self):
        """Test JSON file configuration source."""
        json_config = {
            "llm": {"model": "gpt-3.5-turbo", "temperature": 0.3},
            "react": {"max_steps": 15, "enable_streaming": True},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_config, f)
            temp_path = f.name

        try:
            source = FileConfigSource(temp_path)
            config_data = source.load()

            assert config_data["llm"]["model"] == "gpt-3.5-turbo"
            assert config_data["llm"]["temperature"] == 0.3
            assert config_data["react"]["max_steps"] == 15
            assert config_data["react"]["enable_streaming"] is True

        finally:
            Path(temp_path).unlink()

    def test_file_config_source_missing_optional(self):
        """Test handling of missing optional configuration files."""
        source = FileConfigSource("/nonexistent/path/config.yaml", required=False)
        config_data = source.load()
        assert config_data == {}

    def test_file_config_source_missing_required(self):
        """Test handling of missing required configuration files."""
        source = FileConfigSource("/nonexistent/path/config.yaml", required=True)
        with pytest.raises(FileNotFoundError):
            source.load()

    def test_dict_config_source(self):
        """Test dictionary configuration source."""
        config_dict = {
            "llm": {"model": "custom-model", "temperature": 0.7},
            "tools": {"max_workers": 12},
        }

        source = DictConfigSource(config_dict)
        config_data = source.load()

        assert config_data == config_dict
        assert config_data["llm"]["model"] == "custom-model"
        assert config_data["tools"]["max_workers"] == 12


class TestConfigLoader:
    """Test configuration loader with multiple sources."""

    def test_config_priority_order(self):
        """Test that configuration sources are merged in correct priority order."""
        # Create temporary YAML file (medium priority)
        yaml_config = """
llm:
  model: "file-model"
  timeout: 60

tools:
  max_workers: 4
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_config)
            yaml_path = f.name

        try:
            # Set environment variables (highest priority)
            os.environ["AETHERFLOW_AGENT_LLM_MODEL"] = "env-model"
            os.environ["AETHERFLOW_AGENT_TOOLS_MAX_WORKERS"] = "8"
            os.environ["OPENAI_API_KEY"] = "sk-test-key"

            # Create loader with multiple sources
            loader = ConfigLoader()
            loader.add_dict_source({"llm": {"temperature": 0.5}})  # Lowest priority
            loader.add_file_source(yaml_path)  # Medium priority
            loader.add_env_source()  # Highest priority

            config = loader.load()

            # Environment variables should override file and dict values
            assert config.llm.model == "env-model"  # From env
            assert config.tools.max_workers == 8  # From env

            # File values should override dict values where env not present
            assert config.llm.timeout == 60  # From file

            # Dict values should be used where neither env nor file specify
            assert config.llm.temperature == 0.5  # From dict (lowest priority)

        finally:
            os.environ.pop("AETHERFLOW_AGENT_LLM_MODEL", None)
            os.environ.pop("AETHERFLOW_AGENT_TOOLS_MAX_WORKERS", None)
            os.environ.pop("OPENAI_API_KEY", None)
            Path(yaml_path).unlink()

    def test_config_loader_validation_failure(self):
        """Test configuration loader handling of validation failures."""
        # Create invalid configuration (missing API key)
        loader = ConfigLoader()
        loader.add_dict_source({"llm": {"api_key": ""}})  # Invalid empty API key

        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            loader.load()


class TestUtilityFunctions:
    """Test utility functions for configuration parsing."""

    def test_parse_env_value(self):
        """Test environment variable value parsing."""
        # String values
        assert _parse_env_value("hello") == "hello"
        assert _parse_env_value("gpt-4") == "gpt-4"

        # Boolean values
        assert _parse_env_value("true") is True
        assert _parse_env_value("false") is False
        assert _parse_env_value("True") is True
        assert _parse_env_value("FALSE") is False

        # Integer values
        assert _parse_env_value("42") == 42
        assert _parse_env_value("0") == 0
        assert _parse_env_value("-10") == -10

        # Float values
        assert _parse_env_value("3.14") == 3.14
        assert _parse_env_value("0.5") == 0.5

        # JSON values
        assert _parse_env_value('["item1", "item2"]') == ["item1", "item2"]
        assert _parse_env_value('{"key": "value"}') == {"key": "value"}

    def test_deep_merge_dict(self):
        """Test deep dictionary merging."""
        base = {"a": 1, "b": {"c": 2, "d": 3}, "e": [1, 2, 3]}

        override = {
            "a": 10,  # Should replace
            "b": {"d": 30, "f": 4},  # Should merge nested dict
            "g": 5,  # Should add new key
        }

        result = _deep_merge_dict(base, override)

        expected = {"a": 10, "b": {"c": 2, "d": 30, "f": 4}, "e": [1, 2, 3], "g": 5}

        assert result == expected


class TestAgentConfigIntegration:
    """Integration tests for complete AgentConfig functionality."""

    def test_config_from_env_class_method(self):
        """Test AgentConfig.from_env() class method."""
        # Set up environment
        env_vars = {
            "AETHERFLOW_AGENT_LLM_MODEL": "gpt-3.5-turbo",
            "AETHERFLOW_AGENT_LLM_TEMPERATURE": "0.3",
            "AETHERFLOW_AGENT_REACT_MAX_STEPS": "20",
            "OPENAI_API_KEY": "sk-test-key",
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        try:
            config = AgentConfig.from_env()

            assert config.llm.model == "gpt-3.5-turbo"
            assert config.llm.temperature == 0.3
            assert config.react.max_steps == 20
            assert config.llm.api_key == "sk-test-key"

        finally:
            for key in env_vars:
                os.environ.pop(key, None)

    def test_config_dict_for_container(self):
        """Test converting AgentConfig to container-compatible format."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"

        try:
            config = AgentConfig()
            container_dict = config.dict_for_container()

            # Verify structure is appropriate for dependency injection
            assert isinstance(container_dict, dict)
            assert "llm" in container_dict
            assert "tools" in container_dict
            assert "memory" in container_dict

            # Verify nested structure
            assert isinstance(container_dict["llm"], dict)
            assert "api_key" in container_dict["llm"]
            assert "model" in container_dict["llm"]

        finally:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_config_validate_complete(self):
        """Test complete configuration validation."""
        os.environ["OPENAI_API_KEY"] = "sk-test-key"

        try:
            config = AgentConfig()

            # Should not raise any exceptions
            config.validate_complete()

            # Test with invalid configuration
            config.llm.api_key = ""
            with pytest.raises(ConfigurationError, match="LLM API key is required"):
                config.validate_complete()

        finally:
            os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
