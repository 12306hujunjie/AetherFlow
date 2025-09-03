"""
Agent Configuration System

This module provides comprehensive configuration management for AetherFlow agents,
supporting multiple configuration sources (environment variables, files, programmatic),
type-safe validation, and hierarchical configuration merging.

Features:
- Pydantic-based configuration models with validation
- Environment variable support with automatic type conversion
- YAML/JSON configuration file support
- Configuration priority system (env vars > config files > defaults)
- Thread-safe configuration loading and merging
- Configuration hot-reloading and context management
"""

import json
import os
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator

# Import required for configuration validation and error handling
try:
    import yaml
except ImportError:
    yaml = None


class ConfigurationError(Exception):
    """Configuration-related errors."""

    pass


class RetryConfig(BaseModel):
    """Retry configuration for operations."""

    max_retries: int = Field(3, ge=0, le=10, description="Maximum number of retries")
    initial_delay: float = Field(
        1.0, ge=0.1, le=60.0, description="Initial delay in seconds"
    )
    max_delay: float = Field(
        60.0, ge=1.0, le=3600.0, description="Maximum delay in seconds"
    )
    exponential_base: float = Field(
        2.0, ge=1.1, le=10.0, description="Exponential backoff base"
    )
    jitter: bool = Field(True, description="Add random jitter to delays")

    @validator("max_delay")
    def validate_max_delay(cls, v, values):
        if "initial_delay" in values and v < values["initial_delay"]:
            raise ValueError("max_delay must be greater than or equal to initial_delay")
        return v


class LLMConfig(BaseModel):
    """Large Language Model configuration."""

    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key",
    )
    model: str = Field("gpt-4", description="Model name")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(
        2000, ge=1, le=8192, description="Maximum tokens per response"
    )
    timeout: int = Field(60, ge=5, le=300, description="Request timeout in seconds")
    base_url: str | None = Field(None, description="Base URL for API requests")
    organization: str | None = Field(None, description="OpenAI organization ID")

    @validator("api_key")
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("API key is required")
        if not v.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
        return v

    class Config:
        env_prefix = "AETHERFLOW_LLM_"


class ToolsConfig(BaseModel):
    """Tools system configuration."""

    auto_discover: bool = Field(True, description="Automatically discover tools")
    packages: list[str] = Field(
        default_factory=list, description="Tool packages to load"
    )
    max_workers: int = Field(
        4, ge=1, le=32, description="Maximum concurrent tool workers"
    )
    timeout: int = Field(
        30, ge=1, le=300, description="Tool execution timeout in seconds"
    )
    retry: RetryConfig = Field(
        default_factory=RetryConfig, description="Retry configuration"
    )

    class Config:
        env_prefix = "AETHERFLOW_TOOLS_"


class MemoryConfig(BaseModel):
    """Memory management configuration."""

    max_messages: int = Field(
        1000, ge=10, le=10000, description="Maximum messages to retain"
    )
    max_tokens: int = Field(
        8000, ge=1000, le=32768, description="Maximum tokens in context"
    )
    reserve_tokens: int = Field(
        1000, ge=100, le=4096, description="Reserved tokens for completion"
    )
    enable_compression: bool = Field(True, description="Enable memory compression")
    storage_path: Path = Field(
        Path(".aetherflow/memory"), description="Path for memory storage"
    )
    trimming_strategy: str = Field(
        "recent_first",
        pattern="^(recent_first|oldest_first|importance_based)$",
        description="Memory trimming strategy",
    )

    @validator("reserve_tokens")
    def validate_reserve_tokens(cls, v, values):
        if "max_tokens" in values and v >= values["max_tokens"]:
            raise ValueError("reserve_tokens must be less than max_tokens")
        return v

    class Config:
        env_prefix = "AETHERFLOW_MEMORY_"
        json_encoders = {Path: str}


class StorageConfig(BaseModel):
    """Storage configuration."""

    path: Path = Field(Path(".aetherflow/sessions"), description="Session storage path")
    auto_save: bool = Field(True, description="Auto-save session data")
    save_interval: int = Field(
        30, ge=1, le=3600, description="Auto-save interval in seconds"
    )
    compression: bool = Field(True, description="Enable storage compression")
    cleanup_days: int = Field(30, ge=1, le=365, description="Days to keep old sessions")

    class Config:
        env_prefix = "AETHERFLOW_STORAGE_"
        json_encoders = {Path: str}


class ReActConfig(BaseModel):
    """ReAct engine configuration."""

    max_steps: int = Field(10, ge=1, le=50, description="Maximum reasoning steps")
    step_timeout: int = Field(30, ge=5, le=300, description="Step timeout in seconds")
    enable_streaming: bool = Field(False, description="Enable streaming responses")
    early_stopping: bool = Field(True, description="Enable early stopping on success")

    class Config:
        env_prefix = "AETHERFLOW_REACT_"


class PromptsConfig(BaseModel):
    """Prompt templates configuration."""

    template_path: Path | None = Field(
        None, description="Path to custom prompt templates"
    )
    language: str = Field("en", description="Default prompt language")
    style: str = Field(
        "professional",
        pattern="^(professional|casual|technical|creative)$",
        description="Prompt style",
    )

    class Config:
        env_prefix = "AETHERFLOW_PROMPTS_"
        json_encoders = {Path: str}


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(
        "INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", description="Log level"
    )
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    handlers: list[str] = Field(
        default_factory=lambda: ["console"], description="Log handlers"
    )
    file_path: Path | None = Field(
        None, description="Log file path (if file handler enabled)"
    )

    @validator("handlers")
    def validate_handlers(cls, v):
        valid_handlers = {"console", "file", "rotating"}
        for handler in v:
            if handler not in valid_handlers:
                raise ValueError(
                    f"Invalid handler: {handler}. Must be one of {valid_handlers}"
                )
        return v

    class Config:
        env_prefix = "AETHERFLOW_LOGGING_"
        json_encoders = {Path: str}


class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration."""

    enabled: bool = Field(True, description="Enable monitoring")
    interval: int = Field(
        60, ge=1, le=3600, description="Metrics collection interval in seconds"
    )
    backend: str = Field(
        "memory", pattern="^(memory|file|http)$", description="Metrics storage backend"
    )
    endpoint: str | None = Field(
        None, description="HTTP endpoint for metrics (if using http backend)"
    )

    class Config:
        env_prefix = "AETHERFLOW_MONITORING_"


class AgentConfig(BaseModel):
    """Complete agent configuration model with all subsystems."""

    # Core subsystem configurations
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    tools: ToolsConfig = Field(
        default_factory=ToolsConfig, description="Tools configuration"
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig, description="Memory configuration"
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig, description="Storage configuration"
    )
    react: ReActConfig = Field(
        default_factory=ReActConfig, description="ReAct engine configuration"
    )
    prompts: PromptsConfig = Field(
        default_factory=PromptsConfig, description="Prompts configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig, description="Monitoring configuration"
    )

    class Config:
        env_prefix = "AETHERFLOW_AGENT_"
        case_sensitive = False
        json_encoders = {Path: str}
        validate_assignment = True
        extra = "forbid"  # Prevent unknown fields

    def dict_for_container(self) -> dict[str, Any]:
        """Convert to dictionary format suitable for dependency injection container."""
        return self.dict()

    def validate_complete(self) -> None:
        """Perform additional cross-field validation."""
        # Check that API key is provided when LLM is configured
        if not self.llm.api_key:
            raise ConfigurationError("LLM API key is required")

        # Ensure storage paths exist or can be created
        for path_field in [self.memory.storage_path, self.storage.path]:
            try:
                path_field.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise ConfigurationError(
                    f"Cannot create storage path {path_field}: {e}"
                )

    @classmethod
    def from_env(cls, prefix: str = "AETHERFLOW_AGENT_") -> "AgentConfig":
        """Create configuration from environment variables."""
        env_source = EnvConfigSource(prefix)
        env_data = env_source.load()
        return cls(**env_data)


# Configuration source system
class ConfigSource(ABC):
    """Abstract base class for configuration sources."""

    def __init__(self, required: bool = False):
        self.required = required

    @abstractmethod
    def load(self) -> dict[str, Any]:
        """Load configuration from this source."""
        pass


class EnvConfigSource(ConfigSource):
    """Environment variable configuration source."""

    def __init__(self, prefix: str = "AETHERFLOW_AGENT_", required: bool = False):
        super().__init__(required)
        self.prefix = prefix

    def load(self) -> dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Convert AETHERFLOW_AGENT_LLM_MODEL -> llm.model
                config_key = key[len(self.prefix) :].lower()
                nested_keys = config_key.split("_", 1)  # Only split on first underscore

                if len(nested_keys) == 2:
                    # Two parts: section and key (e.g., "llm" and "model")
                    section, field_key = nested_keys
                    if section not in config:
                        config[section] = {}
                    config[section][field_key] = _parse_env_value(value)
                else:
                    # Single part: top-level key
                    config[nested_keys[0]] = _parse_env_value(value)

        return config


class FileConfigSource(ConfigSource):
    """File-based configuration source supporting YAML and JSON."""

    def __init__(self, file_path: str | Path, required: bool = False):
        super().__init__(required)
        self.file_path = Path(file_path)

    def load(self) -> dict[str, Any]:
        """Load configuration from file."""
        if not self.file_path.exists():
            if self.required:
                raise FileNotFoundError(
                    f"Required configuration file not found: {self.file_path}"
                )
            return {}

        try:
            with open(self.file_path, encoding="utf-8") as f:
                if self.file_path.suffix.lower() in (".yaml", ".yml"):
                    if yaml is None:
                        raise ConfigurationError(
                            "PyYAML is required for YAML configuration files"
                        )
                    return yaml.safe_load(f) or {}
                elif self.file_path.suffix.lower() == ".json":
                    return json.load(f) or {}
                else:
                    raise ValueError(
                        f"Unsupported configuration file format: {self.file_path.suffix}"
                    )
        except Exception as e:
            if self.required:
                raise ConfigurationError(
                    f"Failed to load configuration file {self.file_path}: {e}"
                )
            return {}


class DictConfigSource(ConfigSource):
    """Dictionary-based configuration source for programmatic configuration."""

    def __init__(self, config_dict: dict[str, Any], required: bool = False):
        super().__init__(required)
        self.config_dict = config_dict or {}

    def load(self) -> dict[str, Any]:
        """Load configuration from dictionary."""
        return self.config_dict.copy()


class ConfigLoader:
    """Multi-source configuration loader with priority support."""

    def __init__(self):
        self.config_sources: list[ConfigSource] = []

    def add_env_source(self, prefix: str = "AETHERFLOW_AGENT_") -> "ConfigLoader":
        """Add environment variable configuration source (highest priority)."""
        self.config_sources.append(EnvConfigSource(prefix))
        return self

    def add_file_source(
        self, file_path: str | Path, required: bool = False
    ) -> "ConfigLoader":
        """Add file configuration source."""
        self.config_sources.append(FileConfigSource(file_path, required))
        return self

    def add_dict_source(self, config_dict: dict[str, Any]) -> "ConfigLoader":
        """Add dictionary configuration source (lowest priority)."""
        self.config_sources.append(DictConfigSource(config_dict))
        return self

    def load(self) -> AgentConfig:
        """Load and merge configuration from all sources."""
        merged_config = {}

        # Process sources in order (lowest priority first, highest priority last)
        for source in self.config_sources:
            try:
                source_config = source.load()
                merged_config = _deep_merge_dict(merged_config, source_config)
            except Exception as e:
                if source.required:
                    raise ConfigurationError(
                        f"Failed to load required configuration source: {e}"
                    )
                # Log warning for optional sources but continue
                import logging

                logging.getLogger("aetherflow.agent.config").warning(
                    f"Failed to load optional configuration source, skipping: {e}"
                )

        try:
            config = AgentConfig(**merged_config)
            config.validate_complete()
            return config
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")


# Utility functions
def _parse_env_value(value: str) -> Any:
    """Parse environment variable value to appropriate Python type."""
    # Try JSON parsing first
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Try boolean parsing
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Try numeric parsing
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            # Return as string
            return value


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value

    return result


def with_config(**config_overrides):
    """Decorator for temporarily overriding configuration."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Import here to avoid circular imports
            from .container import ConfigContext, ContainerFactory

            container = ContainerFactory.get_default_container()
            with ConfigContext(container, config_overrides):
                return func(*args, **kwargs)

        return wrapper

    return decorator
