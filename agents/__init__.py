"""AetherFlow Agent System

This module contains the agent infrastructure for LLM-powered reactive agents.
Provides dependency injection, configuration management, and component lifecycle
management for ReAct agents built on AetherFlow.

Main Components:
- AgentContainer: Dependency injection container for agent services
- AgentConfig: Type-safe configuration models
- ContainerFactory: Container creation and management utilities
- ConfigLoader: Multi-source configuration loading system
"""

# Core configuration system
# Main ReActAgent interface
from .agent import ReActAgent, ReActAgentBuilder, create_agent, create_agent_builder
from .config import (
    AgentConfig,
    ConfigLoader,
    ConfigSource,
    ConfigurationError,
    DictConfigSource,
    EnvConfigSource,
    FileConfigSource,
    LLMConfig,
    LoggingConfig,
    MemoryConfig,
    MonitoringConfig,
    PromptsConfig,
    ReActConfig,
    RetryConfig,
    StorageConfig,
    ToolsConfig,
    with_config,
)

# Dependency injection container system
from .container import AgentContainer, ConfigContext, ContainerFactory

# Exceptions
from .exceptions import (
    AgentConfigurationError,
    AgentError,
    AgentExecutionError,
    AgentToolError,
    SessionError,
)

# Data models
from .models import AgentResponse, AgentStreamChunk, ReActStep, Session, ToolInfo


# Main factory function for convenience
def create_agent_container(config_file=None, **config_overrides) -> AgentContainer:
    """
    Convenience function to create a fully configured AgentContainer.

    Args:
        config_file: Optional path to configuration file
        **config_overrides: Configuration overrides

    Returns:
        Configured AgentContainer instance

    Example:
        # Basic usage
        container = create_agent_container()

        # With configuration file
        container = create_agent_container("config.yaml")

        # With overrides
        container = create_agent_container(
            llm_model="gpt-4",
            llm_temperature=0.2
        )
    """
    return ContainerFactory.create_container(
        config_file=config_file, **config_overrides
    )


# Version info
__version__ = "0.1.0"
__all__ = [
    # Main ReActAgent interface
    "ReActAgent",
    "ReActAgentBuilder",
    "create_agent",
    "create_agent_builder",
    # Data models
    "AgentResponse",
    "AgentStreamChunk",
    "ReActStep",
    "Session",
    "ToolInfo",
    # Exceptions
    "AgentError",
    "AgentConfigurationError",
    "AgentExecutionError",
    "AgentToolError",
    "SessionError",
    # Configuration classes
    "AgentConfig",
    "LLMConfig",
    "ToolsConfig",
    "MemoryConfig",
    "StorageConfig",
    "ReActConfig",
    "PromptsConfig",
    "LoggingConfig",
    "MonitoringConfig",
    "RetryConfig",
    # Configuration sources and loaders
    "ConfigLoader",
    "ConfigSource",
    "EnvConfigSource",
    "FileConfigSource",
    "DictConfigSource",
    # Container system
    "AgentContainer",
    "ContainerFactory",
    "ConfigContext",
    # Utilities
    "ConfigurationError",
    "with_config",
    "create_agent_container",
]
