"""
Agent Dependency Injection Container

This module provides the AgentContainer which extends AetherFlow's BaseFlowContext
to provide dependency injection for all ReAct agent components. It manages the
lifecycle of LLM clients, tool registries, memory managers, and other agent services.

Features:
- Extends BaseFlowContext for thread-safe dependency injection
- Component lifecycle management with lazy initialization
- Configuration-driven service creation
- Thread-local isolation for concurrent agent operations
- Graceful resource cleanup and error handling
"""

import logging
from functools import wraps
from pathlib import Path
from typing import Any

from dependency_injector import providers

# Import AetherFlow's base container
from aetherflow import BaseFlowContext

# Import configuration system
from .config import AgentConfig, ConfigLoader, ConfigurationError

# Import memory management components (placeholders for now)
# from .memory import ConversationMemory, ContextManager, StateManager


logger = logging.getLogger("aetherflow.agent.container")


class AgentContainer(BaseFlowContext):
    """
    ReAct agent dependency injection container extending AetherFlow's BaseFlowContext.

    This container provides centralized dependency injection for all agent components,
    including LLM clients, tool systems, memory managers, and monitoring services.
    It inherits thread-safe state management from BaseFlowContext while adding
    agent-specific service providers.
    """

    # Configuration provider - drives all other service configuration
    config = providers.Configuration()

    # === Core LLM Services ===

    # Placeholder providers for LLM components (to be implemented in other tasks)
    llm_client = providers.Singleton(
        dict,  # Placeholder - will be replaced with actual LLMClient
        api_key=config.llm.api_key,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        timeout=config.llm.timeout,
        base_url=config.llm.base_url,
        organization=config.llm.organization,
    )

    prompt_template_manager = providers.Singleton(
        dict,  # Placeholder - will be replaced with actual PromptTemplateManager
        template_path=config.prompts.template_path,
        language=config.prompts.language,
        style=config.prompts.style,
    )

    # === Tool System Services ===

    tool_registry = providers.Singleton(
        dict,  # Placeholder - will be replaced with actual ToolRegistry
        auto_discover=config.tools.auto_discover,
        packages=config.tools.packages,
    )

    tool_executor = providers.Singleton(
        dict,  # Placeholder - will be replaced with actual ToolExecutor
        max_workers=config.tools.max_workers,
        timeout=config.tools.timeout,
        retry_config=config.tools.retry,
    )

    # === Memory Management Services ===

    conversation_memory = providers.Singleton(
        dict,  # Placeholder - will be replaced with actual ConversationMemory
        max_messages=config.memory.max_messages,
        enable_compression=config.memory.enable_compression,
        storage_path=config.memory.storage_path,
    )

    context_manager = providers.Singleton(
        dict,  # Placeholder - will be replaced with actual ContextManager
        max_tokens=config.memory.max_tokens,
        reserve_tokens=config.memory.reserve_tokens,
        trimming_strategy=config.memory.trimming_strategy,
    )

    state_manager = providers.Singleton(
        dict,  # Placeholder - will be replaced with actual StateManager
        storage_path=config.storage.path,
        auto_save=config.storage.auto_save,
        save_interval=config.storage.save_interval,
        compression=config.storage.compression,
    )

    # === ReAct Engine Services ===

    react_engine = providers.Singleton(
        dict,  # Placeholder - will be replaced with actual ReActEngine
        max_steps=config.react.max_steps,
        step_timeout=config.react.step_timeout,
        enable_streaming=config.react.enable_streaming,
        early_stopping=config.react.early_stopping,
    )

    # === Monitoring and Logging Services ===

    agent_logger = providers.Singleton(
        logging.getLogger,  # Using standard library logger
        name="aetherflow.agent",
    )

    metrics_collector = providers.Singleton(
        dict,  # Placeholder - will be replaced with actual MetricsCollector
        enabled=config.monitoring.enabled,
        interval=config.monitoring.interval,
        backend=config.monitoring.backend,
        endpoint=config.monitoring.endpoint,
    )

    @classmethod
    def configure_logging_for_instance(cls, container) -> None:
        """Configure logging for a container instance."""
        try:
            log_config = container.config()["logging"]
            logger = container.agent_logger()

            # Set log level
            level = getattr(logging, log_config.get("level", "INFO").upper())
            logger.setLevel(level)

            # Clear existing handlers
            logger.handlers.clear()

            # Add console handler if requested
            handlers = log_config.get("handlers", ["console"])
            if "console" in handlers:
                console_handler = logging.StreamHandler()
                formatter = logging.Formatter(log_config.get("format"))
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

            # Add file handler if requested
            if "file" in handlers:
                file_path = log_config.get("file_path")
                if file_path:
                    file_handler = logging.FileHandler(file_path)
                    formatter = logging.Formatter(log_config.get("format"))
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)

        except Exception as e:
            # Fallback to basic logging configuration
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            logger.error(f"Failed to configure logging: {e}")

    @classmethod
    def validate_configuration_for_instance(cls, container) -> None:
        """Validate configuration for a container instance."""
        try:
            config_dict = container.config()
            # Validate that required configuration is present
            if not config_dict:
                raise ConfigurationError("Container configuration is empty")

            # Validate LLM configuration
            llm_config = config_dict.get("llm", {})
            if not llm_config.get("api_key"):
                raise ConfigurationError("LLM API key is required")

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    @classmethod
    def health_check_for_instance(cls, container) -> dict[str, Any]:
        """Perform health check on a container instance."""
        health_status = {"container": "healthy", "services": {}, "errors": []}

        try:
            # Check configuration
            cls.validate_configuration_for_instance(container)
            health_status["services"]["config"] = "healthy"
        except Exception as e:
            health_status["services"]["config"] = "unhealthy"
            health_status["errors"].append(f"Config error: {e}")

        try:
            # Check logging
            logger_instance = container.agent_logger()
            if logger_instance:
                health_status["services"]["logging"] = "healthy"
            else:
                health_status["services"]["logging"] = "unhealthy"
        except Exception as e:
            health_status["services"]["logging"] = "unhealthy"
            health_status["errors"].append(f"Logging error: {e}")

        # Set overall health status
        if health_status["errors"]:
            health_status["container"] = (
                "degraded" if len(health_status["errors"]) < 2 else "unhealthy"
            )

        return health_status


class ContainerFactory:
    """
    Factory for creating and managing AgentContainer instances.

    Provides convenient methods for creating containers with different configurations,
    managing default container instances, and handling container lifecycle.
    """

    _default_container: AgentContainer | None = None

    @classmethod
    def create_container(
        cls,
        config: AgentConfig | None = None,
        config_file: str | Path | None = None,
        **kwargs,
    ) -> AgentContainer:
        """
        Create a new AgentContainer instance with the specified configuration.

        Args:
            config: Pre-built AgentConfig instance
            config_file: Path to configuration file
            **kwargs: Additional configuration overrides

        Returns:
            Configured AgentContainer instance

        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        try:
            if config is None:
                config = cls._load_config(config_file, **kwargs)

            # Create and configure container
            container = AgentContainer()
            container.config.from_dict(config.dict_for_container())

            # Configure logging early
            AgentContainer.configure_logging_for_instance(container)

            # Validate configuration
            AgentContainer.validate_configuration_for_instance(container)

            # Wire the container for dependency injection
            container.wire(modules=[__name__])

            # Add instance methods for easier testing
            container.configure_logging = (
                lambda: AgentContainer.configure_logging_for_instance(container)
            )
            container.validate_configuration = (
                lambda: AgentContainer.validate_configuration_for_instance(container)
            )
            container.health_check = lambda: AgentContainer.health_check_for_instance(
                container
            )

            logger.info("Successfully created AgentContainer")
            return container

        except Exception as e:
            logger.error(f"Failed to create AgentContainer: {e}")
            raise ConfigurationError(f"Container creation failed: {e}") from e

    @classmethod
    def get_default_container(cls) -> AgentContainer:
        """
        Get or create the default AgentContainer instance.

        Returns:
            The default container instance
        """
        if cls._default_container is None:
            cls._default_container = cls.create_container()
        return cls._default_container

    @classmethod
    def reset_default_container(cls) -> None:
        """Reset the default container instance, forcing recreation on next access."""
        if cls._default_container is not None:
            try:
                # Cleanup existing container if needed
                cls._default_container = None
                logger.info("Reset default container")
            except Exception as e:
                logger.warning(f"Error during container reset: {e}")
        cls._default_container = None

    @classmethod
    def create_test_container(cls, **config_overrides) -> AgentContainer:
        """
        Create a container configured for testing with sensible test defaults.

        Args:
            **config_overrides: Configuration overrides for testing

        Returns:
            Test-configured AgentContainer
        """
        test_defaults = {
            "llm": {
                "api_key": "sk-test-key-for-testing",
                "model": "gpt-3.5-turbo",
                "temperature": 0.0,
                "max_tokens": 100,
                "timeout": 10,
            },
            "memory": {
                "max_messages": 10,
                "max_tokens": 1000,
                "storage_path": "/tmp/aetherflow_test_memory",  # nosec B108
            },
            "storage": {"path": "/tmp/aetherflow_test_storage", "auto_save": False},  # nosec B108
            "logging": {"level": "DEBUG"},
            "monitoring": {"enabled": False},
        }

        # Merge test defaults with provided overrides (convert flat config first)
        nested_overrides = cls._convert_flat_config(config_overrides)
        merged_config = cls._deep_merge(test_defaults, nested_overrides)

        return cls.create_container(config=AgentConfig(**merged_config))

    @classmethod
    def _load_config(
        cls, config_file: str | Path | None = None, **kwargs
    ) -> AgentConfig:
        """Load configuration from multiple sources with proper priority."""
        loader = ConfigLoader()

        # Add programmatic config (lowest priority - added first)
        if kwargs:
            # Convert flat kwargs to nested structure
            nested_kwargs = cls._convert_flat_config(kwargs)
            loader.add_dict_source(nested_kwargs)

        # Add configuration file if specified (medium priority)
        if config_file:
            loader.add_file_source(config_file, required=True)
        else:
            # Try to find default configuration files
            default_config_files = [
                ".aetherflow/config.yaml",
                ".aetherflow/config.yml",
                ".aetherflow/config.json",
                "aetherflow.config.yaml",
                "aetherflow.config.yml",
                "aetherflow.config.json",
            ]

            for file_path in default_config_files:
                if Path(file_path).exists():
                    loader.add_file_source(file_path)
                    break

        # Add environment variables (highest priority - added last)
        loader.add_env_source()

        return loader.load()

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ContainerFactory._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _convert_flat_config(flat_config: dict[str, Any]) -> dict[str, Any]:
        """Convert flat configuration keys (llm_model) to nested structure (llm.model)."""
        nested = {}
        for key, value in flat_config.items():
            if "_" in key:
                # Split on first underscore only
                parts = key.split("_", 1)
                if len(parts) == 2:
                    section, field = parts
                    if section not in nested:
                        nested[section] = {}
                    nested[section][field] = value
                else:
                    nested[key] = value
            else:
                nested[key] = value
        return nested


class ConfigContext:
    """
    Context manager for temporarily overriding container configuration.

    Allows for temporary configuration changes within a specific scope,
    automatically restoring the original configuration when exiting the context.
    """

    def __init__(self, container: AgentContainer, config_overrides: dict[str, Any]):
        self.container = container
        self.config_overrides = config_overrides
        self.original_config: dict[str, Any] | None = None

    def __enter__(self) -> AgentContainer:
        """Enter the context with configuration overrides applied."""
        try:
            # Save original configuration
            self.original_config = self.container.config()

            # Apply overrides
            merged_config = ContainerFactory._deep_merge(
                self.original_config, self.config_overrides
            )

            # Update container configuration
            self.container.config.from_dict(merged_config)

            logger.debug(
                f"Applied configuration overrides: {list(self.config_overrides.keys())}"
            )

            return self.container

        except Exception as e:
            logger.error(f"Failed to apply configuration overrides: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore original configuration."""
        if self.original_config is not None:
            try:
                self.container.config.from_dict(self.original_config)
                logger.debug("Restored original configuration")
            except Exception as e:
                logger.error(f"Failed to restore original configuration: {e}")
                # Don't raise here as it might mask the original exception


def with_config(**config_overrides):
    """
    Decorator for temporarily overriding configuration in function scope.

    Args:
        **config_overrides: Configuration keys and values to override

    Returns:
        Decorated function with temporary configuration

    Example:
        @with_config(llm_temperature=0.8, react_max_steps=20)
        def creative_agent_task():
            container = ContainerFactory.get_default_container()
            # Function executes with overridden configuration
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            container = ContainerFactory.get_default_container()
            # Convert flat config to nested
            nested_overrides = ContainerFactory._convert_flat_config(config_overrides)
            with ConfigContext(container, nested_overrides):
                return func(*args, **kwargs)

        return wrapper

    return decorator
