"""
Tool System Data Models

Defines the core data structures for the AetherFlow tool system, including
tool metadata, execution results, and type-safe interfaces for tool
registration and invocation.

All models use Pydantic for runtime validation and seamless integration
with the existing ReAct agent models.
"""

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class ToolCategory(str, Enum):
    """Tool category enumeration for organization and filtering"""

    GENERAL = "general"  # General purpose tools
    SEARCH = "search"  # Search and information retrieval
    COMPUTE = "compute"  # Mathematical and computational tools
    WEB = "web"  # Web-based tools and APIs
    IO = "io"  # Input/output and file operations
    DATA = "data"  # Data processing and transformation
    COMMUNICATION = "communication"  # External communication tools


class ParameterInfo(BaseModel):
    """Parameter information for tool validation"""

    name: str = Field(..., description="Parameter name")
    type_hint: str = Field(..., description="Type annotation as string")
    description: str = Field("", description="Parameter description")
    required: bool = Field(True, description="Whether parameter is required")
    default_value: Any = Field(None, description="Default value if not required")


class ToolMetadata(BaseModel):
    """
    Complete metadata for a registered tool.

    Contains all information needed for tool discovery, validation,
    and execution by ReAct agents.
    """

    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Tool functionality description")
    parameters: list[ParameterInfo] = Field(
        default_factory=list, description="Parameter definitions"
    )
    return_type: str = Field("Any", description="Return type as string")
    category: ToolCategory = Field(ToolCategory.GENERAL, description="Tool category")

    # Execution characteristics
    is_async: bool = Field(True, description="Whether tool supports async execution")
    timeout_seconds: float = Field(
        30.0, ge=0.1, le=300.0, description="Execution timeout"
    )

    # Capability flags
    supports_concurrent: bool = Field(
        True, description="Can run concurrently with other tools"
    )
    requires_context: bool = Field(
        False, description="Requires ReAct context for execution"
    )

    # Registration metadata
    module_path: str | None = Field(None, description="Module where tool is defined")
    registration_time: float = Field(
        default_factory=time.time, description="Registration timestamp"
    )

    @validator("name")
    def validate_name_format(cls, v: str) -> str:
        """Validate tool name follows naming conventions"""
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")

        name = v.strip().lower()

        # Check for valid identifier format
        if not name.replace("_", "").isalnum():
            raise ValueError(
                "Tool name must contain only letters, numbers, and underscores"
            )

        if name.startswith("_"):
            raise ValueError("Tool name cannot start with underscore")

        return name

    @validator("description")
    def validate_description_content(cls, v: str) -> str:
        """Ensure description is meaningful"""
        if not v or not v.strip():
            raise ValueError("Tool description cannot be empty")

        description = v.strip()
        if len(description) < 10:
            raise ValueError("Tool description must be at least 10 characters")

        return description


class ToolCall(BaseModel):
    """
    Tool invocation request.

    Represents a request to execute a specific tool with given parameters.
    Used for both synchronous tool calls and concurrent execution batches.
    """

    tool_name: str = Field(..., description="Name of tool to execute")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Tool execution parameters"
    )
    call_id: str | None = Field(None, description="Unique call identifier for tracking")

    # Execution configuration
    timeout_override: float | None = Field(
        None, ge=0.1, le=300.0, description="Override default timeout"
    )
    priority: int = Field(
        0, ge=0, le=10, description="Execution priority (0=lowest, 10=highest)"
    )

    def __post_init__(self):
        """Validate parameters against tool metadata after construction"""
        # This will be called by the ToolRegistry.validate_call method
        pass

    @validator("tool_name")
    def validate_tool_name(cls, v: str) -> str:
        """Ensure tool name is properly formatted"""
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip().lower()


class ToolExecutionStatus(str, Enum):
    """Tool execution status enumeration"""

    PENDING = "pending"  # Waiting to execute
    RUNNING = "running"  # Currently executing
    SUCCESS = "success"  # Completed successfully
    FAILED = "failed"  # Failed with error
    TIMEOUT = "timeout"  # Timed out during execution
    CANCELLED = "cancelled"  # Cancelled before completion


class ToolResult(BaseModel):
    """
    Standardized tool execution result.

    Provides consistent structure for all tool outputs, including
    success/failure status, execution metadata, and error handling.
    """

    # Execution identification
    tool_name: str = Field(..., description="Name of executed tool")
    call_id: str | None = Field(None, description="Call identifier if provided")

    # Execution status
    status: ToolExecutionStatus = Field(..., description="Execution status")
    success: bool = Field(..., description="Whether execution was successful")

    # Result data
    result: Any = Field(None, description="Tool execution result (if successful)")
    error_message: str | None = Field(None, description="Error message (if failed)")
    error_type: str | None = Field(None, description="Error type for debugging")

    # Execution metadata
    execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Execution time in milliseconds"
    )
    start_time: float = Field(
        default_factory=time.time, description="Execution start timestamp"
    )
    end_time: float | None = Field(None, description="Execution end timestamp")

    # Context and debugging
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional execution metadata"
    )
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")

    @validator("success")
    def validate_success_consistency(cls, v: bool, values: dict[str, Any]) -> bool:
        """Ensure success flag is consistent with status"""
        status = values.get("status")
        if status == ToolExecutionStatus.SUCCESS and not v:
            raise ValueError("Success must be True when status is SUCCESS")
        if status in [ToolExecutionStatus.FAILED, ToolExecutionStatus.TIMEOUT] and v:
            raise ValueError("Success must be False when status indicates failure")
        return v

    @validator("error_message")
    def validate_error_for_failed_status(
        cls, v: str | None, values: dict[str, Any]
    ) -> str | None:
        """Ensure error message is provided for failed executions"""
        success = values.get("success", True)
        status = values.get("status")

        if not success and status == ToolExecutionStatus.FAILED and not v:
            raise ValueError("Error message must be provided for failed executions")

        if success and v:
            raise ValueError(
                "Error message should not be provided for successful executions"
            )

        return v

    def mark_completed(self, result: Any = None, error: Exception = None) -> None:
        """Mark the result as completed with success or failure"""
        self.end_time = time.time()
        self.execution_time_ms = (self.end_time - self.start_time) * 1000

        if error:
            self.status = ToolExecutionStatus.FAILED
            self.success = False
            self.error_message = str(error)
            self.error_type = type(error).__name__
        else:
            self.status = ToolExecutionStatus.SUCCESS
            self.success = True
            self.result = result

    def mark_timeout(self) -> None:
        """Mark the result as timed out"""
        self.end_time = time.time()
        self.execution_time_ms = (self.end_time - self.start_time) * 1000
        self.status = ToolExecutionStatus.TIMEOUT
        self.success = False
        self.error_message = (
            f"Tool execution timed out after {self.execution_time_ms:.1f}ms"
        )
        self.error_type = "TimeoutError"


class ConcurrentToolExecution(BaseModel):
    """
    Batch tool execution request for concurrent processing.

    Used by the concurrent execution system to manage multiple
    tool calls with shared configuration and result aggregation.
    """

    tool_calls: list[ToolCall] = Field(
        ..., min_items=1, description="Tools to execute concurrently"
    )
    max_workers: int = Field(
        default=5, ge=1, le=20, description="Maximum concurrent workers"
    )
    timeout_seconds: float = Field(
        default=30.0, ge=0.1, le=300.0, description="Global timeout"
    )
    executor_strategy: str = Field(
        default="auto", description="Execution strategy: auto, thread, async"
    )

    # Result aggregation options
    fail_fast: bool = Field(False, description="Stop all executions on first failure")
    collect_partial_results: bool = Field(
        True, description="Return partial results if some tools fail"
    )

    @validator("tool_calls")
    def validate_unique_call_ids(cls, v: list[ToolCall]) -> list[ToolCall]:
        """Ensure call IDs are unique within the batch"""
        call_ids = [call.call_id for call in v if call.call_id]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("Call IDs must be unique within a batch")
        return v

    @validator("executor_strategy")
    def validate_executor_strategy(cls, v: str) -> str:
        """Validate executor strategy is supported"""
        valid_strategies = ["auto", "thread", "async"]
        if v not in valid_strategies:
            raise ValueError(f"Executor strategy must be one of {valid_strategies}")
        return v


class ConcurrentExecutionResult(BaseModel):
    """
    Result of concurrent tool execution batch.

    Aggregates individual tool results with batch-level metadata
    and performance statistics.
    """

    tool_results: list[ToolResult] = Field(
        ..., description="Individual tool execution results"
    )
    batch_success: bool = Field(..., description="Whether entire batch succeeded")

    # Performance metrics
    total_execution_time_ms: float = Field(
        ..., ge=0.0, description="Total batch execution time"
    )
    average_tool_time_ms: float = Field(
        default=0.0, ge=0.0, description="Average tool execution time"
    )
    parallelism_achieved: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Parallelism efficiency ratio"
    )

    # Execution statistics
    total_tools: int = Field(..., ge=1, description="Total number of tools executed")
    successful_tools: int = Field(
        default=0, ge=0, description="Number of successful executions"
    )
    failed_tools: int = Field(
        default=0, ge=0, description="Number of failed executions"
    )

    # Configuration used
    max_workers_used: int = Field(
        ..., ge=1, description="Number of workers actually used"
    )
    executor_strategy_used: str = Field(
        ..., description="Actual executor strategy applied"
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Calculate derived statistics
        if self.tool_results:
            self.total_tools = len(self.tool_results)
            self.successful_tools = sum(1 for r in self.tool_results if r.success)
            self.failed_tools = self.total_tools - self.successful_tools

            # Calculate average execution time
            execution_times = [r.execution_time_ms for r in self.tool_results]
            self.average_tool_time_ms = (
                sum(execution_times) / len(execution_times) if execution_times else 0.0
            )

            # Calculate parallelism efficiency (actual speedup vs theoretical maximum)
            if self.total_execution_time_ms > 0:
                serial_time = sum(execution_times)
                theoretical_parallel_time = (
                    max(execution_times) if execution_times else 0
                )
                if theoretical_parallel_time > 0:
                    self.parallelism_achieved = min(
                        1.0, theoretical_parallel_time / self.total_execution_time_ms
                    )


# Type aliases for better code readability
ToolFunction = (
    callable | Any
)  # Tool function type (will be more specific in implementation)
ToolParameterDict = dict[str, Any]  # Tool parameter dictionary
ToolResultList = list[ToolResult]  # List of tool results
