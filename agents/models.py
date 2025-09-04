"""
Core data models for ReAct Agent system.

This module defines the primary data structures used by the ReActAgent
for managing execution state, responses, and session information.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agents.react.models import ActionResult, ObservationResult, ReasoningResult
from agents.tools.models import ToolCall


class ReActStep(BaseModel):
    """Detailed information about a single ReAct step."""

    step_number: int = Field(..., description="Step number in the execution sequence")
    reasoning: str = Field(..., description="The agent's reasoning for this step")
    action: ToolCall | None = Field(None, description="Action taken (if any)")
    observation: str | None = Field(None, description="Result of the action")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When this step occurred"
    )
    duration: float = Field(0.0, description="Time taken for this step in seconds")

    # Link to internal results for detailed analysis
    reasoning_result: ReasoningResult | None = Field(
        None, description="Internal reasoning result"
    )
    action_result: ActionResult | None = Field(
        None, description="Internal action result"
    )
    observation_result: ObservationResult | None = Field(
        None, description="Internal observation result"
    )


class AgentResponse(BaseModel):
    """Complete response from a ReAct agent execution."""

    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Final answer from the agent")
    session_id: str = Field(..., description="Session identifier")
    steps: list[ReActStep] = Field(
        default_factory=list, description="Detailed execution steps"
    )
    total_steps: int = Field(0, description="Total number of steps executed")
    execution_time: float = Field(0.0, description="Total execution time in seconds")
    tokens_used: int = Field(0, description="Total tokens consumed")
    tools_called: list[str] = Field(
        default_factory=list, description="Names of tools that were called"
    )
    success: bool = Field(True, description="Whether execution completed successfully")
    error_message: str | None = Field(
        None, description="Error message if execution failed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class AgentStreamChunk(BaseModel):
    """Individual chunk in a streaming agent response."""

    type: str = Field(
        ..., description="Chunk type: reasoning, action, observation, or final"
    )
    content: str = Field(..., description="Chunk content")
    step_number: int = Field(..., description="Current step number")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional chunk metadata"
    )


class Session(BaseModel):
    """Information about an agent session."""

    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Session creation time"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update time"
    )
    message_count: int = Field(0, description="Number of messages in this session")
    status: str = Field("active", description="Session status: active or inactive")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Session metadata"
    )


class ToolInfo(BaseModel):
    """Information about a registered tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Tool parameter schema"
    )
    category: str | None = Field(None, description="Tool category")
    enabled: bool = Field(True, description="Whether the tool is enabled")
