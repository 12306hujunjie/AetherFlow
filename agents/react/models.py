"""ReAct Agent Data Models

定义ReAct循环中各阶段的数据结构，使用Pydantic进行类型安全和验证。
"""

from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, Field, validator


class ActionType(str, Enum):
    """行动类型枚举"""

    TOOL_CALL = "tool_call"  # 工具调用
    FINAL_ANSWER = "final_answer"  # 最终答案
    CONTINUE_THINKING = "continue_thinking"  # 继续思考


class ReasoningResult(BaseModel):
    """推理阶段结果

    包含代理的思考过程、对当前情况的分析，以及计划的下一步行动。
    """

    thought: str = Field(..., description="代理的思考内容")
    analysis: str = Field(..., description="对当前情况的分析")
    next_action_type: ActionType = Field(..., description="计划的下一步行动类型")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="推理置信度")
    reasoning_tokens: int = Field(default=0, ge=0, description="推理消耗的token数")

    @validator("thought", "analysis")
    def validate_non_empty_strings(cls, v: str) -> str:
        """验证字符串字段非空"""
        if not v or not v.strip():
            raise ValueError("Thought and analysis must not be empty")
        return v.strip()


class ToolCall(BaseModel):
    """工具调用定义"""

    name: str = Field(..., description="工具名称")
    parameters: dict[str, Any] = Field(default_factory=dict, description="工具参数")
    call_id: str | None = Field(None, description="调用ID，用于异步追踪")


class ActionResult(BaseModel):
    """行动阶段结果

    记录代理执行的具体行动，可能是工具调用、最终回答或继续思考。
    """

    action_type: ActionType = Field(..., description="实际执行的行动类型")
    content: str = Field(..., description="行动内容描述")
    tool_call: ToolCall | None = Field(None, description="工具调用信息（如果适用）")
    final_answer: str | None = Field(None, description="最终答案（如果适用）")
    execution_time_ms: float = Field(default=0.0, ge=0.0, description="执行耗时(毫秒)")

    @validator("final_answer")
    def validate_final_answer_for_type(
        cls, v: str | None, values: dict[str, Any]
    ) -> str | None:
        """验证最终答案与行动类型的一致性"""
        action_type = values.get("action_type")
        if action_type == ActionType.FINAL_ANSWER and not v:
            raise ValueError(
                "Final answer must be provided when action_type is FINAL_ANSWER"
            )
        if action_type != ActionType.FINAL_ANSWER and v:
            raise ValueError(
                "Final answer should only be provided when action_type is FINAL_ANSWER"
            )
        return v

    @validator("tool_call")
    def validate_tool_call_for_type(
        cls, v: ToolCall | None, values: dict[str, Any]
    ) -> ToolCall | None:
        """验证工具调用与行动类型的一致性"""
        action_type = values.get("action_type")
        if action_type == ActionType.TOOL_CALL and not v:
            raise ValueError("Tool call must be provided when action_type is TOOL_CALL")
        if action_type != ActionType.TOOL_CALL and v:
            raise ValueError(
                "Tool call should only be provided when action_type is TOOL_CALL"
            )
        return v


class ObservationResult(BaseModel):
    """观察阶段结果

    记录对行动结果的观察、处理和状态更新。
    """

    observation: str = Field(..., description="观察到的结果描述")
    tool_output: Any | None = Field(None, description="工具执行的原始输出")
    success: bool = Field(..., description="行动是否成功执行")
    error_message: str | None = Field(None, description="错误信息（如果失败）")
    should_continue: bool = Field(..., description="是否应该继续ReAct循环")
    updated_context: dict[str, Any] = Field(
        default_factory=dict, description="更新的上下文信息"
    )

    @validator("error_message")
    def validate_error_for_success(
        cls, v: str | None, values: dict[str, Any]
    ) -> str | None:
        """验证错误信息与成功状态的一致性"""
        success = values.get("success", True)
        if not success and not v:
            raise ValueError("Error message must be provided when success is False")
        if success and v:
            raise ValueError(
                "Error message should not be provided when success is True"
            )
        return v


class ReActStep(BaseModel):
    """完整的ReAct循环步骤

    包含一个完整循环中的推理、行动和观察结果。
    """

    step_number: int = Field(..., ge=1, description="步骤序号")
    reasoning: ReasoningResult = Field(..., description="推理结果")
    action: ActionResult = Field(..., description="行动结果")
    observation: ObservationResult = Field(..., description="观察结果")
    total_time_ms: float = Field(
        default=0.0, ge=0.0, description="整个步骤总耗时(毫秒)"
    )


class ConversationEntry(BaseModel):
    """会话条目

    记录会话历史中的单条消息。
    """

    role: str = Field(..., description="角色: user, assistant, system, tool")
    content: str = Field(..., description="消息内容")
    timestamp: float | None = Field(None, description="时间戳")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")

    @validator("role")
    def validate_role(cls, v: str) -> str:
        """验证角色值"""
        valid_roles = {"user", "assistant", "system", "tool"}
        if v not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}")
        return v


class ReActExecutionResult(BaseModel):
    """ReAct执行完整结果

    记录整个ReAct循环的执行结果和统计信息。
    """

    success: bool = Field(..., description="整体执行是否成功")
    final_answer: str | None = Field(None, description="最终答案")
    total_steps: int = Field(..., ge=0, description="总执行步骤数")
    execution_time_ms: float = Field(..., ge=0.0, description="总执行时间(毫秒)")
    steps_history: list[ReActStep] = Field(
        default_factory=list, description="步骤执行历史"
    )
    termination_reason: str = Field(..., description="终止原因")
    error_message: str | None = Field(None, description="错误信息（如果失败）")

    # 性能统计
    avg_step_time_ms: float = Field(
        default=0.0, ge=0.0, description="平均步骤耗时(毫秒)"
    )
    total_reasoning_tokens: int = Field(default=0, ge=0, description="总推理token消耗")

    def __init__(self, **data):
        super().__init__(**data)
        # 自动计算平均步骤时间
        if self.total_steps > 0 and self.execution_time_ms > 0:
            self.avg_step_time_ms = self.execution_time_ms / self.total_steps


# 类型别名，提供更好的类型提示
ReActMessage = Union[ReasoningResult, ActionResult, ObservationResult]
