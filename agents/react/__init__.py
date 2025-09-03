"""ReAct Agent Implementation

基于AetherFlow框架的ReAct(Reasoning-Acting-Observing)智能代理实现。

主要组件：
- ReActContext: 扩展BaseFlowContext，管理代理状态和会话历史
- reasoning_step: 推理阶段节点，分析当前状态决定行动
- action_step: 行动阶段节点，执行工具调用或生成回复
- observation_step: 观察阶段节点，处理行动结果更新状态
- create_react_agent: fluent interface组合函数，创建完整ReAct循环
"""

from .context import ReActContext
from .engine import create_react_agent
from .models import (
    ActionResult,
    ObservationResult,
    ReasoningResult,
)
from .nodes import action_step, observation_step, reasoning_step

__all__ = [
    # Core ReAct Engine
    "ReActContext",
    "create_react_agent",
    # Core Node Functions
    "reasoning_step",
    "action_step",
    "observation_step",
    # Data Models
    "ReasoningResult",
    "ActionResult",
    "ObservationResult",
]
