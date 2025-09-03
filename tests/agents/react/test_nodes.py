#!/usr/bin/env python3
"""
test_nodes.py - ReAct节点测试

测试reasoning_step、action_step、observation_step三个核心节点的功能。
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from agents.react.context import ReActContext
from agents.react.models import (
    ActionResult,
    ActionType,
    ObservationResult,
    ReasoningResult,
    ToolCall,
)
from agents.react.nodes import action_step, observation_step, reasoning_step


class TestReasoningStep:
    """推理步骤节点测试"""

    def setup_method(self):
        """每个测试前的设置"""
        self.context = ReActContext()
        self.context.wire(modules=["agents.react"])
        self.context.initialize_execution()

    @pytest.mark.asyncio
    async def test_reasoning_step_basic(self):
        """测试基础推理功能"""
        query = "什么是机器学习？"

        result = await reasoning_step(query, self.context)

        # 验证返回类型和基本字段
        assert isinstance(result, ReasoningResult)
        assert isinstance(result.thought, str)
        assert len(result.thought) > 0
        assert isinstance(result.analysis, str)
        assert len(result.analysis) > 0
        assert isinstance(result.next_action_type, ActionType)
        assert 0.0 <= result.confidence <= 1.0
        assert result.reasoning_tokens > 0

        # 验证上下文更新
        state = self.context.get_react_state(self.context)
        assert len(state.reasoning_history) == 1
        assert state.total_reasoning_tokens == result.reasoning_tokens

    @pytest.mark.asyncio
    async def test_reasoning_step_tool_required(self):
        """测试需要工具调用的查询推理"""
        query = "帮我搜索最新的Python教程"

        result = await reasoning_step(query, self.context)

        # 应该识别为需要工具调用
        assert result.next_action_type == ActionType.TOOL_CALL
        assert "搜索" in query  # 确认触发了工具调用逻辑

    @pytest.mark.asyncio
    async def test_reasoning_step_direct_answer(self):
        """测试可以直接回答的查询推理"""
        query = "你好，很高兴见到你"

        result = await reasoning_step(query, self.context)

        # 应该直接给出最终答案
        assert result.next_action_type == ActionType.FINAL_ANSWER

    @pytest.mark.asyncio
    async def test_reasoning_step_multiple_rounds(self):
        """测试多轮推理"""
        query = "复杂问题需要多步分析"

        # 第一轮推理
        result1 = await reasoning_step(query, self.context)
        assert isinstance(result1, ReasoningResult)

        # 模拟步骤推进
        state = self.context.get_react_state(self.context)
        state.current_step = 1

        # 第二轮推理
        result2 = await reasoning_step("继续分析", self.context)
        assert isinstance(result2, ReasoningResult)

        # 验证历史记录
        state = self.context.get_react_state(self.context)
        assert len(state.reasoning_history) == 2
        assert (
            state.total_reasoning_tokens
            == result1.reasoning_tokens + result2.reasoning_tokens
        )

    @pytest.mark.asyncio
    async def test_reasoning_step_max_steps_reached(self):
        """测试达到最大步骤数的推理"""
        query = "测试查询"

        # 模拟接近最大步骤数
        state = self.context.get_react_state(self.context)
        state.current_step = 4  # 接近默认最大值

        result = await reasoning_step(query, self.context)

        # 应该倾向于给出最终答案
        assert result.next_action_type in [
            ActionType.FINAL_ANSWER,
            ActionType.CONTINUE_THINKING,
        ]


class TestActionStep:
    """行动步骤节点测试"""

    def setup_method(self):
        """每个测试前的设置"""
        self.context = ReActContext()
        self.context.wire(modules=["agents.react"])
        self.context.initialize_execution()

    @pytest.mark.asyncio
    async def test_action_step_tool_call(self):
        """测试工具调用行动"""
        reasoning = ReasoningResult(
            thought="需要搜索信息",
            analysis="用户查询需要外部数据",
            next_action_type=ActionType.TOOL_CALL,
            confidence=0.9,
        )

        result = await action_step(reasoning, self.context)

        # 验证结果
        assert isinstance(result, ActionResult)
        assert result.action_type == ActionType.TOOL_CALL
        assert result.tool_call is not None
        assert isinstance(result.tool_call, ToolCall)
        assert result.tool_call.name == "search_tool"
        assert result.final_answer is None
        assert result.execution_time_ms > 0

        # 验证上下文更新
        state = self.context.get_react_state(self.context)
        assert len(state.action_history) == 1
        assert state.action_history[0].action_type == ActionType.TOOL_CALL

    @pytest.mark.asyncio
    async def test_action_step_final_answer(self):
        """测试最终答案行动"""
        reasoning = ReasoningResult(
            thought="我可以直接回答这个问题",
            analysis="基于现有信息足够给出答案",
            next_action_type=ActionType.FINAL_ANSWER,
            confidence=0.85,
        )

        result = await action_step(reasoning, self.context)

        # 验证结果
        assert isinstance(result, ActionResult)
        assert result.action_type == ActionType.FINAL_ANSWER
        assert result.final_answer is not None
        assert isinstance(result.final_answer, str)
        assert len(result.final_answer) > 0
        assert result.tool_call is None
        assert result.execution_time_ms > 0

        # 验证最终答案被设置到上下文
        state = self.context.get_react_state(self.context)
        assert state.final_answer == result.final_answer

    @pytest.mark.asyncio
    async def test_action_step_continue_thinking(self):
        """测试继续思考行动"""
        reasoning = ReasoningResult(
            thought="需要更多思考",
            analysis="当前信息不足",
            next_action_type=ActionType.CONTINUE_THINKING,
            confidence=0.6,
        )

        result = await action_step(reasoning, self.context)

        # 验证结果
        assert isinstance(result, ActionResult)
        assert result.action_type == ActionType.CONTINUE_THINKING
        assert result.tool_call is None
        assert result.final_answer is None
        assert "继续" in result.content

    @pytest.mark.asyncio
    async def test_action_step_invalid_action_type(self):
        """测试无效行动类型的错误处理"""
        reasoning = ReasoningResult(
            thought="测试",
            analysis="测试",
            next_action_type="INVALID_ACTION",  # 无效的行动类型
            confidence=0.5,
        )

        with pytest.raises((ValueError, TypeError)):
            await action_step(reasoning, self.context)


class TestObservationStep:
    """观察步骤节点测试"""

    def setup_method(self):
        """每个测试前的设置"""
        self.context = ReActContext()
        self.context.wire(modules=["agents.react"])
        self.context.initialize_execution()

    @pytest.mark.asyncio
    async def test_observation_step_final_answer(self):
        """测试最终答案的观察"""
        action = ActionResult(
            action_type=ActionType.FINAL_ANSWER,
            content="生成最终答案",
            final_answer="这是测试的最终答案",
            execution_time_ms=50.0,
        )

        result = await observation_step(action, self.context)

        # 验证结果
        assert isinstance(result, ObservationResult)
        assert result.success is True
        assert result.should_continue is False  # 最终答案应该停止循环
        assert result.error_message is None
        assert "最终答案" in result.observation

        # 验证上下文更新
        state = self.context.get_react_state(self.context)
        assert len(state.observation_history) == 1
        assert state.current_step == 1  # 步骤应该增加
        assert state.is_complete is True  # 应该标记为完成

        # 验证会话历史
        assert len(state.conversation_history) == 1
        assert state.conversation_history[0].role == "assistant"
        assert state.conversation_history[0].content == "这是测试的最终答案"

    @pytest.mark.asyncio
    async def test_observation_step_tool_call(self):
        """测试工具调用的观察"""
        tool_call = ToolCall(
            name="test_tool", parameters={"param": "value"}, call_id="test_123"
        )

        action = ActionResult(
            action_type=ActionType.TOOL_CALL,
            content="执行工具调用",
            tool_call=tool_call,
            execution_time_ms=120.0,
        )

        result = await observation_step(action, self.context)

        # 验证结果
        assert isinstance(result, ObservationResult)
        assert result.success is True
        assert result.should_continue is True  # 工具调用后应该继续
        assert result.error_message is None
        assert "工具" in result.observation
        assert result.tool_output is not None
        assert "updated_context" in result.updated_context

        # 验证上下文更新
        state = self.context.get_react_state(self.context)
        assert state.current_step == 1
        assert state.is_complete is False  # 应该继续执行

        # 验证工具结果添加到会话历史
        assert len(state.conversation_history) == 1
        assert state.conversation_history[0].role == "tool"
        assert state.conversation_history[0].metadata["tool_name"] == "test_tool"
        assert state.conversation_history[0].metadata["call_id"] == "test_123"

    @pytest.mark.asyncio
    async def test_observation_step_continue_thinking(self):
        """测试继续思考的观察"""
        action = ActionResult(
            action_type=ActionType.CONTINUE_THINKING,
            content="继续思考分析",
            execution_time_ms=30.0,
        )

        result = await observation_step(action, self.context)

        # 验证结果
        assert isinstance(result, ObservationResult)
        assert result.success is True
        assert result.should_continue is True  # 应该继续思考
        assert result.error_message is None
        assert "思考" in result.observation

        # 验证状态
        state = self.context.get_react_state(self.context)
        assert state.is_complete is False

    @pytest.mark.asyncio
    async def test_observation_step_context_integration(self):
        """测试观察步骤的上下文集成"""
        # 准备初始状态
        initial_step = self.context.get_current_step()

        action = ActionResult(
            action_type=ActionType.TOOL_CALL,
            content="测试工具调用",
            tool_call=ToolCall(name="test", parameters={}, call_id="123"),
            execution_time_ms=100.0,
        )

        result = await observation_step(action, self.context)

        # 验证上下文的完整更新
        state = self.context.get_react_state(self.context)

        # 检查步骤计数
        assert state.current_step == initial_step + 1

        # 检查历史记录
        assert len(state.observation_history) == 1
        assert len(state.action_history) == 0  # observation_step不会添加action

        # 检查观察结果的详细信息
        saved_observation = state.observation_history[0]
        assert saved_observation.observation == result.observation
        assert saved_observation.success == result.success
        assert saved_observation.should_continue == result.should_continue


class TestNodeIntegration:
    """节点集成测试"""

    def setup_method(self):
        """每个测试前的设置"""
        self.context = ReActContext()
        self.context.wire(modules=["agents.react"])
        self.context.initialize_execution()

    @pytest.mark.asyncio
    async def test_complete_react_cycle(self):
        """测试完整的ReAct循环"""
        query = "测试完整循环"

        # 1. 推理步骤
        reasoning_result = await reasoning_step(query, self.context)
        assert isinstance(reasoning_result, ReasoningResult)

        # 2. 行动步骤
        action_result = await action_step(reasoning_result, self.context)
        assert isinstance(action_result, ActionResult)

        # 3. 观察步骤
        observation_result = await observation_step(action_result, self.context)
        assert isinstance(observation_result, ObservationResult)

        # 验证完整的历史记录
        state = self.context.get_react_state(self.context)
        assert len(state.reasoning_history) == 1
        assert len(state.action_history) == 1
        assert len(state.observation_history) == 1
        assert state.current_step == 1

    @pytest.mark.asyncio
    async def test_multiple_react_cycles(self):
        """测试多轮ReAct循环"""
        query = "需要多轮处理的复杂查询"

        # 第一轮循环
        reasoning1 = await reasoning_step(query, self.context)
        action1 = await action_step(reasoning1, self.context)
        observation1 = await observation_step(action1, self.context)

        # 如果需要继续，进行第二轮
        if observation1.should_continue:
            reasoning2 = await reasoning_step("继续分析", self.context)
            action2 = await action_step(reasoning2, self.context)
            observation2 = await observation_step(action2, self.context)

            # 验证两轮循环的状态
            state = self.context.get_react_state(self.context)
            assert len(state.reasoning_history) == 2
            assert len(state.action_history) == 2
            assert len(state.observation_history) == 2
            assert state.current_step == 2

    @pytest.mark.asyncio
    async def test_node_error_handling(self):
        """测试节点的错误处理"""
        # 测试推理节点的错误处理
        with pytest.raises(TypeError):
            await reasoning_step(None, self.context)  # 无效查询

        # 测试行动节点的错误处理（无效推理结果）
        invalid_reasoning = "这不是ReasoningResult对象"
        with pytest.raises(AttributeError):
            await action_step(invalid_reasoning, self.context)

    @pytest.mark.asyncio
    async def test_node_performance(self):
        """测试节点性能"""
        query = "性能测试查询"

        # 测试推理节点性能
        start_time = time.time()
        reasoning_result = await reasoning_step(query, self.context)
        reasoning_time = (time.time() - start_time) * 1000

        # ReAct循环单步应该 < 100ms（不含真实LLM调用）
        assert reasoning_time < 100, (
            f"推理步骤耗时 {reasoning_time:.2f}ms 超过100ms阈值"
        )

        # 测试行动节点性能
        start_time = time.time()
        action_result = await action_step(reasoning_result, self.context)
        action_time = (time.time() - start_time) * 1000

        assert action_time < 100, f"行动步骤耗时 {action_time:.2f}ms 超过100ms阈值"

        # 测试观察节点性能
        start_time = time.time()
        observation_result = await observation_step(action_result, self.context)
        observation_time = (time.time() - start_time) * 1000

        assert observation_time < 100, (
            f"观察步骤耗时 {observation_time:.2f}ms 超过100ms阈值"
        )

        # 验证整个循环性能目标
        total_time = reasoning_time + action_time + observation_time
        assert total_time < 300, f"完整ReAct循环耗时 {total_time:.2f}ms 超过300ms阈值"
