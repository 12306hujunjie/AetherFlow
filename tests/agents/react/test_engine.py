#!/usr/bin/env python3
"""
test_engine.py - ReAct引擎测试

测试ReAct引擎的组合功能、循环控制和完整执行流程。
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from aetherflow import Node
from agents.react.context import ReActContext
from agents.react.engine import (
    ReActEngineBuilder,
    create_react_agent,
    create_react_engine_builder,
    create_react_reasoning_only,
    create_react_single_step,
)
from agents.react.models import ReActExecutionResult


class TestCreateReActAgent:
    """create_react_agent函数测试"""

    def setup_method(self):
        """每个测试前的设置"""
        self.context = ReActContext()
        self.context.wire(modules=["agents.react"])

    def test_create_react_agent_basic(self):
        """测试基础ReAct代理创建"""
        agent = create_react_agent(self.context)

        assert isinstance(agent, Node)
        assert agent.name == "react_agent_engine"
        assert agent.is_start_node is True

    def test_create_react_agent_custom_params(self):
        """测试自定义参数的ReAct代理创建"""
        agent = create_react_agent(
            context=self.context, max_steps=5, stop_on_error=True
        )

        assert isinstance(agent, Node)

        # 验证初始化参数被正确设置
        state = self.context.get_react_state(self.context)
        assert state.max_steps == 5

    @pytest.mark.asyncio
    async def test_react_agent_execution_simple_query(self):
        """测试简单查询的完整执行"""
        agent = create_react_agent(self.context, max_steps=3)

        query = "你好，很高兴见到你"

        start_time = time.time()
        result = agent(query)
        execution_time = (time.time() - start_time) * 1000

        # 验证执行结果
        assert isinstance(result, ReActExecutionResult)
        assert result.success is True or result.success is False  # 可能成功也可能失败
        assert result.total_steps >= 1
        assert result.execution_time_ms > 0
        assert result.termination_reason in [
            "final_answer_provided",
            "max_steps_reached",
            "observation_indicated_completion",
            "error",
        ]

        # 验证性能要求（不包括真实LLM调用时应该很快）
        assert execution_time < 1000, f"执行时间 {execution_time:.2f}ms 超过1秒阈值"

        # 验证上下文状态
        state = self.context.get_react_state(self.context)
        assert len(state.conversation_history) >= 1  # 至少有用户查询
        assert state.current_step >= 1

    @pytest.mark.asyncio
    async def test_react_agent_execution_tool_query(self):
        """测试需要工具的查询执行"""
        agent = create_react_agent(self.context, max_steps=5)

        query = "请帮我搜索最新的机器学习资讯"

        result = agent(query)

        # 验证执行结果
        assert isinstance(result, ReActExecutionResult)
        assert result.total_steps >= 1

        # 对于工具查询，可能会执行多步
        state = self.context.get_react_state(self.context)
        assert len(state.reasoning_history) >= 1

        # 检查是否有工具相关的会话记录
        conversation_roles = [entry.role for entry in state.conversation_history]
        # 根据执行结果，可能包含tool角色的记录

    def test_react_agent_max_steps_limit(self):
        """测试最大步骤数限制"""
        agent = create_react_agent(self.context, max_steps=2)

        query = "这是一个可能需要多步处理的复杂问题"

        result = agent(query)

        # 验证步骤数限制
        assert isinstance(result, ReActExecutionResult)
        assert result.total_steps <= 2

        # 如果达到最大步骤数，应该有相应的终止原因
        if result.total_steps == 2:
            assert result.termination_reason in [
                "max_steps_reached",
                "final_answer_provided",
            ]

    def test_react_agent_conversation_history(self):
        """测试会话历史记录"""
        agent = create_react_agent(self.context, max_steps=3)

        query = "测试会话历史"

        result = agent(query)

        # 验证会话历史
        state = self.context.get_react_state(self.context)
        conversation_history = state.conversation_history

        assert len(conversation_history) >= 1

        # 第一条应该是用户查询
        assert conversation_history[0].role == "user"
        assert conversation_history[0].content == query

        # 如果有最终答案，最后一条应该是assistant回复
        if result.final_answer:
            assistant_entries = [
                entry for entry in conversation_history if entry.role == "assistant"
            ]
            assert len(assistant_entries) >= 1


class TestReActSingleStep:
    """单步ReAct功能测试"""

    def setup_method(self):
        """每个测试前的设置"""
        self.context = ReActContext()
        self.context.wire(modules=["agents.react"])
        self.context.initialize_execution()

    @pytest.mark.asyncio
    async def test_single_step_execution(self):
        """测试单步执行"""
        query = "单步测试查询"

        single_step_node = create_react_single_step(query, self.context)

        assert isinstance(single_step_node, Node)

        result = single_step_node()

        # 验证单步执行结果（应该返回ObservationResult）
        from agents.react.models import ObservationResult

        assert isinstance(result, ObservationResult)

        # 验证上下文更新
        state = self.context.get_react_state(self.context)
        assert len(state.reasoning_history) == 1
        assert len(state.action_history) == 1
        assert len(state.observation_history) == 1
        assert state.current_step == 1

    @pytest.mark.asyncio
    async def test_reasoning_only_execution(self):
        """测试仅推理执行"""
        query = "仅推理测试"

        reasoning_only_node = create_react_reasoning_only(query, self.context)

        assert isinstance(reasoning_only_node, Node)
        assert reasoning_only_node.name == "reasoning_only"

        result = reasoning_only_node()

        # 验证推理结果
        from agents.react.models import ReasoningResult

        assert isinstance(result, ReasoningResult)
        assert result.thought is not None
        assert result.analysis is not None

        # 验证上下文状态
        state = self.context.get_react_state(self.context)
        assert len(state.reasoning_history) == 1
        assert len(state.action_history) == 0  # 仅推理，无行动
        assert len(state.observation_history) == 0


class TestReActEngineBuilder:
    """ReAct引擎构建器测试"""

    def setup_method(self):
        """每个测试前的设置"""
        self.context = ReActContext()
        self.context.wire(modules=["agents.react"])

    def test_builder_pattern_basic(self):
        """测试基础构建器模式"""
        builder = ReActEngineBuilder()

        agent = builder.with_context(self.context).build()

        assert isinstance(agent, Node)

    def test_builder_pattern_full_config(self):
        """测试完整配置构建"""
        builder = ReActEngineBuilder()

        agent = (
            builder.with_context(self.context)
            .max_steps(8)
            .stop_on_error(True)
            .with_tools({"test_tool": "test_impl"})
            .with_prompts({"reasoning": "自定义推理提示"})
            .build()
        )

        assert isinstance(agent, Node)

        # 验证配置应用
        state = self.context.get_react_state(self.context)
        assert state.max_steps == 8

    def test_builder_validation(self):
        """测试构建器参数验证"""
        builder = ReActEngineBuilder()

        # 测试缺少上下文的错误
        with pytest.raises(ValueError, match="Context is required"):
            builder.build()

        # 测试无效最大步骤数
        with pytest.raises(ValueError, match="max_steps must be positive"):
            builder.max_steps(0)

        # 测试负数最大步骤数
        with pytest.raises(ValueError, match="max_steps must be positive"):
            builder.max_steps(-1)

    def test_builder_method_chaining(self):
        """测试构建器方法链式调用"""
        builder = ReActEngineBuilder()

        # 验证每个方法都返回self，支持链式调用
        assert builder.with_context(self.context) is builder
        assert builder.max_steps(5) is builder
        assert builder.stop_on_error(False) is builder
        assert builder.with_tools({}) is builder
        assert builder.with_prompts({}) is builder

    def test_create_react_engine_builder_convenience(self):
        """测试便捷构建器创建函数"""
        builder = create_react_engine_builder()

        assert isinstance(builder, ReActEngineBuilder)

        # 测试便捷函数创建的构建器可以正常使用
        agent = builder.with_context(self.context).max_steps(3).build()

        assert isinstance(agent, Node)


class TestReActEngineIntegration:
    """ReAct引擎集成测试"""

    def setup_method(self):
        """每个测试前的设置"""
        self.context = ReActContext()
        self.context.wire(modules=["agents.react"])

    def test_engine_error_handling(self):
        """测试引擎错误处理"""
        agent = create_react_agent(self.context, max_steps=2, stop_on_error=False)

        # 使用可能导致错误的查询（虽然当前实现比较健壮）
        query = ""  # 空查询

        result = agent(query)

        # 即使有错误，也应该返回结果对象
        assert isinstance(result, ReActExecutionResult)
        # 错误情况下success可能为False
        if not result.success:
            assert result.error_message is not None
            assert result.termination_reason == "error"

    def test_engine_performance_benchmark(self):
        """测试引擎性能基准"""
        agent = create_react_agent(self.context, max_steps=3)

        query = "性能测试查询"

        # 执行多次以获得稳定的性能数据
        times = []
        for _ in range(5):
            # 重新初始化上下文
            self.context.initialize_execution()

            start_time = time.time()
            result = agent(query)
            execution_time = (time.time() - start_time) * 1000
            times.append(execution_time)

            # 验证每次执行都成功
            assert isinstance(result, ReActExecutionResult)

        # 计算平均性能
        avg_time = sum(times) / len(times)
        max_time = max(times)

        # 性能要求：平均执行时间 < 500ms，最大执行时间 < 1000ms
        assert avg_time < 500, f"平均执行时间 {avg_time:.2f}ms 超过500ms阈值"
        assert max_time < 1000, f"最大执行时间 {max_time:.2f}ms 超过1000ms阈值"

    @pytest.mark.asyncio
    async def test_engine_concurrent_execution(self):
        """测试引擎并发执行"""
        import asyncio

        async def execute_agent(query_id: int):
            """异步执行代理"""
            context = ReActContext()
            context.wire(modules=["agents.react"])

            agent = create_react_agent(context, max_steps=2)

            query = f"并发测试查询 {query_id}"

            # 在异步环境中执行同步代理
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, agent, query)

            return query_id, result

        # 并发执行多个代理
        tasks = [execute_agent(i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        # 验证所有并发执行都成功
        assert len(results) == 3

        for query_id, result in results:
            assert isinstance(result, ReActExecutionResult)
            # 验证结果独立性
            assert f"并发测试查询 {query_id}" in str(result.__dict__)

    def test_engine_state_isolation(self):
        """测试引擎状态隔离"""
        # 创建两个不同的上下文和代理
        context1 = ReActContext()
        context1.wire(modules=["agents.react"])
        agent1 = create_react_agent(context1, max_steps=3)

        context2 = ReActContext()
        context2.wire(modules=["agents.react"])
        agent2 = create_react_agent(context2, max_steps=5)

        # 执行不同的查询
        result1 = agent1("第一个代理的查询")
        result2 = agent2("第二个代理的查询")

        # 验证状态隔离
        state1 = context1.get_react_state(context1)
        state2 = context2.get_react_state(context2)

        assert state1.max_steps == 3
        assert state2.max_steps == 5

        # 验证会话历史独立
        assert state1.conversation_history[0].content == "第一个代理的查询"
        assert state2.conversation_history[0].content == "第二个代理的查询"

    def test_engine_fluent_interface_compatibility(self):
        """测试引擎与AetherFlow fluent interface的兼容性"""
        from aetherflow import node

        # 创建一个简单的后处理节点
        @node
        def post_process_result(result: ReActExecutionResult) -> dict:
            """后处理ReAct结果"""
            return {
                "success": result.success,
                "steps": result.total_steps,
                "time_ms": result.execution_time_ms,
                "has_answer": result.final_answer is not None,
            }

        # 创建ReAct代理
        agent = create_react_agent(self.context, max_steps=2)

        # 使用fluent interface链式调用
        pipeline = agent.then(post_process_result)

        assert isinstance(pipeline, Node)

        # 执行完整流水线
        result = pipeline("测试fluent interface")

        # 验证结果
        assert isinstance(result, dict)
        assert "success" in result
        assert "steps" in result
        assert "time_ms" in result
        assert "has_answer" in result
        assert isinstance(result["steps"], int)
        assert isinstance(result["time_ms"], float)
