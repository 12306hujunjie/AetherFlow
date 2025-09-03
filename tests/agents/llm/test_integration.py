"""OpenAI接口抽象层集成测试。"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from agents.exceptions import OpenAIException
from agents.llm import (
    AgentContainer,
    ChatMessage,
    CircuitBreaker,
    ReActEntry,
    ReActPromptTemplate,
    ReActStep,
    create_agent_container,
)


class TestOpenAIIntegration:
    """测试OpenAI接口抽象层集成功能。"""

    @pytest.fixture
    def container(self) -> AgentContainer:
        """集成测试容器fixture。"""
        return create_agent_container(
            openai_api_key="test-integration-key",
            openai_model="gpt-3.5-turbo",
            temperature=0.7,
        )

    def test_full_stack_configuration(self, container: AgentContainer):
        """测试完整的配置栈。"""
        # 获取所有核心组件
        config = container.openai_config()
        client = container.llm_client()
        template = container.react_template()

        # 验证配置传播
        assert config.api_key == "test-integration-key"
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.7

        # 验证客户端使用正确配置
        assert client.config is config
        assert client.config.api_key == "test-integration-key"

        # 验证模板可用
        assert isinstance(template, ReActPromptTemplate)
        assert template.language == "zh"

    @patch("agents.llm.client.OpenAI")
    def test_sync_chat_completion_workflow(
        self, mock_openai, container: AgentContainer
    ):
        """测试同步聊天完成工作流。"""
        # 模拟OpenAI响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "我理解了您的问题。"
        mock_response.usage.total_tokens = 30

        mock_instance = Mock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_instance

        # 获取客户端并执行聊天完成
        client = container.llm_client()
        messages = [
            ChatMessage(role="system", content="您是一个有用的助手。"),
            ChatMessage(role="user", content="你好！"),
        ]

        result = client.chat_complete(messages)

        assert result == "我理解了您的问题。"

        # 验证调用参数
        expected_messages = [
            {"role": "system", "content": "您是一个有用的助手。"},
            {"role": "user", "content": "你好！"},
        ]
        mock_instance.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=expected_messages,
            temperature=0.7,
            max_tokens=None,
        )

    @patch("agents.llm.client.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_async_chat_completion_workflow(
        self, mock_async_openai, container: AgentContainer
    ):
        """测试异步聊天完成工作流。"""
        # 模拟异步OpenAI响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "异步响应成功。"
        mock_response.usage.total_tokens = 25

        mock_instance = AsyncMock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_async_openai.return_value = mock_instance

        # 获取客户端并执行异步聊天完成
        client = container.llm_client()
        messages = [ChatMessage(role="user", content="异步测试")]

        result = await client.chat_complete_async(messages)

        assert result == "异步响应成功。"
        mock_instance.chat.completions.create.assert_called_once()

    def test_react_template_integration(self, container: AgentContainer):
        """测试ReAct模板集成。"""
        template = container.react_template()

        # 测试完整的ReAct工作流
        query = "计算 2 + 2 * 3 的结果"
        tools = ["calculator", "search"]
        history = []

        # 格式化推理提示
        prompt = template.format_reasoning_prompt(query, history, tools)

        assert query in prompt
        assert "calculator" in prompt
        assert "search" in prompt
        assert "Thought:" in prompt
        assert "Action:" in prompt

        # 模拟LLM响应并解析
        llm_response = """Thought: 我需要计算这个数学表达式。
Action: calculator
Action Input: {"expression": "2 + 2 * 3"}"""

        parsed_response = template.parse_llm_response(llm_response)

        assert parsed_response.thought == "我需要计算这个数学表达式。"
        assert parsed_response.action == "calculator"
        assert parsed_response.action_input == {"expression": "2 + 2 * 3"}
        assert parsed_response.has_action is True
        assert parsed_response.has_answer is False

        # 添加观察结果并格式化新提示
        history.append(ReActEntry(ReActStep.THOUGHT, parsed_response.thought))
        history.append(ReActEntry(ReActStep.ACTION, parsed_response.action))
        observation = "计算结果是 8"

        observation_prompt = template.format_observation_prompt(observation, history)

        assert observation in observation_prompt
        assert "我需要计算这个数学表达式。" in observation_prompt

        # 解析最终答案
        final_response = """Thought: 基于计算器的结果，我现在可以给出答案。
Answer: 2 + 2 * 3 = 8"""

        final_parsed = template.parse_llm_response(final_response)

        assert final_parsed.has_answer is True
        assert "2 + 2 * 3 = 8" in final_parsed.answer

    @patch("agents.llm.client.OpenAI")
    def test_retry_mechanism_integration(self, mock_openai, container: AgentContainer):
        """测试重试机制集成。"""
        call_count = 0

        def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 Rate limit exceeded")  # 触发重试
            # 第三次调用成功
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "重试后成功"
            mock_response.usage.total_tokens = 15
            return mock_response

        mock_instance = Mock()
        mock_instance.chat.completions.create.side_effect = mock_create
        mock_openai.return_value = mock_instance

        client = container.llm_client()

        with patch("time.sleep"):  # 加快测试速度
            result = client.complete("测试重试机制")

        assert result == "重试后成功"
        assert call_count == 3  # 应该重试了3次

    @patch("agents.llm.client.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_streaming_integration(
        self, mock_async_openai, container: AgentContainer
    ):
        """测试流式响应集成。"""
        # 模拟流式响应chunks
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="这"))]),
            Mock(choices=[Mock(delta=Mock(content="是"))]),
            Mock(choices=[Mock(delta=Mock(content="流式"))]),
            Mock(choices=[Mock(delta=Mock(content="响应"))]),
            Mock(choices=[Mock(delta=Mock(content=None))]),  # 结束
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        mock_instance = AsyncMock()
        mock_instance.chat.completions.create.return_value = mock_stream()
        mock_async_openai.return_value = mock_instance

        client = container.llm_client()
        messages = [ChatMessage(role="user", content="测试流式响应")]

        # 获取流式响应
        stream = await client.chat_complete_async(messages, stream=True)

        # 收集所有chunks
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert chunks == ["这", "是", "流式", "响应"]

    def test_multiple_language_templates(self, container: AgentContainer):
        """测试多语言模板支持。"""
        # 测试中文模板
        zh_template = container.react_template_zh()
        zh_prompt = zh_template.format_reasoning_prompt("测试", [], ["工具"])
        assert "你是一个强大的AI助手" in zh_prompt

        # 测试英文模板
        en_template = container.react_template_en()
        en_prompt = en_template.format_reasoning_prompt("test", [], ["tool"])
        assert "You are a powerful AI assistant" in en_prompt

        # 验证不同语言模板是不同实例
        assert zh_template is not en_template
        assert zh_template.language == "zh"
        assert en_template.language == "en"

    def test_error_handling_integration(self, container: AgentContainer):
        """测试错误处理集成。"""
        template = container.react_template()

        # 测试错误处理提示格式化
        error_msg = "API调用超时"
        history = [
            ReActEntry(ReActStep.THOUGHT, "准备调用API"),
            ReActEntry(ReActStep.ACTION, "api_call"),
        ]

        error_prompt = template.format_error_handling_prompt(error_msg, history)

        assert error_msg in error_prompt
        assert "准备调用API" in error_prompt
        assert "api_call" in error_prompt

    def test_circuit_breaker_integration(self):
        """测试断路器集成。"""
        breaker = CircuitBreaker(failure_threshold=2)

        @breaker
        def test_function(should_fail=True):
            if should_fail:
                raise OpenAIException("测试错误")
            return "成功"

        # 触发两次失败，断路器应该打开
        with pytest.raises(OpenAIException):
            test_function()

        with pytest.raises(OpenAIException):
            test_function()

        assert breaker.state == "OPEN"

        # 第三次调用应该被断路器阻止
        with pytest.raises(OpenAIException, match="断路器打开"):
            test_function()

    def test_container_singleton_behavior(self, container: AgentContainer):
        """测试容器单例行为。"""
        # 多次获取应该返回相同实例
        config1 = container.openai_config()
        config2 = container.openai_config()
        assert config1 is config2

        client1 = container.llm_client()
        client2 = container.llm_client()
        assert client1 is client2

        template1 = container.react_template()
        template2 = container.react_template()
        assert template1 is template2

        # 验证客户端使用相同的配置实例
        assert client1.config is config1

    def test_configuration_inheritance(self):
        """测试配置继承。"""
        # 创建容器时设置配置
        container = create_agent_container(
            openai_api_key="inherit-key",
            openai_model="gpt-4",
            temperature=0.5,
            max_tokens=1000,
        )

        # 验证配置正确传播到所有组件
        config = container.openai_config()
        client = container.llm_client()

        assert config.api_key == "inherit-key"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000

        assert client.config.api_key == "inherit-key"
        assert client.config.model == "gpt-4"
        assert client.config.temperature == 0.5
        assert client.config.max_tokens == 1000

    @pytest.mark.asyncio
    async def test_async_context_manager_integration(self, container: AgentContainer):
        """测试异步上下文管理器集成。"""
        client = container.llm_client()

        # 模拟异步客户端
        mock_async_client = AsyncMock()
        client._async_client = mock_async_client

        async with client as ctx_client:
            assert ctx_client is client

        # 验证异步客户端被正确关闭
        mock_async_client.close.assert_called_once()


class TestReActWorkflow:
    """测试完整ReAct工作流。"""

    def test_complete_react_cycle(self):
        """测试完整ReAct循环。"""
        template = ReActPromptTemplate("zh")

        # 初始查询
        query = "北京今天的天气如何？"
        tools = ["weather_api", "location_service"]
        history = []

        # 1. 生成初始推理提示
        initial_prompt = template.format_reasoning_prompt(query, history, tools)
        assert query in initial_prompt
        assert all(tool in initial_prompt for tool in tools)

        # 2. 模拟LLM第一次响应
        response1 = """Thought: 我需要查询北京今天的天气信息。
Action: weather_api
Action Input: {"location": "北京", "date": "today"}"""

        parsed1 = template.parse_llm_response(response1)
        assert parsed1.has_action
        assert parsed1.action == "weather_api"

        # 3. 添加到历史并格式化观察提示
        history.extend(
            [
                ReActEntry(ReActStep.THOUGHT, parsed1.thought),
                ReActEntry(ReActStep.ACTION, parsed1.action),
            ]
        )

        observation = "北京今天多云，温度15-22度，湿度65%"
        obs_prompt = template.format_observation_prompt(observation, history)
        assert observation in obs_prompt

        # 4. 模拟LLM第二次响应（最终答案）
        response2 = """Thought: 我已经获得了天气信息，现在可以回答用户的问题。
Answer: 根据天气API的信息，北京今天是多云天气，温度在15-22度之间，湿度为65%。建议出门时适当增减衣物。"""

        parsed2 = template.parse_llm_response(response2)
        assert parsed2.has_answer
        assert "多云" in parsed2.answer
        assert "15-22度" in parsed2.answer

        # 5. 验证完整历史
        history.extend(
            [
                ReActEntry(ReActStep.OBSERVATION, observation),
                ReActEntry(ReActStep.THOUGHT, parsed2.thought),
                ReActEntry(ReActStep.ANSWER, parsed2.answer),
            ]
        )

        assert len(history) == 5
        assert history[0].step == ReActStep.THOUGHT
        assert history[1].step == ReActStep.ACTION
        assert history[2].step == ReActStep.OBSERVATION
        assert history[3].step == ReActStep.THOUGHT
        assert history[4].step == ReActStep.ANSWER
