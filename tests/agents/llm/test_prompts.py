"""测试ReAct提示模板系统。"""

from agents.llm.prompts import (
    ReActEntry,
    ReActPromptTemplate,
    ReActResponse,
    ReActStep,
    get_react_template,
)


class TestReActPromptTemplate:
    """测试ReAct提示模板类。"""

    def test_template_initialization_chinese(self):
        """测试中文模板初始化。"""
        template = ReActPromptTemplate(language="zh")

        assert template.language == "zh"
        assert "你是一个强大的AI助手" in template.BASE_REASONING_TEMPLATE
        assert "Thought:" in template.BASE_REASONING_TEMPLATE
        assert "Action:" in template.BASE_REASONING_TEMPLATE

    def test_template_initialization_english(self):
        """测试英文模板初始化。"""
        template = ReActPromptTemplate(language="en")

        assert template.language == "en"
        assert "You are a powerful AI assistant" in template.BASE_REASONING_TEMPLATE
        assert "Thought:" in template.BASE_REASONING_TEMPLATE
        assert "Action:" in template.BASE_REASONING_TEMPLATE

    def test_custom_templates(self):
        """测试自定义模板。"""
        custom_templates = {
            "reasoning": "Custom reasoning template: {query}",
            "tool_call": "Custom tool template: {tools}",
        }

        template = ReActPromptTemplate(language="zh", custom_templates=custom_templates)

        assert template.custom_templates == custom_templates

    def test_format_reasoning_prompt(self):
        """测试格式化推理提示。"""
        template = ReActPromptTemplate(language="zh")

        query = "什么是人工智能？"
        history = [
            ReActEntry(ReActStep.THOUGHT, "我需要解释人工智能的概念"),
            ReActEntry(ReActStep.ACTION, "search_knowledge_base"),
        ]
        tools = ["search_knowledge_base", "calculate", "web_search"]

        prompt = template.format_reasoning_prompt(query, history, tools)

        assert query in prompt
        assert "search_knowledge_base" in prompt
        assert "calculate" in prompt
        assert "web_search" in prompt
        assert "Thought: 我需要解释人工智能的概念" in prompt
        assert "Action: search_knowledge_base" in prompt

    def test_format_tool_call_prompt(self):
        """测试格式化工具调用提示。"""
        template = ReActPromptTemplate(language="zh")

        query = "今天天气如何？"
        history = [ReActEntry(ReActStep.THOUGHT, "我需要查询天气信息")]
        tools = ["weather_api", "location_service"]

        prompt = template.format_tool_call_prompt(query, history, tools)

        assert query in prompt
        assert "weather_api" in prompt
        assert "location_service" in prompt
        assert "我需要查询天气信息" in prompt

    def test_format_observation_prompt(self):
        """测试格式化观察提示。"""
        template = ReActPromptTemplate(language="zh")

        observation = "天气查询结果：今天晴天，温度25度"
        history = [
            ReActEntry(ReActStep.THOUGHT, "我需要查询天气"),
            ReActEntry(ReActStep.ACTION, "weather_api"),
        ]

        prompt = template.format_observation_prompt(observation, history)

        assert observation in prompt
        assert "我需要查询天气" in prompt
        assert "weather_api" in prompt

    def test_format_final_answer_prompt(self):
        """测试格式化最终答案提示。"""
        template = ReActPromptTemplate(language="zh")

        history = [
            ReActEntry(ReActStep.THOUGHT, "分析完成"),
            ReActEntry(ReActStep.ACTION, "search"),
            ReActEntry(ReActStep.OBSERVATION, "找到相关信息"),
        ]

        prompt = template.format_final_answer_prompt(history)

        assert "分析完成" in prompt
        assert "search" in prompt
        assert "找到相关信息" in prompt

    def test_format_error_handling_prompt(self):
        """测试格式化错误处理提示。"""
        template = ReActPromptTemplate(language="zh")

        error = "API调用失败：网络连接超时"
        history = [ReActEntry(ReActStep.THOUGHT, "准备调用API")]

        prompt = template.format_error_handling_prompt(error, history)

        assert error in prompt
        assert "准备调用API" in prompt

    def test_parse_llm_response_thought_only(self):
        """测试解析只包含思考的响应。"""
        template = ReActPromptTemplate(language="zh")

        response = "Thought: 我需要先分析这个问题的核心。"

        parsed = template.parse_llm_response(response)

        assert parsed.thought == "我需要先分析这个问题的核心。"
        assert parsed.action is None
        assert parsed.action_input is None
        assert parsed.answer is None
        assert parsed.is_final is False
        assert parsed.has_action is False
        assert parsed.has_answer is False

    def test_parse_llm_response_with_action(self):
        """测试解析包含行动的响应。"""
        template = ReActPromptTemplate(language="zh")

        response = """Thought: 我需要搜索相关信息。
Action: web_search
Action Input: {"query": "人工智能发展历史", "max_results": 5}"""

        parsed = template.parse_llm_response(response)

        assert parsed.thought == "我需要搜索相关信息。"
        assert parsed.action == "web_search"
        assert parsed.action_input == {"query": "人工智能发展历史", "max_results": 5}
        assert parsed.answer is None
        assert parsed.is_final is False
        assert parsed.has_action is True
        assert parsed.has_answer is False

    def test_parse_llm_response_with_simple_action_input(self):
        """测试解析简单文本格式的行动输入。"""
        template = ReActPromptTemplate(language="zh")

        response = """Thought: 我需要计算这个数学问题。
Action: calculator
Action Input: 2 + 2 * 3"""

        parsed = template.parse_llm_response(response)

        assert parsed.action == "calculator"
        assert parsed.action_input == {"input": "2 + 2 * 3"}

    def test_parse_llm_response_with_answer(self):
        """测试解析包含最终答案的响应。"""
        template = ReActPromptTemplate(language="zh")

        response = """Thought: 基于我的分析，我现在可以给出答案。
Answer: 人工智能是一门研究如何让计算机模拟人类智能的科学技术。"""

        parsed = template.parse_llm_response(response)

        assert parsed.thought == "基于我的分析，我现在可以给出答案。"
        assert parsed.answer == "人工智能是一门研究如何让计算机模拟人类智能的科学技术。"
        assert parsed.is_final is True
        assert parsed.has_answer is True

    def test_parse_llm_response_malformed(self):
        """测试解析格式错误的响应。"""
        template = ReActPromptTemplate(language="zh")

        # 这种情况下应该能正常解析，只是某些字段为空
        response = "这是一个没有格式的普通响应。"

        parsed = template.parse_llm_response(response)

        assert parsed.thought is None
        assert parsed.action is None
        assert parsed.answer is None
        assert parsed.is_final is False

    def test_format_tools_with_tools(self):
        """测试格式化工具列表。"""
        template = ReActPromptTemplate(language="zh")
        tools = ["search", "calculator", "weather"]

        result = template._format_tools(tools)

        assert "search" in result
        assert "calculator" in result
        assert "weather" in result
        assert "可用工具:" in result

    def test_format_tools_empty(self):
        """测试格式化空工具列表。"""
        template = ReActPromptTemplate(language="zh")

        result = template._format_tools([])

        assert result == "无可用工具"

    def test_format_history_with_entries(self):
        """测试格式化非空历史。"""
        template = ReActPromptTemplate(language="zh")
        history = [
            ReActEntry(ReActStep.THOUGHT, "分析问题"),
            ReActEntry(ReActStep.ACTION, "搜索"),
            ReActEntry(ReActStep.OBSERVATION, "找到结果"),
        ]

        result = template._format_history(history)

        assert "Thought: 分析问题" in result
        assert "Action: 搜索" in result
        assert "Observation: 找到结果" in result

    def test_format_history_empty(self):
        """测试格式化空历史。"""
        template = ReActPromptTemplate(language="zh")

        result = template._format_history([])

        assert result == ""

    def test_create_system_message_default(self):
        """测试创建默认系统消息。"""
        template = ReActPromptTemplate(language="zh")

        message = template.create_system_message()

        assert message.startswith("System:")
        assert "AI助手" in message
        assert "结构化推理" in message

    def test_create_system_message_english(self):
        """测试创建英文系统消息。"""
        template = ReActPromptTemplate(language="en")

        message = template.create_system_message()

        assert message.startswith("System:")
        assert "AI assistant" in message
        assert "structured reasoning" in message

    def test_create_system_message_custom(self):
        """测试创建自定义系统消息。"""
        template = ReActPromptTemplate(language="zh")
        custom_role = "你是一个专业的数学老师。"

        message = template.create_system_message(custom_role)

        assert message == f"System: {custom_role}"


class TestReActResponse:
    """测试ReActResponse模型。"""

    def test_response_creation(self):
        """测试响应对象创建。"""
        response = ReActResponse(
            thought="我需要思考",
            action="search",
            action_input={"query": "test"},
            answer="这是答案",
            is_final=True,
        )

        assert response.thought == "我需要思考"
        assert response.action == "search"
        assert response.action_input == {"query": "test"}
        assert response.answer == "这是答案"
        assert response.is_final is True
        assert response.has_action is True
        assert response.has_answer is True

    def test_response_defaults(self):
        """测试响应对象默认值。"""
        response = ReActResponse()

        assert response.thought is None
        assert response.action is None
        assert response.action_input is None
        assert response.answer is None
        assert response.is_final is False
        assert response.has_action is False
        assert response.has_answer is False

    def test_has_action_property(self):
        """测试has_action属性。"""
        response1 = ReActResponse(action="search")
        response2 = ReActResponse()

        assert response1.has_action is True
        assert response2.has_action is False

    def test_has_answer_property(self):
        """测试has_answer属性。"""
        response1 = ReActResponse(answer="答案", is_final=True)
        response2 = ReActResponse(answer="答案", is_final=False)
        response3 = ReActResponse()

        assert response1.has_answer is True
        assert response2.has_answer is False  # 需要is_final=True
        assert response3.has_answer is False


class TestReActEntry:
    """测试ReActEntry数据类。"""

    def test_entry_creation(self):
        """测试条目创建。"""
        entry = ReActEntry(
            step=ReActStep.THOUGHT, content="这是思考内容", metadata={"confidence": 0.8}
        )

        assert entry.step == ReActStep.THOUGHT
        assert entry.content == "这是思考内容"
        assert entry.metadata == {"confidence": 0.8}

    def test_entry_without_metadata(self):
        """测试无元数据的条目创建。"""
        entry = ReActEntry(step=ReActStep.ACTION, content="执行搜索")

        assert entry.step == ReActStep.ACTION
        assert entry.content == "执行搜索"
        assert entry.metadata is None


class TestReActStep:
    """测试ReActStep枚举。"""

    def test_step_values(self):
        """测试步骤值。"""
        assert ReActStep.THOUGHT.value == "Thought"
        assert ReActStep.ACTION.value == "Action"
        assert ReActStep.OBSERVATION.value == "Observation"
        assert ReActStep.ANSWER.value == "Answer"


class TestGetReActTemplate:
    """测试获取ReAct模板函数。"""

    def test_get_chinese_template(self):
        """测试获取中文模板。"""
        template = get_react_template("zh")

        assert isinstance(template, ReActPromptTemplate)
        assert template.language == "zh"

    def test_get_english_template(self):
        """测试获取英文模板。"""
        template = get_react_template("en")

        assert isinstance(template, ReActPromptTemplate)
        assert template.language == "en"

    def test_get_default_template(self):
        """测试获取默认模板。"""
        template = get_react_template()

        assert isinstance(template, ReActPromptTemplate)
        assert template.language == "zh"  # 默认中文

    def test_get_unsupported_language(self):
        """测试获取不支持的语言模板。"""
        template = get_react_template("fr")  # 法语不支持

        assert isinstance(template, ReActPromptTemplate)
        assert template.language == "zh"  # 应该返回默认中文模板
