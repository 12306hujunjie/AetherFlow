"""ReAct提示模板系统，支持多语言和自定义模板。"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ..exceptions import PromptTemplateException


class ReActStep(Enum):
    """ReAct推理步骤枚举。"""

    THOUGHT = "Thought"
    ACTION = "Action"
    OBSERVATION = "Observation"
    ANSWER = "Answer"


@dataclass
class ReActEntry:
    """ReAct历史条目。"""

    step: ReActStep
    content: str
    metadata: dict[str, Any] | None = None


class ReActResponse(BaseModel):
    """ReAct LLM响应解析结果。"""

    thought: str | None = Field(None, description="当前思考内容")
    action: str | None = Field(None, description="要执行的行动")
    action_input: dict[str, Any] | None = Field(None, description="行动的输入参数")
    answer: str | None = Field(None, description="最终答案")
    is_final: bool = Field(False, description="是否为最终回答")

    @property
    def has_action(self) -> bool:
        """是否包含行动。"""
        return self.action is not None

    @property
    def has_answer(self) -> bool:
        """是否包含最终答案。"""
        return self.answer is not None and self.is_final


class ReActPromptTemplate:
    """ReAct特定的提示模板管理器。

    支持多种提示格式：
    - 基础推理模板
    - 工具调用模板
    - 观察结果模板
    - 最终答案模板
    """

    # 基础ReAct提示模板
    BASE_REASONING_TEMPLATE = """你是一个强大的AI助手，能够通过推理和行动来解决问题。

请按照以下格式进行推理：

Thought: [你的思考过程，分析当前情况和下一步该做什么]
Action: [如果需要采取行动，描述具体的行动]
Observation: [行动的结果和观察到的信息]

如果你有最终答案，请使用：
Answer: [你的最终答案]

问题: {query}

可用工具:
{tools}

{history}

现在开始推理："""

    # 工具调用格式模板
    TOOL_CALL_TEMPLATE = """你需要使用工具来获取信息或执行操作。

可用工具:
{tools}

使用工具的格式：
Action: tool_name
Action Input: {{"param1": "value1", "param2": "value2"}}

问题: {query}
{history}

Thought:"""

    # 观察结果处理模板
    OBSERVATION_TEMPLATE = """基于以下观察结果，继续你的推理：

Observation: {observation}

{history}

Thought:"""

    # 最终答案模板
    FINAL_ANSWER_TEMPLATE = """基于你的推理过程，请提供最终答案：

{history}

Answer:"""

    # 错误处理模板
    ERROR_HANDLING_TEMPLATE = """执行过程中出现了错误，请重新思考：

错误信息: {error}
{history}

Thought:"""

    def __init__(
        self, language: str = "zh", custom_templates: dict[str, str] | None = None
    ):
        """初始化ReAct提示模板。

        Args:
            language: 语言代码，支持 'zh', 'en' 等
            custom_templates: 自定义模板字典
        """
        self.language = language
        self.custom_templates = custom_templates or {}
        self._load_language_templates()

    def _load_language_templates(self) -> None:
        """加载语言特定的模板。"""
        if self.language == "en":
            self.BASE_REASONING_TEMPLATE = """You are a powerful AI assistant that can solve problems through reasoning and actions.

Please follow this format for reasoning:

Thought: [Your thinking process, analyze the current situation and what to do next]
Action: [If you need to take action, describe the specific action]
Observation: [The result of the action and observed information]

If you have a final answer, use:
Answer: [Your final answer]

Question: {query}

Available tools:
{tools}

{history}

Start reasoning now:"""

            self.TOOL_CALL_TEMPLATE = """You need to use tools to get information or perform operations.

Available tools:
{tools}

Format for using tools:
Action: tool_name
Action Input: {{"param1": "value1", "param2": "value2"}}

Question: {query}
{history}

Thought:"""

    def format_reasoning_prompt(
        self,
        query: str,
        history: list[ReActEntry],
        available_tools: list[str],
    ) -> str:
        """格式化推理阶段提示。

        Args:
            query: 用户查询
            history: 推理历史
            available_tools: 可用工具列表

        Returns:
            格式化的提示文本
        """
        tools_text = self._format_tools(available_tools)
        history_text = self._format_history(history)

        template = self.custom_templates.get("reasoning", self.BASE_REASONING_TEMPLATE)

        return template.format(
            query=query,
            tools=tools_text,
            history=history_text,
        )

    def format_tool_call_prompt(
        self,
        query: str,
        history: list[ReActEntry],
        available_tools: list[str],
    ) -> str:
        """格式化工具调用提示。

        Args:
            query: 用户查询
            history: 推理历史
            available_tools: 可用工具列表

        Returns:
            格式化的提示文本
        """
        tools_text = self._format_tools(available_tools)
        history_text = self._format_history(history)

        template = self.custom_templates.get("tool_call", self.TOOL_CALL_TEMPLATE)

        return template.format(
            query=query,
            tools=tools_text,
            history=history_text,
        )

    def format_observation_prompt(
        self,
        observation: str,
        history: list[ReActEntry],
    ) -> str:
        """格式化观察阶段提示。

        Args:
            observation: 观察结果
            history: 推理历史

        Returns:
            格式化的提示文本
        """
        history_text = self._format_history(history)

        template = self.custom_templates.get("observation", self.OBSERVATION_TEMPLATE)

        return template.format(
            observation=observation,
            history=history_text,
        )

    def format_final_answer_prompt(
        self,
        history: list[ReActEntry],
    ) -> str:
        """格式化最终答案提示。

        Args:
            history: 推理历史

        Returns:
            格式化的提示文本
        """
        history_text = self._format_history(history)

        template = self.custom_templates.get("final_answer", self.FINAL_ANSWER_TEMPLATE)

        return template.format(history=history_text)

    def format_error_handling_prompt(
        self,
        error: str,
        history: list[ReActEntry],
    ) -> str:
        """格式化错误处理提示。

        Args:
            error: 错误信息
            history: 推理历史

        Returns:
            格式化的提示文本
        """
        history_text = self._format_history(history)

        template = self.custom_templates.get(
            "error_handling", self.ERROR_HANDLING_TEMPLATE
        )

        return template.format(
            error=error,
            history=history_text,
        )

    def parse_llm_response(self, response: str) -> ReActResponse:
        """解析LLM响应，提取思考和行动。

        Args:
            response: LLM的原始响应

        Returns:
            解析后的ReAct响应对象

        Raises:
            PromptTemplateException: 响应格式解析失败
        """
        try:
            result = ReActResponse()

            # 解析Thought
            thought_match = re.search(
                r"Thought:\s*(.+?)(?=\n\s*(?:Action|Answer)|$)",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if thought_match:
                result.thought = thought_match.group(1).strip()

            # 解析Action
            action_match = re.search(
                r"Action:\s*(.+?)(?=\n\s*(?:Action Input|Observation|Answer)|$)",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if action_match:
                result.action = action_match.group(1).strip()

            # 解析Action Input
            action_input_match = re.search(
                r"Action Input:\s*(.+?)(?=\n\s*(?:Observation|Answer)|$)",
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if action_input_match:
                action_input_text = action_input_match.group(1).strip()
                try:
                    # 尝试解析JSON格式的输入
                    import json

                    result.action_input = json.loads(action_input_text)
                except json.JSONDecodeError:
                    # 如果不是JSON，就当作普通字符串处理
                    result.action_input = {"input": action_input_text}

            # 解析Answer
            answer_match = re.search(
                r"Answer:\s*(.+)", response, re.DOTALL | re.IGNORECASE
            )
            if answer_match:
                result.answer = answer_match.group(1).strip()
                result.is_final = True

            return result

        except Exception as e:
            raise PromptTemplateException(
                f"解析LLM响应失败: {e}", template_name="llm_response_parser"
            )

    def _format_tools(self, tools: list[str]) -> str:
        """格式化工具列表。

        Args:
            tools: 工具名称列表

        Returns:
            格式化的工具描述
        """
        if not tools:
            return "无可用工具"

        tool_list = "\n".join(f"- {tool}" for tool in tools)
        return f"可用工具:\n{tool_list}"

    def _format_history(self, history: list[ReActEntry]) -> str:
        """格式化推理历史。

        Args:
            history: 推理历史条目列表

        Returns:
            格式化的历史文本
        """
        if not history:
            return ""

        formatted_entries = []
        for entry in history:
            formatted_entries.append(f"{entry.step.value}: {entry.content}")

        return "\n".join(formatted_entries)

    def create_system_message(self, role_description: str | None = None) -> str:
        """创建系统消息。

        Args:
            role_description: 角色描述，可选

        Returns:
            系统消息内容
        """
        default_role = (
            "你是一个强大的AI助手，擅长通过结构化推理来解决复杂问题。"
            "你会仔细思考，必要时使用工具，最后给出准确的答案。"
        )

        if self.language == "en":
            default_role = (
                "You are a powerful AI assistant skilled at solving complex problems "
                "through structured reasoning. You think carefully, use tools when necessary, "
                "and provide accurate answers."
            )

        role = role_description or default_role

        return f"System: {role}"


# 预定义的模板集合
REACT_TEMPLATES = {
    "zh": ReActPromptTemplate("zh"),
    "en": ReActPromptTemplate("en"),
}


def get_react_template(language: str = "zh") -> ReActPromptTemplate:
    """获取指定语言的ReAct模板。

    Args:
        language: 语言代码

    Returns:
        ReAct提示模板实例
    """
    return REACT_TEMPLATES.get(language, REACT_TEMPLATES["zh"])
