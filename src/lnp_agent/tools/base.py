"""Tool 基类定义。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolDefinition:
    """工具定义，注册到 LLM 的 function calling schema。"""

    name: str
    description: str
    parameters: dict[str, dict]
    required: list[str]

    def to_function_schema(self) -> dict:
        """转换为 function calling JSON schema 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


@dataclass
class ToolResult:
    """工具执行结果。"""

    success: bool
    output: str
    error: str | None = None

    def to_message(self, tool_call_id: str = "") -> dict:
        """转换为 tool role message。"""
        content = self.output if self.success else f"ERROR: {self.error}\n{self.output}"
        msg = {"role": "tool", "content": content}
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        return msg


class BaseTool:
    """工具基类。"""

    definition: ToolDefinition

    def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError
