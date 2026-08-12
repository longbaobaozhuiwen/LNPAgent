"""6 个标准系统工具实现。"""

from __future__ import annotations

import json

from lnp_agent.sandbox import Sandbox
from lnp_agent.tools.base import BaseTool, ToolDefinition, ToolResult


class BashExecutor(BaseTool):
    """在沙盒内执行 shell 命令。"""

    definition = ToolDefinition(
        name="bash_executor",
        description=(
            "Execute a shell command in the agent workspace. "
            "Use for running Python scripts, installing packages, checking file contents, etc. "
            "The command runs with the workspace as the current directory. "
            "Dangerous commands (sudo, rm -rf /, etc.) are blocked."
        ),
        parameters={
            "command": {
                "type": "string",
                "description": "Shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120)",
            },
        },
        required=["command"],
    )

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def execute(self, command: str, timeout: int = 120) -> ToolResult:
        result = self.sandbox.safe_bash(command, timeout)
        output = result["stdout"]
        if result["stderr"]:
            output += f"\nSTDERR:\n{result['stderr']}"
        if result["returncode"] != 0:
            return ToolResult(success=False, output=output, error=f"Exit code: {result['returncode']}")
        return ToolResult(success=True, output=output)


class GlobTool(BaseTool):
    """在沙盒内匹配文件路径。"""

    definition = ToolDefinition(
        name="glob",
        description=(
            "Search for files matching a glob pattern in the workspace. "
            "Supports patterns like '**/*.py', 'data/*.csv', etc. "
            "Returns matching file paths."
        ),
        parameters={
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g. '**/*.py', 'data/*.csv')",
            },
            "path": {
                "type": "string",
                "description": "Base directory for search (default '.')",
            },
        },
        required=["pattern"],
    )

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def execute(self, pattern: str, path: str = ".") -> ToolResult:
        matches = self.sandbox.safe_glob(pattern, path)
        if matches:
            return ToolResult(success=True, output="\n".join(matches))
        return ToolResult(success=True, output=f"No files matching '{pattern}' in {path}")


class GrepTool(BaseTool):
    """正则搜索文件内容。"""

    definition = ToolDefinition(
        name="grep",
        description=(
            "Search file contents using a regular expression pattern. "
            "Returns matching lines with file name, line number, and content."
        ),
        parameters={
            "pattern": {
                "type": "string",
                "description": "Regular expression pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search in (default '.')",
            },
            "file_glob": {
                "type": "string",
                "description": "File pattern to filter (e.g. '*.py', default '*')",
            },
        },
        required=["pattern"],
    )

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def execute(self, pattern: str, path: str = ".", file_glob: str = "*") -> ToolResult:
        result = self.sandbox.safe_grep(pattern, path, file_glob)
        return ToolResult(success=True, output=result)


class ReadTool(BaseTool):
    """读取文件内容。"""

    definition = ToolDefinition(
        name="read",
        description=(
            "Read the contents of a file. Returns content with line numbers. "
            "Use offset and limit to read specific line ranges for large files."
        ),
        parameters={
            "file_path": {
                "type": "string",
                "description": "Path to the file (relative to workspace)",
            },
            "offset": {
                "type": "integer",
                "description": "Starting line number (0-indexed, default 0)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read (default 2000)",
            },
        },
        required=["file_path"],
    )

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def execute(self, file_path: str, offset: int = 0, limit: int = 2000) -> ToolResult:
        result = self.sandbox.safe_read(file_path, offset, limit)
        if result.startswith("File not found") or result.startswith("Not a file"):
            return ToolResult(success=False, output=result, error=result)
        return ToolResult(success=True, output=result)


class WriteTool(BaseTool):
    """创建或覆盖文件。"""

    definition = ToolDefinition(
        name="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does. Parent directories are created automatically."
        ),
        parameters={
            "file_path": {
                "type": "string",
                "description": "Path for the file (relative to workspace)",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        required=["file_path", "content"],
    )

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def execute(self, file_path: str, content: str) -> ToolResult:
        result = self.sandbox.safe_write(file_path, content)
        return ToolResult(success=True, output=result)


class EditTool(BaseTool):
    """精确字符串替换编辑。"""

    definition = ToolDefinition(
        name="edit",
        description=(
            "Replace exact text in a file. Finds old_text and replaces it with new_text. "
            "The old_text must be unique in the file (unless replace_all=true). "
            "Use this for bug fixes, parameter changes, or small code modifications."
        ),
        parameters={
            "file_path": {
                "type": "string",
                "description": "Path to the file (relative to workspace)",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to find and replace",
            },
            "new_text": {
                "type": "string",
                "description": "Text to replace with",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences if true (default false)",
            },
        },
        required=["file_path", "old_text", "new_text"],
    )

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def execute(
        self, file_path: str, old_text: str, new_text: str, replace_all: bool = False
    ) -> ToolResult:
        result = self.sandbox.safe_edit(file_path, old_text, new_text, replace_all)
        if "not found" in result.lower() or "found " in result.lower() and "occurrence" not in result.lower():
            return ToolResult(success=False, output=result, error=result)
        return ToolResult(success=True, output=result)
