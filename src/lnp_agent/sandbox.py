"""沙盒安全机制，限制 Agent 文件操作到 AgentWorkspace/。"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path


DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"sudo\s+",
    r"chmod\s+777",
    r"mkfs",
    r"dd\s+if=",
    r">\s*/dev/sd",
    r":\(\)\{.*;\}",
]


class Sandbox:
    """路径沙盒，限制 Agent 操作到指定根目录。"""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve_safe(self, path: str) -> Path:
        """解析路径并验证在沙盒内。"""
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):
            raise PermissionError(f"Path escapes sandbox: {path} -> {target}")
        return target

    # --- 文件操作 ---

    def safe_glob(self, pattern: str, base: str = ".") -> list[str]:
        """在沙盒内执行 glob 搜索。"""
        base_path = self.resolve_safe(base)
        if not base_path.exists():
            return []
        matches = []
        for root, dirs, files in os.walk(base_path):
            for name in files:
                full = Path(root) / name
                rel = full.relative_to(self.root)
                if fnmatch.fnmatch(str(rel), pattern) or fnmatch.fnmatch(name, pattern):
                    matches.append(str(rel))
            for name in dirs:
                full = Path(root) / name
                rel = full.relative_to(self.root)
                if fnmatch.fnmatch(str(rel), pattern):
                    matches.append(str(rel) + "/")
        return sorted(matches)

    def safe_grep(self, pattern: str, path: str = ".", file_glob: str = "*") -> str:
        """在沙盒内执行正则搜索。"""
        search_dir = self.resolve_safe(path)
        if not search_dir.exists():
            return f"Path not found: {path}"

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex: {e}"

        results = []
        for root, _dirs, files in os.walk(search_dir):
            for fname in files:
                if not fnmatch.fnmatch(fname, file_glob):
                    continue
                fpath = Path(root) / fname
                try:
                    text = fpath.read_text(errors="ignore")
                    for i, line in enumerate(text.splitlines(), 1):
                        if regex.search(line):
                            rel = fpath.relative_to(self.root)
                            results.append(f"{rel}:{i}: {line.rstrip()}")
                            if len(results) >= 100:
                                return "\n".join(results) + "\n... (truncated)"
                except (OSError, UnicodeDecodeError):
                    continue

        return "\n".join(results) if results else f"No matches for '{pattern}'"

    def safe_read(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        """读取文件内容 (带行号)。"""
        target = self.resolve_safe(path)
        if not target.exists():
            return f"File not found: {path}"
        if not target.is_file():
            return f"Not a file: {path}"

        try:
            lines = target.read_text(errors="replace").splitlines()
            selected = lines[offset : offset + limit]
            numbered = [f"{offset + i + 1}\t{line}" for i, line in enumerate(selected)]
            return "\n".join(numbered)
        except OSError as e:
            return f"Error reading file: {e}"

    def safe_write(self, path: str, content: str) -> str:
        """写入文件。自动创建父目录。"""
        target = self.resolve_safe(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Written {len(content)} bytes to {path}"

    def safe_edit(
        self, path: str, old_text: str, new_text: str, replace_all: bool = False
    ) -> str:
        """精确字符串替换编辑。"""
        target = self.resolve_safe(path)
        if not target.exists():
            return f"File not found: {path}"

        content = target.read_text()
        count = content.count(old_text)

        if count == 0:
            return f"old_text not found in {path}"
        if count > 1 and not replace_all:
            return (
                f"old_text found {count} times in {path}. "
                "Use replace_all=true or provide more context to make it unique."
            )

        new_content = content.replace(old_text, new_text) if replace_all else content.replace(old_text, new_text, 1)
        target.write_text(new_content)
        return f"Replaced {count} occurrence(s) in {path}"

    def safe_bash(self, command: str, timeout: int = 120) -> dict:
        """在沙盒内执行 bash 命令。"""
        # 危险命令检查
        for pat in DANGEROUS_PATTERNS:
            if re.search(pat, command):
                return {
                    "stdout": "",
                    "stderr": f"Blocked: dangerous command pattern matched: {pat}",
                    "returncode": 126,
                }

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PWD": str(self.root)},
            )
            return {
                "stdout": result.stdout[:10000],  # 截断防止超长输出
                "stderr": result.stderr[:5000],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "returncode": 124,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Execution error: {e}",
                "returncode": 1,
            }
