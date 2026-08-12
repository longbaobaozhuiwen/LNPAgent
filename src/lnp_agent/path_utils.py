"""路径安全工具: 强制路由所有文件 I/O 到版本 Results 目录。

v2.3 变更 (基于 v2.2):
- 新增 atomic_write_csv: 原子写入 CSV 文件 (先写 .tmp 再 rename)
- 继承 v2.1 的 sanitize_output_path / sanitize_save_path / sanitize_input_path
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def sanitize_output_path(
    user_path: str | None,
    default_name: str,
    subdir: str = "working_data",
) -> Path:
    """将输出路径强制路由到 RESULTS_DIR/{subdir}/ 下。

    Parameters
    ----------
    user_path : str | None
        LLM 传入的原始路径 (可能为相对路径、含目录的路径等)。
    default_name : str
        user_path 为 None 时使用的默认文件名。
    subdir : str
        RESULTS_DIR 下的子目录 ("working_data" 或 "figures")。

    Returns
    -------
    Path
        绝对路径，保证在 RESULTS_DIR/{subdir}/ 下。
    """
    from lnp_agent.paths import RESULTS_DIR

    if user_path is None:
        target = RESULTS_DIR / subdir / default_name
    else:
        p = Path(user_path)
        filename = p.name  # 仅取文件名，忽略目录部分
        target = RESULTS_DIR / subdir / filename
        if str(target) != str(Path(user_path).resolve()):
            logger.info(f"Path rerouted: {user_path} → {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def sanitize_save_path(
    user_path: str | None,
    default_name: str,
) -> Path:
    """将图表保存路径强制路由到 RESULTS_DIR/figures/ 下。"""
    return sanitize_output_path(user_path, default_name, subdir="figures")


def sanitize_input_path(
    user_path: str,
    subdir: str = "working_data",
) -> Path:
    """将输入文件路径路由到 RESULTS_DIR/{subdir}/ 下。

    绝对路径直接使用 (用于调试灵活性)，相对路径仅取文件名。
    """
    from lnp_agent.paths import RESULTS_DIR

    p = Path(user_path)
    if p.is_absolute():
        return p
    target = RESULTS_DIR / subdir / p.name
    logger.info(f"Input path resolved: {user_path} → {target}")
    return target


def resolve_round_file(
    user_path: str,
    pattern: str | list[str],
    subdir: str = "working_data",
) -> Path:
    """Try to resolve a file path, falling back to glob pattern matching.

    If the exact file doesn't exist, searches for files matching the
    given glob pattern(s) in the same directory and returns the most recent
    match. This handles LLM file name fabrication gracefully.

    Parameters
    ----------
    user_path : str
        Path provided by the LLM (may be wrong filename).
    pattern : str or list[str]
        Glob pattern(s) to search if exact path not found.
        Supports multiple patterns for broader matching.
    subdir : str
        Subdirectory under RESULTS_DIR.

    Returns
    -------
    Path
        Resolved absolute path.
    """
    direct = sanitize_input_path(user_path, subdir)
    if direct.exists():
        return direct

    # Normalize patterns to list
    patterns = [pattern] if isinstance(pattern, str) else pattern
    search_dir = direct.parent

    # Fallback: try each glob pattern
    for pat in patterns:
        candidates = sorted(search_dir.glob(pat), reverse=True)
        if candidates:
            resolved = candidates[0]
            logger.info(f"Fuzzy path resolution: {user_path} → {resolved.name}")
            return resolved

    # No match found, return original (tool will report clear error)
    return direct


def atomic_write_csv(df, target_path: Path) -> Path:
    """Write DataFrame to CSV atomically via temp file + rename.

    Writes to a .csv.tmp file first, then uses os.replace() for atomic
    rename. On crash, the .tmp file may remain but the target file is
    guaranteed to be either the old version or the complete new version.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(str(tmp_path), str(target_path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return target_path
