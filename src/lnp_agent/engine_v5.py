"""5 状态循环状态机 Agent 引擎 v5 (v2.6 — REPORTING 全域反冗余)。

v2.2 变更 (基于 v2.0 engine_v4):
1. 新增 WET_LAB_TESTING 和 RETRAINING 状态
2. 循环转移: REPORTING → WET_LAB_TESTING → RETRAINING → PREDICTION (循环)
3. 轮次计数器 (max_rounds=3): Round 3 REPORTING 后进入 COMPLETE
4. ConversationManager 适配: 轮次感知的压缩和提示
5. 工具数: 19 (16 + run_wet_lab_experiment + retrain_lnp_predictors + plot_active_learning_trajectory)

v2.3 变更:
1. PREDICTION 和 FILTERING 状态提示增加明确的路径和参数指引
2. WET_LAB_TESTING 候选数从 20 提升至 40 (20 exploitation + 20 exploration)
3. 轮次上下文中累积数据量从 20×(round-1) 调整为 40×(round-1)

v2.4 变更:
1. FORCED_PARAMS: 强制覆盖关键工具参数 (路径+top_n)
2. _build_file_context(): 动态扫描目录生成文件列表
3. _required_plots_called(): 最终轮强制要求轨迹图

v2.5 变更:
1. 反冗余状态锁: 工具签名去重，防止同一状态内重复调用
2. 状态完成自动推进: 检测必需工具完成后提示 LLM 进入下一状态
3. tool_call_log 添加 signature 字段
4. _resolve_tool_defaults 添加 acquisition_method 和 kappa 参数

v2.6 变更:
1. REPORTING 语义签名: 排除 save_path 等输出路径参数
2. REPORTING 一次调用锁定: plot_active_learning_trajectory 每 round 只允许一次
3. STATE_REQUIRED_TOOLS 扩展: 添加 plot_active_learning_trajectory
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from lnp_agent.tools.base import ToolResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Event System (继承自 v2.0)
# ═══════════════════════════════════════════════════════════

class EventType(str, Enum):
    ITERATION_START = "iteration_start"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PERMISSION_CHECK = "permission_check"
    STATE_TRANSITION = "state_transition"
    TASK_COMPLETE = "task_complete"
    ERROR = "error"
    CONVERSATION_TRIMMED = "conversation_trimmed"
    ROUND_START = "round_start"       # v2.2: 新增
    ROUND_COMPLETE = "round_complete"  # v2.2: 新增


@dataclass
class AgentEvent:
    type: EventType
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """轻量同步事件总线。"""

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}

    def subscribe(self, event_type: str | EventType, handler: Callable) -> None:
        key = event_type.value if isinstance(event_type, EventType) else event_type
        self._handlers.setdefault(key, []).append(handler)

    def emit(self, event: AgentEvent) -> None:
        key = event.type.value
        for handler in self._handlers.get(key, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def clear(self) -> None:
        self._handlers.clear()


# ═══════════════════════════════════════════════════════════
# Permission System (继承自 v2.0, 新增 3 个工具)
# ═══════════════════════════════════════════════════════════

class Permission(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


class PermissionManager:
    """工具级权限控制。"""

    DEFAULT_RULES: dict[str, Permission] = {
        "check_compliance": Permission.ALLOW,
        "predict_lnp_performance": Permission.ALLOW,
        "query_pareto_front": Permission.ALLOW,
        "enumerate_missing_cells": Permission.ALLOW,
        "glob": Permission.ALLOW,
        "grep": Permission.ALLOW,
        "read": Permission.ALLOW,
        "bash_executor": Permission.CONFIRM,
        "write": Permission.CONFIRM,
        "edit": Permission.CONFIRM,
        "generate_virtual_library": Permission.ALLOW,
        "batch_predict_lnp": Permission.ALLOW,
        "filter_exploitation_batch": Permission.ALLOW,
        "filter_exploration_batch": Permission.ALLOW,
        "plot_pareto_front": Permission.ALLOW,
        "plot_chemical_space_umap": Permission.ALLOW,
        "run_wet_lab_experiment": Permission.ALLOW,
        "retrain_lnp_predictors": Permission.ALLOW,
        "plot_active_learning_trajectory": Permission.ALLOW,
    }

    def __init__(self, rules: dict[str, str | Permission] | None = None):
        self.rules: dict[str, Permission] = dict(self.DEFAULT_RULES)
        if rules:
            for name, perm in rules.items():
                self.rules[name] = Permission(perm)

    def check(self, tool_name: str, args: dict | None = None) -> Permission:
        return self.rules.get(tool_name, Permission.CONFIRM)

    def configure(self, config: dict) -> None:
        for tool_name, perm_str in config.items():
            self.rules[tool_name] = Permission(perm_str)


# ═══════════════════════════════════════════════════════════
# State Machine Definition (v2.2: 循环状态机)
# ═══════════════════════════════════════════════════════════

class SOPState(str, Enum):
    EXTRACTION = "extraction"
    PREDICTION = "prediction"
    FILTERING = "filtering"
    REPORTING = "reporting"
    WET_LAB_TESTING = "wet_lab_testing"   # v2.2 新增
    RETRAINING = "retraining"             # v2.2 新增
    COMPLETE = "complete"


STATE_ALLOWED_TOOLS: dict[SOPState, set[str]] = {
    SOPState.EXTRACTION: {
        "glob", "grep", "read",
        "enumerate_missing_cells",
        "generate_virtual_library",
        "check_compliance",
    },
    SOPState.PREDICTION: {
        "batch_predict_lnp",
    },
    SOPState.FILTERING: {
        "filter_exploitation_batch",
        "filter_exploration_batch",
    },
    SOPState.REPORTING: {
        "plot_pareto_front",
        "plot_chemical_space_umap",
        "plot_active_learning_trajectory",
        "write",
        "bash_executor",
    },
    SOPState.WET_LAB_TESTING: {
        "run_wet_lab_experiment",
    },
    SOPState.RETRAINING: {
        "retrain_lnp_predictors",
    },
}

STATE_TRANSITIONS: dict[SOPState, list[SOPState]] = {
    SOPState.EXTRACTION: [SOPState.PREDICTION],
    SOPState.PREDICTION: [SOPState.FILTERING],
    SOPState.FILTERING: [SOPState.REPORTING],
    SOPState.REPORTING: [SOPState.WET_LAB_TESTING, SOPState.COMPLETE],
    SOPState.WET_LAB_TESTING: [SOPState.RETRAINING],
    SOPState.RETRAINING: [SOPState.PREDICTION],
    SOPState.COMPLETE: [],
}

STATE_DESCRIPTIONS: dict[SOPState, str] = {
    SOPState.EXTRACTION: "Extract building blocks and generate the virtual library (ugi3cr mode)",
    SOPState.PREDICTION: "Run batch predictions on the virtual library",
    SOPState.FILTERING: "Apply exploitation and exploration filters to select top candidates",
    SOPState.REPORTING: "Generate plots (Pareto front, chemical space, trajectory)",
    SOPState.WET_LAB_TESTING: "Submit 40 candidates (20 exploitation + 20 exploration) for simulated wet-lab experiment",
    SOPState.RETRAINING: "Retrain prediction models with new experimental data",
}

# v2.4: 强制覆盖的参数 — 无论 LLM 传什么都覆盖为标准命名
FORCED_PARAMS: dict[str, set[str]] = {
    "batch_predict_lnp": {"formulations_path", "output_path"},
    "filter_exploitation_batch": {"predictions_path", "output_path", "top_n"},
    "filter_exploration_batch": {"predictions_path", "output_path", "top_n"},
    "run_wet_lab_experiment": {"exploitation_path", "exploration_path", "round_number"},
    "retrain_lnp_predictors": {"wet_lab_results_path"},
    "plot_pareto_front": {"predictions_path", "exploitation_path", "exploration_path"},
    "plot_chemical_space_umap": {"predictions_path", "exploitation_path", "exploration_path"},
}

# v2.5: 状态必需工具集 — 完成后自动提示 LLM 推进
STATE_REQUIRED_TOOLS: dict[SOPState, set[str]] = {
    SOPState.FILTERING: {"filter_exploitation_batch", "filter_exploration_batch"},
    SOPState.REPORTING: {
        "plot_pareto_front",
        "plot_chemical_space_umap",
        "plot_active_learning_trajectory",  # v2.6: 新增
    },
}

# v2.6: REPORTING 工具签名排除参数（不影响逻辑的输出路径参数）
REPORTING_TOOL_IGNORE_ARGS: dict[str, set[str]] = {
    "plot_pareto_front": {"save_path"},
    "plot_chemical_space_umap": {"save_path"},
    "plot_active_learning_trajectory": {"save_path"},
}

# v2.6: 每个 (state, round) 只允许调用一次的工具
REPORTING_SINGLE_CALL_TOOLS: set[str] = {
    "plot_active_learning_trajectory",
}


# ═══════════════════════════════════════════════════════════
# Conversation Manager v5 (轮次感知)
# ═══════════════════════════════════════════════════════════

class ConversationManagerV5:
    """v2.2 对话管理器: 轮次感知的滑动窗口 + 状态压缩。"""

    def __init__(
        self,
        system_prompt: str,
        max_rounds: int = 6,
        tokenizer_encode_fn: Callable | None = None,
        max_context_tokens: int = 8192,
    ):
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.tokenizer_encode_fn = tokenizer_encode_fn
        self.max_context_tokens = max_context_tokens

        self._history: list[dict] = []
        self._state_summaries: dict[str, str] = {}

    def reset(self, task: str) -> None:
        self._history = []
        self._state_summaries = {}

    def add(self, message: dict) -> None:
        self._history.append(message)

    def compress_state(self, state: SOPState, tool_call_log: list[dict],
                       current_round: int = 1) -> None:
        state_tools = [
            log for log in tool_call_log
            if log.get("state") == state.value and log.get("success")
        ]

        if not state_tools:
            summary = f"[Round {current_round} — State {state.value.upper()} completed. No tools called.]"
        else:
            parts = []
            for log in state_tools:
                parts.append(f"{log['tool']} (ok)")
            summary = (
                f"[Round {current_round} — State {state.value.upper()} completed. "
                f"Tools: {', '.join(parts)}.]"
            )

        state_key = f"{state.value}_round_{current_round}"
        self._state_summaries[state_key] = summary
        logger.info(f"Compressed: {summary}")

    def build_messages(self, current_state: SOPState, state_hint: str,
                       current_round: int = 1, max_rounds: int = 3) -> list[dict]:
        messages = []

        messages.append({"role": "system", "content": self.system_prompt})

        if current_round > 1:
            n_new = 40 * (current_round - 1)
            messages.append({"role": "user", "content": (
                f"[ACTIVE LEARNING: Round {current_round}/{max_rounds}. "
                f"Models retrained with {n_new} new experimental data points. "
                f"Predictions should now be more accurate for explored chemical space.]"
            )})
            messages.append({"role": "assistant", "content": "Understood. Proceeding."})

        for state_key, summary in self._state_summaries.items():
            messages.append({"role": "user", "content": summary})
            messages.append({"role": "assistant", "content": "Understood."})

        recent = self._get_recent_history()
        messages.extend(recent)

        est_hint = 200
        total_est = self.get_token_estimate(messages) + est_hint
        if total_est > self.max_context_tokens and recent:
            recent_start = len(messages) - len(recent)
            while total_est > self.max_context_tokens and recent_start < len(messages):
                removed = messages.pop(recent_start)
                if (recent_start < len(messages)
                        and messages[recent_start].get("role") in ("assistant", "tool")):
                    messages.pop(recent_start)
                total_est = self.get_token_estimate(messages) + est_hint
            logger.info(f"Context trimmed to ~{total_est} tokens "
                        f"(budget: {self.max_context_tokens})")

        if current_state != SOPState.COMPLETE:
            allowed = sorted(STATE_ALLOWED_TOOLS[current_state])
            desc = STATE_DESCRIPTIONS.get(current_state, "")
            file_ctx = self._build_file_context()
            trajectory_note = ""
            if current_state == SOPState.REPORTING and current_round >= max_rounds:
                trajectory_note = " MUST call plot_active_learning_trajectory before finishing."
            hint = (
                f"[SYSTEM STATE: Round {current_round}/{max_rounds}, "
                f"{current_state.value.upper()} phase. "
                f"Goal: {desc}.{trajectory_note} "
                f"{file_ctx}"
                f"Available tools: {', '.join(allowed)}. "
                f"Follow the SOP strictly.]"
            )
            messages.append({"role": "user", "content": hint})

        return messages

    def _get_recent_history(self) -> list[dict]:
        clean_history = [
            msg for msg in self._history
            if not (isinstance(msg.get("content"), str)
                    and msg["content"].startswith("[SYSTEM STATE:"))
        ]

        rounds = []
        current_round_msgs = []
        for msg in reversed(clean_history):
            current_round_msgs.insert(0, msg)
            if msg.get("role") == "user" and not msg.get("content", "").startswith("["):
                if len(rounds) < self.max_rounds - 1:
                    rounds.insert(0, current_round_msgs)
                    current_round_msgs = []
                else:
                    break
            elif msg.get("role") == "assistant" and len(current_round_msgs) == 1:
                rounds.insert(0, current_round_msgs)
                current_round_msgs = []
                if len(rounds) >= self.max_rounds:
                    break

        if current_round_msgs:
            rounds.insert(0, current_round_msgs)

        result = []
        for r in rounds:
            result.extend(r)
        return result

    def _build_file_context(self) -> str:
        from lnp_agent.paths import RESULTS_DIR
        parts = []

        working_data = RESULTS_DIR / "working_data"
        if working_data.exists():
            files = sorted(p.name for p in working_data.glob("*.csv"))
            if files:
                parts.append(f"Working data: {', '.join(files)}.")

        wet_lab = RESULTS_DIR / "wet_lab"
        if wet_lab.exists():
            files = sorted(p.name for p in wet_lab.glob("*.csv"))
            if files:
                parts.append(f"Wet-lab results: {', '.join(files)}.")

        return " ".join(parts) + " " if parts else ""

    def get_token_estimate(self, messages: list[dict]) -> int:
        if self.tokenizer_encode_fn:
            try:
                return sum(
                    self.tokenizer_encode_fn(m.get("content", ""))
                    for m in messages if m.get("content")
                )
            except Exception:
                pass
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return int(total_chars / 3.5)

    @property
    def full_history(self) -> list[dict]:
        return list(self._history)


# ═══════════════════════════════════════════════════════════
# Agent Engine V5 (v2.5: 反冗余状态锁)
# ═══════════════════════════════════════════════════════════

class AgentEngineV5:
    """v2.5 循环状态机 Agent 引擎: 3 轮闭环主动学习 + 反冗余状态锁。

    v2.5 新增:
    - 工具签名去重: 防止同一状态内相同参数的重复调用
    - 状态完成自动推进: 必需工具完成后提示 LLM 进入下一状态
    - 采集函数默认参数: acquisition_method, kappa
    """

    def __init__(
        self,
        model_client,
        tools: dict,
        system_prompt: str,
        max_iterations: int = 50,
        max_rounds: int = 3,
        verbose: bool = True,
        permission_manager: PermissionManager | None = None,
        auto_confirm: bool = True,
        max_rejects_per_state: int = 5,
        conversation_window: int = 6,
        max_reporting_iterations: int = 8,
        max_context_tokens: int = 8192,
    ):
        self.model = model_client
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.auto_confirm = auto_confirm
        self.max_rejects_per_state = max_rejects_per_state

        self.event_bus = EventBus()
        self.permissions = permission_manager or PermissionManager()
        self.tool_schemas = {
            name: tool.definition.to_function_schema()
            for name, tool in tools.items()
        }

        self.conv_manager = ConversationManagerV5(
            system_prompt=system_prompt,
            max_rounds=conversation_window,
            max_context_tokens=max_context_tokens,
        )

        self.current_state = SOPState.EXTRACTION
        self.current_round = 1
        self.iteration_count = 0
        self.tool_call_log: list[dict] = []
        self.state_reject_count: dict[SOPState, int] = defaultdict(int)
        self.state_history: list[dict] = []
        self.state_iteration_count: dict[SOPState, int] = defaultdict(int)
        self.round_history: list[dict] = []

        # v2.5: 反冗余签名存储
        # key: (state, round) → set of signatures
        self._executed_signatures: dict[tuple[SOPState, int], set[str]] = defaultdict(set)

    def run(self, task: str) -> str:
        """执行完整的 3 轮闭环主动学习 SOP。"""
        self.conv_manager.reset(task)
        self.conv_manager.add({"role": "user", "content": task})

        self.current_state = SOPState.EXTRACTION
        self.current_round = 1
        self.state_history = []
        self.tool_call_log = []
        self.state_iteration_count = defaultdict(int)
        self.round_history = []
        self._executed_signatures.clear()

        response = {}

        for i in range(self.max_iterations):
            self.iteration_count = i + 1

            self.event_bus.emit(AgentEvent(
                type=EventType.ITERATION_START,
                data={
                    "iteration": i + 1, "max": self.max_iterations,
                    "state": self.current_state.value,
                    "round": self.current_round,
                },
            ))

            if self.verbose:
                logger.info(f"=== Iteration {i + 1}/{self.max_iterations} "
                            f"[State: {self.current_state.value}] [Round: {self.current_round}/{self.max_rounds}] ===")

            active_messages = self.conv_manager.build_messages(
                self.current_state, "",
                current_round=self.current_round,
                max_rounds=self.max_rounds,
            )
            token_estimate = self.conv_manager.get_token_estimate(active_messages)
            if self.verbose:
                logger.info(f"  Conversation: {len(active_messages)} messages, "
                            f"~{token_estimate} tokens")

            try:
                self.event_bus.emit(AgentEvent(
                    type=EventType.MODEL_REQUEST,
                    data={"iteration": i + 1, "state": self.current_state.value,
                          "round": self.current_round, "token_estimate": token_estimate},
                ))
                allowed_schemas = self._get_allowed_tool_schemas()
                response = self.model.chat(
                    messages=active_messages,
                    tools=allowed_schemas if allowed_schemas else None,
                )

                token_usage = {}
                if hasattr(self.model, 'last_token_usage'):
                    token_usage = self.model.last_token_usage

                self.event_bus.emit(AgentEvent(
                    type=EventType.MODEL_RESPONSE,
                    data={
                        "content_preview": str(response.get("content", ""))[:200],
                        "state": self.current_state.value,
                        "round": self.current_round,
                        "token_usage": token_usage,
                    },
                ))
            except Exception as e:
                self.event_bus.emit(AgentEvent(
                    type=EventType.ERROR,
                    data={"phase": "model_call", "error": str(e)},
                ))
                logger.error(f"Model error: {e}")
                self.conv_manager.add({
                    "role": "user",
                    "content": f"System error: {e}. Please try again.",
                })
                continue

            tool_calls = response.get("tool_calls")
            if tool_calls:
                self.conv_manager.add({
                    "role": "assistant",
                    "content": response.get("content", ""),
                })

                for tc in tool_calls:
                    result = self._execute_tool_with_state_check(tc)
                    tc_id = tc.get("id", "")

                    # v2.5: 状态完成自动推进提示
                    result = self._append_state_completion_hint(result)

                    self.conv_manager.add(result.to_message(tc_id))

                    # v2.5: 生成签名
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    tool_args = func.get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                    signature = self._get_tool_signature(tool_name, tool_args)

                    self.tool_call_log.append({
                        "iteration": i + 1,
                        "tool": tool_name,
                        "state": self.current_state.value,
                        "success": result.success,
                        "round": self.current_round,
                        "signature": signature,
                    })
                    self._maybe_transition(tc, result)

            elif self._is_task_complete(response):
                self.event_bus.emit(AgentEvent(
                    type=EventType.TASK_COMPLETE,
                    data={"iterations": i + 1, "state": self.current_state.value,
                          "round": self.current_round, "trigger": "text_marker"},
                ))
                return response.get("content", "Task completed (no output)")
            else:
                self.conv_manager.add({
                    "role": "assistant",
                    "content": response.get("content", ""),
                })

            if self.current_state != SOPState.COMPLETE:
                self.state_iteration_count[self.current_state] += 1
                n_iters = self.state_iteration_count[self.current_state]
                timeout = self._get_state_timeout(self.current_state)
                if n_iters >= timeout:
                    logger.warning(
                        f"State {self.current_state.value} timed out after "
                        f"{n_iters} iterations (limit: {timeout}). Force-advancing."
                    )
                    self._force_advance()

            if (self.current_state != SOPState.COMPLETE
                    and self.state_reject_count[self.current_state] >= self.max_rejects_per_state):
                logger.warning(f"Max rejects in {self.current_state.value}")
                self._force_advance()

            if self.current_state == SOPState.COMPLETE:
                return response.get("content", "SOP completed")

        return (
            f"[MAX ITERATIONS REACHED: {self.max_iterations}] "
            f"Final state: {self.current_state.value}, Round: {self.current_round}\n"
            f"{response.get('content', '')}"
        )

    # ═══════════════════════════════════════════════════════════
    # v2.5: 反冗余状态锁
    # ═══════════════════════════════════════════════════════════

    def _get_tool_signature(self, tool_name: str, tool_args: dict) -> str:
        """生成工具调用的唯一签名 (工具名 + 语义参数哈希)。

        v2.6: 对 REPORTING 工具排除 save_path 等不影响逻辑的参数。
        """
        ignore = REPORTING_TOOL_IGNORE_ARGS.get(tool_name, set())
        filtered_args = {k: v for k, v in tool_args.items() if k not in ignore}
        normalized = json.dumps(filtered_args, sort_keys=True, default=str)
        args_hash = hashlib.md5(normalized.encode()).hexdigest()[:8]
        return f"{tool_name}:{args_hash}"

    def _is_redundant_call(self, signature: str) -> bool:
        """检查当前状态+轮次内是否已有相同签名的成功调用。

        v2.6: 增加 REPORTING 一次调用锁定策略。
        """
        key = (self.current_state, self.current_round)

        # 标准签名检查
        if signature in self._executed_signatures[key]:
            return True

        # v2.6: REPORTING 一次调用检查
        tool_name = signature.split(":")[0]
        if tool_name in REPORTING_SINGLE_CALL_TOOLS:
            executed_tools = {s.split(":")[0] for s in self._executed_signatures[key]}
            if tool_name in executed_tools:
                return True

        return False

    def _record_successful_signature(self, signature: str) -> None:
        """记录成功的工具调用签名。"""
        key = (self.current_state, self.current_round)
        self._executed_signatures[key].add(signature)

    def _all_state_tools_completed(self) -> bool:
        """检测当前状态的所有必需工具是否已成功执行。"""
        required = STATE_REQUIRED_TOOLS.get(self.current_state, set())
        if not required:
            return False

        key = (self.current_state, self.current_round)
        executed_sigs = self._executed_signatures[key]
        executed_tools = {sig.split(":")[0] for sig in executed_sigs}

        return required.issubset(executed_tools)

    def _append_state_completion_hint(self, result: ToolResult) -> ToolResult:
        """v2.5: 如果状态所有必需工具已完成，在结果中附加提示。"""
        if result.success and self._all_state_tools_completed():
            # 检查是否已经附加过提示 (防止重复)
            if not result.output or "[ALL REQUIRED TOOLS COMPLETED]" not in result.output:
                hint = (
                    "\n\n[ALL REQUIRED TOOLS COMPLETED for current state. "
                    "Do NOT call any more tools in this state. "
                    "Proceed to the next state.]"
                )
                # 修改 output 而不是创建新 ToolResult (保持其他字段)
                if result.output:
                    try:
                        data = json.loads(result.output)
                        data["_hint"] = hint.strip()
                        result = ToolResult(success=True,
                                           output=json.dumps(data, ensure_ascii=False),
                                           error=result.error)
                    except (json.JSONDecodeError, TypeError):
                        result = ToolResult(success=True,
                                           output=result.output + hint,
                                           error=result.error)
                else:
                    result = ToolResult(success=True, output=hint, error=result.error)
        return result

    # ═══════════════════════════════════════════════════════════
    # 工具执行
    # ═══════════════════════════════════════════════════════════

    def _get_allowed_tool_schemas(self) -> list[dict]:
        if self.current_state == SOPState.COMPLETE:
            return []
        allowed_names = STATE_ALLOWED_TOOLS[self.current_state]
        return [schema for name, schema in self.tool_schemas.items()
                if name in allowed_names]

    def _resolve_tool_defaults(self, tool_name: str) -> dict:
        """v2.5: 根据当前轮次返回工具的默认参数。LLM 提供的参数优先。"""
        r = self.current_round
        defaults = {}

        if tool_name == "batch_predict_lnp":
            defaults["formulations_path"] = "working_data/virtual_library.csv"
            defaults["output_path"] = f"working_data/predictions_round_{r}.csv"

        elif tool_name == "filter_exploitation_batch":
            defaults["predictions_path"] = f"working_data/predictions_round_{r}.csv"
            defaults["output_path"] = f"working_data/exploitation_round_{r}.csv"
            defaults["top_n"] = 20
            # v2.5: 默认使用 UCB 采集函数
            defaults["acquisition_method"] = "ucb"
            defaults["kappa"] = 2.0

        elif tool_name == "filter_exploration_batch":
            defaults["predictions_path"] = f"working_data/predictions_round_{r}.csv"
            defaults["output_path"] = f"working_data/exploration_round_{r}.csv"
            defaults["top_n"] = 20
            # v2.5: exploration 默认保持 uncertainty (与 UCB 高 kappa 效果类似)
            defaults["diversity_method"] = "uncertainty"

        elif tool_name == "run_wet_lab_experiment":
            defaults["exploitation_path"] = f"working_data/exploitation_round_{r}.csv"
            defaults["exploration_path"] = f"working_data/exploration_round_{r}.csv"
            defaults["round_number"] = r

        elif tool_name == "retrain_lnp_predictors":
            defaults["wet_lab_results_path"] = f"wet_lab_results_round_{r}.csv"

        elif tool_name == "plot_pareto_front":
            defaults["predictions_path"] = f"working_data/predictions_round_{r}.csv"
            defaults["exploitation_path"] = f"working_data/exploitation_round_{r}.csv"
            defaults["exploration_path"] = f"working_data/exploration_round_{r}.csv"

        elif tool_name == "plot_chemical_space_umap":
            defaults["predictions_path"] = f"working_data/predictions_round_{r}.csv"
            defaults["exploitation_path"] = f"working_data/exploitation_round_{r}.csv"
            defaults["exploration_path"] = f"working_data/exploration_round_{r}.csv"

        return defaults

    def _execute_tool_with_state_check(self, tool_call: dict) -> ToolResult:
        """执行工具，检查状态约束、权限和冗余调用。"""
        func = tool_call.get("function", {})
        tool_name = func.get("name", "")
        tool_args = func.get("arguments", {})

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                return ToolResult(success=False, output="",
                                  error=f"Invalid JSON arguments: {tool_args}")

        # v2.4: 分层参数合并 (强制覆盖关键路径 + 缺省填充非关键参数)
        if self.current_state != SOPState.COMPLETE:
            defaults = self._resolve_tool_defaults(tool_name)
            forced_keys = FORCED_PARAMS.get(tool_name, set())
            overridden = []
            for k, v in defaults.items():
                if k in forced_keys:
                    if k in tool_args and tool_args[k] != v:
                        overridden.append(f"{k}: {tool_args[k]}→{v}")
                    tool_args[k] = v  # 强制覆盖
                elif k not in tool_args or tool_args[k] is None:
                    tool_args[k] = v  # 缺省填充
            if overridden:
                logger.warning(f"Forced override for {tool_name}: {overridden}")
            elif defaults:
                logger.info(f"Tool defaults merged for {tool_name}: "
                            f"{list(defaults.keys())}")

        # v2.5: 冗余调用检测 (在参数合并后检查)
        call_signature = self._get_tool_signature(tool_name, tool_args)
        if self._is_redundant_call(call_signature):
            logger.warning(
                f"Redundant call blocked: {tool_name} (signature={call_signature}, "
                f"state={self.current_state.value}, round={self.current_round})"
            )
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Tool '{tool_name}' already executed successfully with identical parameters "
                    f"in the current state ({self.current_state.value}). "
                    f"Do NOT call this tool again with the same parameters. "
                    f"Move to the next step."
                ),
            )

        # 状态检查
        if self.current_state != SOPState.COMPLETE:
            allowed = STATE_ALLOWED_TOOLS[self.current_state]
            if tool_name not in allowed:
                self.state_reject_count[self.current_state] += 1
                allowed_str = ", ".join(sorted(allowed))
                return ToolResult(success=False, output="", error=(
                    f"Tool '{tool_name}' not available in {self.current_state.value} phase. "
                    f"Allowed: {allowed_str}. Follow the SOP."
                ))

        # 权限检查
        permission = self.permissions.check(tool_name, tool_args)
        self.event_bus.emit(AgentEvent(
            type=EventType.PERMISSION_CHECK,
            data={"tool": tool_name, "permission": permission.value},
        ))

        if permission == Permission.DENY:
            return ToolResult(success=False, output="",
                              error=f"Tool '{tool_name}' denied.")
        elif permission == Permission.CONFIRM:
            if self.auto_confirm:
                if self.verbose:
                    logger.info(f"  [CONFIRM] {tool_name}(...auto-approved...)")
            elif not self._prompt_user_confirmation(tool_name, tool_args):
                return ToolResult(success=False, output="",
                                  error=f"Tool '{tool_name}' denied by user.")

        self.event_bus.emit(AgentEvent(
            type=EventType.TOOL_CALL,
            data={"tool": tool_name, "args": tool_args,
                  "state": self.current_state.value, "round": self.current_round},
        ))

        if tool_name not in self.tools:
            result = ToolResult(success=False, output="",
                                error=f"Unknown tool: {tool_name}. "
                                      f"Available: {list(self.tools.keys())}")
        else:
            tool = self.tools[tool_name]
            try:
                result = tool.execute(**tool_args)
            except Exception as e:
                result = ToolResult(success=False, output="",
                                    error=f"Tool error ({tool_name}): {type(e).__name__}: {e}")

        # v2.5: 记录成功调用的签名
        if result.success:
            self._record_successful_signature(call_signature)

        self.event_bus.emit(AgentEvent(
            type=EventType.TOOL_RESULT,
            data={
                "tool": tool_name,
                "success": result.success,
                "output_preview": result.output[:200] if result.output else "",
                "state": self.current_state.value,
                "round": self.current_round,
            },
        ))

        if not result.success and self.current_state != SOPState.COMPLETE:
            self.state_reject_count[self.current_state] += 1

        return result

    def _maybe_transition(self, tool_call: dict, result: ToolResult) -> None:
        """v2.2: 循环转移逻辑。"""
        if not result.success:
            return

        tool_name = tool_call["function"]["name"]

        # === 固定触发器 ===
        transition_triggers = {
            (SOPState.EXTRACTION, "generate_virtual_library"): SOPState.PREDICTION,
            (SOPState.PREDICTION, "batch_predict_lnp"): SOPState.FILTERING,
        }

        key = (self.current_state, tool_name)
        if key in transition_triggers:
            self._transition_to(transition_triggers[key])
            return

        # === FILTERING → REPORTING ===
        if self.current_state == SOPState.FILTERING:
            called_filters = set()
            for log in self.tool_call_log:
                if (log["success"] and log["round"] == self.current_round
                        and log["tool"] in (
                            "filter_exploitation_batch", "filter_exploration_batch")):
                    called_filters.add(log["tool"])
            if len(called_filters) >= 2:
                self._transition_to(SOPState.REPORTING)

        # === REPORTING → WET_LAB_TESTING 或 COMPLETE ===
        if self.current_state == SOPState.REPORTING:
            if tool_name == "write" or self._required_plots_called():
                self._handle_reporting_complete()

        # === WET_LAB_TESTING → RETRAINING ===
        if self.current_state == SOPState.WET_LAB_TESTING:
            if tool_name == "run_wet_lab_experiment":
                self._transition_to(SOPState.RETRAINING)

        # === RETRAINING → PREDICTION (循环) ===
        if self.current_state == SOPState.RETRAINING:
            if tool_name == "retrain_lnp_predictors":
                self.current_round += 1
                self.reporting_iterations = 0
                logger.info(f"=== Round {self.current_round}/{self.max_rounds} starting ===")
                self.event_bus.emit(AgentEvent(
                    type=EventType.ROUND_START,
                    data={"round": self.current_round, "max_rounds": self.max_rounds},
                ))
                self._transition_to(SOPState.PREDICTION)

    def _required_plots_called(self) -> bool:
        """v2.4: 检查当前 round 是否调用了所有必需的绘图工具。"""
        called = set()
        for log in self.tool_call_log:
            if (log["success"] and log["round"] == self.current_round
                    and log["state"] == "reporting"
                    and log["tool"] in (
                        "plot_pareto_front", "plot_chemical_space_umap",
                        "plot_active_learning_trajectory")):
                called.add(log["tool"])

        required = {"plot_pareto_front", "plot_chemical_space_umap"}
        if self.current_round >= self.max_rounds:
            required.add("plot_active_learning_trajectory")

        return required.issubset(called)

    def _handle_reporting_complete(self) -> None:
        if self.current_round >= self.max_rounds:
            logger.info(f"Final round ({self.current_round}) REPORTING complete → COMPLETE")
            self._record_round_metrics()
            self.event_bus.emit(AgentEvent(
                type=EventType.ROUND_COMPLETE,
                data={"round": self.current_round, "final": True},
            ))
            self._transition_to(SOPState.COMPLETE)
        else:
            logger.info(f"Round {self.current_round} REPORTING complete → WET_LAB_TESTING")
            self._record_round_metrics()
            self.event_bus.emit(AgentEvent(
                type=EventType.ROUND_COMPLETE,
                data={"round": self.current_round, "final": False},
            ))
            self._transition_to(SOPState.WET_LAB_TESTING)

    def _record_round_metrics(self) -> None:
        round_tools = [
            log for log in self.tool_call_log
            if log.get("round") == self.current_round and log.get("success")
        ]
        self.round_history.append({
            "round": self.current_round,
            "tools_called": [log["tool"] for log in round_tools],
            "tool_count": len(round_tools),
            "success_count": sum(1 for log in round_tools if log["success"]),
        })

    def _transition_to(self, new_state: SOPState) -> None:
        old_state = self.current_state
        self.current_state = new_state
        self.state_reject_count[new_state] = 0
        self.state_iteration_count[new_state] = 0
        self.state_history.append({
            "from": old_state.value,
            "to": new_state.value,
            "iteration": self.iteration_count,
            "round": self.current_round,
        })
        self.event_bus.emit(AgentEvent(
            type=EventType.STATE_TRANSITION,
            data={"from": old_state.value, "to": new_state.value,
                  "iteration": self.iteration_count, "round": self.current_round},
        ))
        logger.info(f"State transition: {old_state.value} → {new_state.value} "
                    f"(iter {self.iteration_count}, round {self.current_round})")

        if old_state != SOPState.COMPLETE:
            self.conv_manager.compress_state(
                old_state, self.tool_call_log, self.current_round)

    def _get_state_timeout(self, state: SOPState) -> int:
        STATE_TIMEOUTS = {
            SOPState.EXTRACTION: 10,
            SOPState.PREDICTION: 8,
            SOPState.FILTERING: 8,
            SOPState.REPORTING: 8,
            SOPState.WET_LAB_TESTING: 10,
            SOPState.RETRAINING: 8,
            SOPState.COMPLETE: 1,
        }
        return STATE_TIMEOUTS.get(state, 10)

    def _force_advance(self) -> None:
        if self.current_state == SOPState.REPORTING:
            self._handle_reporting_complete()
        else:
            transitions = STATE_TRANSITIONS.get(self.current_state, [])
            if transitions:
                self._transition_to(transitions[0])

    def _is_task_complete(self, response: dict) -> bool:
        content = response.get("content", "").upper()
        markers = [
            "TASK_COMPLETE",
            "[FINAL_REPORT_READY]",
            "FINAL REPORT COMPLETE",
        ]
        return any(marker in content for marker in markers)

    def get_summary(self) -> dict[str, Any]:
        n_calls = len(self.tool_call_log)
        n_success = sum(1 for t in self.tool_call_log if t["success"])
        return {
            "iterations": self.iteration_count,
            "tool_calls": n_calls,
            "tools_used": list({t["tool"] for t in self.tool_call_log}),
            "success_rate": n_success / max(n_calls, 1),
            "final_state": self.current_state.value,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "state_history": self.state_history,
            "round_history": self.round_history,
        }

    def _prompt_user_confirmation(self, tool_name: str, tool_args: dict) -> bool:
        args_preview = json.dumps(tool_args, ensure_ascii=False)[:100]
        print(f"\n  [Permission Required] {tool_name}({args_preview})")
        try:
            resp = input("  Proceed? [y/N]: ").strip().lower()
            return resp in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False
