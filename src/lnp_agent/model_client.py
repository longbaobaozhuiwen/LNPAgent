"""Gemma 4 推理客户端 (v2.0 — 新增 token 计数)。

v2.0 变更:
- 新增 count_tokens() 方法 (使用 tokenizer 精确计数)
- 新增 _last_input_tokens / _last_output_tokens 属性
- 推理后记录 token 用量日志

继承 v1.7 的所有修复:
- 多层 JSON 修复策略处理 Gemma 的 <|"|> 转义伪影
- 每轮推理后清理 KV-cache (torch.cuda.empty_cache)
- max_memory 限制每 GPU 显存分配，防止 OOM
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

# 防止 CUDA 内存碎片化
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)


class GemmaClient:
    """Gemma 4 推理客户端。

    v2.0 修改:
    - token 计数功能
    - 推理 token 用量日志

    v1.7 修改:
    - 多层 JSON 修复策略 (7 层 fallback)
    - KV-cache 清理防止 OOM
    - max_memory 限制每 GPU 显存
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        backend: str = "transformers",
        max_memory_per_gpu: str = "28GiB",
    ):
        self.model_path = Path(model_path) if model_path else None
        self.model_id = str(model_path)
        self.device = device
        self.backend = backend
        self.max_memory_per_gpu = max_memory_per_gpu
        self.model = None
        self.tokenizer = None

        # v2.0: token 计数
        self._last_input_tokens: int = 0
        self._last_output_tokens: int = 0

    def load_model(self) -> None:
        """加载模型到内存。"""
        if self.backend == "transformers":
            self._load_transformers()
        elif self.backend == "llama_cpp":
            self._load_llama_cpp()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _load_transformers(self) -> None:
        """使用 HuggingFace Transformers 加载模型。"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model from {self.model_id} (device={self.device})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
        }
        if self.device == "auto":
            load_kwargs["device_map"] = "auto"
            n_gpus = torch.cuda.device_count()
            load_kwargs["max_memory"] = {
                i: self.max_memory_per_gpu for i in range(n_gpus)
            }
        else:
            load_kwargs["device_map"] = self.device

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        logger.info("Model loaded successfully.")

    def _load_llama_cpp(self) -> None:
        """使用 llama-cpp-python 加载 GGUF 模型。"""
        import llama_cpp

        gguf_files = list(Path(self.model_path).glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf files found in {self.model_path}")

        model_file = gguf_files[0]
        logger.info(f"Loading GGUF model from {model_file}...")
        self.model = llama_cpp.Llama(
            model_path=str(model_file),
            n_ctx=8192,
            n_gpu_layers=0,
            verbose=False,
        )
        logger.info("GGUF model loaded successfully.")

    def _get_input_device(self):
        """获取输入 tensor 应放置的设备。"""
        if self.device == "auto" and hasattr(self.model, "device"):
            return self.model.device
        if self.device == "auto":
            if hasattr(self.model, "hf_device_map"):
                first_device = list(self.model.hf_device_map.values())[0]
                return first_device
            return "cuda:0"
        return self.device

    def unload_model(self) -> None:
        """卸载模型释放 GPU 内存。"""
        import gc

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("Model unloaded, GPU cache cleared.")

    def count_tokens(self, messages: list[dict]) -> int:
        """v2.0: 计算 messages 的 token 数。"""
        if self.tokenizer is None:
            return 0
        formatted = self._format_messages_for_gemma(messages)
        try:
            inputs = self.tokenizer.apply_chat_template(
                formatted,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
            return inputs["input_ids"].shape[1]
        except Exception:
            # Fallback: 启发式
            total_chars = sum(len(m.get("content", "")) for m in formatted)
            return int(total_chars / 3.5)

    @property
    def last_token_usage(self) -> dict[str, int]:
        """返回最近一次推理的 token 用量。"""
        return {
            "input_tokens": self._last_input_tokens,
            "output_tokens": self._last_output_tokens,
        }

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> dict:
        """发送对话请求并返回响应。"""
        if self.backend == "transformers":
            return self._chat_transformers(messages, tools, max_tokens, temperature, top_p)
        elif self.backend == "llama_cpp":
            return self._chat_llama_cpp(messages, tools, max_tokens, temperature, top_p)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _chat_transformers(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        """Transformers 后端的 chat 实现。"""
        import torch

        formatted = self._format_messages_for_gemma(messages)

        if tools and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                inputs = self.tokenizer.apply_chat_template(
                    formatted,
                    tools=tools,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                )
            except Exception:
                formatted = self._inject_tools_to_system(formatted, tools)
                inputs = self.tokenizer.apply_chat_template(
                    formatted,
                    return_tensors="pt",
                    return_dict=True,
                    add_generation_prompt=True,
                )
        else:
            inputs = self.tokenizer.apply_chat_template(
                formatted,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )

        # 移到模型所在设备
        target_device = self._get_input_device()
        input_ids = inputs["input_ids"].to(target_device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

        # v2.0: 记录 token 用量
        self._last_input_tokens = input_ids.shape[1]
        self._last_output_tokens = len(new_tokens)
        logger.info(f"Token usage: input={self._last_input_tokens}, output={self._last_output_tokens}")

        # v1.7: 清理 KV-cache 和 GPU 缓存防止 OOM
        del outputs, new_tokens, input_ids
        if attention_mask is not None:
            del attention_mask
        torch.cuda.empty_cache()

        return self._parse_response(text, tools is not None)

    def _format_messages_for_gemma(self, messages: list[dict]) -> list[dict]:
        """将 tool role 消息转换为 Gemma 可理解的格式。"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "tool":
                formatted.append({
                    "role": "user",
                    "content": f"[Tool Result]\n{content}",
                })
            elif role == "assistant" and msg.get("tool_calls"):
                text = content or ""
                formatted.append({"role": "assistant", "content": text})
            else:
                formatted.append(msg)

        return formatted

    def _chat_llama_cpp(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict:
        """llama-cpp 后端的 chat 实现。"""
        if tools:
            messages = self._inject_tools_to_system(messages, tools)

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        choice = response["choices"][0]
        content = choice["message"].get("content", "")
        finish_reason = choice.get("finish_reason", "stop")

        return {
            "content": content,
            "tool_calls": None,
            "finish_reason": finish_reason,
        }

    def _inject_tools_to_system(
        self, messages: list[dict], tools: list[dict]
    ) -> list[dict]:
        """将工具描述注入 system prompt (fallback 方案)。"""
        tool_descriptions = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            required = params.get("required", [])
            props = params.get("properties", {})

            param_strs = []
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                req = " (required)" if pname in required else ""
                param_strs.append(f"    - {pname}: {ptype}{req} — {pdesc}")

            tool_descriptions.append(
                f"- {name}: {desc}\n  Parameters:\n" + "\n".join(param_strs)
            )

        tools_text = (
            "\n\n## Available Tools\n"
            "You can call tools by responding with JSON in this format:\n"
            '```json\n{"tool_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]}\n```\n\n'
            + "\n".join(tool_descriptions)
        )

        modified = []
        for msg in messages:
            if msg["role"] == "system":
                modified.append({
                    "role": "system",
                    "content": msg["content"] + tools_text,
                })
            else:
                modified.append(msg)
        return modified

    def _parse_response(self, text: str, has_tools: bool) -> dict:
        """解析模型响应，提取 tool calls。"""
        text = text.strip()

        tool_calls = None
        content = text

        if has_tools:
            tool_calls = self._extract_tool_calls(text)
            if tool_calls:
                content = self._remove_tool_call_json(text)

        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"

        return {
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        }

    def _extract_tool_calls(self, text: str) -> list[dict] | None:
        """从文本中提取 tool call。支持多种格式。"""
        gemma_calls = self._extract_gemma_tool_calls(text)
        if gemma_calls:
            return gemma_calls

        json_pattern = r"```json\s*\n?(.*?)\n?\s*```"
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and "tool_calls" in data:
                    return [
                        {
                            "id": f"call_{i}",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc.get("arguments", {}),
                            },
                        }
                        for i, tc in enumerate(data["tool_calls"])
                    ]
            except json.JSONDecodeError:
                continue

        brace_pattern = r"\{[^{}]*\"tool_calls\"[^{}]*\}"
        matches = re.findall(brace_pattern, text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if "tool_calls" in data:
                    return [
                        {
                            "id": f"call_{i}",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc.get("arguments", {}),
                            },
                        }
                        for i, tc in enumerate(data["tool_calls"])
                    ]
            except json.JSONDecodeError:
                continue

        return None

    def _extract_gemma_tool_calls(self, text: str) -> list[dict] | None:
        """解析 Gemma 原生 tool call 格式。"""
        patterns = [
            r"<\|tool_call\|>call:(\w+)\{(.*?)\}(?:<\|end\|>|<tool_call\|>)",
            r"<\|tool_call\|>call:(\w+)\{(.*?)\}",
            r"call:(\w+)\{(.*?)\}(?:<\|end\|>|<tool_call\|>)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return self._parse_gemma_matches(matches)
        return None

    def _parse_gemma_matches(self, matches) -> list[dict]:
        """解析正则匹配到的 tool call。"""
        tool_calls = []
        for i, (name, args_str) in enumerate(matches):
            args = self._fix_json_args(args_str.strip())
            tool_calls.append({
                "id": f"call_{i}",
                "function": {"name": name, "arguments": args},
            })
        return tool_calls

    def _fix_json_args(self, args_str: str) -> dict:
        """多层 JSON 修复策略 (7 层 fallback)。"""
        wrapped = "{" + args_str + "}"

        try:
            return json.loads(wrapped)
        except json.JSONDecodeError:
            pass

        cleaned = wrapped
        cleaned = cleaned.replace('<|"|>', '"')
        cleaned = cleaned.replace("<|'|>", "'")
        cleaned = cleaned.replace('<|"', '"')
        cleaned = cleaned.replace('"|>', '"')
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        cleaned = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        cleaned = cleaned.replace("'", '"')
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        open_braces = cleaned.count('{') - cleaned.count('}')
        open_brackets = cleaned.count('[') - cleaned.count(']')
        cleaned = cleaned + '}' * max(0, open_braces) + ']' * max(0, open_brackets)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        return {"_raw": args_str}

    def _remove_tool_call_json(self, text: str) -> str:
        """从文本中移除 tool call 块。"""
        text = re.sub(r"<\|tool_call\|>call:.*?(?:<\|end\|>|<tool_call\|>)", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|tool_call\|>call:.*", "", text, flags=re.DOTALL)
        text = re.sub(r"```json\s*\n?.*?\n?\s*```", "", text, flags=re.DOTALL)
        text = text.replace("<eos>", "").replace("<|eos|>", "")
        return text.strip()
