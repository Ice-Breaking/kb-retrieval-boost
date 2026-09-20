# -*- coding: utf-8 -*-
"""统一的文本生成后端：ollama（本地）/ transformers（本地或云 GPU）/ dashscope（云 API）。

环境变量：
  LLM_BACKEND=ollama|transformers|dashscope   不设置则自动探测
  OLLAMA_MODEL / OLLAMA_HOST                  默认 qwen2.5-rewrite-lora:latest / 127.0.0.1:11434
  HF_LLM_MODEL                                默认 Qwen/Qwen2.5-1.5B-Instruct
  DASHSCOPE_API_KEY / DASHSCOPE_LLM_MODEL     默认 qwen-turbo-latest
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


class BaseBackend:
    name = "base"

    def chat(self, prompt: str, json_mode: bool = False, temperature: float = 0.3) -> str:
        raise NotImplementedError

    @property
    def tag(self) -> str:
        return self.name


class OllamaBackend(BaseBackend):
    """本机 Ollama 服务：无需下载模型文件、无需 API Key。"""

    name = "ollama"

    def __init__(self, model: str = None, host: str = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-rewrite-lora:latest")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")

    def alive(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def chat(self, prompt: str, json_mode: bool = False, temperature: float = 0.3) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "options": {"temperature": temperature,
                               "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "1200")),
                               "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096"))}}
        if json_mode:
            payload["format"] = "json"
        req = urllib.request.Request(f"{self.host}/api/generate",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return (json.loads(resp.read().decode("utf-8")).get("response") or "").strip()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama 调用失败，请确认已执行 ollama serve：{exc}") from exc

    @property
    def tag(self) -> str:
        return f"ollama:{self.model}"


class TransformersBackend(BaseBackend):
    """在本地或云 GPU 上直接加载因果语言模型（适合魔搭 Notebook 的免费 GPU）。"""

    name = "transformers"

    def __init__(self, model: str = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model or os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.model.eval()

    def chat(self, prompt: str, json_mode: bool = False, temperature: float = 0.3) -> str:
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                  tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=1024, do_sample=temperature > 0,
                                      temperature=max(temperature, 1e-4), top_p=0.9)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

    @property
    def tag(self) -> str:
        return f"transformers:{self.model_id}"


class DashScopeBackend(BaseBackend):
    """阿里云百炼 OpenAI 兼容接口。"""

    name = "dashscope"

    def __init__(self, model: str = None):
        from openai import OpenAI

        key = os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError("未设置 DASHSCOPE_API_KEY")
        self.model = model or os.getenv("DASHSCOPE_LLM_MODEL", "qwen-turbo-latest")
        self.client = OpenAI(api_key=key,
                             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    def chat(self, prompt: str, json_mode: bool = False, temperature: float = 0.3) -> str:
        kwargs = dict(model=self.model, temperature=temperature,
                      messages=[{"role": "user", "content": prompt}])
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return (self.client.chat.completions.create(**kwargs).choices[0].message.content or "").strip()

    @property
    def tag(self) -> str:
        return f"dashscope:{self.model}"


def get_backend(name: str = None) -> BaseBackend:
    """按需实例化后端；未指定时按 ollama -> dashscope -> transformers 顺序探测。"""
    name = (name or os.getenv("LLM_BACKEND") or "").lower()
    if name == "ollama":
        return OllamaBackend()
    if name in ("transformers", "hf", "local"):
        return TransformersBackend()
    if name == "dashscope":
        return DashScopeBackend()
    ollama = OllamaBackend()
    if ollama.alive():
        return ollama
    if os.getenv("DASHSCOPE_API_KEY"):
        return DashScopeBackend()
    return TransformersBackend()


def extract_json(text: str) -> dict:
    """从模型输出中稳健取出 JSON（容忍 ```json 包裹与前后多余文字）。"""
    if not text:
        raise ValueError("模型返回为空")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, depth = None, 0
    for idx, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(cleaned[start:idx + 1])
                except json.JSONDecodeError:
                    start = None
    raise ValueError(f"无法解析 JSON：{text[:120]}")

