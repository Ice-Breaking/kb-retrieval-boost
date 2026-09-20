# -*- coding: utf-8 -*-
"""重排与向量检索组件。

- BgeReranker   : 交叉编码器重排模型（bge-reranker 系列），适合在带 GPU 的环境运行
- OllamaReranker: 用本机大模型给候选文档打分，无需下载任何模型文件
- MockReranker  : 词面重合度打分，仅用于在没有 GPU / 模型的机器上验证流水线
- VectorIndex   : 基于 transformers 的中文向量检索（bge 系列），不依赖 sentence-transformers
"""
from __future__ import annotations

import os
import re

import jieba


class BaseReranker:
    name = "base"

    def score(self, query: str, docs: list) -> list:
        """返回与 docs 等长的相关性分数，越大越相关。"""
        raise NotImplementedError

    @property
    def tag(self) -> str:
        return self.name


class BgeReranker(BaseReranker):
    """bge-reranker 交叉编码器：query 与文档拼接后逐 token 交互，精度高，但只能用于精排。"""

    name = "bge"

    def __init__(self, model_id: str = "BAAI/bge-reranker-base", cache_dir: str = None,
                 max_length: int = 512):
        import torch
        from modelscope import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_id = model_id
        self.max_length = max_length
        cache_dir = cache_dir or os.getenv("MODELSCOPE_CACHE")
        print(f"正在加载重排模型 {model_id}（首次使用会从 ModelScope 下载到缓存目录）")
        model_dir = snapshot_download(model_id, cache_dir=cache_dir) if cache_dir else snapshot_download(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"重排模型已加载，运行设备：{self.device}")

    def score(self, query: str, docs: list) -> list:
        if not docs:
            return []
        pairs = [[query, doc] for doc in docs]
        with self.torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits.view(-1).float().cpu().tolist()
        return logits

    @property
    def tag(self) -> str:
        return f"bge:{self.model_id}"


class OllamaReranker(BaseReranker):
    """用本机大模型给候选文档打分（0-10 分），一次请求评完所有候选，避免逐个调用。"""

    name = "ollama"

    def __init__(self, backend=None):
        from llm_backend import get_backend
        self.llm = backend or get_backend("ollama")
        self.fallback_calls = 0

    def score(self, query: str, docs: list) -> list:
        if not docs:
            return []
        numbered = "\n".join(f"{i}. {doc[:220]}" for i, doc in enumerate(docs, 1))
        prompt = ("你是检索重排模型。请判断每个候选文档与用户问题的相关程度，给 0-10 分"
                  "（10 分最相关，0 分完全不相关）。\n"
                  f"用户问题：{query}\n候选文档：\n{numbered}\n"
                  "只输出一行数字，按文档序号顺序用逗号分隔，例如：9,3,7,0")
        try:
            text = self.llm.chat(prompt, json_mode=False, temperature=0)
            numbers = [min(float(x), 10.0) for x in re.findall(r"\d+(?:\.\d+)?", text)]
            if len(numbers) >= len(docs):
                return numbers[:len(docs)]
        except Exception as exc:  # noqa: BLE001
            print(f"  批量打分失败，改为逐个打分：{exc}")
        self.fallback_calls += 1
        return [self._score_one(query, doc) for doc in docs]

    def _score_one(self, query: str, doc: str) -> float:
        prompt = (f"用户问题：{query}\n文档：{doc[:400]}\n"
                  "该文档与问题的相关程度是几分？只输出 0-10 的整数。")
        try:
            found = re.findall(r"\d+(?:\.\d+)?", self.llm.chat(prompt, temperature=0))
            return min(float(found[0]), 10.0) if found else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    @property
    def tag(self) -> str:
        return f"ollama打分:{getattr(self.llm, 'model', 'llm')}"


class MockReranker(BaseReranker):
    """词面重合度打分，仅用于在没有 GPU / 模型文件的机器上验证流水线是否跑通。"""

    name = "mock"

    def score(self, query: str, docs: list) -> list:
        q = set(w for w in jieba.lcut(query) if len(w) > 1)
        scores = []
        for doc in docs:
            d = set(w for w in jieba.lcut(doc) if len(w) > 1)
            scores.append(len(q & d) / max(len(q), 1) * 10)
        return scores


def get_reranker(name: str = None, model_id: str = None, cache_dir: str = None) -> BaseReranker:
    """按名创建重排器：bge（默认，需要模型）/ ollama（本机大模型）/ mock（冒烟测试）。"""
    name = (name or os.getenv("RERANKER", "bge")).lower()
    if name in ("bge", "bge-reranker", "cross-encoder"):
        return BgeReranker(model_id or os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"), cache_dir)
    if name in ("ollama", "llm"):
        return OllamaReranker()
    if name == "mock":
        return MockReranker()
    raise ValueError(f"未知重排器：{name}")


class VectorIndex:
    """中文向量检索：transformers 直接加载 bge 系列向量模型（CLS 池化 + 归一化 + 余弦相似度）。"""

    def __init__(self, model_id: str = "BAAI/bge-small-zh-v1.5", cache_dir: str = None,
                 max_length: int = 512, batch_size: int = 16):
        import numpy as np
        import torch
        from modelscope import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        self.np, self.torch = np, torch
        self.model_id, self.max_length, self.batch_size = model_id, max_length, batch_size
        cache_dir = cache_dir or os.getenv("MODELSCOPE_CACHE")
        print(f"正在加载向量模型 {model_id}（首次使用会从 ModelScope 下载到缓存目录）")
        model_dir = snapshot_download(model_id, cache_dir=cache_dir) if cache_dir else snapshot_download(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"向量模型已加载，运行设备：{self.device}")

    def encode(self, texts: list):
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state[:, 0]
            vectors.append(self.torch.nn.functional.normalize(hidden, dim=-1).cpu().numpy())
        return self.np.vstack(vectors)

    def similarity(self, query: str, doc_vectors) -> list:
        return list((doc_vectors @ self.encode([query])[0]) * 100)

    @property
    def tag(self) -> str:
        return f"vector:{self.model_id}"
