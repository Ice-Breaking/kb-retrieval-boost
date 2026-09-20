# -*- coding: utf-8 -*-
"""检索与评测通用工具：中文分词、BM25 检索、命中指标、CSV / PNG 报表。"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

STOPWORDS = set("""
的 了 在 是 和 与 及 或 等 有 对 为 于 按 按照 根据 应 应当 须 必须 可以 可 不 未 无 该 本 其 其中
一个 以及 其他 有关 下列 以下 我们 你们 他们 这 那 分别 通过 进行 执行 规定 情况 有关 由 从 向 并 但
""".split())


# ---------------------------------------------------------------- 分词 / 检索
def tokenize(text: str) -> list:
    """中文分词：去标点、去停用词、去单字。"""
    text = re.sub(r"[^\w\s]", " ", text or "")
    return [w for w in jieba.lcut(text) if len(w) > 1 and w not in STOPWORDS]


def load_corpus(path=None) -> list:
    path = Path(path or ROOT / "data" / "knowledge_base.json")
    return json.loads(path.read_text(encoding="utf-8"))


def build_bm25(texts: list) -> BM25Okapi:
    return BM25Okapi([tokenize(t) for t in texts])


def bm25_scores(index, query: str) -> list:
    return [float(s) for s in index.get_scores(tokenize(query))]


def rank_by_scores(scores: list, top_k: int = 20) -> list:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [(i, float(scores[i])) for i in order[:top_k]]


# ---------------------------------------------------------------- 指标
def hit_metrics(ranked_ids: list, gold: set, ks=(1, 3, 5)) -> dict:
    out = {}
    for k in ks:
        out[f"hit@{k}"] = 1.0 if any(g in gold for g in ranked_ids[:k]) else 0.0
    rank = next((i for i, d in enumerate(ranked_ids, 1) if d in gold), -1)
    out["gold_rank"] = rank
    out["mrr"] = round(1.0 / rank, 4) if rank > 0 else 0.0
    return out


def summarize(rows: list, keys=("hit@1", "hit@3", "hit@5", "mrr")) -> dict:
    n = max(len(rows), 1)
    return {k: round(sum(r.get(k, 0.0) for r in rows) / n, 4) for k in keys}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


# ---------------------------------------------------------------- 报表
def save_csv(rows: list, path, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[报表] 已生成: {path}")
    return path


CJK_FONTS = ["PingFang SC", "Heiti TC", "Hiragino Sans GB", "Arial Unicode MS", "Songti SC",
             "STHeiti", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
             "SimHei", "Microsoft YaHei"]


def setup_font(verbose: bool = True) -> str:
    """为 matplotlib 配置可用的中文字体。"""
    from matplotlib import font_manager, rcParams
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False
            if verbose:
                print(f"[报表] 中文字体: {name}")
            return name
    print("[报表] 警告: 未找到中文字体，图中中文可能显示为方框（Linux 可 apt install fonts-noto-cjk）")
    return ""


def render_table_png(title: str, headers: list, rows: list, path, note: str = None,
                     highlight_rows=(), col_widths=None):
    """把对比表渲染成 PNG 截图，便于直接贴到汇报材料里。"""
    setup_font()
    import matplotlib.pyplot as plt
    highlight_rows = set(highlight_rows)
    n_rows = len(rows) + 1
    fig_w = max(9.5, 1.45 * len(headers))
    fig_h = 0.40 * n_rows + 1.7 + (0.4 if note else 0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=18, fontweight="bold")
    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cfd8e3")
        if r == 0:
            cell.set_facecolor("#2f5d8a")
            cell.set_text_props(color="white", fontweight="bold")
        elif r - 1 in highlight_rows:
            cell.set_facecolor("#fff3cd")
        elif r % 2 == 0:
            cell.set_facecolor("#f6f9fc")
    if note:
        fig.text(0.5, 0.015, note, ha="center", fontsize=8, color="#555555")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[报表] 已生成: {path}")
    return path
