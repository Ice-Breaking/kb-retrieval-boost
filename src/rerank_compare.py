# -*- coding: utf-8 -*-
"""高效召回：两阶段检索（粗排召回 + bge-reranker 精排）。

为什么要两阶段：
  粗排（BM25 / 向量检索）速度快、能扫全库，但只做“词面/语义近似匹配”，排序精度有限；
  交叉编码器重排模型把「问题 + 文档」拼在一起逐 token 交互，判断更准，但计算量大，只能用于少量候选。
因此通用做法是：先粗排召回 N 条候选，再用重排模型精排出最终 Top-K。

用法:
    python src/rerank_compare.py --reranker bge                  # 正式实验（GPU 环境，首次自动下载模型）
    python src/rerank_compare.py --reranker bge --stage1 bm25+vector
    python src/rerank_compare.py --reranker ollama               # 用本机大模型打分，无需下载模型
    python src/rerank_compare.py --reranker mock                 # 无模型环境的冒烟测试
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import (RESULTS, ROOT, bm25_scores, build_bm25, hit_metrics, load_corpus,  # noqa: E402
                        pct, rank_by_scores, render_table_png, save_csv, summarize)
from rerankers import get_reranker  # noqa: E402


def normalize(scores: list) -> list:
    top = max(scores) if scores and max(scores) > 0 else 1.0
    return [s / top for s in scores]


def stage1_recall(records: list, queries: list, top_n: int, mode: str = "bm25",
                  vector_index=None) -> dict:
    """粗排召回：BM25 或 BM25 + 向量融合，返回 {query_id: [(切片id, 分数), ...]}。"""
    ids = [r["id"] for r in records]
    index = build_bm25([r["content"] for r in records])
    doc_vectors = None
    if mode == "bm25+vector":
        if vector_index is None:
            raise SystemExit("使用 --stage1 bm25+vector 时需要向量模型，请检查 rerankers.VectorIndex 是否可用")
        print("正在向量化知识切片（一次性）…")
        doc_vectors = vector_index.encode([r["content"] for r in records])
    candidates = {}
    for query in queries:
        c_scores = bm25_scores(index, query["query"])
        if doc_vectors is not None:
            v_scores = vector_index.similarity(query["query"], doc_vectors)
            scores = [0.5 * a + 0.5 * b for a, b in
                      zip(normalize(c_scores), normalize(v_scores))]
        else:
            scores = c_scores
        candidates[query["id"]] = [(ids[i], score) for i, score in rank_by_scores(scores, top_n)]
    return candidates


def evaluate(records: list, queries: list, reranker, top_n: int = 20, stage1: str = "bm25",
             vector_index=None) -> dict:
    """对每条提问执行“粗排 -> 重排”，记录正确条款在重排前后的名次变化。"""
    text_of = {r["id"]: r["content"] for r in records}
    label_of = {r["id"]: r["source_label"] for r in records}
    candidates = stage1_recall(records, queries, top_n, stage1, vector_index)
    before_hits, after_hits = [], []
    rows = []
    print(f"粗排通道：{stage1}（候选 {top_n} 条）-> 精排：{reranker.tag}")
    for query in queries:
        cand = candidates[query["id"]]
        gold = set(query["gold"])
        before_ids = [cid for cid, _ in cand]
        m_before = hit_metrics(before_ids, gold)
        docs = [text_of[cid] for cid, _ in cand]
        start = time.time()
        scores = reranker.score(query["query"], docs)
        cost = time.time() - start
        if len(scores) != len(cand):  # 打分异常时兜底，保持粗排顺序
            scores = [0.0] * len(cand)
        order = sorted(range(len(cand)), key=lambda i: scores[i], reverse=True)
        after_ids = [cand[i][0] for i in order]
        m_after = hit_metrics(after_ids, gold)
        before_hits.append(m_before)
        after_hits.append(m_after)
        rows.append({
            "id": query["id"], "query": query["query"], "qtype": query["qtype"],
            "gold": "|".join(query["gold"]),
            "gold_label": label_of.get(query["gold"][0], ""),
            "before_rank": m_before["gold_rank"], "after_rank": m_after["gold_rank"],
            "before_score": round(cand[0][1], 3),
            "top1_before": label_of.get(before_ids[0], ""),
            "top1_after": label_of.get(after_ids[0], ""),
            "cost_s": round(cost, 3),
            "improved": m_after["gold_rank"] > 0 and (
                m_before["gold_rank"] == -1 or m_after["gold_rank"] < m_before["gold_rank"]),
        })
    return {"rows": rows, "before": summarize(before_hits), "after": summarize(after_hits),
            "reranker": reranker.tag, "stage1": stage1, "top_n": top_n}


def report(result: dict, suffix: str = "bge"):
    rows = result["rows"]
    before, after = result["before"], result["after"]
    avg_cost = sum(r["cost_s"] for r in rows) / max(len(rows), 1)
    table = []
    for key, name in (("hit@1", "Top-1 命中率"), ("hit@3", "Top-3 命中率"),
                      ("hit@5", "Top-5 命中率"), ("mrr", "MRR")):
        delta = after[key] - before[key]
        table.append([name, f"{before[key]:.3f}" if key == "mrr" else pct(before[key]),
                      f"{after[key]:.3f}" if key == "mrr" else pct(after[key]),
                      f"{delta:+.3f}" if key == "mrr" else f"{delta * 100:+.1f} 个百分点"])
    table.append(["单条提问平均重排耗时", "-", f"{avg_cost * 1000:.0f} ms", "-"])
    render_table_png(
        f"两阶段检索效果：粗排（{result['stage1']}，候选 {result['top_n']} 条）→ {result['reranker']} 精排",
        ["指标", "仅粗排", "粗排 + 重排", "变化"], table,
        RESULTS / f"rerank_summary_{suffix}.png",
        note=(f"测试集 {len(rows)} 条真实提问（含口语化提问、术语差异、版本时效、规则冲突等考点）；"
              f"重排只作用于粗排召回的 {result['top_n']} 条候选，因此提升主要体现在排序名次上"),
        highlight_rows=[0, 1, 2])

    ranked = sorted(rows, key=lambda r: (
        (r["before_rank"] if r["before_rank"] > 0 else 999) - (r["after_rank"] if r["after_rank"] > 0 else 999)),
        reverse=True)
    cases = []
    for row in ranked[:15]:
        b = row["before_rank"] if row["before_rank"] > 0 else "未进候选"
        a = row["after_rank"] if row["after_rank"] > 0 else "未命中"
        if row["before_rank"] == -1 and row["after_rank"] > 0:
            verdict = "★ 重排救回"
        elif isinstance(b, int) and isinstance(a, int) and a < b:
            verdict = f"提升 {b - a} 位"
        elif isinstance(b, int) and isinstance(a, int) and a == b:
            verdict = "持平"
        else:
            verdict = "未改善"
        cases.append([row["query"], row["gold_label"], b, a, verdict])
    render_table_png(
        "典型案例：重排前后正确条款的名次变化（名次越小越好）",
        ["用户提问", "正确条款", "重排前名次", "重排后名次", "结论"],
        cases, RESULTS / f"rerank_cases_{suffix}.png",
        note="粗排保留 20 条候选；★ 表示粗排名次之外被重排救回的条款",
        highlight_rows=[i for i, c in enumerate(cases) if c[4].startswith("★")],
        col_widths=[0.36, 0.28, 0.12, 0.12, 0.12])

    save_csv(rows, RESULTS / f"rerank_details_{suffix}.csv")
    print("\n=== 汇总指标 ===")
    for key, name in (("hit@1", "Top-1 命中率"), ("hit@3", "Top-3 命中率"),
                      ("hit@5", "Top-5 命中率"), ("mrr", "MRR")):
        unit = "" if key == "mrr" else " 个百分点"
        print(f"  {name:<14} 粗排 {before[key]:.3f} -> 重排 {after[key]:.3f}"
              f"  ({after[key] - before[key]:+.3f}{unit})")
    improved = [r for r in rows if r["improved"]]
    rescued = [r for r in rows if r["before_rank"] == -1 and r["after_rank"] > 0]
    print(f"  单条提问平均重排耗时 {avg_cost * 1000:.0f} ms")
    print(f"  名次提升的提问：{len(improved)} 条；重排救回：{len(rescued)} 条")
    for row in ranked[:3]:
        print(f"    · {row['query']}\n      重排前第 1 名: {row['top1_before'][:46]}\n"
              f"      重排后第 1 名: {row['top1_after'][:46]}")


def main():
    parser = argparse.ArgumentParser(description="两阶段检索：粗排召回 + 重排精排")
    parser.add_argument("--reranker", default="bge", choices=("bge", "ollama", "mock"))
    parser.add_argument("--rerank-model", default=None, help="默认 BAAI/bge-reranker-base")
    parser.add_argument("--stage1", default="bm25", choices=("bm25", "bm25+vector"))
    parser.add_argument("--vector-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--candidates", type=int, default=20, help="粗排召回的候选数量")
    parser.add_argument("--model-cache", default=None, help="模型缓存目录（魔搭环境可指定持久化目录）")
    args = parser.parse_args()

    records = load_corpus()
    queries = json.loads((ROOT / "data" / "test_queries.json").read_text(encoding="utf-8"))
    vector_index = None
    if args.stage1 == "bm25+vector":
        from rerankers import VectorIndex
        vector_index = VectorIndex(args.vector_model, args.model_cache)
    reranker = get_reranker(args.reranker, args.rerank_model, args.model_cache)
    result = evaluate(records, queries, reranker, args.candidates, args.stage1, vector_index)
    suffix = f"{args.reranker}_{args.stage1}".replace("+", "_")
    report(result, suffix)


if __name__ == "__main__":
    main()