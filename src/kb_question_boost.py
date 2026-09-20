# -*- coding: utf-8 -*-
"""知识库问题生成与检索优化。

思路：用户的提问与政策原文存在“词汇鸿沟”（老百姓问“多少钱”，条文写“按照《目录》补贴标准比例”），
只对原文建索引容易漏召回。本脚本用大模型为每个知识切片预生成多条**口语化问题**，
建立“原文索引 + 问题索引”双通道，并在同一测试集上对比两种索引的召回效果。

用法:
    python src/kb_question_boost.py --stage generate --num-questions 5
    python src/kb_question_boost.py --stage evaluate
    python src/kb_question_boost.py                      # 一键：先生成再评测
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import jieba

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import (RESULTS, ROOT, bm25_scores, build_bm25, hit_metrics, load_corpus,  # noqa: E402
                        pct, rank_by_scores, render_table_png, save_csv, summarize)
from llm_backend import extract_json, get_backend  # noqa: E402

Q_CACHE = RESULTS / "generated_questions.json"

PROMPT = """你是政务咨询热线的坐席培训师。下面是一条政策条款，请站在办事群众的角度，生成 {n} 个他们真实会问的问题。

要求：
1. 每个问题都必须能由这条条款回答，不得引入条款以外的信息；
2. {n} 个问题要用不同问法，例如：直接问金额、问资格条件、问办理流程、问时间期限、对比不同情形；
3. 口语化：像群众打电话咨询那样说话，例如“补贴能拿多少”“钱发给谁”“多久能到账”；
4. 禁止照抄条款原文的表述，禁止出现“《某某办法》第几条”这类引用，禁止重复同一个意思；
5. 问题要短，一句话以内。

只输出 JSON（不要解释、不要输出答案）：
{{"questions": [{{"question": "问题内容", "question_type": "直接问|口语化|条件问|对比问"}}]}}

政策条款（来源：{source}）：
{content}
"""


def dedup_questions(questions: list) -> list:
    """去掉完全重复与高度相似的生成问题（分词 Jaccard 相似度 > 0.8 视为近似重复）。"""
    seen_tokens, kept = [], []
    for item in questions:
        question = (item.get("question") or "").strip()
        if len(question) < 5:
            continue
        tokens = {w for w in jieba.lcut(question) if len(w) > 1}
        if tokens and any(len(tokens & old) / max(len(tokens | old), 1) > 0.8 for old in seen_tokens):
            continue
        seen_tokens.append(tokens)
        kept.append(item)
    return kept


def salvage_questions(text: str) -> list:
    """当模型输出被截断、JSON 不完整时，用正则把已经生成完整的问题抢救出来。"""
    pattern = r'"question"\s*:\s*"([^"]{4,80})"(?:\s*,\s*"question_type"\s*:\s*"([^"]{0,12})")?'
    return [{"question": q, "question_type": t or "未标注"} for q, t in re.findall(pattern, text or "")]


def load_cache() -> dict:
    if Q_CACHE.exists():
        return json.loads(Q_CACHE.read_text(encoding="utf-8"))
    return {}


def generate_questions(records: list, num_questions: int, max_chunks: int = None,
                       backend_name: str = None) -> dict:
    """为知识切片生成问题，结果缓存到 results/generated_questions.json（支持断点续跑）。"""
    cache = load_cache()
    todo = [r for r in records if not cache.get(r["id"])]
    if max_chunks:
        todo = todo[:max_chunks]
    if not todo:
        print(f"问题缓存已完整（{len(cache)} 个切片），跳过生成")
        return cache
    llm = get_backend(backend_name)
    print(f"生成后端：{llm.tag}；待处理切片 {len(todo)} 个，每个生成 {num_questions} 个问题")
    start = time.time()
    for idx, rec in enumerate(todo, 1):
        prompt = PROMPT.format(n=num_questions, source=rec["source_label"], content=rec["content"])
        questions, raw = [], ""
        for attempt in (1, 2):
            try:
                raw = llm.chat(prompt, json_mode=True, temperature=0.4)
                data = extract_json(raw)
                questions = []
                for item in data.get("questions", []):
                    if isinstance(item, str) and item.strip():  # 模型有时直接返回字符串列表
                        questions.append({"question": item.strip(), "question_type": "未标注"})
                    elif isinstance(item, dict) and item.get("question"):
                        questions.append(item)
                if questions:
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"  [{idx}/{len(todo)}] {rec['id']} 第 {attempt} 次解析失败：{exc}")
        if not questions:
            questions = salvage_questions(raw)
            if questions:
                print(f"  [{idx}/{len(todo)}] {rec['id']} JSON 被截断，已从文本抢救出 {len(questions)} 个问题")
        cache[rec["id"]] = questions
        Q_CACHE.parent.mkdir(parents=True, exist_ok=True)
        Q_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        if idx % 5 == 0 or idx == len(todo):
            speed = (time.time() - start) / idx
            print(f"  [{idx}/{len(todo)}] {rec['id']} 生成 {len(questions)} 问；"
                  f"均耗时 {speed:.1f}s，预计剩余 {speed * (len(todo) - idx) / 60:.1f} 分钟")
    return cache


def build_question_index(records: list, cache: dict):
    """问题索引：所有生成问题构成一个 BM25 语料，检索后按切片取最高分汇总。"""
    id_to_idx = {r["id"]: i for i, r in enumerate(records)}
    entries = []
    for rec in records:
        for item in dedup_questions(cache.get(rec["id"], [])):
            question = (item.get("question") or "").strip()
            if question:
                entries.append((question, id_to_idx[rec["id"]]))
    return build_bm25([q for q, _ in entries]), entries

def evaluate(records: list, queries: list, cache: dict, top_k: int = 20,
             q_weight: float = 0.25) -> dict:
    """在同一测试集上对比 原文索引 / 问题索引 / 双索引融合 三条通道。

    q_weight 是问题索引在融合打分里的权重：fused = (1-w)*原文分数 + w*问题分数。
    实测问题索引单独使用召回不稳定，与原文索引按 0.75 : 0.25 融合时效果最好。
    """
    ids = [r["id"] for r in records]
    content_index = build_bm25([r["content"] for r in records])
    q_index, q_entries = build_question_index(records, cache)
    print(f"原文索引：{len(records)} 条切片；问题索引：{len(q_entries)} 条生成问题")
    methods = ("content", "question", "fusion")
    hits = {m: [] for m in methods}
    rows = []
    for query in queries:
        gold = set(query["gold"])
        c_scores = bm25_scores(content_index, query["query"])
        q_scores, best_question = aggregate(bm25_scores(q_index, query["query"]), q_entries, len(records))
        f_scores = [(1 - q_weight) * a + q_weight * b
                    for a, b in zip(normalize(c_scores), normalize(q_scores))]
        ranks = {}
        for name, scores in (("content", c_scores), ("question", q_scores), ("fusion", f_scores)):
            ranked = [ids[i] for i, _ in rank_by_scores(scores, top_k)]
            metric = hit_metrics(ranked, gold)
            hits[name].append(metric)
            ranks[name] = metric["gold_rank"]
        gold_idx = [ids.index(g) for g in query["gold"] if g in ids]
        rows.append({
            "id": query["id"], "query": query["query"], "qtype": query["qtype"],
            "difficulty": query["difficulty"], "gold": "|".join(query["gold"]),
            "content_rank": ranks["content"], "question_rank": ranks["question"],
            "fusion_rank": ranks["fusion"],
            "matched_question": "；".join(best_question[i] for i in gold_idx if best_question[i])
            if (gold_idx := [ids.index(g) for g in query["gold"] if g in ids]) else "",
            "improved": ranks["question"] > 0 and (ranks["content"] == -1 or ranks["question"] < ranks["content"]),
        })
    return {"rows": rows, "summary": {m: summarize(hits[m]) for m in methods}}


def aggregate(scores: list, entries: list, n_chunks: int):
    """问题级分数 -> 切片级分数（取该切片所有生成问题中的最高分），并记录最匹配的问题。"""
    best = [0.0] * n_chunks
    best_question = [""] * n_chunks
    for score, (question, idx) in zip(scores, entries):
        if score > best[idx]:
            best[idx] = score
            best_question[idx] = question
    return best, best_question


def normalize(scores: list) -> list:
    top = max(scores) if scores and max(scores) > 0 else 1.0
    return [s / top for s in scores]


NAMES = {"content": "原文索引（BM25）", "question": "问题索引（生成问题 BM25）", "fusion": "双索引融合（原文 0.75 + 问题 0.25）"}


def report(result: dict, records: list, cache: dict, num_questions: int, top_k: int = 3):
    summary, rows = result["summary"], result["rows"]
    headers = ["检索通道", "Hit@1", "Hit@3", "Hit@5", "MRR"]
    table = []
    for key in ("content", "question", "fusion"):
        s = summary[key]
        delta = s["mrr"] - summary["content"]["mrr"]
        extra = "" if key == "content" else f"（{delta:+.3f}）"
        table.append([NAMES[key], pct(s["hit@1"]), pct(s["hit@3"]), pct(s["hit@5"]),
                      f"{s['mrr']:.3f}{extra}"])
    render_table_png(
        f"知识库问题生成与检索优化：{len(rows)} 条真实提问下的召回效果对比",
        headers, table, RESULTS / "qboost_summary.png",
        note=(f"知识库 {len(records)} 个切片，每切片生成 {num_questions} 个口语化问题，"
              f"问题索引共 {sum(len(v) for v in cache.values())} 条；测试集 {len(rows)} 条"
              f"（含口语化提问、术语差异、版本时效、规则冲突等考点）；括号内为相对原文索引的 MRR 变化"),
        highlight_rows=[1, 2], col_widths=[0.36, 0.16, 0.16, 0.16, 0.16])

    detail = []
    for i, row in enumerate(rows, 1):
        r1 = row["content_rank"] if row["content_rank"] > 0 else "未命中"
        r2 = row["question_rank"] if row["question_rank"] > 0 else "未命中"
        if row["content_rank"] == -1 and row["question_rank"] > 0:
            verdict = "★ 问题索引救回"
        elif row["content_rank"] > 0 and row["question_rank"] > 0 and row["question_rank"] < row["content_rank"]:
            verdict = "排名提升"
        elif row["content_rank"] > 0 and row["question_rank"] == -1:
            verdict = "反而漏召回"
        else:
            verdict = "持平"
        detail.append([i, row["query"], row["qtype"], r1, r2, verdict])
    render_table_png(
        "逐条提问的排名变化：原文索引 vs 问题索引（排名越小越好）",
        ["#", "用户提问", "考点类型", "原文索引排名", "问题索引排名", "结论"],
        detail, RESULTS / "qboost_detail.png",
        note="“未命中”表示该通道前 20 名中没有正确条款；★ 表示问题索引独有召回",
        highlight_rows=[i for i, row in enumerate(detail) if row[5].startswith("★")],
        col_widths=[0.035, 0.39, 0.12, 0.14, 0.14, 0.175])

    examples = []
    for rec in records:
        questions = cache.get(rec["id"], [])[:top_k]
        if questions:
            examples.append([rec["source_label"], " / ".join(q["question"] for q in questions)])
        if len(examples) >= 10:
            break
    if examples:
        render_table_png(
            "问题生成示例：由政策条款自动生成的多样化问题",
            ["知识切片来源", "生成的口语化问题（部分）"],
            examples, RESULTS / "qboost_examples.png",
            note=f"生成后端：{result.get('backend', 'auto')}；完整结果见 results/generated_questions.json",
            col_widths=[0.36, 0.64])

    save_csv(rows, RESULTS / "qboost_details.csv")
    print("\n=== 汇总指标 ===")
    for key in ("content", "question", "fusion"):
        s = summary[key]
        print(f"  {NAMES[key]:<28} Hit@1 {pct(s['hit@1'])}  Hit@3 {pct(s['hit@3'])}  "
              f"Hit@5 {pct(s['hit@5'])}  MRR {s['mrr']:.3f}")
    rescued = [r["id"] for r in rows if r["content_rank"] == -1 and r["question_rank"] > 0]
    improved = [r["id"] for r in rows
                if r["content_rank"] > 0 and r["question_rank"] > 0 and r["question_rank"] < r["content_rank"]]
    dropped = [r["id"] for r in rows if r["content_rank"] > 0 and r["question_rank"] == -1]
    print(f"  问题索引独有召回（原文索引漏掉）: {len(rescued)} 条 {rescued}")
    print(f"  问题索引排名提升: {len(improved)} 条 {improved}")
    print(f"  问题索引漏召回: {len(dropped)} 条 {dropped}")


def main():
    parser = argparse.ArgumentParser(description="知识库问题生成与检索优化")
    parser.add_argument("--stage", choices=("generate", "evaluate", "all"), default="all")
    parser.add_argument("--num-questions", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=None, help="只处理前 N 个切片（快速试跑）")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--backend", default=None, help="ollama|transformers|dashscope，默认自动探测")
    args = parser.parse_args()

    records = load_corpus()
    queries = json.loads((ROOT / "data" / "test_queries.json").read_text(encoding="utf-8"))
    cache = load_cache()
    if args.stage in ("generate", "all"):
        cache = generate_questions(records, args.num_questions, args.max_chunks, args.backend)
    missing = [r["id"] for r in records if not cache.get(r["id"])]
    if missing:
        print(f"提示：仍有 {len(missing)} 个切片没有问题（例如 {missing[:3]}）")
    if args.stage in ("evaluate", "all"):
        result = evaluate(records, queries, cache, args.top_k)
        result["backend"] = args.backend or "auto"
        report(result, records, cache, args.num_questions)


if __name__ == "__main__":
    main()
