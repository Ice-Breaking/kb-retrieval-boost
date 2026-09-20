---
license: Apache License 2.0
tags: []
tasks:
- text-ranking

#model-type:
##如 gpt、phi、llama、chatglm、baichuan 等
#- gpt

#domain:
##如 nlp、cv、audio、multi-modal
#- nlp

#language:
##语言代码列表 https://help.aliyun.com/document_detail/215387.html?spm=a2c4g.11186623.0.0.9f8d7467kni6Aa
#- cn 

#metrics:
##如 CIDEr、Blue、ROUGE 等
#- CIDEr

#tags:
##各种自定义，包括 pretrained、fine-tuned、instruction-tuned、RL-tuned 等训练方法和其他
#- pretrained

#tools:
##如 vllm、fastchat、llamacpp、AdaSeq 等
#- vllm
---

# kb-retrieval-boost

面向**中文政策知识库**的检索质量优化工具包：把"用户的口语化提问"与"政策条文的书面表述"之间的鸿沟填平，并让最终返回的答案片段排序更准。

## 1. 要解决的问题

在政务服务、企业内部制度这类知识库上做检索增强问答，最常遇到两类问题：

| 问题 | 具体表现 | 本项目的做法 |
| --- | --- | --- |
| **词汇鸿沟** | 用户问"培训能补多少钱"，条文写的是"按照《目录》补贴标准给予培训费补贴"，词面几乎不重叠，关键词检索直接漏召回 | **问题生成增强索引**：用大模型为每条知识切片预生成多条口语化问题，建立"原文 + 问题"双索引 |
| **粗排精度不足** | 召回候选里正确条款只排在第 5、第 8 位，直接截断 Top-3 会把无关内容喂给大模型 | **两阶段检索**：BM25 / 向量粗排召回 N 条候选，再用 bge-reranker 交叉编码器精排出 Top-K |

## 2. 两条主线

**主线一：知识库问题生成与检索优化**（`src/kb_question_boost.py`）

```
政策条文 ──► 大模型生成 5 个口语化问题 ──► 问题索引（BM25）┐
                                                          ├──► 同一测试集对比命中率 / MRR
政策条文 ─────────────────────────────► 原文索引（BM25）  ┘
```

**主线二：bge-reranker 两阶段检索**（`src/rerank_compare.py`）

```
知识库（134 个切片）── BM25 / 向量粗排 ──► 候选 20 条 ──► bge-reranker 精排 ──► Top-K
```

## 3. 目录结构

```
kb-retrieval-boost/
├── data/
│   ├── knowledge_base.json      # 知识切片（由原始文件清洗、切分而来）
│   ├── test_queries.json        # 25 条人工标注测试集（含正确条款 id 与考点类型）
│   └── raw/                     # 原始政策文件（已被 .gitignore 排除，仅本地构建语料时使用）
├── src/
│   ├── build_corpus.py          # 原始文件 -> 结构化知识切片
│   ├── text_utils.py            # 中文分词、BM25、命中指标、PNG 报表
│   ├── llm_backend.py           # 统一生成后端：ollama / transformers / dashscope
│   ├── rerankers.py             # 重排器与向量检索组件
│   ├── kb_question_boost.py     # 主线一：问题生成 + 双索引对比
│   └── rerank_compare.py        # 主线二：粗排 + 重排对比
├── results/                     # 实验结果：明细 CSV + 对比表 PNG
└── requirements.txt
```

## 4. 数据

语料来自 6 份公开政策文件，清洗切分为 **134 个可检索切片（27,014 字）**，跨越市、国家两个层级与两个年份版本：

| 文档 | 层级 / 类型 | 切片数 |
| --- | --- | --- |
| 天津市职业技能培训补贴实施办法 | 市级办法 | 35 |
| 天津市职业技能培训监督管理暂行办法 | 市级办法 | 35 |
| 天津市职业培训补贴经办操作规程 | 经办规程 | 25 |
| 天津市市场紧缺职业需求程度及培训补贴标准目录（2024 年版） | 补贴目录 | 3 |
| 天津市市场紧缺职业需求程度及培训补贴标准目录（2026 年版） | 补贴目录 | 3 |
| 就业补助资金管理办法 | 国家级办法 | 33 |

这批语料有三个天然的检索难点，非常适合用来检验检索优化效果：

1. **版本时效**：2024 版与 2026 版目录同时存在，2026 版自 2026 年 5 月 1 日起施行，此前的班期仍按 2024 版执行；
2. **上下位法规**：国家级《就业补助资金管理办法》与市级实施办法有大量近义条款，容易互相干扰；
3. **真实规则冲突**：`经办规程` 规定"每人每年最多可享受三次培训补贴"，`实施办法` 第十五条则是"院校学生一次、其他人员两次"，需要同时返回两条。

测试集 `data/test_queries.json` 共 25 条提问，均由人工标注正确条款 id：

| 考点类型 | 条数 | 说明 |
| --- | --- | --- |
| 跨文档 | 5 | 需区分国家级办法与市级办法 |
| 口语化提问 | 3 | 词汇鸿沟最明显 |
| 条款细节 / 金额条款 / 流程细节 | 9 | 近义干扰条款多 |
| 术语差异 | 2 | 如"鉴定费"与"考试费" |
| 版本时效 | 2 | 2024 版 vs 2026 版目录 |
| 规则冲突 / 有效期 / 主体职责 / 处理措施 | 4 | 一条提问对应多条正确条款 |

## 5. 快速开始

```bash
pip install -r requirements.txt

# 可选：重建知识切片（把原始 doc/docx/pdf 放进 data/raw/ 后执行）
python src/build_corpus.py

# 主线一：生成问题 + 双索引评测（结果写入 results/）
python src/kb_question_boost.py --stage generate --num-questions 5
python src/kb_question_boost.py --stage evaluate

# 主线二：粗排 + 重排
python src/rerank_compare.py --reranker bge --stage1 bm25
```

### 生成后端选择

`src/llm_backend.py` 提供三种后端，可用 `LLM_BACKEND` 环境变量指定，未设置时自动探测：

| 后端 | 适用场景 | 前置条件 |
| --- | --- | --- |
| `ollama` | 本机开发，不需要下载任何模型文件 | 本机已启动 `ollama serve` 并已 pull 模型（`OLLAMA_MODEL` 指定） |
| `transformers` | 云 GPU / 本地 GPU 环境 | `HF_LLM_MODEL` 指定的模型（默认 `Qwen/Qwen2.5-1.5B-Instruct`） |
| `dashscope` | 想用云端大模型、追求生成质量 | 环境变量 `DASHSCOPE_API_KEY` |

### 重排器选择

| 重排器 | 说明 |
| --- | --- |
| `--reranker bge`（默认） | `BAAI/bge-reranker-base` 交叉编码器，效果最好，首次运行从 ModelScope 下载模型 |
| `--reranker ollama` | 用本机大模型给候选文档打分，零模型下载，适合快速验证 |
| `--reranker mock` | 词面重合度打分，仅用于验证流水线是否跑通 |

## 6. 在云 GPU 环境上运行重排实验

重排模型与向量模型都需要下载权重，推荐在带 GPU 的环境里执行：

```bash
pip install -r requirements.txt

# 把模型缓存指到可持久化目录，避免实例释放后重复下载
export MODELSCOPE_CACHE=/mnt/workspace/models

# ① 正式实验：bge-reranker-base 精排（粗排用 BM25）
python src/rerank_compare.py --reranker bge --stage1 bm25

# ② 进阶：粗排换成 BM25 + 向量融合，重排换成更强的 bge-reranker-v2-m3
python src/rerank_compare.py --reranker bge --stage1 bm25+vector \
    --vector-model BAAI/bge-small-zh-v1.5 --rerank-model BAAI/bge-reranker-v2-m3

# ③ 主线一不想调用云端大模型时，用本地小模型生成问题
LLM_BACKEND=transformers HF_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
    python src/kb_question_boost.py --stage generate --num-questions 5
```

所有实验都会把明细写入 `results/*.csv`，并把对比表渲染成 `results/*.png`（内置中文字体自动探测；Linux 上若中文显示为方框，安装 `fonts-noto-cjk` 即可）。

## 7. 实验结果

> 结果均由 `data/test_queries.json` 的 25 条标注提问计算，指标为 Hit@k 与 MRR。

### 7.1 知识库问题生成与检索优化（每切片 5 个生成问题，使用 qwen2.5-rewrite-lora 本地模型）

| 检索通道 | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | --- | --- | --- | --- |
| 原文索引（BM25） | 56.0% | 80.0% | 80.0% | 0.679 |
| 问题索引（生成问题 BM25） | 52.0% | 60.0% | 64.0% | 0.590 |
| 双索引融合（原文 0.75 + 问题 0.25） | 76.0% | 80.0% | 84.0% | 0.793 |

结论：
- 问题索引单独使用时召回不稳定（Hit@1 52.0%），但与原文索引按 **0.75 : 0.25 融合**后，Hit@1 从 56.0% 提升至 **76.0%**（+20 个百分点），MRR 从 0.679 提升至 **0.793**；
- 问题索引独有召回：2 条（原文索引漏掉的提问）；问题索引排名提升：8 条；
- 问题索引漏召回：3 条（q12、q16、q21），说明生成问题质量有待提升（本地小模型限制）；
- 见 `results/qboost_summary.png`、`results/qboost_detail.png`、`results/qboost_examples.png`。

### 7.2 两阶段检索（粗排 BM25 Top-20 → 重排精排）

#### 7.2.1 Mock 重排器（词面重合度，验证流水线）

| 指标 | 仅粗排 | 粗排 + 重排 | 变化 |
| --- | --- | --- | --- |
| Top-1 命中率 | 56.0% | 52.0% | -4.0pp |
| Top-3 命中率 | 80.0% | 76.0% | -4.0pp |
| Top-5 命中率 | 80.0% | 80.0% | 0.0pp |
| MRR | 0.679 | 0.643 | -0.036 |

结论：Mock 重排器基于词面重合度，无法区分高级别相似的候选文档，排序无实质改善（符合预期）。
见 `results/rerank_summary_mock_bm25.png`、`results/rerank_cases_mock_bm25.png`。

#### 7.2.2 bge-reranker-base（正式重排，ModelScope GPU 环境已运行）

| 指标 | 仅粗排 | 粗排 + bge-reranker-base | 变化 |
| --- | --- | --- | --- |
| Top-1 命中率 | 56.0% | **76.0%** | **+20.0pp** |
| Top-3 命中率 | 80.0% | **84.0%** | **+4.0pp** |
| Top-5 命中率 | 80.0% | **92.0%** | **+12.0pp** |
| MRR | 0.679 | **0.818** | **+0.139** |

结论：bge-reranker-base 作为交叉编码器，能捕捉 query 与文档的细粒度语义相关性，
8 条提问名次提升（如"考试费报销"类术语差异提问从国家级办法纠正到市级办法对应条款），
单条提问平均重排耗时 175ms。
见 `results/rerank_summary_bge_bm25.png`、`results/rerank_cases_bge_bm25.png`。
**运行方式**：
```bash
pip install -r requirements.txt
export MODELSCOPE_CACHE=/mnt/workspace/models
python src/rerank_compare.py --reranker bge --stage1 bm25
```

## 8. 已知限制

1. 原始政策文件里有两份是双栏/扫描版 PDF，文本抽取后仍可能残留少量版式噪声（个别数字粘连），
   `src/build_corpus.py` 已做启发式清洗，但未做人工逐字校对；
2. 政策文件本身存在**版本与文号冲突**（详见 4. 数据），本仓库按原样保留，没有做人工合并，
   这正是测试集中"规则冲突"考点的来源；
3. 生成式后端用本地小模型时，生成的问题质量明显低于云端大模型，`salvage_questions()` 会在
   JSON 被截断时抢救已生成的问题，但建议生成阶段使用更强的模型；
4. 重排模型 `bge-reranker-base` 的最大输入长度为 512 token，超长切片会被截断。

## 9. License

Apache-2.0（见 LICENSE）。语料来自公开政策文件，仅用于检索技术研究。

