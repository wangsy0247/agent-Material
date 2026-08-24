# Evaluation data

该目录保存项目可公开、可复核的 RAG 评测集和实验结果。原始 PDF 文献、
数据库内容、API 凭据及运行日志不在仓库中；复现实验前需自行准备知识库。

## 文件说明

| 文件 | 作用 |
|---|---|
| `dataset_parts/*.json` | 91 题固定评测集的可审查源分片 |
| `fixed_eval_dataset.json` | 合并后的问题、参考答案和题型 |
| `fixed_eval_dataset_v2.json` | 增加 Gold evidence、审核状态和核验信息的正式评测集 |
| `gold_review_queue_v2.json` | 自动对齐后仍需复核的样本及候选证据 |
| `index_manifest_v2.json` | 隔离评测索引的构建清单 |
| `index_audit_v2.json` | 评测索引完整性审计结果 |
| `ablation_results_v2.json` | 统一检索管线的消融实验明细与汇总 |
| `weight_sweep_results_v1.json` | Vector/BM25 Weighted RRF 权重扫描结果 |
| `ragas_v3_filled.json` | RAGAS 输入，包括回答和检索上下文 |
| `ragas_v3_scores.json` | RAGAS 逐题评分和汇总 |
| `visual_rag_eval_template.json` | P1 图表问题、Gold artifact 与答案事实模板，不含真实论文内容 |

P1 图表评测通过 `scripts/run_visual_rag_eval.py` 执行。默认推荐先做
retrieval-only；只有确认图片允许发送到外部视觉模型时，才同时使用
`--include-answers --allow-external-images`。报告以 `artifact_id` 精确匹配，
且不写入问题正文、回答正文或本机路径。

## 数据口径

- 总题数为 91，其中 76 题具有自动核验的严格 Gold evidence。
- 严格检索指标以 Gold `chunk_id` 精确匹配，不使用关键词命中代替。
- `unanswerable` 样本不混入检索质量主均值，单独观察拒答边界。
- Gold evidence 为自动词法对齐并经过规则核验，不宣称人工双人标注。
- 实验结果来自单文档、中等规模语料，不代表大规模多租户场景的绝对性能。

## 可复现与隐私

结果文件只保留仓库相对位置、随机种子、依赖版本、配置快照、数据哈希和
Git 提交号。不会记录用户名、本机工作目录、解释器绝对位置、系统平台、
WSL 发行版、Conda 安装位置、数据库连接串或 API 密钥。

在公开前可执行以下检查：

```bash
conda activate langchain
python scripts/sanitize_eval_artifacts.py --check
python scripts/build_fixed_eval_dataset.py
```

重新生成结果时，应先执行清洗检查，再提交 JSON 文件。
