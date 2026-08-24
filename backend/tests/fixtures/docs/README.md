# 冻结评测文档（阶段 0 交付物）

本目录下的三篇 Markdown 文档是 Spec（`specs/rag-p0-stability-hardening-spec.md`
10.2 节）离线集成测试的**冻结测试文档**，由
`backend/tests/test_offline_integration.py` 使用。

- `sic_thermal.md` — 碳化硅热导率主题，唯一术语 A：`ZetaTherm-QX917`
- `dft_functional.md` — DFT 交换关联泛函主题，唯一术语：`PseudoCore-KW304`
- `perovskite_defect.md` — 钙钛矿缺陷主题，唯一术语：`DefectPair-LM552`

## 冻结约定

- 文档内容为合成材料科学文本，含全局唯一术语，用于精确断言检索命中。
- **禁止随意改动文档内容**：内容变化会改变 chunk 切分结果与确定性 ID，
  导致集成测试断言失效。确需修改时，须同步更新集成测试中的相关断言。
- 测试中对"修改文档"场景的模拟一律在 tmp_path 副本上进行，不得改动本目录原件。

## 预期 chunk 数

首次运行记录的基线（ParentFirstSplitter，parent_chunk_size=2000 / child_chunk_size=500）：

- sic_thermal.md: parent=1, child=5
- dft_functional.md: parent=1, child=4
- perovskite_defect.md: parent=1, child=4
