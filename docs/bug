● 指标解读

  核心指标

  ┌───────────────┬─────────┬──────────────────┬───────────────────────────────────┐
  │     指标      │ 理想值  │       含义       │             差的表现              │
  ├──────────────────┼────────┼─────────────────┼────────────────────────────────┤
  │ boundary_score   │ > 0.8  │ chunk 在句首/句 │ < 0.5：句子被拦腰截断，chunk   │
  │                  │        │ 尾切开的比例    │ 开头是半句话                   │ 
  ├──────────────────┼────────┼─────────────────┼────────────────────────────────┤
  │ length_uniformit │        │ chunk           │ < 0.3：有的 chunk 特别短(20字) │
  │ y                │ > 0.5  │ 长度一致性      │ ，有的特别长(3000字)，检索时短 │ 
  │                  │        │                 │  chunk 缺乏上下文              │ 
  ├──────────────────┼────────┼─────────────────┼────────────────────────────────┤
  │                  │ 0.85~0 │                 │ > 0.99：几乎全是实质内容，但可 │
  │ avg_info_density │  .95   │ 有效信息占比    │ 能切得过碎；<                  │
  │                  │        │                 │ 0.7：大量空白/噪音             │
  ├──────────────────┼────────┼─────────────────┼────────────────────────────────┤
  │ duplicate_ratio  │ < 0.02 │ 重复 chunk 比例 │ > 5%：chunk_overlap            │ 
  │                  │        │                 │ 过大或语义合并过激             │
  ├──────────────────┼────────┼─────────────────┼────────────────────────────────┤
  │ avg_adjacent_sim │ 0.1~0. │ 相邻 chunk      │ > 0.5：overlap 太大导致重复；= │
  │ ilarity          │   3    │ 的文本重叠度    │  0：chunk 间完全断裂           │
  ├──────────────────┼────────┼─────────────────┼────────────────────────────────┤
  │ sentence_start_r │ > 0.8  │ 以完整句首开头  │ < 0.6：大量 chunk 从中断开     │
  │ atio             │        │ 的比例          │                                │
  └──────────────────┴────────┴─────────────────┴────────────────────────────────┘

  你的实际结果分析

  以 recursive vs recursive+semantic_merge 对比为例：

                 chunks  边界  均匀  信息密度  相邻相似度  句首%
  仅递归           1032  0.37  0.32   0.992      0.156    52.6%
  递归+语义          732  0.41  0.00   0.993        —       —

  recursive 的问题：

  1. boundary_score = 0.37 偏低 — 只有 52.6% 的 chunk
  以完整句子开头。原因是递归切分按字符数硬切，不管句子边界。措施：在 separator 中增加
  '。', '！', '？' 等中文标点优先级，让切分优先在标点处断开。
  2. length_uniformity = 0.32 偏低 — chunk 长度参差不齐。短 chunk（如 20 字的标题）和长
   chunk（3000 字的段落）混在一起。措施：semantic_merge 合并过短的相邻 chunk。

  recursive+semantic_merge 的问题：

  1. length_uniformity = 0.00 — 合并后长度极不均匀（可能少数 chunk
  合并成超长块）。措施：降低 similarity_threshold（从 0.75 → 0.6），减少过度合并。
  2. boundary_score 略有提升 (0.37 → 0.41)，但依然不高。语义合并只能修复部分边界。



import sys
  sys.path.insert(0, '/mnt/d/Langchain_study/Agent智能体/Agent Material')
  from rag_mcp_server.rag.vector_store import VectorStore

  store = VectorStore()

  # 查看当前文档数
  print(f"入库前: {store.get_document_count()} 条")

  # 扫描目录 → 入队 → Worker 后台处理
  store.load_document()

  # 等待几秒后查询
  import time; time.sleep(5)
  print(f"入库后: {store.get_document_count()} 条")

  ---
  方式三：只执行 Pipeline 不写库（预览 chunk 结果）

  from rag_mcp_server.rag.document_pipeline import ProcessingPipeline

  pipeline = ProcessingPipeline()

  # 处理单个文件，返回切分+增强后的 chunk 列表
  chunks = pipeline.process("datas/某论文.pdf")
  for i, doc in enumerate(chunks[:3]):
      print(f"--- chunk {i} ---")
      print(f"content: {doc.page_content[:200]}...")
      print(f"parent_id: {doc.metadata.get('parent_id', 'N/A')}")
      print(f"parent_doc: {doc.metadata.get('parent_doc', '')[:100]}...")

  ---

  

● 根据日志，逐行映射到完整的函数调用链：

  ---
  完整调用链路
  
  ┌─ rag_load_documents MCP 工具调用 ──────────────────────────────┐
  │                                                                 │
  │  server.py: rag_load_documents()                                │
  │    └─ VectorStore.load_document()                               │
  │         │                                                        │
  │         ├─ [1] FileTracker.scan_directory("data/", [".pdf"])    │
  │         │      └─ file_tracker.py:217                           │
  │         │         扫描完成 | 新增 0 | 修改 0 | 未变 73 | 已删 0  │
  │         │                                                       │
  │         ├─ TaskQueue.enqueue() → 0 个新文件入队（全部未变）       │
  │         │                                                       │
  │         ├─ [2] BackgroundWorker.start()                         │
  │         │      └─ worker.py:45 + worker.py:59                   │
  │         │         "Worker 线程开始运行"                           │
  │         │                                                       │
  │         └─ [3] 返回                                              │
  │              └─ vector_store.py:437                              │
  │                 "扫描入队完成 | 共入队 0"                          │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
                                │
                                ▼  (Worker 线程继续消费之前入队的任务)

  ┌─ Worker 处理任务 #997 ──────────────────────────────────────────┐
  │                                                                 │
  │  worker.py:82  "处理任务 #997: ingest_file"                      │
  │    └─ _handle_ingest(task)                                      │
  │         │                                                        │
  │         ├─ [4] ProcessingPipeline.process(file_path)             │
  │         │    │                                                   │
  │         │    ├─ 阶段1: DocumentLoader.load(file)                │
  │         │    │    ├─ mineru_pdf_loader(file)                    │
  │         │    │    │    ├─ mineru_loader.py:293                   │
  │         │    │    │    │   "MinerU 开始解析: 12页 1.9MB"          │
  │         │    │    │    ├─ MinerU API → Markdown                  │
  │         │    │    │    ├─ MarkdownCleaner.clean()               │
  │         │    │    │    │   └─ markdown_cleaner.py:173             │
  │         │    │    │    │       "清洗完成: 1篇→1篇"                │
  │         │    │    │    └─ file_loader.py:142                     │
  │         │    │    │        "加载+清洗成功 | 1篇文档"               │
  │         │    │    └─ document_pipeline.py:75                     │
  │         │    │        "加载完成 | 1 页/块"                        │
  │         │    │                                                   │
  │         │    ├─ 阶段2: _clean_documents(docs, file)              │
  │         │    │    └─ MarkdownCleaner.clean()                    │
  │         │    │        └─ markdown_cleaner.py:173                 │
  │         │    │            "清洗完成: 1篇→1篇"  ← 二次清洗（幂等） │
  │         │    │                                                   │
  │         │    ├─ 阶段3: HybridSplitter.split_documents(docs)     │
  │         │    │    │  扩展名 .pdf → [markdown_header,              │
  │         │    │    │   sentence_boundary, semantic_merge, parent] │
  │         │    │    │                                               │
  │         │    │    ├─ MarkdownHeaderSplitter.split_documents()   │
  │         │    │    │   └─ text_spliter.py:585                     │
  │         │    │    │       "切分完成 | 89 chunks"                  │
  │         │    │    │   (按 #/##/###/#### 标题切分，大chunk二次细切) │
  │         │    │    │                                               │
  │         │    │    ├─ SentenceBoundarySplitter.split_documents() │
  │         │    │    │   89 chunks → 97 chunks                      │
  │         │    │    │   (双向句边界修剪: 找句尾标点对齐边界)         │
  │         │    │    │                                               │
  │         │    │    ├─ SemanticMergeOnly.split_documents()        │
  │         │    │    │   ├─ async_embeddings.py:65                  │
  │         │    │    │   │   "开始嵌入 97 条文本 | batch_size=10"    │
  │         │    │    │   │   ↓ 10批并发 → DashScope 原生 API        │
  │         │    │    │   ├─ async_embeddings.py:84                  │
  │         │    │    │   │   "嵌入完成 | 97→97 个向量"               │
  │         │    │    │   └─ text_spliter.py:672                     │
  │         │    │    │       "SemanticMergeOnly: 97→97"              │
  │         │    │    │       (相似度均 < 0.9，无需合并)              │
  │         │    │    │                                               │
  │         │    │    └─ ParentDocumentSplitter.split_documents()   │
  │         │    │        └─ text_spliter.py:555                     │
  │         │    │            "切分完成 | 258 chunks"                 │
  │         │    │            父chunk(2000字符)→子chunk(500字符)       │
  │         │    │            97个父chunk → 258个子chunk               │
  │         │    │            (每个子chunk含 parent_id + parent_doc)  │
  │         │    │                                                   │
  │         │    └─ text_spliter.py:716                              │
  │         │        "[HybridSplitter] 1→258 chunks"                  │
  │         │        (步骤链: MarkdownHeader→SentenceBoundary→        │
  │         │         SemanticMerge→Parent)                           │
  │         │                                                        │
  │         │    └─ 阶段4: MetadataEnhancer.enhance(chunks)          │
  │         │         └─ metadata_enhancer.py:114                    │
  │         │             "增强完成: 258 chunks → 6 code, 0 table"    │
  │         │                                                        │
  │         │    └─ document_pipeline.py:188                         │
  │         │        "[Pipeline] 产出 258 chunks"                      │
  │         │                                                        │
  │         ├─ [5] MultiGranularityIndex.insert(child_chunks, src)   │
  │         │    │   (当前 multi_granularity.enabled=false → 单粒度)  │
  │         │    │                                                    │
  │         │    └─ _get_base_store().add_documents(258 chunks)      │
  │         │         └─ PGVector 内部调用 embedding_model            │
  │         │              .embed_documents(258条文本)               │
  │         │              └─ async_embeddings.py:65                  │
  │         │                  "开始嵌入 258 条文本 | batch_size=10"   │
  │         │                  ↓ 26批次(25×10 + 1×8) → DashScope API │
  │         │                  "嵌入完成 | 258→258 个向量"             │
  │         │                  ↓ INSERT INTO langchain_pg_embedding   │
  │         │                                                        │
  │         └─ [6] FileTracker.record_indexed(file_path)             │
  │                 "任务 #997 完成"                                    │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

  ---
  关键数据流
  
  PDF (12页, 1.9MB)
    │ MinerU API → Markdown + MarkdownCleaner (第1次清洗)
    │ DocumentLoader: 1篇文档
    │ MarkdownCleaner (第2次清洗, 幂等)
    │
    ├─ MarkdownHeaderSplitter:  1 → 89 chunks   (按标题切分)
    ├─ SentenceBoundarySplitter: 89 → 97 chunks  (句边界对齐, +8)
    ├─ SemanticMergeOnly:        97 → 97 chunks  (无合并, similarity<0.9)
    └─ ParentDocumentSplitter:   97 → 258 chunks (父→子, ~2.7子/父)
    │
    ├─ MetadataEnhancer: 6个代码块, 0个表格
    │
    ├─ 入库 embedding: 258条文本 → 26批次 → DashScope API
    └─ PGVector INSERT 258条向量 → agent collection

  两次 Embedding 调用

  ┌───────────────────────────────────┬────────┬───────────────────────────┐
  │               位置                │ 文本数 │           用途            │
  ├───────────────────────────────────┼────────┼───────────────────────────┤
  │ SemanticMergeOnly (切分阶段)      │ 97     │ 判断相邻chunk是否需要合并 │
  ├───────────────────────────────────┼────────┼───────────────────────────┤
  │ PGVector.add_documents (入库阶段) │ 258    │ 生成最终存储向量          │
  └───────────────────────────────────┴────────┴───────────────────────────┘

  注意：258 → 26 批次 API 调用，单批次 ≤ 10 条，符合 DashScope 限制。


============================================================
RAGAS 评估结果
============================================================
有效条目: 40 / 40
耗时: 6322.4s (并发: 5)

----------------------------------------
指标                           均值      中位数       最低       最高
----------------------------------------
faithfulness             0.9347   1.0000   0.0000   1.0000
context_recall           0.8875   1.0000   0.0000   1.0000
context_precision        0.2831   0.2500   0.0000   1.0000
answer_correctness       0.6683   0.7056   0.0603   0.9681
----------------------------------------