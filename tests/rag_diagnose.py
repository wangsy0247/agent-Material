"""RAG 系统诊断脚本。

用于分析生产环境（或现有）Chroma 向量库的检索质量，
可以单独运行查看打分过程：

    python tests/rag_diagnose.py

功能：
1. 查看 Chroma 库中文档数量和元数据分布
2. 执行测试查询，查看向量检索、BM25、混合检索、重排序各阶段的打分
3. 输出检索链路的时间消耗
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_chroma import Chroma
try:
    from langchain.retrievers import EnsembleRetriever
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever

from model.model_factory import embedding_model
from utils.config_handler import chroma_config


def diagnose_chroma_store():
    """诊断 Chroma 向量库的基本信息。"""
    print("=" * 70)
    print("📦 Chroma 向量库诊断")
    print("=" * 70)

    store = Chroma(
        collection_name=chroma_config["character_name"],
        embedding_function=embedding_model,
        persist_directory=chroma_config["persist_directory"],
    )

    # 1. 基础统计
    data = store.get()
    doc_count = len(data["documents"]) if data and data.get("documents") else 0
    print(f"\n📄 文档总数: {doc_count}")

    # 2. 元数据分析
    metadatas = data.get("metadatas") or []
    sources = {}
    for meta in metadatas:
        src = meta.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print(f"\n📁 来源文件分布:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1])[:10]:
        print(f"   {src}: {count} chunks")
    if len(sources) > 10:
        print(f"   ... 还有 {len(sources) - 10} 个来源")

    # 3. 检查是否有语义切分标记
    has_semantic = any("semantic_chunk_index" in (m or {}) for m in metadatas)
    print(f"\n🔍 使用语义切分: {'是' if has_semantic else '否'}")

    return store


def diagnose_retrieval_pipeline(store: Chroma, query: str):
    """诊断完整检索链路的打分过程。"""
    print("\n" + "=" * 70)
    print(f"🔎 查询: {query}")
    print("=" * 70)

    k = chroma_config.get("k", 10)
    score_threshold = chroma_config.get("score_threshold", 0.3)

    # ── Stage 1: 向量检索 ──
    print(f"\n📍 Stage 1: 向量检索 (k={k}, score_threshold={score_threshold})")

    t0 = time.time()
    # 使用 similarity_search_with_score 获取原始 distance，然后手动过滤
    # Chroma 返回 cosine distance（范围 [0, 2]），需要转换为 similarity = 1 - distance/2
    raw_results = store.similarity_search_with_score(query, k=k)

    # 按 similarity 阈值过滤
    vector_docs = []
    for doc, distance in raw_results:
        similarity = 1.0 - distance / 2.0
        if similarity >= score_threshold:
            vector_docs.append(doc)

    # 如果过滤后为空，保留 Top-1
    if not vector_docs and raw_results:
        vector_docs.append(raw_results[0][0])

    t1 = time.time()

    print(f"   ⏱️ 耗时: {(t1-t0)*1000:.1f}ms | 返回: {len(vector_docs)} 条")
    print(f"   📊 原始 distance / 转换后 similarity:")
    for i, (doc, distance) in enumerate(raw_results[:5], 1):
        similarity = 1.0 - distance / 2.0
        src = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"      #{i} distance={distance:.4f} sim={similarity:.4f} [{src}] {doc.page_content[:50]}...")
    if score_threshold > 0:
        filtered_count = sum(1 for _, d in raw_results if (1.0 - d / 2.0) >= score_threshold)
        print(f"   🔒 阈值过滤后 (sim>={score_threshold}): {filtered_count} 条")

    # ── Stage 2: BM25 检索 ──
    print(f"\n📍 Stage 2: BM25 关键词检索 (k={k})")
    data = store.get()
    all_documents = []
    if data and data.get("documents"):
        docs_content = data["documents"]
        metas = data.get("metadatas") or [{}] * len(docs_content)
        from langchain_core.documents import Document
        all_documents = [
            Document(page_content=c, metadata=m)
            for c, m in zip(docs_content, metas)
        ]

    bm25_docs = []
    if all_documents:
        try:
            bm25 = BM25Retriever.from_documents(all_documents, k=k)
            t0 = time.time()
            bm25_docs = bm25.invoke(query)
            t1 = time.time()
            print(f"   ⏱️ 耗时: {(t1-t0)*1000:.1f}ms | 返回: {len(bm25_docs)} 条")
            for i, doc in enumerate(bm25_docs[:5], 1):
                src = os.path.basename(doc.metadata.get("source", "unknown"))
                print(f"      #{i} [{src}] {doc.page_content[:50]}...")
            # BM25 本身不输出分数，但可以通过 rank 反映
        except Exception as e:
            print(f"   ❌ BM25 初始化失败: {e}")
    else:
        print("   ⚠️ 无文档，跳过 BM25")

    # ── Stage 3: 混合检索 (RRF) ──
    print(f"\n📍 Stage 3: 混合检索 RRF 融合 (vector_weight={chroma_config.get('vector_weight', 0.7)}, bm25_weight={chroma_config.get('bm25_weight', 0.3)})")
    if all_documents:
        try:
            vector_retriever_raw = store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            )
            bm25_for_hybrid = BM25Retriever.from_documents(all_documents, k=k)
            hybrid = EnsembleRetriever(
                retrievers=[vector_retriever_raw, bm25_for_hybrid],
                weights=[
                    chroma_config.get("vector_weight", 0.7),
                    chroma_config.get("bm25_weight", 0.3),
                ],
            )
            t0 = time.time()
            hybrid_docs = hybrid.invoke(query)
            t1 = time.time()
            print(f"   ⏱️ 耗时: {(t1-t0)*1000:.1f}ms | 返回: {len(hybrid_docs)} 条")
            for i, doc in enumerate(hybrid_docs[:5], 1):
                src = os.path.basename(doc.metadata.get("source", "unknown"))
                print(f"      #{i} [{src}] {doc.page_content[:50]}...")
        except Exception as e:
            print(f"   ❌ 混合检索失败: {e}")
            hybrid_docs = []
    else:
        print("   ⚠️ 无文档，跳过混合检索")
        hybrid_docs = []

    # ── Stage 4: CrossEncoder 重排序 ──
    reranker_model = chroma_config.get("reranker_model", "")
    print(f"\n📍 Stage 4: CrossEncoder 重排序 (model={os.path.basename(reranker_model) or '未配置'})")
    if not reranker_model or not os.path.exists(reranker_model):
        print("   ⚠️ 重排序模型路径不存在，跳过")
        return

    try:
        cross_encoder = HuggingFaceCrossEncoder(
            model_name=reranker_model,
            model_kwargs={"device": "cpu"},
        )
        top_n = chroma_config.get("top_n", 3)
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

        # 基于混合检索结果做重排序
        if hybrid_docs:
            # ContextualCompressionRetriever 需要 retriever，我们直接用 compressor
            t0 = time.time()
            reranked = compressor.compress_documents(hybrid_docs, query)
            t1 = time.time()

            # 获取分数（新 API 使用 score() 而非 predict()）
            pairs = [(query, d.page_content) for d in reranked]
            scores = cross_encoder.score(pairs) if pairs else []

            print(f"   ⏱️ 耗时: {(t1-t0)*1000:.1f}ms | 返回 Top-{top_n}")
            print(f"   📊 CrossEncoder 分数:")
            for i, (doc, score) in enumerate(zip(reranked, scores), 1):
                src = os.path.basename(doc.metadata.get("source", "unknown"))
                print(f"      #{i} score={score:.4f} [{src}] {doc.page_content[:50]}...")

            # 分数分布分析
            if len(scores) > 1:
                import statistics
                print(f"\n   📈 分数统计: min={min(scores):.4f}, max={max(scores):.4f}, "
                      f"mean={statistics.mean(scores):.4f}, std={statistics.stdev(scores):.4f}")
                if max(scores) - min(scores) < 0.1:
                    print("   ⚠️ 分数区分度低！建议检查模型是否匹配领域，或增加召回多样性")
        else:
            print("   ⚠️ 上游无文档，跳过重排序")
    except Exception as e:
        print(f"   ❌ 重排序失败: {e}")


def main():
    print("\n" + "🚀 RAG 系统诊断工具")
    store = diagnose_chroma_store()

    # 内置测试查询
    test_queries = [
        "锂离子电池正极材料有哪些",
        "固态电解质的优缺点",
        "宽禁带半导体应用",
        "硅负极容量衰减原因",
        "高温合金在航空发动机中的应用",
    ]

    print("\n" + "-" * 70)
    print("开始执行测试查询...")
    for q in test_queries:
        diagnose_retrieval_pipeline(store, q)

    print("\n" + "=" * 70)
    print("✅ 诊断完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
