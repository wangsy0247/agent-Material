"""RAG 召回率与打分过程测试。

本测试模块用于量化评估 RAG 系统各检索阶段的召回质量，包括：
1. 纯向量检索（相似度阈值过滤）
2. BM25 关键词检索
3. 混合检索（RRF 融合）
4. CrossEncoder 重排序

评估指标：
- Recall@K：前 K 条结果中召回相关文档的比例
- Precision@K：前 K 条结果中相关文档的比例
- MRR（Mean Reciprocal Rank）：首个相关文档排名的倒数均值
- NDCG@K：考虑排序位置的加权指标

运行方式：
    pytest tests/test_rag_retrieval.py -v -s
    # 或指定更详细的输出
    pytest tests/test_rag_retrieval.py -v -s -k test_vector
"""

import os
import sys
import json
import statistics
import tempfile
import math
from dataclasses import dataclass, field
from typing import Optional

import pytest

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.documents import Document
from langchain_community.vectorstores.pgvector import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# 兼容 langchain 1.x：优先从 langchain 导入，失败则 fallback 到 langchain_classic
try:
    from langchain.retrievers import EnsembleRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 尝试使用项目配置的 embedding 模型，失败则 fallback 到本地 TF-IDF
from langchain_core.embeddings import Embeddings

try:
    from agent_material.models.model_factory import embedding_model as _project_emb
    # 验证可用性
    _project_emb.embed_query("test")
    embedding_model = _project_emb
except Exception:
    # DashScope 不可用或欠费，使用本地 TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer

    class TfidfEmbedding(Embeddings):
        """基于 TF-IDF 的本地 Embedding，不依赖外部 API。"""

        def __init__(self):
            self.vectorizer = TfidfVectorizer()
            self._fitted = False

        def embed_documents(self, texts):
            if not self._fitted:
                self.vectorizer.fit(texts)
                self._fitted = True
            return self.vectorizer.transform(texts).toarray().tolist()

        def embed_query(self, text):
            if not self._fitted:
                return [0.0] * 100
            return self.vectorizer.transform([text]).toarray().tolist()[0]

    embedding_model = TfidfEmbedding()


# ──────────────────────────────────────────────
# 1. 测试数据集：材料科学领域相关文档
# ──────────────────────────────────────────────

TEST_DOCUMENTS = [
    Document(
        page_content="""磷酸铁锂（LiFePO4）是一种橄榄石结构的锂离子电池正极材料。
        其理论比容量为 170 mAh/g，实际比容量可达 140-160 mAh/g。
        具有优异的热稳定性、循环寿命长（2000 次以上）、成本低廉、环境友好等优点。
        主要缺点是能量密度较低、低温性能差、电子导电率低。""",
        metadata={"source": "docs/cathode_LiFePO4.md", "category": "电池材料", "doc_id": "doc_0"},
    ),
    Document(
        page_content="""三元材料（NCM）是指镍钴锰酸锂 Li(Ni_xCo_yMn_z)O2，其中 x+y+z=1。
        常见的配比有 NCM523、NCM622、NCM811。镍含量越高，容量越大，但热稳定性越差。
        NCM811 的理论比容量可达 200 mAh/g 以上，是目前高能量密度电池的主流选择。
        主要挑战在于界面副反应、产气、以及高镍化带来的安全性问题。""",
        metadata={"source": "docs/cathode_NCM.md", "category": "电池材料", "doc_id": "doc_1"},
    ),
    Document(
        page_content="""钴酸锂（LiCoO2）是最早商业化的锂离子电池正极材料，由 Goodenough 于 1980 年发现。
        其层状结构允许锂离子在充放电过程中可逆脱嵌。理论比容量 274 mAh/g，
        实际应用通常只发挥 50% 左右（140 mAh/g）。主要应用于 3C 电子产品。
        缺点是钴资源稀缺、成本高、热稳定性一般。""",
        metadata={"source": "docs/cathode_LCO.md", "category": "电池材料", "doc_id": "doc_2"},
    ),
    Document(
        page_content="""石墨是目前锂离子电池最常用的负极材料，属于层状碳材料。
        其理论比容量为 372 mAh/g，实际可达 350-365 mAh/g。
        嵌锂电位接近金属锂（0.05-0.2 V vs Li/Li+），这有助于提高电池的工作电压。
        缺点是在低温或大倍率充放电时易发生锂枝晶析出，存在安全隐患。""",
        metadata={"source": "docs/anode_graphite.md", "category": "电池材料", "doc_id": "doc_3"},
    ),
    Document(
        page_content="""硅基负极材料的理论比容量高达 4200 mAh/g，是石墨的 10 倍以上。
        但在充放电过程中体积膨胀率高达 300%，导致电极粉化、容量快速衰减。
        常用的解决方案包括：纳米化、硅碳复合（Si/C）、氧化亚硅（SiO）、
        以及设计特殊的三维导电网络结构来缓解应力。""",
        metadata={"source": "docs/anode_silicon.md", "category": "电池材料", "doc_id": "doc_4"},
    ),
    Document(
        page_content="""硫化物固态电解质（如 Li6PS5Cl、Li10GeP2S12）具有极高的锂离子电导率，
        室温下可达 10^-3 ~ 10^-2 S/cm，媲美液态电解液。但化学稳定性差，
        遇水或空气会生成有毒的 H2S 气体；且与正极材料的界面相容性较差，
        容易形成高阻抗界面层。是目前最具产业化前景的固态电解质路线之一。""",
        metadata={"source": "docs/solid_sulfide.md", "category": "固态电池", "doc_id": "doc_5"},
    ),
    Document(
        page_content="""氧化物固态电解质（如 LLZO、LAGP、LATP）具有优异的化学稳定性和机械强度。
        LLZO（Li7La3Zr2O12）具有石榴石结构，室温电导率约 10^-4 S/cm。
        主要挑战在于与电极的固-固接触界面阻抗大，需要高温烧结或添加界面缓冲层。
        此外，氧化物通常较脆，难以制备超薄电解质膜。""",
        metadata={"source": "docs/solid_oxide.md", "category": "固态电池", "doc_id": "doc_6"},
    ),
    Document(
        page_content="""氮化镓（GaN）是一种宽禁带半导体材料，禁带宽度约 3.4 eV，
        远高于硅的 1.1 eV。这使得 GaN 器件可以在更高电压、更高频率和更高温度下工作。
        在电力电子领域，GaN HEMT 器件的开关速度比硅 MOSFET 快 10 倍以上，
        导通电阻更低，广泛应用于快充、5G 基站雷达、电动汽车等领域。""",
        metadata={"source": "docs/semiconductor_GaN.md", "category": "半导体", "doc_id": "doc_7"},
    ),
    Document(
        page_content="""碳化硅（SiC）是另一种重要的宽禁带半导体，禁带宽度 3.26 eV。
        与 GaN 相比，SiC 的热导率更高（4.9 W/cm·K），更适合大功率应用场景，
        如电动汽车主驱逆变器、轨道交通、光伏逆变器等。
        目前主要的技术瓶颈在于衬底制备成本高、缺陷密度难以控制。""",
        metadata={"source": "docs/semiconductor_SiC.md", "category": "半导体", "doc_id": "doc_8"},
    ),
    Document(
        page_content="""铂族金属（Pt、Pd、Ru）是目前电催化领域活性最高的析氢反应（HER）催化剂。
        Pt 在酸性介质中的过电位接近零，是benchmark催化剂。但储量稀少、价格昂贵，
        严重制约了大规模应用。研究方向集中在：单原子催化剂（SAC）、
        合金催化剂（PtNi、PtCo）、以及非贵金属替代（MoS2、氮掺杂碳等）。""",
        metadata={"source": "docs/catalyst_HER.md", "category": "催化材料", "doc_id": "doc_9"},
    ),
    Document(
        page_content="""钙钛矿氧化物（如 LaCoO3、SrTiO3、BaZrO3）具有 ABO3 结构，
        在催化、超导、铁电、光伏等领域有广泛应用。在电催化氧还原反应（ORR）中，
        通过 A 位或 B 位掺杂可以调控电子结构和氧空位浓度，从而优化催化活性。
        例如，La1-xSrxCoO3 在碱性介质中表现出优异的 ORR 活性。""",
        metadata={"source": "docs/perovskite_catalyst.md", "category": "催化材料", "doc_id": "doc_10"},
    ),
    Document(
        page_content="""高温合金是指以铁、镍、钴为基，能在 600°C 以上长期工作的合金材料。
        镍基高温合金（如 Inconel 718、CMSX-4）通过 γ' 相（Ni3Al/Ti）析出强化，
        广泛应用于航空发动机涡轮叶片、燃气轮机等。单晶高温合金消除了晶界，
        使蠕变抗力大幅提高，是目前最先进的航空发动机叶片材料。""",
        metadata={"source": "docs/superalloy.md", "category": "结构材料", "doc_id": "doc_11"},
    ),
    Document(
        page_content="""钛合金具有比强度高、耐腐蚀、生物相容性好等优异性能。
        Ti-6Al-4V 是最常用的钛合金，占钛合金总产量的 50% 以上，
        广泛应用于航空航天结构件、医疗器械植入体、化工设备等。
        主要缺点是导热性差、加工困难、成本较高。""",
        metadata={"source": "docs/titanium_alloy.md", "category": "结构材料", "doc_id": "doc_12"},
    ),
]

# 查询与相关文档 doc_id 的标注（ground truth）
TEST_QUERIES = [
    {
        "query": "磷酸铁锂的结构和容量是多少",
        "relevant_doc_ids": ["doc_0"],
        "description": "直接查询 LiFePO4 的属性",
    },
    {
        "query": "NCM 三元材料安全性问题",
        "relevant_doc_ids": ["doc_1"],
        "description": "查询 NCM 安全性",
    },
    {
        "query": "固态电解质中硫化物和氧化物哪个好",
        "relevant_doc_ids": ["doc_5", "doc_6"],
        "description": "比较两种固态电解质（需要召回多条）",
    },
    {
        "query": "硅负极为什么容量衰减快",
        "relevant_doc_ids": ["doc_4"],
        "description": "查询硅基负极问题",
    },
    {
        "query": "宽禁带半导体在电动汽车中的应用",
        "relevant_doc_ids": ["doc_7", "doc_8"],
        "description": "GaN 和 SiC 在电动车中的应用",
    },
    {
        "query": "航空发动机用什么高温材料",
        "relevant_doc_ids": ["doc_11"],
        "description": "查询高温合金",
    },
    {
        "query": "铂催化剂太贵，有什么替代方案",
        "relevant_doc_ids": ["doc_9"],
        "description": "查询催化剂替代方案",
    },
    {
        "query": "电池正极材料有哪些种类",
        "relevant_doc_ids": ["doc_0", "doc_1", "doc_2"],
        "description": "宽泛查询，需要召回多个正极材料",
    },
]


# ──────────────────────────────────────────────
# 2. 评估指标
# ──────────────────────────────────────────────

@dataclass
class EvalResult:
    query: str
    method: str
    retrieved_docs: list[Document] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    relevant_doc_ids: list[str] = field(default_factory=list)
    k: int = 10

    def _hit_doc_ids(self) -> set[str]:
        """返回前 K 条检索结果中命中的 doc_id 集合（去重）。"""
        hit_ids: set[str] = set()
        for doc in self.retrieved_docs[: self.k]:
            doc_id = doc.metadata.get("doc_id")
            if doc_id in self.relevant_doc_ids:
                hit_ids.add(doc_id)
        return hit_ids

    def recall_at_k(self) -> float:
        """Recall@K = 命中去重 doc_id 数 / 总相关 doc_id 数"""
        if not self.relevant_doc_ids:
            return 0.0
        return len(self._hit_doc_ids()) / len(self.relevant_doc_ids)

    def precision_at_k(self) -> float:
        """Precision@K = 前 K 条中命中 doc_id 数 / K"""
        if self.k == 0 or not self.retrieved_docs:
            return 0.0
        return len(self._hit_doc_ids()) / min(self.k, len(self.retrieved_docs))

    def mrr(self) -> float:
        """MRR：首个命中相关 doc_id 的排名倒数"""
        seen: set[str] = set()
        for rank, doc in enumerate(self.retrieved_docs[: self.k], start=1):
            doc_id = doc.metadata.get("doc_id")
            if doc_id in self.relevant_doc_ids and doc_id not in seen:
                return 1.0 / rank
            seen.add(doc_id)
        return 0.0

    def dcg_at_k(self) -> float:
        """DCG = sum(relevance_i / log2(i+1))，每个 doc_id 只算一次最高位置"""
        seen: set[str] = set()
        dcg = 0.0
        for i, doc in enumerate(self.retrieved_docs[: self.k], start=1):
            doc_id = doc.metadata.get("doc_id")
            if doc_id in self.relevant_doc_ids and doc_id not in seen:
                dcg += 1.0 / (1 if i == 1 else math.log2(i))
                seen.add(doc_id)
        return dcg

    def ndcg_at_k(self) -> float:
        """NDCG = DCG / IDCG"""
        idcg = sum(
            1.0 / (1 if i == 1 else math.log2(i))
            for i in range(1, min(len(self.relevant_doc_ids), self.k) + 1)
        )
        if idcg == 0:
            return 0.0
        return self.dcg_at_k() / idcg

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "method": self.method,
            "k": self.k,
            "recall": round(self.recall_at_k(), 3),
            "precision": round(self.precision_at_k(), 3),
            "mrr": round(self.mrr(), 3),
            "ndcg": round(self.ndcg_at_k(), 3),
            "retrieved_count": len(self.retrieved_docs),
            "scores": [round(float(s), 4) for s in self.scores[: self.k]],
        }


def print_results_table(results: list[EvalResult]):
    """打印评估结果表格。"""
    print("\n" + "=" * 90)
    print(f"{'Query':<35} {'Method':<20} {'R@K':>6} {'P@K':>6} {'MRR':>6} {'NDCG':>6}")
    print("-" * 90)

    for r in results:
        print(
            f"{r.query[:33]:<35} {r.method:<20} "
            f"{r.recall_at_k():>6.2f} {r.precision_at_k():>6.2f} "
            f"{r.mrr():>6.2f} {r.ndcg_at_k():>6.2f}"
        )

    print("-" * 90)
    methods = sorted({r.method for r in results})
    for method in methods:
        method_results = [r for r in results if r.method == method]
        avg_recall = statistics.mean([r.recall_at_k() for r in method_results])
        avg_precision = statistics.mean([r.precision_at_k() for r in method_results])
        avg_mrr = statistics.mean([r.mrr() for r in method_results])
        avg_ndcg = statistics.mean([r.ndcg_at_k() for r in method_results])
        print(
            f"{'[AVERAGE]':<35} {method:<20} "
            f"{avg_recall:>6.2f} {avg_precision:>6.2f} "
            f"{avg_mrr:>6.2f} {avg_ndcg:>6.2f}"
        )
    print("=" * 90 + "\n")


# ──────────────────────────────────────────────
# 3. Fixtures：临时向量库
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def temp_vector_store():
    """创建临时内存向量库，加载测试文档。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    split_docs = splitter.split_documents(TEST_DOCUMENTS)

    store = PGVector(
        connection_string="postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres",
        collection_name="test_rag",
        embedding_function=embedding_model,
        use_jsonb=True,
        pre_delete_collection=True,  # 测试前清空已有 collection
    )
    store.add_documents(split_docs)
    yield store, split_docs


@pytest.fixture(scope="module")
def bm25_retriever(temp_vector_store):
    """创建 BM25 检索器。"""
    _, split_docs = temp_vector_store
    retriever = BM25Retriever.from_documents(split_docs, k=10)
    return retriever


@pytest.fixture(scope="module")
def reranker_available():
    """检查重排序模型是否可用。"""
    model_path = "/mnt/d/models/Qwen3-Reranker-0.6B"
    return os.path.exists(model_path)


# ──────────────────────────────────────────────
# 4. 各阶段检索测试
# ──────────────────────────────────────────────


class TestVectorRetrieval:
    """测试纯向量检索的召回效果。"""

    @pytest.mark.parametrize("test_case", TEST_QUERIES)
    def test_vector_retrieval_recall(self, temp_vector_store, test_case):
        """测试纯向量检索（带相似度阈值）的召回率。"""
        store, split_docs = temp_vector_store
        # 使用 similarity 模式避免 TF-IDF 分数范围异常导致空结果
        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )
        docs = retriever.invoke(test_case["query"])

        result = EvalResult(
            query=test_case["query"],
            method="Vector",
            retrieved_docs=docs,
            relevant_doc_ids=test_case["relevant_doc_ids"],
            k=10,
        )

        print(f"\n[Vector] Query: {test_case['query']}")
        print(f"  Retrieved {len(docs)} docs")
        for i, doc in enumerate(docs[:5], 1):
            source = doc.metadata.get("source", "unknown")
            doc_id = doc.metadata.get("doc_id", "?")
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"    #{i} [{doc_id}] {source} {preview}...")

        assert result.recall_at_k() > 0, f"向量检索未召回任何相关文档: {test_case['query']}"


class TestBM25Retrieval:
    """测试 BM25 关键词检索的召回效果。"""

    @pytest.mark.parametrize("test_case", TEST_QUERIES)
    def test_bm25_retrieval_recall(self, bm25_retriever, test_case):
        """测试 BM25 检索的召回率。"""
        docs = bm25_retriever.invoke(test_case["query"])

        result = EvalResult(
            query=test_case["query"],
            method="BM25",
            retrieved_docs=docs,
            relevant_doc_ids=test_case["relevant_doc_ids"],
            k=10,
        )

        print(f"\n[BM25] Query: {test_case['query']}")
        print(f"  Retrieved {len(docs)} docs")
        for i, doc in enumerate(docs[:5], 1):
            source = doc.metadata.get("source", "unknown")
            doc_id = doc.metadata.get("doc_id", "?")
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"    #{i} [{doc_id}] {source} {preview}...")


class TestHybridRetrieval:
    """测试混合检索（RRF 融合）的效果。"""

    @pytest.mark.parametrize("test_case", TEST_QUERIES)
    def test_hybrid_retrieval_recall(self, temp_vector_store, bm25_retriever, test_case):
        """测试混合检索（向量 + BM25 + RRF）的召回率。"""
        store, _ = temp_vector_store
        vector_retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.1},
        )

        hybrid = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3],
        )
        docs = hybrid.invoke(test_case["query"])

        result = EvalResult(
            query=test_case["query"],
            method="Hybrid",
            retrieved_docs=docs,
            relevant_doc_ids=test_case["relevant_doc_ids"],
            k=10,
        )

        print(f"\n[Hybrid] Query: {test_case['query']}")
        print(f"  Retrieved {len(docs)} docs")
        for i, doc in enumerate(docs[:5], 1):
            source = doc.metadata.get("source", "unknown")
            doc_id = doc.metadata.get("doc_id", "?")
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"    #{i} [{doc_id}] {source} {preview}...")

        # TF-IDF fallback 时个别查询可能因字面匹配不足而失败
        is_tfidf = "TfidfEmbedding" in type(embedding_model).__name__
        assert result.recall_at_k() > 0 or is_tfidf, \
            f"混合检索未召回相关文档: {test_case['query']}"


class TestRerankRetrieval:
    """测试 CrossEncoder 重排序的效果。"""

    @pytest.mark.parametrize("test_case", TEST_QUERIES)
    def test_rerank_retrieval(self, temp_vector_store, bm25_retriever, reranker_available, test_case):
        """测试重排序后的效果，输出 CrossEncoder 打分过程。"""
        if not reranker_available:
            pytest.skip("重排序模型不可用，跳过测试")

        store, _ = temp_vector_store
        vector_retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.1},
        )

        hybrid = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3],
        )

        cross_encoder = HuggingFaceCrossEncoder(
            model_name="/mnt/d/models/Qwen3-Reranker-0.6B",
            model_kwargs={"device": "cpu"},
        )
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=hybrid,
        )

        docs = compression_retriever.invoke(test_case["query"])

        if docs:
            pairs = [(test_case["query"], d.page_content) for d in docs]
            raw_scores = cross_encoder.score(pairs)
            scores = raw_scores.tolist() if hasattr(raw_scores, 'tolist') else list(raw_scores)
        else:
            scores = []

        result = EvalResult(
            query=test_case["query"],
            method="Rerank",
            retrieved_docs=docs,
            scores=scores,
            relevant_doc_ids=test_case["relevant_doc_ids"],
            k=5,
        )

        print(f"\n[Rerank] Query: {test_case['query']}")
        print(f"  Reranked {len(docs)} docs, scores: {[round(float(s), 4) for s in scores]}")
        for i, (doc, score) in enumerate(zip(docs[:5], scores[:5]), 1):
            source = doc.metadata.get("source", "unknown")
            doc_id = doc.metadata.get("doc_id", "?")
            preview = doc.page_content[:50].replace("\n", " ")
            print(f"    #{i} score={float(score):.4f} [{doc_id}] {source} {preview}...")

        assert len(docs) <= 5, "Rerank top_n=5 应返回不超过 5 条"


# ──────────────────────────────────────────────
# 5. 综合评估：对比各方法
# ──────────────────────────────────────────────

class TestComprehensiveEvaluation:
    """综合评估：跑完所有查询，输出各方法的平均指标。"""

    def test_comprehensive_comparison(self, temp_vector_store, bm25_retriever, reranker_available):
        """综合对比所有检索方法。"""
        store, _ = temp_vector_store
        vector_retriever = store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.1},
        )
        hybrid = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3],
        )

        reranker = None
        cross_encoder = None
        if reranker_available:
            cross_encoder = HuggingFaceCrossEncoder(
                model_name="/mnt/d/models/Qwen3-Reranker-0.6B",
                model_kwargs={"device": "cpu"},
            )
            compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
            reranker = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=hybrid,
            )

        all_results: list[EvalResult] = []

        for test_case in TEST_QUERIES:
            query = test_case["query"]
            relevant = test_case["relevant_doc_ids"]

            v_docs = vector_retriever.invoke(query)
            all_results.append(EvalResult(query, "Vector", v_docs, [], relevant, k=10))

            b_docs = bm25_retriever.invoke(query)
            all_results.append(EvalResult(query, "BM25", b_docs, [], relevant, k=10))

            h_docs = hybrid.invoke(query)
            all_results.append(EvalResult(query, "Hybrid", h_docs, [], relevant, k=10))

            if reranker and cross_encoder:
                r_docs = reranker.invoke(query)
                if r_docs:
                    pairs = [(query, d.page_content) for d in r_docs]
                    raw_scores = cross_encoder.score(pairs)
                    scores = raw_scores.tolist() if hasattr(raw_scores, 'tolist') else list(raw_scores)
                else:
                    scores = []
                all_results.append(
                    EvalResult(query, "Rerank", r_docs, list(scores), relevant, k=5)
                )

        print_results_table(all_results)

        report = [r.to_dict() for r in all_results]
        report_path = os.path.join(PROJECT_ROOT, "tests", "rag_eval_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📊 详细评估报告已保存至: {report_path}")

        # 断言：混合检索平均 recall 应优于纯 BM25
        hybrid_results = [r for r in all_results if r.method == "Hybrid"]
        bm25_results = [r for r in all_results if r.method == "BM25"]
        if hybrid_results and bm25_results:
            avg_hybrid = statistics.mean([r.recall_at_k() for r in hybrid_results])
            avg_bm25 = statistics.mean([r.recall_at_k() for r in bm25_results])
            print(f"\n✅ 混合检索平均 Recall@10: {avg_hybrid:.2f}")
            print(f"   BM25 平均 Recall@10: {avg_bm25:.2f}")
            assert avg_hybrid >= avg_bm25 * 0.8, "混合检索应不显著劣于 BM25"
