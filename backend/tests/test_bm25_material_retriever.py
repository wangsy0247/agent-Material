from langchain_core.documents import Document

from agent_material.rag_engine.bm25 import material_tokenize
from agent_material.rag_engine.bm25_retriever import build_material_bm25_retriever


def test_material_tokenize_emits_overlapping_chinese_phrases():
    tokens = material_tokenize("氧化镓热导率")

    assert {"氧化", "化镓", "镓热", "热导", "导率"}.issubset(tokens)
    assert {"氧化镓", "化镓热", "镓热导", "热导率"}.issubset(tokens)


def test_material_tokenize_normalizes_formula_greek_case_and_subscripts():
    tokens = material_tokenize("β-Ga₂O₃ and DFT")

    assert {"beta", "ga2o3", "dft"}.issubset(tokens)
    assert material_tokenize("Ga₂O₃") == material_tokenize("ga2o3")


def test_material_tokenize_keeps_compound_and_components():
    tokens = material_tokenize("first-principles/DFT")

    assert {"first-principles/dft", "first", "principles", "dft"}.issubset(tokens)


def test_material_tokenize_is_deterministic_and_ignores_punctuation():
    text = "热导率，thermal-conductivity!!!"

    assert material_tokenize(text) == material_tokenize(text)
    assert all(token for token in material_tokenize(text))


def test_bm25_ranks_partial_chinese_phrase_match_first():
    docs = [
        Document(page_content="machine learning potential", metadata={"chunk_id": "ml"}),
        Document(page_content="氧化镓热导率随温度升高而降低", metadata={"chunk_id": "relevant"}),
        Document(page_content="其他材料的力学性质", metadata={"chunk_id": "other"}),
    ]
    retriever = build_material_bm25_retriever(docs, k=3)

    results = retriever.invoke("氧化镓热导率")

    assert [doc.metadata["chunk_id"] for doc in results] == ["relevant"]
    assert results[0].metadata["bm25_term_overlap"] > 0
    assert "bm25_score" in results[0].metadata


def test_bm25_returns_empty_for_query_without_lexical_overlap():
    docs = [
        Document(page_content="氧化镓热导率", metadata={"chunk_id": "a"}),
        Document(page_content="神经网络势函数", metadata={"chunk_id": "b"}),
    ]
    retriever = build_material_bm25_retriever(docs, k=2)

    assert retriever.invoke("天气预报") == []


def test_bm25_accepts_valid_zero_or_negative_score_in_small_corpus():
    retriever = build_material_bm25_retriever(
        [Document(page_content="zirconia-xyzzy", metadata={"chunk_id": "only"})],
        k=1,
    )

    results = retriever.invoke("zirconia-xyzzy")

    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "only"


def test_bm25_formula_normalization_matches_subscript_variant():
    retriever = build_material_bm25_retriever(
        [
            Document(page_content="beta-Ga2O3 thermal conductivity", metadata={"chunk_id": "hit"}),
            Document(page_content="silicon mechanical properties", metadata={"chunk_id": "miss"}),
        ],
        k=2,
    )

    results = retriever.invoke("β-Ga₂O₃")

    assert [doc.metadata["chunk_id"] for doc in results] == ["hit"]
