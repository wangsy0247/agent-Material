"""Materials-aware BM25 retriever with explicit no-hit semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from agent_material.rag_engine.bm25 import material_tokenize


class MaterialBM25Retriever(LangChainBM25Retriever):
    """BM25 retriever that does not fabricate results for lexical misses.

    ``rank_bm25.get_top_n`` always returns documents, even if every query term
    is absent.  This implementation retains BM25 ranking but first requires a
    configurable number of query terms to occur in a document.  An optional
    score floor is available, but disabled by default because valid matches in
    very small corpora can have zero or negative BM25 scores.
    """

    min_term_overlap: int = 1
    min_score: float | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        query_tokens = self.preprocess_func(query)
        if not query_tokens or not self.docs:
            return []

        scores = self.vectorizer.get_scores(query_tokens)
        unique_query_tokens = set(query_tokens)
        document_frequencies = getattr(self.vectorizer, "doc_freqs", None)
        ranked: list[tuple[float, int, tuple[str, ...], int, Document]] = []

        for index, (doc, raw_score) in enumerate(zip(self.docs, scores)):
            score = float(raw_score)
            if not math.isfinite(score):
                continue

            if document_frequencies is not None:
                document_tokens = document_frequencies[index]
                overlap = sum(token in document_tokens for token in unique_query_tokens)
            else:
                document_tokens = set(self.preprocess_func(doc.page_content))
                overlap = len(unique_query_tokens.intersection(document_tokens))

            if overlap < self.min_term_overlap:
                continue
            if self.min_score is not None and score < self.min_score:
                continue

            ranked.append((score, overlap, _stable_document_key(doc), index, doc))

        ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        results: list[Document] = []
        for score, overlap, _, _, doc in ranked[: self.k]:
            metadata = dict(doc.metadata)
            metadata["bm25_score"] = score
            metadata["bm25_term_overlap"] = overlap
            results.append(Document(page_content=doc.page_content, metadata=metadata, id=doc.id))
        return results


def build_material_bm25_retriever(
    documents: Sequence[Document],
    *,
    k: int,
    config: Mapping[str, Any] | None = None,
    retriever_cls: type[LangChainBM25Retriever] = MaterialBM25Retriever,
) -> LangChainBM25Retriever:
    """Build every BM25 index with the same analyzer and no-hit policy."""

    bm25_config = (config or {}).get("bm25", {})
    min_score = bm25_config.get("min_score")
    return retriever_cls.from_documents(
        documents,
        k=k,
        preprocess_func=material_tokenize,
        min_term_overlap=int(bm25_config.get("min_term_overlap", 1)),
        min_score=float(min_score) if min_score is not None else None,
    )


def _stable_document_key(doc: Document) -> tuple[str, ...]:
    metadata = doc.metadata
    return (
        str(metadata.get("chunk_id", "")),
        str(metadata.get("document_id", "")),
        str(metadata.get("version_id", "")),
        str(metadata.get("source", "")),
        str(metadata.get("page", "")),
        doc.page_content,
    )


__all__ = [
    "MaterialBM25Retriever",
    "build_material_bm25_retriever",
    "material_tokenize",
]
