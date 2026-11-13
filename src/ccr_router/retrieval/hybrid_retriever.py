"""Hybrid BM25 + Dense Reciprocal Rank Fusion (RRF) retriever."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievedPassage:
    doc_id: str
    text: str
    rrf_score: float
    bm25_score: float
    dense_score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]{2,}\b", text.lower())


class HybridRRFRetriever:
    """Combines sparse BM25 lexical matching with dense embeddings via RRF."""

    def __init__(
        self,
        dense_model_id: str = "BAAI/bge-base-en-v1.5",
        rrf_k: int = 60,
        k1: float = 1.5,
        b: float = 0.75,
        use_dense: bool = True,
    ) -> None:
        self.dense_model_id = dense_model_id
        self.rrf_k = rrf_k
        self.k1 = k1
        self.b = b
        self.use_dense = use_dense

        self.doc_ids: list[str] = []
        self.documents: list[str] = []
        self._doc_tokens: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._avg_dl: float = 1.0
        self._idf: dict[str, float] = {}
        self._dense_encoder = None
        self._doc_embeddings: np.ndarray | None = None

    def _get_encoder(self):
        if not self.use_dense:
            return None
        if self._dense_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._dense_encoder = SentenceTransformer(self.dense_model_id)
            except Exception:
                self.use_dense = False
                self._dense_encoder = None
        return self._dense_encoder

    def index(self, doc_ids: list[str], documents: list[str]) -> None:
        if len(doc_ids) != len(documents):
            raise ValueError("doc_ids and documents must have equal length.")
        self.doc_ids = list(doc_ids)
        self.documents = list(documents)

        self._doc_tokens = [Counter(_tokenize(doc)) for doc in documents]
        self._doc_lens = [sum(c.values()) for c in self._doc_tokens]
        self._avg_dl = float(np.mean(self._doc_lens)) if self._doc_lens else 1.0

        n_docs = len(documents)
        df: Counter[str] = Counter()
        for counts in self._doc_tokens:
            for term in counts:
                df[term] += 1

        self._idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

        encoder = self._get_encoder()
        if encoder is not None and documents:
            embs = encoder.encode(documents, normalize_embeddings=True, show_progress_bar=False)
            self._doc_embeddings = np.asarray(embs, dtype=np.float32)
        else:
            self._doc_embeddings = None

    def _bm25_scores(self, query: str) -> np.ndarray:
        q_tokens = _tokenize(query)
        scores = np.zeros(len(self.documents), dtype=np.float64)
        if not q_tokens or not self.documents:
            return scores

        for idx, counts in enumerate(self._doc_tokens):
            dl = self._doc_lens[idx]
            norm_factor = self.k1 * (1.0 - self.b + self.b * (dl / max(self._avg_dl, 1e-6)))
            s = 0.0
            for term in q_tokens:
                tf = counts.get(term, 0)
                if tf > 0 and term in self._idf:
                    s += self._idf[term] * ((tf * (self.k1 + 1.0)) / (tf + norm_factor))
            scores[idx] = s
        return scores

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPassage]:
        if not self.documents:
            return []

        bm25 = self._bm25_scores(query)
        bm25_order = np.argsort(-bm25)
        bm25_ranks = np.empty_like(bm25_order)
        bm25_ranks[bm25_order] = np.arange(1, len(self.documents) + 1)

        if self._doc_embeddings is not None and self._dense_encoder is not None:
            q_emb = self._dense_encoder.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0]
            dense = self._doc_embeddings @ np.asarray(q_emb, dtype=np.float32)
            dense_order = np.argsort(-dense)
            dense_ranks = np.empty_like(dense_order)
            dense_ranks[dense_order] = np.arange(1, len(self.documents) + 1)
            rrf = (1.0 / (self.rrf_k + bm25_ranks)) + (1.0 / (self.rrf_k + dense_ranks))
        else:
            dense = np.zeros_like(bm25)
            rrf = 1.0 / (self.rrf_k + bm25_ranks)

        top_indices = np.argsort(-rrf)[:top_k]
        return [
            RetrievedPassage(
                doc_id=self.doc_ids[i],
                text=self.documents[i],
                rrf_score=float(rrf[i]),
                bm25_score=float(bm25[i]),
                dense_score=float(dense[i]),
            )
            for i in top_indices
        ]