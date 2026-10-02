"""
Skill Discovery, per spec.md sections 5 and 17.

The point of this layer is architectural, not just functional: the Router must NOT
receive a prompt that enumerates every registered skill, because that prompt would
grow without bound as skills are added (4 today, 30 tomorrow). Discovery's job is to
embed the registry's skill *descriptions* once, embed the incoming *prompt*, and
return only the top-K most relevant skill names - so the Router's prompt stays a
constant size regardless of registry size.

Embedding backend: BAAI/bge-m3 (`BgeEmbeddingIndex`), a multilingual embedding model
trained on cross-lingual parallel data across 100+ languages (including Persian),
specifically so that a Persian query and its semantically-matching English skill
description land close together in vector space - which a lexical method like
TF-IDF cannot do at all when the query and corpus are in different languages. This
was a deliberate choice given this project's Persian/English requirement (see
README.md "Assumptions"). The similarity backend still sits behind the small
`EmbeddingIndex` interface below, so it remains a one-class swap if needed.

Lazy fitting: `SkillDiscovery` only calls `index.fit(...)` the first time
`candidates()` actually needs to rank anything - i.e. once the registry has grown
past `top_k`. With today's 4 registered skills and `top_k=4`, the pass-through branch
below returns immediately and the embedding model is never loaded at all. This keeps
the common case (today) free of BGE-M3's cost (a ~2.2GB model, a one-time download
from Hugging Face Hub, torch as a dependency) entirely, while the exact same code
"just works" once a 5th+ skill makes real narrowing necessary.

`general_chat` is always force-included as a candidate (if registered) regardless of
its similarity rank, since it's this system's fallback target and must always be
available to the Router even if its description scores low on a given prompt.
"""
from __future__ import annotations

from typing import List, Protocol

from src.registry.skill_registry import SkillRegistry

FALLBACK_SKILL_NAME = "general_chat"


class EmbeddingIndex(Protocol):
    """Minimal interface so the similarity backend can be swapped without touching
    SkillDiscovery's logic."""

    def fit(self, names: List[str], corpus: List[str]) -> None: ...
    def rank(self, query: str) -> List[str]:
        """Return skill names ordered by descending relevance to `query`."""
        ...


class BgeEmbeddingIndex:
    """
    Multilingual dense-retrieval backend using BAAI/bge-m3 via the `FlagEmbedding`
    package. The (large) model is loaded once per process via a class-level
    singleton, since instantiating `SkillDiscovery` should not reload it.

    NOTE: needs `pip install FlagEmbedding` (pulls in torch/transformers) and network
    access to Hugging Face Hub the first time it runs, to download the model weights
    (cached locally afterwards).
    """

    _model = None  # class-level: shared across all instances/tests in one process

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = False):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._names: List[str] = []
        self._embeddings = None  # numpy array, shape (N, dim)

    def _get_model(self):
        if BgeEmbeddingIndex._model is None:
            from FlagEmbedding import BGEM3FlagModel  # imported lazily: heavy, optional

            BgeEmbeddingIndex._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
        return BgeEmbeddingIndex._model

    def fit(self, names: List[str], corpus: List[str]) -> None:
        model = self._get_model()
        self._names = list(names)
        out = model.encode(corpus, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        self._embeddings = out["dense_vecs"]

    def rank(self, query: str) -> List[str]:
        if self._embeddings is None or not self._names:
            return []
        import numpy as np

        model = self._get_model()
        out = model.encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False)
        query_vec = out["dense_vecs"][0]

        corpus_norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = np.clip(corpus_norms * query_norm, 1e-9, None)
        scores = (self._embeddings @ query_vec) / denom

        ranked = sorted(zip(self._names, scores), key=lambda pair: pair[1], reverse=True)
        return [name for name, _ in ranked]


class SkillDiscovery:
    def __init__(self, registry: SkillRegistry, top_k: int = 4, index: EmbeddingIndex | None = None):
        self.registry = registry
        self.top_k = top_k
        self._index = index or BgeEmbeddingIndex()
        self._fitted = False

    def _ensure_fitted(self) -> None:
        if self._fitted:
            return
        specs = self.registry.all()
        self._index.fit([s.name for s in specs], [s.description for s in specs])
        self._fitted = True

    def candidates(self, prompt: str) -> List[str]:
        all_names = self.registry.names()

        # Small-registry fast path: if we have top_k or fewer skills total, there is
        # nothing to filter - hand the Router everything, and never touch the
        # (potentially heavyweight) embedding backend at all. This is exactly what
        # keeps today's 4-skill prototype cheap, while the same code path scales
        # down automatically once the registry grows past top_k.
        if len(all_names) <= self.top_k:
            return all_names

        self._ensure_fitted()
        ranked = self._index.rank(prompt)
        top = ranked[: self.top_k]

        if FALLBACK_SKILL_NAME in all_names and FALLBACK_SKILL_NAME not in top:
            top = top[: self.top_k - 1] + [FALLBACK_SKILL_NAME]

        return top