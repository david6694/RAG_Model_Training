"""BM25 search engine with code-aware tokenization and file path boosting."""

from collections import defaultdict
import os
import pickle
import re
from typing import Dict, List, Optional, Set
from rank_bm25 import BM25Plus

from src.chunker import Chunk
from src.models import MinimalSearchResults, MinimalSource

STOP_WORDS: Set[str] = {
    "what", "is", "the", "in", "to", "for", "of", "and", "a", "an", "does",
    "how", "do", "you", "which", "are", "on", "with", "as", "by", "at", "from",
    "it", "its", "that", "this", "be", "or", "can", "when", "where", "why",
    "using", "used", "use", "between", "under", "into", "through", "about",
}


def tokenize_text(text: str) -> List[str]:
    """Code-aware tokenizer handling camelCase, snake_case, and symbols."""
    if not text:
        return []

    # 1. Split camelCase identifiers
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1 \2", text)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s1)

    # 2. Extract alphanumeric tokens and compound identifiers
    raw_tokens = re.findall(r"[a-zA-Z0-9_\.\-]+", s2)

    tokens: List[str] = []
    for tok in raw_tokens:
        tok_lower = tok.lower()
        if len(tok_lower) > 1:
            tokens.append(tok_lower)
        # Split on dots, hyphens, underscores
        subparts = [p for p in re.split(r"[_\.\-/]", tok_lower) if len(p) > 1]
        if len(subparts) > 1:
            tokens.extend(subparts)

    return tokens


class BM25SearchEngine:
    """Fast lexical search engine utilizing BM25Plus algorithm."""

    def __init__(
        self,
        chunks: List[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        path_weight: float = 2.0,
        name_boost: float = 5.0,
    ) -> None:
        """Build BM25 index and inverted path/name index from chunks."""
        self.chunks = chunks
        self.path_weight = path_weight
        self.name_boost = name_boost
        self.path_index: Dict[str, List[int]] = defaultdict(list)
        self.name_index: Dict[str, List[int]] = defaultdict(list)
        self.corpus_tokens: List[List[str]] = []

        file_headers: Dict[str, str] = {}
        for c in chunks:
            if c.file_path not in file_headers and c.file_path.endswith(
                (".md", ".markdown", ".rst", ".txt")
            ):
                try:
                    if os.path.exists(c.file_path):
                        with open(
                            c.file_path, "r", encoding="utf-8", errors="replace"
                        ) as f:
                            lines = f.readlines()
                        for line in lines:
                            if line.startswith("# "):
                                file_headers[c.file_path] = (
                                    line.lstrip("#").strip()
                                )
                                break
                        if c.file_path not in file_headers:
                            file_headers[c.file_path] = ""
                    else:
                        file_headers[c.file_path] = ""
                except Exception:
                    file_headers[c.file_path] = ""

        self.doc_indices: Set[int] = set()
        self.code_indices: Set[int] = set()

        for idx, c in enumerate(chunks):
            if (
                c.file_path.endswith((".md", ".rst", ".txt"))
                or "/docs/" in c.file_path
            ):
                self.doc_indices.add(idx)
            if c.file_path.endswith(".py"):
                self.code_indices.add(idx)

            doc_hdr = file_headers.get(c.file_path, "")
            p_text = f"{c.file_path} {doc_hdr}"
            p_toks = set(
                t
                for t in tokenize_text(p_text)
                if t not in STOP_WORDS and len(t) > 2
            )
            for pt in p_toks:
                self.path_index[pt].append(idx)

            stem = os.path.splitext(os.path.basename(c.file_path))[0]
            for st in tokenize_text(stem):
                if st not in STOP_WORDS and len(st) > 2:
                    self.name_index[st].append(idx)

            path_only_toks = set(
                t
                for t in tokenize_text(c.file_path)
                if t not in STOP_WORDS and len(t) > 2
            )
            c_toks = tokenize_text(c.text) + list(path_only_toks)
            self.corpus_tokens.append(c_toks)

        self.bm25: Optional[BM25Plus] = (
            BM25Plus(self.corpus_tokens, k1=k1, b=b, delta=delta)
            if self.corpus_tokens
            else None
        )

    def search(
        self,
        question_id: str,
        question: str,
        k: int = 10,
        target_type: Optional[str] = None,
    ) -> MinimalSearchResults:
        """Retrieve top-k relevant MinimalSource attributions for a query.

        Crash-proof against empty queries, k <= 0, or missing index.
        """
        if k <= 0 or not self.chunks or self.bm25 is None:
            return MinimalSearchResults(
                question_id=question_id,
                question=question,
                retrieved_sources=[],
            )

        q_tokens = tokenize_text(question)
        if not q_tokens:
            retrieved = [
                c.to_minimal_source()
                for c in self.chunks[: min(k, len(self.chunks))]
            ]
            return MinimalSearchResults(
                question_id=question_id,
                question=question,
                retrieved_sources=retrieved,
            )

        scores = list(self.bm25.get_scores(q_tokens))
        meaningful_q_tokens = set(
            t for t in q_tokens if t not in STOP_WORDS and len(t) > 2
        )

        for qt in meaningful_q_tokens:
            if self.path_weight > 0 and qt in self.path_index:
                for idx in self.path_index[qt]:
                    scores[idx] += self.path_weight
            if self.name_boost > 0 and qt in self.name_index:
                for idx in self.name_index[qt]:
                    scores[idx] += self.name_boost

        doc_bias = 0.0
        if target_type == "docs":
            doc_bias = 2.0
        elif target_type == "code":
            doc_bias = -2.0

        if doc_bias != 0.0:
            doc_idx_set: Set[int] = getattr(self, "doc_indices", set())
            code_idx_set: Set[int] = getattr(self, "code_indices", set())
            for idx in range(len(scores)):
                if idx in doc_idx_set:
                    scores[idx] += doc_bias
                elif idx in code_idx_set:
                    scores[idx] -= doc_bias

        # Sort indices by score descending
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: min(k, len(scores))]

        retrieved_sources: List[MinimalSource] = [
            self.chunks[i].to_minimal_source() for i in top_indices
        ]

        return MinimalSearchResults(
            question_id=question_id,
            question=question,
            retrieved_sources=retrieved_sources,
        )

    def save_index(self, processed_dir: str = "data/processed") -> None:
        """Persist BM25 index to disk for fast reload."""
        os.makedirs(processed_dir, exist_ok=True)
        idx_path = os.path.join(processed_dir, "bm25_engine.pkl")
        try:
            with open(idx_path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            print(f"Warning: Failed to save BM25 index: {e}")

    @classmethod
    def load_index(
        cls, processed_dir: str = "data/processed"
    ) -> "BM25SearchEngine":
        """Load BM25 index from disk or return an empty search engine."""
        idx_path = os.path.join(processed_dir, "bm25_engine.pkl")
        if os.path.exists(idx_path):
            try:
                with open(idx_path, "rb") as f:
                    engine = pickle.load(f)
                    if isinstance(engine, cls):
                        return engine
            except Exception:
                pass
        return cls(chunks=[])
