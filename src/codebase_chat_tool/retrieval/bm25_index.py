from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split code/prose into lowercase tokens, breaking snake_case and camelCase
    identifiers into sub-tokens since identifier pieces carry most of the signal
    in code search (e.g. 'get_user' -> ['get_user', 'get', 'user'])."""
    tokens: list[str] = []
    for raw in _NON_ALNUM.split(text):
        if not raw:
            continue
        lowered = raw.lower()
        tokens.append(lowered)
        for part in _CAMEL_SPLIT.split(raw):
            if part and part.lower() != lowered:
                tokens.append(part.lower())
        if "_" in raw:
            tokens.extend(p.lower() for p in raw.split("_") if p)
    return tokens


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunk_ids: list[str]

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [(cid, score) for cid, score in ranked[:top_k] if score > 0]


def build_bm25_index(chunk_ids: list[str], texts: list[str]) -> BM25Index:
    tokenized_corpus = [tokenize(t) for t in texts]
    return BM25Index(bm25=BM25Okapi(tokenized_corpus), chunk_ids=chunk_ids)


def save_bm25_index(index: BM25Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(index, f)


def load_bm25_index(path: Path) -> BM25Index:
    with path.open("rb") as f:
        return pickle.load(f)
