"""Hash-indexed Engram lookup table — simplified model of DeepSeek's conditional memory.

An embedding table lives in CPU RAM. Token n-grams are hashed to a bucket; only
a subset of buckets are "populated" (the n-grams the model actually memorized
during training). At inference, lookups against populated buckets are hits and
return an embedding; misses return None.

The hash is a vectorized FNV-1a so the same function works for single-ngram
lookups and batched per-request lookups without any divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

_FNV_OFFSET = np.uint64(14695981039346656037)
_FNV_PRIME = np.uint64(1099511628211)


def hash_ngrams_batch(pairs: np.ndarray, num_buckets: int) -> np.ndarray:
    """Vectorized FNV-1a over columns of ``pairs``. Returns bucket ids."""
    if pairs.ndim == 1:
        pairs = pairs.reshape(1, -1)
    pairs64 = pairs.astype(np.uint64, copy=False)
    h = np.full(pairs.shape[0], _FNV_OFFSET, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for col in range(pairs.shape[1]):
            h = h ^ pairs64[:, col]
            h = h * _FNV_PRIME
    return (h % np.uint64(num_buckets)).astype(np.uint32)


def hash_ngram(ngram: tuple[int, ...], num_buckets: int) -> int:
    return int(hash_ngrams_batch(np.asarray(ngram, dtype=np.uint32), num_buckets)[0])


@dataclass
class LookupResult:
    ngram: tuple[int, ...]
    bucket: int
    hit: bool
    embedding: torch.Tensor | None


class EngramTable:
    def __init__(self, num_buckets: int, d_model: int = 0, seed: int = 0):
        """``d_model=0`` skips the embedding allocation — useful when the
        experiment only needs hit/miss telemetry, which saves GBs at scale."""
        self.num_buckets = num_buckets
        self.d_model = d_model
        self.populated_mask = np.zeros(num_buckets, dtype=bool)
        if d_model > 0:
            rng = np.random.default_rng(seed)
            self.embeddings = torch.from_numpy(
                (rng.standard_normal((num_buckets, d_model)) * 0.02).astype(np.float32)
            )
        else:
            self.embeddings = None

    def memorize(self, ngrams: Iterable[tuple[int, ...]]) -> int:
        arr = np.asarray(list(ngrams), dtype=np.uint32)
        buckets = hash_ngrams_batch(arr, self.num_buckets)
        self.populated_mask[buckets] = True
        return int(self.populated_mask.sum())

    def lookup_batch(self, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        buckets = hash_ngrams_batch(pairs, self.num_buckets)
        hits = self.populated_mask[buckets]
        return buckets, hits

    def lookup(self, ngram: tuple[int, ...]) -> LookupResult:
        bucket = hash_ngram(ngram, self.num_buckets)
        hit = bool(self.populated_mask[bucket])
        emb = self.embeddings[bucket] if (hit and self.embeddings is not None) else None
        return LookupResult(ngram=ngram, bucket=bucket, hit=hit, embedding=emb)

    @property
    def populated_count(self) -> int:
        return int(self.populated_mask.sum())
