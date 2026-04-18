"""Synthetic inference driver.

Streams per-lookup telemetry from a model with Engram lookups attached at
specific layers. Writes events as Parquet (much faster than JSONL at scale;
ClickHouse reads it natively via ``file(..., Parquet)``).

Four request classes map to rough LLM workload shapes with varying overlap
with the memorized corpus — so their hit rates differ meaningfully rather
than being uniform noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from engram import EngramTable, hash_ngrams_batch


VOCAB_SIZE = 32_000
NUM_BUCKETS = 1_000_000
ENGRAM_LAYERS = (2, 15)
NUM_MEMORIZED = 200_000

CLASS_OVERLAP = {
    "factual Q&A":     0.95,
    "code completion": 0.60,
    "general chat":    0.25,
    "math reasoning":  0.05,
}


def zipf_tokens(rng: np.random.Generator, n: int, vocab_size: int, a: float = 1.2) -> np.ndarray:
    out = np.empty(n, dtype=np.int32)
    filled = 0
    while filled < n:
        batch = rng.zipf(a, size=max(n * 2, 1024)) - 1
        batch = batch[batch < vocab_size]
        take = min(len(batch), n - filled)
        out[filled : filled + take] = batch[:take]
        filled += take
    return out


def build_class_samplers(rng: np.random.Generator, vocab_size: int):
    train_perm = rng.permutation(vocab_size).astype(np.int32)
    samplers = {}
    for cls, overlap in CLASS_OVERLAP.items():
        own = rng.permutation(vocab_size).astype(np.int32)
        n_overlap = int(vocab_size * overlap)
        mixed = own.copy()
        shared_idx = rng.choice(vocab_size, size=n_overlap, replace=False)
        mixed[shared_idx] = train_perm[shared_idx]
        samplers[cls] = mixed
    return train_perm, samplers


def memorize_training_ngrams(
    table: EngramTable,
    rng: np.random.Generator,
    train_perm: np.ndarray,
    num_ngrams: int,
    vocab_size: int,
) -> None:
    raw_a = zipf_tokens(rng, num_ngrams, vocab_size)
    raw_b = zipf_tokens(rng, num_ngrams, vocab_size)
    pairs = np.stack([train_perm[raw_a], train_perm[raw_b]], axis=1).astype(np.uint32)
    buckets = hash_ngrams_batch(pairs, table.num_buckets)
    table.populated_mask[buckets] = True


def run_experiment(
    n_requests: int,
    tokens_per_request: int,
    out_path: str,
    seed: int = 0,
    chunk_requests: int = 4_000,
) -> dict:
    rng = np.random.default_rng(seed)
    table = EngramTable(NUM_BUCKETS, d_model=0, seed=seed)
    train_perm, samplers = build_class_samplers(rng, VOCAB_SIZE)

    print(f"Memorizing {NUM_MEMORIZED:,} training n-grams...")
    memorize_training_ngrams(table, rng, train_perm, NUM_MEMORIZED, VOCAB_SIZE)
    populated = table.populated_count
    print(f"  → {populated:,} buckets populated ({populated / NUM_BUCKETS:.1%})")

    classes = list(samplers.keys())
    pairs_per_req = tokens_per_request - 1
    layers_arr = np.asarray(ENGRAM_LAYERS, dtype=np.uint8)
    total_events = n_requests * pairs_per_req * len(ENGRAM_LAYERS)
    print(
        f"Running {n_requests:,} requests × {tokens_per_request} tokens × "
        f"{len(ENGRAM_LAYERS)} engram layers → {total_events:,} events"
    )

    writer: pq.ParquetWriter | None = None
    schema = pa.schema(
        [
            ("ts", pa.int64()),
            ("request_id", pa.string()),
            ("request_class", pa.string()),
            ("layer", pa.uint8()),
            ("token_pos", pa.uint16()),
            ("ngram", pa.list_(pa.uint32())),
            ("bucket", pa.uint32()),
            ("hit", pa.uint8()),
        ]
    )
    try:
        writer = pq.ParquetWriter(out_path, schema, compression="zstd")
        ts0 = 1_700_000_000_000_000_000
        for chunk_start in range(0, n_requests, chunk_requests):
            chunk_end = min(chunk_start + chunk_requests, n_requests)
            n_chunk = chunk_end - chunk_start
            n_chunk_events = n_chunk * pairs_per_req * len(ENGRAM_LAYERS)

            ts = np.empty(n_chunk_events, dtype=np.int64)
            request_id = np.empty(n_chunk_events, dtype=object)
            request_class = np.empty(n_chunk_events, dtype=object)
            layer_col = np.empty(n_chunk_events, dtype=np.uint8)
            token_pos = np.empty(n_chunk_events, dtype=np.uint16)
            ngram_a = np.empty(n_chunk_events, dtype=np.uint32)
            ngram_b = np.empty(n_chunk_events, dtype=np.uint32)
            bucket_col = np.empty(n_chunk_events, dtype=np.uint32)
            hit_col = np.empty(n_chunk_events, dtype=np.uint8)

            write_ofs = 0
            for req_idx in range(chunk_start, chunk_end):
                cls = classes[req_idx % len(classes)]
                raw = zipf_tokens(rng, tokens_per_request, VOCAB_SIZE)
                tokens = samplers[cls][raw].astype(np.uint32)
                pairs = np.stack([tokens[:-1], tokens[1:]], axis=1)
                buckets, hits = table.lookup_batch(pairs)

                for li, layer in enumerate(layers_arr):
                    slot = slice(write_ofs, write_ofs + pairs_per_req)
                    base_ts = ts0 + req_idx * 1_000_000 + li * 10
                    ts[slot] = base_ts + np.arange(pairs_per_req, dtype=np.int64) * 1_000
                    request_id[slot] = f"req_{req_idx:08d}"
                    request_class[slot] = cls
                    layer_col[slot] = layer
                    token_pos[slot] = np.arange(1, pairs_per_req + 1, dtype=np.uint16)
                    ngram_a[slot] = pairs[:, 0]
                    ngram_b[slot] = pairs[:, 1]
                    bucket_col[slot] = buckets
                    hit_col[slot] = hits.astype(np.uint8)
                    write_ofs += pairs_per_req

            ngram_col = pa.array(
                list(zip(ngram_a.tolist(), ngram_b.tolist())),
                type=pa.list_(pa.uint32()),
            )
            batch = pa.record_batch(
                [
                    pa.array(ts),
                    pa.array(request_id, type=pa.string()),
                    pa.array(request_class, type=pa.string()),
                    pa.array(layer_col),
                    pa.array(token_pos),
                    ngram_col,
                    pa.array(bucket_col),
                    pa.array(hit_col),
                ],
                schema=schema,
            )
            writer.write_batch(batch)
            print(f"  chunk {chunk_start:,}–{chunk_end:,} written ({n_chunk_events:,} events)")
    finally:
        if writer is not None:
            writer.close()

    print(f"Emitted {total_events:,} events to {out_path}")
    return {
        "num_buckets": NUM_BUCKETS,
        "populated_buckets": populated,
        "total_events": total_events,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-requests", type=int, default=40_000)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--out", default="data/events.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    run_experiment(args.n_requests, args.tokens, args.out, seed=args.seed)
