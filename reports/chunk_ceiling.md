# Chunk ceiling

The maximum `recall@k` any retriever can reach under a given chunking: the share of
gold evidence facts contained in some chunk of their own source document. Computed
over the unfiltered corpus, so a `recency_filter` cell has a lower true ceiling —
compare such a cell against its own recall history, not against this table.

Documents: 609 · queries: 2556 · gold facts: 6084

| chunk size | overlap | chunks | facts reachable | ceiling |
|---:|---:|---:|---:|---:|
| 512 | 128 | 17445 | 6082/6084 | 0.99967 |
| 1024 | 128 | 7880 | 6084/6084 | 1.00000 |
| 2048 | 128 | 3702 | 6084/6084 | 1.00000 |

## Gold-fact lengths (characters)

| p10 | p50 | p90 | max |
|---:|---:|---:|---:|
| 93 | 161 | 257 | 538 |

Chunk-boundary loss is the difference between the two: a fact longer than the chunk
size cannot fit. At the smallest size here that is 2 fact(s) out of
6084 — which is why Phase 2 engineers the split explicitly rather than
waiting for the natural grid to produce it.

Regenerate with `HF_HUB_OFFLINE=1 uv run python scripts/chunk_ceiling.py`.
