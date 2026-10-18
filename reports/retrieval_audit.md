# Candidate-level retrieval audit

Metrics over the top-50 candidates (pre-rerank, pre-truncation), recomputed
from the cached indexes. `window prefix` is the share of stored windows that
equal the candidate prefix; `window set` is the share whose chunks are all in
the candidate list. A set match with a prefix mismatch is an adjacent swap of
near-tied scores (OpenAI embeddings are deterministic per call but differ by
~1e-4 across batch compositions), not a different retrieval result.

| cell | recall@10 | hits@10 | nDCG@10 | MAP@10 | MRR | window prefix | window set |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline/dense/1024/128 | 0.479 | 0.810 | 0.285 | 0.225 | 0.478 | 98.8% | 100.0% |
| chunk_size/2048 | 0.535 | 0.873 | 0.315 | 0.248 | 0.515 | 99.7% | 100.0% |
| chunk_size/512 | 0.403 | 0.733 | 0.244 | 0.195 | 0.415 | 95.1% | 100.0% |
| embedder/bge-small-en-v1.5 | 0.396 | 0.733 | 0.236 | 0.186 | 0.413 | 100.0% | 100.0% |
| overlap/0 | 0.487 | 0.823 | 0.287 | 0.226 | 0.482 | 98.5% | 100.0% |
| overlap/64 | 0.486 | 0.824 | 0.287 | 0.226 | 0.483 | 98.6% | 100.0% |
| recency_filter/before-median | 0.286 | 0.573 | 0.176 | 0.143 | 0.332 | 99.2% | 100.0% |
| rerank/bge-reranker-v2-m3 | 0.479 | 0.810 | 0.285 | 0.225 | 0.478 | 100.0% | 100.0% |
| retriever/bm25 | 0.548 | 0.861 | 0.347 | 0.285 | 0.579 | 99.9% | 100.0% |
| retriever/hybrid_rrf | 0.566 | 0.877 | 0.349 | 0.281 | 0.568 | 99.1% | 100.0% |

Chunk-level metrics. The dataset's official script ranks documents and reports
Hits@4/Hits@10/MAP@10/MRR@10; these values are not directly comparable to its
table. What they do provide is a retrieval-quality measure that does not move
when the context budget fits a different number of chunks.

