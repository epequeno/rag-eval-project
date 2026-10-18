# Phase 2 coverage matrix (regenerated)

`phenomenon` is the paired change in `answer_correctness` (ground truth);
the `answer_correctness` column itself is marked `T`, not scored as a detector.

`+` caught (metric drops and the answer drops) · `-` blind (answer drops, metric does not) ·
`!` false alarm (metric drops, answer does not) · `.` neutral (neither drops).

For retrieval-side classes the perturbation is present by construction; a
retrieval metric that moves while the answer does not is reported as a false
alarm because it is not a proxy for user-visible failure.

slice              phenomenon            recall@1            recall@4           recall@10                 mrr        faithfulness    answer_relevance   context_precision  answer_correctness
chunk_boundary         absent                    !                   !                   !                   !                   .                   .                   .                   T
false_neighbor         absent                    !                   .                   .                   !                   .                   .                   !                   T
retrieval_miss        present                    +                   +                   +                   +                   -                   +                   +                   T
lost_in_middle         absent                    .                   .                   .                   .                   .                   .                   .                   T
hallucination         present                    -                   -                   -                   -                   -                   -                   -                   T
evidence_split           weak                    +                   -                   -                   -                   +                   -                   +                   T

## chunk_boundary

Does recall@k falsely flag a split fact whose answer is still present?  (n=24/24)

| metric | control | engineered | delta | 95% CI |
|---|---|---|---|---|
| recall@1 | 1.000 | 0.000 | -1.000 | [-1.00, -1.00] |
| recall@4 | 1.000 | 0.000 | -1.000 | [-1.00, -1.00] |
| recall@10 | 1.000 | 0.000 | -1.000 | [-1.00, -1.00] |
| mrr | 1.000 | 0.000 | -1.000 | [-1.00, -1.00] |
| faithfulness | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| answer_relevance | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| context_precision | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| answer_correctness | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |

## false_neighbor

Does any metric see that the right evidence is ranked second, not first?  (n=24/24)

| metric | control | engineered | delta | 95% CI |
|---|---|---|---|---|
| recall@1 | 1.000 | 0.000 | -1.000 | [-1.00, -1.00] |
| recall@4 | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| recall@10 | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| mrr | 1.000 | 0.500 | -0.500 | [-0.50, -0.50] |
| faithfulness | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| answer_relevance | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| context_precision | 0.917 | 0.562 | -0.354 | [-0.44, -0.25] |
| answer_correctness | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |

## retrieval_miss

Does faithfulness flag an answer withheld because retrieval missed?  (n=24/24)

| metric | control | engineered | delta | 95% CI |
|---|---|---|---|---|
| recall@1 | 1.000 | 0.000 | -1.000 | [-1.00, -1.00] |
| recall@4 | 1.000 | 0.000 | -1.000 | [-1.00, -1.00] |
| recall@10 | 1.000 | 0.000 | -1.000 | [-1.00, -1.00] |
| mrr | 1.000 | 0.000 | -1.000 | [-1.00, -1.00] |
| faithfulness | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| answer_relevance | 1.000 | 0.208 | -0.792 | [-0.96, -0.62] |
| context_precision | 1.000 | 0.000 | -1.000 | [-1.00, -1.00] |
| answer_correctness | 1.000 | 0.000 | -1.000 | [-1.00, -1.00] |

## lost_in_middle

Does answer correctness drop when the evidence sits mid-prompt?  (n=24/24)

| metric | control | engineered | delta | 95% CI |
|---|---|---|---|---|
| recall@1 | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| recall@4 | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| recall@10 | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| mrr | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| faithfulness | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| answer_relevance | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| context_precision | 0.114 | 0.114 | +0.000 | [+0.00, +0.00] |
| answer_correctness | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |

## hallucination

Does correctness catch a context-supported wrong answer that faithfulness misses?  (n=24/24)

| metric | control | engineered | delta | 95% CI |
|---|---|---|---|---|
| recall@1 | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| recall@4 | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| recall@10 | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| mrr | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| faithfulness | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| answer_relevance | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| context_precision | 0.500 | 0.833 | +0.333 | [+0.23, +0.42] |
| answer_correctness | 1.000 | 0.000 | -1.000 | [-1.00, -1.00] |

## evidence_split

Do end-to-end metrics notice a synthesis demand retrieval metrics cannot see?  (n=24/24)

| metric | control | engineered | delta | 95% CI |
|---|---|---|---|---|
| recall@1 | 1.000 | 0.500 | -0.500 | [-0.50, -0.50] |
| recall@4 | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| recall@10 | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| mrr | 1.000 | 1.000 | +0.000 | [+0.00, +0.00] |
| faithfulness | 1.000 | 0.833 | -0.167 | [-0.27, -0.08] |
| answer_relevance | 1.000 | 0.917 | -0.083 | [-0.21, +0.00] |
| context_precision | 0.938 | 0.500 | -0.438 | [-0.50, -0.33] |
| answer_correctness | 1.000 | 0.917 | -0.083 | [-0.21, +0.00] |

