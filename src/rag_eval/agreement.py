"""
Inter-rater agreement for the Phase 3 judge meta-evaluation.

Cohen's kappa (two raters) and Krippendorff's alpha (nominal, two or more
raters) are computed from scratch rather than pulled from a statistics
package, so the Phase 3 numbers are auditable line by line. Both are validated
against reference values in ``tests/test_agreement.py``.
"""

from __future__ import annotations

import numpy as np


def cohen_kappa(ratings_a: list, ratings_b: list) -> dict:
    """
    Two raters, paired category labels. Returns kappa plus the observed and
    chance-agreement rates. When chance agreement is complete (both raters
    constant and identical), kappa is undefined: report 1.0 only when
    agreement is also perfect, else 0.0.
    """
    if len(ratings_a) != len(ratings_b):
        raise ValueError(f"rating lengths differ: {len(ratings_a)} vs {len(ratings_b)}")
    n = len(ratings_a)
    if n == 0:
        raise ValueError("need at least one rated item")

    p_observed = sum(a == b for a, b in zip(ratings_a, ratings_b)) / n
    p_expected = sum(
        (sum(1 for x in ratings_a if x == c) / n) * (sum(1 for x in ratings_b if x == c) / n)
        for c in set(ratings_a) | set(ratings_b)
    )
    denom = 1 - p_expected
    if denom <= 0:
        kappa = 1.0 if p_observed == 1.0 else 0.0
    else:
        kappa = (p_observed - p_expected) / denom
    return {"kappa": kappa, "p_observed": p_observed, "p_expected": p_expected, "n": n}


def cohen_kappa_ci(
    ratings_a: list,
    ratings_b: list,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    """Row-bootstrap CI for kappa; 60 hand-labeled rows need an interval."""
    pairs = list(zip(ratings_a, ratings_b))
    if not pairs:
        raise ValueError("need at least one rated item")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        sample = rng.integers(0, len(pairs), size=len(pairs))
        a = [pairs[i][0] for i in sample]
        b = [pairs[i][1] for i in sample]
        values.append(cohen_kappa(a, b)["kappa"])
    alpha = 1 - confidence
    return {
        "ci_lower": float(np.percentile(values, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(values, 100 * (1 - alpha / 2))),
    }


def krippendorff_alpha(ratings: list[list]) -> float:
    """
    Nominal Krippendorff's alpha over 2+ raters.

    ``ratings[item][rater]`` is the label; items with fewer than two ratings are
    excluded, as the definition requires. The coincidence matrix is built the
    standard way — ``n_cc = n_c(n_c - 1)/(m_u - 1)`` on the diagonal and
    ``n_cc' = n_c n_c'/(m_u - 1)`` off it — then

        D_o = sum_{c != c'} n_cc' / n
        D_e = sum_{c != c'} n_c n_c' / (n (n - 1))

    with ``n`` the total number of pairable values. An earlier version
    normalised by rater-pair counts and was wrong for three or more raters
    (0.0556 where the reference gives 0.1111 on the test case).
    """
    coincidences: dict[tuple, float] = {}
    n = 0

    for item in ratings:
        k = len(item)
        if k < 2:
            continue
        n += k
        counts: dict = {}
        for label in item:
            counts[label] = counts.get(label, 0) + 1
        for c, cnt_c in counts.items():
            coincidences[(c, c)] = coincidences.get((c, c), 0.0) + cnt_c * (cnt_c - 1) / (k - 1)
            for d, cnt_d in counts.items():
                if c != d:
                    key = (c, d)
                    coincidences[key] = coincidences.get(key, 0.0) + cnt_c * cnt_d / (k - 1)

    if n == 0:
        raise ValueError("need at least one item with two raters")

    d_o = sum(v for (c, d), v in coincidences.items() if c != d) / n

    marginals: dict = {}
    for (c, _d), value in coincidences.items():
        marginals[c] = marginals.get(c, 0.0) + value

    d_e = sum(
        marginals[c] * marginals[d]
        for c in marginals
        for d in marginals
        if c != d
    ) / (n * (n - 1))
    if d_e == 0:
        return 1.0
    return 1 - d_o / d_e
