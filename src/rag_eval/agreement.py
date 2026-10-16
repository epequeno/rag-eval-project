"""
Inter-rater agreement for the Phase 3 judge meta-evaluation.

Cohen's kappa (two raters) and Krippendorff's alpha (nominal, two or more
raters) are computed from scratch rather than pulled from a statistics
package, so the Phase 3 numbers are auditable line by line.
"""

from __future__ import annotations

from collections import Counter


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


def krippendorff_alpha(ratings: list[list]) -> float:
    """
    Nominal Krippendorff's alpha over 2+ raters.

    ``ratings[item][rater]`` is the label; every item needs at least two
    ratings. Coincidence counts are over *ordered* rater pairs within each
    item, which is what makes alpha extend to more than two raters.
    """
    m = 0  # total ordered rater pairs
    n_cc = 0  # ordered pairs that agree
    row = Counter()  # per-category pair participation

    for item in ratings:
        k = len(item)
        if k < 2:
            continue
        counts = Counter(item)
        m += k * (k - 1)
        for label, cnt in counts.items():
            row[label] += cnt * (k - 1)
            n_cc += cnt * (cnt - 1)

    if m == 0:
        raise ValueError("need at least one item with two raters")

    # Expected (chance) disagreement: sum_c n_c (m - n_c) / (m - 1)
    de = sum(row[c] * (m - row[c]) for c in row) / (m - 1)
    if de == 0:
        return 1.0
    return 1 - (m - n_cc) / de