import numpy as np


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = 42,
) -> dict[str, float | None]:
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = np.random.default_rng(seed)

    means = np.array(
        [np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(n_resamples)]
    )

    alpha = 1 - confidence
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(arr)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        # Sample standard deviation is undefined for a single row; reporting
        # 0.0 or nan would hide that, so say so.
        "std": float(np.std(arr, ddof=1)) if n > 1 else None,
        "n": n,
    }


def effect_size_paired(a, b, confidence=0.95, n_resamples=10000, seed=42):
    """
    Paired bootstrap: resample per-instance differences directly.

    ``p_value`` is the two-sided bootstrap tail probability. It is reported so
    a family of tests can be multiplicity-corrected; the original sweep printed
    a bare ``significant`` flag for 24 comparisons at 95%.
    """
    diffs = np.array(a, dtype=float) - np.array(b, dtype=float)
    if len(diffs) == 0:
        raise ValueError("effect_size_paired needs at least one pair")
    rng = np.random.default_rng(seed)

    means = np.array(
        [
            np.mean(rng.choice(diffs, size=len(diffs), replace=True))
            for _ in range(n_resamples)
        ]
    )

    alpha = 1 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    p_value = float(min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0))))
    return {
        "delta": float(np.mean(diffs)),
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": p_value,
        "significant": bool(lower > 0 or upper < 0),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, same order as the input."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        index = order[rank - 1]
        running = min(running, p_values[index] * m / rank)
        adjusted[index] = running
    return adjusted


def aggregate(per_instance, confidence=0.95, n_resamples=10000, seed=42):
    """
    Aggregate per-instance metric dicts into mean + bootstrap CI.

    A metric that is ``None`` for a row is excluded from that metric's
    aggregate rather than counted as zero: retrieval metrics on null queries
    and judge-backed metrics when no judge was injected are undefined, not 0.0.
    The returned ``n`` reports how many instances contributed.
    """
    if not per_instance:
        return {}
    # Union of keys in first-seen order: rows can legitimately carry different
    # key sets, so deriving the metric list from row 0 alone would drop whole
    # metrics from the summary.
    metric_names = dict.fromkeys(name for row in per_instance for name in row)
    result = {}
    for name in metric_names:
        values = [row[name] for row in per_instance if row.get(name) is not None]
        if not values:
            continue
        result[name] = bootstrap_ci(values, confidence, n_resamples, seed)
    return result
