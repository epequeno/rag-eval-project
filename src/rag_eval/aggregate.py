import numpy as np


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = 42,
) -> dict[str, float]:
    arr = np.array(values)
    n = len(arr)
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
        "std": float(np.std(arr, ddof=1)),
        "n": n,
    }


def effect_size(
    a: list[float],
    b: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    arr_a = np.array(a)
    arr_b = np.array(b)

    means_a = np.array(
        [
            np.mean(rng.choice(arr_a, size=len(arr_a), replace=True))
            for _ in range(n_resamples)
        ]
    )

    means_b = np.array(
        [
            np.mean(rng.choice(arr_b, size=len(arr_b), replace=True))
            for _ in range(n_resamples)
        ]
    )

    deltas = means_a - means_b
    alpha = 1 - confidence
    return {
        "delta": float(np.mean(arr_a) - np.mean(arr_b)),
        "ci_upper": float(np.percentile(deltas, 100 * (1 - alpha / 2))),
        "ci_lower": float(np.percentile(deltas, 100 * alpha / 2)),
        "significant": bool(
            np.percentile(deltas, 100 * alpha / 2) > 0
            or np.percentile(deltas, 100 * (1 - alpha / 2)) < 0
        ),
    }


def effect_size_paired(a, b, confidence=0.95, n_resamples=10000, seed=42):
    """Paired bootstrap: resample per-instance differences directly."""
    diffs = np.array(a) - np.array(b)
    rng = np.random.default_rng(seed)

    means = np.array(
        [
            np.mean(rng.choice(diffs, size=len(diffs), replace=True))
            for _ in range(n_resamples)
        ]
    )

    alpha = 1 - confidence
    return {
        "delta": float(np.mean(diffs)),
        "ci_lower": float(np.percentile(means, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "significant": bool(
            np.percentile(means, 100 * alpha / 2) > 0
            or np.percentile(means, 100 * (1 - alpha / 2)) < 0
        ),
    }


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
    # key sets (null queries have no judge-backed metrics), so deriving the
    # metric list from row 0 alone would drop whole metrics from the summary.
    metric_names = dict.fromkeys(name for row in per_instance for name in row)
    result = {}
    for name in metric_names:
        values = [row[name] for row in per_instance if row.get(name) is not None]
        if not values:
            continue
        result[name] = bootstrap_ci(values, confidence, n_resamples, seed)
    return result
