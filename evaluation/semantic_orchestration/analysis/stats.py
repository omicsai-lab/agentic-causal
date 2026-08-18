"""
Statistical primitives used by the semantic-orchestration analysis:
Wilson score confidence intervals, exact McNemar's test, paired bootstrap
CIs (seeded), and confusion-matrix construction. Pure functions, no I/O,
so they are unit-testable in isolation (see tests/test_stats.py).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

try:
    from scipy.stats import binom as _scipy_binom
except Exception:  # pragma: no cover
    _scipy_binom = None


BOOTSTRAP_SEED = 20260818


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float, float]:
    """
    Wilson score interval for a binomial proportion.

    Returns (point_estimate, lower, upper). z defaults to the 97.5th
    percentile of the standard normal (two-sided 95% CI).
    """
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} out of range for n={n}")

    p_hat = successes / n
    denom = 1 + (z**2) / n
    center = p_hat + (z**2) / (2 * n)
    half_width = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2) / (4 * n**2))

    lower = (center - half_width) / denom
    upper = (center + half_width) / denom
    return (p_hat, max(0.0, lower), min(1.0, upper))


def exact_mcnemar(b: int, c: int) -> Dict[str, float]:
    """
    Exact (binomial-based) McNemar's test on the discordant pair counts.

    b = count where model A correct and model B incorrect (or: condition A
        True, condition B False, on matched items)
    c = count where model A incorrect and model B correct

    Returns dict with n_discordant, b, c, p_value.
    Uses the exact two-sided binomial test on Binomial(b+c, 0.5), which is
    exact for any b+c (no continuity-correction chi-square approximation).
    """
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}

    k = min(b, c)

    if _scipy_binom is not None:
        # Two-sided exact binomial test: sum P(X=i) for all i with
        # P(X=i) <= P(X=k) under X ~ Binomial(n, 0.5).
        pmf = [_scipy_binom.pmf(i, n, 0.5) for i in range(n + 1)]
        p_k = pmf[k]
        eps = 1e-12
        p_value = sum(p for p in pmf if p <= p_k + eps)
        p_value = min(1.0, p_value)
    else:  # pragma: no cover - scipy is a pinned dependency, fallback kept for safety
        p_value = _binom_two_sided_fallback(k, n)

    return {"b": b, "c": c, "n_discordant": n, "p_value": p_value}


def _binom_two_sided_fallback(k: int, n: int) -> float:
    def log_choose(n_: int, k_: int) -> float:
        return math.lgamma(n_ + 1) - math.lgamma(k_ + 1) - math.lgamma(n_ - k_ + 1)

    pmf = [math.exp(log_choose(n, i) - n * math.log(2)) for i in range(n + 1)]
    p_k = pmf[k]
    eps = 1e-12
    return min(1.0, sum(p for p in pmf if p <= p_k + eps))


def paired_bootstrap_ci(
    diffs: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Percentile bootstrap CI for the mean of paired differences.

    `diffs` is a sequence of per-item paired differences (e.g. correct_A_i -
    correct_B_i for matched items i). Deterministic given `seed`.
    """
    import random

    n = len(diffs)
    if n == 0:
        return {"mean_diff": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"), "n": 0, "n_resamples": n_resamples, "seed": seed}

    rng = random.Random(seed)
    mean_diff = sum(diffs) / n

    boot_means = []
    idx_range = range(n)
    for _ in range(n_resamples):
        resample = [diffs[rng.randrange(n)] for _ in idx_range]
        boot_means.append(sum(resample) / n)

    boot_means.sort()
    lower_idx = int((alpha / 2) * n_resamples)
    upper_idx = int((1 - alpha / 2) * n_resamples) - 1
    lower_idx = max(0, min(lower_idx, n_resamples - 1))
    upper_idx = max(0, min(upper_idx, n_resamples - 1))

    return {
        "mean_diff": mean_diff,
        "ci_lower": boot_means[lower_idx],
        "ci_upper": boot_means[upper_idx],
        "n": n,
        "n_resamples": n_resamples,
        "seed": seed,
        "alpha": alpha,
    }


def confusion_matrix(
    gold_labels: Sequence[str],
    pred_labels: Sequence[str],
    *,
    label_order: List[str] | None = None,
) -> Dict[str, object]:
    """
    Build a confusion matrix as {gold: {pred: count}} plus a fixed
    `labels` ordering for stable table/figure rendering.
    """
    if len(gold_labels) != len(pred_labels):
        raise ValueError("gold_labels and pred_labels must be the same length")

    labels = label_order or sorted(set(gold_labels) | set(pred_labels))
    matrix: Dict[str, Dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}

    for g, p in zip(gold_labels, pred_labels):
        if g not in matrix:
            matrix[g] = {p2: 0 for p2 in labels}
            labels.append(g)
        if p not in matrix[g]:
            matrix[g][p] = 0
        matrix[g][p] += 1

    return {"labels": labels, "matrix": matrix}


def pairwise_agreement(runs: Sequence[Sequence[str]]) -> Dict[str, float]:
    """
    Given >=2 repeated-run label sequences of equal length (one sequence per
    repeat, aligned by item index), compute mean pairwise agreement rate
    across all repeat pairs, and the fraction of items where ALL repeats
    agree (full consistency).
    """
    import itertools

    if len(runs) < 2:
        raise ValueError("pairwise_agreement requires at least 2 runs")
    n_items = len(runs[0])
    for r in runs:
        if len(r) != n_items:
            raise ValueError("all runs must have the same number of items")

    pair_rates = []
    for r1, r2 in itertools.combinations(runs, 2):
        agree = sum(1 for a, b in zip(r1, r2) if a == b)
        pair_rates.append(agree / n_items if n_items else float("nan"))

    full_agree = 0
    for i in range(n_items):
        vals = {r[i] for r in runs}
        if len(vals) == 1:
            full_agree += 1
    full_consistency_rate = full_agree / n_items if n_items else float("nan")

    return {
        "mean_pairwise_agreement": sum(pair_rates) / len(pair_rates) if pair_rates else float("nan"),
        "pairwise_rates": pair_rates,
        "full_consistency_rate": full_consistency_rate,
        "n_items": n_items,
        "n_runs": len(runs),
    }


def majority_vote(labels: Sequence[str]) -> Tuple[str, bool]:
    """
    Majority vote over a small set of repeat labels (e.g. 3 runs).

    Returns (winning_label, is_unanimous_or_strict_majority). Ties are
    broken by first-occurrence order (deterministic, no randomness) and
    flagged via the second return value being False.
    """
    if not labels:
        raise ValueError("majority_vote requires at least one label")

    counts: Dict[str, int] = {}
    order: List[str] = []
    for lab in labels:
        if lab not in counts:
            counts[lab] = 0
            order.append(lab)
        counts[lab] += 1

    best_count = max(counts.values())
    winners = [lab for lab in order if counts[lab] == best_count]
    is_clean = len(winners) == 1 and best_count > len(labels) / 2
    return winners[0], is_clean
