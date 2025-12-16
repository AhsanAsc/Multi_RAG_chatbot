from __future__ import annotations
from typing import List
import math


# Cosine similarity (vectors already come from the same embedding model)
def _cos(a: List[float], b: List[float]) -> float:
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (da * db)


def mmr(
    query_vec: List[float],
    cand_vecs: List[List[float]],
    k: int,
    lambd: float = 0.7,
) -> List[int]:
    """
    Maximal Marginal Relevance:
    score(i) = λ * sim(query, i) - (1-λ) * max_{j in selected} sim(i, j)
    Returns indices of selected items in order.
    """
    n = len(cand_vecs)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)

    # Precompute similarities to query (relevance)
    rel = [_cos(query_vec, v) for v in cand_vecs]

    selected: List[int] = []
    remaining: set[int] = set(range(n))

    # 1) pick the most relevant first
    best0 = max(remaining, key=lambda i: rel[i])
    selected.append(best0)
    remaining.remove(best0)

    # 2) iteratively add the item that maximizes MMR
    while len(selected) < k and remaining:
        best_i = None
        best_score = -math.inf
        for i in remaining:
            # diversity term: most similar to any already selected
            if selected:
                div = max(_cos(cand_vecs[i], cand_vecs[j]) for j in selected)
            else:
                div = 0.0
            s = lambd * rel[i] - (1.0 - lambd) * div
            if s > best_score:
                best_score = s
                best_i = i
        selected.append(best_i)  # type: ignore
        remaining.remove(best_i)  # type: ignore
    return selected
