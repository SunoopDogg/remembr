"""Novelty-aware result re-ranking utilities."""


def rerank_by_novelty(
    results: list[dict],
    novelty_weight: float,
) -> list[dict]:
    """Re-rank search results using hybrid distance + novelty score."""
    if novelty_weight == 0.0 or len(results) <= 1:
        return results

    max_dist = max(r['distance'] for r in results)
    if max_dist == 0.0:
        return results

    scored = []
    for r in results:
        norm_dist = r['distance'] / max_dist
        novelty = r.get('novelty_score', 0.0)
        hybrid = (1.0 - novelty_weight) * norm_dist + novelty_weight * (1.0 - novelty)
        scored.append((hybrid, r))

    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored]
