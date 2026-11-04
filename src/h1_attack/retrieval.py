"""Offline ATA selection only. No victim access and no attack-loss modification."""
import random
import numpy as np


def normalized(values):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("Features must be a nonempty finite matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("Zero/collapsed features are not valid")
    return matrix / norms


def select_auxiliaries(features, target_feature, ids, hashes, target_hash,
                       method="global", top_l=5, semantic_weight=1.,
                       diversity_weight=.15, concept_features=None, seed=20261026,
                       excluded_hashes=()):
    """Pick exactly two from the same global top-L pool for every control.

    Concept score = mean image/concept cosine; greedy MMR penalizes maximal
    similarity to an already selected auxiliary. Both extra terms are disabled
    for global/random. This is a proposed reranker, not verified grounding.
    """
    if method not in ("global", "random", "diversity", "concept"):
        raise ValueError("Unknown selection method")
    if not isinstance(top_l, int) or top_l < 2:
        raise ValueError("top_l must admit at least two auxiliaries")
    if not np.isfinite([semantic_weight, diversity_weight]).all() or min(semantic_weight, diversity_weight) < 0:
        raise ValueError("Weights must be finite and nonnegative")
    x = normalized(features)
    t = normalized(np.asarray(target_feature).reshape(1, -1))[0]
    if x.shape != (len(ids), len(t)) or len(hashes) != len(ids):
        raise ValueError("Feature/identity shapes disagree")
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("Candidate IDs must be unique nonempty strings")
    if any(not isinstance(h, str) or not h for h in hashes) or not target_hash:
        raise ValueError("Image hashes must be present")
    global_scores = x @ t
    banned = set(excluded_hashes) | {target_hash}
    # Canonical global rank chooses the best representative of byte duplicates.
    order = sorted(range(len(ids)), key=lambda i: (-float(global_scores[i]), ids[i]))
    pool, seen = [], set(banned)
    for i in order:
        if hashes[i] not in seen:
            seen.add(hashes[i])
            pool.append(i)
    pool = pool[:top_l]
    if len(pool) < 2:
        raise ValueError("Fewer than two distinct eligible references; no primary padding")
    semantic = np.zeros(len(ids), dtype=np.float64)
    if concept_features is not None:
        concepts = normalized(concept_features)
        if concepts.shape[1] != x.shape[1]:
            raise ValueError("Concept/image dimensions disagree")
        semantic = (x @ concepts.T).mean(axis=1)
    elif method == "concept" and semantic_weight > 0:
        raise ValueError("Concept reranking requires concept features")
    if method == "random":
        chosen = random.Random(seed).sample(pool, 2)
    else:
        chosen = []
        for _ in range(2):
            def score(i):
                value = float(global_scores[i])
                if method == "concept":
                    value += semantic_weight * float(semantic[i])
                if method in ("concept", "diversity") and chosen:
                    value -= diversity_weight * float(max(x[i] @ x[j] for j in chosen))
                return value
            chosen.append(min((i for i in pool if i not in chosen), key=lambda i: (-score(i), ids[i])))
    return {
        "method": method,
        "selected_ids": [ids[i] for i in chosen],
        "selected_hashes": [hashes[i] for i in chosen],
        "candidate_ids": [ids[i] for i in pool],
        "candidate_hashes": [hashes[i] for i in pool],
        "global_scores": [float(global_scores[i]) for i in pool],
        "semantic_scores": [float(semantic[i]) for i in pool],
        "selected_indices": chosen,
    }
