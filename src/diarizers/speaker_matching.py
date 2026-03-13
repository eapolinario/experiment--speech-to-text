"""Embedding-based speaker label reconciliation across audio chunks."""

import numpy as np


def match_speakers(
    chunk_embeddings: dict[str, np.ndarray],
    global_registry: list[tuple[str, np.ndarray]],
    threshold: float = 0.75,
) -> dict[str, str]:
    """Map local chunk speaker labels to globally consistent labels.

    Compares each chunk speaker's embedding against the global registry via
    cosine similarity. Reuses an existing label when similarity exceeds
    threshold; otherwise mints a new one.

    Args:
        chunk_embeddings: mapping of local label -> embedding for this chunk.
        global_registry: mutable list of (global_label, embedding) pairs,
            updated in-place as new speakers are discovered.
        threshold: cosine similarity above which two embeddings are the same speaker.

    Returns:
        Mapping of local label -> global label.
    """
    mapping: dict[str, str] = {}
    for local_label, emb in chunk_embeddings.items():
        best_score, best_label = max(
            (_cosine_similarity(emb, g_emb), g_label) for g_label, g_emb in global_registry
        ) if global_registry else (0.0, None)

        if best_score >= threshold and best_label is not None:
            mapping[local_label] = best_label
        else:
            new_label = f"SPEAKER_{len(global_registry):02d}"
            global_registry.append((new_label, emb))
            mapping[local_label] = new_label
    return mapping


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
