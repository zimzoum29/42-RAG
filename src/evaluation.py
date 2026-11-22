from .models import (AnsweredQuestion, MinimalSource, RagDataset, StudentSearchResults)

IOU_THRESHOLD = 0.05


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Compute the intersection over union of two sources.

    Two sources in different files never overlap.

    Args:
        a: First source.
        b: Second source.

    Returns:
        Length of the common range divided by the length of the range
        covered by both, from 0.0 (no overlap) to 1.0 (same range).
    """

    if a.file_path != b.file_path:
        return 0.0
    inter = (min(a.last_character_index, b.last_character_index) - max(a.first_character_index, b.first_character_index))
    if inter <= 0:
        return 0.0
    union = (max(a.last_character_index, b.last_character_index) - min(a.first_character_index, b.first_character_index))
    return inter / union


def recall_at_k(results: StudentSearchResults, dataset: RagDataset,
                ks: list[int]) -> dict[int, float]:
    """Compute the average recall@k of search results.

    For each answered question, recall@k is the share of its reference
    sources matched by one of its top-k results, a match being a result
    in the same file with an IoU of at least 0.05. Questions are matched
    by id; a question missing from the results scores 0.

    Args:
        results: Search results to evaluate.
        dataset: Dataset with the reference sources.
        ks: Values of k to evaluate.

    Returns:
        The average recall for each value of k.

    Raises:
        ValueError: If the dataset contains no answered question.
    """
    
    expected = {q.question_id: q.sources for q in dataset.rag_questions if isinstance(q, AnsweredQuestion) and q.sources}
    if not expected:
        raise ValueError("the dataset contains no answered question")
    retrieved = {r.question_id: r.retrieved_sources for r in results.search_results}
    recalls: dict[int, float] = {}
    for k in ks:
        total = 0.0
        for question_id, sources in expected.items():
            top_k = retrieved.get(question_id, [])[:k]
            found = sum(any(iou(source, result) >= IOU_THRESHOLD for result in top_k) for source in sources)
            total += found / len(sources)
        recalls[k] = total / len(expected)
    return recalls