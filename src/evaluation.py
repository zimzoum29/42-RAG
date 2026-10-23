from .models import (AnsweredQuestion, MinimalSource, RagDataset, StudentSearchResults)

IOU_THRESHOLD = 0.05


def iou(a: MinimalSource, b: MinimalSource) -> float:
    if a.file_path != b.file_path:
        return 0.0
    inter = (min(a.last_character_index, b.last_character_index) - max(a.first_character_index, b.first_character_index))
    if inter <= 0:
        return 0.0
    union = (max(a.last_character_index, b.last_character_index) - min(a.first_character_index, b.first_character_index))
    return inter / union


def recall_at_k(results: StudentSearchResults, dataset: RagDataset, ks: list[int]) -> dict[int, float]:
    expected = {q.question_id: q.sources for q in dataset.rag_questions if isinstance(q, AnsweredQuestion) and q.sources}
    if not expected:
        raise ValueError("the dataset contains no answered question")
    retrieved = {r.question_id: r.retrieved_sources
                 for r in results.search_results}
    recalls: dict[int, float] = {}
    for k in ks:
        total = 0.0
        for question_id, sources in expected.items():
            top_k = retrieved.get(question_id, [])[:k]
            found = sum(any(iou(source, result) >= IOU_THRESHOLD for result in top_k) for source in sources)
            total += found / len(sources)
        recalls[k] = total / len(expected)
    return recalls