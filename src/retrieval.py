from pathlib import Path

from tqdm import tqdm

from .cache import Postings, QueryCache, index_fingerprint, load_index
from .indexing import tokenize
from .models import (MinimalSearchResults, MinimalSource, RagDataset, StudentSearchResults)


class Retriever:

    def __init__(self, index_dir: str = "data/processed", use_cache: bool = True) -> None:
        directory = Path(index_dir)
        self.chunks, index = load_index(directory, use_cache)
        self.k1: float = index["k1"]
        self.b: float = index["b"]
        self.avgdl: float = index["avgdl"]
        self.doc_lengths: list[int] = index["doc_lengths"]
        self.idf: dict[str, float] = index["idf"]
        self.postings: Postings = index["postings"]
        if len(self.chunks) != len(self.doc_lengths):
            raise ValueError("chunks.json and bm25_index.json do not match")
        self.cache: QueryCache | None = None
        if use_cache:
            self.cache = QueryCache(directory, index_fingerprint(directory))
        self.cache_hits = 0

    def search(self, question: str, k: int = 5) -> list[MinimalSource]:
        if self.cache is not None:
            cached = self.cache.get(question, k)
            if cached is not None:
                self.cache_hits += 1
                return cached
        sources = self.rank(question, k)
        if self.cache is not None:
            self.cache.put(question, k, sources)
        return sources

    def rank(self, question: str, k: int) -> list[MinimalSource]:
        scores: dict[int, float] = {}
        for token in dict.fromkeys(tokenize(question)):
            idf = self.idf.get(token)
            if idf is None:
                continue
            ids, tfs = self.postings[token]
            for chunk_id, tf in zip(ids.tolist(), tfs.tolist()):
                length_ratio = self.doc_lengths[chunk_id] / self.avgdl
                norm = self.k1 * (1 - self.b + self.b * length_ratio)
                scores[chunk_id] = (scores.get(chunk_id, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm))
        best = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
        return [self.chunks[cid] for cid in best]

    def save_cache(self) -> None:
        if self.cache is not None:
            try:
                self.cache.save()
            except OSError:
                pass


def search_dataset(retriever: Retriever, dataset_path: str, k: int, save_directory: str) -> Path:
    with open(dataset_path, encoding="utf-8") as f:
        dataset = RagDataset.model_validate_json(f.read())
    results: list[MinimalSearchResults] = []
    for question in tqdm(dataset.rag_questions, desc="Searching"):
        results.append(MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=retriever.search(question.question, k),
        ))
    output = StudentSearchResults(search_results=results, k=k)
    out_dir = Path(save_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(dataset_path).name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))
    retriever.save_cache()
    return out_path