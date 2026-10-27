import json
from pathlib import Path
from typing import Any
from tqdm import tqdm

from .models import (MinimalSearchResults, MinimalSource, RagDataset, StudentSearchResults)
from .indexing import tokenize


class Retriever: 
    def __init__(self, index_dir: str = "data/processed") -> None:

        directory = Path(index_dir)
        with open(directory / "chunks.json", encoding="utf-8") as f:
            self.chunks = [MinimalSource.model_validate(c) for c in json.load(f)]
        with open(directory / "bm25_index.json", encoding="utf-8") as f:
            index: dict[str, Any] = json.load(f)
        self.k1: float = index["k1"]
        self.b: float = index["b"]
        self.avgdl: float = index["avgdl"]
        self.doc_lengths: list[int] = index["doc_lengths"]
        self.idf: dict[str, float] = index["idf"]
        self.postings: dict[str, list[list[int]]] = index["postings"]
        if len(self.chunks) != len(self.doc_lengths):
            raise ValueError("chunks.json and bm25_index.json do not match")

    def search(self, question: str, k: int = 5) -> list[MinimalSource]:
        
        scores: dict[int, float] = {}
        for token in dict.fromkeys(tokenize(question)):
            idf = self.idf.get(token)
            if idf is None:
                continue
            for chunk_id, tf in self.postings[token]:
                length_ratio = self.doc_lengths[chunk_id] / self.avgdl
                norm = self.k1 * (1 - self.b + self.b * length_ratio)
                scores[chunk_id] = (scores.get(chunk_id, 0.0)
                                    + idf * tf * (self.k1 + 1) / (tf + norm))
        best = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
        return [self.chunks[cid] for cid in best]

def search_dataset(retriever: Retriever, dataset_path: str, k: int, save_directory: str) -> Path:

    with open(dataset_path, encoding="utf-8") as f:
        dataset = RagDataset.model_validate_json(f.read())
    results = []
    for question in tqdm(dataset.rag_questions, desc="Searching"):
        results.append(MinimalSearchResults(question_id=question.question_id, question=question.question, retrieved_sources=retriever.search(question.question, k)))
        
    output = StudentSearchResults(search_results=results, k=k)

    out_dir = Path(save_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(dataset_path).name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))
    return out_path