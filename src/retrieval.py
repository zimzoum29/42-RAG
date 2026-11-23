from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from .cache import Postings, QueryCache, index_fingerprint, load_index
from .indexing import tokenize
from .models import (MinimalSearchResults, MinimalSource, RagDataset,
                     StudentSearchResults)

if TYPE_CHECKING:
    from .semantic import SemanticIndex

METHODS = ("bm25", "semantic")


class Retriever:
    """Search over the index built by the index command.

    BM25 is always available; the semantic method needs the embeddings
    built by index --semantic, loaded on first use.

    Attributes:
        index_dir: Directory of the index.
        chunks: Locations of the chunks, by chunk id.
        k1: BM25 saturation of the term frequency.
        b: BM25 strength of the chunk length normalisation.
        avgdl: Average chunk length, in tokens.
        doc_lengths: Length of each chunk, by chunk id.
        idf: Inverse document frequency of each token.
        postings: Compact posting lists of each token.
        semantic: Embedding index, or None until first needed.
        cache: Cache of query results, or None if caching is off.
        cache_hits: Number of searches answered by the cache.
    """

    def __init__(self, index_dir: str = "data/processed",
                 use_cache: bool = True) -> None:
        """Load the index and, if allowed, the query cache.

        Args:
            index_dir: Directory of the index.
            use_cache: Whether to use the index and query caches.

        Raises:
            FileNotFoundError: If the index has not been built yet.
            ValueError: If the two index files do not match.
        """
        self.index_dir = index_dir
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
        self.semantic: SemanticIndex | None = None
        self.cache: QueryCache | None = None
        if use_cache:
            self.cache = QueryCache(directory, index_fingerprint(directory))
        self.cache_hits = 0

    def search(self, question: str, k: int = 5,
               method: str = "bm25") -> list[MinimalSource]:
        """Return the k most relevant chunks for a question.

        Results already in the query cache are returned directly;
        others are ranked with the chosen method and then cached.

        Args:
            question: Text of the question.
            k: Number of results to return.
            method: "bm25" or "semantic".

        Returns:
            Up to k sources, best first.

        Raises:
            ValueError: If method is unknown, or the embeddings do not
                match the index.
            FileNotFoundError: If method is "semantic" and the
                embeddings were not built.
            ImportError: If method is "semantic" and
                sentence-transformers is not installed.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {', '.join(METHODS)}"
                             f", got {method!r}")
        if self.cache is not None:
            cached = self.cache.get(question, k, method)
            if cached is not None:
                self.cache_hits += 1
                return cached
        if method == "semantic":
            ids = self.semantic_index().rank(question, k)
        else:
            ids = self.bm25_ids(question, k)
        sources = [self.chunks[chunk_id] for chunk_id in ids]
        if self.cache is not None:
            self.cache.put(question, k, method, sources)
        return sources

    def semantic_index(self) -> "SemanticIndex":
        """Return the embedding index, loading it on first use.

        Returns:
            The embedding index of the chunks.
        """
        if self.semantic is None:
            from .semantic import SemanticIndex
            self.semantic = SemanticIndex(self.index_dir, len(self.chunks))
        return self.semantic

    def bm25_ids(self, question: str, n: int) -> list[int]:
        """Rank the chunks by BM25 score for a question.

        Only the chunks that contain at least one token of the question
        are scored, thanks to the inverted index.

        Args:
            question: Text of the question.
            n: Number of chunk ids to return.

        Returns:
            Up to n chunk ids, by decreasing score.
        """
        scores: dict[int, float] = {}
        for token in dict.fromkeys(tokenize(question)):
            idf = self.idf.get(token)
            if idf is None:
                continue
            ids, tfs = self.postings[token]
            for chunk_id, tf in zip(ids.tolist(), tfs.tolist()):
                length_ratio = self.doc_lengths[chunk_id] / self.avgdl
                norm = self.k1 * (1 - self.b + self.b * length_ratio)
                scores[chunk_id] = (scores.get(chunk_id, 0.0)
                                    + idf * tf * (self.k1 + 1) / (tf + norm))
        return sorted(scores, key=lambda cid: scores[cid], reverse=True)[:n]

    def save_cache(self) -> None:
        """Write the query cache to disk, ignoring write errors."""
        if self.cache is not None:
            try:
                self.cache.save()
            except OSError:
                pass


def search_dataset(retriever: Retriever, dataset_path: str, k: int,
                   save_directory: str, method: str = "bm25") -> Path:
    """Search every question of a dataset and save the results.

    The output is a StudentSearchResults JSON file with the same name
    as the dataset, written in save_directory.

    Args:
        retriever: Loaded retriever, reused for every question.
        dataset_path: Path of the RagDataset JSON file.
        k: Number of sources to retrieve per question.
        save_directory: Directory of the output, created if needed.
        method: Retrieval method, "bm25" or "semantic".

    Returns:
        Path of the written file.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is not valid JSON or not a RagDataset.
    """
    with open(dataset_path, encoding="utf-8") as f:
        dataset = RagDataset.model_validate_json(f.read())
    results: list[MinimalSearchResults] = []
    for question in tqdm(dataset.rag_questions, desc="Searching"):
        results.append(MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=retriever.search(question.question, k,
                                               method),
        ))
    output = StudentSearchResults(search_results=results, k=k)
    out_dir = Path(save_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(dataset_path).name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))
    retriever.save_cache()
    return out_path