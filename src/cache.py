import json
import pickle
from array import array
from pathlib import Path
from typing import TypedDict
from indexing import BM25Index
from .models import MinimalSource

INDEX_FILES = ("chunks.json", "bm25_index.json")
INDEX_CACHE = "index_cache.pkl"
QUERY_CACHE = "query_cache.json"

Postings = dict[str, tuple["array[int]", "array[int]"]]

class CompactIndex(TypedDict):
    """BM25 index held in memory, with posting lists stored as arrays.

    Attributes:
        k1: Saturation of the term frequency.
        b: Strength of the chunk length normalisation.
        avgdl: Average chunk length, in tokens.
        doc_lengths: Length of each chunk, by chunk id.
        idf: Inverse document frequency of each token.
        postings: For each token, the chunk ids and the term frequencies
            of the chunks that contain it, as two int arrays.
    """

    k1: float
    b: float
    avgdl: float
    doc_lengths: list[int]
    idf: dict[str, float]
    postings: Postings


def index_fingerprint(directory: Path) -> str:
    """Identify the current index by the size and mtime of its files.

    Any new run of the index command rewrites both files, which changes
    the fingerprint and invalidates every cache built on the old index.

    Args:
        directory: Directory of the index.

    Returns:
        A string that changes whenever an index file changes.

    Raises:
        FileNotFoundError: If the index has not been built yet.
    """

    parts: list[str] = []
    for name in INDEX_FILES:
        stat = (directory / name).stat()
        parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def compact_postings(postings: dict[str, list[list[int]]]) -> Postings:
    """Store each posting list as two int arrays.

    One array holds the chunk ids and the other the term frequencies.
    A few arrays per token load much faster than millions of small
    Python lists.

    Args:
        postings: Posting lists as saved in bm25_index.json.

    Returns:
        The same posting lists, as (chunk_ids, frequencies) arrays.
    """
    return {token: (array("i", [chunk_id for chunk_id, _ in posting]),
                    array("i", [tf for _, tf in posting]))
            for token, posting in postings.items()}


def read_json_index(directory: Path
                    ) -> tuple[list[MinimalSource], CompactIndex]:
    with open(directory / "chunks.json", encoding="utf-8") as f:
        chunks = [MinimalSource.model_validate(c) for c in json.load(f)]
    with open(directory / "bm25_index.json", encoding="utf-8") as f:
        index: BM25Index = json.load(f)
    compact: CompactIndex = {
        "k1": index["k1"],
        "b": index["b"],
        "avgdl": index["avgdl"],
        "doc_lengths": index["doc_lengths"],
        "idf": index["idf"],
        "postings": compact_postings(index["postings"]),
    }
    return chunks, compact


def load_index(directory: Path, use_cache: bool = True
               ) -> tuple[list[MinimalSource], CompactIndex]:
    """Load the index, through a binary cache when allowed.

    The JSON files are read only when the cache is missing, unreadable
    or built from an older index; the result is then saved to the cache
    for the next run.

    Args:
        directory: Directory of the index.
        use_cache: Whether to read and write index_cache.pkl.

    Returns:
        The chunks and the BM25 index, with compact posting lists.

    Raises:
        FileNotFoundError: If the index has not been built yet.
    """

    fingerprint = index_fingerprint(directory)
    cache_path = directory / INDEX_CACHE
    if use_cache:
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached["fingerprint"] == fingerprint:
                chunks = [MinimalSource.model_construct(
                              file_path=path, first_character_index=start,
                              last_character_index=end)
                          for path, start, end in cached["chunks"]]
                return chunks, cached["index"]
        except Exception:
            pass
    chunks, index = read_json_index(directory)
    if use_cache:
        rows = [(c.file_path, c.first_character_index,
                 c.last_character_index) for c in chunks]
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"fingerprint": fingerprint, "chunks": rows,
                             "index": index}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            pass
    return chunks, index


class QueryCache:
    """Search results saved on disk, valid for one version of the index.

    Attributes:
        path: Path of the cache file.
        fingerprint: Fingerprint of the index the results come from.
        entries: Cached results, by query key.
        changed: Whether entries changed since the cache was loaded.
    """

    def __init__(self, directory: Path, fingerprint: str) -> None:
        """Load the cached results that match the current index.

        A missing or unreadable cache file, or one built from another
        index, gives an empty cache.

        Args:
            directory: Directory of the index, where the cache lives.
            fingerprint: Fingerprint of the current index.
        """

        self.path = directory / QUERY_CACHE
        self.fingerprint = fingerprint
        self.entries: dict[str, list[dict[str, str | int]]] = {}
        self.changed = False
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("fingerprint") == fingerprint:
                self.entries = dict(data["entries"])
        except Exception:
            pass

    @staticmethod
    def key(question: str, k: int) -> str:
        """Build the cache key of a query.

        Args:
            question: Text of the question.
            k: Number of results requested.

        Returns:
            A key made of k and the stripped question.
        """

        return f"{k}\t{question.strip()}"

    def get(self, question: str, k: int) -> list[MinimalSource] | None:
        """Return the cached results of a query.

        Args:
            question: Text of the question.
            k: Number of results requested.

        Returns:
            The cached sources, or None if the query is not cached.
        """

        entry = self.entries.get(self.key(question, k))
        if entry is None:
            return None
        return [MinimalSource.model_validate(s) for s in entry]

    def put(self, question: str, k: int,
            sources: list[MinimalSource]) -> None:
        """Store the results of a query in memory.

        Args:
            question: Text of the question.
            k: Number of results requested.
            sources: Sources found for the question.
        """

        self.entries[self.key(question, k)] = [s.model_dump()
                                               for s in sources]
        self.changed = True

    def save(self) -> None:
        """Write the cache to disk, only if it changed.

        Raises:
            OSError: If the cache file cannot be written.
        """
        
        if not self.changed:
            return
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": self.fingerprint,
                       "entries": self.entries}, f)
        self.changed = False