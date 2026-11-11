import json
import pickle
from array import array
from pathlib import Path
from typing import Any

from .models import MinimalSource

INDEX_FILES = ("chunks.json", "bm25_index.json")
INDEX_CACHE = "index_cache.pkl"
QUERY_CACHE = "query_cache.json"

Postings = dict[str, tuple["array[int]", "array[int]"]]


def index_fingerprint(directory: Path) -> str:
    parts: list[str] = []
    for name in INDEX_FILES:
        stat = (directory / name).stat()
        parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def compact_postings(postings: dict[str, list[list[int]]]) -> Postings:
    return {token: (array("i", [chunk_id for chunk_id, _ in posting]),
                    array("i", [tf for _, tf in posting]))
            for token, posting in postings.items()}


def read_json_index(directory: Path) -> tuple[list[MinimalSource], dict[str, Any]]:
    with open(directory / "chunks.json", encoding="utf-8") as f:
        chunks = [MinimalSource.model_validate(c) for c in json.load(f)]
    with open(directory / "bm25_index.json", encoding="utf-8") as f:
        index: dict[str, Any] = json.load(f)
    index["postings"] = compact_postings(index["postings"])
    return chunks, index


def load_index(directory: Path, use_cache: bool = True) -> tuple[list[MinimalSource], dict[str, Any]]:
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

    def __init__(self, directory: Path, fingerprint: str) -> None:
        self.path = directory / QUERY_CACHE
        self.fingerprint = fingerprint
        self.entries: dict[str, list[dict[str, Any]]] = {}
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
        return f"{k}\t{question.strip()}"

    def get(self, question: str, k: int) -> list[MinimalSource] | None:
        entry = self.entries.get(self.key(question, k))
        if entry is None:
            return None
        return [MinimalSource.model_validate(s) for s in entry]

    def put(self, question: str, k: int, sources: list[MinimalSource]) -> None:
        self.entries[self.key(question, k)] = [s.model_dump()
                                               for s in sources]
        self.changed = True

    def save(self) -> None:
        if not self.changed:
            return
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": self.fingerprint,
                       "entries": self.entries}, f)
        self.changed = False