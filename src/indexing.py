import re
import json
import math
from pathlib import Path

from .chunking import chunk_corpus
from .models import Chunk

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
CAMEL_RE = re.compile(r"(?<=.)(?=[A-Z][a-z]{2,})")
STOPWORDS = frozenset("""
    a an and are as at be by can do does for from how if in into is it its
    of on or so that the this to was what when where which who why will
    with you
""".split())


def tokenize(text: str) -> list[str]:

    tokens: list[str] = []
    for word in WORD_RE.findall(text):
        parts = []
        for piece in word.split("_"):
            for part in CAMEL_RE.split(piece):
                if part:
                    parts.append(part.lower())

        for token in dict.fromkeys([word.lower(), *parts]):

            if len(token) > 1 and token not in STOPWORDS:
                tokens.append(token)

    return tokens


def build_bm25(tokenized: list[list[str]], k1: float = 1.2, b: float = 0.5) -> dict:
    postings: dict[str, list[list[int]]] = {}
    doc_lengths: list[int] = []
    for chunk_id, tokens in enumerate(tokenized):
        doc_lengths.append(len(tokens))
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, tf in counts.items():
            postings.setdefault(token, []).append([chunk_id, tf])

    n_chunks = len(tokenized)
    avgdl = sum(doc_lengths) / n_chunks if n_chunks else 0.0
    idf = {}
    for token, posting in postings.items():
        idf[token] = math.log(1 + (n_chunks - len(posting) + 0.5) / (len(posting) + 0.5))

    return {
        "k1": k1,
        "b": b,
        "avgdl": avgdl,
        "doc_lengths": doc_lengths,
        "idf": idf,
        "postings": postings,
    }

def save_index(out_dir: str, chunks: list[Chunk], index: dict) -> None:

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "chunks.json", "w", encoding="utf-8") as f:
        json.dump([chunk.model_dump() for chunk in chunks], f)

    with open(out / "bm25_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)


def build_index(raw_dir: str = "data/raw", out_dir: str = "data/processed", max_chunk_size: int = 2000) -> int:
    
    chunks = chunk_corpus(raw_dir, max_chunk_size)
    tokenized = [tokenize(chunk.text) + tokenize(chunk.file_path)
                 for chunk in chunks]
    index = build_bm25(tokenized)
    save_index(out_dir, chunks, index)
    return len(chunks)