import re
import json
import math
from pathlib import Path
from typing import TypedDict
from .chunking import chunk_corpus, read_file
from .models import MinimalSource

class BM25Index(TypedDict):
    """BM25 statistics of the chunks, as saved in bm25_index.json.

    Attributes:
        k1: Saturation of the term frequency.
        b: Strength of the chunk length normalisation.
        avgdl: Average chunk length, in tokens.
        doc_lengths: Length of each chunk, by chunk id.
        idf: Inverse document frequency of each token.
        postings: For each token, the [chunk_id, term_frequency] pairs
            of the chunks that contain it.
    """

    k1: float
    b: float
    avgdl: float
    doc_lengths: list[int]
    idf: dict[str, float]
    postings: dict[str, list[list[int]]]

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
CAMEL_RE = re.compile(r"(?<=.)(?=[A-Z][a-z]{2,})")
STOPWORDS = frozenset("""
    a an and are as at be by can do does for from how if in into is it its
    of on or so that the this to was what when where which who why will
    with you
""".split())


def tokenize(text: str) -> list[str]:
    """Split a text into lowercase search tokens.

    Each identifier is kept whole and also split on underscores and
    camelCase, so "enable_lora" gives "enable_lora", "enable" and
    "lora". Stopwords and one-character tokens are dropped. The same
    function tokenizes the chunks and the questions.

    Args:
        text: Text of a chunk, a file path or a question.

    Returns:
        The tokens, in order of appearance.
    """

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


def build_bm25(tokenized: list[list[str]], k1: float = 1.2,
               b: float = 0.5) -> BM25Index:
    """Compute the BM25 statistics of the tokenized chunks.

    Args:
        tokenized: Tokens of each chunk; the list index is the chunk id.
        k1: Saturation of the term frequency.
        b: Strength of the chunk length normalisation.

    Returns:
        A dict with the parameters ("k1", "b"), the average chunk
        length ("avgdl"), the length of each chunk ("doc_lengths"), the
        IDF of each token ("idf") and the inverted index ("postings"):
        for each token, the [chunk_id, term_frequency] pairs of the
        chunks that contain it.
    """

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
    idf: dict[str, float] = {}
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

def save_index(out_dir: str, chunks: list[MinimalSource],
               index: BM25Index) -> None:
    """Write the chunks and the BM25 index as JSON files in out_dir.

    The position of a chunk in chunks.json is its id in the index, so
    both files are always written together.

    Args:
        out_dir: Directory of the index, created if needed.
        chunks: Locations of the chunks, in chunk id order.
        index: BM25 statistics returned by build_bm25.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "chunks.json", "w", encoding="utf-8") as f:
        json.dump([chunk.model_dump() for chunk in chunks], f)

    with open(out / "bm25_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)


def build_index(raw_dir: str = "data/raw", out_dir: str = "data/processed",
                max_chunk_size: int = 2000, semantic: bool = False) -> int:
    """Chunk the corpus, build the BM25 index and save it in out_dir.

    The text of each chunk is read back from its file and tokenized,
    together with the file path, before computing the BM25 statistics.
    With semantic, the chunk embeddings are also computed; otherwise any
    old embeddings are removed, since they no longer match the chunks.

    Args:
        raw_dir: Root directory of the corpus.
        out_dir: Directory where the index is written.
        max_chunk_size: Maximum length of a chunk, from 100 to 2000.
        semantic: Whether to also build the embedding index.

    Returns:
        Number of chunks indexed.

    Raises:
        ValueError: If max_chunk_size is out of range or no indexable
            file is found.
        FileNotFoundError: If raw_dir is not a directory.
        ImportError: If semantic is set without sentence-transformers.
    """

    chunks = chunk_corpus(raw_dir, max_chunk_size)
    tokenized: list[list[str]] = []
    path, content = "", ""
    for chunk in chunks:
        if chunk.file_path != path:
            path = chunk.file_path
            content = read_file(path) or ""
        text = content[chunk.first_character_index:
                        chunk.last_character_index]
        tokenized.append(tokenize(text) + tokenize(chunk.file_path))
    index = build_bm25(tokenized)
    save_index(out_dir, chunks, index)
    if semantic:
        from .semantic import build_embeddings
        build_embeddings(chunks, out_dir)
    else:
        (Path(out_dir) / "embeddings.npy").unlink(missing_ok=True)
    return len(chunks)