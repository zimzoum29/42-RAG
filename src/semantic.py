from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .models import MinimalSource

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_FILE = "embeddings.npy"
BATCH_SIZE = 64


def load_encoder(model_name: str = EMBEDDING_MODEL) -> Any:
    """Load the sentence embedding model on the CPU.

    Args:
        model_name: Hugging Face name of the model.

    Returns:
        The SentenceTransformer model.

    Raises:
        ImportError: If sentence-transformers is not installed.
        OSError: If the model cannot be downloaded or loaded.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def chunk_texts(chunks: list[MinimalSource]) -> list[str]:
    """Return the text to embed for each chunk: its path, then its text.

    Args:
        chunks: Locations of the chunks, in chunk id order.

    Returns:
        One text per chunk, in the same order.
    """
    texts: list[str] = []
    path, content = "", ""
    for chunk in chunks:
        if chunk.file_path != path:
            path = chunk.file_path
            try:
                with open(path, encoding="utf-8", errors="replace",
                          newline="") as f:
                    content = f.read()
            except OSError:
                content = ""
        text = content[chunk.first_character_index:
                       chunk.last_character_index]
        texts.append(f"{chunk.file_path}\n{text}")
    return texts


def build_embeddings(chunks: list[MinimalSource], out_dir: str) -> None:
    """Embed every chunk and save the vectors in out_dir.

    Row i of the saved matrix is the normalised vector of chunk i, so
    the matrix is aligned with chunks.json.

    Args:
        chunks: Locations of the chunks, in chunk id order.
        out_dir: Directory of the index.

    Raises:
        ImportError: If sentence-transformers is not installed.
        OSError: If the model cannot be loaded or the file written.
    """
    encoder = load_encoder()
    texts = chunk_texts(chunks)
    batches: list[npt.NDArray[np.float32]] = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        vectors = encoder.encode(texts[start:start + BATCH_SIZE],
                                 normalize_embeddings=True,
                                 convert_to_numpy=True)
        batches.append(np.asarray(vectors, dtype=np.float32))
    matrix = np.concatenate(batches) if batches else np.zeros((0, 0))
    np.save(Path(out_dir) / EMBEDDINGS_FILE, matrix.astype(np.float32))


class SemanticIndex:
    """Nearest-neighbour search over the chunk embeddings.

    Attributes:
        embeddings: Normalised vector of each chunk, by chunk id.
        encoder: Model used to embed the questions.
    """

    def __init__(self, index_dir: str, n_chunks: int) -> None:
        """Load the chunk embeddings and the embedding model.

        Args:
            index_dir: Directory of the index.
            n_chunks: Number of chunks in the lexical index.

        Raises:
            FileNotFoundError: If the embeddings were not built.
            ValueError: If they do not match the lexical index.
            ImportError: If sentence-transformers is not installed.
        """
        path = Path(index_dir) / EMBEDDINGS_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"no embeddings in {index_dir}/, run index --semantic")
        self.embeddings: npt.NDArray[np.float32] = np.load(path)
        if self.embeddings.shape[0] != n_chunks:
            raise ValueError("embeddings do not match the index, "
                             "run index --semantic again")
        self.encoder = load_encoder()

    def rank(self, question: str, n: int) -> list[int]:
        """Return the ids of the n chunks closest in meaning to question.

        Args:
            question: Text of the question.
            n: Number of chunk ids to return.

        Returns:
            Chunk ids, by decreasing cosine similarity.
        """
        if n <= 0 or len(self.embeddings) == 0:
            return []
        query = np.asarray(self.encoder.encode(
            [question], normalize_embeddings=True,
            convert_to_numpy=True)[0], dtype=np.float32)
        scores = self.embeddings @ query
        n = min(n, len(scores))
        best = np.argpartition(-scores, n - 1)[:n]
        return [int(i) for i in best[np.argsort(-scores[best])]]