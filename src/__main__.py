import fire
from pydantic import ValidationError
from .evaluation import recall_at_k
from .generation import Generator, answer_dataset
from .indexing import build_index
from .models import RagDataset, StudentSearchResults
from .retrieval import Retriever, search_dataset


def check_k(k: object) -> int:
    """Check that k is a positive integer.

    Fire converts command-line values itself, so k may arrive as a
    string ("abc"), a float (2.5) or True (a bare --k flag).

    Args:
        k: Value received for k.

    Returns:
        k, unchanged.

    Raises:
        ValueError: If k is not an int greater than 0.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k

def describe(error: Exception) -> str:
    """Turn an exception into a short, one-line error message.

    Pydantic validation errors are reduced to their first problem and
    where it is in the file, instead of the full multi-line report.

    Args:
        error: Exception to describe.

    Returns:
        A one-line description of the error.
    """

    if isinstance(error, ValidationError):
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        message = first["msg"]
        if where:
            message += f" (at {where})"
        return message
    return str(error)


class RagCLI:
    """Command-line interface of the RAG system, exposed with Fire.

    Each public method is a command, and each of its parameters an
    option: uv run python -m src <command> [--option value].
    """


    def index(self, max_chunk_size: int = 2000, raw_dir: str = "data/raw",
              out_dir: str = "data/processed") -> None:
        """Chunk the corpus and build the BM25 index.

        Args:
            max_chunk_size: Maximum length of a chunk, from 1 to 2000.
            raw_dir: Root directory of the corpus.
            out_dir: Directory where the index is written.
        """

        try:
            if isinstance(max_chunk_size, bool) or not isinstance(max_chunk_size, int):
                raise ValueError(f"max_chunk_size must be an integer, got {max_chunk_size!r}")
            n_chunks = build_index(raw_dir, out_dir, max_chunk_size)
        except (OSError, ValueError) as error:
            print(f"Error: {describe(error)}")
            return
        print(f"Ingestion complete! Indexed {n_chunks} chunks under {out_dir}/")

    def search(self, query: str, k: int = 5,
               index_dir: str = "data/processed",
               cache: bool = True) -> None:
        """Print the top-k sources for a single query.

        Args:
            query: Text of the question.
            k: Number of sources to return.
            index_dir: Directory of the index.
            cache: Whether to use the index and query caches.
        """

        try:
            k = check_k(k)
            query = str(query)
            if not query.strip():
                raise ValueError("query must not be empty")
            retriever = Retriever(index_dir, use_cache=bool(cache))
            sources = retriever.search(query, k)
            retriever.save_cache()
        except FileNotFoundError:
            print(f"Error: no index found in {index_dir}/, "
                  "run the index command first")
            return
        except (OSError, ValueError) as error:
            print(f"Error: {describe(error)}")
            return
        if retriever.cache_hits:
            print("(results served from cache)")
        if not sources:
            print("No relevant source found.")
            return
        for rank, source in enumerate(sources, start=1):
            print(f"{rank}. {source.file_path} "
                  f"[{source.first_character_index}:"
                  f"{source.last_character_index}]")

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/search_results",
                       index_dir: str = "data/processed",
                       cache: bool = True) -> None:
        """Search every question of a dataset and save the results.

        Args:
            dataset_path: Path of the RagDataset JSON file.
            k: Number of sources to retrieve per question.
            save_directory: Directory of the StudentSearchResults file.
            index_dir: Directory of the index.
            cache: Whether to use the index and query caches.
        """

        try:
            k = check_k(k)
            retriever = Retriever(index_dir, use_cache=bool(cache))
        except FileNotFoundError:
            print(f"Error: no index found in {index_dir}/, run the index command first")
            return
        except (OSError, ValueError) as error:
            print(f"Error: {describe(error)}")
            return
        try:
            out_path = search_dataset(retriever, str(dataset_path), k, str(save_directory))
        except FileNotFoundError:
            print(f"Error: dataset not found: {dataset_path}")
            return
        except (OSError, ValueError) as error:
            print(f"Error: invalid dataset {dataset_path}: {describe(error)}")
            return
        print(f"Saved student_search_results to {out_path}")

    def answer(self, query: str, k: int = 5,
               index_dir: str = "data/processed") -> None:
        """Answer a single query from its top-k sources.

        Args:
            query: Text of the question.
            k: Number of sources given to the model.
            index_dir: Directory of the index.
        """

        try:
            k = check_k(k)
            query = str(query)
            if not query.strip():
                raise ValueError("query must not be empty")
            sources = Retriever(index_dir).search(query, k)
        except FileNotFoundError:
            print(f"Error: no index found in {index_dir}/, run the index command first")
            return
        except (OSError, ValueError) as error:
            print(f"Error: {describe(error)}")
            return
        try:
            answer = Generator().answer(query, sources)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            print(f"Error: answer generation failed: {describe(error)}")
            return
        print(answer)
        print("\nSources:")
        for rank, source in enumerate(sources, start=1):
            print(f"{rank}. {source.file_path} "
                  f"[{source.first_character_index}:"
                  f"{source.last_character_index}]")

    def answer_dataset(
            self, student_search_results_path: str,
            save_directory: str = "data/output/search_results_and_answer"
    ) -> None:
        """Answer every question of a search results file and save them.

        Args:
            student_search_results_path: Path of a file written by
                search_dataset.
            save_directory: Directory of the
                StudentSearchResultsAndAnswer file.
        """

        path = str(student_search_results_path)
        try:
            with open(path, encoding="utf-8") as f:
                StudentSearchResults.model_validate_json(f.read())
        except FileNotFoundError:
            print(f"Error: file not found: {path}")
            return
        except (OSError, ValueError) as error:
            print(f"Error: invalid search results {path}: "
                  f"{describe(error)}")
            return
        try:
            generator = Generator()
            out_path = answer_dataset(generator, path, str(save_directory))
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            print(f"Error: {describe(error)}")
            return
        print(f"Saved student_search_results_and_answer to {out_path}")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Print the recall@1, 3, 5 and 10 of search results.

        Args:
            student_search_results_path: Path of a file written by
                search_dataset.
            dataset_path: Path of the dataset with the reference
                sources (AnsweredQuestions).
        """

        try:
            with open(student_search_results_path, encoding="utf-8") as f:
                results = StudentSearchResults.model_validate_json(f.read())
            with open(dataset_path, encoding="utf-8") as f:
                dataset = RagDataset.model_validate_json(f.read())
            recalls = recall_at_k(results, dataset, [1, 3, 5, 10])
        except (OSError, ValueError) as error:
            print(f"Error: {describe(error)}")
            return
        print("  ".join(f"Recall@{k}: {value:.3f}" for k, value in recalls.items()))

    def serve(self, host: str = "127.0.0.1", port: int = 8000,
              index_dir: str = "data/processed") -> None:
        """Serve /search and /answer over a local HTTP API.

        Args:
            host: Address to listen on; 127.0.0.1 keeps it local.
            port: Port to listen on.
            index_dir: Directory of the index.
        """
        if isinstance(port, bool) or not isinstance(port, int) \
                or not 1 <= port <= 65535:
            print(f"Error: port must be an integer from 1 to 65535, "
                  f"got {port!r}")
            return
        try:
            import uvicorn

            from .api import create_app
            app = create_app(str(index_dir))
        except FileNotFoundError:
            print(f"Error: no index found in {index_dir}/, "
                  "run the index command first")
            return
        except (ImportError, OSError, ValueError) as error:
            print(f"Error: {describe(error)}")
            return
        print(f"Serving on http://{host}:{port} (docs: /docs)")
        uvicorn.run(app, host=str(host), port=port)


if __name__ == "__main__":
    fire.Fire(RagCLI)