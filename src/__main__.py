import fire

from .indexing import build_index
from .retrieval import Retriever, search_dataset


def check_k(k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k


class RagCLI:

    def index(self, max_chunk_size: int = 2000, raw_dir: str = "data/raw", out_dir: str = "data/processed") -> None:
        try:
            if isinstance(max_chunk_size, bool) or not isinstance(max_chunk_size, int):
                raise ValueError(f"max_chunk_size must be an integer, got {max_chunk_size!r}")
            n_chunks = build_index(raw_dir, out_dir, max_chunk_size)
        except (OSError, ValueError) as error:
            print(f"Error: {error}")
            return
        print(f"Ingestion complete! Indexed {n_chunks} chunks "
              f"under {out_dir}/")

    def search(self, query: str, k: int = 5,
               index_dir: str = "data/processed") -> None:
        try:
            k = check_k(k)
            query = str(query)
            if not query.strip():
                raise ValueError("query must not be empty")
            retriever = Retriever(index_dir)
            sources = retriever.search(query, k)
        except FileNotFoundError:
            print(f"Error: no index found in {index_dir}/, run the index command first")
            return
        except (OSError, ValueError) as error:
            print(f"Error: {error}")
            return
        if not sources:
            print("No relevant source found.")
            return
        for rank, source in enumerate(sources, start=1):
            print(f"{rank}. {source.file_path} "
                  f"[{source.first_character_index}:"
                  f"{source.last_character_index}]")

    def search_dataset(self, dataset_path: str, k: int = 10, save_directory: str = "data/output/search_results", index_dir: str = "data/processed") -> None:
        try:
            k = check_k(k)
            retriever = Retriever(index_dir)
        except FileNotFoundError:
            print(f"Error: no index found in {index_dir}/, "
                  "run the index command first")
            return
        except (OSError, ValueError) as error:
            print(f"Error: {error}")
            return
        try:
            out_path = search_dataset(retriever, str(dataset_path), k, str(save_directory))
        except FileNotFoundError:
            print(f"Error: dataset not found: {dataset_path}")
            return
        except (OSError, ValueError) as error:
            print(f"Error: invalid dataset {dataset_path}: {error}")
            return
        print(f"Saved student_search_results to {out_path}")


if __name__ == "__main__":
    fire.Fire(RagCLI)