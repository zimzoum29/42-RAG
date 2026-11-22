import threading
import uuid

from fastapi import FastAPI, HTTPException, Query

from .generation import Generator
from .models import MinimalAnswer, MinimalSearchResults
from .retrieval import Retriever


def create_app(index_dir: str = "data/processed") -> FastAPI:
    """Build the HTTP API around a retriever loaded once.

    The language model is loaded on the first call to /answer only, so
    the server starts fast and /search never waits for it.

    Args:
        index_dir: Directory of the index.

    Returns:
        The FastAPI application.

    Raises:
        FileNotFoundError: If the index has not been built yet.
        ValueError: If the two index files do not match.
    """
    retriever = Retriever(index_dir)
    generators: list[Generator] = []
    lock = threading.Lock()
    app = FastAPI(title="RAG against the machine")

    def check_query(q: str) -> str:
        """Reject a query made only of whitespace."""
        if not q.strip():
            raise HTTPException(status_code=400,
                                detail="query must not be empty")
        return q

    @app.get("/health")
    def health() -> dict[str, str]:
        """Tell that the server is running."""
        return {"status": "ok"}

    @app.get("/search", response_model=MinimalSearchResults)
    def search(q: str = Query(..., description="The question"),
               k: int = Query(5, gt=0, description="Number of sources")
               ) -> MinimalSearchResults:
        """Return the top-k sources for a question."""
        question = check_query(q)
        with lock:
            sources = retriever.search(question, k)
            retriever.save_cache()
        return MinimalSearchResults(question_id=str(uuid.uuid4()),
                                    question=question,
                                    retrieved_sources=sources)

    @app.get("/answer", response_model=MinimalAnswer)
    def answer(q: str = Query(..., description="The question"),
               k: int = Query(5, gt=0, description="Number of sources")
               ) -> MinimalAnswer:
        """Answer a question from its top-k sources."""
        question = check_query(q)
        with lock:
            sources = retriever.search(question, k)
            retriever.save_cache()
            try:
                if not generators:
                    generators.append(Generator())
                text = generators[0].answer(question, sources)
            except (ImportError, OSError, RuntimeError) as error:
                raise HTTPException(
                    status_code=503,
                    detail=f"answer generation failed: {error}") from error
        return MinimalAnswer(question_id=str(uuid.uuid4()),
                             question=question,
                             retrieved_sources=sources, answer=text)

    return app