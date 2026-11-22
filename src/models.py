import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Location of a passage in the corpus.

    The passage is content[first_character_index:last_character_index]
    of the file at file_path; the end index is exclusive.

    Attributes:
        file_path: Path relative to the project root, exactly as in the
            corpus (e.g. "data/raw/vllm-0.10.1/README.md").
        first_character_index: Offset of the first character.
        last_character_index: Offset just after the last character.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question to answer, identified by a unique id.

    Attributes:
        question_id: Unique identifier, generated if not provided.
        question: Text of the question.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference answer and sources.

    Attributes:
        sources: Locations of the passages that contain the answer.
        answer: Reference answer.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, with or without reference answers.

    Attributes:
        rag_questions: The questions of the dataset.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Sources retrieved for one question.

    Attributes:
        question_id: Identifier of the question.
        question: Text of the question.
        retrieved_sources: Retrieved sources, best first.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Sources retrieved for one question, with the generated answer.

    Attributes:
        answer: Answer generated from the retrieved sources.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Output of search_dataset: the search results of a dataset.

    Attributes:
        search_results: Results for each question of the dataset.
        k: Number of sources requested per question.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of answer_dataset: search results with generated answers.

    Attributes:
        search_results: Results and answer for each question.
        k: Number of sources requested per question.
    """

    search_results: List[MinimalAnswer]
    k: int