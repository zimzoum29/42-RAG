from pathlib import Path

from tqdm import tqdm

from .models import (MinimalAnswer, MinimalSource, StudentSearchResults, StudentSearchResultsAndAnswer)

MODEL_NAME = "Qwen/Qwen3-0.6B"
SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase. "
    "Use only the information in the provided context. "
    "If the context does not contain the answer, say that you do not know. "
    "Answer concisely and name the relevant functions, classes or files."
)
NO_CONTEXT_ANSWER = "I could not find relevant information to answer."


def read_source(source: MinimalSource) -> str:

    with open(source.file_path, encoding="utf-8", errors="replace", newline="") as f:
        content = f.read()
    return content[source.first_character_index: source.last_character_index]


class Generator:

    def __init__(self, model_name: str = MODEL_NAME, max_context_tokens: int = 3000, max_new_tokens: int = 256) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto").to(self.device)
        self.model.eval()
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens

    def build_context(self, sources: list[MinimalSource]) -> str:
        blocks: list[str] = []
        used = 0
        for number, source in enumerate(sources, start=1):
            try:
                text = read_source(source)
            except OSError:
                continue
            block = f"[Source {number}] {source.file_path}\n{text}\n"
            size = len(self.tokenizer.encode(block))
            if used + size > self.max_context_tokens:
                break
            blocks.append(block)
            used += size
        return "\n".join(blocks)

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        context = self.build_context(sources)
        if not context:
            return NO_CONTEXT_ANSWER
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\nQuestion: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, repetition_penalty=1.1)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text: str = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip() or NO_CONTEXT_ANSWER


def answer_dataset(generator: Generator, search_results_path: str, save_directory: str) -> Path:
    with open(search_results_path, encoding="utf-8") as f:
        results = StudentSearchResults.model_validate_json(f.read())
    answers: list[MinimalAnswer] = []
    for result in tqdm(results.search_results, desc="Answering"):
        answers.append(MinimalAnswer(
            question_id=result.question_id,
            question=result.question,
            retrieved_sources=result.retrieved_sources,
            answer=generator.answer(result.question, result.retrieved_sources),
        ))
    output = StudentSearchResultsAndAnswer(search_results=answers, k=results.k)
    out_dir = Path(save_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(search_results_path).name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))
    return out_path