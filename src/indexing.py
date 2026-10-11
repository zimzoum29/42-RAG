import os
import re
from typing import Literal
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel

FileKind = Literal["code", "doc"]

class Chunk(BaseModel):
        file_path: str
        first_character_index: int
        last_character_index: int
        kind: FileKind
        text: str

    
def collect_files(raw_dir: str) -> list[tuple[str, FileKind]]:

    root = Path(raw_dir)
    files = []
    
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {raw_dir}")

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        relative_parts = path.relative_to(root).parts
        
        if any(part for part in relative_parts if part.startswith('.') or part == "__pycache__"): 
            continue

        files_path = Path(os.path.relpath(path, Path.cwd())).as_posix()
        extension = path.suffix.lower()

        if extension == ".py":
            files.append((files_path, "code"))
        elif extension == ".md":
            files.append((files_path, "doc"))
        else:
            continue

    files.sort()
    return files


def read_file(path: str) -> str:

    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            return f.read()
        
    except OSError:
        return None


def split_by_size(content: str, start: int, end: int, max_size: int) -> list[tuple[int, int]]:

    spans: list[tuple[int, int]] = []
    pos = start

    while end - pos > max_size:

        window_end = pos + max_size
        newline = content.rfind("\n", pos, window_end)
        cut = newline + 1 if newline > pos else window_end
        spans.append((pos, cut))
        pos = cut

    if pos < end:
        spans.append((pos, end))

    return spans


def make_chunks(path: str, content: str, kind: FileKind, spans: list[tuple[int, int]]) -> list[Chunk]:
    return [
        Chunk(file_path=path, first_character_index=s, last_character_index=e, kind=kind, text=content[s:e])
        for s, e in spans
        if content[s:e].strip()
    ]


def chunk_python(path: str, content: str, max_size: int) -> list[Chunk]:

    return make_chunks(path, content, "code", split_by_size(content, 0, len(content), max_size))


def chunk_markdown(path: str, content: str, max_size: int) -> list[Chunk]:

    starts = [0]
    in_fence = False
    fence = ""
    pos = 0
    for line in content.splitlines(keepends=True):
        match = re.compile(r"[ ]{0,3}(`{3,}|~{3,})").match(line)
        if match:
            marker = match.group(1)
            if not in_fence:
                in_fence, fence = True, marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                in_fence = False
        elif not in_fence and re.compile(r"#{1,6}(?:[ \t]|$)").match(line) and pos != 0:
            starts.append(pos)
        pos += len(line)
    ends = starts[1:] + [len(content)]

    chunks: list[Chunk] = []
    for start, end in zip(starts, ends):
        pos = start
        while pos < end:
            cut = end
            if end - pos > max_size:
                window_end = pos + max_size
                cut = window_end
                for sep in ("\n\n", "\n"):
                    idx = content.rfind(sep, pos, window_end)
                    if idx > pos:
                        cut = idx + len(sep)
                        break
            text = content[pos:cut]
            if text.strip():
                chunks.append(Chunk(
                    file_path=path,
                    first_character_index=pos,
                    last_character_index=cut,
                    kind="doc",
                    text=text,
                ))
            pos = cut
    return chunks


def chunk_corpus(raw_dir: str, max_chunk_size: int = 2000) -> list[Chunk]:

    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("max_chunk_size must be between 1 and 2000")

    files = collect_files(raw_dir)

    if not files:
        raise ValueError(f"No indexable files found in {raw_dir}")

    chunks: list[Chunk] = []

    for path, kind in tqdm(files, desc="Chunking"):

        content = read_file(path)

        if content is None or not content.strip():
            continue
        
        if kind == "code":
            chunks.extend(chunk_python(path, content, max_chunk_size))
        else:
            chunks.extend(chunk_markdown(path, content, max_chunk_size))

    return chunks