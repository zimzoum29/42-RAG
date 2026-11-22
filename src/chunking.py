import os
import re
import ast
from pathlib import Path
from tqdm import tqdm
from .models import MinimalSource
from typing import Literal

FileKind = Literal["code", "doc"]

    
def collect_files(raw_dir: str) -> list[tuple[str, FileKind]]:
    """List the files to index under raw_dir, with their kind.

    Python files are "code"; Markdown and text files are "doc". Hidden
    directories and __pycache__ are skipped.

    Args:
        raw_dir: Root directory of the corpus (e.g. "data/raw").

    Returns:
        Sorted (path, kind) pairs. Paths are relative to the current
        directory, so they match the corpus paths expected by the grader.

    Raises:
        FileNotFoundError: If raw_dir is not a directory.
    """

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
        elif extension == ".md" or extension == ".txt":
            files.append((files_path, "doc"))
        else:
            continue

    files.sort()
    return files


def read_file(path: str) -> str | None:
    """Read a whole file as text, without changing its characters.

    Undecodable bytes are replaced by one character each, and line
    endings are kept, so character offsets match the file.

    Args:
        path: Path of the file to read.

    Returns:
        The content of the file, or None if it cannot be read.
    """

    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            return f.read()
        
    except OSError:
        return None


def chunk_python(path: str, content: str,
                 max_size: int) -> list[MinimalSource]:
    """Split a Python file into chunks that follow its structure.

    Each top-level function or class (with its decorators) becomes its
    own region, and so does the code between them. A class longer than
    max_size is also split between its methods. Regions still longer
    than max_size are cut on blank lines, then on line ends. A file that
    does not parse is only split by size.

    Args:
        path: Path of the file, stored in each chunk.
        content: Content of the file.
        max_size: Maximum length of a chunk, in characters.

    Returns:
        The non-blank chunks of the file, in order.
    """

    line_starts = [0]

    for i, char in enumerate(content):
        if char == "\n":
            line_starts.append(i + 1)

    line_starts.append(len(content))
    cuts = {0, len(content)}

    try:
        tree: ast.Module | None = ast.parse(content)

    except (SyntaxError, ValueError):
        tree = None

    if tree is not None:
        for node in tree.body:

            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            
            first_line = min([node.lineno] + [d.lineno for d in node.decorator_list])
            start = line_starts[first_line - 1]
            end = line_starts[node.end_lineno or node.lineno]
            cuts.update((start, end))

            if isinstance(node, ast.ClassDef) and end - start > max_size:

                for child in node.body:

                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        child_line = min([child.lineno] + [d.lineno for d in child.decorator_list])
                        cuts.add(line_starts[child_line - 1])
                        cuts.add(line_starts[child.end_lineno or child.lineno])

    bounds = sorted(cuts)
    chunks: list[MinimalSource] = []

    for start, end in zip(bounds, bounds[1:]):

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
                chunks.append(MinimalSource(file_path=path, first_character_index=pos, last_character_index=cut))

            pos = cut
              
    return chunks


def chunk_markdown(path: str, content: str,
                   max_size: int) -> list[MinimalSource]:
    """Split a Markdown file into one chunk per section.

    A section runs from a heading to the next one; "#" lines inside
    fenced code blocks are not headings. Sections longer than max_size
    are cut between paragraphs, then on line ends.

    Args:
        path: Path of the file, stored in each chunk.
        content: Content of the file.
        max_size: Maximum length of a chunk, in characters.

    Returns:
        The non-blank chunks of the file, in order.
    """

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
    chunks: list[MinimalSource] = []

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
                chunks.append(MinimalSource(file_path=path, first_character_index=pos, last_character_index=cut))
            
            pos = cut

    return chunks


def chunk_corpus(raw_dir: str,
                 max_chunk_size: int = 2000) -> list[MinimalSource]:
    """Chunk every indexable file of the corpus.

    Each file is read and split with the strategy of its kind: Python
    code or Markdown/text. Empty or unreadable files are skipped.

    Args:
        raw_dir: Root directory of the corpus.
        max_chunk_size: Maximum length of a chunk, from 1 to 2000.

    Returns:
        All the chunks, grouped by file in sorted order.

    Raises:
        ValueError: If max_chunk_size is out of range or no indexable
            file is found.
        FileNotFoundError: If raw_dir is not a directory.
    """

    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("max_chunk_size must be between 1 and 2000")

    files = collect_files(raw_dir)

    if not files:
        raise ValueError(f"No indexable files found in {raw_dir}")

    chunks: list[MinimalSource] = []

    for path, kind in tqdm(files, desc="Chunking"):

        content = read_file(path)

        if content is None or not content.strip():
            continue
        
        if kind == "code":
            chunks.extend(chunk_python(path, content, max_chunk_size))
        else:
            chunks.extend(chunk_markdown(path, content, max_chunk_size))

    return chunks