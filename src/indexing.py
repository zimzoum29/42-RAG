from typing import Literal
from pathlib import Path
import os
from pydantic import BaseModel

FileKind = Literal["code", "doc"]

class Chunk(BaseModel):
    def __init__(self):
        file_path: str
        first_character_index: int
        last_character_inde: int
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

