from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

ChunkKind = Literal["module", "class", "function", "method"]


@dataclass
class Chunk:
    file_path: str  # posix path relative to repo root
    qualname: str  # e.g. "service.UserService.get_user" or "service" for module-level
    kind: ChunkKind
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    text: str
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    class_name: str | None = None

    @property
    def chunk_id(self) -> str:
        return f"{self.file_path}::{self.qualname}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.to_dict() for c in chunks], indent=2))


def load_chunks(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text())
    return [Chunk.from_dict(d) for d in data]
