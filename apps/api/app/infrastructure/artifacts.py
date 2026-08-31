"""Local immutable artifact storage for cleaned source text."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import UUID


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    async def save_page(self, run_id: UUID, source_id: UUID, text: str) -> str:
        relative = Path("runs") / str(run_id) / "sources" / f"{source_id}.txt"
        destination = (self._root / relative).resolve()
        if self._root not in destination.parents:
            raise ValueError("artifact path escaped configured root")
        await asyncio.to_thread(self._write_atomic, destination, text)
        return relative.as_posix()

    @staticmethod
    def _write_atomic(destination: Path, text: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
