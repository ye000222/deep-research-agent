"""Local immutable artifact storage for cleaned source text."""

from __future__ import annotations

import asyncio
import json
import os
from hashlib import sha256
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

    async def read_text(self, artifact_uri: str) -> str:
        """Read a stored artifact while enforcing the configured-root boundary."""

        relative = Path(artifact_uri)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact path escaped configured root")
        destination = (self._root / relative).resolve()
        if self._root not in destination.parents:
            raise ValueError("artifact path escaped configured root")
        return await asyncio.to_thread(destination.read_text, encoding="utf-8")

    async def save_json(self, artifact_uri: str, value: object) -> str:
        """Atomically persist a derived JSON artifact below the configured root."""

        relative = Path(artifact_uri)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact path escaped configured root")
        destination = (self._root / relative).resolve()
        if self._root not in destination.parents:
            raise ValueError("artifact path escaped configured root")
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        await asyncio.to_thread(self._write_atomic, destination, payload)
        return relative.as_posix()

    async def create_json_once(self, artifact_uri: str, value: object) -> bool:
        """Create an immutable at-most-once marker; return False when it exists."""

        relative = Path(artifact_uri)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact path escaped configured root")
        destination = (self._root / relative).resolve()
        if self._root not in destination.parents:
            raise ValueError("artifact path escaped configured root")
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return await asyncio.to_thread(self._create_once, destination, payload)

    async def save_immutable_json(self, artifact_uri: str, value: object) -> str:
        """Persist a content-addressed artifact without ever replacing prior bytes."""

        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        created = await self.create_json_once(artifact_uri, value)
        if not created:
            existing = await self.read_text(artifact_uri)
            existing_hash = sha256(existing.encode("utf-8")).digest()
            payload_hash = sha256(payload.encode("utf-8")).digest()
            if existing_hash != payload_hash:
                raise ValueError("immutable_artifact_content_conflict")
        return artifact_uri

    async def read_json(self, artifact_uri: str) -> object:
        """Read a JSON artifact while enforcing the configured-root boundary."""

        raw = await self.read_text(artifact_uri)
        return json.loads(raw)

    @staticmethod
    def _write_atomic(destination: Path, text: str) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)

    @staticmethod
    def _create_once(destination: Path, text: str) -> bool:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(text)
        except FileExistsError:
            return False
        return True
