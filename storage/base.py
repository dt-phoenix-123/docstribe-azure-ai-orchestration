"""Abstract base classes and defaults for storage backends."""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class StorageError(RuntimeError):
    """Raised when a storage backend cannot fulfil an operation."""


class StorageBackend(ABC):
    """Minimal interface each storage backend must implement."""

    name: str = "base"

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """Return raw bytes stored at *key*.

        Should raise :class:`FileNotFoundError` if the object does not exist.
        """

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None:
        """Store raw *data* at *key*, overwriting any existing object."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the object identified by *key* if it exists."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return ``True`` when *key* is present in the backend."""

    # ------------------------------------------------------------------
    # Convenience helpers shared by all backends
    # ------------------------------------------------------------------

    def append_json_line(self, key: str, record: Dict[str, Any]) -> None:
        """Append *record* as a JSONL entry to *key*.

        Falls back to rewriting the entire payload when the backend does not
        provide native append semantics. Optimised backends can override this
        method for better performance.
        """

        payload = json.dumps(record, ensure_ascii=False).encode("utf-8")
        if self.exists(key):
            existing = self.read_bytes(key)
            if existing and not existing.endswith(b"\n"):
                existing += b"\n"
            payload = existing + payload
        self.write_bytes(key, payload)

    def line_count(self, key: str) -> int:
        """Return the number of non-empty lines stored at *key*."""

        if not self.exists(key):
            return 0
        content = self.read_bytes(key)
        text = content.decode("utf-8")
        return sum(1 for line in text.splitlines() if line.strip())

    def download_to_path(self, key: str, destination: Path) -> None:
        """Download *key* to *destination* on the local filesystem."""

        data = self.read_bytes(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)

    def upload_from_path(self, key: str, source: Path) -> None:
        """Upload local file *source* into storage as *key*."""

        data = source.read_bytes()
        self.write_bytes(key, data)

    def download_to_tempfile(self, key: str, suffix: str = "") -> Path:
        """Return a temporary file populated with the contents of *key*."""

        data = self.read_bytes(key)
        file_suffix = suffix or Path(key).suffix or ".tmp"
        fd, tmp_name = tempfile.mkstemp(suffix=file_suffix)
        path = Path(tmp_name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return path

