"""Storage backend factory exposed for orchestrator and automation modules."""

from .base import StorageBackend, StorageError
from .factory import get_storage_backend
from .config import load_storage_settings, StorageSettings

__all__ = [
    "StorageBackend",
    "StorageError",
    "StorageSettings",
    "get_storage_backend",
    "load_storage_settings",
]

