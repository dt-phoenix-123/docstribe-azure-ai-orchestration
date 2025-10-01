"""Factory helpers that create storage backends based on environment settings."""

from __future__ import annotations

from functools import lru_cache

from .base import StorageBackend, StorageError
from .config import load_storage_settings
from .providers import (
    AzureBlobStorageBackend,
    GCSStorageBackend,
    LocalStorageBackend,
    S3StorageBackend,
)


@lru_cache(maxsize=1)
def get_storage_backend() -> StorageBackend:
    settings = load_storage_settings()
    backend = settings.backend

    if backend in {"local", "filesystem", "fs"}:
        return LocalStorageBackend(root=settings.local_root)

    if backend in {"aws", "s3"}:
        if not settings.container:
            raise StorageError("STORAGE_CONTAINER must be set for the AWS storage backend")
        return S3StorageBackend(
            bucket=settings.container,
            prefix=settings.prefix,
            region=settings.aws_region,
        )

    if backend in {"azure", "azure_blob"}:
        if not settings.container:
            raise StorageError("STORAGE_CONTAINER must be set for the Azure storage backend")
        return AzureBlobStorageBackend(
            container=settings.container,
            prefix=settings.prefix,
            connection_string=settings.azure_connection_string,
            account_url=settings.azure_account_url,
            credential=settings.azure_credential,
        )

    if backend in {"gcp", "gcs", "google"}:
        if not settings.container:
            raise StorageError("STORAGE_CONTAINER must be set for the GCP storage backend")
        return GCSStorageBackend(
            bucket=settings.container,
            prefix=settings.prefix,
            project=settings.gcp_project,
        )

    raise StorageError(f"Unsupported storage backend '{backend}'")

