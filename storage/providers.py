"""Concrete storage backend providers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from .base import StorageBackend, StorageError


class LocalStorageBackend(StorageBackend):
    """Filesystem-backed storage using the local disk."""

    name = "local"

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = Path(root).expanduser().resolve() if root else None

    def _resolve(self, key: str) -> Path:
        path = Path(key)
        if not path.is_absolute() and self.root is not None:
            path = (self.root / key).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def read_bytes(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(key)
        return path.read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.write_bytes(data)

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()

    def exists(self, key: str) -> bool:
        path = self._resolve(key)
        return path.exists()


class S3StorageBackend(StorageBackend):
    """Amazon S3 backed storage."""

    name = "aws"

    def __init__(self, bucket: str, prefix: str = "", region: Optional[str] = None) -> None:
        try:
            import boto3  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise StorageError("boto3 is required for the AWS storage backend") from exc

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client("s3", region_name=region)

    def _object_key(self, key: str) -> str:
        key = key.lstrip("/")
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def read_bytes(self, key: str) -> bytes:
        object_key = self._object_key(key)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=object_key)
        except Exception as exc:  # pragma: no cover - surfaced at runtime
            raise FileNotFoundError(object_key) from exc
        body = response.get("Body")
        if body is None:
            return b""
        return body.read()

    def write_bytes(self, key: str, data: bytes) -> None:
        object_key = self._object_key(key)
        self.client.put_object(Bucket=self.bucket, Key=object_key, Body=data)

    def delete(self, key: str) -> None:
        object_key = self._object_key(key)
        self.client.delete_object(Bucket=self.bucket, Key=object_key)

    def exists(self, key: str) -> bool:
        object_key = self._object_key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=object_key)
            return True
        except Exception:  # pragma: no cover - head failure implies absence
            return False


class AzureBlobStorageBackend(StorageBackend):
    """Azure Blob Storage backend."""

    name = "azure"

    def __init__(
        self,
        container: str,
        prefix: str = "",
        connection_string: Optional[str] = None,
        account_url: Optional[str] = None,
        credential: Optional[str] = None,
    ) -> None:
        try:
            from azure.storage.blob import BlobServiceClient  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise StorageError("azure-storage-blob is required for the Azure backend") from exc

        if connection_string:
            service_client = BlobServiceClient.from_connection_string(connection_string)
        elif account_url:
            service_client = BlobServiceClient(account_url=account_url, credential=credential)
        else:  # pragma: no cover - configuration error
            raise StorageError(
                "Provide either AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL"
            )

        self.container = container
        self.prefix = prefix.strip("/")
        self.container_client = service_client.get_container_client(container)
        try:
            self.container_client.create_container()
        except Exception:
            # Container likely exists already; swallow errors to stay idempotent
            pass

    def _blob_name(self, key: str) -> str:
        key = key.lstrip("/")
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def read_bytes(self, key: str) -> bytes:
        blob_name = self._blob_name(key)
        blob = self.container_client.get_blob_client(blob_name)
        try:
            stream = blob.download_blob()
        except Exception as exc:  # pragma: no cover - surfaced at runtime
            raise FileNotFoundError(blob_name) from exc
        buffer = io.BytesIO()
        stream.readinto(buffer)
        return buffer.getvalue()

    def write_bytes(self, key: str, data: bytes) -> None:
        blob_name = self._blob_name(key)
        blob = self.container_client.get_blob_client(blob_name)
        blob.upload_blob(data, overwrite=True)

    def delete(self, key: str) -> None:
        blob_name = self._blob_name(key)
        blob = self.container_client.get_blob_client(blob_name)
        try:
            blob.delete_blob()
        except Exception:
            pass

    def exists(self, key: str) -> bool:
        blob_name = self._blob_name(key)
        blob = self.container_client.get_blob_client(blob_name)
        try:
            blob.get_blob_properties()
            return True
        except Exception:
            return False


class GCSStorageBackend(StorageBackend):
    """Google Cloud Storage backend."""

    name = "gcp"

    def __init__(self, bucket: str, prefix: str = "", project: Optional[str] = None) -> None:
        try:
            from google.cloud import storage as gcs  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise StorageError("google-cloud-storage is required for the GCP backend") from exc

        self.prefix = prefix.strip("/")
        self.client = gcs.Client(project=project)
        self.bucket = self.client.bucket(bucket)

    def _blob_name(self, key: str) -> str:
        key = key.lstrip("/")
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def read_bytes(self, key: str) -> bytes:
        blob = self.bucket.blob(self._blob_name(key))
        if not blob.exists():
            raise FileNotFoundError(key)
        return blob.download_as_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        blob = self.bucket.blob(self._blob_name(key))
        blob.upload_from_string(data)

    def delete(self, key: str) -> None:
        blob = self.bucket.blob(self._blob_name(key))
        try:
            blob.delete()
        except Exception:
            pass

    def exists(self, key: str) -> bool:
        blob = self.bucket.blob(self._blob_name(key))
        return blob.exists()

