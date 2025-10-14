"""Storage module for handling different storage backends (local, AWS, Azure, GCP)."""

from __future__ import annotations

import os
import tempfile
import fcntl
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, ContextManager
from contextlib import contextmanager


@dataclass
class StorageSettings:
    """Configuration settings for storage backend."""
    backend: str
    container: str
    prefix: str
    local_root: str
    opd_key: str
    pdcm_key: str


def load_storage_settings() -> StorageSettings:
    """Load storage configuration from environment variables."""
    return StorageSettings(
        backend=os.getenv("STORAGE_BACKEND", "local"),
        container=os.getenv("STORAGE_CONTAINER", ""),
        prefix=os.getenv("STORAGE_PREFIX", ""),
        local_root=os.getenv("LOCAL_STORAGE_ROOT", "./data/jsonl"),
        opd_key=os.getenv("OPD_STORAGE_KEY", "continental_opd_jsonl_file.jsonl"),
        pdcm_key=os.getenv("PDCM_STORAGE_KEY", "ipd_jsonl_batch_file.jsonl"),
    )


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def line_count(self, key: str) -> int:
        """Return the number of lines in the storage object."""
        pass

    @abstractmethod
    def download_to_tempfile(self, key: str, suffix: str = "") -> Optional[Path]:
        """Download storage object to a temporary file and return the path."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the storage object."""
        pass

    @abstractmethod
    def append_line(self, key: str, line: str) -> None:
        """Append a line to the storage object."""
        pass
        
    def append_lines(self, key: str, lines: list[str]) -> None:
        """Append multiple lines to the storage object.
        
        Default implementation calls append_line for each line.
        Backends can override this for better performance.
        """
        for line in lines:
            self.append_line(key, line)


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend with file locking to prevent race conditions."""

    def __init__(self, root_path: str, prefix: str = ""):
        self.root_path = Path(root_path)
        self.prefix = prefix
        # Ensure root directory exists
        self.root_path.mkdir(parents=True, exist_ok=True)
        
    @contextmanager
    def _file_lock(self, file_path: Path) -> ContextManager:
        """Context manager for file locking to prevent race conditions.
        
        Uses advisory file locking (fcntl.flock) to ensure atomic operations.
        """
        lock_path = file_path.with_suffix(file_path.suffix + '.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open the lock file (create it if it doesn't exist)
        lock_file = None
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            try:
                lock_file = open(lock_path, 'w+')
                # Get exclusive, non-blocking lock
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError:
                # Another process has the lock, wait a bit and retry
                if lock_file:
                    lock_file.close()
                attempt += 1
                time.sleep(0.1 * attempt)  # Exponential backoff
        
        if attempt == max_attempts:
            raise TimeoutError(f"Could not acquire lock for {file_path} after {max_attempts} attempts")
            
        try:
            yield
        finally:
            if lock_file:
                # Release the lock and close/remove the lock file
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                try:
                    lock_path.unlink(missing_ok=True)
                except Exception:
                    pass  # Best-effort cleanup

    def _get_file_path(self, key: str) -> Path:
        """Get the full file path for a storage key."""
        if self.prefix:
            key = f"{self.prefix}/{key}"
        return self.root_path / key

    def line_count(self, key: str) -> int:
        """Return the number of lines in the file.
        
        Uses file locking to ensure consistent reads even during writes.
        """
        file_path = self._get_file_path(key)
        if not file_path.exists():
            return 0
        
        try:
            with self._file_lock(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return sum(1 for _ in f)
        except Exception:
            return 0

    def download_to_tempfile(self, key: str, suffix: str = "") -> Optional[Path]:
        """Copy local file to a temporary file and return the path.
        
        Uses file locking to ensure a consistent copy even during writes.
        """
        file_path = self._get_file_path(key)
        if not file_path.exists():
            raise FileNotFoundError(f"Storage object {key} not found")

        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
        temp_path_obj = Path(temp_path)
        
        try:
            # Use file locking to ensure consistent reads
            with self._file_lock(file_path):
                # Copy file content to temp file
                with open(temp_fd, 'wb') as temp_file, open(file_path, 'rb') as source_file:
                    temp_file.write(source_file.read())
            return temp_path_obj
        except Exception as e:
            # Clean up temp file if copy failed
            temp_path_obj.unlink(missing_ok=True)
            raise e

    def delete(self, key: str) -> None:
        """Delete the local file with proper locking."""
        file_path = self._get_file_path(key)
        with self._file_lock(file_path):
            file_path.unlink(missing_ok=True)

    def append_line(self, key: str, line: str) -> None:
        """Append a line to the local file with proper locking to prevent race conditions."""
        file_path = self._get_file_path(key)
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use file locking to ensure atomic appends
        with self._file_lock(file_path):
            with open(file_path, 'a', encoding='utf-8') as f:
                if not line.endswith('\n'):
                    line += '\n'
                f.write(line)
    
    def append_lines(self, key: str, lines: list[str]) -> None:
        """Append multiple lines to the local file efficiently with proper locking.
        
        More efficient than multiple append_line calls as it only acquires the lock once.
        """
        if not lines:
            return
            
        file_path = self._get_file_path(key)
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use file locking to ensure atomic batch append
        with self._file_lock(file_path):
            with open(file_path, 'a', encoding='utf-8') as f:
                for line in lines:
                    if not line.endswith('\n'):
                        line += '\n'
                    f.write(line)


class AWSStorageBackend(StorageBackend):
    """AWS S3 storage backend."""

    def __init__(self, container: str, prefix: str = ""):
        self.container = container
        self.prefix = prefix
        try:
            import boto3
            self.client = boto3.client('s3')
        except ImportError:
            raise ImportError("boto3 is required for AWS storage backend")

    def _get_key(self, key: str) -> str:
        """Get the full S3 key with prefix."""
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def line_count(self, key: str) -> int:
        """Return the number of lines in the S3 object."""
        s3_key = self._get_key(key)
        try:
            response = self.client.get_object(Bucket=self.container, Key=s3_key)
            content = response['Body'].read().decode('utf-8')
            return len(content.splitlines())
        except Exception:
            return 0

    def download_to_tempfile(self, key: str, suffix: str = "") -> Optional[Path]:
        """Download S3 object to a temporary file and return the path."""
        s3_key = self._get_key(key)
        
        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
            temp_path_obj = Path(temp_path)
            
            # Download from S3
            with open(temp_fd, 'wb') as temp_file:
                self.client.download_fileobj(self.container, s3_key, temp_file)
            
            return temp_path_obj
        except Exception as e:
            if 'NoSuchKey' in str(e):
                raise FileNotFoundError(f"Storage object {key} not found")
            raise e

    def delete(self, key: str) -> None:
        """Delete the S3 object."""
        s3_key = self._get_key(key)
        self.client.delete_object(Bucket=self.container, Key=s3_key)

    def append_line(self, key: str, line: str) -> None:
        """Append a line to the S3 object."""
        s3_key = self._get_key(key)
        if not line.endswith('\n'):
            line += '\n'
        
        # For S3, we need to download, append, and re-upload
        try:
            # Try to get existing content
            response = self.client.get_object(Bucket=self.container, Key=s3_key)
            existing_content = response['Body'].read().decode('utf-8')
            new_content = existing_content + line
        except Exception:
            # File doesn't exist, create new
            new_content = line
        
        # Upload the updated content
        self.client.put_object(
            Bucket=self.container,
            Key=s3_key,
            Body=new_content.encode('utf-8')
        )


class AzureStorageBackend(StorageBackend):
    """Azure Blob Storage backend."""

    def __init__(self, container: str, prefix: str = ""):
        self.container = container
        self.prefix = prefix
        try:
            from azure.storage.blob import BlobServiceClient
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
            credential = os.getenv("AZURE_STORAGE_CREDENTIAL")
            
            if connection_string:
                self.client = BlobServiceClient.from_connection_string(connection_string)
            elif account_url:
                self.client = BlobServiceClient(account_url=account_url, credential=credential)
            else:
                raise ValueError("AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL must be set")
        except ImportError:
            raise ImportError("azure-storage-blob is required for Azure storage backend")

    def _get_blob_name(self, key: str) -> str:
        """Get the full blob name with prefix."""
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def line_count(self, key: str) -> int:
        """Return the number of lines in the blob."""
        blob_name = self._get_blob_name(key)
        try:
            blob_client = self.client.get_blob_client(container=self.container, blob=blob_name)
            content = blob_client.download_blob().readall().decode('utf-8')
            return len(content.splitlines())
        except Exception:
            return 0

    def download_to_tempfile(self, key: str, suffix: str = "") -> Optional[Path]:
        """Download blob to a temporary file and return the path."""
        blob_name = self._get_blob_name(key)
        
        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
            temp_path_obj = Path(temp_path)
            
            # Download from Azure Blob
            blob_client = self.client.get_blob_client(container=self.container, blob=blob_name)
            with open(temp_fd, 'wb') as temp_file:
                download_stream = blob_client.download_blob()
                temp_file.write(download_stream.readall())
            
            return temp_path_obj
        except Exception as e:
            if 'BlobNotFound' in str(e):
                raise FileNotFoundError(f"Storage object {key} not found")
            raise e

    def delete(self, key: str) -> None:
        """Delete the blob."""
        blob_name = self._get_blob_name(key)
        blob_client = self.client.get_blob_client(container=self.container, blob=blob_name)
        blob_client.delete_blob()

    def append_line(self, key: str, line: str) -> None:
        """Append a line to the blob."""
        blob_name = self._get_blob_name(key)
        if not line.endswith('\n'):
            line += '\n'
        
        blob_client = self.client.get_blob_client(container=self.container, blob=blob_name)
        
        try:
            # Try to get existing content
            existing_content = blob_client.download_blob().readall().decode('utf-8')
            new_content = existing_content + line
        except Exception:
            # Blob doesn't exist, create new
            new_content = line
        
        # Upload the updated content
        blob_client.upload_blob(new_content.encode('utf-8'), overwrite=True)


class GCPStorageBackend(StorageBackend):
    """Google Cloud Storage backend."""

    def __init__(self, container: str, prefix: str = ""):
        self.container = container
        self.prefix = prefix
        try:
            from google.cloud import storage
            project = os.getenv("GCP_PROJECT")
            self.client = storage.Client(project=project)
            self.bucket = self.client.bucket(container)
        except ImportError:
            raise ImportError("google-cloud-storage is required for GCP storage backend")

    def _get_blob_name(self, key: str) -> str:
        """Get the full blob name with prefix."""
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def line_count(self, key: str) -> int:
        """Return the number of lines in the blob."""
        blob_name = self._get_blob_name(key)
        try:
            blob = self.bucket.blob(blob_name)
            content = blob.download_as_text()
            return len(content.splitlines())
        except Exception:
            return 0

    def download_to_tempfile(self, key: str, suffix: str = "") -> Optional[Path]:
        """Download blob to a temporary file and return the path."""
        blob_name = self._get_blob_name(key)
        
        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
            temp_path_obj = Path(temp_path)
            
            # Download from GCS
            blob = self.bucket.blob(blob_name)
            with open(temp_fd, 'wb') as temp_file:
                blob.download_to_file(temp_file)
            
            return temp_path_obj
        except Exception as e:
            if 'Not Found' in str(e):
                raise FileNotFoundError(f"Storage object {key} not found")
            raise e

    def delete(self, key: str) -> None:
        """Delete the blob."""
        blob_name = self._get_blob_name(key)
        blob = self.bucket.blob(blob_name)
        blob.delete()

    def append_line(self, key: str, line: str) -> None:
        """Append a line to the blob."""
        blob_name = self._get_blob_name(key)
        if not line.endswith('\n'):
            line += '\n'
        
        blob = self.bucket.blob(blob_name)
        
        try:
            # Try to get existing content
            existing_content = blob.download_as_text()
            new_content = existing_content + line
        except Exception:
            # Blob doesn't exist, create new
            new_content = line
        
        # Upload the updated content
        blob.upload_from_string(new_content)


def get_storage_backend() -> StorageBackend:
    """Get the appropriate storage backend based on configuration."""
    settings = load_storage_settings()
    
    if settings.backend == "local":
        return LocalStorageBackend(settings.local_root, settings.prefix)
    elif settings.backend == "aws":
        return AWSStorageBackend(settings.container, settings.prefix)
    elif settings.backend == "azure":
        return AzureStorageBackend(settings.container, settings.prefix)
    elif settings.backend == "gcp":
        return GCPStorageBackend(settings.container, settings.prefix)
    else:
        raise ValueError(f"Unsupported storage backend: {settings.backend}")