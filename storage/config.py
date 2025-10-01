"""Environment-driven configuration for storage backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class StorageSettings:
    backend: str
    container: str
    prefix: str
    local_root: str
    opd_key: str
    pdcm_key: str
    aws_region: Optional[str]
    azure_connection_string: Optional[str]
    azure_account_url: Optional[str]
    azure_credential: Optional[str]
    gcp_project: Optional[str]


def load_storage_settings() -> StorageSettings:
    backend = os.getenv("STORAGE_BACKEND", "local").strip().lower()
    container = os.getenv("STORAGE_CONTAINER", "").strip()
    prefix = os.getenv("STORAGE_PREFIX", "").strip()

    default_root = (
        Path(os.getenv("DOCSTRIBE_PROJECT_ROOT", Path.cwd())) / "data" / "jsonl"
    )
    local_root = os.getenv("LOCAL_STORAGE_ROOT", str(default_root))

    opd_key = os.getenv("OPD_STORAGE_KEY", "continental_opd_jsonl_file.jsonl")
    pdcm_key = os.getenv("PDCM_STORAGE_KEY", "ipd_jsonl_batch_file.jsonl")

    return StorageSettings(
        backend=backend,
        container=container,
        prefix=prefix,
        local_root=local_root,
        opd_key=opd_key,
        pdcm_key=pdcm_key,
        aws_region=os.getenv("AWS_REGION_NAME"),
        azure_connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        azure_account_url=os.getenv("AZURE_STORAGE_ACCOUNT_URL"),
        azure_credential=os.getenv("AZURE_STORAGE_CREDENTIAL"),
        gcp_project=os.getenv("GCP_PROJECT"),
    )

