"""Configuration helpers for the automation orchestrator."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from docstribe_agent_config import (
    DOCSTRIBE_MONGODB_URI,
    LLM_ORCHESTRATOR_DB,
    OPD_BATCH_COLLECTION,
    OPD_COLLECTION,
    PDCM_BATCH_COLLECTION,
    PDCM_COLLECTION,
)
from storage import load_storage_settings


_storage_settings = load_storage_settings()


@dataclass
class AutomationConfig:
    base_url: str = os.getenv("AUTOMATION_BASE_URL", "http://localhost:8080")
    redis_url: str = os.getenv("AUTOMATION_REDIS_URL", "redis://localhost:6379/0")
    enabled: bool = os.getenv("AUTOMATION_ENABLED", "false").lower() == "true"
    batch_threshold: int = int(os.getenv("AUTOMATION_BATCH_THRESHOLD", "100"))
    poll_interval_seconds: int = int(os.getenv("AUTOMATION_POLL_INTERVAL", "180"))
    incoming_queue: str = os.getenv("AUTOMATION_INCOMING_QUEUE", "opd:incoming")
    collect_queue: str = os.getenv("AUTOMATION_COLLECT_QUEUE", "opd:collect")
    batch_queue: str = os.getenv("AUTOMATION_BATCH_QUEUE", "opd:batch")
    completed_queue: str = os.getenv("AUTOMATION_COMPLETED_QUEUE", "opd:completed")
    mongodb_uri: str = DOCSTRIBE_MONGODB_URI
    mongodb_db: str = LLM_ORCHESTRATOR_DB
    opd_collection: str = OPD_COLLECTION
    pdcm_collection: str = PDCM_COLLECTION
    opd_batch_collection: str = OPD_BATCH_COLLECTION
    pdcm_batch_collection: str = PDCM_BATCH_COLLECTION
    storage_backend: str = _storage_settings.backend
    storage_container: str = _storage_settings.container
    storage_prefix: str = _storage_settings.prefix
    local_storage_root: str = _storage_settings.local_root
    opd_storage_key: str = _storage_settings.opd_key
    pdcm_storage_key: str = _storage_settings.pdcm_key


config = AutomationConfig()
