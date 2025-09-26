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
    jsonl_path: str = os.getenv(
        "AUTOMATION_JSONL_PATH",
        "/Users/Dr.NidhiChandra/Desktop/max/Max_AI_Orchestration/data/jsonl/continental_opd_jsonl_file.jsonl",
    )
    pdcm_jsonl_path: str = os.getenv(
        "AUTOMATION_PDCM_JSONL_PATH",
        "/Users/Dr.NidhiChandra/Desktop/max/Max_AI_Orchestration/data/jsonl/ipd_jsonl_batch_file.jsonl",
    )
    mongodb_uri: str = DOCSTRIBE_MONGODB_URI
    mongodb_db: str = LLM_ORCHESTRATOR_DB
    opd_collection: str = OPD_COLLECTION
    pdcm_collection: str = PDCM_COLLECTION
    opd_batch_collection: str = OPD_BATCH_COLLECTION
    pdcm_batch_collection: str = PDCM_BATCH_COLLECTION


config = AutomationConfig()
