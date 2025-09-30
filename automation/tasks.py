"""Task functions used by automation workers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from .api_client import DocstribeAPIClient
from .config import config
from .mongo_store import MongoStore

logger = logging.getLogger(__name__)


def queue_process_opd_message(api: DocstribeAPIClient, payload: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Submitting process_opd_message for patient_id=%s", payload.get("data_payload", {}).get("patient_id"))
    return api.process_opd_message(payload)


def collect_pending_opd(api: DocstribeAPIClient, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    responses = []
    for doc in docs:
        responses.append({
            "patient_details": doc.get("patient_details", {}),
            "abnormalities_identified": doc.get("abnormalities_identified", []),
            "clinical_assessment": doc.get("clinical_assessment", {}),
            "agentic_summary": doc.get("agentic_summary", {}),
            "radiology_findings": doc.get("radiology_findings", {}),
        })
    return api.collect_opd_pending_requests(responses)


def submit_opd_batch(api: DocstribeAPIClient, file_path: Path) -> Dict[str, Any]:
    logger.info("Uploading OPD JSONL batch from %s", file_path)
    return api.upload_batch("OPD", str(file_path))


def check_batches(api: DocstribeAPIClient, batch_ids: List[str]) -> List[Dict[str, Any]]:
    results = []
    for batch_id in batch_ids:
        result = api.check_batch_status(batch_id)
        results.append(result)
    return results


def retrieve_opd_results(api: DocstribeAPIClient, file_id: str) -> Dict[str, Any]:
    return api.retrieve_results("OPD", file_id)


def fetch_pending_docs(limit: int = 100) -> List[Dict[str, Any]]:
    store = MongoStore()
    try:
        return store.fetch_pending_opd_documents(limit=limit)
    finally:
        store.close()


def count_pending_docs() -> int:
    store = MongoStore()
    try:
        return store.count_pending_opd_documents()
    finally:
        store.close()


def count_processing_batches() -> int:
    store = MongoStore()
    try:
        return store.count_processing_batches()
    finally:
        store.close()


def fetch_batches_for_polling() -> Dict[str, List[Dict[str, Any]]]:
    store = MongoStore()
    try:
        return {
            "OPD": store.fetch_incomplete_batches(config.opd_batch_collection),
            "PDCM": store.fetch_incomplete_batches(config.pdcm_batch_collection),
        }
    finally:
        store.close()
