"""MongoDB access utilities for automation workflows."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

try:
    from pymongo import MongoClient  # type: ignore
except ImportError:  # pragma: no cover
    MongoClient = None  # type: ignore

from .config import config

FINAL_BATCH_STATES = {"completed", "failed", "cancelled"}
DEFAULT_POLL_STATES = {"processing", "queued"}


class MongoStore:
    def __init__(self) -> None:
        self.client: Optional[MongoClient] = None
        self.db = None
        if MongoClient is not None and config.mongodb_uri:
            try:
                self.client = MongoClient(config.mongodb_uri)
                self.db = self.client[config.mongodb_db]
            except Exception:
                self.client = None
                self.db = None

    def fetch_pending_opd_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        if self.db is None:
            return []
        cursor = (
            self.db[config.opd_collection]
            .find({"status": "pending"})
            .sort("current_date", -1)
            .limit(limit)
        )
        return list(cursor)

    def count_pending_opd_documents(self) -> int:
        if self.db is None:
            return 0
        return self.db[config.opd_collection].count_documents({"status": "pending"})

    def fetch_processing_batches(self) -> List[Dict[str, Any]]:
        if self.db is None:
            return []
        cursor = self.db[config.opd_batch_collection].find({"status": {"$in": list(DEFAULT_POLL_STATES)}})
        return list(cursor)

    def count_processing_batches(self) -> int:
        if self.db is None:
            return 0
        return self.db[config.opd_batch_collection].count_documents({"status": {"$in": list(DEFAULT_POLL_STATES)}})

    def fetch_batches_for_polling(
        self,
        collection_name: str,
        active_statuses: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.db is None:
            return []
        statuses = list(active_statuses) if active_statuses is not None else list(DEFAULT_POLL_STATES)
        cursor = self.db[collection_name].find({"status": {"$in": statuses}})
        return list(cursor)

    def fetch_incomplete_batches(
        self,
        collection_name: str,
        terminal_statuses: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.db is None:
            return []
        terminals = list(terminal_statuses) if terminal_statuses is not None else list(FINAL_BATCH_STATES)
        cursor = self.db[collection_name].find({"status": {"$nin": terminals}})
        return list(cursor)

    def update_batch_status(
        self,
        batch_id: str,
        status: str,
        output_file_id: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        if self.db is None:
            return
        coll_name = collection_name or config.opd_batch_collection
        update_doc: Dict[str, Any] = {"status": status}
        if output_file_id:
            update_doc["output_file_id"] = output_file_id
        self.db[coll_name].update_one({"batch_id": batch_id}, {"$set": update_doc})

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
