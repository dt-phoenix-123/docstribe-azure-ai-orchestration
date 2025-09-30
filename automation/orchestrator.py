"""Automation orchestrator coordinating Redis queues and API calls."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:  # pragma: no cover - script execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from .api_client import DocstribeAPIClient
from .config import config
from .redis_queue import RedisQueue
from . import tasks
from storage import get_storage_backend

logger = logging.getLogger(__name__)


class BaseWorker(threading.Thread):
    def __init__(self, name: str, stop_event: threading.Event) -> None:
        super().__init__(name=name, daemon=True)
        self.stop_event = stop_event

    def stopped(self) -> bool:
        return self.stop_event.is_set()


class IncomingWorker(BaseWorker):
    def __init__(self, queue: RedisQueue, api: DocstribeAPIClient, stop_event: threading.Event) -> None:
        super().__init__(name="IncomingWorker", stop_event=stop_event)
        self.queue = queue
        self.api = api

    def run(self) -> None:  # pragma: no cover - threading
        while not self.stopped():
            job = self.queue.pop(timeout=5)
            if job is None:
                continue
            try:
                tasks.queue_process_opd_message(self.api, job)
            except Exception as exc:  # pragma: no cover - log only
                logger.exception("Failed to process incoming OPD job: %s", exc)


class PendingCollectorWorker(BaseWorker):
    def __init__(self, api: DocstribeAPIClient, stop_event: threading.Event, interval: int) -> None:
        super().__init__(name="PendingCollectorWorker", stop_event=stop_event)
        self.api = api
        self.interval = interval

    def run(self) -> None:  # pragma: no cover
        while not self.stopped():
            docs = tasks.fetch_pending_docs(limit=config.batch_threshold)
            if docs:
                logger.info("Collector found %d pending OPD documents", len(docs))
                try:
                    tasks.collect_pending_opd(self.api, docs)
                except Exception as exc:
                    logger.exception("collect_opd_pending_requests failed: %s", exc)
            time.sleep(self.interval)


class BatchSubmitWorker(BaseWorker):
    def __init__(
        self,
        api: DocstribeAPIClient,
        orchestrator: "AutomationOrchestrator",
        storage_backend,
        storage_key: str,
        stop_event: threading.Event,
        interval: int,
    ) -> None:
        super().__init__(name="BatchSubmitWorker", stop_event=stop_event)
        self.api = api
        self.interval = interval
        self.orchestrator = orchestrator
        self.storage = storage_backend
        self.storage_key = storage_key

    def run(self) -> None:  # pragma: no cover
        while not self.stopped():
            try:
                line_count = self.storage.line_count(self.storage_key)
                pending_count = tasks.count_pending_docs()
                processing_count = tasks.count_processing_batches()

                submit_reason = None
                if line_count >= config.batch_threshold:
                    submit_reason = f"threshold reached ({line_count} lines)"
                elif line_count > 0 and pending_count == 0:
                    submit_reason = "pending queue empty"
                elif line_count > 0 and processing_count >= config.batch_threshold:
                    submit_reason = f"processing backlog {processing_count}"

                if not submit_reason:
                    time.sleep(self.interval)
                    continue

                logger.info("Submitting batch because %s", submit_reason)
                tmp_path: Optional[Path] = None
                try:
                    tmp_path = self.storage.download_to_tempfile(self.storage_key, suffix=".jsonl")
                except FileNotFoundError:
                    logger.warning(
                        "Batch submission triggered but storage object %s was not found",
                        self.storage_key,
                    )

                if tmp_path is None:
                    time.sleep(self.interval)
                    continue

                try:
                    response = tasks.submit_opd_batch(self.api, tmp_path)
                    batch_id = response.get("batch_id") or response.get("id")
                    if batch_id:
                        logger.info("Submitted batch %s", batch_id)
                        self.orchestrator.register_batch(batch_id, "OPD")
                        try:
                            self.storage.delete(self.storage_key)
                            logger.debug(
                                "Cleared storage object %s after batch submission",
                                self.storage_key,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed to delete storage object %s: %s",
                                self.storage_key,
                                exc,
                            )
                finally:
                    if tmp_path is not None:
                        tmp_path.unlink(missing_ok=True)

                time.sleep(self.interval)
            except Exception as exc:
                logger.exception("Batch submission failed: %s", exc)
                time.sleep(self.interval)


class BatchPollerWorker(BaseWorker):
    def __init__(self, api: DocstribeAPIClient, orchestrator: "AutomationOrchestrator", stop_event: threading.Event, interval: int = 180) -> None:
        super().__init__(name="BatchPollerWorker", stop_event=stop_event)
        self.api = api
        self.orchestrator = orchestrator
        self.interval = interval

    def run(self) -> None:  # pragma: no cover
        while not self.stopped():
            batches_by_type = tasks.fetch_batches_for_polling()
            aggregated: Dict[str, str] = {}
            for batch_type, docs in batches_by_type.items():
                if not docs:
                    continue
                for doc in docs:
                    batch_id = doc.get("batch_id")
                    if not batch_id:
                        continue
                    aggregated.setdefault(batch_id, (batch_type or "OPD").upper())

            for batch_id, batch_type in self.orchestrator.get_active_batches():
                aggregated.setdefault(batch_id, (batch_type or "OPD").upper())

            logger.debug("Batch poller waking up with %d batches to check", len(aggregated))

            for batch_id, batch_type in aggregated.items():
                try:
                    status = self.api.check_batch_status(batch_id)
                    state = status.get("status")
                    logger.info("Batch %s status=%s", batch_id, state)
                    if state == "completed" and status.get("output_file_id"):
                        file_id = status.get("output_file_id")
                        file_type = "PDCM" if batch_type == "PDCM" else "OPD"
                        self.orchestrator.enqueue_completed_batch(batch_id, file_id, file_type)
                        self.orchestrator.unregister_batch(batch_id)
                    elif state in {"failed", "cancelled"}:
                        logger.warning("Batch %s ended with status %s", batch_id, state)
                        self.orchestrator.unregister_batch(batch_id)
                except Exception as exc:
                    logger.exception("Failed to poll batch %s: %s", batch_id, exc)
            time.sleep(self.interval)


class ResultWorker(BaseWorker):
    def __init__(self, api: DocstribeAPIClient, completed_queue: RedisQueue, stop_event: threading.Event) -> None:
        super().__init__(name="ResultWorker", stop_event=stop_event)
        self.api = api
        self.queue = completed_queue

    def run(self) -> None:  # pragma: no cover
        while not self.stopped():
            item = self.queue.pop(timeout=5)
            if item is None:
                continue
            file_id = item.get("file_id")
            file_type = (item.get("file_type") or "OPD").upper()
            if not file_id:
                continue
            try:
                logger.info("Retrieving %s results for file_id=%s", file_type, file_id)
                self.api.retrieve_results(file_type, file_id)
            except Exception as exc:
                logger.exception("Failed to retrieve results for %s: %s", file_id, exc)


class AutomationOrchestrator:
    """High-level controller coordinating Redis queues, Mongo, and REST APIs."""

    def __init__(self, enable: Optional[bool] = None) -> None:
        self.config_enabled = config.enabled if enable is None else enable
        self.stop_event = threading.Event()
        self.api = DocstribeAPIClient()
        self.incoming_queue = RedisQueue(config.incoming_queue)
        self.completed_queue = RedisQueue(config.completed_queue)
        self.workers: List[BaseWorker] = []
        self._active_batches: Dict[str, str] = {}
        self._batch_lock = threading.Lock()
        self.storage_backend = get_storage_backend()

    def register_batch(self, batch_id: str, batch_type: str) -> None:
        with self._batch_lock:
            self._active_batches[batch_id] = (batch_type or "OPD").upper()

    def unregister_batch(self, batch_id: str) -> None:
        with self._batch_lock:
            self._active_batches.pop(batch_id, None)

    def get_active_batches(self) -> List[Tuple[str, str]]:
        with self._batch_lock:
            return list(self._active_batches.items())

    def enqueue_completed_batch(self, batch_id: str, file_id: str, file_type: str) -> None:
        file_type_norm = (file_type or "OPD").upper()
        logger.info(
            "Batch %s completed; enqueueing %s results fetch (file_id=%s)",
            batch_id,
            file_type_norm,
            file_id,
        )
        self.completed_queue.push(
            {"batch_id": batch_id, "file_id": file_id, "file_type": file_type_norm}
        )

    def start(self) -> None:
        if not self.config_enabled:
            logger.warning("Automation orchestrator disabled. Set AUTOMATION_ENABLED=true to run.")
            return

        logger.info("Starting automation orchestrator")
        incoming_worker = IncomingWorker(self.incoming_queue, self.api, self.stop_event)
        collector_worker = PendingCollectorWorker(self.api, self.stop_event, interval=60)
        batch_submit_worker = BatchSubmitWorker(
            self.api,
            self,
            self.storage_backend,
            config.opd_storage_key,
            self.stop_event,
            interval=30,
        )
        batch_poller_worker = BatchPollerWorker(self.api, self, self.stop_event, interval=config.poll_interval_seconds)
        result_worker = ResultWorker(self.api, self.completed_queue, self.stop_event)

        self.workers = [
            incoming_worker,
            collector_worker,
            batch_submit_worker,
            batch_poller_worker,
            result_worker,
        ]

        for worker in self.workers:
            worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=5)
        logger.info("Automation orchestrator stopped")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    orchestrator = AutomationOrchestrator()
    orchestrator.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        orchestrator.stop()
