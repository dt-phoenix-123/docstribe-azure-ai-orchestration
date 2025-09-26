"""HTTP client for interacting with the orchestrator API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from .config import config


class DocstribeAPIClient:
    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or config.base_url).rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def process_opd_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(self._url("/process_opd_message"), json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def collect_opd_pending_requests(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        resp = requests.post(
            self._url("/collect_opd_pending_requests"),
            json={"responses": responses},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def upload_batch(self, batch_type: str, file_path: str) -> Dict[str, Any]:
        resp = requests.post(
            self._url("/upload_batch"),
            json={"type": batch_type, "file_url": file_path},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        resp = requests.post(
            self._url("/check_batch_status"),
            json={"batch_id": batch_id},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def retrieve_results(self, file_type: str, file_id: str) -> Dict[str, Any]:
        resp = requests.post(
            self._url("/retrieve_results"),
            json={"file_type": file_type, "file_id": file_id},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def process_pending_pdcm(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(self._url("/process_pdcm_message"), json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def collect_pdcm_pending_requests(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        resp = requests.post(
            self._url("/collect_pending_requests"),
            json={"responses": responses},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
