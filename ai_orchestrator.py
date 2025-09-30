#!/usr/bin/env python3
"""Docstribe Orchestrator Flask application.

This module exposes a Flask app that wraps the DocstribeOrchestrator class, which
encapsulates all route handlers and supporting helpers. The orchestrator is designed
around local processing (no Google Cloud dependencies) and loads prompt templates
from ``prompt_config.yaml`` so that system/user prompts can be adjusted without
changing the Python code.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from flask import Flask, Response, jsonify, request

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

if load_dotenv is not None:  # pragma: no cover - depends on runtime package install
    load_dotenv()

try:  # Optional dependency; keep import guarded
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None  # type: ignore

try:  # Optional MongoDB support; fall back to local storage if unavailable
    from pymongo.mongo_client import MongoClient
    from pymongo.server_api import ServerApi
except ImportError:  # pragma: no cover - handled at runtime
    MongoClient = None  # type: ignore
    ServerApi = None  # type: ignore

try:  # OpenAI client
    from openai import AzureOpenAI
except ImportError as exc:  # pragma: no cover - surfaced clearly at runtime
    raise ImportError("openai package is required for Docstribe Orchestrator") from exc

try:  # Domain-specific helpers
    from docstribe_summarizer import (
        summarize_diagnostics,
        summarize_agent,
        summarize_lab_data_agent,
        detect_op_abnormalities,
        cohort_generation_agent,
        reasoning_llm,
        reasoning_llm_openai,
        reasoning_llm_v2,
    )
except ImportError:  # pragma: no cover - surfaced clearly at runtime
    summarize_diagnostics = None  # type: ignore
    summarize_agent = None  # type: ignore
    summarize_lab_data_agent = None  # type: ignore
    detect_op_abnormalities = None  # type: ignore
    cohort_generation_agent = None  # type: ignore
    reasoning_llm = None  # type: ignore
    reasoning_llm_openai = None  # type: ignore
    reasoning_llm_v2 = None  # type: ignore


def _missing_dependency(name: str):
    raise RuntimeError(
        f"{name} is unavailable because docstribe_summarizer (or its dependencies) could not be imported."
    )


if summarize_diagnostics is None:  # pragma: no cover - depends on runtime setup
    def summarize_diagnostics(*args, **kwargs):  # type: ignore
        _missing_dependency("summarize_diagnostics")


if summarize_agent is None:  # pragma: no cover
    def summarize_agent(medical_text: str, discharge_date: str, *_, **__):  # type: ignore
        logger.warning(
            "summarize_agent unavailable; returning minimal discharge summary stub"
        )
        return {
            "medical_history": [],
            "key_findings": [],
            "chief_complaints": [],
            "procedures_done": [],
            "follow_up": [],
            "medications": [],
            "death_flag": "false",
            "cohorts": {"primary": "", "secondary": []},
            "critical_medications": [],
            "actionables": [],
        }


if summarize_lab_data_agent is None:  # pragma: no cover
    def summarize_lab_data_agent(patient_data: Dict[str, Any]):  # type: ignore
        logger.warning(
            "summarize_lab_data_agent unavailable; returning empty clinical assessment"
        )
        return {"clinical_assessment": {}}


if detect_op_abnormalities is None:  # pragma: no cover
    def detect_op_abnormalities(test_values, gender="Male"):  # type: ignore
        logger.warning(
            "detect_op_abnormalities unavailable; returning tests without tagging"
        )
        return test_values


if cohort_generation_agent is None:  # pragma: no cover
    def cohort_generation_agent(*args, **kwargs):  # type: ignore
        _missing_dependency("cohort_generation_agent")


try:
    from docstribe_agent_config import *  # noqa: F401,F403
    import docstribe_agent_config as agent_cfg  # type: ignore
except ImportError:  # pragma: no cover - allow orchestrator to run with fallbacks
    agent_cfg = None  # type: ignore

try:
    from docstribe_knowledge_script import *  # noqa: F401,F403
except ImportError:  # pragma: no cover
    pass

try:
    from pdcm_pydantic_validator import parse_json  # type: ignore
except ImportError:  # pragma: no cover
    parse_json = None  # type: ignore

try:
    from opd_pydantic_validator import opd_parse_json  # type: ignore
except ImportError:  # pragma: no cover
    opd_parse_json = None  # type: ignore

from dateutil import parser  # noqa: F401  # parity with original code
from fuzzywuzzy import process  # type: ignore
import pandas as pd  # noqa: F401
import fcntl  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ensure_yaml_loaded() -> None:
    if yaml is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "PyYAML is required to load prompt_config.yaml. Please install 'pyyaml'."
        )


class PromptManager:
    """Load and format prompt templates from a YAML configuration file."""

    def __init__(self, config_path: Path):
        _ensure_yaml_loaded()
        self.config_path = config_path
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Prompt configuration not found at {self.config_path}"  # pragma: no cover
            )
        with self.config_path.open("r", encoding="utf-8") as fh:
            self.prompts = yaml.safe_load(fh) or {}

    def get(self, dotted_key: str, default: Optional[str] = None) -> Optional[str]:
        parts = dotted_key.split(".")
        node: Any = self.prompts
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def render(self, dotted_key: str, **kwargs: Any) -> str:
        template = self.get(dotted_key)
        if template is None:
            raise KeyError(f"Prompt template '{dotted_key}' not found in {self.config_path}")
        return template.format(**kwargs)


class StateStore:
    """Simple JSON-backed store used when MongoDB is not configured."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                try:
                    return json.load(fh)
                except json.JSONDecodeError:
                    logger.warning("State file corrupt; starting with empty state")
        return {"collections": {}, "batches": {}}

    def _persist(self) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(self.state, fh, ensure_ascii=False, indent=2)
        tmp_path.replace(self.path)

    # ------------------------------------------------------------------
    def get_collection(self, name: str) -> List[Dict[str, Any]]:
        with self.lock:
            col = self.state.setdefault("collections", {}).setdefault(name, [])
            return deepcopy(col)

    def save_collection(self, name: str, docs: List[Dict[str, Any]]) -> None:
        with self.lock:
            self.state.setdefault("collections", {})[name] = deepcopy(docs)
            self._persist()

    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return deepcopy(self.state.setdefault("batches", {}).get(batch_id))

    def upsert_batch(self, batch_id: str, payload: Dict[str, Any]) -> None:
        with self.lock:
            self.state.setdefault("batches", {})[batch_id] = deepcopy(payload)
            self._persist()


# ---------------------------------------------------------------------------
# Collection adapters
# ---------------------------------------------------------------------------

class MongoCollectionAdapter:
    """Adapter around a pymongo collection to provide a uniform API."""

    def __init__(self, collection):
        self.collection = collection

    def insert_one(self, document: Dict[str, Any]):
        return self.collection.insert_one(document)

    def find_one(self, query: Dict[str, Any]):
        return self.collection.find_one(query)

    def update_one(self, query: Dict[str, Any], update: Dict[str, Any]):
        return self.collection.update_one(query, update)


class LocalCollectionAdapter:
    """Minimal Mongo-like collection backed by StateStore."""

    def __init__(self, state_store: StateStore, name: str):
        self.state_store = state_store
        self.name = name

    # --------------------------------------------------------------
    @staticmethod
    def _get_nested(document: Dict[str, Any], dotted_key: str) -> Any:
        parts = dotted_key.split(".")
        value: Any = document
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    @classmethod
    def _matches(cls, document: Dict[str, Any], query: Dict[str, Any]) -> bool:
        for key, value in query.items():
            actual = cls._get_nested(document, key)
            if actual != value:
                return False
        return True

    @staticmethod
    def _apply_update(document: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        updated = deepcopy(document)
        set_values = update.get("$set", {})
        for dotted_key, value in set_values.items():
            parts = dotted_key.split(".")
            target = updated
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        return updated

    # --------------------------------------------------------------
    def insert_one(self, document: Dict[str, Any]):
        docs = self.state_store.get_collection(self.name)
        doc = deepcopy(document)
        doc.setdefault("_id", uuid.uuid4().hex)
        docs.append(doc)
        self.state_store.save_collection(self.name, docs)
        return type("InsertOneResult", (), {"inserted_id": doc["_id"]})()

    def find_one(self, query: Dict[str, Any]):
        docs = self.state_store.get_collection(self.name)
        for doc in docs:
            if self._matches(doc, query):
                return deepcopy(doc)
        return None

    def update_one(self, query: Dict[str, Any], update: Dict[str, Any]):
        docs = self.state_store.get_collection(self.name)
        matched = 0
        modified = 0
        for idx, doc in enumerate(docs):
            if self._matches(doc, query):
                matched += 1
                docs[idx] = self._apply_update(doc, update)
                modified += 1
        self.state_store.save_collection(self.name, docs)
        return type("UpdateResult", (), {"matched_count": matched, "modified_count": modified})()


# ---------------------------------------------------------------------------
# Local storage helpers
# ---------------------------------------------------------------------------

class LocalJSONLStore:
    """Utility to maintain JSONL payloads on the local filesystem."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, filename: str, record: Dict[str, Any]) -> Path:
        path = self.base_path / filename
        with self.lock:
            with path.open("a+", encoding="utf-8") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                fh.seek(0, os.SEEK_END)
                if fh.tell() > 0:
                    fh.write("\n")
                fh.write(json.dumps(record, ensure_ascii=False))
                fcntl.flock(fh, fcntl.LOCK_UN)
        return path

    def overwrite(self, filename: str, records: Iterable[Dict[str, Any]]) -> Path:
        path = self.base_path / filename
        with self.lock, path.open("w", encoding="utf-8") as fh:
            for idx, record in enumerate(records):
                if idx:
                    fh.write("\n")
                fh.write(json.dumps(record, ensure_ascii=False))
        return path

    def read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not path.exists():
            return records
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSONL line in %s", path)
        return records


class LocalPublisher:
    """Minimal Pub/Sub-like publisher writing payloads to JSONL topics."""

    def __init__(self, base_path: Path):
        self.jsonl_store = LocalJSONLStore(base_path)

    def publish(self, topic: str, data: bytes) -> str:
        message_id = uuid.uuid4().hex
        payload = {"message_id": message_id, "topic": topic, "data": data.decode("utf-8")}
        filename = f"{topic}.jsonl"
        self.jsonl_store.append(filename, payload)
        return message_id


# ---------------------------------------------------------------------------
# Docstribe Orchestrator implementation
# ---------------------------------------------------------------------------


class DocstribeOrchestrator:
    expected_keys = [
        "next_visit_action",
        "purpose_of_visit",
        "consultations",
        "monitoring_tests",
        "secondary_complications",
        "department",
        "screenings",
        "disease_name",
        "likeliness",
        "visit_date",
        "simple_discharge_summarization",
        "simplified_discharge_summary",
    ]

    opd_expected_keys = [
        "admission_evidence",
        "chronic_condition",
        "clinical_problem_representation",
        "clinical_problem_representation_summary",
        "impact_on_health",
        "differential_diagnoses",
        "department",
        "disease_name",
        "risk_score",
        "evidence",
        "primary_goal",
        "consultations",
        "lab_screenings",
        "visit_date",
        "secondary_complications",
    ]

    def __init__(
        self,
        prompt_config_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        debug: bool = False,
    ) -> None:
        self.debug = debug
        base_dir = Path.cwd()
        self.data_dir = data_dir or (base_dir / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = prompt_config_path or (base_dir / "prompt_config.yaml")
        self.prompt_manager = PromptManager(prompt_path)

        self.state_store = StateStore(self.data_dir / "state.json")
        self.jsonl_store = LocalJSONLStore(self.data_dir / "jsonl")
        self.publisher = LocalPublisher(self.data_dir / "pubsub")

        mongo_uri = os.getenv("DOCSTRIBE_MONGODB_URI")
        if not mongo_uri and agent_cfg is not None:
            mongo_uri = getattr(agent_cfg, "MONGODB_URI", None)
        self.mongo_client = None
        if mongo_uri and MongoClient is not None:
            self.mongo_client = MongoClient(mongo_uri, server_api=ServerApi("1"))
            db_name = os.getenv("DOCSTRIBE_DB", getattr(agent_cfg, "LLM_ORCHESTRATOR_DB", "docstribe"))
            self.mongo_database = self.mongo_client[db_name]
        else:
            self.mongo_database = None
            if mongo_uri and MongoClient is None:
                logger.warning("pymongo not installed; falling back to local state store")
            else:
                logger.info("MongoDB URI not provided; using local state store")

        self.collections = self._init_collections()

        self.azure_endpoint = os.getenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://docstribeazureaifoundry.cognitiveservices.azure.com/",
        )
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "Docstribe-o3")

        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not azure_api_key:
            fallback_key = os.getenv("OPENAI_API_KEY")
            if not fallback_key and agent_cfg is not None:
                fallback_key = getattr(agent_cfg, "OPENAI_API_KEY", None)
            azure_api_key = fallback_key

        if not azure_api_key:
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY (or OPENAI_API_KEY as fallback) must be configured to use the orchestrator"
            )

        self.openai_client = AzureOpenAI(
            api_key=azure_api_key,
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint,
        )

        self.project_settings = {
            "project_id": os.getenv("DOCSTRIBE_PROJECT_ID", "local-docstribe"),
            "topic_id": os.getenv("DOCSTRIBE_TOPIC_ID", "pdcm-payload"),
            "opd_topic_id": os.getenv("DOCSTRIBE_OPD_TOPIC_ID", "opd-payload"),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _select_reasoning_llm(alias: str):
        mapping = {
            "groq": reasoning_llm,
            "openai": reasoning_llm_openai,
            "deepseek": reasoning_llm_v2,
        }
        llm = mapping.get(alias.lower(), reasoning_llm)
        if llm is None:
            raise RuntimeError(
                "Requested reasoning model is not available. Ensure the corresponding API key is configured."
            )
        return llm

    def _azure_batch_url(self) -> str:
        return (
            f"/openai/deployments/{self.azure_deployment}/chat/completions"
            f"?api-version={self.azure_api_version}"
        )

    # ------------------------------------------------------------------
    def _init_collections(self) -> Dict[str, Any]:
        names = {
            "pdcm_collection": getattr(agent_cfg, "PDCM_COLLECTION", "pdcm"),
            "batch_collection": getattr(agent_cfg, "PDCM_BATCH_COLLECTION", "pdcm_batches"),
            "opd_collection": getattr(agent_cfg, "OPD_COLLECTION", "opd"),
            "opd_batch_collection": getattr(agent_cfg, "OPD_BATCH_COLLECTION", "opd_batches"),
            "opd_initial_collection": getattr(agent_cfg, "OPD_INITIAL_COLLECTION", "opd_initial"),
            "pdcm_initial_collection": getattr(agent_cfg, "PDCM_INITIAL_COLLECTION", "pdcm_initial"),
        }

        collections: Dict[str, Any] = {}
        for alias, collection_name in names.items():
            if self.mongo_database is not None:
                collections[alias] = MongoCollectionAdapter(self.mongo_database[collection_name])
            else:
                collections[alias] = LocalCollectionAdapter(self.state_store, collection_name)
        return collections

    # ------------------------------------------------------------------
    @staticmethod
    def set_cors_headers(response: Response) -> Response:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, x-http-method-override, access-control-request-method, "
            "Authorization, access-control-allow-headers, access-control-max-age, "
            "access-control-allow-methods, access-control-allow-origin"
        )
        response.headers["Access-Control-Max-Age"] = "86400"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    @staticmethod
    def _options_ok() -> Response:
        return DocstribeOrchestrator.set_cors_headers(jsonify(success=True))

    # ------------------------------------------------------------------
    @staticmethod
    def correct_key_typo(key: str, expected_keys: List[str]) -> str:
        correct_key, score = process.extractOne(key, expected_keys)
        if score > 93:
            return correct_key
        return key

    def correct_json_keys(self, data: Dict[str, Any], expected_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        expected = expected_keys or self.expected_keys
        corrected: Dict[str, Any] = {}
        for key, value in data.items():
            corrected_value = (
                self.correct_json_keys(value, expected) if isinstance(value, dict) else value
            )
            corrected_key = self.correct_key_typo(key, expected)
            corrected[corrected_key] = corrected_value
        return corrected

    @staticmethod
    def check_same_elements(list1: List[str], list2: List[str]) -> bool:
        return set(list1) == set(list2)

    def standardize_json(self, json_payload: Dict[str, Any]) -> Dict[str, Any]:
        comprehensive_care_package = json_payload.get("comprehensive_care_package", {}) or {}
        if comprehensive_care_package:
            annual_consultations = comprehensive_care_package.get("annual_consultations", [])
            for ac in annual_consultations:
                keys = list(ac.keys())
                for key in keys:
                    if key != "department":
                        ac["department"] = ac[key]
                        ac.pop(key)
                        break

            annual_screenings = comprehensive_care_package.get("annual_screenings", [])
            for asc in annual_screenings:
                keys = list(asc.keys())
                for key in keys:
                    if key not in {"frequency", "screening_name"}:
                        asc["frequency"] = ""
                        asc["screening_name"] = ""
                        break

        next_visit_action = json_payload.get("next_visit_action", {}) or {}
        secondary_complications = next_visit_action.get("secondary_complications", [])
        filtered_secondary = [
            sc
            for sc in secondary_complications
            if self.check_same_elements(
                ["department", "disease_name", "likeliness", "screenings"], list(sc.keys())
            )
        ]
        next_visit_action["secondary_complications"] = filtered_secondary
        json_payload["next_visit_action"] = next_visit_action

        return self.correct_json_keys(json_payload)

    # ------------------------------------------------------------------
    def upload_jsonl_file(self, file_path: str) -> str:
        with open(file_path, "rb") as fh:
            batch_input_file = self.openai_client.files.create(file=fh, purpose="batch")
        return batch_input_file.id

    def create_batch(self, batch_input_file_id: str) -> Any:
        return self.openai_client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/chat/completions",
            completion_window="24h",
            model=self.azure_deployment,
            metadata={"description": f"batch job for file id -{batch_input_file_id}"},
        )

    # ------------------------------------------------------------------
    def _read_jsonl_file(self, file_path: Path, mongo_collection) -> None:
        json_data = self.jsonl_store.read_jsonl(file_path)
        request_ids = [payload.get("custom_id") for payload in json_data]

        bodies = []
        for entry in json_data:
            response = entry.get("response", {})
            body = response.get("body", {})
            content = body.get("choices", [{}])[0].get("message", {}).get("content")
            if content:
                bodies.append(content)
        bodies_json: List[Any] = []
        for body in bodies:
            start_idx = body.find("{")
            end_idx = body.rfind("}")
            if start_idx >= 0 and end_idx >= 0:
                fragment = body[start_idx : end_idx + 1]
            else:
                fragment = body
            try:
                bodies_json.append(json.loads(fragment))
            except json.JSONDecodeError:
                bodies_json.append(fragment)

        for idx, body in enumerate(bodies_json):
            try:
                body["request_id"] = request_ids[idx]
                patient_id = request_ids[idx].split("_")[1]
                mongo_collection.update_one(
                    {"patient_details.patient_id": patient_id},
                    {"$set": {"status": "completed", "event_response": body}},
                )
            except Exception:  # pragma: no cover - defensive: keep parity with legacy behaviour
                logger.debug("Did not update document for request %s", request_ids[idx])

    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_to_string(obj: Any) -> str:
        if isinstance(obj, dict):
            return ", ".join(f"{k}: {DocstribeOrchestrator._flatten_to_string(v)}" for k, v in obj.items())
        if isinstance(obj, list):
            return ", ".join(DocstribeOrchestrator._flatten_to_string(item) for item in obj)
        return str(obj)

    def _fix_discharge_summary(self, discharge_summary: Dict[str, Any]) -> Dict[str, Any]:
        keys_to_check = [
            "medical_history",
            "key_findings",
            "chief_complaints",
            "procedures_done",
            "follow_up",
            "medications",
        ]
        for key in keys_to_check:
            value = discharge_summary.get(key)
            if value is not None and not isinstance(value, list):
                discharge_summary[key] = [value]

        procedures = discharge_summary.get("procedures_done", [])
        for idx, procedure in enumerate(procedures):
            if isinstance(procedure, dict):
                p_name = procedure.get("procedure", "")
                findings = procedure.get("findings", "")
                if not isinstance(p_name, str):
                    p_name = self._flatten_to_string(p_name)
                if not isinstance(findings, str):
                    findings = self._flatten_to_string(findings)
                procedures[idx] = {"procedure": p_name, "findings": findings}
            elif isinstance(procedure, str):
                procedures[idx] = {"procedure": procedure, "findings": ""}
        discharge_summary["procedures_done"] = procedures

        for key in keys_to_check:
            if key == "procedures_done":
                continue
            value = discharge_summary.get(key)
            if value is None:
                continue
            discharge_summary[key] = [
                item if isinstance(item, str) else self._flatten_to_string(item) for item in value
            ]
        return discharge_summary

    # ------------------------------------------------------------------
    def _sort_visits_by_date(self, data: List[Dict[str, Any]], dict_key: str = "date") -> List[Dict[str, Any]]:
        def _parse(date_str: str) -> dt.datetime:
            return dt.datetime.strptime(date_str, "%d-%b-%Y")

        return sorted(data, key=lambda x: _parse(x[dict_key]), reverse=True)

    # ------------------------------------------------------------------
    # Route handlers (public methods)
    # ------------------------------------------------------------------

    def handle_check_batch_status(self, payload: Dict[str, Any]):
        batch_id = payload["batch_id"]
        resp = self.openai_client.batches.retrieve(batch_id)
        batch_dict = {"batch_id": resp.id, "status": resp.status}
        if resp.status == "completed":
            batch_dict["output_file_id"] = resp.output_file_id
            update_payload = {"status": "completed", "output_file_id": resp.output_file_id}
            result = self.collections["batch_collection"].update_one(
                {"batch_id": batch_id},
                {"$set": update_payload},
            )
            if getattr(result, "matched_count", 0) == 0:
                self.collections["opd_batch_collection"].update_one(
                    {"batch_id": batch_id},
                    {"$set": update_payload},
                )
            self.state_store.upsert_batch(batch_id, batch_dict)
        return batch_dict

    def handle_upload_batch(self, payload: Dict[str, Any]):
        data_type = payload.get("type", "undefined")
        file_url = payload["file_url"]
        tmp_path = Path(file_url)
        if not tmp_path.exists():
            raise FileNotFoundError(f"Batch file not found at {file_url}")

        temp_copy = self.data_dir / ("pdcm_jsonl_for_processing.jsonl" if data_type == "PDCM" else "opd_jsonl_for_processing.jsonl")
        temp_copy.write_bytes(tmp_path.read_bytes())

        batch_file_id = self.upload_jsonl_file(str(temp_copy))
        resp = self.create_batch(batch_file_id)
        batch_id = resp.id
        batch_resp = {
            "batch_id": batch_id,
            "batch_file_id": batch_file_id,
            "status": "processing",
        }

        if data_type == "PDCM":
            insert_result = self.collections["batch_collection"].insert_one(batch_resp.copy())
        elif data_type == "OPD":
            insert_result = self.collections["opd_batch_collection"].insert_one(batch_resp.copy())
        else:
            raise ValueError("Please provide valid type")

        batch_resp["_id"] = str(getattr(insert_result, "inserted_id", ""))
        return batch_resp

    def handle_retrieve_results(self, payload: Dict[str, Any]):
        file_type = payload["file_type"]
        file_id = payload["file_id"]

        content = self.openai_client.files.content(file_id)
        tmp_file = self.data_dir / f"output_{file_id}.jsonl"
        with tmp_file.open("wb") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(content.content)
            fcntl.flock(fh, fcntl.LOCK_UN)

        if file_type == "PDCM":
            collection = self.collections["pdcm_collection"]
        else:
            collection = self.collections["opd_collection"]

        self._read_jsonl_file(tmp_file, collection)
        tmp_file.unlink(missing_ok=True)
        return {"status": "success"}

    # ------------------------------------------------------------------
    # Many subsequent handlers mirror the legacy implementation but
    # operate through collection adapters and local utilities.
    # ------------------------------------------------------------------

    @staticmethod
    def _simplify_cohorts(cohorts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        simplified: Dict[str, Dict[str, Any]] = {}
        for spec, info in cohorts.items():
            primary_val = info.get("primary_gold_standard_disease") or "Others"
            secondary_val = info.get("secondary_gold_standard_disease") or "Others"
            simplified[spec] = {"primary": primary_val, "secondary": secondary_val}
        return simplified

    def handle_view_opd_output(self, payload: Dict[str, Any]):
        patient_ids = payload["patient_ids"]
        array_response: List[Dict[str, Any]] = []
        opd_collection = self.collections["opd_collection"]

        for patient in patient_ids:
            patient_document = opd_collection.find_one(
                {"patient_details.patient_id": patient, "status": "completed"}
            )
            if not patient_document:
                array_response.append({})
                continue

            radiology_findings = patient_document.get("radiology_findings", {})
            visits = patient_document.get("abnormalities_identified") or []
            if not visits:
                array_response.append({})
                continue

            visit_latest = visits[0]
            response_array: List[Dict[str, Any]] = []
            prescription_flag = any(
                visit_latest.get(k, "") != ""
                for k in [
                    "medications",
                    "investigations",
                    "diagnosis_advised_list",
                    "follow_up_date",
                    "ip_advised",
                    "clinical_note",
                ]
            )

            tests = visit_latest.get("tests", [])
            for test in tests:
                classification = test.get("classification", "")
                test["classification"] = classification or ""
                if classification and classification.lower() != "normal":
                    test["isAbnormal"] = True
                else:
                    test["isAbnormal"] = not bool(classification)
                response_array.append(test)

            abnormalities_identified = response_array
            resp_content = patient_document.get("event_response", {})
            clinical_condition = patient_document.get("clinical_condition", {})
            resp_content = deepcopy(resp_content)
            resp_content["abnormalities_identified"] = abnormalities_identified
            resp_content.pop("clinical_condition", None)
            resp_content["doctors_notes"] = clinical_condition
            resp_content["overall_risk_score"] = (
                resp_content.get("risk_stratification", {}).get("overall_risk_score", 0)
            )

            further_management = resp_content.get("further_management", {})
            management_advice = further_management.get("management_advice", "")
            admission_evidence = further_management.get("evidence", "")
            overall_admission_probability = "High" if management_advice == "immediate IPD admission" else "Low"

            resp_content["overall_admission_probability"] = overall_admission_probability
            resp_content["admission_evidence"] = admission_evidence
            resp_content["primary_cohort"] = resp_content.get("primary_cohort", {})
            resp_content["secondary_cohort"] = resp_content.get("secondary_cohort", {})
            resp_content["care_gaps_present"] = resp_content.get("care_gaps_present", {})
            resp_content["status"] = "completed"
            resp_content["version"] = "4"
            resp_content["clinical_summary"] = patient_document.get("agentic_summary", {})
            resp_content.setdefault("clinical_summary", {}).setdefault(
                "clinical_assessment", {}
            )["radiology_findings"] = radiology_findings

            if opd_parse_json is None:
                raise RuntimeError("opd_pydantic_validator is not available in this environment")

            try:
                parsed_resp = opd_parse_json(resp_content)  # type: ignore[misc]
                if not parsed_resp:
                    parsed_resp = {"status": "failed", "request_id": f"request_{patient}"}
                    opd_collection.update_one(
                        {"patient_details.patient_id": patient},
                        {"$set": {"status": "pydantic_failed"}},
                    )
                simple_primary = parsed_resp.get("primary_cohort", {})
                simple_secondary = parsed_resp.get("secondary_cohort", {})
                parsed_resp["primary_cohort"] = simple_primary
                parsed_resp["secondary_cohort"] = simple_secondary
                opd_collection.update_one(
                    {"patient_details.patient_id": patient},
                    {"$set": {
                        "primary_cohort": simple_primary,
                        "secondary_cohort": simple_secondary,
                        "cohort_calculated": True,
                    }},
                )
                array_response.append(parsed_resp)
            except Exception:
                logger.exception("JSON doesn't fit the pydantic for OPD")
                opd_collection.update_one(
                    {"patient_details.patient_id": patient},
                    {"$set": {"status": "pydantic_failed"}},
                )
                array_response.append({"status": "failed", "request_id": f"request_{patient}"})

        return {"responses": array_response}

    # ------------------------------------------------------------------
    def handle_view_pdcm_output(self, payload: Dict[str, Any]):
        if request.method == "OPTIONS":  # rely on Flask request context
            return self._options_ok()

        patient_ids = payload["patient_ids"]
        responses: List[Dict[str, Any]] = []
        pdcm_collection = self.collections["pdcm_collection"]

        for patient_id in patient_ids:
            resp = pdcm_collection.find_one(
                {"patient_details.patient_id": patient_id, "status": "completed"}
            )
            if not resp:
                responses.append({"status": "failed", "request_id": f"request_{patient_id}", "version": "2"})
                continue

            event_response = deepcopy(resp.get("event_response", {}))
            visits = resp.get("visits", [])
            if not visits:
                responses.append({"status": "failed", "request_id": f"request_{patient_id}", "version": "2"})
                continue

            discharge_summary = deepcopy(visits[0].get("discharge_summary", {}))
            discharge_summary = self._fix_discharge_summary(discharge_summary)

            primary_procedure_name = discharge_summary.get("primary_procedure_name")
            simplification_url = getattr(agent_cfg, "SIMPLIFICATION_URL", "") if agent_cfg else ""
            if simplification_url and primary_procedure_name:
                try:
                    resp_simple = requests.post(
                        simplification_url,
                        json={"procedure_name": primary_procedure_name},
                        timeout=30,
                    )
                    if resp_simple.status_code == 200:
                        resp_content = resp_simple.json()
                        discharge_summary["primary_procedure_name"] = resp_content.get(
                            "simple_procedure_name", primary_procedure_name
                        )
                        discharge_summary["procedure_tags"] = resp_content.get("tags", [])
                except Exception:
                    logger.debug("Procedure simplification service unavailable")

            event_response["discharge_summary"] = discharge_summary
            if parse_json is None:
                raise RuntimeError("pdcm_pydantic_validator is not available in this environment")

            try:
                parsed = parse_json(event_response)  # type: ignore[misc]
                parsed["status"] = "completed"
                parsed["request_id"] = f"request_{patient_id}"
                parsed["version"] = "2"
                responses.append(parsed)
            except Exception:
                logger.exception("JSON doesn't fit PDCM pydantic")
                pdcm_collection.update_one(
                    {"patient_details.patient_id": patient_id},
                    {"$set": {"status": "pydantic_failed"}},
                )

        return {"responses": responses}

    # ------------------------------------------------------------------
    def handle_process_cohort(self, payload: Dict[str, Any]):
        patient_id = payload.get("patient_id")
        visit_type = payload.get("visit_type")
        if visit_type == "OPD":
            resp = self.collections["opd_collection"].find_one(
                {"patient_details.patient_id": patient_id}
            )
        else:
            resp = self.collections["pdcm_collection"].find_one(
                {"patient_details.patient_id": patient_id}
            )
        resp_content = cohort_generation_agent(resp)  # type: ignore[attr-defined]
        return resp_content

    # ------------------------------------------------------------------
    def handle_process_opd_message(self, payload: Dict[str, Any]):
        current_date = dt.datetime.now().strftime("%d-%b-%Y")
        if request.method == "OPTIONS":
            return self._options_ok()

        prompt_code = payload.get("prompt_code", "")
        data_payload = payload.get("data_payload", {})
        visits_payload = data_payload.get("visits", [])

        gender = data_payload.get("gender", "Male")
        if gender.lower() == "male":
            gender = "Male"
        elif gender.lower() == "female":
            gender = "Female"

        diagnostic_summary: Dict[str, Any] = {}
        tests_detected = 0
        prescription_flag = False

        for visit in visits_payload:
            is_processed = visit.get("is_processed", False)
            date_of_opd = visit.get("date", current_date)
            date_obj = dt.datetime.strptime(date_of_opd, "%d/%m/%Y")
            visit["date"] = date_obj.strftime("%d-%b-%Y")
            for key in [
                "medications",
                "investigations",
                "diagnosis_advised_list",
                "follow_up_date",
                "ip_advised",
            ]:
                value = visit.get(key, "")
                if value:
                    prescription_flag = True
                    if key == "follow_up_date":
                        date_obj_v = dt.datetime.strptime(value, "%d/%m/%Y")
                        visit["follow_up_date"] = date_obj_v.strftime("%d-%b-%Y")

            test_list = visit.get("tests", [])
            if not is_processed and test_list:
                tests_detected += 1
                test_list = detect_op_abnormalities(test_list, gender)  # type: ignore[attr-defined]
            visit["tests"] = test_list

            radiology_findings = visit.get("radiology_findings", {})
            if radiology_findings and not is_processed:
                pdf_url = radiology_findings.get("pdf_url", "")
                if pdf_url:
                    try:
                        diagnostic_summary = summarize_diagnostics(pdf_url)  # type: ignore[attr-defined]
                    except Exception:
                        logger.debug("Radiology summarization failed", exc_info=True)

        if prompt_code == "openai_gen_op_1":
            resp_content = {
                "abnormalities_identified": visits_payload,
                "patient_details": {
                    "age": data_payload.get("age"),
                    "gender": gender,
                    "patient_id": data_payload.get("patient_id"),
                },
                "radiology_findings": diagnostic_summary,
            }
            if tests_detected > 0 or prescription_flag:
                resp_content["status"] = "pending"
                resp_content["current_date"] = current_date
            else:
                resp_content["status"] = "completed"
                resp_content["event_response"] = {}
                resp_content["current_date"] = current_date
            resp_to_summarize = deepcopy(resp_content)
            resp_to_summarize["visits"] = resp_to_summarize.pop("abnormalities_identified")
            summary_resp = summarize_lab_data_agent(resp_to_summarize)  # type: ignore[attr-defined]
            resp_content["clinical_assessment"] = summary_resp.get("clinical_assessment", {})
            resp_content["agentic_summary"] = summary_resp
            resp_content["radiology_findings"] = resp_content["clinical_assessment"].pop(
                "radiology_findings", {}
            )
            resp_content["abnormalities_identified"] = resp_to_summarize.get("visits", visits_payload)
            insert_result = self.collections["opd_collection"].insert_one(resp_content)
            resp_content["_id"] = str(getattr(insert_result, "inserted_id", ""))
            return resp_content

        return {
            "abnormalities_identified": visits_payload,
            "message": "please pass a valid prompt code",
        }

    # ------------------------------------------------------------------
    def handle_process_pdcm_message(self, payload: Dict[str, Any]):
        if request.method == "OPTIONS":
            return self._options_ok()

        prompt_code = payload.get("prompt_code", "")
        data_payload = payload.get("data_payload", {})
        gender = data_payload.get("gender", "Male")
        age = data_payload.get("age")
        patient_id = data_payload.get("patient_id")
        current_date = dt.datetime.now().strftime("%d-%b-%Y")
        visits = data_payload.get("visits", [])

        for visit in visits:
            is_processed = visit.get("is_processed", False)
            discharge_date = visit.get("discharge_date", "")
            if discharge_date:
                discharge_date_obj = dt.datetime.strptime(discharge_date, "%d/%m/%Y")
                visit["discharge_date"] = discharge_date_obj.strftime("%d-%b-%Y")
            date_of_visit = visit.get("date", current_date)
            if date_of_visit:
                date_obj = dt.datetime.strptime(date_of_visit, "%d/%m/%Y")
                visit["date"] = date_obj.strftime("%d-%b-%Y")

            test_list = visit.get("tests", [])
            if not is_processed:
                test_list = detect_op_abnormalities(test_list, gender)  # type: ignore[attr-defined]
            visit["tests"] = test_list

            if not is_processed:
                discharge_summary = visit.get("discharge_summary", "")
                discharge_date_fmt = visit.get("discharge_date", "")
                if discharge_summary:
                    visit["discharge_summary"] = summarize_agent(  # type: ignore[attr-defined]
                        discharge_summary,
                        discharge_date_fmt,
                    )
            else:
                if visit.get("discharge_summary") and visit.get("visit_type", "").lower() == "ip":
                    visit["discharge_summary"] = visit.get("discharge_summary")

        resp_content = {
            "visits": visits,
            "patient_details": {
                "age": age,
                "gender": gender,
                "patient_id": patient_id,
            },
            "status": "pending",
            "current_date": current_date,
        }

        insert_result = self.collections["pdcm_collection"].insert_one(resp_content)
        resp_content["_id"] = str(getattr(insert_result, "inserted_id", ""))
        return {"visits": resp_content["visits"]}

    # ------------------------------------------------------------------
    def handle_summarize_discharge(self, payload: Dict[str, Any]):
        medical_text = payload["medical_text"]
        discharge_date = payload.get("discharge_date", "")
        model_alias = payload.get("model", "groq")
        llm = self._select_reasoning_llm(model_alias)
        return summarize_agent(medical_text, discharge_date, model=llm)

    def handle_summarize_opd_data(self, payload: Dict[str, Any]):
        patient_data = payload["patient_data"]
        return summarize_lab_data_agent(patient_data)

    def handle_summarize_diagnostics(self, payload: Dict[str, Any]):
        pdf_url = payload["pdf_url"]
        return summarize_diagnostics(pdf_url)

    # ------------------------------------------------------------------
    def append_to_opd_blob(self, data: Dict[str, Any]) -> None:
        self.jsonl_store.append("continental_opd_jsonl_file.jsonl", data)

    def append_to_pdcm_blob(self, data: Dict[str, Any]) -> None:
        self.jsonl_store.append("ipd_jsonl_batch_file.jsonl", data)

    # ------------------------------------------------------------------
    def opd_openai_prepare_batch_v2(
        self,
        patient_id: str,
        patient_age: Any,
        patient_gender: str,
        current_date: str,
        agentic_summary: Dict[str, Any],
        clinical_assessment: Dict[str, Any],
        latest_visit_date: str,
        radiology_findings: Any = "",
    ) -> Dict[str, Any]:
        sys_cohort_prompt = self.prompt_manager.get("master_prompts.sys_cohort_prompt", "")
        output_format = self.prompt_manager.get(
            "templates.opd_output_json_format_medications_v2", "{}"
        )

        context_segments = [
            f"You are assessing an Indian {patient_gender} patient aged {patient_age} as of {current_date}.",
            f"Latest visit date: {latest_visit_date}.",
        ]
        if agentic_summary:
            context_segments.append(f"Agentic summary: {agentic_summary}.")
        if clinical_assessment:
            context_segments.append(f"Clinical assessment: {clinical_assessment}.")
        if radiology_findings:
            context_segments.append(f"Radiology findings: {radiology_findings}.")
        context = " ".join(context_segments)

        system_prompt = self.prompt_manager.render(
            "prompts.opd_batch_v2.system",
            context=context,
            sys_cohort_prompt=sys_cohort_prompt,
        )
        user_prompt = self.prompt_manager.render(
            "prompts.opd_batch_v2.user",
            output_format=output_format,
        )

        payload = {
            "model": self.azure_deployment,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        return {
            "custom_id": f"request_{patient_id}",
            "method": "POST",
            "url": self._azure_batch_url(),
            "body": payload,
        }

    # ------------------------------------------------------------------
    def handle_collect_opd_pending_requests(self, payload: Dict[str, Any]):
        if request.method == "OPTIONS":
            return self._options_ok()

        responses = payload.get("responses", [])
        opd_collection = self.collections["opd_collection"]
        for resp in responses:
            patient_details = resp.get("patient_details", {})
            patient_id = patient_details.get("patient_id")
            patient_age = patient_details.get("age")
            patient_gender = patient_details.get("gender")

            patient_doc = opd_collection.find_one({"patient_details.patient_id": patient_id})
            if patient_doc and patient_doc.get("status") == "processing":
                continue

            current_date = dt.datetime.now().strftime("%d-%b-%Y")
            agentic_summary = resp.get("agentic_summary", {})
            clinical_assessment = resp.get("clinical_assessment", {})
            radiology_findings = resp.get("radiology_findings", {})
            visits = self._sort_visits_by_date(resp.get("abnormalities_identified", []))
            latest_visit_date = visits[0].get("date", "") if visits else ""
            opd_payload = self.opd_openai_prepare_batch_v2(
                patient_id,
                patient_age,
                patient_gender,
                current_date,
                agentic_summary,
                clinical_assessment,
                latest_visit_date,
                radiology_findings,
            )
            self.append_to_opd_blob(opd_payload)
            opd_collection.update_one(
                {"patient_details.patient_id": patient_id},
                {"$set": {"status": "processing"}},
            )
        return {"status": "success"}

    # ------------------------------------------------------------------
    def opd_openai_prepare_batch(
        self,
        patient_id: str,
        patient_age: Any,
        patient_gender: str,
        current_date: str,
        diagnoses_json: List[Dict[str, Any]],
        clinical_condition: Dict[str, Any],
        radiology_findings: Any = "",
    ) -> Dict[str, Any]:
        output_format_medications = self.prompt_manager.get(
            "templates.opd_output_json_format_medications", "{}"
        )
        output_format_default = self.prompt_manager.get(
            "templates.opd_output_json_format_v3", "{}"
        )
        sys_cohort_prompt = self.prompt_manager.get("master_prompts.sys_cohort_prompt", "")

        context_segments: List[str] = []
        prescription_flag = False
        prescription_content: Dict[str, Any] = {}

        if len(diagnoses_json) == 1:
            visit = diagnoses_json[0]
            opd_date = visit.get("date", current_date)
            tests = visit.get("tests", [])
            for key in [
                "medications",
                "investigations",
                "diagnosis_advised_list",
                "follow_up_date",
                "ip_advised",
                "clinical_note",
            ]:
                value = visit.get(key)
                if value:
                    prescription_content[key] = value
                    prescription_flag = True
            if prescription_flag and tests:
                context_segments.append(
                    f"Examining Indian {patient_gender} aged {patient_age} for visit {opd_date} "
                    f"with prescriptions {prescription_content} and tests {tests}."
                )
            elif prescription_flag:
                context_segments.append(
                    f"Examining Indian {patient_gender} aged {patient_age} for visit {opd_date} "
                    f"with prescriptions {prescription_content}."
                )
            else:
                context_segments.append(
                    f"Examining Indian {patient_gender} aged {patient_age} for visit {opd_date} "
                    f"with tests {tests}."
                )
        else:
            sorted_visits = self._sort_visits_by_date(diagnoses_json)
            visit_latest = sorted_visits[0]
            visit_prior = sorted_visits[1]
            latest_date = visit_latest.get("date", current_date)
            prior_date = visit_prior.get("date", current_date)
            latest_tests = visit_latest.get("tests", [])
            prior_tests = visit_prior.get("tests", [])
            for visit in sorted_visits[:2]:
                for key in [
                    "medications",
                    "investigations",
                    "diagnosis_advised_list",
                    "follow_up_date",
                    "ip_advised",
                ]:
                    value = visit.get(key)
                    if value:
                        prescription_content.setdefault(key, []).append(value)
                        prescription_flag = True
            if prescription_flag:
                context_segments.append(
                    f"Latest visit {latest_date} tests {latest_tests} with prescriptions {prescription_content}. "
                    f"Prior visit {prior_date} tests {prior_tests}."
                )
            else:
                context_segments.append(
                    f"Latest visit {latest_date} tests {latest_tests}. Prior visit {prior_date} tests {prior_tests}."
                )

        if radiology_findings:
            context_segments.append(f"Radiology findings: {radiology_findings}.")
        if clinical_condition:
            context_segments.append(f"Clinical condition: {clinical_condition}.")

        context = " ".join(context_segments)
        system_prompt = self.prompt_manager.render(
            "prompts.opd_batch_legacy.system",
            context=context,
            sys_cohort_prompt=sys_cohort_prompt,
        )
        user_prompt = self.prompt_manager.render(
            "prompts.opd_batch_legacy.user",
            output_format=output_format_medications if prescription_flag else output_format_default,
            current_date=current_date,
        )

        payload = {
            "model": self.azure_deployment,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        return {
            "custom_id": f"request_{patient_id}",
            "method": "POST",
            "url": self._azure_batch_url(),
            "body": payload,
        }

    # ------------------------------------------------------------------
    def ipd_openai_prepare_batch(
        self,
        patient_id: str,
        patient_age: Any,
        patient_gender: str,
        current_date: str,
        resp_content: Dict[str, Any],
        language: str = "english",
    ) -> Dict[str, Any]:
        visits = self._sort_visits_by_date(resp_content.get("visits", []), "date")
        number_of_visits = len(visits)
        care_plan_system_prompt = self.prompt_manager.get(
            "master_prompts.post_discharge_care_plan_system_prompt",
            "Generate a post discharge care plan.",
        )
        care_plan_user_prompt = self.prompt_manager.get(
            "master_prompts.post_discharge_care_plan_user_prompt",
            "Return JSON care plan.",
        )
        openai_output_format = self.prompt_manager.get(
            "templates.openai_gen_ip2_json_format",
            "{}",
        )

        if number_of_visits == 0:
            raise ValueError("No visits supplied for IPD batch")

        if number_of_visits == 1:
            visit_info = visits[0]
            discharge_summary = visit_info.get("discharge_summary", {})
            tests = visit_info.get("tests", [])
            discharge_date = visit_info.get("discharge_date", current_date)
            discharge_summary = discharge_summary or {}
            death_flag = discharge_summary.get("death_flag", "false")
            if death_flag == "true":
                return {}
            system_prompt = care_plan_system_prompt.format(
                patient_gender=patient_gender,
                patient_age=patient_age,
                tests=tests,
                discharge_summary=discharge_summary,
            )
            user_prompt = care_plan_user_prompt.format(
                OPENAI_GEN_IP2_JSON_FORMAT=openai_output_format,
                current_date=current_date,
                discharge_date=discharge_date,
                language=language,
            )
        else:
            visit_latest = visits[0]
            visit_old = visits[-1]
            discharge_summary = visit_old.get("discharge_summary", {})
            discharge_date = visit_old.get("discharge_date", "")
            death_flag = discharge_summary.get("death_flag", "false") if isinstance(discharge_summary, dict) else "false"
            if death_flag == "true":
                return {}
            visit_type_latest = visit_latest.get("visit_type", "")
            tests_latest = visit_latest.get("tests", [])
            tests_old = visit_old.get("tests", [])
            system_prompt = (
                f"You are an experienced strategist and medical professional assessing an Indian {patient_gender} "
                f"patient aged {patient_age}. Latest {visit_type_latest} visit on {visit_latest.get('date')} "
                f"tests {tests_latest}; prior discharge on {discharge_date} summary {discharge_summary} "
                f"and tests {tests_old}. Today is {current_date}."
            )
            user_prompt = (
                "Create a detailed post discharge care management plan in the following JSON format -"
                f"{openai_output_format}"
            )

        model_name = self.prompt_manager.get("models.ipd_batch_model", self.azure_deployment)
        payload = {
            "model": model_name,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        return {
            "custom_id": f"request_{patient_id}",
            "method": "POST",
            "url": self._azure_batch_url(),
            "body": payload,
        }

    # ------------------------------------------------------------------
    def handle_collect_pending_requests(self, payload: Dict[str, Any]):
        if request.method == "OPTIONS":
            return self._options_ok()

        responses = payload.get("responses", [])
        for resp in responses:
            patient_details = resp.get("patient_details", {})
            patient_id = patient_details.get("patient_id")
            patient_age = patient_details.get("age")
            patient_gender = patient_details.get("gender")
            language = resp.get("language", "english")
            visits = resp.get("visits", [])
            current_date = resp.get("current_date", dt.datetime.now().strftime("%d-%b-%Y"))

            status_doc = self.collections["pdcm_collection"].find_one(
                {"patient_details.patient_id": patient_id}
            )
            if status_doc and status_doc.get("status") == "processing":
                continue

            resp_content = {"visits": visits}
            try:
                jsonl_payload = self.ipd_openai_prepare_batch(
                    patient_id,
                    patient_age,
                    patient_gender,
                    current_date,
                    resp_content,
                    language,
                )
            except Exception as exc:
                logger.exception("Failed to prepare IPD batch", exc_info=True)
                self.collections["pdcm_collection"].update_one(
                    {"patient_details.patient_id": patient_id},
                    {"$set": {"status": str(exc)}},
                )
                continue

            if not jsonl_payload:
                self.collections["pdcm_collection"].update_one(
                    {"patient_details.patient_id": patient_id},
                    {"$set": {"status": "failed", "death_flag": True}},
                )
                continue

            self.append_to_pdcm_blob(jsonl_payload)
            self.collections["pdcm_collection"].update_one(
                {"patient_details.patient_id": patient_id},
                {"$set": {"status": "processing"}},
            )

        return {
            "status": "success",
            "file_url": str(self.data_dir / "jsonl" / "ipd_jsonl_batch_file.jsonl"),
        }

    # ------------------------------------------------------------------
    def handle_upload_opd_records(self, file_stream) -> Dict[str, Any]:
        df = pd.read_excel(file_stream)
        if "json_output" not in df.columns:
            raise ValueError("'json_output' column not found in the file")

        json_output_list = df["json_output"].tolist()
        topic = self.project_settings["opd_topic_id"]
        for json_output in json_output_list:
            message_id = self.publisher.publish(topic, json_output.encode("utf-8"))
            logger.debug("Published OPD message %s", message_id)
        return {"message": "File received and processed successfully", "json_output": json_output_list}

    # ------------------------------------------------------------------
    def handle_process_op(self, payload: Dict[str, Any]):
        if request.method == "OPTIONS":
            return self._options_ok()

        json_output = json.dumps(payload)
        self.collections["opd_initial_collection"].insert_one(payload)
        topic = self.project_settings["opd_topic_id"]
        message_id = self.publisher.publish(topic, json_output.encode("utf-8"))
        return {"status": "success", "message": f"Published to topic with {message_id}"}

    def handle_process_ip(self, payload: Dict[str, Any]):
        if request.method == "OPTIONS":
            return self._options_ok()

        json_output = json.dumps(payload)
        self.collections["pdcm_initial_collection"].insert_one(payload)
        topic = self.project_settings["topic_id"]
        message_id = self.publisher.publish(topic, json_output.encode("utf-8"))
        return {"status": "success", "message": f"Published to topic with {message_id}"}

    # ------------------------------------------------------------------
    def handle_upload_pdcm_records(self, file_stream) -> Dict[str, Any]:
        df = pd.read_excel(file_stream)
        if "json_output" not in df.columns:
            raise ValueError("'json_output' column not found in the file")
        json_output_list = df["json_output"].tolist()
        topic = self.project_settings["topic_id"]
        for json_output in json_output_list:
            self.publisher.publish(topic, json_output.encode("utf-8"))
        return {"message": "File received and processed successfully", "json_output": json_output_list}


# ---------------------------------------------------------------------------
# Flask application setup
# ---------------------------------------------------------------------------


def create_app(debug: bool = False) -> Flask:
    orchestrator = DocstribeOrchestrator(debug=debug)
    app = Flask(__name__)

    @app.after_request
    def apply_cors_headers(response):  # pragma: no cover - Flask hook
        return orchestrator.set_cors_headers(response)

    @app.route("/check_batch_status", methods=["POST", "OPTIONS"])
    def check_batch_status():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_check_batch_status(request.json)
        return jsonify(result)

    @app.route("/upload_batch", methods=["POST", "OPTIONS"])
    def upload_batch():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_upload_batch(request.json)
        return jsonify(result)

    @app.route("/retrieve_results", methods=["POST", "OPTIONS"])
    def retrieve_results():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_retrieve_results(request.json)
        return jsonify(result)

    @app.route("/view_opd_output", methods=["POST", "OPTIONS"])
    def view_opd_output():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_view_opd_output(request.json)
        return jsonify(result)

    @app.route("/view_pdcm_output", methods=["POST", "OPTIONS"])
    def view_pdcm_output():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_view_pdcm_output(request.json)
        return jsonify(result)

    @app.route("/process_cohort", methods=["POST", "OPTIONS"])
    def process_cohort():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_process_cohort(request.json)
        return jsonify(result)

    @app.route("/process_opd_message", methods=["POST", "OPTIONS"])
    def process_opd_message():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_process_opd_message(request.json)
        return jsonify(result)

    @app.route("/process_pdcm_message", methods=["POST", "OPTIONS"])
    def process_pdcm_message():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_process_pdcm_message(request.json)
        return jsonify(result)

    @app.route("/summaries/discharge", methods=["POST", "OPTIONS"])
    def summarize_discharge():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_summarize_discharge(request.json)
        return jsonify(result)

    @app.route("/summaries/opd", methods=["POST", "OPTIONS"])
    def summarize_opd():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_summarize_opd_data(request.json)
        return jsonify(result)

    @app.route("/summaries/diagnostics", methods=["POST", "OPTIONS"])
    def summarize_diagnostics_route():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_summarize_diagnostics(request.json)
        return jsonify(result)

    @app.route("/collect_opd_pending_requests", methods=["POST", "OPTIONS"])
    def collect_opd_pending_requests():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_collect_opd_pending_requests(request.json)
        return jsonify(result)

    @app.route("/collect_pending_requests", methods=["POST", "OPTIONS"])
    def collect_pending_requests():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_collect_pending_requests(request.json)
        return jsonify(result)

    @app.route("/upload_opd_records", methods=["POST"])
    def upload_opd_records():
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No selected file"}), 400
        try:
            result = orchestrator.handle_upload_opd_records(file)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/process_op", methods=["POST", "OPTIONS"])
    def process_op():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_process_op(request.json)
        return jsonify(result)

    @app.route("/process_ip", methods=["POST", "OPTIONS"])
    def process_ip():
        if request.method == "OPTIONS":
            return orchestrator._options_ok()
        result = orchestrator.handle_process_ip(request.json)
        return jsonify(result)

    @app.route("/upload_pdcm_records", methods=["POST"])
    def upload_pdcm_records():
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No selected file"}), 400
        try:
            result = orchestrator.handle_upload_pdcm_records(file)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


if __name__ == "__main__":
    debug_mode = os.getenv("DOCSTRIBE_DEBUG", "false").lower() == "true"
    app = create_app(debug=debug_mode)
    app.run(debug=debug_mode, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
