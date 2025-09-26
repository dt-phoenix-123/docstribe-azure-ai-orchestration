#!/usr/bin/env python3
"""Docstribe agent configuration utilities.

This module centralises access to environment-driven settings, prompt templates,
and reference data required by the Docstribe orchestration stack. Secrets are
never hardcoded—instead, they are sourced from environment variables—while large
reference datasets (OPD ranges, etc.) can be supplied via optional JSON files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

import yaml

load_dotenv()
# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _resolve_path(path_like: str, default_root: Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (default_root / path)


PROJECT_ROOT = Path(os.getenv("DOCSTRIBE_PROJECT_ROOT", Path.cwd()))
PROMPT_FILE = _resolve_path(os.getenv("DOCSTRIBE_PROMPT_FILE", "prompt_config.yaml"), PROJECT_ROOT)
OPD_RANGE_UNIVERSE_FILE = _resolve_path(
    os.getenv("OPD_RANGE_UNIVERSE_FILE", "data/opd_range_universe.json"), PROJECT_ROOT
)
OPD_STRING_UNIVERSE_FILE = _resolve_path(
    os.getenv("OPD_STRING_UNIVERSE_FILE", "data/opd_string_universe.json"), PROJECT_ROOT
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt configuration file not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_json_list(path: Path) -> List[Any]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:  # pragma: no cover - surfaced at runtime
            raise ValueError(f"Invalid JSON structure in {path}: {exc}") from exc
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected a JSON list in {path}, found {type(payload).__name__}")


PROMPT_DATA = _load_yaml(PROMPT_FILE)
TEMPLATES: Dict[str, Any] = PROMPT_DATA.get("templates", {})
PROMPTS: Dict[str, Any] = PROMPT_DATA.get("prompts", {})
MASTER_PROMPTS: Dict[str, Any] = PROMPT_DATA.get("master_prompts", {})
MODELS: Dict[str, Any] = PROMPT_DATA.get("models", {})


def load_prompts() -> Dict[str, Any]:
    """Return a shallow copy of the `master_prompts` section."""
    return dict(MASTER_PROMPTS)


SYS_COHORT_PROMPT = MASTER_PROMPTS.get(
    "sys_cohort_prompt",
    TEMPLATES.get("sys_cohort_prompt", ""),
)
OPENAI_GEN_IP2_JSON_FORMAT = TEMPLATES.get("openai_gen_ip2_json_format", "{}")
OPENAI_GEN_IP2_JSON_FORMAT_V2 = TEMPLATES.get("openai_gen_ip2_json_format_v2", "{}")
OPD_OUTPUT_JSON_FORMAT_MEDICATIONS = TEMPLATES.get("opd_output_json_format_medications", "{}")
OPD_OUTPUT_JSON_FORMAT_MEDICATIONS_V2 = TEMPLATES.get(
    "opd_output_json_format_medications_v2",
    "{}",
)
OPD_OUTPUT_JSON_FORMAT_V3 = TEMPLATES.get("opd_output_json_format_v3", "{}")
SUMMARIZE_FORMAT = TEMPLATES.get("summarize_format", "{}")

# Optional large reference datasets. Supply via JSON files if classification is required.
OPD_RANGE_UNIVERSE: List[Dict[str, Any]] = _load_json_list(OPD_RANGE_UNIVERSE_FILE) or PROMPT_DATA.get(
    "opd_range_universe",
    [],
)
OPD_STRING_UNIVERSE: List[str] = _load_json_list(OPD_STRING_UNIVERSE_FILE) or PROMPT_DATA.get(
    "opd_string_universe",
    [],
)

# ---------------------------------------------------------------------------
# Environment-driven secrets and service endpoints
# ---------------------------------------------------------------------------

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "Docstribe-o3")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    AZURE_OPENAI_DEPLOYMENT,
)
AZURE_OPENAI_BASE_URL = os.getenv("AZURE_OPENAI_BASE_URL", "")
AZURE_GROK_BASE_URL = os.getenv(
    "AZURE_GROK_BASE_URL",
    "https://docstribeazureaifoundry.services.ai.azure.com/openai/v1/",
)
AZURE_GROK_DEPLOYMENT = os.getenv("AZURE_GROK_DEPLOYMENT", "Docstribe-grok-3")

SIMPLIFICATION_URL = os.getenv(
    "SIMPLIFICATION_URL",
    "https://docstribe-fine-tuning-service-877719534636.asia-south2.run.app/process_procedure",
)
PROJECT_ID = os.getenv("PROJECT_ID", "prod-408107")
IPD_SUBSCRIPTION_ID = os.getenv("IPD_SUBSCRIPTION_ID", "pdcm_payload-sub")
PDCM_AGENT_THRESHOLD = int(os.getenv("PDCM_AGENT_THRESHOLD", "4"))
LLM_ORCHESTRATOR_DB = os.getenv("LLM_ORCHESTRATOR_DB", "max_azure_check")
PDCM_COLLECTION = os.getenv("PDCM_COLLECTION", "pdcm_workflow")
PDCM_BATCH_COLLECTION = os.getenv("PDCM_BATCH_COLLECTION", "pdcm_batch_log")
OPD_COLLECTION = os.getenv("OPD_COLLECTION", "opd_workflow")
OPD_BATCH_COLLECTION = os.getenv("OPD_BATCH_COLLECTION", "opd_batch_log")
PDCM_INITIAL_COLLECTION = os.getenv("PDCM_INITIAL_COLLECTION", "pdcm_workflow_log")
OPD_INITIAL_COLLECTION = os.getenv("OPD_INITIAL_COLLECTION", "opd_workflow_log")
OPENAI_URL = os.getenv("OPENAI_URL", "https://api.openai.com/v1/chat/completions")

OPENAI_GPT4_TURBO_MODEL = os.getenv("OPENAI_GPT4_TURBO_MODEL", "gpt-4-turbo")
OPENAI_GPT4O_MODEL = os.getenv("OPENAI_GPT4O_MODEL", "gpt-4o")
OPENAI_GPT4O_MINI = os.getenv("OPENAI_GPT4O_MINI", "gpt-4o-mini")
OPENAI_GPTO3_MODEL = os.getenv("OPENAI_GPTO3_MODEL", "o3")

__all__ = [
    "MASTER_PROMPTS",
    "PROMPTS",
    "TEMPLATES",
    "MODELS",
    "SYS_COHORT_PROMPT",
    "OPENAI_GEN_IP2_JSON_FORMAT",
    "OPENAI_GEN_IP2_JSON_FORMAT_V2",
    "OPD_OUTPUT_JSON_FORMAT_MEDICATIONS",
    "OPD_OUTPUT_JSON_FORMAT_MEDICATIONS_V2",
    "OPD_OUTPUT_JSON_FORMAT_V3",
    "SUMMARIZE_FORMAT",
    "OPD_RANGE_UNIVERSE",
    "OPD_STRING_UNIVERSE",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_BASE_URL",
    "AZURE_GROK_BASE_URL",
    "AZURE_GROK_DEPLOYMENT",
    "SIMPLIFICATION_URL",
    "PROJECT_ID",
    "IPD_SUBSCRIPTION_ID",
    "PDCM_AGENT_THRESHOLD",
    "LLM_ORCHESTRATOR_DB",
    "PDCM_COLLECTION",
    "PDCM_BATCH_COLLECTION",
    "OPD_COLLECTION",
    "OPD_BATCH_COLLECTION",
    "PDCM_INITIAL_COLLECTION",
    "OPD_INITIAL_COLLECTION",
    "OPENAI_URL",
    "OPENAI_GPT4_TURBO_MODEL",
    "OPENAI_GPT4O_MODEL",
    "OPENAI_GPT4O_MINI",
    "OPENAI_GPTO3_MODEL",
    "load_prompts",
]


OPD_RANGE_UNIVERSE = [{'test': 'MCV', 'ranges': {'significant_low': [None, 75], 'borderline_low': [75, 80], 'normal': [81, 95], 'borderline_high': [96, 100], 'significant_high': [100, None]}}, {'test': 'NRBCs', 'ranges': {'normal': [0, None], 'significant_high': [10, None]}}, {'test': 'PCV (Haematocrit)', 'ranges': {'significant_low': {'males': [None, 30], 'females': [None, 27]}, 'borderline_low': {'males': [30, 37], 'females': [27, 33]}, 'normal': {'males': [37, 47], 'females': [33, 43]}, 'borderline_high': {'males': [47, 52], 'females': [43, 48]}, 'significant_high': {'males': [52, None], 'females': [48, None]}}}, {'test': 'Platelet Count', 'ranges': {'significant_low': [None, 50], 'borderline_low': [50, 149], 'normal': [149, 450], 'borderline_high': [450, 900], 'significant_high': [900, None]}}, {'test': 'RBC Count (Red Blood Cell)', 'ranges': {'significant_low': {'males': [None, 3.5], 'females': [None, 3.1]}, 'borderline_low': {'males': [3.5, 4.2], 'females': [3.1, 3.6]}, 'normal': {'males': [4.2, 5.5], 'females': [3.6, 5.0]}, 'borderline_high': {'males': [5.5, 6.2], 'females': [5.0, 5.7]}, 'significant_high': {'males': [6.2, None], 'females': [5.7, None]}}}, {'test': 'RDW', 'ranges': {'significant_low': [None, 7], 'borderline_low': [7, 10], 'normal': [10, 14], 'borderline_high': [14, 16], 'significant_high': [16, None]}}, {'test': 'Total Leukocyte Count (TLC)', 'ranges': {'significant_low': [None, 3], 'borderline_low': [3, 4], 'normal': [4, 10], 'borderline_high': [10, 20], 'significant_high': [20, None]}}, {'test': 'Absolute Basophil Count', 'ranges': {'significant_low': [None, 0.01], 'borderline_low': [0.01, 0.02], 'normal': [0.02, 0.1], 'borderline_high': [0.1, 0.2], 'significant_high': [0.2, None]}}, {'test': 'Absolute Eosinophil Count (AEC)', 'ranges': {'significant_low': [None, 0.05], 'borderline_low': [0.05, 0.1], 'normal': [0.1, 1.5], 'borderline_high': [1.5, 3], 'significant_high': [3, None]}}, {'test': 'Absolute Lymphocyte Count', 'ranges': {'significant_low': [None, 0.8], 'borderline_low': [0.8, 1], 'normal': [1, 4], 'borderline_high': [4, 5], 'significant_high': [5, None]}}, {'test': 'Absolute Monocyte Count', 'ranges': {'significant_low': [None, 0.1], 'borderline_low': [0.1, 0.19], 'normal': [0.19, 1], 'borderline_high': [1, 1.5], 'significant_high': [1.5, None]}}, {'test': 'Absolute Neutrophil Count', 'ranges': {'significant_low': [None, 1.5], 'borderline_low': [1.5, 2], 'normal': [2, 7.5], 'borderline_high': [7.5, 9], 'significant_high': [9, None]}}, {'test': 'DLC - Band Cells', 'ranges': {'normal': [0, 11], 'borderline_high': [11, 14], 'significant_high': [14, None]}}, {'test': 'DLC - Basophils', 'ranges': {'significant_low': [None, 0.2], 'borderline_low': [0.2, 0.4], 'normal': [0.4, 1], 'borderline_high': [1.1, 2.0], 'significant_high': [2.0, None]}}, {'test': 'DLC - Blasts', 'ranges': {'normal': [0, None], 'significant_high': [0.1, None]}}, {'test': 'DLC - Eosinophils', 'ranges': {'significant_low': [None, 0.5], 'borderline_low': [0.5, 1.0], 'normal': [1, 10], 'borderline_high': [10, 15], 'significant_high': [15, None]}}, {'test': 'DLC - Metamyelocytes', 'ranges': {'normal': [0, 0.2], 'borderline_high': [0.2, 0.5], 'significant_high': [0.5, None]}}, {'test': 'DLC - Lymphocytes', 'ranges': {'significant_low': [None, 15], 'borderline_low': [15, 20], 'normal': [20, 45], 'borderline_high': [45, 55], 'significant_high': [55, None]}}, {'test': 'DLC - Monocytes', 'ranges': {'significant_low': [None, 1], 'borderline_low': [1, 2], 'normal': [2, 10], 'borderline_high': [10, 15], 'significant_high': [15, None]}}, {'test': 'DLC - Myelocytes', 'ranges': {'normal': [0, 0.5], 'borderline_high': [0.5, 2], 'significant_high': [2, None]}}, {'test': 'DLC - Neutrophils', 'ranges': {'significant_low': [None, 30], 'borderline_low': [30, 40], 'normal': [40, 75], 'borderline_high': [75, 80], 'significant_high': [80, None]}}, {'test': 'DLC - Promyelocytes', 'ranges': {'normal': [0, 0.3], 'borderline_high': [0.3, 1], 'significant_high': [1, None]}}, {'test': 'Haemoglobin Estimation (Hb)', 'ranges': {'significant_low': {'males': [None, 7], 'females': [None, 6]}, 'borderline_low': {'males': [7, 13], 'females': [6, 12]}, 'normal': {'males': [13, 17], 'females': [12, 15]}, 'borderline_high': {'males': [17, 19], 'females': [15, 17]}, 'significant_high': {'males': [19, None], 'females': [17, None]}}},
{'test': 'MCH', 'ranges': {'significant_low': [None, 15], 'borderline_low': [15, 27], 'normal': [27, 34], 'borderline_high': [34, 40], 'significant_high': [40, None]}}, {'test': 'MCHC', 'ranges': {'significant_low': [None, 22], 'borderline_low': [22, 30], 'normal': [30, 37], 'borderline_high': [37, 45], 'significant_high': [45, None]}}, {'test': 'Creatinine', 'ranges': {'significant_low': {'males': [None, 0.3], 'females': [None, 0.2]}, 'borderline_low': {'males': [0.3, 0.6], 'females': [0.2, 0.5]}, 'normal': {'males': [0.6, 1.35], 'females': [0.5, 1.1]}, 'borderline_high': {'males': [1.35, 1.5], 'females': [1.1, 1.3]}, 'significant_high': {'males': [1.5, None], 'females': [1.3, None]}}}, {'test': 'A/G Ratio (Albumin/Globulin Ratio)', 'ranges': {'significant_low': [None, 0.8], 'borderline_low': [0.8, 1.0], 'normal': [1.0, 2.5], 'borderline_high': [2.5, 2.9], 'significant_high': [2.9, None]}}, {'test': 'Albumin', 'ranges': {'significant_low': [None, 2.0], 'borderline_low': [2.0, 3.4], 'normal': [3.4, 5.0], 'borderline_high': [5.0, 5.9], 'significant_high': [6.0, None]}}, {'test': 'Alkaline Phosphatase', 'ranges': {'significant_low': [None, 15], 'borderline_low': [15, 30], 'normal': [30, 130], 'borderline_high': [130, 350], 'significant_high': [350, None]}}, {'test': 'Bilirubin Conjugated (Bc)', 'ranges': {'normal': [0, 0.3], 'borderline_high': [0.3, 0.7], 'significant_high': [0.7, None]}}, {'test': 'Bilirubin Total (T Bil)', 'ranges': {'normal': [0, 5], 'borderline_high': [5, 12], 'significant_high': [12, None]}}, {'test': 'Bilirubin Unconjugated (Bu)', 'ranges': {'normal': [0, 1.6], 'borderline_high': [1.6, 3], 'significant_high': [3, None]}}, {'test': 'GGTP (GAMMA GT)', 'ranges': {'normal': {'males': [0, 97], 'females': [0, 59]}, 'borderline_high': {'males': [97, 240], 'females': [59, 145]}, 'significant_high': {'males': [240, None], 'females': [145, None]}}}, {'test': 'Globulin', 'ranges': {'significant_low': [None, 1.25], 'borderline_low': [1.25, 1.5], 'normal': [1.5, 3.6], 'borderline_high': [3.6, 4.5], 'significant_high': [4.5, None]}}, {'test': 'Protein Total', 'ranges': {'significant_low': [None, 5.0], 'borderline_low': [5.0, 6.0], 'normal': [6.0, 8.3], 'borderline_high': [8.3, 9.0], 'significant_high': [9.0, None]}}, {'test': 'SGOT (AST)', 'ranges': {'borderline_low': {'males': [None, 10], 'females': [None, 6]}, 'normal': {'males': [10, 80], 'females': [6, 70]}, 'borderline_high': {'males': [80, 350], 'females': [70, 340]}, 'significant_high': {'males': [350, None], 'females': [340, None]}}}, {'test': 'SGPT (ALTV)', 'ranges': {'normal': [0, 100], 'borderline_high': [100, 250], 'significant_high': [250, None]}}, {'test': 'Prothrombin Time (PT) Control', 'ranges': {'significant_low': [None, 9], 'borderline_low': [9, 11], 'normal': [11, 13.5], 'borderline_high': [13.5, 16], 'significant_high': [16, None]}}, {'test': 'Prothrombin Time (PT) INR Value', 'ranges': {'significant_low': [None, 0.75], 'borderline_low': [0.75, 0.8], 'normal': [0.8, 1.2], 'borderline_high': [1.2, 2.0], 'significant_high': [2.0, None]}}, {'test': 'Prothrombin Time Patient Value', 'ranges': {'significant_low': [None, 8], 'borderline_low': [8, 10], 'normal': [10, 12.5], 'borderline_high': [12.5, 15], 'significant_high': [15, None]}}, {'test': 'Calcium', 'ranges': {'significant_low': [None, 7.5], 'borderline_low': [7.5, 8.0], 'normal': [8.0, 12.0], 'borderline_high': [12.0, 13.9], 'significant_high': [13.9, None]}}, {'test': 'Chloride', 'ranges': {'significant_low': [None, 88], 'borderline_low': [88, 98], 'normal': [98, 107], 'borderline_high': [107, 115], 'significant_high': [115, None]}}, {'test': 'Phosphorus', 'ranges': {'significant_low': [None, 1.5], 'borderline_low': [1.5, 2.5], 'normal': [2.5, 4.5], 'borderline_high': [4.5, 7.0], 'significant_high': [7.0, None]}}, {'test': 'Potassium', 'ranges': {'significant_low': [None, 2.5], 'borderline_low': [2.5, 3.4], 'normal': [3.4, 5.5], 'borderline_high': [5.0, 7.5], 'significant_high': [7.5, None]}}, {'test': 'Sodium', 'ranges': {'significant_low': [None, 125], 'borderline_low': [125, 135], 'normal': [135, 145], 'borderline_high': [145, 165], 'significant_high': [165, None]}}, {'test': 'UREA', 'ranges': {'significant_low': [None, 10], 'borderline_low': [10, 13], 'normal': [13, 43], 'borderline_high': [43, 53], 'significant_high': [53, None]}}, {'test': 'Uric Acid', 'ranges': {'significant_low': {'males': [None, 2.0], 'females': [None, 1.5]}, 'borderline_low': {'males': [2.0, 3.5], 'females': [1.5, 2.6]}, 'normal': {'males': [3.5, 7.2], 'females': [2.6, 6.0]}, 'borderline_high': {'males': [7.2, 14.0], 'females': [6.0, 12]}, 'significant_high': {'males': [14.0, None], 'females': [12, None]}}}, {'test': 'CRP - C Reactive Protein', 'ranges': {'normal': [0, 10], 'borderline_high': [10, 50], 'significant_high': [50, None]}}, {'test': 'Iron', 'ranges': {'significant_low': {'males': [None, 30], 'females': [None, 20]}, 'borderline_low': {'males': [30, 50], 'females': [20, 40]}, 'normal': {'males': [50, 180], 'females': [40, 160]}, 'borderline_high': {'males': [180, 200], 'females': [160, 180]}, 'significant_high': {'males': [200, None], 'females': [180, None]}}}, {'test': 'TIBC', 'ranges': {'significant_low': [None, 200], 'borderline_low': [200, 250], 'normal': [250, 480], 'borderline_high': [480, 550], 'significant_high': [550, None]}}, {'test': 'Transferin Saturation Index', 'ranges': {'significant_low': [None, 10], 'borderline_low': [10, 13], 'normal': [13, 45], 'borderline_high': [45, 60], 'significant_high': [60, None]}}, {'test': 'Ferritin', 'ranges': {'significant_low': {'males': [None, 10], 'females': [None, 5]}, 'borderline_low': {'males': [10, 20], 'females': [5, 10]}, 'normal': {'males': [20, 500], 'females': [10, 400]}, 'borderline_high': {'males': [500, 800], 'females': [400, 700]}, 'significant_high': {'males': [800, None], 'females': [700, None]}}}, {'test': 'Cholesterol Total : HDL Cholesterol Ratio', 'ranges': {'significant_low': [None, 3], 'borderline_low': [3, 4], 'normal': [4, 5], 'borderline_high': [5.0, 6.0], 'significant_high': [6.0, None]}},
{'test': 'Cholesterol-Total', 'ranges': {'normal': [None, 240], 'borderline_high': [240, 280], 'significant_high': [280, None]}}, {'test': 'HDL Cholesterol', 'ranges': {'significant_low': {'males': [None, 30], 'females': [None, 40]}, 'normal': {'males': [30, 60], 'females': [40, 60]}}}, {'test': 'LDL Cholesterol', 'ranges': {'normal': [0, 150], 'borderline_high': [150, 190], 'significant_high': [190, None]}}, 
{'test': 'LDL Cholesterol : HDL Cholesterol Ratio', 'ranges': {'normal': [0, 4], 'borderline_high': [4.0, 5.0], 'significant_high': [5.0, None]}}, {'test': 'Triglycerides', 'ranges': {'normal': [0, 200], 'borderline_high': [200, 500], 'significant_high': [500, None]}}, {'test': 'VLDL Cholesterol', 'ranges': {'normal': [0, 40], 'borderline_high': [40, 100], 'significant_high': [100, None]}}, {'test': 'NT-proBNP', 'ranges': {'normal': [0, 450], 'borderline_high': [450, 1000], 'significant_high': [1000, None]}}, {'test': 'Estimated average glucose', 'ranges': {'borderline_low': [70, 100], 'normal': [100, 125], 'borderline_high': [125, 183], 'significant_high': [183, None]}}, {'test': 'HBA1C - Glycosylated Hemoglobin', 'ranges': {'normal': [None, 6], 'borderline_high': [6.0, 7.0], 'significant_high': [7.0, None]}}, {'test': 'Blood Sugar (Fasting)', 'ranges': {'significant_low': [None, 50], 'borderline_low': [50, 70], 'normal': [70, 140], 'borderline_high': [140, 200], 'significant_high': [200, None]}}, {'test': 'Troponin-I (Quantitative)', 'ranges': {'normal': [0, 0.05], 'borderline_high': [0.05, 0.12], 'significant_high': [0.12, None]}}, {'test': 'FT3 - Free T3', 'ranges': {'significant_low': [None, 1.5], 'borderline_low': [1.5, 2.7], 'normal': [2.7, 5.3], 'borderline_high': [5.3, 6.5], 'significant_high': [6.5, None]}}, {'test': 'FT4 - Free T4', 'ranges': {'significant_low': [None, 0.5], 'borderline_low': [0.5, 0.7], 'normal': [0.7, 2], 'borderline_high': [2, 2.5], 'significant_high': [2.5, None]}}, {'test': 'TSH 3 -Thyroid Stimulating Hormone (3rd Generation)', 'ranges': {'significant_low': [None, 0.1], 'borderline_low': [0.1, 0.4], 'normal': [0.4, 6.5], 'borderline_high': [6.5, 9], 'significant_high': [9.0, None]}}, {'test': 'CK / CPK (Creatine Kinase)', 'ranges': {'normal': {'males': [0, 250], 'females': [0, 200]}, 'borderline_high': {'males': [250, 800], 'females': [200, 650]}, 'significant_high': {'males': [800, None], 'females': [650, None]}}}, {'test': 'CA 125', 'ranges': {'normal': [None, 35], 'borderline_high': [35, 100], 'significant_high': [100, None]}}, {'test': 'CORTISOL (AM)', 'ranges': {'significant_low': [None, 3], 'borderline_low': [3, 6], 'normal': [6, 25], 'borderline_high': [25, 45], 'significant_high': [45, None]}}, {'test': 'CEA', 'ranges': {'normal': [None, 3.0], 'borderline_high': [3.0, 5.0], 'significant_high': [5.0, None]}}, {'test': 'ALPHA-FETOPROTEIN(AFP) TUMOR MARKER', 'ranges': {'normal': [0, 10], 'borderline_high': [10, 100], 'significant_high': [100, None]}}, {'test': 'Vitamin B12', 'ranges': {'significant_low': [None, 100], 'borderline_low': [100, 150], 'normal': [150, 2000], 'borderline_high': [2000, 5000], 'significant_high': [5000, None]}}, {'test': 'Anti Mullerian Hormone Amh', 'ranges': {'significant_low': [None, 0.5], 'borderline_low': [0.5, 1], 'normal': [1, 3], 'borderline_high': [3, 5], 'significant_high': [5, None]}}, {'test': 'Prolactin', 'ranges': {'normal': {'males': [0, 18.6], 'females': [0, 23]}, 'borderline_high': {'males': [18.6, 34], 'females': [23, 38]}, 'significant_high': {'males': [34, None], 'females': [38, None]}}}, {'test': 'Grade A (Progressive Motile)', 'ranges': {'significant_low': [None, 15], 'borderline_low': [15, 32], 'normal': [32, None]}}, {'test': 'Morphology, Abnormal', 'ranges': {'normal': [None, 20], 'borderline_high': [20, 96], 'significant_high': [96, None]}}, {'test': 'Morphology, Normal', 'ranges': {'significant_low': [None, 4], 'borderline_low': [4, 50], 'normal': [50, None]}}, {'test': 'Non-Motile', 'ranges': {'normal': [None, 60], 'borderline_high': [60, 90], 'significant_high': [90, None]}}, {'test': 'Percentage Motility { Grade A + Grade B }', 'ranges': {'significant_low': [None, 20], 'borderline_low': [20, 39], 'normal': [39, None]}}, {'test': 'Total Sperm Count', 'ranges': {'significant_low': [None, 5], 'borderline_low': [5, 15], 'normal': [15, None]}}, {'test': 'Volume', 'panel': 'Sperm', 'ranges': {'significant_low': [None, 1], 'borderline_low': [1, 1.5], 'normal': [1.5, 5]}}, {'test': 'Urine Analysis - Ph', 'ranges': {'significant_low': [None, 3], 'borderline_low': [3, 4.5], 'normal': [4.5, 8], 'borderline_high': [8, 10], 'significant_high': [10, None]}}, {'test': 'Urine Analysis - Quantity', 'ranges': {'significant_low': [None, 5], 'borderline_low': [5, 15], 'normal': [15, 80]}}, {'test': 'Urine Analysis - Specific Gravity', 'ranges': {'normal': [1.005, 1.03], 'borderline_high': [1.03, 1.035], 'significant_high': [1.035, None]}}, {'test': '25 Hydroxy Vitamin D-Total', 'ranges': {'significant_low': [None, 12], 'borderline_low': [12, 30], 'normal': [30, 75], 'borderline_high': [75, 100], 'significant_high': [100, None]}}, {'test': 'Glucose Pp (2 Hour Post Prandial)', 'ranges': {'significant_low': [None, 64], 'borderline_low': [64, 75], 'normal': [75, 140], 'borderline_high': [140, 200], 'significant_high': [200, None]}}, {'test': 'Activtd.Partial Thromboplastin Time (Aptt) Control', 'ranges': {'normal': [0, 40], 'borderline_high': [40, 70], 'significant_high': [70, None]}}, {'test': 'Activtd.Partial Thromboplastin Time Ratio', 'ranges': {'significant_low': [None, 0.6], 'borderline_low': [0.6, 1], 'normal': [1, 1.5], 'borderline_high': [1.5, 2], 'significant_high': [2, None]}}, {'test': 'Actvtd.Partial Thromboplastin Time (Aptt) Patient', 'ranges': {'significant_low': [None, 20], 'borderline_low': [20, 25], 'normal': [25, 45], 'borderline_high': [45, 70], 'significant_high': [70, None]}}, {'test': 'Fdp (Fibrin Degradation Product)', 'ranges': {'normal': [None, 20], 'borderline_high': [20, 40], 'significant_high': [40, None]}}, {'test': 'Ldh', 'ranges': {'normal': [120, 350], 'borderline_high': [350, 700], 'significant_high': [700, None]}}, {'test': 'Antibody To Hepatitis C Virus (Eci)', 'ranges': {'non-reactive': [None, 0.99], 'reactive': [1.0, None]}}, {'test': 'Hepatitis B Surface Antigen (Hbsag) (Eci)', 'ranges': {'non-reactive': [None, 0.99], 'reactive': [1.0, None]}}, {'test': 'Hiv 1 & 2 Antibody (Eci)', 'ranges': {'non-reactive': [None, 0.99], 'reactive': [1.0, None]}},
{'test': 'Glucose (Rbs/Random Blood Sugar)', 'ranges': {'significant_low': [None, 54], 'borderline_low': [54, 70], 'normal': [70, 140], 'borderline_high': [140, 190], 'significant_high': [190, None]}}, {'test': 'Psa Total (Prostate Specific Antigen)', 'ranges': {'normal': [None, 4], 'borderline_high': [4, 10], 'significant_high': [10, None]}}, {'test': 'Total Ige', 'ranges': {'normal': [None, 150], 'borderline_high': [150, 501], 'significant_high': [501, None]}}, {'test': 'Microalbumin Creatinine Ratio In Random Urine', 'ranges': {'normal': [None, 30], 'borderline_high': [30, 300], 'significant_high': [300, None]}}, {'test': 'Urine Creatinine (Spot)', 'ranges': {'significant_low': {'males': [None, 10], 'females': [None, 10]}, 'borderline_low': {'males': [10, 20], 'females': [10, 20]}, 'normal': {'males': [20, 370], 'females': [20, 320]}, 'borderline_high': {'males': [370, 400], 'females': [320, 350]}, 'significant_high': {'males': [400, None], 'females': [350, None]}}}, {'test': 'Urine Microalbumin (Random Urine)', 'ranges': {'normal': [None, 30], 'borderline_high': [30, 250], 'significant_high': [250, None]}}, {'test': 'Lipase', 'ranges': {'normal': [10, 140], 'borderline_high': [140, 600], 'significant_high': [600, None]}}, {'test': 'Creatinine Kinase Mb Mass', 'ranges': {'normal': [0, 4], 'borderline_high': [4, 25], 'significant_high': [25, None]}}, 
{'test': 'Esr (Erythrocyte Sed.Rate)', 'ranges': {'normal': {'males': [0, 30], 'females': [0, 25]}, 'borderline_high': {'males': [30, 75], 'females': [25, 70]}, 'significant_high': {'males': [75, None], 'females': [70, None]}}}, {'test': 'Grade A {Progressive Motile}', 'ranges': {'significant_low': [None, 10], 'borderline_low': [10, 32], 'normal': [32, None]}},
{'test': 'Motility, Non-Motile', 'ranges': {'normal': [0, 60], 'borderline_high': [60, 80], 'significant_high': [80, None]}}, {'test': 'Amylase', 'ranges': {'normal': [30, 180], 'borderline_high': [180, 300], 'significant_high': [300, None]}}, {'test': 'Urea (Post Dialysis)', 'ranges': {'significant_low': [None, 5], 'borderline_low': [5, 10], 'normal': [10, 40], 'borderline_high': [40, 50], 'significant_high': [50, None]}}, {'test': 'Urine Protein Creatinine Ratio', 'ranges': {'normal': [0, 300], 'borderline_high': [300, 500], 'significant_high': [500, None]}}, {'test': 'Urine Protein, Random', 'ranges': {'normal': [1, 150], 'borderline_high': [150, 500], 'significant_high': [500, None]}}, {'test': 'Ra Factor/ Rf/ Rheumatoid Factor', 'ranges': {'normal': [0, 30], 'borderline_high': [30, 100], 'significant_high': [100, None]}}, {'test': 'Anti Cyclic Citrullinated Peptide (Anti Ccp)', 'ranges': {'normal': [1, 30], 'borderline_high': [30, 40], 'significant_high': [40, None]}}, 
{'test': 'Protein', 'ranges': {'significant_low': [None, 5], 'borderline_low': [5, 5.5], 'normal': [5.5, 9], 'borderline_high': [9, 10], 'significant_high': [10, None]}}, {'test': 'Pre Dialysis Screen (Eci) Reading', 'ranges': {'significant_low': [0, 1500], 'borderline_low': [1500, 3000], 'normal': [3000, 15000], 'borderline_high': [15000, 20000], 'significant_high': [20000, None]}}, {'test': 'Pth (Parathyroid Hormone), Intact', 'ranges': {'normal': [0, 100], 'borderline_high': [100, 300], 'significant_high': [300, None]}}, 
{'test': 'Anti Thyroglobulin (Anti-Tg)', 'ranges': {'negative': [6.4, 18], 'positive': [18, None]}}, {'test': '24 Hour Urine Total Volume', 'ranges': {'significant_low': [0, 400], 'borderline_low': [400, 600], 'normal': [600, 2000], 'borderline_high': [2000, 2500], 'significant_high': [2500, None]}}, {'test': 'Total Urine Protein', 'ranges': {'normal': [0, 15], 'borderline_high': [15, 50], 'significant_high': [50, None]}}, {'test': 'Urine Protein (24 Hours)', 'ranges': {'normal': [0, 200], 'borderline_high': [200, 500], 'significant_high': [500, None]}}, {'test': 'Reticulocyte Count', 'ranges': {'significant_low': [None, 0.3], 'normal': [0.3, 3.5], 'borderline_high': [3.5, 6], 'significant_high': [6, None]}}, {'test': 'Blood Sugar (1 Hr.)', 'ranges': {'normal': [0, 200], 'borderline_high': [200, 250], 'significant_high': [250, None]}}, {'test': 'Blood Sugar (2 Hrs.)', 'ranges': {'normal': [0, 200], 'borderline_high': [200, 250], 'significant_high': [250, None]}}, {'test': 'Beta Hcg (Cancer Marker)', 'ranges': {'normal': [0, 15], 'borderline_high': [15, 100], 'significant_high': [100, None]}}, {'test': 'D- Dimer', 'ranges': {'normal': [0, 800], 'borderline_high': [800, 2000], 'significant_high': [2000, None]}}, {'test': 'Cortisol (Random)', 'ranges': {'normal': [0, 300], 'borderline_high': [300, 500], 'significant_high': [500, None]}}, 
{'test': 'G6Pd Deficiency Test', 'ranges': {'significant_low': {'males': [None, 0.5], 'females': [None, 0.5]}, 'normal': {'males': [0.5, 6], 'females': [0.5, 5]}, 'significant_high': {'males': [6, None], 'females': [5, None]}}}, 
{'test': 'Procalcitonin (Quantitative)', 'ranges': {'normal': [0, 3], 'borderline_high': [3, 10], 'significant_high': [10, None]}}, {'test': 'Osmolality - Serum', 'ranges': {'significant_low': [None, 245], 'borderline_low': [245, 260], 'normal': [260, 300], 'borderline_high': [300, 330], 'significant_high': [330, None]}}, 
{'test': 'Bun - Blood Urea Nitrogen', 'ranges': {'significant_low': [None, 3], 'borderline_low': [3, 5], 'normal': [5, 25], 'borderline_high': [25, 60], 'significant_high': [60, None]}}, {'test': 'Urine Sodium (Spot)', 'ranges': {'significant_low': [None, 5], 'borderline_low': [5, 10], 'normal': [10, 300], 'borderline_high': [300, 500], 'significant_high': [500, None]}},
 
    {
        "test": "Mean Platelet Volume (MPV)",
        "ranges": {
            "significant_low": [None, 6.0],
            "borderline_low": [6.0, 7.4],
            "normal": [7.5, 12.0],
            "borderline_high": [12.1, 14.0],
            "significant_high": [14.0, None]
        }
    },
    {
        "test": "Platelets",
        "ranges": {
            "significant_low": [None, 100000],
            "borderline_low": [100000, 149000],
            "normal": [150000, 450000],
            "borderline_high": [451000, 600000],
            "significant_high": [600000, None]
        }
    },
    {
        "test": "RBCs",
        "ranges": {
            "significant_low": {
                "males": [None, 3.5],
                "females": [None, 3.0]
            },
            "borderline_low": {
                "males": [3.5, 4.4],
                "females": [3.0, 3.9]
            },
            "normal": {
                "males": [4.5, 5.5],
                "females": [4.0, 5.0]
            },
            "borderline_high": {
                "males": [5.6, 7.0],
                "females": [5.1, 6.0]
            },
            "significant_high": {
                "males": [7, None],
                "females": [6, None]
            }
        }
    },
    {
        "test": "WBCs",
        "ranges": {
            "significant_low": [None, 3000],
            "borderline_low": [3000, 3999],
            "normal": [4000, 11000],
            "borderline_high": [11001, 19000],
            "significant_high": [19000, None]
        }
    },
    {
        "test": "Ca 19.9 Pancreatic Cancer Marker",
        "ranges": {
            "normal": [0, 37],
            "borderline_high": [38, 100],
            "significant_high": [100, None]
        }
    },
    {
        "test": "Fdp (Fibrin Degradation Product)",
        "ranges": {
            "significant_low": [None, 10],
            "normal": [10, 35],
            "significant_high": [35, None]
        }
    },
    {
        "test": "Dhea Sulphate Dehydroepiandrosterone Sulphate",
        "ranges": {
            "significant_low": {
                "males": [None, 40],
                "females": [None, 20]
            },
            "borderline_low": {
                "males": [40, 79],
                "females": [20, 34]
            },
            "normal": {
                "males": [80, 560],
                "females": [35, 430]
            },
            "borderline_high": {
                "males": [561, 1200],
                "females": [431, 750]
            },
            "significant_high": {
                "males": [1200, None],
                "females": [750, None]
            }
        }
    },
    {
        "test": "Insulin Post Prandial (PP)",
        "ranges": {
            "significant_low": [None, 5],
            "borderline_low": [5, 9],
            "normal": [10, 120],
            "borderline_high": [121, 150],
            "significant_high": [150, None]
        }
    },
    {
        "test": "C-Reactive Protein Cardio; HsCRP",
        "ranges": {
            "normal": [None, 1.0],
            "borderline_high": [1.0, 10],
            "significant_high": [10, None]
        }
    },
    {
        "test": "Homocysteine Quantitative, Serum",
        "ranges": {
            "normal": [5, 25],
            "borderline_high": [25, 90],
            "significant_high": [90, None]
        }
    },
    {
        "test": "Lipoprotein (A) Lp(A)",
        "ranges": {
            "normal": [None, 30],
            "borderline_high": [30, 100],
            "significant_high": [100, None]
        }
    },
    {
        "test": "Psa Profile",
        "ranges": {
            "normal": [None, 4.0],
            "significant_high": [4.0, None]
        }
    },
    {
        "test": "Ph",
        "ranges": {
            "significant_low": [None, 7.3],
            "normal": [7.3, 7.5],
            "significant_high": [7.5, None]
        }
    },
    {
        "test": "Ammonia Blood",
        "ranges": {
            "normal": [15, 50],
            "borderline_high": [50, 150],
            "significant_high": [150, None]
        }
    },
    {
        "test": "Ammonia Blood",
        "ranges": {
            "normal": [15, 50],
            "borderline_high": [50, 150],
            "significant_high": [150, None]
        }
    },
    {
        "test": "Urine Albumin",
        "ranges": {
            "significant_low": [None, 30],
            "borderline_high": [31, 300],
            "significant_high": [300, None]
        }
    },
    {
        "test": "Folate (Folic Acid), Serum",
        "ranges": {
            "significant_low": [None, 2.5],
            "borderline_low": [2.5, 3.9],
            "normal": [2.5, 20],
            "borderline_high": [20.1, 25],
            "significant_high": [25, None]
        }
    },
    {
        "test": "Factor VIII Functional / Activity",
        "ranges": {
            "significant_low": [None, 30],
            "borderline_low": [30, 49],
            "normal": [50, 200],
            "borderline_high": [200, 250],
            "significant_high": [250, None]
        }
    },
    {
        "test": "Fibrinogen, Clotting Activity",
        "ranges": {
            "significant_low": [None, 150],
            "borderline_low": [150, 199],
            "normal": [200, 400],
            "borderline_high": [401, 500],
            "significant_high": [500, None]
        }
    },
    {
        "test": "Hepatitis B Surface Antigen (Eci) Reading",
        "ranges": {
            "normal": [None, 1.0],
            "significant_high": [1.0, None]
        }
    },
    {
        "test": "Mantoux / Ppd/ Tuberculin Test",
        "ranges": {
            "normal": [None, 5],
            "borderline_high": [5, 9],
            "significant_high": [9, None]
        }
    },
    {
        "test": "Tissue Transglutaminase (Ttg) Antibody Iga",
        "ranges": {
            "normal": [None, 7],
            "borderline_high": [7, 15],
            "significant_high": [15, None]
        }
    },
    {
        "test": "Calcium, Ionized",
        "ranges": {
            "significant_low": [None, 1.0],
            "borderline_low": [1.0, 1.11],
            "normal": [1.12, 1.32],
            "borderline_high": [1.33, 1.40],
            "significant_high": [1.40, None]
        }
    },
    {
        "test": "Cytomegalovirus (CMV) DNA, Quantitative PCR",
        "ranges": {
            "normal": [None, 1000],
            "borderline_high": [1000, 9999],
            "significant_high": [10000, None]
        }
    },
    {
        "test": "Hepatitis B Virus Surface Antibody (Hbsab) Quantitative",
        "ranges": {
            "normal": [None, 5],
            "significant_high": [10, None]
        }
    },
    {
        "test": "Insulin Fasting",
        "ranges": {
            "significant_low": [None, 2],
            "normal": [2, 30],
            "borderline_high": [30, 60],
            "significant_high": [60, None]
        }
    },
    {
        "test": "Magnesium",
        "ranges": {
            "significant_low": [None, 1.5],
            "borderline_low": [1.5, 1.6],
            "normal": [1.6, 2.5],
            "borderline_high": [2.5, 2.6],
            "significant_high": [2.6, None]
        }
    },
    {
        "test": "Aldosterone, Plasma",
        "ranges": {
            "significant_low": [None, 3],
            "borderline_low": [3, 4],  # This appears to be a typo, and might need correction
            "normal": [4, 30],
            "borderline_high": [30, 40],
            "significant_high": [40, None]
        }
    },
    {
        "test": "Plasma Renin, Direct",
        "ranges": {
            "significant_low": [None, 0.3],
            "borderline_low": [0.3, 0.5],
            "normal": [0.5, 3],
            "borderline_high": [3, 5],  # Note: There seems to be a gap between the normal and borderline high range
            "significant_high": [5, None]
        }
    },
    {
        "test": "Protein Electrophoresis, Serum",
        "ranges": {
            "significant_low": {
                "Albumin": [None, 3.0],
                "Globulin": [None, 1.5]
            },
            "borderline_low": {
                "Albumin": [3.0, 3.4],
                "Globulin": [1.5, 1.9]
            },
            "normal": {
                "Albumin": [3.5, 5.0],
                "Globulin": [2.0, 3.5]
            },
            "borderline_high": {
                "Albumin": [5.1, 6.0],
                "Globulin": [3.6, 4.5]
            },
            "significant_high": {
                "Albumin": [6.0, None],
                "Globulin": [4.5, None]
            }
        }
    },
    {
        "test": "Progesterone",
        "ranges": {
            "significant_low": {
                "Follicular": [None, 0.1],
                "Luteal": [None, 4.0]
            },
            "borderline_low": {
                "Follicular": [0.1, 0.2],
                "Luteal": [4.0, 4.0]  # Typo, as it does not range, may need to adjust this value.
            },
            "normal": {
                "Follicular": [0.1, 0.3],
                "Luteal": [4.1, 34.0]
            },
            "borderline_high": {
                "Follicular": [0.3, 0.4],
                "Luteal": [34.1, 40.0]
            },
            "significant_high": {
                "Follicular": [0.4, None],
                "Luteal": [40.0, None]
            }
        }
    },
    {
        "test": "Immunophenotyping By Flow Cytometry, CD19",
        "ranges": {
            "significant_low": [None, 2],
            "borderline_low": [2, 5],
            "normal": [5, 20],
            "borderline_high": [20, 30],
            "significant_high": [30, None]
        }
    },
    {
        "test": "Adenosine Deaminase; ADA",
        "ranges": {
            "normal": [None, 40],
            "borderline_high": [40, 60],
            "significant_high": [60, None]
        }
    },
    {
        "test": "Eosinophils",
        "ranges": {
            "normal": [0, 10],
            "borderline_high": [10, 20],
            "significant_high": [20, None]
        }
    },
    {
        "test": "Erythropoietin; Epo",
        "ranges": {
            "significant_low": [None, 3],
            "borderline_low": [3, 4],
            "normal": [4, 24],
            "borderline_high": [25, 30],
            "significant_high": [30, None]
        }
    },
    {
        "test": "Psa (Prostate-Specific Antigen) Free",
        "ranges": {
            "normal": [None, 4.0],
            "borderline_high": [4, 10],
            "significant_high": [10, None]
        }
    },
    {
        "test": "Psa (Prostate-Specific Antigen) Free Percent",
        "ranges": {
            "significant_low": [None, "15%"],
            "borderline_low": ["15%", "25%"],
            "normal": ["25%", None]
        }
    },
    {
        "test": "Bicarbonate, Serum",
        "ranges": {
            "significant_low": [None, 20],
            "borderline_low": [20, 21],
            "normal": [22, 28],
            "borderline_high": [29, 32],
            "significant_high": [32, None]
        }
    },
    {
        "test": "Immunoglobulin IgG, Serum",
        "ranges": {
            "significant_low": [None, 500],
            "borderline_low": [500, 699],
            "normal": [700, 1600],
            "borderline_high": [1601, 2000],
            "significant_high": [2000, None]
        }
    },
    {
        "test": "Kappa / Lambda Light Chains Free, Serum",
        "ranges": {
            "significant_low": {
                "Kappa": [None, 2.0],
                "Lambda": [None, 4.0]
            },
            "borderline_low": {
                "Kappa": [2.0, 3.2],
                "Lambda": [4.0, 5.6]
            },
            "normal": {
                "Kappa": [3.3, 19.4],
                "Lambda": [5.7, 26.3]
            },
            "borderline_high": {
                "Kappa": [19.5, 25.0],
                "Lambda": [26.4, 30.0]
            },
            "significant_high": {
                "Kappa": [25.0, None],
                "Lambda": [30.0, None]
            }
        }
    },
    {
        "test": "Protein Electrophoresis & Immunotyping, Serum",
        "ranges": {
            "significant_low": {
                "Albumin": [None, 3.0],
                "Globulin": [None, 1.5]
            },
            "borderline_low": {
                "Albumin": [3.0, 3.4],
                "Globulin": [1.5, 1.9]
            },
            "normal": {
                "Albumin": [3.5, 5.0],
                "Globulin": [2.0, 3.5]
            },
            "borderline_high": {
                "Albumin": [5.1, 6.0],
                "Globulin": [3.6, 4.5]
            },
            "significant_high": {
                "Albumin": [6.0, None],
                "Globulin": [4.5, None]
            }
        }
    },
    {
        "test": "Hepatitis B Viral DNA (HBV DNA) Quantitative Real Time PCR",
        "ranges": {
            "normal": [None, 20],
            "borderline_high": [20, 20000],
            "significant_high": [20000, None]
        }
    },
    {
        "test": "Copper, Serum",
        "ranges": {
            "significant_low": [None, 50],
            "borderline_low": [50, 70],
            "normal": [70, 150],
            "borderline_high": [150, 200],
            "significant_high": [200, None]
        }
    },
    {
        "test": "Zinc, Serum / Plasma",
        "ranges": {
            "significant_low": [None, 50],
            "borderline_low": [50, 60],
            "normal": [60, 120],
            "borderline_high": [121, 150],
            "significant_high": [150, None]
        }
    },
    {
        "test": "Aso Titer/Anti Streptolysin O Titer",
        "ranges": {
            "normal": [None, 200],
            "borderline_high": [200, 400],
            "significant_high": [400, None]
        }
    },
    {
        "test": "Immune Deficiency Panel 4; CD4 Counts",
        "ranges": {
            "significant_low": [None, 200],
            "borderline_low": [200, 499],
            "normal": [500, 1500],
            "borderline_high": [1501, 2000],
            "significant_high": [2000, None]
        }
    },
    {
        "test": "GDH (Glutamate Dehydrogenase)",
        "ranges": {
            "normal": [0, 10],
            "borderline_high": [10, 20],
            "significant_high": [20, None]
        }
    },
    {
        "test": "Fecal Calprotectin",
        "ranges": {
            "normal": [None, 100],
            "borderline_high": [100, 500],
            "significant_high": [500, None]
        }
    },
    {
        "test": "Angiotensin Converting Enzyme; ACE",
        "ranges": {
            "normal": [0, 52],
            "borderline_high": [53, 75],
            "significant_high": [75, None]
        }
    },
    {
        "test": "Cardiolipin Antibody IgG",
        "ranges": {
            "normal": [None, 20],
            "borderline_high": [20, 80],
            "significant_high": [80, None]
        }
    },
    {
        "test": "Cardiolipin Antibody IgM",
        "ranges": {
            "normal": [None, 20],
            "borderline_high": [20, 80],
            "significant_high": [80, None]
        }
    },
    {
        "test": "Protein S Functional / Activity",
        "ranges": {
            "significant_low": [None, 20],
            "borderline_low": [20, 40],
            "normal": [40, 150]
        }
    },
    {
        "test": "Protein C Functional",
        "ranges": {
            "significant_low": [None, 20],
            "borderline_low": [20, 40],
            "normal": [40, 150]
        }
    },
    {
        "test": "Haptoglobin",
        "ranges": {
            "significant_low": [None, 10],
            "borderline_low": [10, 20],
            "normal": [20, 300],
            "borderline_high": [300, 400],
            "significant_high": [400, None]
        }
    },
    {
        "test": "ACTH",
        "ranges": {
            "significant_low": [None, 4],
            "borderline_low": [4, 8],
            "normal": [8, 80],
            "borderline_high": [80, 200],
            "significant_high": [200, None]
        }
    },
    {
        "test": "Bile Acids Total, Serum",
        "ranges": {
            "normal": [None, 20],
            "borderline_high": [20, 40],
            "significant_high": [40, None]
        }
    },
    {
        "test": "Beta 2 Glycoprotein 1 Panel IgG, IgM & IgA",
        "ranges": {
            "normal": [None, 35],
            "borderline_high": [35, 80],
            "significant_high": [80, None]
        }
    },
    {
        "test": "Cardiolipin Antibodies Panel IgG, IgA & IgM",
        "ranges": {
            "normal": [None, 40],
            "borderline_high": [40, 80],
            "significant_high": [80, None]
        }
    },
    {
        "test": "ABG - %So2C",
        "ranges": {
            "significant_low": [None, 75],
            "borderline_low": [75, 89],
            "normal": [90, 100]
        }
    },
    {
        "test": "ABG - A-Ado2",
        "ranges": {
            "normal": [None, 30],
            "borderline_high": [30, 50],
            "significant_high": [50, None]
        }
    },
    {
        "test": "ABG - Beb",
        "ranges": {
            "significant_low": [None, -5],
            "borderline_low": [-5, -3],
            "normal": [-2, 2],
            "borderline_high": [3, 5],
            "significant_high": [5, None]
        }
    },
    {
        "test": "ABG - Beecf",
        "ranges": {
            "significant_low": [None, -5],
            "borderline_low": [-5, -3],
            "normal": [-2, 2],
            "borderline_high": [3, 5],
            "significant_high": [5, None]
        }
    },
    {
        "test": "ABG - Hco3",
        "ranges": {
            "significant_low": [None, 15],
            "borderline_low": [15, 18],
            "normal": [18, 30],
            "borderline_high": [31, 34],
            "significant_high": [35, None]
        }
    },
    {
        "test": "ABG - O2Ct",
        "ranges": {
            "significant_low": [None, 10],
            "borderline_low": [10, 12],
            "normal": [12, 22],
            "borderline_high": [24, 28],
            "significant_high": [28, None]
        }
    },
    {
        "test": "ABG - Pco2",
        "ranges": {
            "significant_low": [None, 25],
            "borderline_low": [25, 30],
            "normal": [30, 50],
            "borderline_high": [50, 60],
            "significant_high": [60, None]
        }
    },
    {
        "test": "ABG - Ph",
        "ranges": {
            "significant_low": [None, 7.2],
            "borderline_low": [7.2, 7.3],
            "normal": [7.3, 7.5],
            "borderline_high": [7.5, 7.6],
            "significant_high": [7.60, None]
        }
    },
    {
        "test": "ABG - Po2",
        "ranges": {
            "significant_low": [None, 40],
            "borderline_low": [40, 60],
            "normal": [60, 100]
        }
    },
    {
        "test": "ABG - Sbc",
        "ranges": {
            "significant_low": [None, 15],
            "borderline_low": [15, 18],
            "normal": [18, 30],
            "borderline_high": [30, 35],
            "significant_high": [35, None]
        }
    },
    {
        "test": "ABG - Tco2",
        "ranges": {
            "significant_low": [None, 15],
            "borderline_low": [15, 18],
            "normal": [18, 35],
            "borderline_high": [30, 35],
            "significant_high": [40, None]
        }
    },
    {
        "test": "Beta - 2 - Microglobulin, Serum",
        "ranges": {
            "normal": [0.7, 3],
            "borderline_high": [3.1, 5],
            "significant_high": [5, None]
        }
    },
    {
        "test": "Osmolality, Urine",
        "ranges": {
            "significant_low": [None, 100],
            "borderline_low": [100, 200],
            "normal": [200, 1100],
            "borderline_high": [1100, 1300],
            "significant_high": [1300, None]
        }
    },
    {
        "test": "GAD-65 (Glutamic Acid Decarboxylase- 65), IgG",
        "ranges": {
            "normal": [None, 15],
            "borderline_high": [15, 200],
            "significant_high": [200, None]
        }
    },
    {
        "test": "Fecal Elastase",
        "ranges": {
            "significant_low": [None, 100],
            "borderline_low": [100, 199],
            "normal": [200, None]
        }
    },
    {
        "test": "Bleeding Time (BT)",
        "ranges": {
            "normal": [1, 7],
            "significant_high": [7, None]
        }
    },
    {
        "test": "Clotting Time (CT)",
        "ranges": {
            "normal": [5, 15],
            "significant_high": [15, None]
        }
    },
    {
        "test": "Troponin-T",
        "ranges": {
            "normal": [None, 0.05],
            "borderline_high": [0.05, 0.5],
            "significant_high": [0.5, None]
        }
    },
    {
        "test": "Chromogranin A (CgA)",
        "ranges": {
            "normal": [None, 100],
            "borderline_high": [100, 500],
            "significant_high": [500, None]
        }
    },
    {
        "test": "Gastrin",
        "ranges": {
            "normal": [None, 100],
            "borderline_high": [100, 500],
            "significant_high": [500, None]
        }
    },
    {
        "test": "Urine Calcium (Spot)",
        "ranges": {
            "significant_low": [None, 10],
            "borderline_low": [10, 30],
            "normal": [30, 400],
            "borderline_high": [400, 500],
            "significant_high": [500, None]
        }
    },
    {
        "test": "Vitamin B6; Pyridoxine",
        "ranges": {
            "significant_low": [None, 2],
            "borderline_low": [2, 4],
            "normal": [4, 50],
            "borderline_high": [50, 100],
            "significant_high": [100, None]
        }
    },
    {
        "test": "Valproic Acid; Valproate",
        "ranges": {
            "significant_low": [None, 50],
            "normal": [50, 100],
            "borderline_high": [100, 150],
            "significant_high": [150, None]
        }
    },
    {
        "test": "HIV 1 RNA Quantitative Real Time PCR",
        "ranges": {
            "normal": [None, 20],
            "borderline_high": [20, 1000],
            "significant_high": [1000, None]
        }
    },
    {
        "test": "Interleukin-6 (IL-6)",
        "ranges": {
            "normal": [None, 7],
            "borderline_high": [7, 30],
            "significant_high": [30, None]
        }
    },
    {
        "test": "Beta 2 Glycoprotein 1, IgG",
        "ranges": {
            "normal": [None, 40],
            "borderline_high": [40, 80],
            "significant_high": [80, None]
        }
    },
    {
        "test": "Beta 2 Glycoprotein 1, IgM",
        "ranges": {
            "normal": [None, 40],
            "borderline_high": [40, 80],
            "significant_high": [80, None]
        }
    },
    {
        "test": "Des-Gamma Carboxy Prothrombin (DCP)/PIVKA II, Serum",
        "ranges": {
            "normal": [None, 7.5],
            "borderline_high": [7.6, 40],
            "significant_high": [40, None]
        }
    },
    {
        "test": "Neuron Specific Enolase (NSE), Serum",
        "ranges": {
            "normal": [None, 12.5],
            "borderline_high": [12.6, 40],
            "significant_high": [40, None]
        }
    },
    {
        "test": "C-Peptide Fasting, Serum",
        "ranges": {
            "significant_low": [None, 0.8],
            "normal": [0.8, 3.1],
            "significant_high": [3.1, None]
        }
    },
    {
        "test": "Immunoglobulin IgM, Serum",
        "ranges": {
            "significant_low": [None, 30],
            "borderline_low": [30, 40],
            "normal": [40, 300],
            "borderline_high": [300, 500],
            "significant_high": [500, None]
        }
    },
    {
        "test": "Immunoglobulin IgA, Serum",
        "ranges": {
            "significant_low": [None, 50],
            "borderline_low": [50, 70],
            "normal": [70, 500],
            "borderline_high": [500, 700],
            "significant_high": [700, None]
        }
    },
    {
        "test": "Urine Protein (Spot)",
        "ranges": {
            "normal": [None, 150],
            "borderline_high": [151, 500],
            "significant_high": [500, None]
        }
    },
    {
        "test": "Paras Labs 25 Hydroxy Vitamin D-Total",
        "ranges": {
            "significant_low": [None, 10],
            "borderline_low": [10, 19],
            "normal": [20, 50],
            "borderline_high": [51, 70],
            "significant_high": [70, None]
        }
    },
    {
        "test": "C3 Complement",
        "ranges": {
            "significant_low": [None, 30],
            "borderline_low": [30, 60],
            "normal": [60, 180],
            "significant_high": [180, None]
        }
    },
    {
        "test": "C4 Complement",
        "ranges": {
            "significant_low": [None, 3],
            "normal": [3, 7],
            "borderline_high": [7, 40]
        }
    },
    {
        "test": "Immunoglobulin IgG Subclass 4 (Outsource)",
        "ranges": {
            "significant_low": [None, 4],
            "borderline_low": [4, 7],
            "normal": [7, 500],
            "borderline_high": [300, 500],  # Note potential overlap in range values
            "significant_high": [500, None]
        }
    },
    {
        "test": "Allergy Specific IgG: Aspergillus Fumigatus",
        "ranges": {
            "normal": [None, 50],
            "borderline_high": [50, 100],
            "significant_high": [100, None]
        }
    },
    {
        "test": "C-Reactive Proteins (CRP) - COVID",
        "ranges": {
            "normal": [None, 10],
            "borderline_high": [10, 40],
            "significant_high": [40, None]
        }
    },
    {
        "test": "Methotrexate",
        "ranges": {
            "normal": [None, 0.6],
            "borderline_high": [0.6, 1.0],
            "significant_high": [1.0, None]
        }
    },
    {
        "test": "Urine Sugar",
        "ranges": {
            "normal": [None, 15],
            "significant_high": [15, None]
        }
    },
    {
        "test": "Transferrin",
        "ranges": {
            "significant_low": [None, 100],
            "borderline_low": [100, 150],
            "normal": [200, 360],
            "significant_high": [360, None]
        }
    },
    {
        "test": "Pleural Fluid for LDH",
        "ranges": {
            "normal": [None, 200],
            "significant_high": [200, None]
        }
    },
    {
        "test": "Fluid for Glucose",
        "ranges": {
            "significant_low": [None, 20],
            "borderline_low": [20, 40],
            "normal": [40, 80],
            "significant_high": [80, None]
        }
    },
    {
        "test": "Antithrombin Activity, Functional",
        "ranges": {
            "significant_low": [None, 50],
            "borderline_low": [50, 70],
            "normal": [70, 120],
            "significant_high": [120, None]
        }
    },
    {
        "test": "Anti - Ds DNA Antibody, EIA",
        "ranges": {
            "normal": [None, 30],
            "borderline_high": [31, 100],
            "significant_high": [100, None]
        }
    },
    {
        "test": "(1, 3)-Beta-D-Glucan (BDG) (Outsource)",
        "ranges": {
            "normal": [None, 80],
            "borderline_high": [80, 200],
            "significant_high": [200, None]
        }
    },
    {
        "test": "Growth Hormone; GH",
        "ranges": {
            "normal": [None, 5],
            "significant_high": [5, None]
        }
    },
    {
        "test": "IGF - I; Somatomedin - C",
        "ranges": {
            "significant_low": [None, 100],
            "normal": [100, 300],
            "significant_high": [300, None]
        }
    },
    {
        "test": "Cortisol (PM)",
        "ranges": {
            "significant_low": [None, 3],
            "borderline_low": [3, 11],
            "normal": [11, 20],
            "significant_high": [20, None]
        }
    },
    {
        "test": "Urine Potassium (Spot)",
        "ranges": {
            "significant_low": [None, 10],
            "borderline_low": [10, 15],
            "normal": [15, 150],
            "borderline_high": [150, 175],
            "significant_high": [175, None]
        }
    },
    {
        "test": "Peritoneal Fluid - Diff Count, Monocytes",
        "ranges": {
            "normal": [None, "10%"],
            "borderline_high": ["10%", "25%"],
            "significant_high": ["25%", None]
        }
    },
    {
        "test": "Peritoneal Fluid - Diff. Count, Eosinophils",
        "ranges": {
            "normal": [None, "4%"],
            "borderline_high": ["4%", "10%"],
            "significant_high": ["10%", None]
        }
    },
    {
        "test": "Peritoneal Fluid - Diff. Count, Lymphocytes",
        "ranges": {
            "normal": [None, "40%"],
            "borderline_high": ["40%", "60%"],
            "significant_high": ["60%", None]
        }
    },
    {
        "test": "Peritoneal Fluid - Diff. Count, Neutrophils",
        "ranges": {
            "normal": [None, 350],
            "borderline_high": [350, 1000],
            "significant_high": [1000, None]
        }
    },
    {
        "test": "Peritoneal Fluid - WBC Count",
        "ranges": {
            "normal": [None, 750],
            "borderline_high": [750, 2500],
            "significant_high": [2500, None]
        }
    },
    {
        "test": "Hepatitis C Viral RNA (HCV RNA) Quantitative Real Time PCR",
        "ranges": {
            "normal": [None, 20],
            "borderline_high": [20, 100000],
            "significant_high": [1000000, None]
        }
    },
    {
        "test": "Folate (Folic Acid), RBC",
        "ranges": {
            "significant_low": [None, 200],
            "borderline_low": [200, 279],
            "normal": [280, 791],
            "borderline_high": [792, 1000],
            "significant_high": [1000, None]
        }
    },
    {
        "test": "Urine Chloride (Spot)",
        "ranges": {
            "significant_low": [None, 60],
            "borderline_low": [60, 90],
            "normal": [90, 300],
            "borderline_high": [300, 350],
            "significant_high": [350, None]
        }
    },
    {
        "test": "LDH (Fluid)",
        "ranges": {
            "normal": [None, 300],
            "borderline_high": [300, 600],
            "significant_high": [600, None]
        }
    },
    {
        "test": "Paras Labs Partial Thromboplastin Time (APTT)",
        "ranges": {
            "normal": [25, 35],
            "borderline_high": [36, 45],
            "significant_high": [45, None]
        }
    },
    {
        "test": "Vitamin D 1, 25-Dihydroxy",
        "ranges": {
            "significant_low": [None, 10],
            "borderline_low": [10, 19],
            "normal": [20, 60],
            "borderline_high": [61, 100],
            "significant_high": [100, None]
        }
    },
    {
        "test": "Stem Cell CD34 Count",
        "ranges": {
            "normal": [2, 10],
            "borderline_high": [10, 100],
            "significant_high": [100, None]
        }
    },
    {
        "test": "C3 & C4 Complement Panel",
        "ranges": {
            "significant_low": {
                "C3": [None, 60],
                "C4": [None, 10]
            },
            "borderline_low": {
                "C3": [60, 89],
                "C4": [10, 15]
            },
            "normal": {
                "C3": [90, 180],
                "C4": [10, 40]
            },
            "borderline_high": {
                "C3": [181, 200],
                "C4": [41, 50]
            },
            "significant_high": {
                "C3": [200, None],
                "C4": [50, None]
            }
        }
    },
    {
        "test": "Ceruloplasmin",
        "ranges": {
            "significant_low": [None, 5],
            "borderline_low": [5, 15],
            "normal": [15, 40],
            "significant_high": [40, None]
        }
    },
    {
        "test": "Epstein Barr Virus (EBV), Quantitative PCR",
        "ranges": {
            "normal": [None, 500],
            "borderline_high": [500, 1000],
            "significant_high": [1000, None]
        }
    },
    {
        "test": "Copper, 24 Hour Urine",
        "ranges": {
            "normal": [None, 50],
            "borderline_high": [50, 200],
            "significant_high": [200, None]
        }
    },
    {
        "test": "Tissue Transglutaminase (TTG) Antibody IgG (Outsource)",
        "ranges": {
            "normal": [None, 30],
            "borderline_high": [30, 100],
            "significant_high": [100, None]
        }
    },
    {
        "test": "Metanephrines Free, Plasma",
        "ranges": {
            "normal": [None, 150],
            "borderline_high": [150, 400],
            "significant_high": [400, None]
        }
    },
    {
        "test": "Paras Labs TSH (Ultrasensitive)",
        "ranges": {
            "significant_low": [None, 0.3],
            "borderline_low": [0.3, 0.4],
            "normal": [0.4, 5.0],
            "borderline_high": [5.1, 10],
            "significant_high": [10, None]
        }
    },
    {
        "test": "Paras Labs Total IgE",
        "ranges": {
            "significant_low": [None, 100],
            "normal": [100, 500],
            "significant_high": [500, None]
        }
    },
    {
        "test": "Fluid for Protein",
        "ranges": {
            "significant_low": [None, 3.6],
            "normal": [3.6, 4],
            "significant_high": [4, None]
        }
    },
    {
        "test": "Colony Count",
        "ranges": {
            "normal": [None, 1000],
            "borderline_high": [1000, 10000],
            "significant_high": [10000, None]
        }
    },
    {
        "test": "Paras Labs Creatinine Kinase Mb Mass",
        "ranges": {
            "normal": [None, 10],
            "borderline_high": [10, 20],
            "significant_high": [20, None]
        }
    },
    {
        "test": "CA-125 (Cancer Antigen 125)",
        "ranges": {
            "normal": [None, 35],
            "significant_high": [35, None]
        }
    },
    {
        "test": "Myoglobin Serum",
        "ranges": {
            "normal": [None, 100],
            "borderline_high": [100, 500],
            "significant_high": [500, None]
        }
    },
    {
        "test": "PTHrP; Parathyroid Hormone Related Protein",
        "ranges": {
            "normal": [None, 2],
            "significant_high": [2, None]
        }
    },
    {
        "test": "Immunophenotyping By Flow Cytometry, CD4",
        "ranges": {
            "significant_low": [None, 200],
            "borderline_low": [200, 499],
            "normal": [500, 1500]
        }
    },
    {
        "test": "AFP (Alpha Fetoprotein): Maternal",
        "ranges": {
            "normal": [None, 10],
            "borderline_high": [10, 150],
            "significant_high": [150, None]
        }
    },
    {
        "test": "Free Plasma Haemoglobin",
        "ranges": {
            "normal": [None, 10],
            "borderline_high": [10, 50],
            "significant_high": [50, None]
        }
    },
    {
        "test": "Fluid Albumin",
        "ranges": {
            "significant_low": [None, 1.1],
            "normal": [1.1, 3.5]
        }
    },
    {
        "test": "SAAG (Serum Ascites-Albumin Gradient)",
        "ranges": {
            "significant_low": [None, 1.1],
            "significant_high": [1.1, None]
        }
    },
    {
        "test": "Isohemagglutinin Titres",
        "ranges": {
            "normal": [None, 1.8],
            "borderline_high": [1.16, 1.32],
            "significant_high": [1.32, None]
        }
    },
    {
        "test": "Factor IX Functional",
        "ranges": {
            "significant_low": [None, 5],
            "borderline_low": [5, 40],
            "normal": [40, 150]
        }
    },
    {
        "test": "Mantoux Test Induration",
        "ranges": {
            "normal": [None, 5],
            "borderline_high": [5, 10],
            "significant_high": [10, None]
        }
    },
    {
        "test": "5α-Dihydrotestosterone (5α-DHT)",
        "ranges": {
            "significant_low": [None, 10],
            "borderline_low": [10, 20],
            "normal": [20, 120],
            "borderline_high": [120, 150],
            "significant_high": [150, None]
        }
    },
    {
        "test": "Phenytoin",
        "ranges": {
            "significant_low": [None, 5],
            "borderline_low": [5, 9],
            "normal": [10, 20],
            "borderline_high": [21, 30],
            "significant_high": [30, None]
        }
    },
    {
        "test": "VMA (Vanillyl Mandelic Acid), 24 Hour Urine",
        "ranges": {
            "normal": [None, 10],
            "borderline_high": [10, 30],
            "significant_high": [30, None]
        }
    },
    {
        "test": "eGFR",
        "ranges": {
            "significant_low": [None, 30],
            "borderline_low": [30, 80],
            "normal": [80, None]
        }
    },
    {
        "test": "Amylase (Fluid)",
        "ranges": {
            "normal": [None, 150],
            "borderline_high": [150, 500],
            "significant_high": [500, None]
        }
    },
    {
        "test": "Lactate (Plasma)",
        "ranges": {
            "normal": [None, 4],
            "borderline_high": [4, 8],
            "significant_high": [8, None]
        }
    },
    {
        "test": "Tacrolimus; FK506",
        "ranges": {
            "significant_low": [None, 5],
            "normal": [5, 15],
            "borderline_high": [15, 20],
            "significant_high": [20, None]
        }
    },
    {
        "test": "LDH Total (Serum) - COVID",
        "ranges": {
            "normal": [None, 400],
            "borderline_high": [400, 600],
            "significant_high": [600, None]
        }
    },
    {
        "test": "Creatinine Kinase MB",
        "ranges": {
            "normal": [None, 10],
            "borderline_high": [10, 20],
            "significant_high": [20, None]
        }
    },
    {
        "test": "Paras Labs Vitamin B12",
        "ranges": {
            "significant_low": [None, 100],
            "borderline_low": [100, 150],
            "normal": [150, 900]
        }
    },
    {
        "test": "Paras Labs Blood Sugar (Random)",
        "ranges": {
            "significant_low": [None, 60],
            "normal": [60, 200],
            "borderline_high": [200, 300],
            "significant_high": [300, None]
        }
    },
    {
        "test": "Paras Labs Prothrombin Time (With INR)",
        "ranges": {
            "normal": [None, 15],
            "borderline_high": [15, 25],
            "significant_high": [25, None]
        }
    },
    {
        "test": "C1 Esterase Inhibitor Protein Quantitation; C1 Inhibitor Quantitative",
        "ranges": {
            "significant_low": [None, 10],
            "borderline_low": [10, 15],
            "normal": [15, 40],
            "significant_high": [40, None]
        }
    },
    {
        "test": "ADAMTS13 Activity",
        "ranges": {
            "significant_low": [None, 20],
            "borderline_low": [20, 50],
            "normal": [50, 100],
            "significant_high": [100, None]
        }
    },
    {
        "test": "Ascitic/Peritoneal Fluid for LDH",
        "ranges": {
            "normal": [None, 400],
            "borderline_high": [400, 600],
            "significant_high": [600, None]
        }
    },
    {
        "test": "Digoxin",
        "ranges": {
            "significant_low": [None, 0.5],
            "normal": [0.5, 2.0],
            "borderline_high": [2, 2.5],
            "significant_high": [2.5, None]
        }
    },
    {
        "test": "Paras Labs Creatinine Kinase (CPK)",
        "ranges": {
            "normal": [None, 400],
            "borderline_high": [400, 1000],
            "significant_high": [1000, None]
        }
    },
    {
        "test": "BNP B-Type Natriuretic Peptide",
        "ranges": {
            "normal": [None, 100],
            "borderline_high": [100, 300],
            "significant_high": [300, None]
        }
    },
    {
        "test": "Paras Labs Hepatitis B Virus Surface Antibody (HBsAb) Quantitative",
        "ranges": {
            "normal": [None, 10],
            "significant_high": [10, None]
        }
    },
    {
        "test": "Lithium",
        "ranges": {
            "significant_low": [None, 0.3],
            "borderline_low": [0.3, 0.5],
            "normal": [0.6, 1.2],
            "borderline_high": [1.3, 2.0],
            "significant_high": [2.0, None]
        }
    },
    {
        "test": "Circulating Tumor Cells Enumeration",
        "ranges": {
            "normal": [None, 1],
            "significant_high": [1, None]
        }
    },
    {
        "test": "Paras Labs Total Proteins with Albumin/Globulin (A/G)",
        "ranges": {
            "significant_low": [None, 5],
            "borderline_low": [5, 5.9],
            "normal": [6, 8.3],
            "borderline_high": [8.4, 9.0],
            "significant_high": [9.0, None]
        }
    },
    {
        "test": "Paras Labs Troponin I (Quantitative)",
        "ranges": {
            "normal": [None, 0.04],
            "borderline_high": [0.04, 0.5],
            "significant_high": [0.5, None]
        }
    },
    {
        "test": "HE4; Human Epididymis Protein 4",
        "ranges": {
            "normal": [None, 140],
            "significant_high": [140, None]
        }
    },
    {
        "test": "17 - Hydroxyprogesterone (17-OHP)",
        "ranges": {
            "normal": [None, 150],
            "borderline_high": [150, 500],
            "significant_high": [500, None]
        }
    },
    {
        "test": "Paras Labs Cortisol AM",
        "ranges": {
            "significant_low": [None, 1],
            "borderline_low": [1, 3],
            "normal": [3, 35],
            "borderline_high": [35, 50],
            "significant_high": [50, None]
        }
    },
    {
        "test": "Paras Labs Urine Protein/Creatinine Ratio",
        "ranges": {
            "normal": [None, 250],
            "borderline_high": [250, 850],
            "significant_high": [850, None]
        }
    },
    {
        "test": "Lead, Blood",
        "ranges": {
            "normal": [None, 9],
            "borderline_high": [9, 20],
            "significant_high": [20, None]
        }
    },
    {
        "test": "Serum Ascites Albumin Gradient (SAAG)",
        "ranges": {
            "significant_low": [None, 1.1],
            "significant_high": [1.1, None]
        }
    },
    {
        "test": "Factor V, Functional",
        "ranges": {
            "significant_low": [None, 20],
            "borderline_low": [20, 40],
            "normal": [40, 150]
        }
    },
    {
        "test": "Phospholipase A2 Receptor Antibody (PLA2R) Quantitative",
        "ranges": {
            "normal": [None, 20],
            "borderline_high": [20, 50],
            "significant_high": [50, None]
        }
    }

]



OPD_STRING_UNIVERSE =['Troponin-T', 'Liquefaction', 'Ph', 'Rbcs', 'Viscosity', 'Urine Analysis - Appearance', 'Urine Analysis - Bilirubin', 'Urine Analysis - Casts', 'Urine Analysis - Color', 'Urine Analysis - Crystals', 'Urine Analysis - Glucose', 'Urine Analysis - Ketone', 'Urine Analysis - Nitrite', 'Urine Analysis - Others', 'Urine Analysis - Protein', 'Urine Analysis - Urobilinogen', 'Urine Analysis- Blood', 'Blood Group (Abo And Rh)', 'Others', 'Stool Analysis - Consistency', 'Stool Analysis - Mucus', 'Stool Analysis- Blood', 'Stool Analysis- Color', 'Stool Analysis- Cysts', 'Stool Analysis- Ova', 'Stool Analysis- Parasite', 'Stool Analysis- Trophozoites', 'Stool For Reducing Substance', 'Stool For Occult Blood', 'Malaria Antigen (Rapid Test For Malaria)', 'Appearance', 'Semen Fructose(Qualitative)', 'Advice', 'Dlc', 'Hemoparasites', 'Impression', 'Mean Platelet Volume (Mpv)', 'Peripheral Smear Examination', 'Platelets', 'Wbcs', 'Color', 'Eosinophils', 'Glucose', 'Pre Dialysis Screen (Eci) Result', 'Rpr / Vdrl (Rapid Plasma Reagin)', 'Gdh (Glutamate Dehydrogenase)', 'Toxin A', 'Toxin B', 'Aso Titer/Anti Streptolysin O Titer', 'Urine Sugar (1 Hr.)', 'Urine Sugar (2 Hrs.)', 'Urine Sugar (Fasting)', 'Typhoid Igm', 'Malarial Parasite', 'Abg - Beb', 'Abg - A-Ado2', 'Abg - %So2C', 'Abg - Tco2', 'Abg - Po2', 'Abg - Ph', 'Abg - Pco2', 'Abg - O2Ct', 'Abg - Hco3', 'Abg - Beecf', 'Abg - Sbc', 'Epithelial Cells', 'Round Cells/Pus Cells', 'Urine Analysis - Epithelial Cells', 'Urine Analysis - Pus Cells', 'Urine Analysis - Glucose', 'Urine Analysis - Rbc', 'Urine Analysis - Protein', 'Stool Analysis- Pus Cells', 'Rbcs', 'Stool Analysis- Rbcs', 'Estradiol (E2)', 'Progesterone', 'Lh- Leutenizing Hormone', 'Lymphocytes', 'Monocytes', 'Neutrophils', 'Wbc Count', 'Testosterone, Total', 'Progesterone, Serum', 'Fsh- Follicle Stimulating Hormone', 'Hcg - Beta Specific', 'Hiv I & Ii Screening', 'Anti Thyroid (Anti-Tpo)', 'Hiv 1 & 2 Antibody (Rapid)', 'Dengue Ns1 Ag(Elfa)']
