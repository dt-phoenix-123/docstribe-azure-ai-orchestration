#!/usr/bin/env python3
"""Docstribe summarisation helpers backed by CrewAI workflows and LLM APIs."""

from __future__ import annotations

import datetime
import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
# from crewai import Agent, Task, Crew, Process
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_groq import ChatGroq
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from openai import OpenAI

from docstribe_agent_config import (
    MASTER_PROMPTS,
    SUMMARIZE_FORMAT,
    SYS_COHORT_PROMPT,
    OPENAI_API_KEY,
    GROQ_API_KEY,
    DEEPSEEK_API_KEY,
    MISTRAL_API_KEY,
    OPD_RANGE_UNIVERSE,
    OPD_STRING_UNIVERSE,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_GROK_BASE_URL,
    AZURE_GROK_DEPLOYMENT,
)


# ---------------------------------------------------------------------------
# LLM initialisation helpers (lazy instantiation with graceful fallbacks)
# ---------------------------------------------------------------------------


def _build_chat_openai(model: str, **kwargs: Any) -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Export it or set it in the environment before using summariser endpoints."
        )
    return ChatOpenAI(api_key=OPENAI_API_KEY, model=model, **kwargs)


def _build_chat_groq(model: str) -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Export it or set it in the environment before using Groq-backed reasoning models."
        )
    return ChatGroq(model_name=model, groq_api_key=GROQ_API_KEY)


def _build_chat_deepseek(model: str) -> ChatDeepSeek:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not configured. Export it or set it in the environment before using DeepSeek-backed reasoning models."
        )
    return ChatDeepSeek(model_name=model, api_key=DEEPSEEK_API_KEY)


def _build_chat_mistral(model: str, **kwargs: Any) -> ChatMistralAI:
    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY is not configured. Export it or set it in the environment before using Mistral-backed agents."
        )
    return ChatMistralAI(api_key=MISTRAL_API_KEY, model=model, **kwargs)


try:
    reasoning_llm = _build_chat_groq("deepseek-r1-distill-llama-70b")
except RuntimeError:
    reasoning_llm = None

try:
    reasoning_llm_openai = _build_chat_openai(model="o3")
except RuntimeError:
    reasoning_llm_openai = None

try:
    reasoning_llm_v2 = _build_chat_deepseek("deepseek-reasoner")
except RuntimeError:
    reasoning_llm_v2 = None

try:
    mistral_llm = _build_chat_mistral(
        model="mistral-small-latest",
        temperature=0.56,
        top_p=0.1,
        max_retries=5,
        max_tokens=None,
    )
except RuntimeError:
    mistral_llm = None

try:
    openai_llm = _build_chat_openai(
        model="gpt-4.1",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
except RuntimeError:
    openai_llm = None


def _require_llm(llm: Optional[Any], name: str) -> Any:
    if llm is None:
        raise RuntimeError(
            f"The LLM '{name}' is not initialised. Ensure the relevant API key is set before invoking this operation."
        )
    return llm


def _grok_base_url() -> str:
    return AZURE_GROK_BASE_URL.rstrip("/") + "/"


@lru_cache(maxsize=1)
def _get_grok_client() -> OpenAI:
    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not configured for summariser usage")
    return OpenAI(base_url=_grok_base_url(), api_key=AZURE_OPENAI_API_KEY)


def _grok_chat_completion(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    client = _get_grok_client()
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    completion = client.chat.completions.create(
        model=AZURE_GROK_DEPLOYMENT,
        messages=messages,
        temperature=temperature,
    )
    message = completion.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, list):
        return "".join(
            segment.get("text", "") if isinstance(segment, dict) else str(segment)
            for segment in content
        )
    if hasattr(message, "content") and isinstance(message.content, str):
        return message.content
    if isinstance(message, dict):
        return message.get("content", "")
    return content or ""


# ---------------------------------------------------------------------------
# Agent definitions (lazily created when prerequisites are available)
# ---------------------------------------------------------------------------

# if openai_llm is not None:
#     medical_summarization_agent = Agent(
#         role="Medical History summarizer",
#         goal="Understand the medical history and summarize it in a way that helps a healthcare professional to deliver a better care",
#         backstory=(
#             "You are a harvard graduated medical person and you want that a patient's medical context is readily available with the care team"
#         ),
#         verbose=False,
#         memory=False,
#         llm=openai_llm,
#         allow_delegation=False,
#     )
# else:  # pragma: no cover - depends on runtime config
#     medical_summarization_agent = None

# if openai_llm is not None:
#     pdf_agent = Agent(
#         role="Radiologist",
#         goal="Analyze the given pdf report dump and summarize it in crisp format",
#         verbose=True,
#         memory=False,
#         backstory=(
#             "As a radiologist from harvard, your mission is to extract critical medical information from PDF documents and provide concise summaries."
#         ),
#         llm=openai_llm,
#         allow_delegation=False,
#     )
# else:  # pragma: no cover - depends on runtime config
#     pdf_agent = None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def extract_diagnostic_texts(pdf_url: str) -> Dict[str, Any]:
    url = "https://docstribe-parser-kmo2wwblnq-em.a.run.app/extract_texts"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json={"pdf_url": pdf_url}, timeout=120)
    response.raise_for_status()
    return response.json()


def clean_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text()


def mistral_radiology_summarizer_task(pdf_url: str, *, agent: Optional[Agent] = None) -> Task:
    agent = agent or _require_llm(pdf_agent, "pdf_agent")
    pdf_resp = extract_diagnostic_texts(pdf_url)
    description = (
        "Given the radiological findings - {pdf_resp} , extract the text and summarize the main points.".format(
            pdf_resp=pdf_resp
        )
    )
    expected_output = (
        'A json output without personal information. Keep the results strictly in this format - {"findings":[{"procedure_name":<name of the procedure>, "findings":[<finding in the procedure>], "impressions":[<impressions mentioned if any>]}]}.'
    )
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        async_execution=False,
    )


def abnormality_detection_classification(
    test_name: str,
    result: Any,
    gender: str,
    ranges: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    ranges = ranges or OPD_RANGE_UNIVERSE
    if test_name in OPD_STRING_UNIVERSE:
        return {"test": test_name, "value": result}
    if pd.isna(result) or result == "":
        return None

    classification = ""
    try:
        numeric_result = float(result)
    except (ValueError, TypeError):
        match = re.search(r"([<>])?\s*(\d*\.?\d+)", str(result))
        if match:
            numeric_result = float(match.group(2))
        else:
            return {"test": test_name, "value": result, "classification": ""}

    gender_key = gender.lower()
    for test_range in ranges:
        if test_range.get("test", "").lower() != test_name.lower():
            continue
        for class_key, class_range in test_range.get("ranges", {}).items():
            if class_key == "borderline_low":
                classification_key = "mildly_low"
            elif class_key == "borderline_high":
                classification_key = "slightly_above_normal"
            else:
                classification_key = class_key

            if isinstance(class_range, dict):
                class_range = class_range.get(gender_key, [None, None])
            low, high = class_range if isinstance(class_range, (list, tuple)) else (None, None)
            if low is None and high is not None and numeric_result <= high:
                classification = classification_key
                break
            if high is None and low is not None and numeric_result >= low:
                classification = classification_key
                break
            if low is not None and high is not None and low <= numeric_result <= high:
                classification = classification_key
                break
        break

    return {"test": test_name, "value": numeric_result, "classification": classification}


def detect_op_abnormalities(test_values: List[Dict[str, Any]], gender: str = "Male") -> List[Dict[str, Any]]:
    tagged_tests: List[Dict[str, Any]] = []
    gender_key = "males" if gender == "Male" else "females"
    for test_value in test_values:
        test_name = test_value.get("test")
        value = test_value.get("value")
        if value == "-":
            continue
        res = abnormality_detection_classification(test_name, value, gender_key)
        if res is not None:
            tagged_tests.append(res)
    return tagged_tests


def run_crew(medical_summarization_tasks: List[Task], *, agent: Optional[Agent] = None) -> List[Any]:
    agent = agent or _require_llm(medical_summarization_agent, "medical_summarization_agent")
    crew = Crew(
        agents=[agent],
        tasks=medical_summarization_tasks,
        memory=False,
        process=Process.sequential,
        verbose=True,
    )
    crew.kickoff()
    return [task.raw_output for task in medical_summarization_tasks]


def summarize_diagnostics(pdf_url: str) -> Dict[str, Any]:
    agent = _require_llm(pdf_agent, "pdf_agent")
    diagnostic_task = mistral_radiology_summarizer_task(pdf_url, agent=agent)
    crew = Crew(agents=[agent], tasks=[diagnostic_task], memory=False, verbose=True)
    result = crew.kickoff()
    start_idx = result.find("{")
    end_idx = result.rfind("}")
    res = result[start_idx : end_idx + 1]
    try:
        return json.loads(res)
    except json.JSONDecodeError:
        res = res.replace("\\_", "_")
        return json.loads(res)


def cohort_generation_agent(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    llm = _require_llm(reasoning_llm_openai, "reasoning_llm_openai")
    cohort_gen_prompt = (
        """
    Please provide the answer of the cohort in below format - 
    {
        "primary_cohort":{
            "<name_of_the_speciality>":{
            "primary":<name of the primary disease>,
            "secondary":<name of the secondary disease>
            }
        },
        "secondary_cohort":{
            "<name_of_the_speciality>":{
            "primary":<name of the primary disease>,
            "secondary":<name of the secondary disease>
            }

    }
    """
    )
    prompt = [
        SystemMessage(content=SYS_COHORT_PROMPT),
        HumanMessage(content=cohort_gen_prompt),
        HumanMessage(content=f"Given the patient data - {patient_data}"),
    ]
    resp = llm.invoke(prompt)
    raw = resp.content
    start_idx = raw.find("{")
    end_idx = raw.rfind("}")
    json_str = raw[start_idx : end_idx + 1]
    return json.loads(json_str)


def summarize_lab_data_agent(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    visits = patient_data.get("visits", [])[:3]
    meds_list = []
    tests_list = []
    imaging_list = []
    clinical_notes: List[Dict[str, Any]] = []
    for visit in visits:
        meds_list.append(visit.get("medications", ""))
        tests_list.append(visit.get("tests", []))
        imaging_list.append(visit.get("imaging_results", []))
        clinical_notes.append({"visit_date": visit.get("date"), "note": visit.get("clinical_note", "")})

    aggregated_meds = "\n".join(m for m in meds_list if m)
    aggregated_tests = json.dumps(tests_list, ensure_ascii=False)
    aggregated_imaging = json.dumps(imaging_list, ensure_ascii=False)
    clinical_note_blob = json.dumps(clinical_notes, ensure_ascii=False)

    initial_prompt = MASTER_PROMPTS.get("opd_data_summarizer_v2", "")
    formatted_prompt = initial_prompt.format(
        medications=aggregated_meds,
        lab_data=aggregated_tests,
        imaging_results=aggregated_imaging,
        clinical_note=clinical_note_blob,
    )

    raw = _grok_chat_completion(formatted_prompt)
    raw = raw.split("</think>", 1)[-1].strip()
    start_idx = raw.find("{")
    end_idx = raw.rfind("}")
    json_str = raw[start_idx : end_idx + 1]
    return json.loads(json_str)


def summarize_agent(
    medical_text: str,
    discharge_date: str,
    *,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    cleaned_text = clean_html(medical_text)
    initial_prompt = MASTER_PROMPTS.get("discharge_summary_summarizer", "")
    formatted_prompt = initial_prompt.format(
        patient_detail_summary=cleaned_text,
        summarized_json_format=SUMMARIZE_FORMAT,
        discharge_date=discharge_date,
    )
    raw = _grok_chat_completion(formatted_prompt)
    start_idx = raw.find("{")
    end_idx = raw.rfind("}")
    res = raw[start_idx : end_idx + 1]
    return json.loads(res)


__all__ = [
    "reasoning_llm",
    "reasoning_llm_openai",
    "reasoning_llm_v2",
    "mistral_llm",
    "openai_llm",
    "medical_summarization_agent",
    "pdf_agent",
    "extract_diagnostic_texts",
    "clean_html",
    "mistral_radiology_summarizer_task",
    "abnormality_detection_classification",
    "detect_op_abnormalities",
    "run_crew",
    "summarize_diagnostics",
    "cohort_generation_agent",
    "summarize_lab_data_agent",
    "summarize_agent",
]
