"""OPD response validator and transformer.

This module parses OPD JSON payloads into a structured pydantic model, enriches
it with cohort mappings from MongoDB when available, and produces a dict ready
for storage. Database access is optional: if the MongoDB URI or database are not
configured, the validator still runs and simply skips cohort enrichment.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, validator

try:  # MongoDB is optional; skip silently if unavailable
    from pymongo import MongoClient  # type: ignore
except ImportError:  # pragma: no cover
    MongoClient = None  # type: ignore

# ---------------------------------------------------------------------------
# MongoDB connection
# ---------------------------------------------------------------------------

_DOCSTRIBE_URI = os.getenv(
    "DOCSTRIBE_MONGODB_URI",
    "mongodb+srv://dtadmin:docstribe%40123@dt-redcliffe-instance-1.teoq61l.mongodb.net/?"
    "retryWrites=true&w=majority&tlsAllowInvalidCertificates=true",
)
_OPD_DB_NAME = os.getenv("LLM_ORCHESTRATOR_DB", "batch_processing_continental")

if MongoClient is not None:
    try:
        _mongo_client = MongoClient(_DOCSTRIBE_URI)
        _mongo_db = _mongo_client[_OPD_DB_NAME]
    except Exception:  # pragma: no cover - defensive fallback
        _mongo_client = None
        _mongo_db = None
else:  # pragma: no cover - pymongo not installed
    _mongo_client = None
    _mongo_db = None

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Abnormality(BaseModel):
    classification: Optional[str] = ""
    isAbnormal: bool
    test: str
    value: Union[float, str, int]


class MissedFollowUp(BaseModel):
    evidence: str
    follow_up_date: str  # dd-mm-yyyy
    type: str
    criticality: Optional[str] = ""


class CareGaps(BaseModel):
    doctor_care_gap_flag: bool
    protocol_care_gap_flag: bool
    missed_follow_ups: List[MissedFollowUp] = []

    @validator("doctor_care_gap_flag", "protocol_care_gap_flag", pre=True)
    def str_to_bool(cls, v):
        if isinstance(v, str):
            return v.lower() == "true"
        return bool(v)


class ChronicFlag(BaseModel):
    chronic_evidence: str
    has_chronic_disease: bool
    has_multiple_chronic_disease: bool


class MedicationItem(BaseModel):
    medication_name: str
    dosage: str
    frequency: str
    duration: str
    instructions: str


class MedicationsPrescribed(BaseModel):
    current_medications: List[MedicationItem]


class PatientSummary(BaseModel):
    existing_complaints: List[str]
    existing_conditions: List[str]
    medical_history: List[str]
    medications_prescribed: MedicationsPrescribed


class FollowUpAdvice(BaseModel):
    advised: bool
    follow_up_instructions: List[str] = []


class IPRecommendationAdvice(BaseModel):
    advised: bool
    ip_recommendation_list: List[str] = []


class LabTestAdvice(BaseModel):
    advised: bool
    lab_test_date_advised: Optional[str] = ""
    lab_tests_ordered: List[str] = []


class RadiologyTestAdvice(BaseModel):
    advised: bool
    radiology_tests_date_advised: Optional[str] = ""
    radiology_tests_ordered: List[str] = []


class ReferralAdvice(BaseModel):
    advised: bool
    referrals_string: List[str] = []


class DoctorAdvice(BaseModel):
    follow_up: FollowUpAdvice
    ip_recommendations: IPRecommendationAdvice
    lab_tests: LabTestAdvice
    radiology_tests: RadiologyTestAdvice
    referrals: ReferralAdvice


class ClinicalCondition(BaseModel):
    condition: str
    evidence: str
    severity: str
    associated_symptoms: List[str]


class LabResult(BaseModel):
    test_name: str
    value: Union[str, int, float]
    unit: str
    reference_range: str
    status: str
    date: str


class Measurement(BaseModel):
    parameter: str
    value: Union[str, int, float]
    unit: Union[str, int, float]


class KeyFinding(BaseModel):
    anatomical_site: str
    finding_description: str
    measurements: List[Measurement] = []
    clinical_significance: str


class RadiologyFinding(BaseModel):
    study_date: str
    modality: str
    body_region: str
    key_findings: List[KeyFinding]
    radiologist_impression: str
    recommendations: List[str] = []


class ClinicalAssessment(BaseModel):
    risk_factors: List[str]
    clinical_conditions: List[ClinicalCondition]
    lab_results: List[LabResult]
    radiology_findings: List[RadiologyFinding]


class TemporalClinicalData(BaseModel):
    visit_date: str
    clinical_progression: List[str]


class ClinicalSummary(BaseModel):
    clinical_assessment: ClinicalAssessment
    doctor_advice: DoctorAdvice
    patient_summary: PatientSummary
    temporal_clinical_data: Union[List[TemporalClinicalData], Dict[str, Any]]

    @validator("temporal_clinical_data", pre=True, always=True)
    def normalize_temporal_clinical_data(cls, v):
        def is_valid_visit(obj):
            return (
                isinstance(obj, dict)
                and "visit_date" in obj
                and "clinical_progression" in obj
                and isinstance(obj["clinical_progression"], list)
            )

        def parse_date(date_str):
            try:
                return datetime.strptime(date_str, "%d-%b-%Y")
            except Exception:
                return None

        if isinstance(v, dict) and "visits" in v:
            visits = v["visits"]
            if not isinstance(visits, list) or not all(is_valid_visit(visit) for visit in visits):
                raise ValueError(
                    "All items in 'visits' must be visit dicts with 'visit_date' and 'clinical_progression' list"
                )
            visits = sorted(visits, key=lambda visit: parse_date(visit["visit_date"]) or datetime.min)
            return {"visits": visits}

        if is_valid_visit(v):
            return {"visits": [v]}

        if isinstance(v, list) and all(is_valid_visit(visit) for visit in v):
            visits = sorted(v, key=lambda visit: parse_date(visit["visit_date"]) or datetime.min)
            return {"visits": visits}

        raise ValueError("Invalid format for temporal_clinical_data")


class Task(BaseModel):
    body_part: Optional[str] = ""
    criticality: Optional[str] = ""
    department: str
    due_date: str
    evidence: str
    source: str
    specialist: Optional[str] = ""
    task_name: str
    task_type: str
    urgency: Optional[str] = ""
    revenue_potential: Optional[int] = 0


class PrimaryGoal(BaseModel):
    tasks: List[Task]
    visit_date: str

    @field_validator("tasks")
    @classmethod
    def filter_guideline_led_tasks(cls, tasks):
        return [
            task
            for task in tasks
            if not (
                task.source == "guideline_led"
                and (task.urgency == "routine" or task.criticality in {"medium", "low"})
            )
        ]


class ActionItem(BaseModel):
    action_type: str
    date: str
    test_name: Optional[str] = None
    department: Optional[str] = None
    specialist: Optional[str] = None
    urgency: Optional[str] = None
    details: Optional[str] = None
    revenue_potential: Optional[int] = None


class Actions(BaseModel):
    pending: Dict[str, List[ActionItem]] = Field(default_factory=dict)
    upcoming: Dict[str, List[ActionItem]] = Field(default_factory=dict)


class DifferentialDiagnosis(BaseModel):
    department: str
    disease_name: str
    evidence: str
    risk_score: Union[int, float, str]
    symptoms: List[str]


class FurtherManagement(BaseModel):
    evidence: str
    management_advice: str


class PatientHealthCondition(BaseModel):
    current_condition: List[str]
    impact_on_health: List[dict]
    upcoming_appointment: List[str]
    symptoms_monitoring: List[str]

    @validator("impact_on_health", pre=True, always=True)
    def ensure_list_of_dicts(cls, v):
        if isinstance(v, dict):
            return [v]
        if v is None:
            return []
        return v


class RiskStratification(BaseModel):
    overall_risk_score: Union[int, float, str]
    risk_category: str


class SecondaryComplication(BaseModel):
    department_name: str
    disease_name: str
    evidence: str


class CohortInfo(BaseModel):
    primary: str
    secondary: str
    primary_gold_standard_disease: Optional[str] = None
    primary_icd10_code: Optional[str] = None
    primary_icd10_description: Optional[str] = None
    primary_mapping_confidence: Optional[str] = None
    secondary_gold_standard_disease: Optional[str] = None
    secondary_icd10_code: Optional[str] = None
    secondary_icd10_description: Optional[str] = None
    secondary_mapping_confidence: Optional[str] = None


class MedicalRecord(BaseModel):
    abnormalities_identified: List[Abnormality]
    admission_evidence: str
    care_gaps_present: CareGaps
    chronic_flag: ChronicFlag
    clinical_summary: ClinicalSummary
    differential_diagnoses: List[DifferentialDiagnosis]
    doctors_notes: Union[dict, List[Any], None] = Field(default_factory=dict)
    further_management: FurtherManagement
    overall_admission_probability: str
    overall_risk_score: Union[int, float, str]
    patient_health_condition: PatientHealthCondition
    primary_cohort: Dict[str, CohortInfo]
    primary_goal: PrimaryGoal
    progression_stage: Dict[str, Any] = Field(default_factory=dict)
    request_id: str
    risk_stratification: RiskStratification
    secondary_cohort: Dict[str, CohortInfo]
    secondary_complications: List[SecondaryComplication]
    status: str
    version: str
    actions: Actions = Field(default_factory=Actions)

    def compute_actions_and_cohort_mapping(self, db):
        pg: PrimaryGoal = self.primary_goal
        today = date.today()
        action_map = {
            "lab-test": "Pathology Test",
            "radiology": "Radiology Test",
            "appointment": "Follow Up Appointment",
            "referral": "Referral Appointment",
            "ip-recommendation": "Admission Advised",
            "procedure": "Radiology Test",
        }

        pending: Dict[str, List[ActionItem]] = {}
        upcoming: Dict[str, List[ActionItem]] = {}

        if pg and getattr(pg, "tasks", None):
            for task in pg.tasks:
                if task.task_type == "ip-recommendation" and task.source == "guideline_led":
                    continue
                try:
                    due = datetime.strptime(task.due_date, "%d-%m-%Y").date()
                except Exception:
                    continue

                data = {
                    "action_type": action_map.get(task.task_type, "Other"),
                    "date": task.due_date,
                    "revenue_potential": task.revenue_potential,
                }
                if task.task_type in [
                    "lab-test",
                    "radiology",
                    "procedure",
                    "ip-recommendation",
                    "referral",
                    "appointment",
                ]:
                    data["test_name"] = task.task_name
                if task.task_type in ["appointment", "procedure", "ip-recommendation"]:
                    data.update({"department": task.department, "urgency": task.urgency})
                if task.task_type == "referral":
                    data.update(
                        {
                            "department": task.department,
                            "specialist": task.specialist,
                            "urgency": task.urgency,
                        }
                    )
                if task.source not in ["doctor_advised", "guideline_led"] and getattr(task, "evidence", None):
                    data["details"] = task.evidence

                (pending if due <= today else upcoming).setdefault(task.source, []).append(ActionItem(**data))

        self.actions = Actions(pending=pending, upcoming=upcoming)

        if db is None:
            return self

        try:
            mapping_coll = db["cohort_icd_mappings"]
            for spec, info in (self.primary_cohort or {}).items():
                doc = mapping_coll.find_one({"specialty": spec}) or {}
                cohort_map = (doc.get("cohort_mappings") or {}).get(getattr(info, "primary", None), {})
                if cohort_map:
                    info.primary_gold_standard_disease = cohort_map.get("mapped_to_gold_standard")
                    info.primary_icd10_code = cohort_map.get("icd10_code")
                    info.primary_icd10_description = cohort_map.get("icd10_description")
                    info.primary_mapping_confidence = cohort_map.get("confidence")

            for spec, info in (self.secondary_cohort or {}).items():
                doc = mapping_coll.find_one({"specialty": spec}) or {}
                cohort_map = (doc.get("cohort_mappings") or {}).get(getattr(info, "secondary", None), {})
                if cohort_map:
                    info.secondary_gold_standard_disease = cohort_map.get("mapped_to_gold_standard")
                    info.secondary_icd10_code = cohort_map.get("icd10_code")
                    info.secondary_icd10_description = cohort_map.get("icd10_description")
                    info.secondary_mapping_confidence = cohort_map.get("confidence")
        except Exception:  # pragma: no cover - mapping is best-effort
            pass

        return self


def opd_parse_json(data: dict) -> dict:
    try:
        record = MedicalRecord.model_validate(data)
        record.compute_actions_and_cohort_mapping(_mongo_db)
        return json.loads(record.json())
    except ValidationError as exc:
        print("Validation failed:")
        print(exc.json(indent=2))
        return {}


if __name__ == "__main__":  # pragma: no cover - manual testing utility
    sample_path = os.environ.get("OPD_SAMPLE_FILE", "sample_opd_record.json")
    with open(sample_path, "r", encoding="utf-8") as fh:
        sample = json.load(fh)
    result = opd_parse_json(sample)
    print(json.dumps(result, indent=2))
    if _mongo_client is not None:
        _mongo_client.close()
