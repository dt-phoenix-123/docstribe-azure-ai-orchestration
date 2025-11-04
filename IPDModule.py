from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class DeferredTimeline(BaseModel):
    """Details of deferred procedure timeline and emergency symptoms.
    Red flags should be string of not more than 2 words each"""
    time: str
    red_flags: List[str]


class Stage(BaseModel):
    """Current stage in IP recommendation workflow."""
    primary_status: Optional[str]
    secondary_status: Optional[str]


class RemarksActionables(BaseModel):
    """Pointers for communication and follow-up planning. Include crisp pointers as per the given clinical note only."""
    important_pointers: Optional[List[str]]
    follow_up_date: Optional[str]
    tackle_pointers: Optional[List[str]]


class Actionables(BaseModel):
    """Clinical + care coordination tasks related to procedure execution."""
    procedure_criticiality: Literal["Critical", "High", "Medium", "Low"]
    procedure_criticiality_evidence: Optional[str]
    symptoms_monitoring: Optional[List[str]]
    remarks_actionables: Optional[RemarksActionables]
    simple_surgery_name: Optional[str]
    surgery_advised: bool
    deferred_timeline: DeferredTimeline
    stage: Optional[Stage]
    procedure_name: Optional[str]
    admission_advised_date: Optional[str]


class TalkingPoints(BaseModel):
    """Conversation helper and patient guidance pointers."""
    important_pointers: Optional[List[str]]
    tackle_pointers: Optional[List[str]]
    actionables: Optional[Actionables]


class SurgeryInfo(BaseModel):
    """Basic metadata of surgery or procedure advised."""
    simple_name: Optional[str]
    surgery_advised: bool
    admission_advised_date: Optional[str]


class IPAdvisedMeta(BaseModel):
    """
    Metadata for IP / Daycare / OP-procedure advice.
    
    Helps classify the nature of admission, care setting, and urgency.
    Used by Care Coordinators & workflow engines.
    """

    admission_type: Optional[str] = Field(
        None,
        description="Admission category e.g. inpatient, outpatient, daycare_procedure"
    )
    procedure_name: Optional[str] = Field(
        None, 
        description="Full medical procedure name"
    )
    procedure_setting: Optional[str] = Field(
        None,
        description="Where procedure is performed: inpatient / outpatient / daycare"
    )
    procedure_type: Optional[str] = Field(
        None,
        description="Elective / Emergency / Semi-elective"
    )


class PatientCarePlan(BaseModel):
    """
    Docstribe Patient Care Plan schema.
    Captures IP-advised journey, outreach logic, and follow-ups.
    """
    talking_points: TalkingPoints
    surgery_info: Optional[SurgeryInfo]
    doctor_name: Optional[str]
    follow_up_date: Optional[str]
    last_visit_date: Optional[str]
    center: Optional[str]
    department: Optional[str]
    prescription: Optional[List[str]]
    procedure_name: Optional[str]

    # ✅ NEW BLOCK ADDED HERE
    ip_advised_meta: Optional[IPAdvisedMeta]