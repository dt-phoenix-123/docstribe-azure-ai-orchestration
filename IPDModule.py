from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class DeferredTimeline(BaseModel):
    """Details of deferred procedure timeline and emergency symptoms.
    Red flags should be string of not more than 2 words each"""
    time: Literal["<15 days", "15-30 days", ">30 days"]
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
    
    Helps classify the nature of admission, care setting, cost of procedure and urgency.
    Used by Care Coordinators & workflow engines.
    """

    admission_type: Literal["surgical", "medical", "daycare"] = Field(
        None,
        description="Type of admission as per the defined in the clinical note. Medical means management of patient in IPD. So use that if no definitive procedure is found in the clinical note."
    )
    procedure_name: Optional[str] = Field(
        None, 
        description="Full medical procedure name as per clinical note."
    )

    procedure_description: Optional[str] = Field(
        None,
        description="Brief description of the procedure as per clinical note so that any non medical user can understand. Don't make it more than one line."
    )

    procedure_setting: Literal["inpatient","outpatient","daycare"] = Field(
        None,
        description="Where procedure is performed: inpatient / outpatient / daycare"
    )
    procedure_type: Optional[str] = Field(
        None,
        description="Elective / Emergency / Semi-elective"
    )

    procedure_cost: Optional[float] = Field(
        None,
        description="Estimated cost of the procedure in INR as per standard pricing of tertiary care hospitals in India."
    )

    is_robotic: Literal["true","false"] = Field(
        None,
        description="Whether the procedure in question is robotic-assisted or not. If yes then mark it true else false."
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