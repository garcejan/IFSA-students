"""Princeton-Plainsboro patient registry (synthetic).

All records are fictional and deterministic — no PHI, no external calls.
"""

from typing import List

from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission
from pydantic import BaseModel


class PatientRecord(BaseModel):
    patient_id: str
    name: str
    age: int
    sex: str
    presenting_complaint: str
    symptoms: List[str]
    vitals: str
    history: str
    current_medications: List[str]
    immune_status: str
    admitted_days: int


_REGISTRY = {
    "PPTH-001": PatientRecord(
        patient_id="PPTH-001",
        name="Rebecca Adler",
        age=29,
        sex="F",
        presenting_complaint="Sudden aphasia during a kindergarten class, then seizure",
        symptoms=["aphasia", "seizure", "fever", "nausea"],
        vitals="BP 104/68, HR 96, Temp 38.4C, RR 18, SpO2 98%",
        history="No significant past medical history. Recent unpasteurised farm cheese.",
        current_medications=["none"],
        immune_status="immunocompetent",
        admitted_days=2,
    ),
    "PPTH-002": PatientRecord(
        patient_id="PPTH-002",
        name="John Henry Giles",
        age=41,
        sex="M",
        presenting_complaint="Progressive weakness, now ventilator dependent",
        symptoms=["ascending paralysis", "dyspnea", "areflexia", "fatigue"],
        vitals="BP 132/84, HR 88, Temp 37.1C, RR 12 (ventilated), SpO2 95%",
        history="Marathon runner. Diagnosed elsewhere with ALS two years ago.",
        current_medications=["riluzole", "baclofen"],
        immune_status="immunocompetent",
        admitted_days=9,
    ),
    "PPTH-003": PatientRecord(
        patient_id="PPTH-003",
        name="Carly Forlano",
        age=34,
        sex="F",
        presenting_complaint="Fever, migrating joint pain and a spreading rash",
        symptoms=["fever", "rash", "arthralgia", "night sweats", "weight loss"],
        vitals="BP 118/72, HR 104, Temp 39.1C, RR 20, SpO2 97%",
        history="Renal transplant 14 months ago, on maintenance immunosuppression.",
        current_medications=["tacrolimus", "mycophenolate", "prednisone"],
        immune_status="immunosuppressed",
        admitted_days=1,
    ),
    "PPTH-004": PatientRecord(
        patient_id="PPTH-004",
        name="Ezra Powell",
        age=78,
        sex="M",
        presenting_complaint="Confusion, chest pain and intermittent syncope",
        symptoms=["confusion", "chest pain", "syncope", "palpitations"],
        vitals="BP 96/58, HR 148 irregular, Temp 36.9C, RR 22, SpO2 92%",
        history="Hypertension, paroxysmal atrial fibrillation, prior TIA.",
        current_medications=["warfarin", "metoprolol", "lisinopril"],
        immune_status="immunocompetent",
        admitted_days=4,
    ),
}


@tool(permission=ToolPermission.READ_ONLY)
def get_patient_record(patient_id: str) -> PatientRecord:
    """
    Look up a patient's chart in the Princeton-Plainsboro registry by patient ID.

    Args:
        patient_id (str): The patient identifier, for example PPTH-001.
    Returns:
        PatientRecord: Demographics, presenting complaint, symptoms, vitals, history, medications and immune status.
    """
    key = patient_id.strip().upper()
    if key not in _REGISTRY:
        raise ValueError(
            f"No patient with ID '{patient_id}'. Known IDs: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[key]
