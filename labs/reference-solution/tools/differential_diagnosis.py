"""Differential diagnosis ranking for the diagnostics team (synthetic ruleset)."""

from typing import List

from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission
from pydantic import BaseModel


class Candidate(BaseModel):
    condition: str
    likelihood: str
    supporting: str
    next_test: str


class Differential(BaseModel):
    presentation: str
    candidates: List[Candidate]
    red_flags: List[str]
    note: str


_RULES = [
    (
        {"aphasia", "seizure"},
        Candidate(
            condition="Neurocysticercosis",
            likelihood="high",
            supporting="Focal neurological deficit plus seizure in a patient with possible pork tapeworm exposure",
            next_test="Contrast MRI brain",
        ),
    ),
    (
        {"fever", "rash"},
        Candidate(
            condition="Disseminated infection (bacterial or fungal)",
            likelihood="high",
            supporting="Fever with cutaneous involvement suggests haematogenous spread",
            next_test="Blood cultures x2 and skin biopsy",
        ),
    ),
    (
        {"fever", "arthralgia"},
        Candidate(
            condition="Systemic lupus erythematosus",
            likelihood="moderate",
            supporting="Migratory arthralgia with fever and rash is a classic connective-tissue pattern",
            next_test="ANA, anti-dsDNA, complement C3/C4",
        ),
    ),
    (
        {"ascending paralysis", "areflexia"},
        Candidate(
            condition="Guillain-Barre syndrome",
            likelihood="high",
            supporting="Ascending weakness with lost reflexes and respiratory involvement",
            next_test="Lumbar puncture for albuminocytologic dissociation, nerve conduction study",
        ),
    ),
    (
        {"confusion", "palpitations"},
        Candidate(
            condition="Cardioembolic stroke from atrial fibrillation",
            likelihood="high",
            supporting="Irregular tachycardia with acute cognitive change and hypoperfusion",
            next_test="12-lead ECG, CT head, echocardiogram",
        ),
    ),
    (
        {"chest pain", "syncope"},
        Candidate(
            condition="Critical aortic stenosis or arrhythmic syncope",
            likelihood="moderate",
            supporting="Exertional chest pain with syncope in an older patient",
            next_test="Transthoracic echocardiogram, troponin, telemetry",
        ),
    ),
    (
        {"night sweats", "weight loss"},
        Candidate(
            condition="Lymphoma",
            likelihood="moderate",
            supporting="B symptoms: fever, night sweats and unintentional weight loss",
            next_test="CT chest/abdomen/pelvis, LDH, excisional lymph-node biopsy",
        ),
    ),
]

_RED_FLAGS = {
    "seizure": "Status epilepticus risk - secure the airway before imaging",
    "dyspnea": "Impending respiratory failure - measure vital capacity hourly",
    "syncope": "Sudden cardiac death risk - continuous telemetry",
    "confusion": "Rule out hypoglycaemia and hypoxia before anything exotic",
}


@tool(permission=ToolPermission.READ_ONLY)
def differential_diagnosis(symptoms: str, age: int, immune_status: str) -> Differential:
    """
    Rank the most likely diagnoses for a clinical presentation and name the next test for each.

    Args:
        symptoms (str): Comma-separated symptoms, for example "fever, rash, arthralgia".
        age (int): Patient age in years.
        immune_status (str): One of immunocompetent, immunosuppressed or unknown.
    Returns:
        Differential: Ranked candidate conditions with supporting evidence, the next diagnostic test, and red flags.
    """
    observed = {s.strip().lower() for s in symptoms.split(",") if s.strip()}

    candidates: List[Candidate] = []
    for trigger, candidate in _RULES:
        if trigger.issubset(observed):
            candidates.append(candidate)

    if immune_status.strip().lower() == "immunosuppressed":
        candidates.insert(
            0,
            Candidate(
                condition="Opportunistic infection (CMV, aspergillus, nocardia)",
                likelihood="high",
                supporting="Immunosuppression widens the differential to organisms a normal immune system clears silently",
                next_test="Galactomannan, CMV PCR, fungal blood cultures",
            ),
        )

    if age >= 65:
        candidates.append(
            Candidate(
                condition="Polypharmacy adverse effect",
                likelihood="moderate",
                supporting="Age over 65 with multiple active prescriptions - always on the list",
                next_test="Full medication reconciliation and drug interaction review",
            ),
        )

    if not candidates:
        candidates.append(
            Candidate(
                condition="Insufficient data for a ranked differential",
                likelihood="unknown",
                supporting="No rule matched the supplied symptom set",
                next_test="Complete history, physical examination and basic metabolic panel",
            )
        )

    return Differential(
        presentation=f"{age}y {immune_status}: {', '.join(sorted(observed))}",
        candidates=candidates,
        red_flags=[_RED_FLAGS[s] for s in sorted(observed) if s in _RED_FLAGS],
        note="Synthetic teaching ruleset. Not for clinical use.",
    )
