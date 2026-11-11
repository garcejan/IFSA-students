"""Drug-drug interaction checking against the hospital formulary.

LAB DEVICE (deliberate, do not "fix"):
    This tool raises ValueError for any drug that is not in the small synthetic
    formulary below. That is the *planted* failure mode of this workshop: it produces
    genuine failed tool calls, which is what the tool-call success rate and error-rate
    panels in Part 3 are measuring. A team that never fails produces a boring dashboard.
"""

from typing import List

from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission
from pydantic import BaseModel


class Interaction(BaseModel):
    drug_a: str
    drug_b: str
    severity: str
    mechanism: str
    management: str
    formulary_version: str


class FormularyEntry(BaseModel):
    drug: str
    drug_class: str


class Formulary(BaseModel):
    entries: List[FormularyEntry]
    formulary_version: str


_FORMULARY = {
    "warfarin": "vitamin K antagonist",
    "metoprolol": "beta blocker",
    "lisinopril": "ACE inhibitor",
    "tacrolimus": "calcineurin inhibitor",
    "mycophenolate": "antimetabolite",
    "prednisone": "corticosteroid",
    "riluzole": "glutamate antagonist",
    "baclofen": "muscle relaxant",
    "amiodarone": "class III antiarrhythmic",
    "fluconazole": "azole antifungal",
    "rifampin": "rifamycin antibiotic",
    "ibuprofen": "NSAID",
    "clarithromycin": "macrolide antibiotic",
}

_INTERACTIONS = {
    frozenset({"warfarin", "amiodarone"}): (
        "MAJOR",
        "Amiodarone inhibits CYP2C9 and raises warfarin exposure",
        "Reduce the warfarin dose by 30-50% and recheck INR in 3 days",
    ),
    frozenset({"warfarin", "ibuprofen"}): (
        "MAJOR",
        "Additive bleeding risk plus displacement from plasma protein binding",
        "Avoid; use paracetamol for analgesia",
    ),
    frozenset({"warfarin", "rifampin"}): (
        "MAJOR",
        "Rifampin induces CYP2C9 and collapses the INR",
        "Expect a large dose increase; monitor INR twice weekly",
    ),
    frozenset({"tacrolimus", "fluconazole"}): (
        "MAJOR",
        "Fluconazole inhibits CYP3A4 and raises tacrolimus to nephrotoxic levels",
        "Reduce tacrolimus dose by half and monitor trough levels",
    ),
    frozenset({"tacrolimus", "clarithromycin"}): (
        "MAJOR",
        "Strong CYP3A4 inhibition raises tacrolimus levels sharply",
        "Choose a non-macrolide antibiotic if possible",
    ),
    frozenset({"metoprolol", "amiodarone"}): (
        "MODERATE",
        "Additive negative chronotropy and AV nodal blockade",
        "Telemetry; hold if heart rate falls below 50",
    ),
    frozenset({"lisinopril", "ibuprofen"}): (
        "MODERATE",
        "NSAIDs blunt the antihypertensive effect and add renal risk",
        "Monitor creatinine and blood pressure",
    ),
}


@tool(permission=ToolPermission.READ_ONLY)
def check_drug_interaction(drug_a: str, drug_b: str) -> Interaction:
    """
    Check two drugs against the hospital formulary for a documented interaction.

    Args:
        drug_a (str): The first drug name, for example warfarin.
        drug_b (str): The second drug name, for example amiodarone.
    Returns:
        Interaction: Severity, mechanism and management for the pair.
    """
    a = drug_a.strip().lower()
    b = drug_b.strip().lower()

    unknown = [d for d in (a, b) if d not in _FORMULARY]
    if unknown:
        # Planted failure mode - see the module docstring.
        raise ValueError(
            f"Not in the PPTH formulary: {', '.join(unknown)}. "
            f"Call list_formulary to see stocked drugs."
        )

    hit = _INTERACTIONS.get(frozenset({a, b}))
    if hit is None:
        severity, mechanism, management = (
            "NONE DOCUMENTED",
            "No interaction recorded in this formulary version",
            "No change required; clinical judgement still applies",
        )
    else:
        severity, mechanism, management = hit

    return Interaction(
        drug_a=a,
        drug_b=b,
        severity=severity,
        mechanism=mechanism,
        management=management,
        formulary_version="PPTH-2026.1",
    )


@tool(permission=ToolPermission.READ_ONLY)
def list_formulary() -> Formulary:
    """
    List every drug stocked in the Princeton-Plainsboro formulary with its drug class.

    Args:
    Returns:
        Formulary: All stocked drugs and their classes.
    """
    return Formulary(
        entries=[FormularyEntry(drug=d, drug_class=c) for d, c in sorted(_FORMULARY.items())],
        formulary_version="PPTH-2026.1",
    )
