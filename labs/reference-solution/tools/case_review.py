"""Scoring rubric used by the case_reviewer observer agent.

The observer agent is an in-graph critic: it reads a proposed diagnostic answer and
scores it against a fixed rubric. The rubric lives here (deterministic, auditable)
rather than in the model's head so that two runs of the same answer score the same.
"""

from typing import List

from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission
from pydantic import BaseModel


class RubricLine(BaseModel):
    criterion: str
    weight: int
    met: bool
    comment: str


class ReviewScore(BaseModel):
    total_score: int
    max_score: int
    verdict: str
    lines: List[RubricLine]
    blocking_issues: List[str]


@tool(permission=ToolPermission.READ_ONLY)
def score_case_review(
    named_differential: bool,
    cited_next_test: bool,
    flagged_red_flags: bool,
    used_patient_record: bool,
    gave_treatment_without_confirmation: bool,
) -> ReviewScore:
    """
    Score a proposed diagnostic answer against the Princeton-Plainsboro review rubric.

    Args:
        named_differential (bool): True if the answer named at least two candidate diagnoses rather than one.
        cited_next_test (bool): True if the answer named the next diagnostic test to run.
        flagged_red_flags (bool): True if the answer surfaced the immediate risks to the patient.
        used_patient_record (bool): True if the answer was grounded in the retrieved patient chart rather than assumed.
        gave_treatment_without_confirmation (bool): True if the answer prescribed a treatment before the diagnosis was confirmed.
    Returns:
        ReviewScore: Weighted score, per-criterion breakdown, blocking issues and a pass or fail verdict.
    """
    lines = [
        RubricLine(
            criterion="Named a differential, not a single guess",
            weight=30,
            met=named_differential,
            comment="At least two competing candidates are required",
        ),
        RubricLine(
            criterion="Named the next diagnostic test",
            weight=25,
            met=cited_next_test,
            comment="An answer without a next step is not actionable",
        ),
        RubricLine(
            criterion="Surfaced red flags",
            weight=20,
            met=flagged_red_flags,
            comment="Immediate risks must precede the intellectual exercise",
        ),
        RubricLine(
            criterion="Grounded in the patient record",
            weight=25,
            met=used_patient_record,
            comment="The chart must be retrieved, not assumed",
        ),
    ]

    total = sum(line.weight for line in lines if line.met)

    blocking: List[str] = []
    if gave_treatment_without_confirmation:
        blocking.append(
            "Treatment recommended before the diagnosis was confirmed - protocol violation"
        )
    if not used_patient_record:
        blocking.append("Answer was not grounded in the retrieved patient chart")

    if blocking:
        verdict = "FAIL"
    elif total >= 75:
        verdict = "PASS"
    else:
        verdict = "NEEDS REVISION"

    return ReviewScore(
        total_score=total,
        max_score=100,
        verdict=verdict,
        lines=lines,
        blocking_issues=blocking,
    )
