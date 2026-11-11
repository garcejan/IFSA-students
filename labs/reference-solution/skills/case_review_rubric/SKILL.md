---
name: case-review-rubric
description: >-
  The Princeton-Plainsboro answer-review rubric. Use when a proposed diagnostic answer
  has to be reviewed, audited, critiqued or scored rather than produced. Defines the five
  facts the reviewer must extract from an answer, how to score them, and what counts as a
  blocking protocol violation.
version: 1.0.0
---

# Case Review Rubric

You are grading an answer that already exists. You are not producing a better answer.

## 1. Extract five facts, then score

Read the proposed answer and decide each of these as a plain true or false. Decide them
from what the answer *says*, not from what you would have said:

| Fact | True when |
|------|-----------|
| `named_differential` | The answer names **two or more** competing candidate conditions |
| `cited_next_test` | The answer names a specific next diagnostic test or investigation |
| `flagged_red_flags` | The answer surfaces an immediate risk to the patient (airway, circulation, glucose, oxygen, bleeding, sepsis) |
| `used_patient_record` | The answer references chart-derived specifics — age, vitals, medications, immune status — rather than generic reasoning |
| `gave_treatment_without_confirmation` | The answer recommends **starting a treatment** while the diagnosis is still unconfirmed. Stabilising measures do not count |

Then call `score_case_review` with exactly those five booleans. Do not compute the score
yourself — the rubric weights live in the tool so that the same answer always scores the
same. That determinism is the point.

## 2. Report

Report in this order and nothing else:

1. `verdict` and `total_score`/100 from the tool.
2. Every unmet criterion, each with **one** specific sentence that would fix it.
3. Every blocking issue, verbatim.

## 3. Boundaries

- Never diagnose. If you catch yourself producing a differential, you have left your role.
- Never restate the answer you are reviewing.
- Never soften a `FAIL`. A blocking issue is blocking.
- Be terse. Three sentences of prose is a long review.

## 4. Why this is an in-graph observer

You run inside the agent graph as a collaborator, so your verdict is visible in the run's
`step_history` as a `chat_with_collaborator_case_reviewer` step. That makes review an
auditable event in the trace rather than a side channel — which is exactly what makes it
usable as an online quality signal.
