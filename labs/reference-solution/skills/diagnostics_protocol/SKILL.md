---
name: diagnostics-protocol
description: >-
  The Princeton-Plainsboro differential diagnosis protocol. Use whenever a diagnostic
  question is raised about a patient: it fixes the order of work (chart first, then
  differential, then the next test), the escalation rules for routing to a specialist,
  and the hard safety rules about treatment and confirmation.
version: 1.0.0
---

# Princeton-Plainsboro Diagnostics Protocol

You follow this protocol on every diagnostic case. It is the order of work, not advice.

## 1. Chart before theory

Retrieve the patient record before you reason about the patient. If a patient ID was
given, call `get_patient_record` first. Never invent a symptom, a vital sign or a
medication. If the chart is unavailable, say so and stop.

## 2. Differential before diagnosis

Produce **at least two** competing candidate conditions. A single answer is a guess.
For each candidate state:

- the condition,
- the finding that supports it,
- the single next test that would confirm or kill it.

Kill the cheap and dangerous ones first.

## 3. Red flags precede the puzzle

Anything that can kill the patient in the next hour is named at the top of the answer,
before the intellectual work. Airway, breathing, circulation, glucose, oxygen.

## 4. Escalation rules

Route to the specialist whose domain owns the finding. Do not answer outside your
domain when a colleague owns it:

| Finding | Route to |
|---------|----------|
| Mass, B symptoms, malignancy, staging, palliative or end-of-life decisions | Dr Wilson (oncology) |
| Weakness, seizure, neuropathy, cognition, anything above the neck | Dr Foreman (neurology) |
| Chest pain, arrhythmia, syncope, perfusion, anything needing surgery or ICU | Dr Chase (cardiology and intensive care) |
| Fever, rash, transplant, immunosuppression, antibiotic and drug choice | Dr Cameron (immunology and infectious disease) |
| Consent, budget, legal exposure, transfer, staffing, anything that is not medicine | Dr Cuddy (hospital administration) |

Delegate to exactly one specialist unless two domains genuinely both own the finding.

## 5. Treatment requires confirmation

Do not recommend starting a treatment before the diagnosis is confirmed by the next
test naming it. You may recommend *stabilising* measures at any time. State the
difference explicitly.

## 6. Never a real prescription

Every output ends with the reminder that this is a synthetic teaching case and is not
medical advice.
