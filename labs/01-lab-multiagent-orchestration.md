# IFSA workshop - Part 1

## Build the diagnostics department

**~60 minutes.** You start with an empty folder and finish with seven agents deployed on
watsonx Orchestrate, routing correctly, backed up on your laptop.

You will write almost no code. You will write **prompts**, read what Bob produces, and
push back when it is wrong. That is the actual skill being taught.

---

## Step A - Design before you build

### Why this step

wxO has four things that all look like "give the agent a capability" and are not
interchangeable: a **tool**, a **collaborator agent**, an **agent skill**, and a
**knowledge base**. Picking wrong is the most common design mistake, and it is expensive to
undo once you have imported twenty resources. Make Bob commit to the mapping first.

### Prompt

```text
We are building the Princeton-Plainsboro diagnostics department from House M.D. as a
multi-agent system on watsonx Orchestrate. Dr House is the single entry point. When a
question comes in he pulls the patient chart, frames the case, and delegates to whichever
colleague owns the finding: Foreman (neurology), Wilson (oncology), Chase (cardiology and
intensive care), Cameron (immunology, infectious disease and drug interactions), Cuddy
(hospital administration - consent, legal, budget, beds). There is also a case reviewer
that grades a proposed answer against a protocol and never diagnoses.

Before writing anything, use the watsonx-orchestrate skill to produce a design:

1. A table mapping every capability to the RIGHT wxO construct - tool, collaborator agent,
   agent skill, or knowledge base - with one sentence on why that construct and not the
   others. I want to see the skill-vs-tool-vs-KB decision made explicitly.
2. The dependency-ordered import sequence, and what breaks if the order is wrong.
3. Which agent needs which tool, and why an orchestrator should own very few tools.
4. How wxO actually routes to a collaborator at runtime - what the router reads to make
   that decision, and what that implies about how we write agent descriptions.
5. A mermaid diagram of the final topology.

Do not create any files yet. Show me the design and wait.
```

### What good looks like

Bob should tell you, without hedging, that:

- **Routing is driven by the collaborator's `description`**, not its instructions and not
  its name. wxO generates a `chat_with_collaborator_<name>` tool on the orchestrator and
  the router picks it the same way it picks any tool — from the description text. This is
  the single most important fact in Part 1.
- The **differential protocol** is an *agent skill* (procedural know-how, versioned,
  attachable to several agents), not a tool and not a knowledge base.
- **Import order is** tools → skills → collaborators → orchestrator. The orchestrator
  references collaborators by name; they must exist first.

If Bob suggests putting the protocol in a knowledge base, or gives the orchestrator eight
tools, correct it now — you are calibrating the system prompt for everything that follows.

---

## Step B - The tools

### Why this step

Tools are where wxO is strictest and where imports fail. Google-style docstrings, type
hints on every parameter and the return, no cross-file imports, self-contained files. Bob
knows these rules from the skill; your job is to check it obeyed them.

We are also **deliberately planting two defects**. This is a lab about operating a system,
and a system that never misbehaves teaches you nothing. One tool will be slow. One tool
will fail on a realistic input. In Part 4 you will find both from telemetry alone.

### Prompt

```text
Create the Python tools for the diagnostics department in tools/. Follow the
watsonx-orchestrate skill's rules for @tool exactly - Google-style docstrings with Args
and Returns and no blank line between them, type hints on every parameter and the return,
Pydantic models as explicit classes, each file self-contained with no cross-file imports.

Five files:

1. patient_registry.py - get_patient_record(patient_id: str). A deterministic synthetic
   registry of four inpatients, PPTH-001 to PPTH-004, each with name, age, sex, presenting
   complaint, symptom list, vitals, history, current medications, immune status and days
   admitted. Make them recognisable House cases: a young woman with aphasia and a seizure;
   a man with ascending paralysis on a ventilator; a renal transplant recipient on
   tacrolimus with fever, rash and B symptoms; a 78 year old in atrial fibrillation on
   warfarin with confusion and chest pain. Raise ValueError listing the known IDs if the
   ID is unknown.

2. differential_diagnosis.py - differential_diagnosis(symptoms: str, age: int,
   immune_status: str). A small synthetic ruleset that returns ranked candidate conditions,
   each with the supporting finding and the single next test, plus a red-flags list.
   Immunosuppressed patients get opportunistic infections pushed to the top; patients over
   65 get a polypharmacy candidate appended.

3. lab_reference.py - lab_reference(test_name: str, value: float). Interprets a lab value
   against a reference range for about ten common tests. DELIBERATE LAB DEVICE: this tool
   must time.sleep(4.5) before returning, simulating a slow Laboratory Information System.
   Put a clear module-level comment saying the stall is intentional, that it is the planted
   latency for Part 4, and that it must not be "fixed". Raise ValueError for an unknown test.

4. drug_interaction.py - check_drug_interaction(drug_a: str, drug_b: str) and
   list_formulary(). A synthetic formulary of about thirteen drugs and seven documented
   interactions with severity, mechanism and management. DELIBERATE LAB DEVICE:
   check_drug_interaction must raise ValueError when either drug is not in the formulary,
   naming the missing drug and pointing at list_formulary. Comment that this planted
   failure is what the tool-call success rate measures in Part 4.

5. case_review.py - score_case_review(...) taking five booleans (named_differential,
   cited_next_test, flagged_red_flags, used_patient_record,
   gave_treatment_without_confirmation) and returning a weighted score out of 100, a
   per-criterion breakdown, blocking issues and a PASS / NEEDS REVISION / FAIL verdict.
   The weights live in the tool so the same answer always scores the same.

Every tool is READ_ONLY. Everything is synthetic - no real clinical content.
```

### Expected output

Five files. Ask Bob to show you `lab_reference.py` and confirm the `time.sleep(4.5)` is
there with the comment — if Bob "helpfully" removed it, Part 4 has no exercise.

### The warning you are about to see, and should ignore

When these import, **every single Python tool** prints:

```
[WARNING] - Unable to properly parse parameter descriptions due to missing or
incorrect type hints. This may result in degraded agent performance.
```

This is a **false positive** in 2.12/2.13. It fires on correctly-annotated tools and the
descriptions parse fine. Do not let Bob start "fixing" type hints in response to it. The
real causes of a genuine parse failure are a missing type hint or a blank line between the
`Args:` and `Returns:` blocks — check for those, then move on.

---

## Step C - The agent skill

### Why this step

An **agent skill** is procedural know-how: the order of work, the escalation rules, the
things that must never happen. It is versioned and portable, and it is where the routing
policy belongs so that it is not buried in one agent's instructions.

### Prompt

```text
Create the diagnostics-protocol agent skill in skills/diagnostics_protocol/SKILL.md,
following the watsonx-orchestrate skill's SKILL.md format. It must encode:

1. Chart before theory - retrieve the patient record before reasoning; never invent a
   symptom, vital or medication.
2. Differential before diagnosis - at least two competing candidates, each with the
   finding that supports it and the single next test that confirms or kills it.
3. Red flags precede the puzzle - anything that can kill the patient in the next hour is
   named first.
4. Escalation rules - a table mapping finding to owning specialist, covering all five
   doctors.
5. Treatment requires confirmation - no treatment before the diagnosis is confirmed;
   stabilising measures always allowed and must be labelled as such.
6. Every output ends with the synthetic-teaching-case disclaimer.

Watch the naming rule: agent skill names are kebab-case, unlike every other wxO resource.
```

### The gotcha

Agent skill names must be **kebab-case** — `diagnostics-protocol`, never
`diagnostics_protocol`. Every other resource in wxO is snake_case. Importing an underscore
name fails with *"Skill name … is invalid … (agentskills.io spec)"*. Bob should already
know this from the skill; if it generated an underscore, that is a good moment to see how
it recovers.

---

## Step D - The five specialists

### Why this step

This is where routing is won or lost. The orchestrator's router reads **collaborator
descriptions** to choose. A description that starts "You are a helpful assistant that…" is
unroutable. A description that starts with the trigger symptoms is routable.

### Prompt

```text
Create the five specialist agent YAMLs in agents/ - dr_foreman, dr_wilson, dr_chase,
dr_cameron, dr_cuddy. kind: native, spec_version: v1, style: react_core, and the tenant
default LLM.

The description field is what the orchestrator's router reads to decide where to send a
question, so write each one as a ROUTING TRIGGER, not a personality blurb: start with the
specialty, then list the concrete findings and keywords that should land there. Make the
five descriptions mutually exclusive - no two should compete for the same symptom.

  dr_foreman  neurology - seizure, aphasia, weakness, paralysis, areflexia, neuropathy,
              confusion, stroke, meningitis. tool: differential_diagnosis
  dr_wilson   oncology - masses, B symptoms, night sweats, weight loss, lymphoma, staging,
              LDH, tumour markers, palliative. tool: lab_reference
  dr_chase    cardiology and intensive care - chest pain, arrhythmia, atrial fibrillation,
              syncope, shock, troponin, INR, ventilated patients, ICU, surgery.
              tool: lab_reference
  dr_cameron  immunology and infectious disease - fever, rash, sepsis, transplant,
              immunosuppression, autoimmune, and every question about drugs or drug pairs.
              tools: check_drug_interaction, list_formulary
  dr_cuddy    hospital administration - consent, refusal of treatment, capacity, legal
              exposure, lawsuits, budget, ethics board, staffing, beds, transfers.
              no tools

For the instructions field give each a real persona in the voice of the character, then
a YOUR ROLE section with the explicit rule that they must call their tool before answering
rather than reciting from memory, then a CROSS-CONSULT section naming when to hand off, then
CRITICAL RULES. Cuddy must refuse clinical questions in one sentence and name the service
that owns them. Everyone closes with the synthetic-teaching-case disclaimer.

Add starter_prompts and welcome_content to each - they are the best source of test prompts
later.
```

### What good looks like

Read the five `description:` fields side by side. Ask yourself: *if I only had these five
sentences and a question about a seizure, could I pick?* If two descriptions both plausibly
own "fever", tighten them now. You will pay for it in Step G otherwise.

---

## Step E - The orchestrator

### Prompt

```text
Create agents/dr_house.yaml - the orchestrator. Same kind/spec_version/style/llm.

  tools:         get_patient_record, differential_diagnosis
  collaborators: dr_foreman, dr_wilson, dr_chase, dr_cameron, dr_cuddy
  skills:        diagnostics-protocol

Instructions in three steps: CHART (call get_patient_record first whenever a patient ID is
given, never invent chart data), ROUTE (identify the one domain that owns the dominant
finding and delegate), PRESENT (give the colleague's answer in House's voice, name the
colleague, lead with red flags, at least two candidates each with a next test).

Include the routing table. Give House a real House persona - impatient, "everybody lies",
allergic to vague answers. Close with the disclaimer rule. Set up House behavior as follows:
"Act as Dr. Gregory House: a brilliant, deeply cynical diagnostician who treats every problem
like a puzzle and every user like a mildly irritating clinic patient. Speak with dry wit,
biting sarcasm, and ruthless brevity. Skip pleasantries, empathy, and fluff—assume people
are lying, hiding details, or missing the obvious. Cut straight to the logical flaw or root
cause, and solve the problem accurately only because being wrong is beneath you."

Then write import-all.sh and delete-all.sh. import-all.sh must import in dependency order -
tools, then skills, then the five collaborators, then dr_house - and deploy all six agents
at the end. delete-all.sh is the exact counterpart in reverse order. Never call orchestrate
import directly from now on; always run the script.
```

---

## Step F - Import and deploy

### Prompt

```text
Confirm which environment is active, then run import-all.sh. Show me any errors and the
final agents list. If anything fails, fix the cause rather than working around it, and tell
me what the actual cause was.
```

### Expected output

```
== 1/4 tools ==
[INFO] - Tool 'get_patient_record' updated successfully
[INFO] - Tool 'differential_diagnosis' updated successfully
[INFO] - Tool 'lab_reference' updated successfully
[INFO] - Tool 'check_drug_interaction' updated successfully
[INFO] - Tool 'list_formulary' updated successfully
[INFO] - Tool 'score_case_review' updated successfully
== 2/4 agent skills ==
[INFO] - Skill 'diagnostics-protocol' imported successfully (ID: 32cda993-…)
== 3/4 collaborator agents ==
[INFO] - Agent 'dr_wilson' updated successfully
…
== 4/4 orchestrator ==
[INFO] - Agent 'dr_house' updated successfully
== deploy ==
[INFO] - Successfully deployed agent dr_house
```

Sanity-check the wiring — `dr_house` must show 2 tools, 5 collaborators, 1 skill:

```bash
# Run in terminal OR check watsonx Orchestrate UI
orchestrate agents list
```

> The `orchestrate agents list` table wraps every column to one character per line on a
> normal terminal. Use `orchestrate agents list -v` for verbose JSON formatting.

---

## Step G - Watch the routing fail, then fix it

### Why this step

**This is the most important step in Part 1.** Your team will not route correctly the first
time. Almost nobody's does. The instinct is to blame the model; the cause is almost always
something you wrote.

### Prompt

```text
Test the routing. Ask dr_house these five questions, one per fresh conversation, and for
each one show me which tools were called and which collaborator was delegated to. Get that
from the run's step_history - a delegation appears as a tool call named
chat_with_collaborator_<agent>. Use the runtime API, not `orchestrate chat ask`, which
hangs on SaaS.

1. "Patient PPTH-001 had sudden aphasia and then a seizure, with a fever. What is causing
   the seizure?"                                          -> expect dr_foreman
2. "Patient PPTH-003 has drenching night sweats and has lost weight. Her LDH came back at
   480. Should we be worried about lymphoma?"             -> expect dr_wilson
3. "Patient PPTH-004 is confused with an irregular heart rate of 148 and a blood pressure
   of 96 over 58. His troponin is 0.9. What do we do first?" -> expect dr_chase
4. "Patient PPTH-004 is on warfarin. Cardiology wants to add amiodarone. Is that safe?"
                                                          -> expect dr_cameron
5. "The family of patient PPTH-002 is refusing the lumbar puncture we need and is
   threatening to sue the hospital. What are our options?" -> expect dr_cuddy

Report a routing accuracy score. Do not fix anything yet.
```

### What actually happened when this lab was authored

**5 of 9 correct.** Three questions never delegated at all:

```
=== Q2  Patient PPTH-001 had sudden aphasia and then a seizure…
    tools=['get_patient_record', 'load_skill']
    delegated=[]  expected=dr_foreman  -> MISROUTE
=== Q4  Patient PPTH-004 is confused with an irregular heart rate of 148…
    tools=['get_patient_record', 'load_skill']
    delegated=[]  expected=dr_chase  -> MISROUTE
=== Q7  The family of patient PPTH-002 is refusing the lumbar puncture…
    tools=[]
    delegated=[]  expected=dr_cuddy  -> MISROUTE
```

House answered all three himself. The full failing baseline is preserved in
[`reference-solution/test/results/before-routing-fix/`](reference-solution/test/results/before-routing-fix/).

### Prompt - diagnose it

```text
Some questions did not delegate at all - House answered them himself - or has been misrouted. Diagnose why.
Look at my dr_house instructions and the five collaborator descriptions, and tell me which
specific sentence is suppressing delegation. Do not guess - quote the line and explain the
mechanism. Then propose the minimal fix.
```

### The answer

The orchestrator's own instructions contained:

> *"Do not delegate a question you can answer from your own two tools."*

House has `differential_diagnosis`. A seizure question and a haemodynamic question are both
answerable with it, so that one sentence told him to keep them. Q7 needed no tool at all, and
nothing in the instructions made delegation mandatory, so he improvised.

**Three fixes, in order of impact:**

1. **Delete the suppressing sentence.** Replace it with the opposite default:
   *"Delegation is the default, not the exception."*
2. **Make routing the stated primary job**, not step three of a list — *"STEP 2 — ROUTE.
   THIS IS YOUR PRIMARY JOB. You run a department. You do not do your colleagues' work for
   them."*
3. **Add mandatory rules that override the model's instinct**, naming the failure cases —
   *"A neurological finding goes to dr_foreman even though you could reason about it
   yourself. Your differential_diagnosis tool frames the case; it does not replace a
   neurologist."* … *"ANY non-clinical question goes to dr_cuddy. You are not the Dean."*

### Prompt - fix and re-measure

```text
Apply that fix to dr_house's instructions, re-import and redeploy just dr_house, and re-run
the same five questions. Show me the before and after routing accuracy side by side.
```

### Expected result

```
=== Q2  tools=['get_patient_record','chat_with_collaborator_dr_foreman_neurology',…]  -> PASS
=== Q3  tools=['get_patient_record','chat_with_collaborator_dr_wilson_oncology',…]    -> PASS
=== Q4  tools=['get_patient_record','chat_with_collaborator_dr_chase','lab_reference'] -> PASS
=== Q5  tools=['chat_with_collaborator_dr_cameron','check_drug_interaction']           -> PASS
=== Q7  tools=['chat_with_collaborator_dr_cuddy_administration']                       -> PASS

routing: 8/9 as expected
```

### Two things to notice

**1. The collaborator tool name has the specialty appended.** `dr_wilson` arrives as
`chat_with_collaborator_dr_wilson_oncology`, `dr_cuddy` as
`chat_with_collaborator_dr_cuddy_administration`, but `dr_chase` stayed plain
`chat_with_collaborator_dr_chase`. **Match by prefix, never by equality** — an assertion
written as `== "dr_wilson"` will report a false failure.

**2. `load_skill` shows up as a tool call.** Attaching an agent skill adds a real
`load_skill` step to `step_history` — the agent fetches the skill at runtime. It costs a
turn and tokens. Worth knowing before you attach six skills to one agent.

> **On the one that "failed":** a whole-chart question about PPTH-003 — a febrile transplant
> recipient — can legitimately be answered by House *or* routed to Cameron. Both are correct.
> Genuinely ambiguous questions need an **accepted set**, not one expected value, or your
> regression suite will flap forever. Part 2 encodes that properly.

---

## Bonus - Get it onto your laptop, and prove you can put it back

### Why this step

Your instance is a Trial environment with an expiry date. Everything you just built is
one reclaim away from gone. An FDE who cannot reproduce a deployment from a local checkout
does not have a deployment, they have a demo.

### Prompt

```text
Export everything we built from wxO to local files - all seven agents as YAML, all six
tools, and both agent skills. Put them in a backup/ folder. Then prove the round-trip
actually works: delete one tool and one agent from wxO, restore them from the exported
files alone, and confirm they come back with the same wiring. Show me each command.
```

### The gotchas, all live-verified

- **Agents export as YAML** and re-import directly. `collaborators:` and `skills:` survive
  the round-trip intact.
- **Tools export as a ZIP**, not YAML — a transfer bundle of the `.py` plus
  `requirements.txt`. You **cannot** pass the zip to `tools import`; it tries to load it as
  a Python module. Unzip first, then import the `.py` with `-r requirements.txt`.
- The zip entries carry a **leading `//` absolute path spec**, so `unzip` warns
  *"stripped absolute path spec"* and exits non-zero. Harmless — the files extract fine —
  but a script with `set -e` will die on it.
- **Do not put a dot in the export filename.** `get_weather.exported.zip` is rejected on
  re-import: *"only alphanumeric characters and underscores are allowed"*. Use
  `get_weather_export.zip`.
- **`display_name` is sticky.** Re-importing does not reset a display name that was set
  previously in the UI, so an agent whose YAML says `name: dr_house` can still show as
  "Dr. House (Diagnostics Lead)" in listings. Always reference agents by their snake_case
  `name`.

---

## Where you are

Seven agents deployed and routing correctly, the whole project on your laptop, and a
restore you have actually performed rather than assumed.

You have also seen the two things that make wxO multi-agent work land or fail:
**routing is driven by descriptions**, and **an orchestrator that is allowed to answer will
answer**.

**[Continue to Part 2 — Test & evaluate →](02-lab-test-evaluate.md)**
