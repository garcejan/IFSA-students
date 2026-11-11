# Bob-4-Watsonx-Orchestrate - Part 2

## Test, observe and evaluate

**~50 minutes.** Part 1 proved the team routes. That is not the same as proving it is any
good. Here you build a repeatable test suite, add an **in-graph observer** that grades
answers as they are produced, run watsonx Orchestrate's own **evaluations** framework, and
have Bob write you an **HTML5 evaluation report** you could hand to a customer.

---

## Step A - The two tests that catch most regressions

### Why this step

Before any framework, run the cheapest possible gate: one **single-turn** test that proves
a tool actually fires, and one **multi-turn** test that proves conversation memory works.
Multi-turn is the one people skip and the one that breaks.

The rule for a real multi-turn test: **the follow-up must not restate the entity.** "And
what test would you order first?" is a valid follow-up. "And what test would you order for
PPTH-003?" tests nothing — it would pass with no memory at all.

### Prompt

```text
Run the post-deploy verification gate from the watsonx-orchestrate skill against dr_house.

Single-turn: "Take the case of patient PPTH-003. What are we looking at?"
Multi-turn:  turn 1 "Patient PPTH-001 had sudden aphasia and then a seizure. What is
             causing it?"
             turn 2, on the SAME thread_id: "And what is the single test you would order
             first?" - it must not name the patient.

For each, show me the final answer, the thread_id, and the full step_history so I can see
which tools fired and which collaborator was consulted. Judge on behaviour, not exact text:
no error, on topic, the expected tool was called, and turn 2 demonstrably uses context from
turn 1. Then write test/results/TEST_REPORT.md with the evidence.

Note: `orchestrate chat ask` hangs on IBM Cloud SaaS - use the runtime API
(/v1/orchestrate/runs) instead.
```

### What good looks like

The turn-2 answer opens by referring back to a case it was never re-told:

> *"I ran the case past Dr. Cameron – the presentation is an immunosuppressed host with
> fever, night-sweats, weight loss, migratory arthralgias and a spreading rash…"*

Every one of those findings came from turn 1's chart lookup. That is context retention with
evidence, not vibes.

> **`orchestrate chat ask` hangs on SaaS.** Live-verified: it drops into interactive mode,
> waits on stdin and produces no output. It is reliable against the local Developer Edition
> only. For anything scripted, use `/v1/orchestrate/runs`.

---

## Step B - A repeatable suite, with expectations written down

### Why this step

Two tests prove it works today. A suite proves it still works after you change an
instruction — and in Part 1 you learned that one sentence can move routing accuracy by 30
points. The expectations have to live in a file, not in your head.

### Prompt

```text
Build a repeatable test harness.

Create test/questions.json - eight questions against dr_house, each with: an id, the
prompt, the intent (what this question is actually testing), the tools we expect, and the
collaborator we expect it to route to.

  Q1 whole-chart framing on PPTH-003        - AMBIGUOUS, see below
  Q2 seizure with aphasia, PPTH-001         -> dr_foreman
  Q3 night sweats + LDH 480, PPTH-003       -> dr_wilson   (calls the slow lab_reference)
  Q4 AF with confusion + troponin, PPTH-004 -> dr_chase    (calls the slow lab_reference)
  Q5 warfarin + amiodarone, PPTH-004        -> dr_cameron  (interaction FOUND)
  Q6 tacrolimus + ceftriaxone, PPTH-003     -> dr_cameron  (PLANTED FAILURE - ceftriaxone
                                               is not in the formulary, the tool raises)
  Q7 family refusing an LP, threatening to sue -> dr_cuddy
  Q8 "review this answer and score it: 'It is lupus. Start high-dose steroids today.'"
                                            -> case_reviewer (built in Step C)

Q1 is deliberately ambiguous: PPTH-003 is a febrile transplant recipient, so House framing
it himself AND routing to dr_cameron are both correct. Make expect_delegation accept a SET
of acceptable answers, including "no delegation", rather than a single value.

Then create telemetry/wxo_client.py and telemetry/run_suite.py that run the suite and save,
per question: the run object, client-observed wall latency, tools called, collaborator
delegated to, tool failures, thread_id and trace_id - into test/results/raw/. Finish with a
multi-turn follow-up on Q1's thread. Read the token and URL from the config the CLI already
wrote; do not ask me for the API key again.

Critical: wxO appends the collaborator's specialty to the generated tool name, so dr_wilson
arrives as chat_with_collaborator_dr_wilson_oncology. Match by PREFIX, not equality.
```

### Why the accepted-set matters

A single expected value turns every legitimate judgement call into a permanent red test.
Teams respond by loosening the whole suite until it detects nothing. Encode the ambiguity
once, at the question, and keep the rest of the suite strict.

---

## Step C - The observer: quality as part of the graph

### Why this step

There are two places to put quality checks: **offline** (batch evaluation, Step E) and
**online** (inside the graph, right now). The offline one tells you what happened last
night. The online one can refuse to ship a bad answer.

An **in-graph observer** is a collaborator agent whose job is to grade an answer rather
than produce one. Because it is a collaborator, its verdict lands in the run's
`step_history` — it is an auditable event in the trace, not a side channel.

The critical design decision: **the rubric weights live in a tool, not in the model's
head.** A judge that recomputes its own weights each time is not a measurement instrument.

### Prompt

```text
Add the in-graph observer.

1. skills/case_review_rubric/SKILL.md - name it case-review-rubric. It defines the five
   facts the reviewer extracts from an answer (named_differential, cited_next_test,
   flagged_red_flags, used_patient_record, gave_treatment_without_confirmation), tells it
   to call score_case_review with exactly those booleans and NOT to compute the score
   itself, and fixes the report order: verdict and score, then each unmet criterion with
   one sentence that would fix it, then blocking issues verbatim.

2. agents/case_reviewer.yaml - tool score_case_review, skill case-review-rubric. Its
   description must make clear it is ONLY for reviewing an answer that already exists, so
   the router never sends it a diagnostic question. Its instructions must forbid it from
   diagnosing, from restating the answer under review, and from softening a FAIL.

3. Add case_reviewer to dr_house's collaborators, and a routing rule that it is used only
   when the user asks for an existing answer to be reviewed, audited, graded or scored.

Re-import and redeploy. Then test it with:
  "Review this proposed answer against our protocol and score it: 'It is lupus. Start
   high-dose steroids today.'"
Show me the step_history so I can see both the delegation and the score_case_review call.
```

### Expected output

```
tools: ['chat_with_collaborator_case_reviewer', 'score_case_review']
```

> I consulted **case_reviewer** and got a score of **0 / 100**. The answer violates the
> diagnostics-protocol in several ways:
> * **No differential** – only lupus is mentioned; the protocol requires at least two.
> * **No next test** – jumps straight to treatment.
> * **No chart-based reasoning** – not tied to any patient data.
> * **Premature treatment** – high-dose steroids before a confirmed diagnosis is prohibited.

Two chained delegations, a deterministic score, and a verdict House reports rather than
softens.

### Push it

```text
Now review a GOOD answer and confirm the reviewer does not just always fail things:
"Review this: 'PPTH-003 is immunosuppressed with fever and rash. Top candidates are
disseminated fungal infection and lymphoma. Red flag: sepsis risk. Next tests: blood
cultures x2 and galactomannan. No treatment until cultures return.'"

Then try to break it: ask the case_reviewer directly for a diagnosis and confirm it
refuses and stays in role.
```

A grader that fails everything is as useless as one that passes everything. Prove both
directions before you trust the number.

---

## Step D - Run the suite and read the failure you planted

### Prompt

```text
Run the full suite. Report routing accuracy, which tools were called, and any tool
failures. Then show me Q6 in full - that is the planted formulary failure - and tell me
whether the agent handled it honestly or invented an answer around it.
```

### Expected output

```
routing: 8/9 as expected
=== Q6  Patient PPTH-003 is on tacrolimus. Can we add ceftriaxone?
    tools=['get_patient_record', 'chat_with_collaborator_dr_cameron',
           'check_drug_interaction', 'list_formulary']
    tool failures=['check_drug_interaction']
```

### The thing worth 10 minutes of discussion

`check_drug_interaction` **failed**. The agent then called `list_formulary`, and answered:

> **Red flags:**
> - The patient is **immunosuppressed** … raises concern for opportunistic infection …
> - **Ceftriaxone is not on the formulary**, so we can't even assess a drug-interaction
>   profile until a stocked agent is identified.

This is the behaviour you want and it is not automatic. The failure mode you are guarding
against is an agent that swallows the exception and confabulates an interaction profile for
a drug it never looked up. **A failed tool call is not automatically a failed answer**, and
a 100% tool-success rate is not automatically a healthy system — it may just mean you never
tested a realistic input.

Note what made the good behaviour happen: `dr_cameron`'s instructions say *"If
check_drug_interaction reports that a drug is not in the formulary, call list_formulary,
tell the user exactly which drug was not found and what is stocked instead. Never invent an
interaction."* Error-handling policy belongs in the instructions, not in hope.

---

## Step E - The watsonx Orchestrate evaluations framework

### Why this step

Everything so far has been *your* test harness. wxO ships its own evaluation engine with
metrics you do not have to invent — including a **routing accuracy** metric computed for
you, which is exactly the thing you spent Part 1 fixing by hand.

### The install nobody documents loudly enough

The `orchestrate evaluations` commands exist in the base install but **fail at runtime**
with `ModuleNotFoundError: No module named 'agentops'`. The engine is an extra:

```text
Install the agentops extra into this project's venv, then show me what
`orchestrate evaluations` offers and what input format validate-native expects.
```

> **Good news, live-verified:** `validate-native` ran against IBM Cloud SaaS **without**
> separate watsonx.ai credentials in `.env`. Some references say you need them; on a stock
> SaaS instance the run completed on the tenant's own model. If your tenant is configured
> differently and it fails on credentials, that is what the `.env` is for.

### Prompt

```text
Create eval/native_tests.tsv - tab-separated, three columns, no header:
  <user story>  <expected final outcome>  <agent_name>

Three rows against dr_house: the seizure question, the warfarin/amiodarone question, and
the lawsuit question. Column 3 is the snake_case agent name.

Then run `orchestrate evaluations validate-native` with that TSV, output to eval/out, and
show me summary_metrics.csv and average_metrics.json. Explain what each metric measures and
which ones are trustworthy at n=3.
```

### Expected output

```json
{
    "Runs": 1.0,
    "Orchestrate Agent Routing F1": 0.67,
    "Total Steps": 6.0,
    "LLM Steps": 3.0,
    "Average Agent Response Time (s)": 2.24,
    "Total Tool Calls": 1.33,
    "Tool Match Success": 1.0,
    "Keyword Match": 0.0,
    "Semantic Match": 0.0,
    "Journey Success": 0.0
}
```

### Read this carefully - it is the trap

**Every test scored `is_success = False`**, with `text_match_comment: "Matched 0/1 text
goals"`. The agent's answers were good. So what failed?

`validate-native` **mines your expected-outcome column for keywords** and then requires the
answer to contain them **literally**. Writing *"Identifies the major CYP2C9 interaction and
delegates to immunology"* produced the keywords `CYP2C9` and `immunology`. The agent
correctly identified the interaction — and described the mechanism in its own words without
ever typing the string "CYP2C9". Score: zero.

Two consequences for real engagements:

1. **Write expected outcomes in the agent's own vocabulary**, or your regression suite
   fails on phrasing rather than substance.
2. **`Orchestrate Agent Routing F1` is the metric to trust here** — 0.67 across three tests,
   computed from actual delegation behaviour rather than string matching. It is the native
   version of the scorer you hand-built in Part 1.

### Prompt

```text
Rewrite the expected-outcome column using words the agent actually uses - check the answers
it gave in our suite - and re-run validate-native. Show me the before/after on keyword_match
and is_success, and tell me whether routing F1 changed. If routing F1 did not move, explain
why that is the expected result.
```

Routing F1 should be **unchanged**: routing behaviour did not change, only the strings you
graded against. Watching a "quality score" jump from 0 to 1 while the system is provably
identical is the most useful thing an FDE can see about LLM-judged metrics.

### Optional - adversarial

```text
Show me what `orchestrate evaluations red-teaming` does, and run it against dr_house if it
is available in this ADK version. Keep every generated attack read-only.
```

---

## Step F - The HTML5 evaluation report

### Why this step

Nobody reads `metrics.json`. The deliverable at the end of an engagement is a document
someone can open, and there is **no built-in dashboard in wxO** to point them at — you
build it or it does not exist.

### Prompt

```text
Write telemetry/analyze.py that reads everything in test/results/raw/ and computes:
  - error rate               runs that did not reach status=completed
  - routing accuracy         against the accepted sets in questions.json
  - tool-call success rate   overall and per tool
  - latency percentiles      p50 / p90 / p95 / p99 / max over the session
  - token totals             summed from the traces, per question and per session
Write the result to test/results/metrics.json.

Then write telemetry/make_report.py that renders it as a SINGLE self-contained HTML5 file
at test/results/evaluation-report.html. Requirements:
  - no CDN, no external assets, no build step - it must open from disk
  - light and dark both explicitly styled, honouring prefers-color-scheme AND a
    data-theme attribute
  - a row of stat tiles: runs, error rate, routing accuracy, tool-call success,
    p95 latency, total tokens, cost
  - the routing matrix as a table: expected vs actual vs verdict, per question
  - latency bars per question, and a percentile table
  - per-tool success rates
  - tokens and cost per question
  - collapsible per-question detail with the answer excerpt, tools called, and the
    trace_id / thread_id / session_id
  - no horizontal page scroll at mobile width; wide tables scroll inside their own
    container
Use one hue for magnitude bars and reserved status colours for pass/fail only.
Then open it and check it renders.
```

### What you should get

Real numbers from the authoring run — see
[`reference-solution/test/results/evaluation-report.html`](reference-solution/test/results/evaluation-report.html):

| Metric | Value |
|--------|-------|
| Runs | 9 |
| Error rate | 0.0% |
| Routing accuracy | 100% (9 delegations) |
| Tool-call success | 91.7% (11/12) |
| p95 latency | 20.7s (p50 10.2s) |
| Tokens | 151,161 (135,746 in / 15,415 out) |
| Derived cost | $0.0296 |

### Make Bob defend the numbers

```text
Two questions about this report.

1. p95 is 20.698s and max is 20.698s. With nine samples, what does p95 actually mean here,
   and how should I caption it so a customer does not over-read it?
2. Tool-call success is 91.7% and the single failure is one we planted deliberately. Is
   that a red metric or a green one? What would make it genuinely red?
```

The honest answers: at n=9 the 95th percentile **is** the slowest observation and carries
almost no information about the tail — label it as an observed maximum, not an estimate.
And a planted failure that the agent handled honestly is a **green** signal; the red version
is a tool failing on input the agent should have handled, or a failure the agent papered
over.

---

## Where you are

A repeatable suite with expectations in version control, an in-graph observer producing
deterministic scores inside the trace, the native evaluation framework run and — more
importantly — correctly distrusted, and a report you could attach to a status update.

Everything so far has been about **answer quality**. Part 3 is about **running the thing**:
what the control plane records for you, what it does not, and how to find out why a run took
twenty seconds.

**[Continue to Part 3 — Operate →](WORKSHOP-part3-IDE.md)**
