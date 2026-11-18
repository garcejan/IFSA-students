# Reference solution — Princeton-Plainsboro diagnostics department

The complete, working project from **IFSA workshop**, exactly as it was built
and validated while authoring the lab, including the captured evidence.

**Use it as a checkpoint, not a shortcut.** The learning is in prompting Bob to produce
these artifacts and debugging what comes back. Come here when you are stuck, when you want
to compare your agent instructions against ones that route correctly, or when you want to
know what the captured telemetry is supposed to look like.

## Deploy it

```bash
export WXO_INSTANCE_URL="https://api.<region>.watson-orchestrate.cloud.ibm.com/instances/<INSTANCE_ID>"
export WXO_API_KEY="<your-api-key>"

source ../../SKILL/watsonx-orchestrate/references/setup-venv.sh   # or Bob's copy of the skill
orchestrate env add -n houselab -u "$WXO_INSTANCE_URL"
orchestrate env activate houselab --api-key "$WXO_API_KEY"

./import-all.sh          # tools -> skills -> collaborators -> orchestrator -> deploy
```

> **`import` is upsert-by-name.** On a shared instance this **overwrites** any existing
> `dr_house`, `dr_wilson`, `dr_foreman`, `dr_chase`, `dr_cameron`, `dr_cuddy` or
> `diagnostics-protocol` with no backup. Export what is there first, or rename everything.

## Run the suite

```bash
python telemetry/run_suite.py         # 8 questions + a multi-turn follow-up
python telemetry/analyze.py           # -> test/results/metrics.json
python telemetry/make_report.py       # -> test/results/evaluation-report.html
open test/results/evaluation-report.html
```

## Layout

```
agents/          dr_house (orchestrator) + 5 specialists + case_reviewer (observer)
tools/           patient_registry · differential_diagnosis · lab_reference
                 drug_interaction (2 tools) · case_review
skills/          diagnostics-protocol · case-review-rubric
telemetry/       wxo_client.py    runtime API client, reads the CLI's own token
                 run_suite.py     runs the questions, captures runs + traces
                 analyze.py       derives error rate, percentiles, tool success, cost
                 make_report.py   renders the self-contained HTML5 report
                 rate_card.json   YOUR prices - wxO reports totalCost = 0
test/
  questions.json          the 8 questions with accepted-answer sets
  results/
    session.json          captured runs
    metrics.json          computed metrics
    evaluation-report.html
    suite.log
    before-routing-fix/   the FAILING baseline: 5/9 routing, before the fix
    example_trace_with_judge_score.json
eval/out/        a real `orchestrate evaluations validate-native` run
webchat/         index.html - embedded web-chat page (placeholders, paste your own snippet)
import-all.sh · delete-all.sh
```

## The two deliberate defects

Both are load-bearing for Part 3. **Do not fix them.**

| Where | What | Found in |
|-------|------|----------|
| `tools/lab_reference.py` | `time.sleep(4.5)` — a slow Laboratory Information System | Part 3 Step 3, from the trace |
| `tools/drug_interaction.py` | raises for any drug outside the formulary | Part 2 Step D, Part 3 Step 5 |

## Measured results from the authoring run

ADK 2.13.0 · IBM Cloud SaaS · `groq/openai/gpt-oss-120b` · 2026-08-05

| Metric | Before routing fix | After |
|---|---|---|
| Routing accuracy | 5/9 | **8/9** strict · **9/9** with accepted sets¹ |
| Error rate | — | **0.0%** |
| Tool-call success | — | **91.7%** (11/12, one planted) |
| Latency p50 / p95 / max | — | **10.2s / 20.7s / 20.7s** |
| Tokens (9 runs) | — | **151,161** (135,746 in / 15,415 out) |
| Derived cost @ placeholder rates | — | **$0.0296** |
| Cost reported by `trace.totalCost` | — | **$0** |

¹ Q1 is deliberately ambiguous — a whole-chart question about a febrile transplant recipient
can legitimately be framed by House *or* routed to Cameron. `questions.json` encodes an
accepted **set** for it; `suite.log` shows the strict 8/9, `metrics.json` the 9/9.

Session ids, thread ids and trace ids in the captured results are from the authoring
instance; account and instance identifiers have been replaced with placeholders. Your run
will produce different ids and different token counts.

## Not for clinical use

Every patient, chart, laboratory range, formulary entry and interaction in this project is
**fictional**. The medicine is deliberately shallow — the orchestration is the subject.
