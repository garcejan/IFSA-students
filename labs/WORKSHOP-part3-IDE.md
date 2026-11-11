# Bob-4-Watsonx-Orchestrate — Part 3

## Operate it: the control plane

**~60 minutes.** Parts 1 and 2 were about whether the team gives good answers. This part is
about running it in production: what watsonx Orchestrate records **for** you, what it
records **nothing** about, how to find out why a run took twenty seconds, what a
conversation costs, and how to ship the thing to a web page.

The organising question for the whole part:

> **Which of these numbers does the platform give me, and which am I on the hook to
> compute?**

Getting that boundary wrong is how teams end up either rebuilding telemetry that already
exists, or promising a customer a cost dashboard that no API can populate.

---

## Step 1 - Find the boundary yourself

### Why this step

Do not take a table's word for it, including this lab's. The surfaces move between
versions. Have Bob interrogate the actual payloads on **your** tenant and report what is
really there.

### Prompt

```text
I want to map exactly what the watsonx Orchestrate control plane captures natively for a
single run, versus what I would have to derive myself.

Take one completed run from our suite and, for it, dump and compare THREE surfaces:

  1. the runs API response            GET /v1/orchestrate/runs/{run_id}
  2. the CLI trace export             orchestrate observability traces export --trace-id ...
  3. the AgentOps v3 trace API        GET /v1/agentops-v3/traces/{trace_id}

For each surface list the top-level keys and, for observations, the full key set. Then give
me one table: signal (latency, per-step latency, token counts, model, cost, error state,
session id, quality score, span parent/child), which surface carries it, and whether the
value is actually POPULATED or just present-and-empty. I care about the difference between
"the field exists" and "the field has a number in it".
```

### What you will find

Three surfaces, three different levels of detail. Live-verified, ADK 2.13.0, IBM Cloud SaaS:

**The runs API** carries `trace_id`, `thread_id`, `status`, `started_at`, `completed_at`,
`step_history` — and a `usage` key that is **`null`**.

```json
{ "status": "completed",
  "trace_id": "291eb99557526f679a362846b2aeb4b1",
  "thread_id": "d2a6a5d6-…",
  "usage": null,
  "llm_params": null }
```

> **Do not build token accounting on `run.usage`.** The field is there; it is empty. This is
> the single most common wrong turn in wxO observability work.

**The CLI export** (`traces export --trace-id`) returns *observations*, 11 fields each:
`id · traceId · type · name · model · input · output · usage · startTime · endTime ·
metadata`. Enough for tokens, and for latency **if you subtract the timestamps yourself**.

**The AgentOps v3 API** returns those 11 plus **30 more**, including the ones you actually
need for forensics:

| Field | Why you need it |
|-------|-----------------|
| `latency` | per-observation seconds, precomputed |
| `parentObservationId` | **the only way to rebuild the span tree** |
| `level`, `statusMessage` | error state per span |
| `timeToFirstToken` | streaming responsiveness |
| `calculatedTotalCost`, `totalPrice`, `costDetails` | present — and zero, see Step 4 |

**Rule of thumb:** CLI export for a quick look or a token count; **AgentOps v3 for anything
that has to explain a number.** Without `parentObservationId` you cannot tell a 6-second
outer span wrapping a 2-second inner one from a genuinely 6-second operation — and that
distinction is the whole of Step 3.

### The table to walk through

| Signal | Native? | Where | What you do |
|---|---|---|---|
| End-to-end latency | **native** | `trace.latency` | read it |
| Per-step latency | **native** | `observation.latency` (v3 API only) | read it |
| Token counts | **native** | `observation.usage.input/.output` | sum them |
| Model identity | **native** | `observation.model` | needed to price tokens |
| Session / caller | **native** | `trace.sessionId`, `trace.userId` | your roll-up key |
| Quality scores | **native** | `trace.scores[]` | async + sampled — Step 6 |
| Span tree | **native** | `observation.parentObservationId` | v3 API only |
| Tokens in runs API | **absent** | `run.usage` is `null` | use the trace |
| **Cost** | **not computed** | `trace.totalCost` is `0` | your rate card — Step 4 |
| **Error rate** | **derive** | — | runs not `completed` ÷ total |
| **Tail latency** | **derive** | — | percentiles over a window |
| **Tool-call success** | **derive** | — | inspect each `tool_response` |
| **Dashboards** | **none** | — | you build it — Step 5 |

---

## Step 2 - Search traces like an operator

### Prompt

```text
Show me how to find traces without already knowing the trace id. Use
`orchestrate observability traces search` with a relative window, and also filtered by the
session id from our suite. Explain which filters actually work in this ADK version and
which are deprecated no-ops, and show me the raw command each time.
```

### Expected output

```
                                   Found 83 traces
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Timestamp               ┃ Trace ID                         ┃ Agent Name ┃ Latency ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━┩
│ 08/05/2026 , 10:14:59AM │ 291eb99557526f679a362846b2aeb4b1 │ dr_house   │ 18.792s │
│ 08/05/2026 , 10:14:42AM │ 79597fb9ef79d7274ec9686ff1f6aa35 │ dr_house   │ 15.817s │
│ 08/05/2026 , 10:14:31AM │ 2afc921e3dfa2444c3d44558575fff3e │ dr_house   │  8.565s │
│ 08/05/2026 , 10:14:07AM │ 57b26b031acbb15e232efef5f64fa4e7 │ -          │   200ms │
```

**What works:** `--last 30m|3h|10d`, or `--start-time`/`--end-time`; `--user-id` and
`--session-id`, both repeatable; `--sort-field`/`--sort-direction`.

**What is deprecated and silently does nothing:** `--agent-name`, `--agent-id`,
`--service-name`, `--min-spans`, `--max-spans`. The API no longer filters by agent — note
that the *result table still shows an Agent Name column*, which makes it very easy to
believe `--agent-name` worked. Filter client-side instead.

The `-` rows with sub-second latency are infrastructure calls, not agent runs. Your error
rate must exclude them or it will be meaningless.

---

## Step 3 - Read a trace to diagnose latency

### Why this step

This is the exercise the whole lab was built around. Q4 took **20.7 seconds**. A customer
will ask why. "The LLM is slow" is not an answer, and in this case it is not even true.

### Prompt

```text
Q4 - the atrial fibrillation question - was our slowest run at about 20 seconds. Diagnose
it from the trace alone. Pull the AgentOps v3 trace, list every observation sorted by
latency descending with its type and name, reconstruct the parent/child structure using
parentObservationId, and tell me:

  1. how much of the wall time was LLM generation, how much was tool execution, and how
     much was orchestration and delegation overhead
  2. which single span is the biggest contributor
  3. whether any span looks like a cold start rather than genuine work - and how you can
     tell the difference from the trace

Do not guess. Quote the observation names and latencies.
```

### What the trace actually says

```
type        name                                           latency
CHAIN       LangGraph                                       17.393
CHAIN       collaborator                                     7.206
CHAIN       LangGraph[collaborator:4eacc15b-…]               6.987
CHAIN       tools                                            6.450
SPAN        get_patient_record                               6.368   <-- outer
SPAN        POST /api/v1/runtime/tools/:toolId/run            6.210
SPAN        runtime.invoke_tool                              6.188
SPAN        HTTP POST                                        6.188
SPAN        POST /execute                                    6.180
CHAIN       tools                                            4.932
SPAN        lab_reference                                    4.862   <-- outer
SPAN        POST /execute                                    4.648
SPAN        lab_reference                                    4.506   <-- inner: the real work
SPAN        get_patient_record                               1.807   <-- inner: the real work
CHAIN       agent                                            1.616
GENERATION  WatsonxChatModel.chat                            1.567    712 in / 646 out
CHAIN       agent                                            1.447
GENERATION  WatsonxChatModel.chat                            1.350   6230 in / 1084 out
```

Three findings, none of which you could have guessed:

**1. The LLM is not the problem.** Every `GENERATION` span totals well under 4 seconds
across the whole run. The model is fast; the plumbing is not.

**2. `lab_reference` is a flat 4.506s of real work.** Inner and outer spans nearly match
(4.506 vs 4.862), so the time is genuinely inside the tool. That is the **planted
`time.sleep(4.5)`** from Part 1 Step B — found from telemetry, exactly as an unfamiliar slow
downstream system would be found.

**3. `get_patient_record` is a cold start, and the trace proves it.** The outer span is
**6.368s** but the inner one is **1.807s** — a 4.5-second gap that is *not* inside your
code. That gap is the tool runtime spinning up a container. The same tool later in the
session runs in a fraction of the time.

> **The diagnostic rule worth writing down:** when an outer span is much larger than the
> inner span it wraps, the time is in the *platform* — cold start, queueing, transport.
> When outer and inner are close, the time is in *your code* or the system it calls. Those
> two problems have completely different fixes, and you can only tell them apart with
> `parentObservationId`, which is why you need the v3 API rather than the CLI export.

### Prompt - make the finding actionable

```text
Given that breakdown, write me the three-bullet answer I would give a customer asking why
this took 20 seconds, and for each cause say whether it is fixable by us, fixable by the
platform, or inherent. Then tell me what the run would cost in latency if lab_reference
returned in 200ms, and how I would verify that claim rather than assert it.
```

---

## Step 4 - Cost and tokens at session level

### Why this step

"What will this cost us at 10,000 conversations a day?" is asked in every engagement. The
platform gives you **exact token counts** and **no money at all**.

### The finding

`trace.totalCost` and every observation's `calculatedTotalCost` come back **`0`** on a stock
SaaS instance. The fields exist, the pricing tier fields exist (`usagePricingTierName`,
`inputPrice`, `outputPrice`) — and no price sheet is attached, so everything computes to
zero. Report `$0.00` from that field and you will be wrong by exactly your whole bill.

**Tokens are native and exact. Pricing is yours.**

### Prompt

```text
Build session-level cost and token accounting.

Group our runs by trace.sessionId. For each session compute total input tokens, total output
tokens, and cost from a rate card I control. Put the rates in telemetry/rate_card.json as
USD per million tokens, keyed by the model id from observation.model, with a loud comment
that these are placeholders to be replaced with our contracted rate - because wxO reports
totalCost = 0 and will not price this for us.

Report per question and per session: tokens in, tokens out, derived cost, and the native
cost field alongside it so the gap is visible. Then answer: what does one conversation with
this team cost, and what would 10,000 a day cost at these rates?
```

### Expected output

| | |
|---|---|
| Tokens, 9 runs | **151,161** (135,746 in / 15,415 out) |
| Derived cost | **$0.0296** |
| `trace.totalCost` reported natively | **$0** |
| Mean per run | ~16,800 tokens |

### The two things that will surprise the room

**Input tokens are 90% of the bill.** 135,746 in versus 15,415 out — roughly **nine to one**.
Every collaborator delegation re-sends context, the agent skill is loaded into the prompt,
and tool results are fed back in. Optimising for shorter answers is optimising the 10%.

**Delegation is not free.** Compare a self-answered run against one that delegates: the
sub-agent gets its own context, its own system prompt, and its own generation calls, all
flattened into the parent trace. Ask Bob to quantify it:

```text
Compare token usage between a run where House answered himself and one where he delegated
to a specialist. Quantify the delegation overhead in tokens and in dollars at our rates, and
tell me at what daily volume that overhead would justify collapsing two agents into one.
```

That is the real architectural trade-off in multi-agent design: **routing quality costs
tokens**, and the number is knowable.

---

## Step 5 - The dashboards

### Why this step

There is **no built-in operations dashboard** in watsonx Orchestrate. No error-rate panel,
no latency histogram, no tool-success view. The control plane gives you the raw material;
the panel is yours. That is not a criticism — it is the boundary you have to be able to
state clearly to a customer before they assume otherwise.

### Prompt

```text
Extend telemetry/analyze.py and telemetry/make_report.py into an operations dashboard
covering the three panels an SRE would actually want:

1. ERROR RATE - runs that did not reach status=completed, over total. Separately, the
   tool-level error rate from tool_response payloads carrying an error. Exclude the
   sub-second infrastructure traces that traces search returns with no agent name -
   they are not agent runs and they will flatter the number.

2. TAIL LATENCY - p50, p90, p95, p99 and max over the session, from client wall time AND
   from trace.latency, shown side by side so the gap between what the platform measured
   and what the caller experienced is visible. Caption the percentiles with the sample
   size, and if n is small say plainly that p95 is the slowest observation rather than an
   estimate.

3. TOOL-CALL SUCCESS - per tool: calls, failures, success rate. Plus a stacked bar per
   run showing where the time went - LLM generation vs tool execution vs orchestration.

Render into the same self-contained HTML5 report. One hue for magnitude, reserved status
colours for pass/fail only, no external assets, works from disk, light and dark.
```

### Expected result

The [reference report](reference-solution/test/results/evaluation-report.html) from the
authoring run:

| Panel | Value |
|---|---|
| Error rate | 0.0% — 0 failed runs of 9 |
| Tool-call success | 91.7% — 11/12 |
| p50 / p95 / max | 10.163s / 20.698s / 20.698s |
| Routing accuracy | 100% |

### The honesty checks

Have Bob argue against its own dashboard:

```text
Three challenges to this dashboard.

1. p95 equals max. Explain why, and rewrite the caption so a customer cannot over-read it.
2. Our error rate is 0% but one tool call failed. Are those consistent? What is the
   difference between a failed tool call and a failed run, and which one should page
   someone at 3am?
3. What is this dashboard structurally unable to tell me about production, no matter how
   many runs I feed it?
```

The third answer is the important one. This measures **the runs you chose to make**. It says
nothing about the questions real users ask, the ones the agent refuses, or the ones it
answers confidently and wrongly. A synthetic suite is a regression detector, not a
production monitor — to get one of those you sample real traces by session id on a schedule.

### Optional - make it continuous

```text
Write a script I can cron that pulls the last hour of traces, computes these metrics, appends
them to a rolling JSONL, and regenerates the dashboard. Include a threshold check that exits
non-zero if error rate exceeds 2% or p95 exceeds 30s, so it can gate a pipeline.
```

---

## Step 6 - The scores you did not ask for

### Why this step

The control plane runs LLM-as-judge evaluation over traces on its own, and attaches the
results to the trace. Most people never look.

### Prompt

```text
Check whether the control plane attached any quality scores to our traces. Look at
trace.scores[] in the AgentOps v3 payload across all our runs, and report the metric name,
the value, the source and the rationale. If some traces have scores and others do not, tell
me what that implies about how the scoring job works.
```

### What you will find

A scored trace looks like this — a real one from the authoring run, preserved at
[`reference-solution/test/results/example_trace_with_judge_score.json`](reference-solution/test/results/example_trace_with_judge_score.json):

```json
{ "name": "hallucination",
  "value": 0.85,
  "source": "EVAL",
  "comment": "The answer invents extensive clinical details (age, transplant status, labs,
              differential diagnoses) that were never provided in the query, creating a
              plausible but unfounded scenario; thus it largely hallucinates while remaining
              internally consistent." }
```

**Two things every FDE needs to know about this.**

**1. It is asynchronous and sampled.** The score arrived roughly 30 seconds after the run.
Re-fetching the nine suite traces several minutes later returned **zero** scores. Do not
build a gate that blocks on a score arriving, and do not report "0 hallucinations detected"
when the truth is "no trace in this batch was sampled".

**2. That 0.85 is a false positive, and understanding why matters.** The agent did **not**
invent the transplant status or the labs — it read them from the chart via
`get_patient_record`. The judge compared the final answer against the **user's question**
alone, without the tool output in scope, so correctly-retrieved facts look like
fabrication.

This is the central failure mode of automated hallucination scoring on **tool-using**
agents, and it will bite any customer who wires that number to an alert.

```text
Given that the hallucination judge scored a correct, tool-grounded answer as 0.85, how would
you design a hallucination check for THIS agent that does not produce that false positive?
Consider what evidence a judge would need in scope. Compare that to what our in-graph
case_reviewer from Part 2 does differently, and say which of the two you would put in front
of a customer.
```

The case_reviewer scores against a **deterministic rubric in a tool**, using criteria that
reference retrieved evidence. It is narrower and far more defensible. Native judges are for
triage across volume; a rubric observer is for a claim you have to stand behind.

---

## Step 7 - Ship it: the embedded web chat

### Why this step

The demo that lands is the one someone can click. wxO generates a drop-in `<script>`
snippet — with two failure modes that will stop you cold if you do not know them.

### Prompt

```text
Generate the embedded web chat snippet for dr_house and build a standalone HTML5 page
around it in webchat/index.html - a Princeton-Plainsboro landing page with a few suggested
prompts and the div the launcher mounts into.

Two known problems to handle:
  1. `orchestrate channels webchat embed` tries to fetch the instance CRN from the IBM Cloud
     resource-controller and gets 403 with an instance-scoped API key, then prompts on stdin
     and aborts. Extract the CRN from the bearer token's unique_instance_crns claim and pipe
     it into the command.
  2. The CLI renders the snippet to the terminal width, so it arrives hard-wrapped and
     broken. Set COLUMNS wide before running it.

Then serve the page locally and confirm the launcher renders.
```

### The commands, live-verified

```bash
CRN=$(python3 -c "
import yaml,os,json,base64
t=yaml.safe_load(open(os.path.expanduser('~/.cache/orchestrate/credentials.yaml')))['auth']['$WXO_ENV_NAME']['wxo_mcsp_token']
p=t.split('.')[1]; p+='='*(-len(p)%4)
print(json.loads(base64.urlsafe_b64decode(p))['unique_instance_crns'][0])")

echo "$CRN" | COLUMNS=400 orchestrate channels webchat embed --agent-name dr_house --env live
```

Without the pipe you get:

```
[ERROR] - Error retrieving CRN: 403 Client Error: Forbidden for url:
          https://resource-controller.cloud.ibm.com/v2/resource_instances/<INSTANCE_ID>
[WARNING] - Could not get CRN automatically
Please enter your CRN which can be retrieved from the IBM Cloud UI: Aborted.
```

Without `COLUMNS=400` the snippet arrives wrapped mid-string and the page silently fails.

### What to expect when you open it

The page renders, `wxoLoader.js` loads from the host URL, and **the launcher appears**.

Then, from a plain `localhost` origin with no authenticated session, the loader's backend
calls return **401**:

```
[log]   [WindowEventBus] Initialized on window object
[warn]  [WxOChat] Remote landing not available in this environment
[error] Failed to load resource: the server responded with a status of 401 ()
```

This is **not** a missing channel — `channels create --type webchat` refuses with *"Webchat
channels cannot be created using the create command. Webchat is automatically available for
all agents."* It is authentication: the embedded chat expects an authenticated session
against the tenant. To serve it to anonymous users you configure the web chat's security
settings for your agent and origin **in the watsonx Orchestrate UI** — the ADK does not
expose that.

Budget for this. "Paste the snippet and it works" is true only inside an authenticated
context, and finding that out during a customer demo is expensive.

### Cross-check in the UI

```text
Walk me through what I should now be able to see for dr_house in the watsonx Orchestrate UI
that we cannot see from the CLI - the chat preview, the agent's reasoning trace, and the
channel settings - and tell me which of the numbers in our dashboard I can corroborate there.
```

The UI is the right place to **watch** an agent think and to configure channel security. The
CLI and the trace APIs are the right place to **measure** it. An FDE should be fluent in
both and clear about which answers which question.

---

## Step 8 - Tear down

```text
Run delete-all.sh to remove everything this lab created - all seven agents, both agent
skills, all six tools - then confirm with an agents list and a tools list that nothing of
ours remains. Leave anything on this instance that we did not create untouched.
```

Check that list carefully before you walk away, especially on a shared instance.

---

## What you actually learned

**Building** a multi-agent team on wxO is the easy part — an afternoon of prompting.

**Routing** is won and lost in the collaborator `description` field and in whether the
orchestrator's instructions permit it to answer. One sentence moved accuracy from 5/9 to
8/9.

**Evaluation** comes in three flavours and you need all three: a smoke gate, a repeatable
suite with accepted-answer sets, and an in-graph observer with a deterministic rubric. The
platform's own judge is useful for triage and will produce confident false positives on
tool-using agents.

**Observability** has a hard boundary. Latency, tokens, model, session, span tree and
sampled quality scores are **native**. Cost, error rate, tail latency, tool-call success and
every dashboard are **yours to build** — and knowing which side of that line a number falls
on is the difference between a scoping conversation that goes well and one that does not.

---

### Take it further

- Add `dr_kutner` (experimental treatments) and `dr_thirteen` (rare and genetic disease) as
  a second tier, and measure what a 9-agent graph does to routing accuracy and token cost.
- Give one specialist a **knowledge base** — an ingested formulary — and contrast retrieval
  grounding against the tool-call grounding used here.
- Nest the graph: let `dr_foreman` delegate onward to a sub-specialist and watch the whole
  chain flatten into one `step_history`.
- Wire the threshold check from Step 5 into CI so a regression in routing accuracy fails the
  build.
