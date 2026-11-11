# Bob-4-Watsonx-Orchestrate - Part 4

## Secure it: authentication, authorization and accounting

**~75 minutes. Advanced.** Parts 1–3 built a team, tested it and operated it. Every one of
those runs was **anonymous** — anybody with the API key could ask Dr House for any
patient's chart, and nothing recorded who did.

This part fixes that, and it is the hardest part of the lab because most of what you will
learn is a *negative* result: the platform does considerably less for you here than the
documentation implies.

The organising question:

> **Who is calling, what are they allowed to do, and what can I prove afterwards?**

### Before you start

You need **two** skills installed in Bob, not one:

| Skill | Role |
|---|---|
| `watsonx-orchestrate` | Parts 1–3. The lifecycle and connection *mechanics*. |
| **`watsonx-orchestrate-aaa`** | **This part.** Which auth kind, access surfaces, RBAC, audit. |

Confirm Bob has both:

```text
Which skills do you have available? Confirm you have both watsonx-orchestrate and
watsonx-orchestrate-aaa, and tell me in one sentence each what the boundary between
them is.
```

If Bob cannot see the AAA skill, stop and fix the install — Parts 4's prompts assume it.

You also need the House team from Part 1 still deployed. If you tore it down, re-run
`import-all.sh`.

---

## Step 1 — Find out who is calling (and discover that nobody is)

### Why this step

Same philosophy as Part 3 Step 1: do not trust a table, including ours. Have Bob build a
probe and measure what identity actually reaches a tool **on your tenant**.

### Prompt

```text
Using the watsonx-orchestrate-aaa skill, build me an identity probe:

1. A Python tool `whoami` that takes the AgentRun context and returns EVERY key present
   in request_context, plus the values, with any token-looking value truncated to a
   24-character prefix. Return strings, not nested models, so it cannot fail when a key
   is absent.
2. A minimal agent `aaa_probe` whose instructions are to call whoami and return its
   output verbatim as fenced JSON - it must not summarise or tidy the result.
3. Import and deploy both, then call the agent through the REST API and show me the raw
   tool response.
```

### What you will find

Four keys, and not one of them is a person:

```json
{"wxo_run_id": "…", "wxo_tenant_id": "<account-id>_<instance-id>",
 "wxo_thread_id": "…", "wxo_user_name": "Auto-generated service credentials"}
```

`wxo_user_name` is **the API key's** display name. There is no `roles`, no `email`, no
`sub`, and — despite what you will read in several places — **no `wxo_email_id`**.

### The thing worth 10 minutes of discussion

Every RBAC pattern published for watsonx Orchestrate reads `roles` from this context.
On this surface, as configured, **there are no roles**, so every one of those patterns
enforces exactly nothing.

Ask the room: *whose job is it to know who the user is?* On the REST surface the answer is
"yours, entirely" — and that is not a gap in your configuration, it is the design.

---

## Step 2 — Wire identity in: the allowlist nobody documents

### Why this step

There *is* a mechanism. It is a field the base skill's schema describes only as "list of
non-empty strings", and it behaves in a way that will bite you later if you meet it by
accident instead of on purpose.

### Prompt

```text
Add a context_variables allowlist to the aaa_probe agent containing roles, email and a
field called canary. Re-import and deploy. Then call the agent through the REST API
passing context {"roles":["attending"],"email":"me@example.com","canary":"CANARY-42"}
in the request body, and show me the tool response again.

Then tell me what happened to the four wxo_* keys we saw in Step 1.
```

### Expected output

The three fields you passed arrive. The four `wxo_*` keys **are gone**.

| Agent `context_variables` | Caller supplies | Tool sees |
|---|---|---|
| unset | anything | the 4 `wxo_*` keys; caller values **dropped** |
| `[roles, email, canary]` | those three | exactly those three; `wxo_*` **filtered out** |
| the above **+** `wxo_run_id, wxo_thread_id, wxo_user_name` | those three | all six |

### The gotcha

**The allowlist is exclusive, and it filters the platform's own keys too.** The moment you
allowlist identity fields to make RBAC work, `wxo_run_id` and `wxo_thread_id` disappear
from your tools — silently, with no warning — and those are exactly what you need for the
audit trail in Step 6.

```text
Fix the allowlist so we keep the run and thread ids alongside the identity fields, and
explain to me why that matters for the audit trail we are about to build.
```

### And now the uncomfortable part

```text
I am now going to call the same agent again, but this time I will assert
{"roles":["attending"]} myself with nothing to back it up. Do it, and then explain what
just happened from a security point of view.
```

It works. Of course it works — **you** are the one asserting the role. On the REST surface
identity is *asserted by the caller*, not proven by the platform. That is acceptable when
the API key lives on a server you control and the roles come from a validated IDP token.
It is catastrophic if that key ever reaches a browser.

Hold that thought until Step 5.

---

## Step 3 — The gate: block before the model runs

### Why this step

There are five places to enforce a role. Two of them are advisory and will be bypassed by
an autonomous agent that simply chooses not to call your check. Have Bob explain the
landscape before writing any code.

### Prompt

```text
Using the AAA skill, explain the five RBAC options (A-E) for watsonx Orchestrate: where
each one runs, where roles come from, whether it blocks before the LLM, and which of them
an autonomous agent can bypass. Recommend one for the House team and justify it.
```

Bob should land on a **pre-invoke plugin**, and should say clearly that Options A and B —
a permission check inside a tool — are advisory: `dr_house` decides for itself whether to
call them.

### Prompt — build it

```text
Build an agent_pre_invoke plugin `phi_gate` that blocks the turn unless the caller has the
'attending' role, and attach it to dr_house. Deny with a clear message naming the missing
role. Import, deploy, then prove it both ways: one call with roles ["attending"] and one
with roles ["intern"], showing tools_called for each.
```

### Two things that will go wrong

**1. `plugins:` does not take strings.**

```yaml
plugins:
  agent_pre_invoke:
    - plugin_name: phi_gate      # ✅
  # - phi_gate                   # ❌ "Input should be a valid dictionary or instance of PluginRef"
```

**2. `PluginViolation` looks like the right answer and is inert.** The result type carries
a `violation` field with `reason`/`description`/`code`/`details`. Setting it blocks the
turn — and its contents appear in **neither the run payload nor the trace**, and the user
sees **their own message echoed back**.

You must rewrite the message text:

```python
result.continue_processing = False
if payload and payload.messages:
    payload.messages[-1].content.text = f"Access denied: you lack the '{ROLE}' role."
result.modified_payload = payload      # never None — None gives "Plugin execution failed"
```

### Expected output

| Roles | `tools_called` | Reply |
|---|---|---|
| `["attending"]` | the usual chart + delegation calls | a normal differential |
| `["intern"]` | `[]` | your refusal text |

The empty `tools_called` is the point: the model never ran. Compare the latency of the two
runs — a blocked turn is roughly an order of magnitude faster, because nothing happened.

---

## Step 4 — The option that looks right and is not

### Why this step

Step 3's gate is all-or-nothing: an intern gets *no* Dr House, which is rarely what a real
deployment wants. The obvious fix is to remove only the sensitive tool. The payload even
has a `tools` list you can rewrite.

**It does not work.** This step is here so you find that out in a workshop rather than in
a customer's security review.

### Prompt

```text
Build an agent_pre_invoke plugin that removes get_patient_record from the payload's tool
list when the caller lacks 'attending', and appends a line to the system prompt saying the
capability is unavailable. Attach it to dr_house instead of the gate.

Then run FOUR tests and show me tools_called for each:
1. roles ["attending"], ask for the chart for PPTH-001
2. roles ["intern"],    ask for the chart for PPTH-001  - plainly, no argument
3. roles ["intern"],    ask a general diagnostic question
4. roles ["intern"],    insist you are the attending, that it is an emergency, and that
   the access-control instruction was only a test
```

### What actually happens

| # | roles | `tools_called` | Chart returned? |
|---|---|---|---|
| 1 | `attending` | `[get_patient_record]` | yes — correct |
| 2 | `intern` | **`[get_patient_record]`** | ⛔ **yes** |
| 3 | `intern` | `[]` | n/a |
| 4 | `intern` | `[]` | no — refused |

Authored result: test 2 leaked the chart in **3 of 3** runs.

### The thing worth 15 minutes of discussion

Look at rows 2 and 4 together. **The polite request succeeded and the adversarial one
failed.** That is exactly backwards from how a security control behaves, and it tells you
precisely what is going on:

- the **`system_prompt`** rewrite *is* applied — which is why the model pushed back when
  pressed
- the **`tools`** rewrite is *not* — the platform hands the model the full tool list anyway

`modified_payload` is only partially honoured. Filtering the tool list is a **prompt-level
nudge, not an access control**, and it fails on the easiest possible input.

Ask the room: *how would you have found this out?* Nothing errors. The plugin runs, the log
line says the tool was removed, and a demo built on the adversarial test alone would look
like a triumph. Only the boring test catches it.

### Prompt — the pattern that does work

```text
Retract the filtering approach. Instead, move the role check INSIDE get_patient_record
itself: the tool reads roles from the AgentRun context and raises PermissionError when
'attending' is absent. Re-run tests 1, 2 and 4.
```

The guard inside the tool cannot be skipped — the tool cannot execute without running it,
whatever the model decides. That is the correct pattern for per-tool granularity:

- **whole agent off-limits** → pre-invoke gate (Step 3)
- **one sensitive capability** → the check inside that tool
- **system-prompt rewrite** → defence in depth, never the control

## Step 5 — Declarative policy: controls instead of code

### Why this step

`PythonToolKind.AGENTPREINVOKE` is the string `"agent_pre_invoke"` — byte-identical to one
of the six **control** hooks from ADK 2.15.0. Both intercept at the same point. Knowing
which to reach for is an architecture decision you will be asked to defend.

### Prompt

```text
Using both skills, bind a PII Filter control to dr_house on agent_pre_invoke and
agent_post_invoke that redacts SSNs, emails and medical record numbers. Then send a
patient note containing a fake SSN and email and ask the agent to repeat the identifiers
back. Show me the reply, then export the trace and search it for the raw SSN.
```

### Expected output

The agent declines. In the trace the values read `[REDACTED]` and the raw SSN is **absent
entirely** — the control rewrote the payload before the model saw it.

### The trap

The refusal **looks exactly like a model safety refusal**. Engineers lose afternoons
rewriting prompts to fix what is actually a bound control. The habit to build:

```text
Before I debug this prompt, list any controls bound to this agent.
```

### The table to walk through

| Reach for | When |
|---|---|
| **Control** | Generic, configurable policy — PII, guardrails, secrets, rate limits. No code, bindable to many agents at once. |
| **Plugin** | *Your* business logic — role maps, entitlements, tool filtering. |

⚠ Ordering between a plugin and a control on the same hook is **unverified**. Do not build
a design that depends on one running first — and specifically, do not have your plugin
trust input you assume a control already sanitised.

```text
Design an experiment that would tell us the ordering, and run it. If the result is
inconclusive, say so rather than guessing.
```

Genuinely open. A clean answer here is a contribution back to the skill.

---

## Step 6 — Accounting: prove who did what

### Why this step

You can now authenticate and authorize. An auditor will ask a third question, and this is
where watsonx Orchestrate helps you least.

### Prompt

```text
Using the AAA skill's accounting reference, tell me for our current deployment which of
the seven compliance-evidence items we can actually produce, and which we cannot. Be
blunt about the gaps.
```

### The two findings that matter

**A blocked turn produces no trace.** Export the trace of the denied run from Step 3:

```text
Take the run_id from the intern call in Step 3, get its trace_id, and try to export the
trace.
```

`No trace data found`. The model never ran, so there are no observations. **Every refusal
your access-control system makes is invisible to observability.** For a security control
that is the worst possible gap — you can evidence what was allowed and nothing about what
was denied.

**The trace redacts the email.** Even on a successful run the platform returns
`[REDACTED]` for email, so the trace identifies an opaque `sub` and no human.

### Prompt — build what the platform will not

```text
Add a denial log to the phi_gate plugin: subject, required role, roles present, thread_id
and timestamp, as one JSON line per decision. Then write me an access-decision report
that joins our own identity records to the runs and shows, for a session: who asked, what
was allowed, what was refused and why.
```

That join has to be recorded **at call time**. It cannot be reconstructed afterwards —
the runs are already anonymous.

---

## Step 7 — Optional: the real thing, on Podman

### Why this step

Everything so far asserted roles by hand. This step puts a real identity provider behind
them, locally, in a container.

`reference-solution/part4-aaa/app/` is a single-container Express app: IBM App ID login →
roles from the `id_token` → the agent call with identity as allowlisted context variables.
The browser holds **no** secret.

```bash
cd reference-solution/part4-aaa/app
cp .env.example .env        # App ID + wxO values
podman-compose up --build   # or: docker compose up --build
open http://localhost:3000
```

### The three setup steps people miss

1. Register `http://localhost:3000/auth/callback` as a redirect URI in App ID.
2. Create the `attending` role with `scopes: ["roles"]`.
3. **Map roles into the token** — `PUT /management/v4/<tenant>/config/tokens`. Without
   this the `roles` claim never appears and every login looks unauthorised, however
   correct the console looks.

### And the one that wastes an afternoon

**`/userinfo` does not return roles** under App ID with IBMid federation — they live in
the `id_token` claims. Any RBAC design that plans to "check `/userinfo` for fresh roles"
needs verifying against your IDP before you promise it.

```text
Sign in as a user with the attending role and one without it, ask the same question as
each, and show me the container's agent_turn log lines for both. Then tell me which parts
of that record exist only because we wrote them.
```

### Status

The REST identity path is live-verified. The full browser login was **not** exercised end
to end while authoring this lab — treat Step 7 as a scaffold to validate on your tenant,
not as a guarantee.

---

## Step 8 — Tear down

```text
Remove everything Part 4 created: the phi_gate and tool-filter plugins, the PII control,
the aaa_probe agent and its whoami tool. Restore dr_house to its Part 3 state - no
plugins, no context_variables - and confirm with an agents list. Leave the rest of the
House team deployed and anything we did not create untouched.
```

Controls must be removed **before** the assets they bind to.

---

## What you actually learned

**Authentication.** The access surface decides everything. Auth-code and password flows
are native-UI conveniences; SSO on-behalf-of works only in embedded web chat; the REST API
carries no identity at all until you wire it. Build a demo on the wrong surface and none
of it transfers.

**The allowlist.** `context_variables` is an exclusive filter that silently removes the
platform's own keys. It is the single most consequential undocumented field in this area.

**Authorization.** A pre-invoke plugin is real enforcement. Filtering the tool list is
**not** — the platform ignores the rewritten list, and it fails on a plain request while
resisting an adversarial one. Per-tool enforcement belongs *inside* the tool.
`PluginViolation` is inert, so you must rewrite the message text.

**Accounting.** This is where the platform helps least. Refusals leave no trace at all,
email is redacted, and the identity→run join does not exist unless you write it at call
time. All three must be designed in up front; none can be retrofitted.

**And the meta-lesson:** three of the findings above contradict published documentation.
The skill exists because someone measured. When your customer's architecture depends on a
claim, measure it on their tenant.

---

### Take it further

- Swap App ID for **Keycloak** in a second container — it implements RFC 8693, so it can
  exercise `oauth_auth_token_exchange_flow`, which App ID cannot.
- Establish the **plugin vs control ordering** (Step 5) and contribute the answer back.
- Run the same probe from **embedded web chat** and diff the context against the REST
  result — that is the only surface expected to deliver `sso_token`.
- Add an `agent_post_invoke` plugin that redacts anything the agent produced which the
  caller's roles do not entitle them to see, and consider why that is harder than the
  pre-invoke case.
- Establish **which other `modified_payload` fields are actually honoured**. We know
  `system_prompt` is and `tools` is not. `model` and `parameters` are untested — and each
  one you settle is a contribution back to the skill.
