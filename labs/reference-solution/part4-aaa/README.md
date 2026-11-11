# Part 4 reference solution — AAA

The working artifacts for [Part 4](../../WORKSHOP-part4-IDE.md). **A checkpoint, not a
shortcut** — the learning is in prompting Bob to produce these and debugging what comes back.

```
part4-aaa/
├── tools/whoami.py                  Step 1 — the identity probe
├── agents/aaa_probe.yaml            Step 1/2 — probe agent (allowlist commented out)
├── agents/dr_house_aaa.patch.yaml   Step 2/3 — blocks to MERGE into your dr_house YAML
├── plugins/phi_gate.py              Step 3 — Option C, all-or-nothing gate
├── plugins/phi_tool_filter.py       Step 4 — ⛔ tool filtering: the NEGATIVE result
├── controls/phi_pii_control.yaml    Step 5 — declarative PII policy at the same hook
├── app/                             Step 7 — App ID + wxO demo, one container (Podman)
└── import-all.sh · delete-all.sh
```

## The four things that will cost you an hour if you meet them by accident

All live-verified on ADK 2.15.0:

1. **`plugins:` takes `PluginRef` objects.** `- plugin_name: phi_gate`, never `- phi_gate`.
2. **`PluginViolation` is inert.** It blocks the turn, but its contents reach neither the
   run payload nor the trace, and the user sees their own message echoed back. **Rewrite
   `payload.messages[-1].content.text`.** Never pass `None` as `modified_payload`.
3. **`context_variables` is an exclusive allowlist** — it filters the platform's own
   `wxo_*` keys too. List them explicitly or lose your audit join.
4. **Tool-list filtering does not enforce.** Rewriting `payload.tools` is ignored — the
   removed tool was still called 3/3 times by an unauthorised caller. `system_prompt`
   rewrites *are* honoured. Put per-tool checks inside the tool.
5. **A blocked turn produces no trace.** `traces export` returns "No trace data found".
   Every refusal is invisible unless your plugin or app logs it — which is why both
   plugins here print a decision line.

## Verification status

| Verified live on 2.15.0 | Not verified |
|---|---|
| The probe and the four-key REST context | The App ID browser login end to end (`app/`) |
| The allowlist behaviour, all three rows | Plugin vs control ordering on the same hook |
| The gate blocking (`tools_called: []`, model never runs) | Whether `model` / `parameters` in `modified_payload` are honoured |
| Tool filtering **failing** — 3/3 leaks on a plain request | |
| `PluginViolation` being inert; no trace on a blocked turn | |
| The PII control redacting before the model sees the payload | |

Where something is unverified the workshop says so in place. If you settle one of the open
questions, that is a contribution back to the `watsonx-orchestrate-aaa` skill.
