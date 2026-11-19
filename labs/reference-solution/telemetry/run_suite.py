"""Run the lab question set against dr_house and capture everything.

For each question this records:
  - the run object (status, thread_id, trace_id, step_history)
  - client-observed wall latency
  - which tools were called and which collaborator was delegated to
  - the AgentOps v3 trace (per-observation latency, token usage, model, judge scores)

Everything lands in test/results/raw/ as JSON so Part 4 can compute metrics offline
without re-running the agents.

Usage:
    python telemetry/run_suite.py                 # full suite + multi-turn follow-up
    python telemetry/run_suite.py Q2 Q5           # only those questions
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wxo_client import (  # noqa: E402
    WxOClient,
    delegations,
    final_text,
    routed_to,
    tool_calls,
    tool_failures,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "test", "results", "raw")

# The trace is written asynchronously; give the collector a moment before fetching.
TRACE_SETTLE_SECONDS = 12


def main() -> None:
    wanted = set(sys.argv[1:])
    questions = json.load(open(os.path.join(ROOT, "test", "questions.json")))
    if wanted:
        questions = [q for q in questions if q["id"] in wanted]

    os.makedirs(RAW, exist_ok=True)
    client = WxOClient()
    print(f"env={client.env}\n")

    session = []
    for q in questions:
        print(f"=== {q['id']}  {q['prompt'][:70]}")
        run = client.run(q["agent"], q["prompt"])

        called = tool_calls(run)
        deleg = delegations(run)
        fails = tool_failures(run)
        routed_ok = (
            routed_to(run, q["expect_delegation"])
            if q["expect_delegation"]
            else (len(deleg) == 0)
        )

        print(f"    status={run.get('status')}  wall={run['_wall_seconds']}s")
        print(f"    tools={called}")
        print(f"    delegated={deleg}  expected={q['expect_delegation']}  -> {'PASS' if routed_ok else 'MISROUTE'}")
        if fails:
            print(f"    tool failures={[f['tool'] for f in fails]}")

        session.append(
            {
                "id": q["id"],
                "intent": q["intent"],
                "prompt": q["prompt"],
                "status": run.get("status"),
                "wall_seconds": run["_wall_seconds"],
                "thread_id": run.get("thread_id"),
                "trace_id": run.get("trace_id"),
                "tool_calls": called,
                "delegations": deleg,
                "expected_delegation": q["expect_delegation"],
                "routed_as_expected": routed_ok,
                "tool_failures": fails,
                "answer": final_text(run),
            }
        )
        json.dump(run, open(os.path.join(RAW, f"{q['id']}_run.json"), "w"), indent=2)

    # ---- multi-turn: resume Q1's thread with a context-dependent follow-up --------
    q1 = next((s for s in session if s["id"] == "Q1"), None)
    if q1 and q1["thread_id"]:
        follow = "And what is the single test you would order first?"
        print(f"\n=== Q1b (multi-turn on thread {q1['thread_id'][:8]})  {follow}")
        run = client.run("dr_house", follow, thread_id=q1["thread_id"])
        print(f"    status={run.get('status')}  wall={run['_wall_seconds']}s")
        session.append(
            {
                "id": "Q1b",
                "intent": "Multi-turn context retention on Q1's thread. The follow-up "
                          "never names the patient - it is only resolvable from turn 1.",
                "prompt": follow,
                "status": run.get("status"),
                "wall_seconds": run["_wall_seconds"],
                "thread_id": run.get("thread_id"),
                "trace_id": run.get("trace_id"),
                "tool_calls": tool_calls(run),
                "delegations": delegations(run),
                "expected_delegation": None,
                "routed_as_expected": True,
                "tool_failures": tool_failures(run),
                "answer": final_text(run),
            }
        )
        json.dump(run, open(os.path.join(RAW, "Q1b_run.json"), "w"), indent=2)

    # ---- pull the AgentOps v3 trace for every run --------------------------------
    print(f"\nwaiting {TRACE_SETTLE_SECONDS}s for traces to settle...")
    time.sleep(TRACE_SETTLE_SECONDS)
    for s in session:
        if not s.get("trace_id"):
            continue
        trace = client.trace(s["trace_id"])
        json.dump(trace, open(os.path.join(RAW, f"{s['id']}_trace.json"), "w"), indent=2)
        err = trace.get("_http_error") if isinstance(trace, dict) else None
        n = len(trace.get("observations", [])) if not err else 0
        print(f"  {s['id']}: trace {s['trace_id'][:12]}... obs={n} err={err}")

    json.dump(session, open(os.path.join(ROOT, "test", "results", "session.json"), "w"), indent=2)

    ok = sum(1 for s in session if s["routed_as_expected"])
    print(f"\nrouting: {ok}/{len(session)} as expected")
    print(f"wrote {os.path.join(ROOT, 'test', 'results', 'session.json')}")


if __name__ == "__main__":
    main()
