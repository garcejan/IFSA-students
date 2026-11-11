"""Turn captured runs + traces into the metrics the control plane does not give you.

Reads   test/results/raw/*_run.json, *_trace.json  and test/questions.json
Writes  test/results/metrics.json

What is NATIVE (read straight out of the trace):
    trace.latency                        end-to-end server-side seconds
    observation.latency                  per-span seconds
    observation.usage.{input,output}     per-generation token counts
    observation.model                    which model produced them
    trace.scores[]                       control-plane LLM-as-judge scores
    trace.sessionId / trace.userId       session and caller identity

What is DERIVED here (nothing computes these for you):
    error rate                           runs that did not complete
    tool-call success rate               tool responses that carried an error
    p50 / p90 / p95 / p99 tail latency   percentiles over the session
    session token + cost roll-up         tokens summed per session, priced by rate_card.json
    routing accuracy                     delegation vs the expected owner
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "test", "results", "raw")
OUT = os.path.join(ROOT, "test", "results", "metrics.json")

DELEG = "chat_with_collaborator_"


# ------------------------------------------------------------------ helpers
def percentile(values: List[float], p: float) -> Optional[float]:
    """Nearest-rank percentile. Small samples: say so rather than interpolating noise."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * len(s) + 0.5)) - 1))
    return round(s[k], 3)


def load(pattern: str) -> Dict[str, Any]:
    out = {}
    if not os.path.isdir(RAW):
        return out
    for fn in os.listdir(RAW):
        m = re.match(rf"^(.+)_{pattern}\.json$", fn)
        if m:
            out[m.group(1)] = json.load(open(os.path.join(RAW, fn)))
    return out


def steps_of(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    return ((run.get("result") or {}).get("data", {}).get("message", {}).get("step_history") or [])


def final_text(run: Dict[str, Any]) -> str:
    msg = (run.get("result") or {}).get("data", {}).get("message", {})
    c = msg.get("content", "")
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if x.get("response_type") != "thinking").strip()
    return str(c)


def matches(actual: str, expected: Optional[str]) -> bool:
    """wxO appends the collaborator's specialty: dr_wilson -> dr_wilson_oncology."""
    if expected is None:
        return False
    return actual == expected or actual.startswith(expected + "_")


# ------------------------------------------------------------------ analysis
def analyse_run(qid: str, run: Dict[str, Any], trace: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    calls: List[str] = []
    responses: List[Dict[str, Any]] = []
    for step in steps_of(run):
        for d in step.get("step_details", []) or []:
            if d.get("type") == "tool_calls":
                for tc in d.get("tool_calls", []) or []:
                    calls.append(tc.get("name") or tc.get("function", {}).get("name") or "?")
            elif d.get("type") == "tool_response":
                responses.append(d)

    tool_calls = [c for c in calls if not c.startswith(DELEG)]
    delegations = [c[len(DELEG):] for c in calls if c.startswith(DELEG)]

    failures = []
    for r in responses:
        body = json.dumps(r.get("content", "")).lower()
        if any(k in body for k in ('"error"', "traceback", "not in the ppth formulary")):
            failures.append(r.get("name", "?"))

    # --- routing verdict, allowing an accepted SET for genuinely ambiguous cases
    expected = spec.get("expect_delegation")
    accepted = expected if isinstance(expected, list) else [expected]
    if any(a is None for a in accepted) and not delegations:
        routed_ok, routing_note = True, "self-answered (accepted)"
    elif any(matches(d, a) for d in delegations for a in accepted if a):
        routed_ok, routing_note = True, "delegated as expected"
    elif delegations:
        routed_ok, routing_note = False, f"delegated to {delegations} instead"
    else:
        routed_ok, routing_note = False, "did not delegate at all"

    # --- native telemetry out of the trace
    obs = trace.get("observations", []) if isinstance(trace, dict) else []
    gens = [o for o in obs if o.get("type") == "GENERATION"]
    tok_in = sum((o.get("usage") or {}).get("input") or 0 for o in gens)
    tok_out = sum((o.get("usage") or {}).get("output") or 0 for o in gens)
    models = sorted({o.get("model") for o in gens if o.get("model")})

    spans = [
        {"name": o.get("name"), "type": o.get("type"), "latency": o.get("latency") or 0}
        for o in obs
        if (o.get("latency") or 0) > 0
    ]
    spans.sort(key=lambda s: -s["latency"])

    llm_seconds = round(sum(o.get("latency") or 0 for o in gens), 3)
    tool_span_names = set(tool_calls)
    tool_seconds = round(
        sum(o.get("latency") or 0 for o in obs
            if o.get("type") == "SPAN" and o.get("name") in tool_span_names),
        3,
    )

    return {
        "id": qid,
        "prompt": spec.get("prompt", ""),
        "intent": spec.get("intent", ""),
        "status": run.get("status"),
        "completed": run.get("status") == "completed",
        "thread_id": run.get("thread_id"),
        "trace_id": run.get("trace_id"),
        "session_id": trace.get("sessionId") if isinstance(trace, dict) else None,
        "wall_seconds": run.get("_wall_seconds"),
        "trace_latency": trace.get("latency") if isinstance(trace, dict) else None,
        "llm_seconds": llm_seconds,
        "tool_seconds": tool_seconds,
        "tool_calls": tool_calls,
        "delegations": delegations,
        "expected_delegation": expected,
        "routed_as_expected": routed_ok,
        "routing_note": routing_note,
        "tool_failures": failures,
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "tokens_total": tok_in + tok_out,
        "models": models,
        "native_total_cost": trace.get("totalCost") if isinstance(trace, dict) else None,
        "judge_scores": [
            {"name": s.get("name"), "value": s.get("value"), "comment": s.get("comment"), "source": s.get("source")}
            for s in (trace.get("scores") or [])
        ] if isinstance(trace, dict) else [],
        "top_spans": spans[:12],
        "answer_excerpt": final_text(run)[:600],
        "observation_count": len(obs),
    }


def main() -> None:
    runs = load("run")
    traces = load("trace")
    specs = {q["id"]: q for q in json.load(open(os.path.join(ROOT, "test", "questions.json")))}
    specs.setdefault("Q1b", {
        "prompt": "And what is the single test you would order first?",
        "intent": "Multi-turn context retention on Q1's thread.",
        "expect_delegation": [None, "dr_cameron", "dr_wilson", "dr_foreman", "dr_chase"],
    })

    rate = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate_card.json")))
    prices = rate["per_million_tokens"]

    results = [
        analyse_run(qid, runs[qid], traces.get(qid, {}), specs.get(qid, {}))
        for qid in sorted(runs, key=lambda k: (len(k), k))
    ]

    # ---- costing (derived: tokens are native, the money is yours) -------------
    for r in results:
        model = r["models"][0] if r["models"] else "_default"
        p = prices.get(model, prices["_default"])
        r["cost_usd"] = round(
            r["tokens_in"] / 1e6 * p["input"] + r["tokens_out"] / 1e6 * p["output"], 6
        )

    walls = [r["wall_seconds"] for r in results if r["wall_seconds"]]
    traces_lat = [r["trace_latency"] for r in results if r["trace_latency"]]

    total_tool_calls = sum(len(r["tool_calls"]) for r in results)
    total_tool_fail = sum(len(r["tool_failures"]) for r in results)
    routed = [r for r in results if r["expected_delegation"] is not None or r["delegations"]]

    per_tool: Dict[str, Dict[str, int]] = {}
    for r in results:
        for t in r["tool_calls"]:
            per_tool.setdefault(t, {"calls": 0, "failures": 0})
            per_tool[t]["calls"] += 1
        for t in r["tool_failures"]:
            per_tool.setdefault(t, {"calls": 0, "failures": 0})
            per_tool[t]["failures"] += 1
    for t, v in per_tool.items():
        v["success_rate"] = round(100.0 * (v["calls"] - v["failures"]) / v["calls"], 1) if v["calls"] else 0.0

    summary = {
        "runs": len(results),
        "completed": sum(1 for r in results if r["completed"]),
        "error_rate_pct": round(100.0 * sum(1 for r in results if not r["completed"]) / len(results), 1) if results else 0.0,
        "routing_accuracy_pct": round(100.0 * sum(1 for r in results if r["routed_as_expected"]) / len(results), 1) if results else 0.0,
        "tool_calls": total_tool_calls,
        "tool_failures": total_tool_fail,
        "tool_success_pct": round(100.0 * (total_tool_calls - total_tool_fail) / total_tool_calls, 1) if total_tool_calls else 0.0,
        "latency": {
            "p50": percentile(walls, 50), "p90": percentile(walls, 90),
            "p95": percentile(walls, 95), "p99": percentile(walls, 99),
            "max": round(max(walls), 3) if walls else None,
            "mean": round(sum(walls) / len(walls), 3) if walls else None,
            "sample_size": len(walls),
        },
        "trace_latency": {
            "p50": percentile(traces_lat, 50), "p95": percentile(traces_lat, 95),
            "max": round(max(traces_lat), 3) if traces_lat else None,
        },
        "tokens_in": sum(r["tokens_in"] for r in results),
        "tokens_out": sum(r["tokens_out"] for r in results),
        "tokens_total": sum(r["tokens_total"] for r in results),
        "cost_usd": round(sum(r["cost_usd"] for r in results), 6),
        "native_cost_reported": sum(r["native_total_cost"] or 0 for r in results),
        "sessions": sorted({r["session_id"] for r in results if r["session_id"]}),
        "judge_scored_runs": sum(1 for r in results if r["judge_scores"]),
        "per_tool": per_tool,
        "rate_card": prices,
        "delegation_count": sum(len(r["delegations"]) for r in results),
        "routed_questions": len(routed),
    }

    json.dump({"summary": summary, "results": results}, open(OUT, "w"), indent=2)

    print(f"runs={summary['runs']}  completed={summary['completed']}  "
          f"error_rate={summary['error_rate_pct']}%")
    print(f"routing_accuracy={summary['routing_accuracy_pct']}%  "
          f"delegations={summary['delegation_count']}")
    print(f"tool_success={summary['tool_success_pct']}%  "
          f"({summary['tool_calls'] - summary['tool_failures']}/{summary['tool_calls']})")
    print(f"latency p50={summary['latency']['p50']}s  p95={summary['latency']['p95']}s  "
          f"max={summary['latency']['max']}s  (n={summary['latency']['sample_size']})")
    print(f"tokens={summary['tokens_total']}  derived_cost=${summary['cost_usd']}  "
          f"native_cost_reported={summary['native_cost_reported']}")
    print(f"judge-scored runs={summary['judge_scored_runs']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
