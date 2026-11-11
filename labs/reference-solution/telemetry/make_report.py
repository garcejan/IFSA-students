"""Render test/results/metrics.json as a self-contained HTML5 evaluation report.

No external assets, no CDN, no build step - one file you can email, attach to a
handover, or open from disk. Light and dark are both selected, not auto-flipped.

Usage:
    python telemetry/analyze.py && python telemetry/make_report.py
    open test/results/evaluation-report.html
"""

from __future__ import annotations

import csv
import datetime as dt
import glob
import html
import json
import os
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "test", "results", "metrics.json")
OUT = os.path.join(ROOT, "test", "results", "evaluation-report.html")

# Palette roles. Magnitude is single-hue; status colours are reserved and never
# reused as a series. The three-way time split uses categorical slots 1-3, the
# only three that validate all-pairs in both modes.
CSS = """
:root{
  color-scheme:light;
  --plane:#f9f9f7; --surface:#fcfcfb;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
  --bar:#2a78d6; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b;
  --good-ink:#006300;
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])){
    color-scheme:dark;
    --plane:#0d0d0d; --surface:#1a1a19;
    --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
    --bar:#3987e5; --s1:#3987e5; --s2:#d95926; --s3:#199e70;
    --good-ink:#0ca30c;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --plane:#0d0d0d; --surface:#1a1a19;
  --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --bar:#3987e5; --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --good-ink:#0ca30c;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:2.5rem 1.5rem 5rem}
h1{font-size:1.75rem;margin:0 0 .35rem;letter-spacing:-.01em}
h2{font-size:1.15rem;margin:2.75rem 0 .35rem;letter-spacing:-.005em}
h3{font-size:.95rem;margin:1.5rem 0 .5rem;color:var(--ink2)}
p{color:var(--ink2);margin:.35rem 0 1rem}
.sub{color:var(--muted);font-size:.875rem;margin:0 0 2rem}
.note{font-size:.85rem;color:var(--muted);margin:.4rem 0 1.25rem}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;
  padding:1.25rem 1.4rem;margin:0 0 1rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.75rem;margin:0 0 1rem}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:.9rem 1rem}
.tile .k{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.tile .v{font-size:1.7rem;line-height:1.15;margin-top:.3rem;font-variant-numeric:tabular-nums}
.tile .u{font-size:.8rem;color:var(--muted);margin-top:.15rem}
table{width:100%;border-collapse:collapse;font-size:.875rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
th{text-align:left;font-weight:600;color:var(--ink2);border-bottom:1px solid var(--axis);
  padding:.5rem .6rem;white-space:nowrap}
td{border-bottom:1px solid var(--grid);padding:.5rem .6rem;vertical-align:top}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
code,.mono{font:.85em ui-monospace,SFMono-Regular,Menlo,monospace}
.pill{display:inline-flex;align-items:center;gap:.35rem;font-size:.78rem;font-weight:600;
  padding:.12rem .5rem;border-radius:999px;border:1px solid var(--ring);white-space:nowrap}
.pill.ok{color:var(--good-ink)} .pill.bad{color:var(--crit)} .pill.warn{color:var(--serious)}
.dot{width:.5rem;height:.5rem;border-radius:50%;flex:0 0 auto}
.dot.ok{background:var(--good)} .dot.bad{background:var(--crit)} .dot.warn{background:var(--warn)}
.bar-row{display:grid;grid-template-columns:3.2rem 1fr 4.5rem;align-items:center;gap:.6rem;margin:.3rem 0}
.bar-track{height:14px;background:var(--grid);border-radius:4px;overflow:hidden;display:flex}
.bar-fill{height:100%;background:var(--bar);border-radius:0 4px 4px 0}
.seg{height:100%}
.seg+.seg{box-shadow:-2px 0 0 var(--surface)}
.lbl{color:var(--ink2);font-size:.82rem}
.val{text-align:right;font-variant-numeric:tabular-nums;font-size:.82rem;color:var(--ink2)}
.legend{display:flex;flex-wrap:wrap;gap:1rem;margin:.6rem 0 0;font-size:.8rem;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:.4rem}
.sw{width:.65rem;height:.65rem;border-radius:2px;flex:0 0 auto}
.two{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:760px){.two{grid-template-columns:1fr}}
details{border-top:1px solid var(--grid);padding:.6rem 0}
summary{cursor:pointer;font-weight:600;font-size:.9rem}
details pre{white-space:pre-wrap;font-size:.82rem;color:var(--ink2);
  background:var(--plane);padding:.75rem;border-radius:8px;border:1px solid var(--ring)}
.kicker{font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
"""


def esc(x: Any) -> str:
    return html.escape(str(x if x is not None else ""))


def pill(ok: bool, yes: str = "PASS", no: str = "FAIL") -> str:
    cls = "ok" if ok else "bad"
    return f'<span class="pill {cls}"><span class="dot {cls}"></span>{yes if ok else no}</span>'


def bars(rows: List[tuple], unit: str = "s") -> str:
    """rows: (label, value, optional_flag). Single-hue magnitude bars."""
    if not rows:
        return ""
    mx = max(r[1] for r in rows) or 1
    out = []
    for label, value, *rest in rows:
        pct = max(1.5, 100.0 * value / mx)
        out.append(
            f'<div class="bar-row"><div class="lbl">{esc(label)}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="val">{value:g}{unit}</div></div>'
        )
    return "".join(out)


def stacked(rows: List[tuple]) -> str:
    """rows: (label, [(seconds, css_var), ...], total)"""
    mx = max(r[2] for r in rows) or 1
    out = []
    for label, segs, total in rows:
        inner = "".join(
            f'<div class="seg" style="width:{100.0*v/total if total else 0:.1f}%;background:var(--{c})"></div>'
            for v, c in segs if v > 0
        )
        pct = max(1.5, 100.0 * total / mx)
        out.append(
            f'<div class="bar-row"><div class="lbl">{esc(label)}</div>'
            f'<div class="bar-track" style="width:{pct:.1f}%">{inner}</div>'
            f'<div class="val">{total:g}s</div></div>'
        )
    return "".join(out)


def load_agentops() -> Optional[List[Dict[str, str]]]:
    runs = sorted(glob.glob(os.path.join(ROOT, "eval", "out", "**", "summary_metrics.csv"), recursive=True))
    if not runs:
        return None
    with open(runs[-1]) as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    data = json.load(open(METRICS))
    s, results = data["summary"], data["results"]
    lat = s["latency"]
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---------- tiles
    tiles = [
        ("Runs", s["runs"], "questions executed"),
        ("Error rate", f'{s["error_rate_pct"]}%', f'{s["runs"] - s["completed"]} failed runs'),
        ("Routing accuracy", f'{s["routing_accuracy_pct"]}%', f'{s["delegation_count"]} delegations'),
        ("Tool-call success", f'{s["tool_success_pct"]}%', f'{s["tool_calls"] - s["tool_failures"]}/{s["tool_calls"]} calls'),
        ("p95 latency", f'{lat["p95"]}s', f'p50 {lat["p50"]}s · max {lat["max"]}s'),
        ("Tokens", f'{s["tokens_total"]:,}', f'{s["tokens_in"]:,} in · {s["tokens_out"]:,} out'),
        ("Derived cost", f'${s["cost_usd"]:.4f}', f'native reported ${s["native_cost_reported"]}'),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>'
        f'<div class="u">{esc(u)}</div></div>' for k, v, u in tiles
    )

    # ---------- routing table
    routing_rows = "".join(
        f"<tr><td class='mono'>{esc(r['id'])}</td>"
        f"<td>{esc(r['prompt'][:78])}{'…' if len(r['prompt'])>78 else ''}</td>"
        f"<td class='mono'>{esc(fmt_expected(r['expected_delegation']))}</td>"
        f"<td class='mono'>{esc(', '.join(r['delegations']) or '— self-answered')}</td>"
        f"<td>{pill(r['routed_as_expected'])}</td>"
        f"<td class='lbl'>{esc(r['routing_note'])}</td></tr>"
        for r in results
    )

    # ---------- latency
    lat_bars = bars([(r["id"], r["wall_seconds"] or 0) for r in results])
    pct_rows = "".join(
        f"<tr><td>{esc(k)}</td><td class='num'>{esc(v)}{'s' if v is not None else ''}</td></tr>"
        for k, v in [("p50", lat["p50"]), ("p90", lat["p90"]), ("p95", lat["p95"]),
                     ("p99", lat["p99"]), ("max", lat["max"]), ("mean", lat["mean"])]
    )

    # ---------- where the time went
    split_rows = []
    for r in results:
        llm = r["llm_seconds"] or 0
        tool = r["tool_seconds"] or 0
        total = r["trace_latency"] or r["wall_seconds"] or 0
        other = max(0.0, round(total - llm - tool, 3))
        split_rows.append((r["id"], [(llm, "s1"), (tool, "s2"), (other, "s3")], round(total, 3)))
    split_html = stacked(split_rows)

    # ---------- slowest spans across the session
    all_spans: Dict[str, float] = {}
    for r in results:
        for sp in r["top_spans"]:
            all_spans[sp["name"]] = max(all_spans.get(sp["name"], 0), sp["latency"])
    slowest = sorted(all_spans.items(), key=lambda kv: -kv[1])[:10]
    slow_bars = bars([(n[:34], round(v, 2)) for n, v in slowest])

    # ---------- per-tool success
    tool_rows = "".join(
        f"<tr><td class='mono'>{esc(t)}</td><td class='num'>{v['calls']}</td>"
        f"<td class='num'>{v['failures']}</td><td class='num'>{v['success_rate']}%</td>"
        f"<td>{pill(v['failures']==0,'100%','degraded')}</td></tr>"
        for t, v in sorted(s["per_tool"].items(), key=lambda kv: kv[0])
    )

    # ---------- tokens & cost
    cost_rows = "".join(
        f"<tr><td class='mono'>{esc(r['id'])}</td><td class='num'>{r['tokens_in']:,}</td>"
        f"<td class='num'>{r['tokens_out']:,}</td><td class='num'>{r['tokens_total']:,}</td>"
        f"<td class='num'>${r['cost_usd']:.5f}</td>"
        f"<td class='num'>{esc(r['native_total_cost'])}</td></tr>"
        for r in results
    )

    # ---------- agentops
    ao = load_agentops()
    if ao:
        keys = ["dataset_name", "orchestrate_agent_routing_accuracy", "total_steps", "llm_steps",
                "average_agent_response_time", "total_tool_calls", "tool_match_success",
                "keyword_match", "semantic_match", "is_success"]
        ao_head = "".join(f"<th>{esc(k.replace('_',' '))}</th>" for k in keys)
        ao_rows = "".join(
            "<tr>" + "".join(f"<td class='{'num' if k!='dataset_name' else 'mono'}'>{esc(row.get(k,''))}</td>" for k in keys) + "</tr>"
            for row in ao
        )
        ao_html = (
            f'<div class="card scroll"><table><thead><tr>{ao_head}</tr></thead>'
            f'<tbody>{ao_rows}</tbody></table></div>'
        )
    else:
        ao_html = '<p class="note">No AgentOps run found under <code>eval/out/</code>. Run <code>orchestrate evaluations validate-native</code> first.</p>'

    # ---------- judge scores
    judged = [(r["id"], sc) for r in results for sc in r["judge_scores"]]
    if judged:
        judge_rows = "".join(
            f"<tr><td class='mono'>{esc(qid)}</td><td class='mono'>{esc(sc['name'])}</td>"
            f"<td class='num'>{esc(sc['value'])}</td><td class='mono'>{esc(sc['source'])}</td>"
            f"<td class='lbl'>{esc((sc.get('comment') or '')[:200])}</td></tr>"
            for qid, sc in judged
        )
        judge_html = (
            '<div class="card scroll"><table><thead><tr><th>Run</th><th>Metric</th>'
            '<th>Value</th><th>Source</th><th>Rationale</th></tr></thead>'
            f'<tbody>{judge_rows}</tbody></table></div>'
        )
    else:
        judge_html = ('<p class="note">No judge scores attached to these traces yet. The control plane '
                      'attaches them <strong>asynchronously and by sampling</strong> — re-fetch the traces a '
                      'few minutes after the run, and do not assume every trace gets scored.</p>')

    # ---------- per-question detail
    details = "".join(
        f"<details><summary>{esc(r['id'])} — {esc(r['prompt'][:80])}</summary>"
        f"<p class='note'>{esc(r['intent'])}</p>"
        f"<p class='note'><strong>trace</strong> <code>{esc(r['trace_id'])}</code> · "
        f"<strong>thread</strong> <code>{esc(r['thread_id'])}</code> · "
        f"<strong>session</strong> <code>{esc(r['session_id'])}</code> · "
        f"{r['observation_count']} observations</p>"
        f"<p class='note'><strong>tools</strong> <code>{esc(', '.join(r['tool_calls']) or 'none')}</code>"
        + (f" · <strong>failed</strong> <code>{esc(', '.join(r['tool_failures']))}</code>" if r['tool_failures'] else "")
        + f"</p><pre>{esc(r['answer_excerpt'])}</pre></details>"
        for r in results
    )

    native_table = """
<div class="card scroll"><table>
<thead><tr><th>Signal</th><th>Where it lives</th><th>Native?</th><th>What you do about it</th></tr></thead>
<tbody>
<tr><td>End-to-end latency</td><td class="mono">trace.latency</td><td>{ok}</td><td>Read it. Also present per-observation.</td></tr>
<tr><td>Per-step latency</td><td class="mono">observation.latency</td><td>{ok}</td><td>The only way to find which span is slow.</td></tr>
<tr><td>Token counts</td><td class="mono">observation.usage.input / .output</td><td>{ok}</td><td>Exact, per LLM call. Sum them yourself.</td></tr>
<tr><td>Model identity</td><td class="mono">observation.model</td><td>{ok}</td><td>Needed to price the tokens.</td></tr>
<tr><td>Session / caller</td><td class="mono">trace.sessionId, trace.userId</td><td>{ok}</td><td>The grouping key for session-level roll-ups.</td></tr>
<tr><td>Quality scores</td><td class="mono">trace.scores[]</td><td>{ok}</td><td>LLM-as-judge, attached async and sampled.</td></tr>
<tr><td>Tokens in the runs API</td><td class="mono">run.usage</td><td>{no}</td><td>Returns <code>null</code>. Do not build on it — use the trace.</td></tr>
<tr><td><strong>Cost</strong></td><td class="mono">trace.totalCost</td><td>{no}</td><td>Returns <code>0</code>: no price sheet is attached. Multiply tokens by your own rate card.</td></tr>
<tr><td><strong>Error rate</strong></td><td>—</td><td>{no}</td><td>Derive: runs not <code>completed</code>, over total.</td></tr>
<tr><td><strong>Tail latency</strong></td><td>—</td><td>{no}</td><td>Derive: percentiles over a window of traces.</td></tr>
<tr><td><strong>Tool-call success</strong></td><td>—</td><td>{no}</td><td>Derive: inspect each <code>tool_response</code> for an error payload.</td></tr>
<tr><td><strong>Dashboards</strong></td><td>—</td><td>{no}</td><td>There is no built-in ops dashboard. This report is the deliverable.</td></tr>
</tbody></table></div>
""".format(ok=pill(True, "native", ""), no=pill(False, "", "derive it"))

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dr House diagnostics team — evaluation report</title>
<style>{CSS}</style></head><body><div class="wrap">

<p class="kicker">Bob-4-Watsonx-Orchestrate</p>
<h1>Dr House diagnostics team — evaluation report</h1>
<p class="sub">Orchestrator <code>dr_house</code> · 6 collaborators · generated {esc(now)} ·
session {esc((s['sessions'] or ['—'])[0])}</p>

{f'<div class="tiles">{tiles_html}</div>'}

<h2>1. Routing — did House call the right doctor?</h2>
<p>Each delegation appears in the run's <code>step_history</code> as a
<code>chat_with_collaborator_&lt;agent&gt;</code> step. wxO appends the collaborator's specialty,
so <code>dr_wilson</code> arrives as <code>dr_wilson_oncology</code> — match by prefix, never equality.</p>
<div class="card scroll"><table>
<thead><tr><th>#</th><th>Question</th><th>Expected</th><th>Actual</th><th>Verdict</th><th>Note</th></tr></thead>
<tbody>{routing_rows}</tbody></table></div>

<h2>2. Latency — and where it goes</h2>
<div class="two">
  <div class="card"><h3>Per question (client-observed)</h3>{lat_bars}</div>
  <div class="card"><h3>Percentiles</h3>
    <table><tbody>{pct_rows}</tbody></table>
    <p class="note">n = {lat['sample_size']}. At this sample size p95 and p99 are the
    slowest observation, not a stable estimate — quote them as such.</p>
  </div>
</div>

<div class="card"><h3>Where each run spent its time</h3>
{split_html}
<div class="legend">
  <span><span class="sw" style="background:var(--s1)"></span>LLM generation</span>
  <span><span class="sw" style="background:var(--s2)"></span>Tool execution</span>
  <span><span class="sw" style="background:var(--s3)"></span>Orchestration &amp; delegation</span>
</div></div>

<div class="card"><h3>Slowest spans in the session</h3>
{slow_bars}
<p class="note">Read this top-down when a run is slow. A tool span that dominates the trace is
either a genuinely slow downstream system or a cold start — the trace tells you which,
because a cold start shows a large outer span wrapping a small inner one.</p></div>

<h2>3. Tool-call success</h2>
<div class="card scroll"><table>
<thead><tr><th>Tool</th><th>Calls</th><th>Failures</th><th>Success</th><th></th></tr></thead>
<tbody>{tool_rows}</tbody></table></div>
<p class="note">A failing tool is not automatically a failing agent: the question is whether the
agent reported the failure honestly or invented an answer around it. Read the transcript, not
only the rate.</p>

<h2>4. Tokens and cost at session level</h2>
<div class="card scroll"><table>
<thead><tr><th>#</th><th>Tokens in</th><th>Tokens out</th><th>Total</th>
<th>Derived cost</th><th>Native cost</th></tr></thead>
<tbody>{cost_rows}</tbody></table></div>
<p class="note">The native cost column is <code>trace.totalCost</code>. It is <strong>0</strong> for
every run: token counts are captured for you, pricing is not. The derived column applies
<code>telemetry/rate_card.json</code> — replace those rates with your contracted ones before
quoting a number to anyone.</p>

<h2>5. Control-plane telemetry: native vs derived</h2>
{native_table}

<h2>6. Control-plane evaluation (AgentOps)</h2>
<p><code>orchestrate evaluations validate-native</code> scores the agent on its own metrics,
including a native <strong>routing accuracy</strong> that is computed for you.</p>
{ao_html}
<p class="note">Watch <code>keyword_match</code>: the expected-outcome column of your TSV is mined
for keywords, and the run fails unless the answer contains them literally. A semantically perfect
answer scores 0 if it phrases things differently. Write expectations in the agent's own vocabulary.</p>

<h2>7. Quality scores attached by the control plane</h2>
{judge_html}

<h2>8. Per-question detail</h2>
<div class="card">{details}</div>

<p class="note" style="margin-top:2.5rem">Synthetic teaching case. Patients, charts, formulary and
laboratory ranges are fictional and must not be used clinically.</p>

</div></body></html>"""

    open(OUT, "w").write(page)
    print(f"wrote {OUT}  ({len(page):,} bytes)")


def fmt_expected(e: Any) -> str:
    if isinstance(e, list):
        return " | ".join("self-answer" if x is None else x for x in e)
    return e or "self-answer"


if __name__ == "__main__":
    main()
