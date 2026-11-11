"""Minimal watsonx Orchestrate runtime client used by the lab harness.

Reads the active environment and bearer token that `orchestrate env activate` already
minted, so nothing here needs your API key a second time:

    ~/.config/orchestrate/config.yaml      -> active env + wxo_url
    ~/.cache/orchestrate/credentials.yaml  -> wxo_mcsp_token

Everything talks to the documented runtime API:
    POST <url>/v1/orchestrate/runs          start a run
    GET  <url>/v1/orchestrate/runs/{id}     poll it
    GET  <url>/v1/orchestrate/agents        resolve agent name -> id
    GET  <url>/v1/agentops-v3/traces/{id}   rich trace (observations, latency, cost)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import yaml

CONFIG = os.path.expanduser("~/.config/orchestrate/config.yaml")
CREDS = os.path.expanduser("~/.cache/orchestrate/credentials.yaml")


class WxOClient:
    def __init__(self, env: Optional[str] = None, timeout: int = 300):
        cfg = yaml.safe_load(open(CONFIG))
        creds = yaml.safe_load(open(CREDS))
        self.env = env or cfg["context"]["active_environment"]
        self.url = cfg["environments"][self.env]["wxo_url"].rstrip("/")
        self.token = creds["auth"][self.env]["wxo_mcsp_token"]
        self.timeout = timeout
        self._agents: Optional[List[Dict[str, Any]]] = None

    # ---------------------------------------------------------------- transport
    def _req(self, path: str, method: str = "GET", body: Any = None) -> Any:
        req = urllib.request.Request(
            f"{self.url}{path}",
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read() or b"null")

    # ------------------------------------------------------------------ agents
    def agents(self) -> List[Dict[str, Any]]:
        if self._agents is None:
            raw = self._req("/v1/orchestrate/agents")
            self._agents = raw if isinstance(raw, list) else raw.get("data", raw)
        return self._agents

    def agent_id(self, name: str) -> str:
        for a in self.agents():
            if a.get("name") == name:
                return a.get("agent_id") or a.get("id")
        raise KeyError(f"agent {name!r} not found in env {self.env!r}")

    # -------------------------------------------------------------------- runs
    def run(
        self,
        agent: str,
        message: str,
        thread_id: Optional[str] = None,
        poll: float = 1.0,
    ) -> Dict[str, Any]:
        """Start a run, poll to completion, return the full run object plus timing.

        Adds three keys the API does not return itself:
            _wall_seconds   client-observed end-to-end latency
            _started_at     unix epoch when the POST was issued
            _polls          how many GETs it took
        """
        body: Dict[str, Any] = {
            "message": {"role": "user", "content": message},
            "agent_id": self.agent_id(agent),
        }
        if thread_id:
            body["thread_id"] = thread_id

        t0 = time.time()
        started = self._req("/v1/orchestrate/runs", "POST", body)
        run_id = started["run_id"]

        polls = 0
        while True:
            run = self._req(f"/v1/orchestrate/runs/{run_id}")
            polls += 1
            if str(run.get("status", "")).lower() in ("completed", "failed", "cancelled"):
                break
            if time.time() - t0 > self.timeout:
                run = {"status": "timeout", "run_id": run_id}
                break
            time.sleep(poll)

        run["_wall_seconds"] = round(time.time() - t0, 3)
        run["_started_at"] = t0
        run["_polls"] = polls
        run.setdefault("thread_id", started.get("thread_id"))
        return run

    # ------------------------------------------------------------------ traces
    def trace(self, trace_id: str) -> Any:
        """AgentOps v3 trace. Returns {'_http_error': code} instead of raising."""
        try:
            return self._req(f"/v1/agentops-v3/traces/{trace_id}")
        except urllib.error.HTTPError as e:
            return {"_http_error": e.code, "_body": e.read().decode()[:400]}


# ------------------------------------------------------------------ extraction
def final_text(run: Dict[str, Any]) -> str:
    msg = (run.get("result") or {}).get("data", {}).get("message", {})
    content = msg.get("content", "")
    if isinstance(content, list):
        return "\n".join(
            c.get("text", "") for c in content if c.get("response_type") != "thinking"
        ).strip()
    return str(content)


def steps(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    msg = (run.get("result") or {}).get("data", {}).get("message", {})
    return msg.get("step_history", []) or []


def tool_calls(run: Dict[str, Any]) -> List[str]:
    """Every tool the run invoked, in order. Collaborator delegations appear here as
    `chat_with_collaborator_<agent>` — that is the routing evidence."""
    names: List[str] = []
    for step in steps(run):
        for detail in step.get("step_details", []) or []:
            if detail.get("type") != "tool_calls":
                continue
            for tc in detail.get("tool_calls", []) or []:
                names.append(tc.get("name") or tc.get("function", {}).get("name") or "?")
    return names


def delegations(run: Dict[str, Any]) -> List[str]:
    """Just the collaborator delegations, with the prefix stripped.

    wxO appends the collaborator's specialty to the generated tool name, so a
    delegation to `dr_wilson` arrives as `chat_with_collaborator_dr_wilson_oncology`.
    Match by prefix, never by equality — live-verified 2.13.0.
    """
    return [
        t[len("chat_with_collaborator_"):]
        for t in tool_calls(run)
        if t.startswith("chat_with_collaborator_")
    ]


def routed_to(run: Dict[str, Any], expected_agent: str) -> bool:
    """True if the run delegated to `expected_agent`, tolerating the specialty suffix."""
    return any(d == expected_agent or d.startswith(expected_agent + "_")
               for d in delegations(run))


def tool_failures(run: Dict[str, Any]) -> List[Dict[str, str]]:
    """Tool responses that carry an error. Feeds the tool-call success rate."""
    out: List[Dict[str, str]] = []
    for step in steps(run):
        for detail in step.get("step_details", []) or []:
            if detail.get("type") != "tool_response":
                continue
            content = json.dumps(detail.get("content", ""))
            low = content.lower()
            if any(k in low for k in ('"error"', "traceback", "not in the ppth formulary")):
                out.append({"tool": detail.get("name", "?"), "content": content[:300]})
    return out
