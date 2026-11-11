"""Part 4 Step 1 — the identity probe.

Dumps every key the runtime puts in the tool context so the same agent can be called from
each access surface and the results diffed. Returns STRINGS, not nested models: the keys
differ per surface and a loose schema keeps the tool from erroring when one is absent —
an error tells you nothing.

⚠ Emits a PREFIX of any token-shaped value, never the whole thing. A full sso_token in a
trace is a credential in a log.
"""

import json

from pydantic import BaseModel

from ibm_watsonx_orchestrate.agent_builder.tools import tool
from ibm_watsonx_orchestrate.run.context import AgentRun

SENSITIVE = ("sso_token", "user_payload", "token", "access_token")


class ContextProbe(BaseModel):
    surface_hint: str
    all_keys: str
    values: str


@tool(name="whoami", description="Report the caller's identity as seen by the tool runtime.")
def whoami(context: AgentRun, surface_hint: str = "unspecified") -> ContextProbe:
    """Report every identity value the watsonx Orchestrate runtime passes to this tool.

    Args:
        context (AgentRun): Injected agent-run context carrying the request context.
        surface_hint (str): Label for how this call was made, for example 'rest_api' or
            'embedded_webchat'. Echoed back unchanged so surfaces can be told apart.

    Returns:
        ContextProbe: The full list of context keys and their values, tokens truncated.
    """
    rc = context.request_context if context and context.request_context else {}
    keys = sorted(rc.keys()) if hasattr(rc, "keys") else []

    def show(k):
        v = rc.get(k)
        if k in SENSITIVE and isinstance(v, str) and v:
            return f"<{len(v)} chars> {v[:24]}..."
        return v if isinstance(v, str) else json.dumps(v, default=str)

    return ContextProbe(
        surface_hint=surface_hint,
        all_keys=json.dumps(keys),
        values=json.dumps({k: show(k) for k in keys}, default=str)[:2000],
    )
