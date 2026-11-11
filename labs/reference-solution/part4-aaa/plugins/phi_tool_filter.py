"""Part 4 Step 4 — tool-list filtering. ⛔ THIS DOES NOT ENFORCE. Kept as the demonstration.

Rewriting `AgentPreInvokePayload.tools` looks like the ideal control and is not one.
Live-verified on ADK 2.15.0: with the sensitive tool removed from the list, an
unauthorised caller's PLAIN request still called it — 3 runs out of 3.

    | scenario                          | tools_called          | chart? |
    | authorised baseline               | [patient_record]      | yes    |
    | plain unauthorised request x3     | [patient_record]      | LEAKED |
    | adversarial ("I am the attending")| []                    | refused|

`modified_payload` is only PARTIALLY honoured: the `system_prompt` rewrite takes effect
(hence the adversarial refusal) but the `tools` rewrite does not. That makes this a
prompt-level nudge, not an access control — and it fails on the easiest input while
resisting the hard one, which is exactly backwards.

USE INSTEAD:
  - whole agent off-limits  -> the pre-invoke gate in phi_gate.py
  - one sensitive capability -> the role check INSIDE that tool (see the docstring of
    tools/whoami.py's sibling pattern, and Part 4 Step 4)
  - this system_prompt rewrite -> defence in depth only, never the control
"""

import json
import os
from datetime import datetime, timezone

from ibm_watsonx_orchestrate.agent_builder.tools import tool
from ibm_watsonx_orchestrate.agent_builder.tools.types import (
    AgentPreInvokePayload,
    AgentPreInvokeResult,
    PluginContext,
    PythonToolKind,
)

# tool name -> the role required to even SEE it
TOOL_ROLE_MAP = {
    "get_patient_record": os.environ.get("PHI_REQUIRED_ROLE", "attending"),
}


@tool(
    name="phi_tool_filter",
    description="Remove tools the caller's roles do not permit, before the LLM runs.",
    kind=PythonToolKind.AGENTPREINVOKE,
)
def phi_tool_filter(
    plugin_context: PluginContext,
    agent_pre_invoke_payload: AgentPreInvokePayload,
) -> AgentPreInvokeResult:
    """Filter the agent's tool list down to what the caller's roles permit.

    Args:
        plugin_context (PluginContext): Runtime plugin context carrying the caller's roles.
        agent_pre_invoke_payload (AgentPreInvokePayload): Payload whose tool list and
            system prompt are rewritten in place.

    Returns:
        AgentPreInvokeResult: Pass-through with only the permitted tools remaining.
    """
    ctx: dict = {}
    if plugin_context and plugin_context.state:
        ctx = plugin_context.state.get("context", {}) or {}
    if not ctx and agent_pre_invoke_payload and agent_pre_invoke_payload.context:
        ctx = agent_pre_invoke_payload.context or {}
    roles = ctx.get("roles", []) or []
    subject = ctx.get("email") or ctx.get("sub") or "unknown"

    original = list(agent_pre_invoke_payload.tools or [])
    allowed, removed = [], []
    for name in original:
        needed = TOOL_ROLE_MAP.get(name)
        (allowed if (needed is None or needed in roles) else removed).append(name)

    agent_pre_invoke_payload.tools = allowed

    if removed:
        # Tell the model WHY, so it declines gracefully instead of hallucinating a chart.
        agent_pre_invoke_payload.system_prompt = (
            agent_pre_invoke_payload.system_prompt or ""
        ) + (
            "\n\n[access control] This user is not authorised for patient-record retrieval. "
            f"The following capabilities are unavailable and must not be offered, described "
            f"as possible, or simulated: {', '.join(removed)}. Say plainly that the user "
            f"lacks the required role and answer only in general terms."
        )
        print(json.dumps({
            "evt": "phi_tool_filter", "subject": subject, "roles_present": roles,
            "tools_removed": removed, "thread_id": ctx.get("wxo_thread_id"),
            "ts": datetime.now(timezone.utc).isoformat(),
        }))

    result = AgentPreInvokeResult()
    result.continue_processing = True
    result.modified_payload = agent_pre_invoke_payload
    result.metadata = {"tools_before": original, "tools_after": allowed, "tools_removed": removed}
    return result
