"""Part 4 Step 3 — the pre-invoke gate (RBAC Option C).

Blocks the whole agent turn unless the caller holds the required role. The model is
never invoked: `tools_called` comes back empty and the turn is roughly an order of
magnitude faster than a real one.

⚠ TWO THINGS THAT ARE NOT OBVIOUS (both live-verified on ADK 2.15.0):

1. `PluginViolation` is INERT. Setting `result.violation` does block the turn, but its
   reason/description/code/details appear in NEITHER the run payload NOR the trace, and
   the user sees their own message echoed back. You MUST rewrite the message text or the
   refusal is invisible. We still populate `violation` — it costs nothing and may become
   observable later — but it is not an audit record.

2. `result.modified_payload` must never be None. None produces a generic
   "Plugin execution failed" instead of your message.

Attach it in the agent YAML as a PluginRef OBJECT, not a string:

    plugins:
      agent_pre_invoke:
        - plugin_name: phi_gate
      agent_post_invoke: []
"""

import json
import os
from datetime import datetime, timezone

from ibm_watsonx_orchestrate.agent_builder.tools import tool
from ibm_watsonx_orchestrate.agent_builder.tools.types import (
    AgentPreInvokePayload,
    AgentPreInvokeResult,
    PluginContext,
    PluginViolation,
    PythonToolKind,
)

REQUIRED_ROLE = os.environ.get("PHI_REQUIRED_ROLE", "attending")


def _identity(plugin_context: PluginContext, payload: AgentPreInvokePayload):
    """Read roles and subject from whichever context slot is populated."""
    ctx: dict = {}
    if plugin_context and plugin_context.state:
        ctx = plugin_context.state.get("context", {}) or {}
    if not ctx and payload and payload.context:
        ctx = payload.context or {}
    subject = ctx.get("email") or ctx.get("sub") or ctx.get("wxo_user_name") or "unknown"
    return (ctx.get("roles", []) or []), subject, ctx


@tool(
    name="phi_gate",
    description="Block the agent turn unless the caller holds the PHI-access role.",
    kind=PythonToolKind.AGENTPREINVOKE,
)
def phi_gate(
    plugin_context: PluginContext,
    agent_pre_invoke_payload: AgentPreInvokePayload,
) -> AgentPreInvokeResult:
    """Deny the turn when the caller lacks the role required to see patient data.

    Args:
        plugin_context (PluginContext): Runtime plugin context; `state` is a plain dict.
        agent_pre_invoke_payload (AgentPreInvokePayload): Incoming payload, returned
            unchanged when the caller is authorised.

    Returns:
        AgentPreInvokeResult: A blocking result carrying the refusal text when the role
            is missing, or a clean pass-through when it is present.
    """
    roles, subject, ctx = _identity(plugin_context, agent_pre_invoke_payload)
    allowed = REQUIRED_ROLE in roles

    # ACCOUNTING: a blocked turn produces NO TRACE, so this line is the only record that
    # the refusal ever happened. In production, ship it somewhere durable.
    print(json.dumps({
        "evt": "phi_gate_decision", "allowed": allowed, "subject": subject,
        "required_role": REQUIRED_ROLE, "roles_present": roles,
        "thread_id": ctx.get("wxo_thread_id"), "run_id": ctx.get("wxo_run_id"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }))

    result = AgentPreInvokeResult()
    if allowed:
        result.continue_processing = True
        result.modified_payload = agent_pre_invoke_payload
        return result

    denial = (
        f"Access denied: your account ({subject}) does not have the '{REQUIRED_ROLE}' "
        f"role required to access patient information. Contact your administrator."
    )
    # The rewrite is what the user actually sees. Without it they get their own message back.
    if agent_pre_invoke_payload and agent_pre_invoke_payload.messages:
        agent_pre_invoke_payload.messages[-1].content.text = denial

    result.continue_processing = False
    result.modified_payload = agent_pre_invoke_payload      # never None
    result.violation = PluginViolation(
        reason="missing_required_role",
        description=denial,
        code="PHI_RBAC_001",
        details={"subject": subject, "required_role": REQUIRED_ROLE, "roles_present": roles},
    )
    return result
