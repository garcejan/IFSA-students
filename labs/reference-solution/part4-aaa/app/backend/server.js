/**
 * wxO AAA demo — Express backend.
 *
 * Demonstrates the TRUSTED SERVER zone of the AAA architecture:
 *   - performs the OIDC login against IBM App ID (client secret never leaves here)
 *   - reads `roles` from the id_token claims  (App ID + IBMid federation does NOT
 *     return roles from /userinfo — the claim must be mapped into the token)
 *   - calls watsonx Orchestrate with the instance API key (also never leaves here)
 *   - passes identity to the agent as ALLOWLISTED context variables
 *
 * The browser holds no secret and never talks to Orchestrate directly. That is the
 * whole point: on the REST surface roles are ASSERTED by whoever holds the API key,
 * so the API key must stay server-side.
 */
const express = require("express");
const session = require("express-session");
const { Issuer, generators } = require("openid-client");

const {
  APPID_OAUTH_SERVER_URL, APPID_CLIENT_ID, APPID_CLIENT_SECRET,
  WXO_INSTANCE_URL, WXO_API_KEY, WXO_AGENT_NAME = "dr_house",
  APP_BASE_URL = "http://localhost:3000",
  SESSION_SECRET = "dev-only-secret",
  REQUIRED_ROLE = "attending",
  PORT = 3000,
} = process.env;

const app = express();
app.use(express.json());
app.use(express.static(`${__dirname}/public`));
app.use(session({
  secret: SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: { httpOnly: true, sameSite: "lax" },   // httpOnly: the browser cannot read it
}));

let client;
async function oidc() {
  if (client) return client;
  const issuer = await Issuer.discover(`${APPID_OAUTH_SERVER_URL}/.well-known/openid-configuration`);
  client = new issuer.Client({
    client_id: APPID_CLIENT_ID,
    client_secret: APPID_CLIENT_SECRET,
    redirect_uris: [`${APP_BASE_URL}/auth/callback`],
    response_types: ["code"],
  });
  return client;
}

// --- auth -----------------------------------------------------------------
app.get("/auth/login", async (req, res) => {
  const c = await oidc();
  req.session.state = generators.state();
  res.redirect(c.authorizationUrl({ scope: "openid", state: req.session.state }));
});

app.get("/auth/callback", async (req, res) => {
  try {
    const c = await oidc();
    const params = c.callbackParams(req);
    const tokenSet = await c.callback(`${APP_BASE_URL}/auth/callback`, params, { state: req.session.state });
    const claims = tokenSet.claims();

    // ⚠ Roles come from the id_token claims, NOT from /userinfo. App ID needs a token
    //   claims mapping (PUT /management/v4/<tenant>/config/tokens) or this is always [].
    req.session.user = {
      sub: claims.sub,
      email: claims.email || "",
      name: claims.name || claims.email || claims.sub,
      roles: claims.roles || [],
      accessToken: tokenSet.access_token,
    };
    res.redirect("/");
  } catch (e) {
    res.status(500).send(`Login failed: ${e.message}`);
  }
});

app.get("/auth/me", (req, res) =>
  req.session.user
    ? res.json({ authenticated: true, ...req.session.user, accessToken: undefined })
    : res.status(401).json({ authenticated: false }));

app.post("/auth/logout", (req, res) => req.session.destroy(() => res.json({ ok: true })));

// --- the agent call -------------------------------------------------------
async function wxo(path, method = "GET", body) {
  const r = await fetch(`${WXO_INSTANCE_URL.replace(/\/$/, "")}${path}`, {
    method,
    headers: { Authorization: `Bearer ${WXO_API_KEY}`, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`wxO ${path} -> ${r.status} ${await r.text()}`);
  return r.json();
}

let agentIdCache;
async function agentId() {
  if (agentIdCache) return agentIdCache;
  const agents = await wxo("/v1/orchestrate/agents");
  const a = agents.find((x) => x.name === WXO_AGENT_NAME);
  if (!a) throw new Error(`agent ${WXO_AGENT_NAME} not found`);
  return (agentIdCache = a.id);
}

app.post("/api/chat", async (req, res) => {
  const u = req.session.user;
  if (!u) return res.status(401).json({ error: "not authenticated" });

  try {
    // Identity is injected as context variables. The AGENT must allowlist each key in
    // its `context_variables:` field or the tools will never see them — and note the
    // allowlist is EXCLUSIVE, so wxo_run_id/wxo_thread_id must be listed there too.
    const run = await wxo("/v1/orchestrate/runs", "POST", {
      agent_id: await agentId(),
      message: { role: "user", content: req.body.message },
      thread_id: req.body.thread_id || undefined,
      context: { roles: u.roles, email: u.email, sub: u.sub, clientID: "wxo-aaa-demo" },
    });

    let out = {};
    for (let i = 0; i < 60; i++) {
      out = await wxo(`/v1/orchestrate/runs/${run.run_id}`);
      if (["completed", "failed", "cancelled"].includes(out.status)) break;
      await new Promise((r) => setTimeout(r, 2000));
    }

    const msg = out.result?.data?.message || {};
    const text = (msg.content || []).map((c) => c.text || "").join("");
    const tools = (msg.step_history || []).flatMap((s) =>
      (s.step_details || []).filter((d) => d.type === "tool_calls")
        .flatMap((d) => (d.tool_calls || []).map((t) => t.name)));

    // ⚠ ACCOUNTING: a turn blocked by a pre-invoke gate produces NO TRACE. This log line
    //   is the only record that the refusal ever happened. Ship it to real storage.
    console.log(JSON.stringify({
      evt: "agent_turn", sub: u.sub, email: u.email, roles: u.roles,
      required_role: REQUIRED_ROLE, allowed: u.roles.includes(REQUIRED_ROLE),
      thread_id: out.thread_id, run_id: run.run_id, trace_id: out.trace_id,
      tools_called: tools, ts: new Date().toISOString(),
    }));

    res.json({ text, tools, thread_id: out.thread_id, run_id: run.run_id, trace_id: out.trace_id });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, "0.0.0.0", () => console.log(`AAA demo on ${APP_BASE_URL} (agent: ${WXO_AGENT_NAME})`));
