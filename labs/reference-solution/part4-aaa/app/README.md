# Part 4 reference app — AAA demo on Podman

The **trusted server** zone of the AAA architecture, in one container:
IBM App ID login → roles from the `id_token` → a watsonx Orchestrate agent call with
identity passed as **allowlisted context variables**.

The browser holds **no secret** and never talks to Orchestrate directly. On the REST
surface roles are *asserted* by whoever holds the API key — so the key stays server-side.
That is the entire security argument; see the AAA skill §4.6.

## Run it

```bash
cp .env.example .env      # fill in App ID + wxO values
podman-compose up --build # or: docker compose up --build
open http://localhost:3000
```

## Before it will work

1. **App ID → Applications** — add `http://localhost:3000/auth/callback` as a redirect URI.
2. **App ID → Profiles and roles → Roles** — create `attending` with `scopes: ["roles"]`.
3. **Map roles into the token** — `PUT /management/v4/<tenant>/config/tokens`.
   Without this the `roles` claim never appears and every login looks unauthorised.
4. **The agent must allowlist the context keys**, and the allowlist is *exclusive*:

```yaml
context_variables: [roles, email, sub, clientID, wxo_run_id, wxo_thread_id, wxo_user_name]
```

Omit `wxo_run_id`/`wxo_thread_id` and they vanish from your tools — which breaks the
identity→run join the accounting step depends on.

## What to watch

- Sign in as a user **with** `attending` and **without** it — same app, different agent behaviour.
- The container log prints one `agent_turn` JSON line per turn with subject, roles,
  allow/deny, `thread_id`, `run_id`, `trace_id`. **A blocked turn produces no trace**, so
  that log line is the only record the refusal happened.

## Status

The REST identity path is live-verified. The full App ID browser login was **not**
exercised end to end while authoring — verify it on your tenant before demoing.
