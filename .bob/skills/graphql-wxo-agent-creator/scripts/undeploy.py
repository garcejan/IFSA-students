#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
watsonx Orchestrate Agent Undeploy Script
Removes the deployed agent and toolkit from watsonx Orchestrate.
Does NOT delete local files.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Variables checked by group so the error message points to the right section of .env
REQUIRED_VARS_STEPZEN = [
    "STEPZEN_INSTANCE_API_KEY",
    "STEPZEN_ENVIRONMENT",
    "STEPZEN_DOMAIN",
    "STEPZEN_INSTANCE_ID",
]

REQUIRED_VARS_WXO = [
    "WXO_ENV_NAME",
    "WXO_IBM_CLOUD_API_KEY",
    "WXO_INSTANCE_URL",
    "WXO_CONNECTION_NAME",
    "WXO_TOOLKIT_NAME",
    "WXO_AGENT_NAME",
    "WXO_AGENT_DESCRIPTION",
    "WXO_AGENT_MODEL",
]

# Combined list
REQUIRED_VARS = REQUIRED_VARS_STEPZEN + REQUIRED_VARS_WXO

# Authentication type constants — mirrors deploy.py so undeploy targets the
# same kind of environment (ibm_iam vs mcsp_v2) that deploy.py registered.
# The orchestrate CLI infers the MCSP v2 IAM host itself (account-iam.platform
# .saas.ibm.com for AWS regions, confirmed working) — WXO_IAM_URL/--iam-url
# should only be set to override that for a region the CLI genuinely gets
# wrong, never defaulted to a guessed host.
WXO_AUTH_TYPE_IBM_IAM   = "ibm_iam"
WXO_AUTH_TYPE_MCSP_V2   = "mcsp_v2"
VALID_AUTH_TYPES        = {WXO_AUTH_TYPE_IBM_IAM, WXO_AUTH_TYPE_MCSP_V2}

# ---------------------------------------------------------------------------
# Path resolution
# This script lives at .bob/skills/graphql-wxo-agent-creator/scripts/undeploy.py
# so the project root is four levels up.
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parents[3]


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _resolve_cmd(cmd: list[str]) -> tuple[list[str], bool]:
    """On Windows resolve the bare executable to the venv Scripts/ path.

    Uses shell=False so arguments are always passed safely.
    Falls back to the bare name (PATH lookup) if the venv exe is not found.
    """
    if sys.platform != 'win32':
        return cmd, False
    exe = cmd[0]
    venv_exe = Path(sys.executable).parent / (exe + '.exe')
    if venv_exe.exists():
        return [str(venv_exe)] + list(cmd[1:]), False
    return cmd, False


def run_command(cmd: list[str], check: bool = True, quiet: bool = False,
                timeout: int = 120, cwd: Path | None = None):
    """Run a CLI command using Popen+communicate to avoid the Windows
    shell=True pipe-deadlock that breaks subprocess.run timeouts.

    Output is captured and printed after the process exits so it is always
    visible in the console.
    """
    if not quiet:
        print(f"$ {' '.join(str(a) for a in cmd)}", flush=True)

    resolved, use_shell = _resolve_cmd(cmd)
    try:
        proc = subprocess.Popen(
            resolved, shell=use_shell,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace',
            cwd=str(cwd) if cwd else None,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        print(f"[!] Timeout after {timeout}s", flush=True)
        if check:
            sys.exit(1)
        return None
    except FileNotFoundError:
        print(f"[X] Command not found: {cmd[0]}", flush=True)
        if check:
            sys.exit(1)
        return None

    if stdout and stdout.strip():
        print(stdout.rstrip(), flush=True)
    if stderr and stderr.strip():
        print(stderr.rstrip(), file=sys.stderr, flush=True)

    class _Result:
        def __init__(self, returncode, stdout, stderr):
            self.returncode = returncode
            self.stdout     = stdout
            self.stderr     = stderr

    result = _Result(proc.returncode, stdout, stderr)
    if check and result.returncode != 0:
        print(f"[X] Failed (exit {result.returncode})", file=sys.stderr, flush=True)
        sys.exit(1)
    return result


def run_command_capture(cmd: list[str], timeout: int = 30,
                        cwd: Path | None = None):
    """Like run_command but always captures output (never streams) and
    never exits on failure — returns None on timeout/not-found."""
    resolved, use_shell = _resolve_cmd(cmd)
    try:
        proc = subprocess.Popen(
            resolved, shell=use_shell,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace',
            cwd=str(cwd) if cwd else None,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None
    except FileNotFoundError:
        return None

    class _Result:
        def __init__(self, rc, out, err):
            self.returncode = rc
            self.stdout     = out
            self.stderr     = err

    return _Result(proc.returncode, stdout, stderr)


# ---------------------------------------------------------------------------
# CLI check
# ---------------------------------------------------------------------------

def check_orchestrate_cli() -> None:
    """Abort with a clear message if the orchestrate CLI is not on PATH."""
    import shutil
    if shutil.which("orchestrate") is None:
        print("\n[X] 'orchestrate' CLI not found on PATH.", file=sys.stderr)
        print("    Install and source it before running this script:", file=sys.stderr)
        print("    https://www.ibm.com/docs/en/watsonx/watson-orchestrate/current"
              "?topic=cli-installing", file=sys.stderr)
        print(file=sys.stderr)
        print("    Quick setup (run once per shell session):", file=sys.stderr)
        print("      pip install ibm-watsonx-orchestrate", file=sys.stderr)
        print("    Then verify with:  orchestrate --version", file=sys.stderr)
        sys.exit(1)
    result = run_command_capture(["orchestrate", "--version"])
    version = (result.stdout or result.stderr or "").strip().splitlines()[0] if result else "unknown"
    print(f"[i] orchestrate CLI: {version}")


# ---------------------------------------------------------------------------
# Hint messages
# ---------------------------------------------------------------------------

def print_wxo_setup_hint() -> None:
    """Print setup instructions when watsonx Orchestrate is not yet configured."""
    print("\n" + "=" * 60, file=sys.stderr)
    print("  watsonx Orchestrate — Not Configured", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(file=sys.stderr)
    print("  0. If you don't have an instance yet, start a free trial:", file=sys.stderr)
    print("     No credit card (AWS/Azure trial): https://www.ibm.com/account/reg/signup?formid=urx-54418", file=sys.stderr)
    print("     IBM Cloud (credit card required):  https://www.ibm.com/account/reg/us-en/signup?formid=urx-54418", file=sys.stderr)
    print("     -> Sign in with an IBMid to activate your free instance.", file=sys.stderr)
    print(file=sys.stderr)
    print("  Once you have an instance, collect the required values:", file=sys.stderr)
    print(file=sys.stderr)
    print("  WXO_INSTANCE_URL — the base URL of your Orchestrate instance.", file=sys.stderr)
    print("    IBM Cloud:  https://dl.watson-orchestrate.ibm.com/instances/<id>", file=sys.stderr)
    print("                Found on the IBM Cloud resource page for your instance.", file=sys.stderr)
    print("    AWS trial:  log in to your watsonx Orchestrate instance, click your", file=sys.stderr)
    print("                User icon (top-right) -> Settings -> API details tab, and", file=sys.stderr)
    print("                copy the 'Service instance URL' shown there.", file=sys.stderr)
    print(file=sys.stderr)
    print("  WXO_IBM_CLOUD_API_KEY — the API key for your Orchestrate instance.", file=sys.stderr)
    print("    IBM Cloud:  https://cloud.ibm.com/iam/apikeys ('Viewer' access is enough).", file=sys.stderr)
    print("    AWS trial:  same Settings -> API details tab, click 'Generate API key'.", file=sys.stderr)
    print("                Copy it immediately — it cannot be viewed again.", file=sys.stderr)
    print(file=sys.stderr)
    print("  WXO_ENV_NAME — the exact resource name of your watsonx Orchestrate instance", file=sys.stderr)
    print("    as it appears in the IBM Cloud resource list under AI / Machine Learning:", file=sys.stderr)
    print("    https://cloud.ibm.com/resources", file=sys.stderr)
    print(file=sys.stderr)
    print("  WXO_CONNECTION_NAME — the app_id of an existing connection in your", file=sys.stderr)
    print("    Orchestrate instance (links the MCP toolkit to the gateway).", file=sys.stderr)
    print("    How to create a connection:", file=sys.stderr)
    print("    https://www.ibm.com/docs/en/watsonx/watson-orchestrate/base", file=sys.stderr)
    print("          ?topic=credentials-connection-overview", file=sys.stderr)
    print(file=sys.stderr)
    print("  Add all values to your .env file, then re-run this script.", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(file=sys.stderr)


def print_wxo_auth_hint() -> None:
    """Print guidance when authentication against watsonx Orchestrate fails."""
    auth_type = os.getenv("WXO_AUTH_TYPE", WXO_AUTH_TYPE_IBM_IAM).strip()
    print("\n" + "=" * 60, file=sys.stderr)
    print("  Authentication Failed — watsonx Orchestrate", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(file=sys.stderr)

    if auth_type == WXO_AUTH_TYPE_MCSP_V2:
        print("  Detected: AWS / Azure trial tenant (WXO_AUTH_TYPE=mcsp_v2)", file=sys.stderr)
        print(file=sys.stderr)
        print("  Steps to verify:", file=sys.stderr)
        print("    1. Confirm WXO_IBM_CLOUD_API_KEY was generated in the watsonx", file=sys.stderr)
        print("       Orchestrate UI itself: User icon -> Settings -> API details ->", file=sys.stderr)
        print("       Generate API key (NOT an IBM Cloud platform key).", file=sys.stderr)
        print("    2. Confirm WXO_INSTANCE_URL is the exact 'Service instance URL' from", file=sys.stderr)
        print("       that same API details page (it must contain ?mcsp_metadata=).", file=sys.stderr)
        print("    3. Re-register the environment if it was previously added as 'ibm_iam':", file=sys.stderr)
        print("         orchestrate env list", file=sys.stderr)
        print("         orchestrate env add -n <name> -u <WXO_INSTANCE_URL> --type mcsp_v2",
              file=sys.stderr)
        print("         orchestrate env activate <name> --api-key <WXO_API_KEY>",
              file=sys.stderr)
    else:
        print("  Detected: IBM Cloud tenant (WXO_AUTH_TYPE=ibm_iam)", file=sys.stderr)
        print(file=sys.stderr)
        print("  WXO_IBM_CLOUD_API_KEY must be a valid IBM Cloud API key", file=sys.stderr)
        print("  with access to the target Orchestrate instance.", file=sys.stderr)
        print(file=sys.stderr)
        print("  Steps to verify:", file=sys.stderr)
        print("    1. Go to https://cloud.ibm.com/iam/apikeys", file=sys.stderr)
        print("    2. Confirm the key is active and not expired.", file=sys.stderr)
        print("    3. Confirm it has at least 'Viewer' access to the", file=sys.stderr)
        print("       watsonx Orchestrate instance.", file=sys.stderr)
        print("    4. Update WXO_IBM_CLOUD_API_KEY in .env and re-run.", file=sys.stderr)
        print(file=sys.stderr)
        print("    If you have an AWS/Azure trial tenant, set WXO_AUTH_TYPE=mcsp_v2 in your .env.",
              file=sys.stderr)

    print(file=sys.stderr)
    print("  Free trial (no credit card): https://www.ibm.com/account/reg/signup?formid=urx-54418", file=sys.stderr)
    print("  IBM Cloud (credit card):     https://www.ibm.com/account/reg/us-en/signup?formid=urx-54418", file=sys.stderr)
    print("  Docs:        https://www.ibm.com/docs/en/watsonx/watson-orchestrate/base", file=sys.stderr)
    print("               ?topic=started-getting-access", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(file=sys.stderr)


# ---------------------------------------------------------------------------
# Env loading
# ---------------------------------------------------------------------------

def load_env() -> None:
    """Load .env from the project root and validate required variables by group."""
    possible_paths = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "api-connect" / ".env",
    ]
    env_path = next((p for p in possible_paths if p.is_file()), None)

    if not env_path:
        print("[X] .env file not found", file=sys.stderr)
        for p in possible_paths:
            print(f"    Looked in: {p}", file=sys.stderr)
        print_wxo_setup_hint()
        sys.exit(1)

    print(f"[i] Loading environment from: {env_path.resolve()}")
    load_dotenv(env_path)

    # ── Validate by group so the error message is actionable ──────────────
    missing_stepzen = [v for v in REQUIRED_VARS_STEPZEN if not os.getenv(v, "").strip()]
    missing_wxo     = [v for v in REQUIRED_VARS_WXO     if not os.getenv(v, "").strip()]

    if missing_stepzen or missing_wxo:
        print("[X] Missing required environment variables:", file=sys.stderr)
        if missing_stepzen:
            print("    [StepZen / IBM API Connect for GraphQL]", file=sys.stderr)
            for var in missing_stepzen:
                print(f"      - {var}", file=sys.stderr)
        if missing_wxo:
            print("    [watsonx Orchestrate]", file=sys.stderr)
            for var in missing_wxo:
                print(f"      - {var}", file=sys.stderr)
            core = {"WXO_ENV_NAME", "WXO_INSTANCE_URL", "WXO_IBM_CLOUD_API_KEY"}
            if core & set(missing_wxo):
                print_wxo_setup_hint()
        sys.exit(1)

    auth_type = os.getenv("WXO_AUTH_TYPE", WXO_AUTH_TYPE_IBM_IAM).strip()
    if auth_type not in VALID_AUTH_TYPES:
        print(f"[X] Invalid WXO_AUTH_TYPE: '{auth_type}'. "
              f"Allowed: {', '.join(sorted(VALID_AUTH_TYPES))}", file=sys.stderr)
        print("    IBM Cloud tenants  → WXO_AUTH_TYPE=ibm_iam  (default)",
              file=sys.stderr)
        print("    AWS/Azure trial    → WXO_AUTH_TYPE=mcsp_v2",
              file=sys.stderr)
        sys.exit(1)

    print("✅ Environment variables loaded:")
    print("   [StepZen / IBM API Connect for GraphQL]")
    for var in REQUIRED_VARS_STEPZEN:
        value = os.getenv(var, "")
        display = value[:8] + "****" if "KEY" in var or "API" in var else value
        print(f"     {var}: {display}")
    print("   [watsonx Orchestrate]")
    for var in REQUIRED_VARS_WXO:
        value = os.getenv(var, "")
        display = value[:8] + "****" if "KEY" in var or "API" in var else value
        print(f"     {var}: {display}")


# ---------------------------------------------------------------------------
# Environment activation
# ---------------------------------------------------------------------------

def _orchestrate_env_state(env_name: str) -> tuple[bool, str | None]:
    """Parse 'orchestrate env list' to get (env_exists, active_env_name).

    Uses only the CLI's own documented output — never reads config.yaml —
    since 'env list' tags whichever environment is currently active with
    '(active)'.
    """
    result = run_command_capture(["orchestrate", "env", "list"])
    output = result.stdout if result else ""
    exists = bool(env_name and env_name in output)
    active = None
    for line in output.splitlines():
        if "(active)" in line:
            parts = line.split()
            if parts:
                active = parts[0]
            break
    return exists, active


def _activate_environment(env_name: str, api_key: str) -> bool:
    """Run 'orchestrate env activate'. Returns success/failure, never exits.

    'env activate' already re-authenticates only if its own cached token
    for env_name has expired, so calling it is cheap when the session is
    still fresh.
    """
    print("  Activating...")
    result = run_command([
        "orchestrate", "env", "activate", env_name,
        "--api-key", api_key,
    ], check=False)
    if result and result.returncode == 0:
        return True
    if result:
        output = (result.stdout or "") + (result.stderr or "")
        if any(kw in output.lower() for kw in ("unauthorized", "forbidden", "401", "403", "invalid")):
            print_wxo_auth_hint()
    return False


def ensure_environment() -> None:
    """Add (if missing) and activate the watsonx Orchestrate environment.

    Mirrors deploy.py's ensure_environment: validity is judged purely from
    'orchestrate env list'. If WXO_ENV_NAME is already the '(active)'
    environment, it's trusted as-is and no login call is made. Otherwise
    'env activate' is run for real; if that fails for an entry that already
    existed (stale auth type or an old WXO_INSTANCE_URL), it's removed and
    re-added once, then retried — so undeploy never silently targets the
    wrong or an invalid environment.
    """
    print("\n[*] Environment...", flush=True)
    env_name     = os.getenv("WXO_ENV_NAME")
    instance_url = os.getenv("WXO_INSTANCE_URL")
    api_key      = os.getenv("WXO_IBM_CLOUD_API_KEY")
    auth_type    = os.getenv("WXO_AUTH_TYPE", WXO_AUTH_TYPE_IBM_IAM).strip()
    iam_url      = os.getenv("WXO_IAM_URL", "").strip() or None

    def _add_env() -> None:
        print(f"  Adding environment '{env_name}' ({auth_type})...")
        cmd = [
            "orchestrate", "env", "add",
            "-n", env_name,
            "-u", instance_url,
            "--type", auth_type,
        ]
        if auth_type == WXO_AUTH_TYPE_MCSP_V2 and iam_url:
            cmd += ["--iam-url", iam_url]
        run_command(cmd)

    env_exists, active_env = _orchestrate_env_state(env_name)

    if not env_exists:
        _add_env()
        if not _activate_environment(env_name, api_key):
            sys.exit(1)
        return

    if active_env == env_name:
        print(f"  ✅ '{env_name}' is already the active environment — skipping login")
        return

    print(f"  '{env_name}' exists but is not the active environment "
          f"(active: {active_env or 'none'}) — activating...")
    if _activate_environment(env_name, api_key):
        return

    print(f"  Activation failed for '{env_name}' — it may be stale (wrong "
          "auth type or an old WXO_INSTANCE_URL). Re-registering and retrying once...")
    run_command(["orchestrate", "env", "remove", "-n", env_name])
    _add_env()
    if not _activate_environment(env_name, api_key):
        sys.exit(1)


# ---------------------------------------------------------------------------
# Existence checks (capture-only, never abort)
# ---------------------------------------------------------------------------

def check_agent_exists(target_name: str) -> bool:
    """Return True if the agent exists in watsonx Orchestrate."""
    print(f"\n📋 Checking agents for '{target_name}':", flush=True)
    result = run_command_capture(["orchestrate", "agents", "list", "-v"])
    if not result or not result.stdout:
        print(f"  [i] Could not list agents")
        return False
    try:
        data = json.loads(result.stdout)
        all_agents: list = []
        if isinstance(data, dict):
            for section in ('native', 'assistant', 'external'):
                all_agents.extend(data.get(section, []))
        elif isinstance(data, list):
            all_agents = data
        for agent in all_agents:
            if agent.get("name") == target_name:
                print(f"  [✓] FOUND: '{target_name}'")
                return True
    except (json.JSONDecodeError, Exception):
        # Fallback: plain-text search
        if target_name in (result.stdout or ""):
            print(f"  [✓] FOUND (text match): '{target_name}'")
            return True
    print(f"  [i] '{target_name}' NOT FOUND")
    return False


def check_toolkit_exists(target_name: str) -> bool:
    """Return True if the toolkit exists in watsonx Orchestrate."""
    print(f"\n📋 Checking toolkits for '{target_name}':", flush=True)
    result = run_command_capture(["orchestrate", "toolkits", "list", "-v"])
    if not result or not result.stdout:
        print(f"  [i] Could not list toolkits")
        return False

    # Clean control characters that can appear in JSON output
    cleaned = ''.join(
        c if (ord(c) >= 32 or c in '\n\r\t') else ' '
        for c in result.stdout
    ).replace('\r', '').replace('\n', ' ')

    try:
        toolkits = json.loads(cleaned)
        if isinstance(toolkits, list):
            for tk in toolkits:
                if tk.get('name') == target_name:
                    print(f"  [✓] FOUND: '{target_name}'")
                    return True
        elif isinstance(toolkits, dict):
            for section in toolkits.values():
                if isinstance(section, list):
                    for tk in section:
                        if tk.get('name') == target_name:
                            print(f"  [✓] FOUND: '{target_name}'")
                            return True
    except (json.JSONDecodeError, Exception):
        # Fallback: plain-text search in raw output
        if target_name in (result.stdout or ""):
            print(f"  [✓] FOUND (text match): '{target_name}'")
            return True

    print(f"  [i] '{target_name}' NOT FOUND")
    return False


def wait_for_cleanup(name: str, check_func, resource_type: str,
                     max_retries: int = 5, delay: int = 3) -> bool:
    """Poll until the resource is gone or retries are exhausted."""
    print(f"  [~] Verifying {resource_type} '{name}' removal...")
    for attempt in range(1, max_retries + 1):
        print(f"  Check {attempt}/{max_retries}:")
        if not check_func(name):
            print(f"  [OK] ✅ '{name}' removed")
            return True
        time.sleep(delay)
    print(f"  [!] '{name}' still exists after {max_retries * delay}s")
    return False


# ---------------------------------------------------------------------------
# Remove steps
# ---------------------------------------------------------------------------

def undeploy_agent(agent_name: str) -> None:
    """Remove the native agent from watsonx Orchestrate."""
    print(f"\n🚫 AGENT: {agent_name}", flush=True)
    if check_agent_exists(agent_name):
        print("\n🔥 Removing agent...")
        result = run_command(
            ["orchestrate", "agents", "remove", "--name", agent_name, "--kind", "native"],
            check=False,
        )
        if result:
            print(f"  Exit code: {result.returncode}")
        wait_for_cleanup(agent_name, check_agent_exists, "AGENT")
    else:
        print("  [OK] Already gone")


def undeploy_toolkit(toolkit_name: str) -> None:
    """Remove the MCP toolkit from watsonx Orchestrate."""
    print(f"\n🚫 TOOLKIT: {toolkit_name}", flush=True)
    if check_toolkit_exists(toolkit_name):
        print("\n🔥 Removing toolkit...")
        result = run_command(
            ["orchestrate", "toolkits", "remove", "-n", toolkit_name],
            check=False,
        )
        if result:
            print(f"  Exit code: {result.returncode}")
            if result.returncode == 0:
                wait_for_cleanup(toolkit_name, check_toolkit_exists, "TOOLKIT")
            else:
                print("  [!] Remove command failed")
    else:
        print("  [OK] Already gone")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def final_status(toolkit_name: str, agent_name: str) -> None:
    """Print a final pass/fail summary and exit with the appropriate code."""
    print("\n" + "=" * 60)
    print("🏁 FINAL STATUS")
    print("=" * 60)

    print("\n🔍 AGENTS:")
    agent_gone   = not check_agent_exists(agent_name)
    print("\n🔍 TOOLKITS:")
    toolkit_gone = not check_toolkit_exists(toolkit_name)

    print("\n📊 RESULT:")
    print(f"  TOOLKIT '{toolkit_name}': {'✅ CLEAN' if toolkit_gone else '❌ EXISTS'}")
    print(f"  AGENT   '{agent_name}':   {'✅ CLEAN' if agent_gone   else '❌ EXISTS'}")

    if toolkit_gone and agent_gone:
        print("\n🎉 CLEANUP SUCCESS!")
        print("=" * 60)
        sys.exit(0)

    print("\n⚠  INCOMPLETE — some resources could not be removed")
    print("=" * 60)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        print("🗑️  WATSONX ORCHESTRATE CLEANUP")
        print("=" * 60)

        load_env()
        check_orchestrate_cli()
        ensure_environment()

        toolkit_name = os.getenv("WXO_TOOLKIT_NAME")
        agent_name   = os.getenv("WXO_AGENT_NAME")

        print(f"\n🎯 Targets:")
        print(f"   Agent:   '{agent_name}'")
        print(f"   Toolkit: '{toolkit_name}'")

        undeploy_agent(agent_name)
        undeploy_toolkit(toolkit_name)
        final_status(toolkit_name, agent_name)

    except KeyboardInterrupt:
        print("\n[!] Cleanup cancelled by user")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[X] Unexpected error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with IBM Bob
