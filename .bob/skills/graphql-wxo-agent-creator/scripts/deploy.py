#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
watsonx Orchestrate Agent Deploy Script
Deploys the GraphQL MCP toolkit and native agent to watsonx Orchestrate.
Does NOT create files or folders - only deploys existing schema and agent.yaml.
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

# Authentication type constants
# ibm_iam  — IBM Cloud tenants (default)
# mcsp_v2  — AWS / Azure trial tenants
#            The CLI infers the IAM endpoint itself via infer_mcspv2_iam_url()
#            (https://account-iam.platform.saas.ibm.com for AWS regions such
#            as eu-central-1, confirmed working; a special-cased host for
#            Germany West Central). WXO_IAM_URL / --iam-url should only be
#            set to override that for a region the CLI genuinely gets wrong —
#            never default it to a guessed host.
WXO_AUTH_TYPE_IBM_IAM   = "ibm_iam"
WXO_AUTH_TYPE_MCSP_V2   = "mcsp_v2"
VALID_AUTH_TYPES        = {WXO_AUTH_TYPE_IBM_IAM, WXO_AUTH_TYPE_MCSP_V2}

# Combined list
REQUIRED_VARS = REQUIRED_VARS_STEPZEN + REQUIRED_VARS_WXO

VALID_AGENT_KINDS  = {"native", "external", "assistant"}
# react_core is the current default (watsonx Orchestrate 2.13.0+).
# default/react/planner are kept for backward compatibility but are deprecated.
VALID_AGENT_STYLES = {"react_core", "default", "react", "planner"}
DEPRECATED_AGENT_STYLES = {"default", "react", "planner"}

# ---------------------------------------------------------------------------
# Path resolution
# This script lives at .bob/skills/graphql-wxo-agent-creator/scripts/deploy.py
# so the project root is four levels up.
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parents[3]
GRAPHQL_DIR  = PROJECT_ROOT / "api-connect" / "graphql"
WATSONX_DIR  = PROJECT_ROOT / "watsonx"


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
                        cwd: Path | None = None) -> '_Result | None':
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
# Env / file loading
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
    print("    IBM Cloud:  https://cloud.ibm.com/iam/apikeys", file=sys.stderr)
    print("    AWS trial:  same Settings -> API details tab, click 'Generate API key'.", file=sys.stderr)
    print("                Copy it immediately — it cannot be viewed again.", file=sys.stderr)
    print(file=sys.stderr)
    print("  WXO_ENV_NAME — the environment / resource name of your Orchestrate instance.", file=sys.stderr)
    print("    IBM Cloud: visible in the resource list under AI / Machine Learning:", file=sys.stderr)
    print("               https://cloud.ibm.com/resources", file=sys.stderr)
    print("    AWS trial: any short name you choose (used only locally).", file=sys.stderr)
    print(file=sys.stderr)
    print("  WXO_AUTH_TYPE — authentication mechanism (optional, default: ibm_iam).", file=sys.stderr)
    print("    IBM Cloud:  WXO_AUTH_TYPE=ibm_iam  (default — no further settings needed)", file=sys.stderr)
    print("    AWS trial:  WXO_AUTH_TYPE=mcsp_v2", file=sys.stderr)
    print(file=sys.stderr)
    print("  WXO_IAM_URL — MCSP IAM endpoint override (optional, mcsp_v2 only).", file=sys.stderr)
    print("    Leave unset for AWS trial tenants — the orchestrate CLI infers the", file=sys.stderr)
    print("    correct host (https://account-iam.platform.saas.ibm.com) itself.", file=sys.stderr)
    print("    Only set this if activation reports the wrong IAM host for your region.", file=sys.stderr)
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

    # Validate WXO_AUTH_TYPE
    auth_type = os.getenv("WXO_AUTH_TYPE", WXO_AUTH_TYPE_IBM_IAM).strip()
    if auth_type not in VALID_AUTH_TYPES:
        print(f"[X] Invalid WXO_AUTH_TYPE: '{auth_type}'. "
              f"Allowed: {', '.join(sorted(VALID_AUTH_TYPES))}", file=sys.stderr)
        print("    IBM Cloud tenants  → WXO_AUTH_TYPE=ibm_iam  (default)",
              file=sys.stderr)
        print("    AWS/Azure trial    → WXO_AUTH_TYPE=mcsp_v2",
              file=sys.stderr)
        sys.exit(1)

    # Validate enumerated variables
    agent_kind = os.getenv("WXO_AGENT_KIND", "native").strip()
    if agent_kind not in VALID_AGENT_KINDS:
        print(f"[X] Invalid WXO_AGENT_KIND: '{agent_kind}'. "
              f"Allowed: {', '.join(sorted(VALID_AGENT_KINDS))}", file=sys.stderr)
        sys.exit(1)

    agent_style = os.getenv("WXO_AGENT_STYLE", "react_core").strip()
    if agent_style not in VALID_AGENT_STYLES:
        print(f"[X] Invalid WXO_AGENT_STYLE: '{agent_style}'. "
              f"Allowed: {', '.join(sorted(VALID_AGENT_STYLES))}", file=sys.stderr)
        sys.exit(1)
    if agent_style in DEPRECATED_AGENT_STYLES:
        print(f"[!] WXO_AGENT_STYLE='{agent_style}' is deprecated. "
              f"Update to 'react_core' to avoid future issues.", file=sys.stderr)
        print("    Migration guide: https://www.ibm.com/docs/SSAVQO/agent_builder/agent-style-migration.html",
              file=sys.stderr)

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


def verify_required_files() -> None:
    """Verify all files that must exist before deployment."""
    print("\n[*] Verifying required files...")

    required = [
        ("watsonx/agent.yaml",                     WATSONX_DIR / "agent.yaml"),
        ("api-connect/graphql/stepzen.config.json", GRAPHQL_DIR / "stepzen.config.json"),
    ]
    missing = []
    for label, path in required:
        if path.exists():
            print(f"  ✓ Found: {label}")
        else:
            print(f"  ❌ Missing: {label}")
            missing.append(label)

    if missing:
        print(f"\n[X] Cannot deploy — {len(missing)} required file(s) missing",
              file=sys.stderr)
        if "watsonx/agent.yaml" in missing:
            print("    Run the graphql-wxo-agent-creator skill first to generate agent.yaml",
                  file=sys.stderr)
        sys.exit(1)

    print("✅ All required files present")


def get_mcp_url() -> str:
    """Build the MCP endpoint URL from STEPZEN_ENVIRONMENT + STEPZEN_DOMAIN + stepzen.config.json."""
    config_path = GRAPHQL_DIR / "stepzen.config.json"
    try:
        config   = json.loads(config_path.read_text(encoding="utf-8"))
        endpoint = config.get("endpoint", "")
        if not endpoint:
            print("[X] stepzen.config.json has no 'endpoint' field", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        print(f"[X] Cannot read stepzen.config.json: {exc}", file=sys.stderr)
        sys.exit(1)

    environment = os.getenv("STEPZEN_ENVIRONMENT", "").strip()
    domain      = os.getenv("STEPZEN_DOMAIN", "").strip()
    mcp_url     = f"https://{environment}.{domain}/{endpoint}/mcp"
    print(f"[i] MCP URL: {mcp_url}")
    return mcp_url


# ---------------------------------------------------------------------------
# Deployment steps
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
    still fresh — the real cost we want to avoid is calling it needlessly.
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
        if any(kw in output.lower() for kw in ("unauthorized", "forbidden", "401", "403", "invalid",
                                                "ibm_iam", "mcsp", "overriding", "token")):
            print_wxo_auth_hint()
    return False


def ensure_environment() -> None:
    """Add (if missing) and activate the watsonx Orchestrate environment.

    Supports two authentication types controlled by WXO_AUTH_TYPE (.env):
      ibm_iam  — IBM Cloud tenants (default)
      mcsp_v2  — AWS / Azure trial tenants

    The auth type is declared via --type on 'env add' only.
    'env activate' accepts only --api-key; the stored env config carries
    the auth type, so no flag is needed at activation time.

    Validity is judged purely from 'orchestrate env list': if WXO_ENV_NAME
    is already the environment marked '(active)', it is trusted as-is and
    no login call is made at all — this is what makes repeat deploys fast.
    If a *different* environment is active, none is active, or WXO_ENV_NAME
    isn't registered yet, we never just reuse it blindly — 'env activate'
    is run for real, so a genuinely invalid/expired session always gets a
    real login. If that activation fails for an entry that already existed
    (e.g. it was registered under the wrong auth type, or against an old
    WXO_INSTANCE_URL), it is removed and re-added once, then retried — so a
    stale registration self-heals instead of failing silently or forever.

    WXO_IAM_URL is only passed through (via --iam-url on 'env add') when the
    user explicitly sets it. It must NOT default to a guessed host: the
    orchestrate CLI's own infer_mcspv2_iam_url() already resolves the
    correct MCSP v2 IAM host per WXO region (https://account-iam.platform.
    saas.ibm.com for AWS regions such as eu-central-1, confirmed working;
    a special-cased host for Germany West Central) — overriding that with
    a wrong guess (a previous version of this script defaulted to
    https://iam.platform.saas.ibm.com, which 404s) breaks activation with a
    cryptic 'Error getting MCSP_V2 Token' / HTML 404 body.
    """
    print("\n🏗️  Environment...")
    env_name     = os.getenv("WXO_ENV_NAME")
    instance_url = os.getenv("WXO_INSTANCE_URL")
    api_key      = os.getenv("WXO_IBM_CLOUD_API_KEY")
    auth_type    = os.getenv("WXO_AUTH_TYPE", WXO_AUTH_TYPE_IBM_IAM).strip()
    iam_url      = os.getenv("WXO_IAM_URL", "").strip() or None

    print(f"  Auth type: {auth_type}")
    if auth_type == WXO_AUTH_TYPE_MCSP_V2:
        print(f"  IAM URL:   {iam_url or '(inferred automatically by the orchestrate CLI)'}")

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
        print(f"  ✅ Environment '{env_name}' added ({auth_type})")

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


def create_toolkit(mcp_url: str) -> None:
    """Create the MCP toolkit in watsonx Orchestrate if it does not yet exist."""
    print("\n🔧 Managing toolkit...")
    toolkit_name    = os.getenv("WXO_TOOLKIT_NAME")
    connection_name = os.getenv("WXO_CONNECTION_NAME")

    # Read tool names from agent.yaml (single source of truth)
    agent_yaml_path = WATSONX_DIR / "agent.yaml"
    try:
        import yaml
        agent_config = yaml.safe_load(agent_yaml_path.read_text(encoding='utf-8'))
    except Exception as exc:
        print(f"[X] Cannot read agent.yaml: {exc}", file=sys.stderr)
        sys.exit(1)

    tools_list = agent_config.get("tools", [])
    if not tools_list:
        print("[X] No tools found in agent.yaml", file=sys.stderr)
        sys.exit(1)

    tool_names = list(dict.fromkeys(
        tool.split(":")[-1] for tool in tools_list if ":" in tool
    ))
    if not tool_names:
        print("[X] No valid tool names found in agent.yaml (expected 'toolkit:name' format)",
              file=sys.stderr)
        sys.exit(1)

    tools_param = ",".join(tool_names)
    print(f"  MCP URL:             {mcp_url}")
    print(f"  Connection (app_id): '{connection_name}'")
    print(f"  Tools ({len(tool_names)}):")
    for name in tool_names:
        print(f"     - {name}")

    # Verify the connection exists
    print("  Checking connection presence...", flush=True)
    result = run_command_capture(["orchestrate", "connections", "list", "--verbose"])
    if not result or not result.stdout:
        print("[X] Cannot list connections — check environment and credentials.",
              file=sys.stderr)
        sys.exit(1)
    if connection_name and connection_name not in result.stdout:
        print(f"[X] Connection '{connection_name}' not found in this environment.",
              file=sys.stderr)
        print("    Create the gateway/connection in the UI or via CLI first, "
              "then re-run this script.", file=sys.stderr)
        sys.exit(1)

    # Check whether the toolkit already exists
    result = run_command_capture(["orchestrate", "toolkits", "list"])
    if result and toolkit_name and toolkit_name in (result.stdout or ""):
        print(f"  ✅ Toolkit '{toolkit_name}' already exists — reusing it")
        return

    # Create toolkit
    print(f"  Creating toolkit '{toolkit_name}'...")
    run_command([
        "orchestrate", "toolkits", "add",
        "--kind", "mcp",
        "--name", toolkit_name,
        "--description", "auto_generated_via_IBM_BOB",
        "--url", mcp_url,
        "--transport", "streamable_http",
        "--app-id", connection_name,
        "--tools", tools_param,
    ])
    print(f"  ✅ Toolkit '{toolkit_name}' created")

    print("  ⏳ Waiting for tool discovery (15 s)...")
    time.sleep(15)

    result = run_command_capture(
        ["orchestrate", "tools", "list", "--toolkit", toolkit_name]
    )
    if result and result.stdout:
        tool_count = max(0, result.stdout.count('\n') - 3)
        print(f"  ✅ Found {tool_count} tools in toolkit")
    else:
        print("  ⚠  Could not verify tools — proceeding anyway")


def deploy_agent() -> None:
    """Import and deploy the watsonx Orchestrate agent."""
    print("\n🚀 Deploying agent...")
    agent_file = WATSONX_DIR / "agent.yaml"

    try:
        import yaml
        agent_config = yaml.safe_load(agent_file.read_text(encoding='utf-8'))
        agent_name   = agent_config.get("name") or os.getenv("WXO_AGENT_NAME")
    except Exception as exc:
        print(f"[X] Cannot read agent.yaml: {exc}", file=sys.stderr)
        sys.exit(1)

    run_command(["orchestrate", "agents", "import", "--file", str(agent_file)])
    run_command(["orchestrate", "agents", "deploy", "--name", agent_name])
    print(f"  ✅ Agent '{agent_name}' deployed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        print("🚀 WATSONX ORCHESTRATE DEPLOY")
        print("=" * 60)

        load_env()
        check_orchestrate_cli()

        if not GRAPHQL_DIR.exists():
            print(f"\n[X] GraphQL directory not found: {GRAPHQL_DIR}", file=sys.stderr)
            sys.exit(1)
        if not WATSONX_DIR.exists():
            print(f"\n[X] watsonx directory not found: {WATSONX_DIR}", file=sys.stderr)
            print("    Run the graphql-wxo-agent-creator skill first.", file=sys.stderr)
            sys.exit(1)

        verify_required_files()

        mcp_url      = get_mcp_url()
        toolkit_name = os.getenv("WXO_TOOLKIT_NAME")
        agent_name   = os.getenv("WXO_AGENT_NAME")

        print(f"\n🎯 Deploying:")
        print(f"   Toolkit: '{toolkit_name}'")
        print(f"   Agent:   '{agent_name}'")
        print(f"   MCP URL: '{mcp_url}'")

        ensure_environment()
        create_toolkit(mcp_url)
        deploy_agent()

        print("\n" + "=" * 60)
        print("🎉 DEPLOYMENT COMPLETE!")
        print("=" * 60)
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n[!] Deployment cancelled by user")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[X] Unexpected error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with IBM Bob
