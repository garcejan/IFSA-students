---
name: graphql-wxo-agent-creator
description: >
  Generates a watsonx Orchestrate native-agent configuration (watsonx/agent.yaml) from GraphQL
  @tool definitions and environment variables, and deploys the MCP toolkit and agent to watsonx
  Orchestrate. Triggers on: create agent yaml, generate agent yaml, create watsonx agent,
  configure watsonx agent, deploy watsonx agent, deploy agent, create agent configuration,
  init watsonx project, scaffold watsonx project, graphql-wxo-agent-creator.
---

# watsonx Orchestrate Agent Configuration Guide

This skill generates a watsonx Orchestrate native-agent configuration file at `watsonx/agent.yaml`
and provides scripts to deploy it together with the GraphQL MCP toolkit.

It uses:

- Environment variables for agent metadata and behavior (loaded via `set-env.py`)
- Tool definitions from `api-connect/graphql/tools/{input-filename}.graphql` to derive tool names

## Scope

This skill:

- Creates or overwrites `watsonx/agent.yaml`
- Extracts tool names from `@tool` directives in `tools/{input-filename}.graphql`
- Converts those tool names into agent tool names with toolkit prefix
- Writes a valid YAML agent definition
- Deploys or removes the MCP toolkit and agent via `watsonx/deploy.py` / `watsonx/undeploy.py`

This skill does **not**:

- Create or update GraphQL schema or tools files (that is `graphql-api-creator`)
- Validate that tool names already exist in the target environment before generating YAML

---

## 0. Sensitive Data Handling (Mandatory)

**Bob MUST NEVER display literal secret values in chat output.** This applies to any value read
from `.env`, environment variables, or any other credential source.

When Bob reads a secret value from any config file and needs to reference it in chat, replace the
middle portion with asterisks, keeping at most the first 4 and last 4 characters visible:

```
71e83fc2-8bc6-420f-aed2-af1aadf6bc55  →  71e8****bc55
```

If the value is shorter than 8 characters, replace it entirely with `****`.

---

## 1. Project Initialization

### 1.0 Intent Recognition — "init" trigger

If the user's message contains any of the following intents (case-insensitive),
treat it as an **init request** and follow section 1.1 exclusively:

- "init the watsonx project"
- "init watsonx project"
- "initialize the watsonx project"
- "set up the watsonx project"
- "scaffold the watsonx project"
- "create the watsonx project"
- "bootstrap the watsonx project"
- "new watsonx project"

**Enforcement:** When an init intent is detected, the ONLY action Bob takes is
the scaffold procedure in section 1.1. Do NOT proceed to agent YAML generation
or deployment until the user explicitly asks for those steps.

---

### 1.1 Scaffold procedure (init path)

**Step 1 — Check if the project already exists:**

Check whether `watsonx/agent.yaml` exists at the project root.
If it does, tell the user:

> "The watsonx agent project is already initialised at `watsonx/`.
> Say **'Bob, create the agent YAML'** to regenerate `agent.yaml` from
> the current environment variables and GraphQL tools file."

Then STOP — do not re-scaffold.

**Step 2 — Check graphql-api-creator project exists:**

Check whether `api-connect/graphql/` exists. If it does **not**, tell the user:

> "The GraphQL project must be initialised first. Say **'Bob, init the GraphQL project'**
> to set it up, then come back to initialise the watsonx agent project."

Then STOP.

**Step 3 — Run the scaffold script:**

```
python .bob/skills/graphql-wxo-agent-creator/scripts/scaffold_project.py
```

**Step 4 — Confirm and guide next step:**

Tell the user:

> "watsonx agent project initialised successfully. Here is what was created:
>
> ```
> watsonx/
>   agent.yaml          <- edit or regenerate with 'Bob, create the agent YAML'
>   deploy.py           <- run after agent.yaml is ready
>   undeploy.py         <- removes agent and toolkit from watsonx Orchestrate
> ```
>
> **Next steps:**
> 1. Ensure your `.env` has all `WXO_*` variables set (see README.md)
> 2. Load env vars: `python set-env.py | Invoke-Expression` (PowerShell)
>    or `source <(python3 set-env.py)` (bash/zsh)
> 3. Say **'Bob, create the agent YAML'** to generate `watsonx/agent.yaml`
> 4. Then run `python watsonx/deploy.py` to deploy"

---

### 1.2 Normal path (agent YAML generation)

When the user asks to create or regenerate the agent YAML, follow this sequence:

- Check that `set-env.py` exists at the project root; stop with a clear error if missing
- Validate all required environment variables are present in the process environment
- Determine input filename from `api-connect/graphql/input/` directory
- Extract tool names from `api-connect/graphql/tools/{input-filename}.graphql`
- Generate `watsonx/agent.yaml` using the PowerShell script in §9 (Approach A)
  or the direct file creation approach in §10 (Approach B)

---

## 2. Implementation Approaches

### Approach A: PowerShell Script (Recommended)
- Full-featured with validation and error handling
- Requires PowerShell execution permissions
- See §9 for complete implementation

### Approach B: Direct File Creation (Fallback)
- Works when PowerShell execution is restricted
- Avoids shell escaping issues
- See §10 for implementation pattern

Choose Approach B when:
- PowerShell execution is denied or restricted
- Shell escaping becomes problematic
- Cross-platform compatibility is critical

---

## 3. Prerequisites

Required:

- `set-env.py` exists in the project root
- Environment variables are loaded in the current session by running `set-env.py` before this skill
- GraphQL tools file exists at `api-connect/graphql/tools/{input-filename}.graphql`
- PowerShell 5.1 or later (for Approach A)
- Input filename can be determined from the `input/` directory
- Write access to the `watsonx/` folder

### How to Load Environment Variables

**Windows PowerShell:**
```powershell
python set-env.py | Invoke-Expression
```

**All platforms (bash / zsh / Git Bash / WSL):**
```bash
source <(python3 set-env.py)
```

> If `set-env.py` does not exist in the project root, stop immediately and tell
> the user to run `python .bob/skills/graphql-api-creator/scripts/scaffold_project.py` first.

### Upfront Environment Variable Validation

**Before doing any work**, verify:
1. `set-env.py` exists in the project root
2. All required variables are present and non-empty in the process environment

If either check fails, stop immediately and report what is missing.

```powershell
if (-not (Test-Path "set-env.py")) {
    throw "set-env.py not found. Run the graphql-api-creator scaffold first."
}
$missing = @()
foreach ($var in @("WXO_AGENT_NAME","WXO_AGENT_DESCRIPTION","WXO_AGENT_MODEL","WXO_TOOLKIT_NAME")) {
    if ([string]::IsNullOrWhiteSpace([System.Environment]::GetEnvironmentVariable($var, "Process"))) {
        $missing += $var
    }
}
if ($missing.Count -gt 0) {
    throw "Missing required environment variables: $($missing -join ', ').`nRun: python set-env.py | Invoke-Expression"
}
```

---

## 4. Environment Variables

### 4.1 Required variables

- `WXO_AGENT_NAME`
- `WXO_AGENT_DESCRIPTION`
- `WXO_AGENT_MODEL`
- `WXO_TOOLKIT_NAME`

### 4.2 Optional variables with defaults

- `WXO_AGENT_KIND=native` (one of: `native`, `external`, `assistant`)
- `WXO_AGENT_STYLE=react_core` (one of: `react_core`, `default`, `react`, `planner` — `default`/`react`/`planner` are deprecated since 2.13.0, see [migration guide](https://www.ibm.com/docs/SSAVQO/agent_builder/agent-style-migration.html))
- `WXO_AGENT_INSTRUCTIONS=` (empty string)
- `WXO_AGENT_HIDE_REASONING=false`
- `WXO_CONTEXT_ACCESS_ENABLED=false`
- `WXO_CONTEXT_VARIABLES=` (comma-separated list)
- `WXO_CONNECTION_NAME=` (when set, written into `connections.global_requirements.required_app_ids`)

### 4.3 Additional variables required by deploy.py

These are also required when running `python watsonx/deploy.py`:

- `STEPZEN_ENVIRONMENT` — environment name from the StepZen login page
- `STEPZEN_DOMAIN` — domain name from the StepZen login page
  (The MCP/Query endpoint base URL is `https://<STEPZEN_ENVIRONMENT>.<STEPZEN_DOMAIN>/`)
- `WXO_ENV_NAME` — the **exact resource name** of your watsonx Orchestrate instance as shown in the [IBM Cloud resource list](https://cloud.ibm.com/resources) under **AI / Machine Learning**
- `WXO_INSTANCE_URL` — watsonx Orchestrate instance URL
- `WXO_IBM_CLOUD_API_KEY` — IBM Cloud API key

#### AWS/Azure trial tenants (`WXO_AUTH_TYPE=mcsp_v2`)

The values above assume an IBM Cloud tenant (`WXO_AUTH_TYPE=ibm_iam`, the
default when the variable is unset). For an AWS/Azure trial tenant, set
`WXO_AUTH_TYPE=mcsp_v2` in `.env` and retrieve the values differently:

- `WXO_ENV_NAME` — any short name you choose (used only locally)
- `WXO_INSTANCE_URL` — in the watsonx Orchestrate UI: User icon (top-right) →
  Settings → API details tab → **Service instance URL**
- `WXO_IBM_CLOUD_API_KEY` — same API details tab → **Generate API key**
  (copy immediately — it cannot be viewed again)
- `WXO_IAM_URL` — optional override, rarely needed. The orchestrate CLI
  infers the correct MCSP v2 IAM host itself
  (`https://account-iam.platform.saas.ibm.com` for AWS regions such as
  eu-central-1). Only set this if activation reports the wrong IAM host.

No tenant yet?
- Free trial, no credit card (AWS/Azure): https://www.ibm.com/account/reg/signup?formid=urx-54418
- IBM Cloud trial, credit card required: https://www.ibm.com/account/reg/us-en/signup?formid=urx-54418

### 4.4 Tool naming behavior

Tool names are generated from GraphQL operation names with a **required toolkit prefix**:

- Format: `<WXO_TOOLKIT_NAME>:<operationName>`
- The delimiter `:` is fixed
- `WXO_TOOLKIT_NAME` must not be empty

---

## 5. File Handling Rules

- Ensure the `watsonx/` directory exists before writing
- Overwrite `watsonx/agent.yaml` deterministically on every run
- Do not delete unrelated files
- Do not modify `.env` or environment variables
- Do not modify the GraphQL schema or tools files

---

## 6. Tool Extraction Rules

Read:

- `api-connect/graphql/tools/{input-filename}.graphql`
  (where `{input-filename}` is the base name of the file in `input/`)

Extract tool names from `@tool` directives:

- Each `extend schema @tool(...)` directive defines one tool
- Extract the `name:` attribute value using pattern: `name:\s*"([^"]+)"`
- Preserve source order; remove duplicates (keep first occurrence)
- Ignore comments

### 6.1 Example

Input:
```graphql
extend schema
  @tool(
    name: "getAccountDetails"
    description: "..."
  )
extend schema
  @tool(
    name: "searchAccountsByName"
    description: "..."
  )
```

Extracted tool names → prefixed with `WXO_TOOLKIT_NAME=my_toolkit`:
- `my_toolkit:getAccountDetails`
- `my_toolkit:searchAccountsByName`

---

## 7. Agent YAML Structure

```yaml
spec_version: v1
kind: native
name: my_agent
description: |
  My agent description
instructions: |
  My instructions
llm: watsonx/ibm/granite-4-h-small
style: default
hide_reasoning: false
guidelines: []
collaborators: []
connections:                          # only when WXO_CONNECTION_NAME is non-empty
  global_requirements:
    required_app_ids:
      - my-connection-name
context_access_enabled: true          # only when WXO_CONTEXT_ACCESS_ENABLED=true
context_variables:                    # only when context is enabled
  - variable_name
tools:
  - my_toolkit:getAccountDetails
knowledge_base: []
```

### 7.1 Required fields

Always generate: `spec_version`, `kind`, `name`, `description`, `instructions`,
`llm`, `style`, `hide_reasoning`, `guidelines`, `collaborators`, `tools`, `knowledge_base`.

### 7.2 Optional connections block

Generate only when `WXO_CONNECTION_NAME` is non-empty:
```yaml
connections:
  global_requirements:
    required_app_ids:
      - {WXO_CONNECTION_NAME}
```

Place after `collaborators`, before `tools`. Omit entirely when empty.

### 7.3 Optional context section

Generate only when `WXO_CONTEXT_ACCESS_ENABLED=true`:
```yaml
context_access_enabled: true
context_variables:
  - {var}
```

---

## 8. Validation Rules

Before writing `agent.yaml`:

- `WXO_AGENT_NAME`, `WXO_AGENT_DESCRIPTION`, `WXO_AGENT_MODEL`, `WXO_TOOLKIT_NAME` — non-empty
- `WXO_AGENT_KIND` ∈ {`native`, `external`, `assistant`}
- `WXO_AGENT_STYLE` ∈ {`react_core`, `default`, `react`, `planner`} — `react_core` is the current default; `default`/`react`/`planner` are deprecated
- `WXO_AGENT_HIDE_REASONING` ∈ {`true`, `false`}
- `WXO_CONTEXT_ACCESS_ENABLED` ∈ {`true`, `false`}
- Tools file must exist and contain at least one `@tool` directive
- If `WXO_CONTEXT_ACCESS_ENABLED=true`, `WXO_CONTEXT_VARIABLES` must be non-empty

---

## 9. PowerShell Implementation (Approach A)

Use this script:

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path $Path)) { New-Item -ItemType Directory -Path $Path -Force | Out-Null }
}

function ConvertTo-Bool {
    param([string]$Value, [string]$VariableName)
    switch ($Value.Trim().ToLowerInvariant()) {
        "true"  { return $true }
        "false" { return $false }
        default { throw "Invalid boolean value for $VariableName: '$Value'. Use true or false." }
    }
}

function ConvertTo-YamlBlock {
    param([string]$Text, [int]$Indent = 2)
    $padding = " " * $Indent
    if ([string]::IsNullOrEmpty($Text)) { return "$padding" }
    $normalized = $Text -replace "`r`n", "`n" -replace "`r", "`n"
    return ($normalized -split "`n" | ForEach-Object { "$padding$_" }) -join "`n"
}

function Validate-EnvLoaded {
    if (-not (Test-Path "set-env.py")) {
        throw "set-env.py not found. Run the graphql-api-creator scaffold first."
    }
    $required = @("WXO_AGENT_NAME", "WXO_AGENT_DESCRIPTION", "WXO_AGENT_MODEL", "WXO_TOOLKIT_NAME")
    $missing  = $required | Where-Object {
        [string]::IsNullOrWhiteSpace([System.Environment]::GetEnvironmentVariable($_, "Process"))
    }
    if ($missing) {
        throw "Missing required environment variables: $($missing -join ', ').`nRun: python set-env.py | Invoke-Expression"
    }
}

function Get-EnvOrDefault { param([string]$Name, [string]$Default = "")
    $v = [System.Environment]::GetEnvironmentVariable($Name, "Process")
    if ($null -eq $v -or $v -eq "") { return $Default }
    return $v
}

function Get-RequiredEnv { param([string]$Name)
    $v = [System.Environment]::GetEnvironmentVariable($Name, "Process")
    if ($null -eq $v -or $v.Trim() -eq "") { throw "Required variable '$Name' is missing." }
    return $v.Trim()
}

function Validate-Enum { param([string]$Name, [string]$Value, [string[]]$Allowed)
    if ($Allowed -notcontains $Value) {
        throw "Invalid value for $Name: '$Value'. Allowed: $($Allowed -join ', ')"
    }
}

function Get-ToolNamesFromGraphQL { param([string]$ToolsFilePath)
    if (-not (Test-Path $ToolsFilePath)) {
        throw "GraphQL tools file not found: '$ToolsFilePath'"
    }
    $toolsText = Get-Content $ToolsFilePath -Raw
    $matches   = [regex]::Matches($toolsText, 'name:\s*"([^"]+)"')
    $names = New-Object System.Collections.Generic.List[string]
    $seen  = @{}
    foreach ($m in $matches) {
        $n = $m.Groups[1].Value
        if (-not $seen.ContainsKey($n)) { $seen[$n] = $true; $names.Add($n) }
    }
    if ($names.Count -eq 0) {
        throw "No @tool directives with name attributes found in '$ToolsFilePath'"
    }
    return $names
}

function Get-ContextVariables { param([string]$RawValue)
    if ([string]::IsNullOrWhiteSpace($RawValue)) { return @() }
    return @($RawValue.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
}

function Build-ToolNames { param([System.Collections.Generic.List[string]]$Names, [string]$Prefix)
    $tools = New-Object System.Collections.Generic.List[string]
    foreach ($n in $Names) {
        if ([string]::IsNullOrWhiteSpace($Prefix)) { $tools.Add($n) }
        else { $tools.Add("$Prefix`:$n") }
    }
    return $tools
}

function Write-AgentYaml { param([string]$OutputPath, [hashtable]$Agent)
    $yaml = New-Object System.Collections.Generic.List[string]
    $yaml.Add("# Generated by graphql-wxo-agent-creator skill")
    $yaml.Add("# Source variables: WXO_AGENT_NAME, WXO_AGENT_MODEL, WXO_TOOLKIT_NAME, WXO_CONNECTION_NAME")
    $yaml.Add("# Do not edit manually — re-run after: python set-env.py | Invoke-Expression")
    $yaml.Add("spec_version: v1")
    $yaml.Add("kind: $($Agent.kind)")
    $yaml.Add("name: $($Agent.name)")
    $yaml.Add("description: |")
    $yaml.Add((ConvertTo-YamlBlock -Text $Agent.description))
    $yaml.Add("instructions: |")
    $yaml.Add((ConvertTo-YamlBlock -Text $Agent.instructions))
    $yaml.Add("llm: $($Agent.llm)")
    $yaml.Add("style: $($Agent.style)")
    $yaml.Add("hide_reasoning: $($Agent.hide_reasoning.ToString().ToLowerInvariant())")
    $yaml.Add("guidelines: []")
    $yaml.Add("collaborators: []")
    if (-not [string]::IsNullOrWhiteSpace($Agent.app_id)) {
        $yaml.Add("connections:")
        $yaml.Add("  global_requirements:")
        $yaml.Add("    required_app_ids:")
        $yaml.Add("      - $($Agent.app_id)")
    }
    if ($Agent.context_access_enabled) {
        $yaml.Add("context_access_enabled: true")
        $yaml.Add("context_variables:")
        foreach ($var in $Agent.context_variables) { $yaml.Add("  - $var") }
    }
    if ($Agent.tools.Count -eq 0) { $yaml.Add("tools: []") }
    else {
        $yaml.Add("tools:")
        foreach ($t in $Agent.tools) { $yaml.Add("  - $t") }
    }
    $yaml.Add("knowledge_base: []")
    $content   = $yaml -join "`n"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($OutputPath, $content, $utf8NoBom)
}

Write-Host "Validating environment variables..." -ForegroundColor Cyan
Validate-EnvLoaded

$agentName        = Get-RequiredEnv "WXO_AGENT_NAME"
$agentDescription = Get-RequiredEnv "WXO_AGENT_DESCRIPTION"
$agentModel       = Get-RequiredEnv "WXO_AGENT_MODEL"
$toolkitName      = Get-RequiredEnv "WXO_TOOLKIT_NAME"

$agentKind           = Get-EnvOrDefault "WXO_AGENT_KIND"             "native"
$agentStyle          = Get-EnvOrDefault "WXO_AGENT_STYLE"            "react_core"
$agentInstructions   = Get-EnvOrDefault "WXO_AGENT_INSTRUCTIONS"     ""
$hideReasoningRaw    = Get-EnvOrDefault "WXO_AGENT_HIDE_REASONING"   "false"
$contextEnabledRaw   = Get-EnvOrDefault "WXO_CONTEXT_ACCESS_ENABLED" "false"
$contextVariablesRaw = Get-EnvOrDefault "WXO_CONTEXT_VARIABLES"      ""
$connectionName      = Get-EnvOrDefault "WXO_CONNECTION_NAME"        ""

Validate-Enum -Name "WXO_AGENT_KIND"  -Value $agentKind  -Allowed @("native", "external", "assistant")
Validate-Enum -Name "WXO_AGENT_STYLE" -Value $agentStyle -Allowed @("react_core", "default", "react", "planner")
if ($agentStyle -in @("default", "react", "planner")) {
    Write-Warning "WXO_AGENT_STYLE='$agentStyle' is deprecated. Update to 'react_core' to avoid future issues."
    Write-Warning "Migration guide: https://www.ibm.com/docs/SSAVQO/agent_builder/agent-style-migration.html"
}

$hideReasoning    = ConvertTo-Bool -Value $hideReasoningRaw  -VariableName "WXO_AGENT_HIDE_REASONING"
$contextEnabled   = ConvertTo-Bool -Value $contextEnabledRaw -VariableName "WXO_CONTEXT_ACCESS_ENABLED"
$contextVariables = Get-ContextVariables -RawValue $contextVariablesRaw

if ($contextEnabled -and $contextVariables.Count -eq 0) {
    throw "WXO_CONTEXT_ACCESS_ENABLED is true but WXO_CONTEXT_VARIABLES is empty."
}

# Determine input filename from input/ directory
$inputFiles = Get-ChildItem -Path "api-connect/graphql/input" -Filter "*.json","*.yaml","*.yml" -ErrorAction SilentlyContinue
if (-not $inputFiles -or $inputFiles.Count -eq 0) {
    throw "No input files found in api-connect/graphql/input/"
}
$inputBaseName = [System.IO.Path]::GetFileNameWithoutExtension($inputFiles[0].Name)

$toolsFilePath = "api-connect/graphql/tools/$inputBaseName.graphql"
$toolBaseNames = Get-ToolNamesFromGraphQL -ToolsFilePath $toolsFilePath
$toolNames     = Build-ToolNames -Names $toolBaseNames -Prefix $toolkitName

Ensure-Directory -Path "watsonx"

$agent = @{
    kind                   = $agentKind
    name                   = $agentName
    description            = $agentDescription
    instructions           = $agentInstructions
    llm                    = $agentModel
    style                  = $agentStyle
    hide_reasoning         = $hideReasoning
    app_id                 = $connectionName
    context_access_enabled = $contextEnabled
    context_variables      = $contextVariables
    tools                  = $toolNames
}

$outputPath = "watsonx/agent.yaml"
Write-AgentYaml -OutputPath $outputPath -Agent $agent

Write-Host "✅ Created $outputPath" -ForegroundColor Green
Write-Host ""
Write-Host "Generated tools:" -ForegroundColor Cyan
$toolNames | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }
```

---

## 10. Alternative: Direct File Creation (Approach B)

When PowerShell execution is restricted, Bob creates `watsonx/agent.yaml` directly.

### 10.1 Steps

```
1. Check set-env.py exists in the project root; abort if missing
2. Validate all required env vars are in the process environment; abort listing any missing
3. Determine input filename from api-connect/graphql/input/ directory
4. Read api-connect/graphql/tools/{input-filename}.graphql
5. Extract tool names using regex: name:\s*"([^"]+)"
6. Build tool names list: {WXO_TOOLKIT_NAME}:{toolName}
7. Construct YAML string with proper indentation
8. Write to watsonx/agent.yaml (UTF-8, no BOM)
```

### 10.2 YAML template

```yaml
# Generated by graphql-wxo-agent-creator skill
spec_version: v1
kind: {WXO_AGENT_KIND}
name: {WXO_AGENT_NAME}
description: |
  {WXO_AGENT_DESCRIPTION}
instructions: |
  {WXO_AGENT_INSTRUCTIONS}
llm: {WXO_AGENT_MODEL}
style: {WXO_AGENT_STYLE}
hide_reasoning: {WXO_AGENT_HIDE_REASONING}
guidelines: []
collaborators: []
# include connections block only when WXO_CONNECTION_NAME is non-empty:
connections:
  global_requirements:
    required_app_ids:
      - {WXO_CONNECTION_NAME}
# include context block only when WXO_CONTEXT_ACCESS_ENABLED=true:
context_access_enabled: true
context_variables:
  - {var1}
tools:
  - {WXO_TOOLKIT_NAME}:{tool_1}
  - {WXO_TOOLKIT_NAME}:{tool_2}
knowledge_base: []
```

---

## 11. Expected Output Examples

**With `WXO_TOOLKIT_NAME=my_toolkit` and `WXO_CONNECTION_NAME=my-connection`:**

```yaml
spec_version: v1
kind: native
name: customer_support_agent
description: |
  Answers account and customer questions.
instructions: |

llm: watsonx/ibm/granite-4-h-small
style: default
hide_reasoning: false
guidelines: []
collaborators: []
connections:
  global_requirements:
    required_app_ids:
      - my-connection
tools:
  - my_toolkit:getAccountDetails
  - my_toolkit:searchAccountsByName
knowledge_base: []
```

**With `WXO_CONNECTION_NAME` absent or empty (no connections block):**

```yaml
spec_version: v1
kind: native
name: customer_support_agent
description: |
  Answers account and customer questions.
instructions: |

llm: watsonx/ibm/granite-4-h-small
style: default
hide_reasoning: false
guidelines: []
collaborators: []
tools:
  - my_toolkit:getAccountDetails
  - my_toolkit:searchAccountsByName
knowledge_base: []
```

---

## 12. Deployment

After `agent.yaml` is generated:

```bash
python watsonx/deploy.py
```

This script:
1. Loads `.env` from the project root
2. Activates the watsonx Orchestrate environment
3. Creates the MCP toolkit (skips if it already exists)
4. Imports and deploys the agent from `watsonx/agent.yaml`

To remove all deployed resources:

```bash
python watsonx/undeploy.py
```

To import an agent manually:

```bash
orchestrate agents import -f watsonx/agent.yaml
```

---

## 13. Troubleshooting

### Problem: `agent.yaml` is not created

Check:
- `set-env.py` exists in the project root (created by `graphql-api-creator` init)
- All required env vars are loaded: `python set-env.py | Invoke-Expression`
- Input file exists in `api-connect/graphql/input/`
- `api-connect/graphql/tools/{input-filename}.graphql` exists and has `@tool` directives

### Problem: zero tools generated

Likely causes:
- Tools file has no `@tool` directives — run the `graphql-api-creator` skill first
- `@tool` directives are missing `name:` attributes

### Problem: tools exist in YAML but agent cannot use them

- Run `orchestrate tools list -v` and compare tool names
- Verify `WXO_TOOLKIT_NAME` matches the actual toolkit name in the environment

### Problem: deploy.py fails with "Connection not found"

- The connection named in `WXO_CONNECTION_NAME` must already exist in the environment
- Create it via the watsonx Orchestrate UI or CLI first

### Problem: invalid agent style or kind

- `WXO_AGENT_KIND` must be one of: `native`, `external`, `assistant`
- `WXO_AGENT_STYLE` must be one of: `react_core`, `default`, `react`, `planner`
  (`default`/`react`/`planner` still work but are deprecated — update to `react_core`)

---

## 14. Instructions for BOB

### Environment Setup (MANDATORY — do this first, every time)

Bob CANNOT read `.env` directly (it is gitignored). The only way to load environment
variables into the shell session is to run `set-env.py` via `execute_command`.

**Step 1 — Load all variables into the session (PowerShell):**

Use `execute_command` with exactly this command:

```
python set-env.py | Invoke-Expression
```

If `set-env.py` is not found at the project root, stop immediately and tell the user:
> "set-env.py is missing. Run the graphql-api-creator scaffold first:
> `python .bob/skills/graphql-api-creator/scripts/scaffold_project.py`"

**Step 2 — Verify all required variables are present:**

Use `execute_command` with exactly this command:

```
python set-env.py --check WXO_AGENT_NAME,WXO_AGENT_DESCRIPTION,WXO_AGENT_MODEL,WXO_TOOLKIT_NAME,WXO_CONNECTION_NAME,WXO_ENV_NAME,WXO_IBM_CLOUD_API_KEY,WXO_INSTANCE_URL
```

If the check reports any variable as missing, stop and tell the user exactly which
variables need to be added to `.env` before proceeding.

**CRITICAL:** Do NOT attempt to read `$env:WXO_*` variables directly in chat —
they will always appear empty unless `set-env.py | Invoke-Expression` has been
run first via `execute_command` in the same session.

### When generating agent.yaml

1. Run `python set-env.py | Invoke-Expression` via `execute_command` (Step 1 above)
2. Run `python set-env.py --check ...` via `execute_command` to verify vars (Step 2 above)
3. Only proceed after all required vars are confirmed present
4. Determine input filename from `api-connect/graphql/input/`
5. Extract tool names from `api-connect/graphql/tools/{input-filename}.graphql`
6. Build prefixed tool names: `{WXO_TOOLKIT_NAME}:{toolName}`
7. Generate `watsonx/agent.yaml` using Approach A (PowerShell) or Approach B (direct)
8. Confirm the file was written and show the generated tool list

### When in doubt

- **never display literal API keys, tokens, or secrets in chat** — obfuscate using §0
- **never assume WXO_TOOLKIT_NAME is empty** — it is always required
- **never overwrite agent.yaml without user confirmation** if it contains non-placeholder content
- **always read tool names from the tools graphql file** — never invent them

### After agent.yaml generation, suggest next steps

- **Deploy**: `python watsonx/deploy.py`
- **Verify tools**: `orchestrate tools list -v`
- **Import agent manually**: `orchestrate agents import -f watsonx/agent.yaml`

---

## 15. Further Documentation

- **watsonx Orchestrate CLI reference** — https://www.ibm.com/docs/en/watsonx/watson-orchestrate/current?topic=cli-agent-management
- **Importing agents** — https://www.ibm.com/docs/en/watsonx/watson-orchestrate/current?topic=agents-importing
- **MCP toolkit integration** — https://www.ibm.com/docs/en/watsonx/watson-orchestrate/current?topic=tools-model-context-protocol
