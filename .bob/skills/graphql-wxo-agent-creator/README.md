# graphql-wxo-agent-creator

Generates a watsonx Orchestrate native-agent configuration (`watsonx/agent.yaml`) from GraphQL tool definitions and environment variables, then deploys the MCP toolkit and agent to watsonx Orchestrate.

## What this skill does

1. Reads `@tool` directive names from `api-connect/graphql/tools/{input-filename}.graphql`
2. Prefixes each tool name with `WXO_TOOLKIT_NAME` to produce `toolkit:toolName` entries
3. Generates `watsonx/agent.yaml` with the full agent configuration
4. Provides `watsonx/deploy.py` and `watsonx/undeploy.py` wrappers for one-command deployment

## Project layout

```
<project-root>/
  .env                          ← credentials (never commit)
  set-env.py                    ← env-var loader
  api-connect/
    graphql/
      tools/{name}.graphql      ← input: @tool directives
      stepzen.config.json       ← input: endpoint name
  watsonx/
    agent.yaml                  ← output: generated agent config
    deploy.py                   ← wrapper → skill canonical deploy.py
    undeploy.py                 ← wrapper → skill canonical undeploy.py
```

## Prerequisites

- `graphql-api-creator` skill has been initialised and the GraphQL schema built
- `set-env.py` exists at the project root (created by `graphql-api-creator` init)
- `.env` is populated with all required variables (see below)
- `orchestrate` CLI is installed and on PATH
- `python-dotenv` and `pyyaml` are installed: `pip install python-dotenv pyyaml`

## Required environment variables

| Variable | Description |
|---|---|
| `STEPZEN_ENVIRONMENT` | Environment name from the StepZen login page |
| `STEPZEN_DOMAIN` | Domain name from the StepZen login page (endpoint base: `https://<env>.<domain>/`) |
| `WXO_ENV_NAME` | The **exact resource name** of your watsonx Orchestrate instance as shown in [IBM Cloud resource list](https://cloud.ibm.com/resources) under **AI / Machine Learning** |
| `WXO_INSTANCE_URL` | watsonx Orchestrate instance URL |
| `WXO_IBM_CLOUD_API_KEY` | IBM Cloud API key for environment activation |
| `WXO_TOOLKIT_NAME` | MCP toolkit name in watsonx Orchestrate |
| `WXO_CONNECTION_NAME` | Existing watsonx Orchestrate connection `app_id` |
| `WXO_AGENT_NAME` | Agent name |
| `WXO_AGENT_DESCRIPTION` | Agent description |
| `WXO_AGENT_MODEL` | LLM model identifier (e.g. `watsonx/ibm/granite-4-h-small`) |

## Optional environment variables

| Variable | Default | Description |
|---|---|---|
| `WXO_AGENT_KIND` | `native` | Agent kind: `native`, `external`, `assistant` |
| `WXO_AGENT_STYLE` | `react_core` | Agent style: `react_core` (recommended), `default`, `react`, `planner` — `default`/`react`/`planner` deprecated since 2.13.0 |
| `WXO_AGENT_INSTRUCTIONS` | _(empty)_ | Agent system instructions |
| `WXO_AGENT_HIDE_REASONING` | `false` | Hide reasoning steps |
| `WXO_CONTEXT_ACCESS_ENABLED` | `false` | Enable context variable access |
| `WXO_CONTEXT_VARIABLES` | _(empty)_ | Comma-separated context variable names |

## Usage

### 1. Scaffold (once per project)

The scaffold is run automatically when you init the `graphql-api-creator` project. To run it standalone:

```bash
python .bob/skills/graphql-wxo-agent-creator/scripts/scaffold_project.py
```

### 2. Load environment variables

```powershell
# PowerShell
python set-env.py | Invoke-Expression
```
```bash
# bash / zsh
source <(python3 set-env.py)
```

### 3. Generate agent.yaml

Ask Bob: **"Bob, create the agent YAML"**

Or run the PowerShell script from the SKILL.md directly.

### 4. Deploy

```bash
python watsonx/deploy.py
```

### 5. Undeploy

```bash
python watsonx/undeploy.py
```

## Scripts

| Script | Location | Purpose |
|---|---|---|
| `scaffold_project.py` | skill `scripts/` | Creates `watsonx/` dir and wrapper scripts |
| `deploy.py` | skill `scripts/` | Deploys toolkit + agent to watsonx Orchestrate |
| `undeploy.py` | skill `scripts/` | Removes agent and toolkit from watsonx Orchestrate |
| `watsonx/deploy.py` | project root | Thin wrapper → skill `deploy.py` |
| `watsonx/undeploy.py` | project root | Thin wrapper → skill `undeploy.py` |

## Related skills

- **`graphql-api-creator`** — builds the GraphQL schema and MCP tools that this skill reads
- **`graphql-field-optimization`** — cleans up field names before generating agent.yaml
- **`graphql-query-orchestrator`** — adds cross-resource queries before generating agent.yaml

<!-- Made with IBM Bob -->
