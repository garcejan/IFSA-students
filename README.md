# IFSA x IBM workshop: multi-agent teams on watsonx Orchestrate

Hands-on workshop for building, routing, testing and operating a multi-agent system on **IBM watsonx Orchestrate (wxO)**, driven from the **IBM Bob** IDE.

![`Instructors`](/img/instructors.png)

Over four parts you build a team of AI agents, deploy it to watsonx Orchestrate, test and evaluate it, embed it in a web app, and operate it with traces and dashboards.

**Start here: [`labs/README.md`](labs/README.md).** It has the architecture, the four lab parts, pre-flight setup, ground rules and the reference solution.

## Dependencies

| Dependency | Version | Used for |
|------------|---------|----------|
| [IBM Bob](https://bob.ibm.com) | Trial: 50 bobcoins for 30 days | The agentic IDE that you drive the labs from |
| [IBM watsonx Orchestrate](https://www.ibm.com/products/watsonx-orchestrate) | Trial: 30 days | The platform that hosts the agents. Your own instance, or one provisioned for you |
| [Python](https://www.python.org/) | 3.12 or later (pinned in [`.python-version`](.python-version)) | Tools, telemetry scripts and the `orchestrate` CLI |
| [uv](https://docs.astral.sh/uv/) | any recent version | Creating the virtual environment and installing packages |
| [`ibm-watsonx-orchestrate`](https://pypi.org/project/ibm-watsonx-orchestrate/) | 2.20.0 or later (locked to 2.20.0 in [`uv.lock`](uv.lock)) | watsonx Orchestrate ADK and the `orchestrate` CLI |
| [Node.js](https://nodejs.org/) and npm | Node.js 20 or later | Next.js web app in [`labs/ppth-web/`](labs/ppth-web/) (Part 2) |

Python dependencies are declared in [`pyproject.toml`](pyproject.toml). The web app's dependencies are in [`labs/ppth-web/package.json`](labs/ppth-web/package.json).

## Getting started

1. Clone this repository.
2. Install the Python dependencies:

   ```bash
   uv sync
   source .venv/bin/activate
   ```

3. Install the web app dependencies (only needed for Part 2):

   ```bash
   cd labs/ppth-web
   npm install
   ```

4. Follow the pre-flight steps in [`labs/README.md`](labs/README.md) to install the Bob skill and connect to your watsonx Orchestrate instance.

## Repository structure

```
IFSA-students/
├── README.md
├── pyproject.toml           # Python project (ibm-watsonx-orchestrate)
├── workspace_config.yaml    # wxO ADK folder paths
├── img/                     # Images used in the docs
└── labs/
    ├── README.md            # Lab overview, architecture, pre-flight
    ├── 01-lab-multiagent-orchestration.md
    ├── 02-lab-webchat-embed.md
    ├── 03-lab-test-evaluate.md
    ├── 04-extra-lab-control-plane.md
    ├── ppth-web/            # Next.js app for the web chat embed
    └── reference-solution/  # Complete, working project to compare against
```

You create the working project (agents, tools, skills and scripts) in the repository root as you go through the labs. Those folders are git-ignored.

## Additional resources

- [IBM Bob](https://bob.ibm.com): trial access and downloads
- [IBM watsonx Orchestrate](https://www.ibm.com/products/watsonx-orchestrate): product page and 30-day trial
- [IBM watsonx Orchestrate documentation](https://www.ibm.com/docs/en/watsonx/watson-orchestrate)
- [watsonx Orchestrate ADK](https://developer.watson-orchestrate.ibm.com/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Next.js documentation](https://nextjs.org/docs)

## Support

1. Re-read the relevant lab and its pre-flight and ground rules.
2. Compare your work with the [reference solution](labs/reference-solution/).
3. Ask Bob to diagnose the error output.
4. Contact your instructor or teaching assistant.

---

**Author**: Jan Garček
