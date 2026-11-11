#!/bin/bash
# Part 4 teardown. Controls come off BEFORE the assets they bind to.
cd "$(dirname "$0")"
orchestrate controls remove -n phi_pii_guard        || true
orchestrate agents undeploy -n aaa_probe            || true
orchestrate agents remove   -n aaa_probe -k native  || true
orchestrate tools remove    -n whoami               || true
orchestrate tools remove    -n phi_gate             || true
orchestrate tools remove    -n phi_tool_filter      || true
echo "Now restore dr_house to its Part 3 state: remove the context_variables and"
echo "plugins blocks from its YAML, re-import and re-deploy."
