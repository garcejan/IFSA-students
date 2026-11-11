#!/bin/bash
# Part 4 — import the AAA layer on top of the Part 1 House team.
# Run AFTER the main reference-solution import-all.sh.
set -uo pipefail
cd "$(dirname "$0")"

echo "== probe tool + agent (Step 1) =="
orchestrate tools import  -k python -f tools/whoami.py
orchestrate agents import -f agents/aaa_probe.yaml
orchestrate agents deploy -n aaa_probe

echo "== plugins (Steps 3 and 4) =="
orchestrate tools import -k python -f plugins/phi_gate.py
orchestrate tools import -k python -f plugins/phi_tool_filter.py
# Attach one of them by merging agents/dr_house_aaa.patch.yaml into your dr_house YAML,
# then: orchestrate agents import -f agents/dr_house.yaml && orchestrate agents deploy -n dr_house

echo "== control (Step 5) — create-only, so remove first =="
orchestrate controls remove -n phi_pii_guard 2>/dev/null || true
orchestrate controls import -f controls/phi_pii_control.yaml

echo "== done =="
orchestrate controls list
