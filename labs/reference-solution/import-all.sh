#!/usr/bin/env bash
# import-all.sh - dependency-ordered import of the Princeton-Plainsboro diagnostics team.
#
# Order matters: tools -> agent skills -> collaborator agents -> orchestrator -> deploy.
# Re-running this script is safe: every import overwrites by name.
#
# Usage:
#   source .venv/bin/activate
#   orchestrate env activate <your-env> --api-key "$WXO_API_KEY"
#   ./import-all.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "== active environment =="
orchestrate env list

echo
echo "== 1/4 tools =="
for f in tools/patient_registry.py \
         tools/differential_diagnosis.py \
         tools/lab_reference.py \
         tools/drug_interaction.py \
         tools/case_review.py; do
  echo "-- $f"
  orchestrate tools import -k python -f "$f"
done

echo
echo "== 2/4 agent skills =="
orchestrate skills import -f skills/diagnostics_protocol/SKILL.md --upsert
orchestrate skills import -f skills/case_review_rubric/SKILL.md --upsert

echo
echo "== 3/4 collaborator agents (before the orchestrator that references them) =="
for a in dr_wilson dr_foreman dr_chase dr_cameron dr_cuddy case_reviewer; do
  echo "-- $a"
  orchestrate agents import -f "agents/${a}.yaml"
done

echo
echo "== 4/4 orchestrator =="
orchestrate agents import -f agents/dr_house.yaml

echo
echo "== deploy =="
for a in dr_wilson dr_foreman dr_chase dr_cameron dr_cuddy case_reviewer dr_house; do
  orchestrate agents deploy -n "$a"
done

echo
echo "== result =="
orchestrate agents list
