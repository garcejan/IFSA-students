#!/usr/bin/env bash
# delete-all.sh - counterpart to import-all.sh. Removes everything this lab created,
# in reverse dependency order (orchestrator -> collaborators -> skill -> tools).

set -uo pipefail
cd "$(dirname "$0")"

echo "== active environment =="
orchestrate env list

echo
echo "== agents =="
for a in dr_house dr_wilson dr_foreman dr_chase dr_cameron dr_cuddy case_reviewer; do
  orchestrate agents remove -n "$a" -k native || echo "   (skip $a)"
done

echo
echo "== agent skills =="
orchestrate skills remove -n diagnostics-protocol || echo "   (skip diagnostics-protocol)"
orchestrate skills remove -n case-review-rubric || echo "   (skip case-review-rubric)"

echo
echo "== tools =="
for t in get_patient_record differential_diagnosis lab_reference \
         check_drug_interaction list_formulary score_case_review; do
  orchestrate tools remove -n "$t" || echo "   (skip $t)"
done

echo
echo "== result =="
orchestrate agents list
orchestrate tools list
