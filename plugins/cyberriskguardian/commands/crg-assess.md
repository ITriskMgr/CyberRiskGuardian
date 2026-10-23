---
description: Run a CyberRiskGuardian Core risk assessment end to end
argument-hint: "[organization documents or scope] [--scenarios N]"
---

Use the `cyber-risk-assessment` skill to perform a scenario-driven cybersecurity risk assessment for: $ARGUMENTS

1. Review every attached or referenced organizational document first, then build the evidence register and context with `crg-context-intake`.
2. Ask once, in one question set, for scope, approved risk appetite, number of scenarios (default 10) and deliverable format if not given.
3. Set the appetite with `crg-risk-appetite`, build and select scenarios with `crg-scenario-builder`, and quantify them with `crg-quantify`, computing every risk index and CVSS score with the calculator (the approved or estimated appetite recorded; never a placeholder).
4. If CRG Pro skills are installed (Educational and Professional editions), hand treatment, sensitivity, budget, KRIs and the workbook to them; otherwise list those as next steps (steps 12-19).
5. Write the management summary with `crg-summary-report`.
6. Reply briefly: counts above / at / below tolerance, top priorities, decisions required (each marked **Management decision required**), main information gaps.
