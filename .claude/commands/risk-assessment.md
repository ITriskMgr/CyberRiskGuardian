---
description: Run a full CyberRiskGuardian risk assessment
argument-hint: "[organization documents or scope] [--scenarios N]"
---

Use the `cyber-risk-assessment` skill to perform a complete scenario-driven cybersecurity risk assessment for: $ARGUMENTS

1. Review every attached or referenced organizational document first and build the evidence register.
2. Ask once (single question set) for scope, approved risk appetite, number of scenarios (default 10) and deliverable format (Word + Excel recommended) if not given.
3. Select scenarios with the `scenario-selection` skill, quantify them, compute KRIs with the bundled calculator (never by eye), build the treatment portfolio, calibrate the budget with the `cyber-budget-calibration` skill, and produce the management report and live-formula workbook.
4. Recalculate the workbook, verify zero formula errors, cross-check key numbers, render the report and review it before delivering.
5. Reply briefly: counts above / at / below tolerance, portfolio cost, budget position, decisions required, key caveats.
