---
description: Compute CyberRiskGuardian KRIs for one scenario
argument-hint: "PbA Pbx de dm theta mu red_p red_i CVSS-vector [appetite] [factor]"
---

Compute the CyberRiskGuardian indicators for the scenario parameters in: $ARGUMENTS

Use the `crg-calculator` MCP tool `crg_risk` if available; otherwise run `python3` with the `crg_calc` module from the `cyber-risk-assessment` skill's `scripts/` directory. Default appetite 0.30 and factor 1,000 unless given. Show the formula substitution, estimated, tolerated, mitigated and residual risk, the residual-to-tolerated ratio, the classification (below / approximately at / above tolerance), the CVSS v4.0 score and severity, and a one-line interpretation. Remind that Residual/Tolerated depends only on damage, reductions and appetite.
