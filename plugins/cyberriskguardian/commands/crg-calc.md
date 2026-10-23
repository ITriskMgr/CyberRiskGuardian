---
description: Compute CyberRiskGuardian risk indices for one scenario
argument-hint: "PbA Pbx de dm theta mu red_p red_i CVSS-vector appetite [factor] [cost]"
---

Compute the CyberRiskGuardian indicators for the scenario parameters in: $ARGUMENTS

Use the `crg-calculator` MCP tool `crg_risk` if available; otherwise run `python3` with the `crg_calc` module from the `crg-quantify` skill's `scripts/` directory. Factor 1,000 unless given. If no appetite is given, say that the default 0.30 is used and mark it **Analytical estimate — validation required**. The calculator rejects out-of-range inputs (0-1, dm >= de, theta in (0,1], appetite strictly in (0,1), CVSS required) - report the error rather than changing a value. Show the formula substitution, estimated, tolerated, mitigated and residual risk, the risk-to-tolerance ratio, the classification (below / approximately at / above tolerance), the CVSS v4.0 score and severity, the cost-effectiveness when a cost is given, and a one-line interpretation. Remind that Residual/Tolerated depends only on damage, reductions and appetite.
