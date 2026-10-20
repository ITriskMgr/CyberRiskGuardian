---
description: Quick CyberRiskGuardian scan - 20 candidates, 10 scenarios, pre-treatment verdicts
argument-hint: "[organization description or documents]"
---

Run a quick CyberRiskGuardian scan for: $ARGUMENTS

1. Build a short context and evidence register with `crg-context-intake` from what is supplied. Do not interview at length: list gaps instead.
2. Estimate the appetite with `crg-risk-appetite` unless one is given.
3. With `crg-scenario-builder`, generate 20 causal-chain candidates and select the 10 most material, with a one-line rationale each.
4. With `crg-quantify`, estimate parameters and compute estimated and tolerated risk with the calculator (no treatment).
5. Return one table (ID, scenario, estimated, tolerated, ratio, classification, main evidence gap), the top three priorities and the decisions required. Label every estimate.
