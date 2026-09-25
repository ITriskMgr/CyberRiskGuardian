# Cybersecurity budget guideline

Source: rule stated by the CyberRiskGuardian author (Marc-André Léger), September 2026.

**Rule.** An appropriate cybersecurity budget is **4% to 12% of the total IT budget, including salaries**.

| Position | % of IT budget | Meaning | Appetite anchor (0–1) |
|---|---|---|---|
| Floor | 4% | Risk seeking | 0.70 |
| Median | 7.8% | Risk neutral | 0.50 |
| Ceiling | 12% | Risk averse | 0.30 |

**Appetite → target.** Linear interpolation between anchors, clamped outside (≤ 0.30 → 12%; ≥ 0.70 → 4%). Examples: 0.40 → 9.9%; 0.60 → 5.9%.
**Implied appetite.** The inverse mapping turns a planned budget % into the appetite it expresses; below 4% it is outside the guideline.

## Calibration step (after the investment portfolio, before KRIs)
1. Set or confirm the risk appetite.
2. Derive the appetite-consistent budget (% and currency).
3. Estimate current security spend (staff, tooling, security embedded in IT contracts); flag it if below the 4% floor.
4. Build the risk-driven budget bottom-up from the treatment portfolio:
   - phase by fiscal year (one-time costs spread between start and end month; recurring from completion);
   - split shared initiatives between the cybersecurity and IT budgets, stating the share and why;
   - count shared controls once; add new security roles, risk-transfer premiums and a contingency (≈10%).
5. Compute the implied appetite of the risk-driven budget and compare with the approved appetite.
6. Reconcile — one of: fund to the appetite (ramped to delivery capacity; any balancing reserve released only against scenario-linked business cases), formally revise the appetite, or accept residual risk. **Management decision required.**
7. Show what each funding level buys: tolerance status counts and total residual risk per option.
8. Report the cybersecurity project budget and the security-relevant IT project budget separately (IT-funded prerequisites do not count toward the cyber %).
9. Test sensitivity to the IT budget figure; if the plan exceeds 12%, phase or trim starting with the lowest cost-effectiveness items.
10. Monitor with KRIs: cyber spend as % of IT budget (target = appetite-consistent %, warning < 7.8%, critical < 4%); share of spend tied to assessed scenarios; reserve releases tied to scenario IDs.

## Typical cyber/IT split conventions (state and justify per case)
| Initiative type | Cyber share |
|---|---|
| MFA, PAM, MDR/SIEM, cloud posture, access governance, vendor risk, awareness, DDoS/WAF, brand protection | 100% |
| Vulnerability management (scanning, SLAs, analyst) vs patch deployment | ~75% |
| IR readiness (plan, retainer, exercises) vs clinical/business downtime procedures | ~70% |
| Device security programme vs biomedical/OT engineering effort | ~70% |
| Immutable/isolated backup vs backup platform; segmentation vs network re-architecture; integrity monitoring vs application support | ~50% |
| MDM/endpoint encryption, secure communications platforms | ~30% |
| Carrier redundancy, SD-WAN, secondary site, device replacement | 0–20% (IT project) |

## First application (MediBec, Sept 2026 — fictional teaching case)
IT budget CAD 100M (evidence gap). Appetite 0.30 → 12% = CAD 12.0M. Current spend ≈ CAD 0.9M (0.9%, below floor). Risk-driven budget CAD 5.1M / 5.5M / 4.9M (5.1–5.5%, within band, implied appetite ≈ 0.64 — inconsistent with 0.30). Recommended ramp CAD 7.0M / 9.5M / 12.0M with a gated reserve. If the IT budget were ≈ CAD 46–50M, the risk-driven budget alone would sit at 10–12%.
