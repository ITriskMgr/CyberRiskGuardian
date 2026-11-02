# Clinique Nord — management summary (draft)

Fictional teaching fixture for the CyberRiskGuardian label-check eval. © 2026 Marc-André Léger, CC BY-NC 4.0.

| ID | Scenario | Pb(A) | Pb(ψ,A) | δe | δm | θ(ψ,A) | μ(E) | Estimated risk | Tolerated risk | Ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | Admin credential compromise through phishing and no MFA on remote access | 0.70 | 0.60 | 0.60 | 0.80 | 0.50 | 0.65 | 3,554 | 1,777 | 2.00 |
| S2 | Supplier portal data leak through an unpatched file-transfer server | 0.40 | 0.35 | 0.50 | 0.70 | 0.55 | 0.60 | 612 | 306 | 2.00 |

Parameters: Analyst estimate — validation required.

The supplier list was not provided. Residual risk will be computed as ABS(L − R) once measures are chosen.

## Recommendations
1. Enforce MFA on all remote and privileged access.
2. Patch the file-transfer server and restrict it to known partners.
