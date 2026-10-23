#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Marc-André Léger
"""CyberRiskGuardian Core calculator - local MCP server (stdio).

Exposes deterministic tools so the agent never computes risk indices or CVSS scores "by eye":
  - crg_risk           : estimated / tolerated / mitigated / residual risk, risk-to-tolerance ratio,
                         classification and, when a cost is given, cost-effectiveness
  - cvss4_score        : CVSS v4.0 base score and severity from a vector
  - crg_budget_target  : appetite-consistent cybersecurity budget target (12% / 7.8% / 4% of total IT
                         budget incl. salaries at appetite 0.30 / 0.50 / 0.70, linear), gap and implied appetite

All inputs are validated by the engine (crg-quantify/scripts/crg_calc.py). Sensitivity analysis and the
costed treatment portfolio are part of CRG Pro Quantitative.

Requires: pip install "mcp>=1.2" cvss (mcp 1.x FastMCP or 2.x MCPServer)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skills", "crg-quantify", "scripts"))
try:                                      # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ImportError:                       # mcp 2.x renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
import crg_calc as C

mcp = _Server("crg-calculator")


def _scenario(pba, pbx, de, dm, theta, mu, red_p, red_i, cvss_vector=None, cvss_score=None, cost=None):
    s = {"id": "X", "params": {"PbA": {"v": pba}, "Pbx": {"v": pbx}, "De": {"v": de}, "Dm": {"v": dm},
                               "Th": {"v": theta}, "Mu": {"v": mu}}, "red_p": red_p, "red_i": red_i}
    if cvss_vector:
        s["cvss"] = cvss_vector
    if cvss_score is not None:
        s["cvss_score"] = cvss_score
    if cost:
        s["cost"] = cost
    return s


@mcp.tool()
def crg_risk(pba: float, pbx: float, de: float, dm: float, theta: float, mu: float,
             red_p: float = 0.0, red_i: float = 0.0, appetite: float = -1.0,
             factor: float = C.DEFAULT_FACTOR, cvss_vector: str = "", cvss_score: float = -1,
             cost: float = 0.0) -> dict:
    """CyberRiskGuardian risk indices for one scenario (CRG formulas, unmodified).
    pba=Pb(A) threat presence, pbx=Pb(psi,A) exploitation, de/dm expected/maximum damage (dm >= de),
    theta=resilience in (0,1], mu=organizational criticality (expected utility), red_p/red_i = probability/impact
    reduction of the treatment package (0-1; combine several measures with 1 - prod(1 - r_i) per lever).
    appetite: the approved or estimated appetite, strictly between 0 and 1 (if omitted, the default is used and
    reported as an Analytical estimate). Give cvss_vector (preferred; the score is recomputed from it and a
    disagreeing cvss_score is flagged) or cvss_score (0-10). cost (optional, > 0) adds cost-effectiveness."""
    warnings = []
    if appetite is None or appetite < 0:
        appetite = C.DEFAULT_APPETITE
        warnings.append(f"appetite not given: defaulted to {appetite:.2f} - Analytical estimate — validation required")
    s = _scenario(pba, pbx, de, dm, theta, mu, red_p, red_i, cvss_vector or None,
                  None if cvss_score < 0 else cvss_score, cost or None)
    r = C.calc(s, appetite, factor)
    cur, post = C.case_normalized(s)
    r["warnings"] = warnings + r["warnings"]
    r.update(appetite=appetite, factor=factor, classification=C.classify(r["ratio"]),
             pre_treatment_classification=C.classify(r["pre_ratio"]),
             normalized_current=cur, normalized_post=post,
             note="Risk-to-tolerance ratio = ((de+dm)/2) x (1 - red_p x red_i) / appetite; probabilities, CVSS, "
                  "resilience and criticality scale magnitude (ranking) only. Below tolerance (< 0.90) needs "
                  "red_p x red_i > 1 - 0.90 x appetite / average damage; ratio <= 1.0 needs red_p x red_i >= 1 - appetite / average damage.")
    return r


@mcp.tool()
def cvss4_score(vector: str) -> dict:
    """CVSS v4.0 base score (CVSS-B) and severity for a vector such as CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N."""
    score, sev = C.cvss4(vector)
    return {"vector": vector, "base_score": score, "severity": sev}


@mcp.tool()
def crg_budget_target(appetite: float, it_budget: float = 0.0, current_cyber_spend: float = -1.0,
                      portfolio_cost: float = 0.0) -> dict:
    """Appetite-consistent cybersecurity budget (best practice, a reasonableness check, not a formula that
    dictates the budget): 12% of total IT budget incl. salaries at appetite <= 0.30, 7.8% at 0.50, 4% at >= 0.70,
    linear in between. With it_budget, returns the target amount; with current_cyber_spend (and the first-year
    portfolio_cost), the planned share, the gap to target and the implied appetite. The gap is reported and
    closing it or revising the appetite is a Management decision required."""
    if it_budget and it_budget > 0:
        return C.budget_position(appetite, it_budget, None if current_cyber_spend < 0 else current_cyber_spend, portfolio_cost)
    return {"appetite": appetite, "target_pct": C.budget_target(appetite)}


if __name__ == "__main__":
    mcp.run()
