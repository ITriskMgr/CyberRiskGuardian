#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Marc-André Léger
"""CyberRiskGuardian calculator - local MCP server (stdio).

Exposes deterministic tools so the agent never computes KRIs, CVSS scores or budget targets "by eye":
  - crg_risk          : estimated / tolerated / mitigated / residual risk, ratio and classification
  - cvss4_score       : CVSS v4.0 base score and severity from a vector
  - budget_target     : appetite-consistent cybersecurity budget (% and amount) within the 4-12% band
  - budget_check      : band status and implied appetite for planned spend
  - sensitivity       : lower / central / higher case for one scenario

Requires: pip install "mcp>=1.2" cvss
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skills", "cyber-risk-assessment", "scripts"))
from mcp.server.fastmcp import FastMCP
import crg_calc as C

mcp = FastMCP("crg-calculator")


def _scenario(pba, pbx, de, dm, theta, mu, red_p, red_i, cvss_vector=None, cvss_score=None):
    s = {"id": "X", "params": {"PbA": {"v": pba}, "Pbx": {"v": pbx}, "De": {"v": de}, "Dm": {"v": dm},
                               "Th": {"v": theta}, "Mu": {"v": mu}}, "red_p": red_p, "red_i": red_i}
    if cvss_score is not None:
        s["cvss_score"] = cvss_score
    elif cvss_vector:
        s["cvss"] = cvss_vector
    else:
        raise ValueError("Provide cvss_vector or cvss_score")
    return s


@mcp.tool()
def crg_risk(pba: float, pbx: float, de: float, dm: float, theta: float, mu: float,
             red_p: float = 0.0, red_i: float = 0.0, appetite: float = C.DEFAULT_APPETITE,
             factor: float = C.DEFAULT_FACTOR, cvss_vector: str = "", cvss_score: float = -1) -> dict:
    """CyberRiskGuardian KRIs for one scenario (Excel Guide v1.0c formulas, unmodified).
    pba=Pb(A) threat presence, pbx=Pb(psi,A) exploitation, de/dm expected/maximum damage, theta=resilience,
    mu=criticality, red_p/red_i = probability/impact reduction of the treatment package (0-1).
    Give either cvss_vector (CVSS:4.0/...) or cvss_score (0-10)."""
    s = _scenario(pba, pbx, de, dm, theta, mu, red_p, red_i, cvss_vector or None, None if cvss_score < 0 else cvss_score)
    r = C.calc(s, appetite, factor)
    cur, post = C.case_normalized(s)
    r.update(classification=C.classify(r["ratio"]), normalized_current=cur, normalized_post=post,
             note="Residual/Tolerated = ((de+dm)/2) x (1 - red_p x red_i) / appetite; probabilities, CVSS, resilience and criticality scale magnitude only.")
    return r


@mcp.tool()
def cvss4_score(vector: str) -> dict:
    """CVSS v4.0 base score (CVSS-B) and severity for a vector such as CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N."""
    score, sev = C.cvss4(vector)
    return {"vector": vector, "base_score": score, "severity": sev}


@mcp.tool()
def budget_target(appetite: float, it_budget: float) -> dict:
    """Appetite-consistent cybersecurity budget. Guideline: 4% (risk seeking, appetite 0.70) - 7.8% (neutral, 0.50) -
    12% (risk averse, 0.30) of total IT budget including salaries; linear between anchors, clamped outside."""
    pct = C.budget_target(appetite)
    return {"appetite": appetite, "target_pct": pct, "target_amount": pct * it_budget,
            "band": {k: v * it_budget for k, v in C.BUDGET_BAND.items()}}


@mcp.tool()
def budget_check(spend: list[float], it_budget: float) -> list[dict]:
    """Band status and implied risk appetite for each year of planned cybersecurity spend."""
    out = []
    for i, s in enumerate(spend, 1):
        pct = s / it_budget
        out.append({"year": i, "spend": s, "pct": pct, "status": C.band_status(pct), "implied_appetite": C.implied_appetite(pct)})
    return out


@mcp.tool()
def sensitivity(pba: float, pbx: float, de: float, dm: float, theta: float, mu: float, red_p: float, red_i: float,
                delta: float = 0.10, appetite: float = C.DEFAULT_APPETITE, factor: float = C.DEFAULT_FACTOR,
                cvss_vector: str = "", cvss_score: float = -1) -> dict:
    """Lower / central / higher case: all uncertain parameters shifted by +/- delta against or in favour of the organization."""
    s = _scenario(pba, pbx, de, dm, theta, mu, red_p, red_i, cvss_vector or None, None if cvss_score < 0 else cvss_score)
    out = {}
    for lab, d in (("lower", -delta), ("central", 0.0), ("higher", delta)):
        r = C.calc(C.shift(s, d), appetite, factor)
        out[lab] = {"estimated": r["est"], "residual": r["res"], "ratio": r["ratio"], "classification": C.classify(r["ratio"])}
    out["robust"] = out["lower"]["classification"] == out["higher"]["classification"]
    return out


if __name__ == "__main__":
    mcp.run()
