#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Marc-André Léger
"""CyberRiskGuardian Core calculator - local MCP server (stdio).

Exposes deterministic tools so the agent never computes KRIs or CVSS scores "by eye":
  - crg_risk     : estimated / tolerated / mitigated / residual risk, ratio and classification
  - cvss4_score  : CVSS v4.0 base score and severity from a vector

Sensitivity analysis and budget calibration tools are part of CRG Pro Quantitative.

Requires: pip install "mcp>=1.2" cvss
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "skills", "crg-quantify", "scripts"))
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
    r.update(classification=C.classify(r["ratio"]), pre_treatment_classification=C.classify(r["pre_ratio"]),
             normalized_current=cur, normalized_post=post,
             note="Residual/Tolerated = ((de+dm)/2) x (1 - red_p x red_i) / appetite; probabilities, CVSS, resilience and criticality scale magnitude only.")
    return r


@mcp.tool()
def cvss4_score(vector: str) -> dict:
    """CVSS v4.0 base score (CVSS-B) and severity for a vector such as CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N."""
    score, sev = C.cvss4(vector)
    return {"vector": vector, "base_score": score, "severity": sev}


if __name__ == "__main__":
    mcp.run()
