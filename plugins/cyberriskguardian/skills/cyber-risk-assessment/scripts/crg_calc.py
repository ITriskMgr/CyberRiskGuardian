#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Marc-André Léger
"""CyberRiskGuardian calculation engine.

Implements the KRI formulas of the CyberRiskGuardian Excel Guide v1.0c (s.11) without modification,
CVSS v4.0 base scoring (via the `cvss` package), tolerance classification, sensitivity shifts,
the case-style normalized cross-check and the cybersecurity budget calibration rule.

Usage:
    python3 crg_calc.py scenarios.json [--appetite 0.30] [--factor 1000] [--json]

The input JSON follows assets/scenario-template.json (keys: APPETITE, FACTOR, SCEN[...]).
"""
import argparse, copy, json, os, sys

BAND_LOW, BAND_HIGH = 0.90, 1.10          # "approximately at tolerance" band on Residual/Tolerated
BUDGET_BAND = {"min": 0.04, "median": 0.078, "max": 0.12}   # % of total IT budget incl. salaries
ANCHORS = {"max": 0.30, "median": 0.50, "min": 0.70}        # appetite anchors for the band

DEFAULT_APPETITE = float(os.environ.get("CRG_DEFAULT_APPETITE", "0.30"))
DEFAULT_FACTOR = float(os.environ.get("CRG_FACTOR", "1000"))


def cvss4(vector):
    """Return (base score, severity) for a CVSS v4.0 vector string."""
    from cvss import CVSS4
    v = CVSS4(vector)
    return float(v.base_score), v.severities()[0]


def calc(s, appetite=DEFAULT_APPETITE, factor=DEFAULT_FACTOR):
    """Estimated, tolerated, mitigated, residual risk and ratios for one scenario dict."""
    p = {k: v["v"] if isinstance(v, dict) else v for k, v in s["params"].items()}
    c = s.get("cvss_score")
    if c is None:
        c = cvss4(s["cvss"])[0]
    if not (0 < p["Th"] <= 1):
        raise ValueError(f'{s.get("id")}: resilience (Th) must be in (0, 1]')
    for k in ("red_p", "red_i"):
        if not (0 <= s[k] <= 1):
            raise ValueError(f'{s.get("id")}: {k} must be within 0-1 (a reduction above 1 is invalid)')
    base = p["PbA"] * p["Pbx"] * c * p["Mu"] / p["Th"] * factor
    est = base * (p["De"] + p["Dm"]) / 2
    tol = base * appetite
    mit = est * s["red_p"] * s["red_i"]
    res = est - mit
    return dict(cvss=c, est=est, tol=tol, mit=mit, res=res, ratio=res / tol, pre_ratio=est / tol)


def classify(ratio):
    if ratio < BAND_LOW:
        return "Below tolerance"
    if ratio <= BAND_HIGH:
        return "Approximately at tolerance"
    return "Above tolerance"


def case_normalized(s):
    """Secondary cross-check: T x E x I x (1 - C), C proxied by resilience; post = cur x (1 - P x Q)."""
    p = {k: v["v"] if isinstance(v, dict) else v for k, v in s["params"].items()}
    cur = p["PbA"] * p["Pbx"] * (p["De"] + p["Dm"]) / 2 * (1 - p["Th"])
    return cur, cur * (1 - s["red_p"] * s["red_i"])


def shift(s, d):
    """Sensitivity case: raise threat, exploitation and damages by d; lower resilience and reductions by d."""
    t = copy.deepcopy(s)
    clamp = lambda x: max(0.05, min(0.99, x))
    for k in ("PbA", "Pbx", "De", "Dm"):
        t["params"][k]["v"] = clamp(t["params"][k]["v"] + d)
    t["params"]["Th"]["v"] = clamp(t["params"]["Th"]["v"] - d)
    t["red_p"] = clamp(t["red_p"] - d)
    t["red_i"] = clamp(t["red_i"] - d)
    return t


def budget_target(appetite):
    """Appetite (0-1) -> appetite-consistent cybersecurity budget as % of total IT budget."""
    a, B, A = appetite, BUDGET_BAND, ANCHORS
    if a <= A["max"]:
        return B["max"]
    if a <= A["median"]:
        return B["max"] + (a - A["max"]) / (A["median"] - A["max"]) * (B["median"] - B["max"])
    if a <= A["min"]:
        return B["median"] + (a - A["median"]) / (A["min"] - A["median"]) * (B["min"] - B["median"])
    return B["min"]


def implied_appetite(pct):
    """% of IT budget -> implied appetite; None if below the 4% floor."""
    B, A = BUDGET_BAND, ANCHORS
    if pct >= B["max"]:
        return A["max"]
    if pct >= B["median"]:
        return A["max"] + (B["max"] - pct) / (B["max"] - B["median"]) * (A["median"] - A["max"])
    if pct >= B["min"]:
        return A["median"] + (B["median"] - pct) / (B["median"] - B["min"]) * (A["min"] - A["median"])
    return None


def band_status(pct):
    if pct < BUDGET_BAND["min"]:
        return "Below floor (risk seeking beyond guideline)"
    if pct > BUDGET_BAND["max"]:
        return "Above ceiling"
    return "Within band"


def run(data, appetite=None, factor=None):
    appetite = appetite if appetite is not None else data.get("APPETITE", DEFAULT_APPETITE)
    factor = factor if factor is not None else data.get("FACTOR", DEFAULT_FACTOR)
    out = []
    for s in data["SCEN"]:
        r = calc(s, appetite, factor)
        cur, post = case_normalized(s)
        out.append(dict(id=s["id"], name=s.get("name", ""), **r, cls=classify(r["ratio"]), norm_cur=cur, norm_post=post))
    return dict(appetite=appetite, factor=factor, results=out,
                totals=dict(est=sum(x["est"] for x in out), res=sum(x["res"] for x in out)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios")
    ap.add_argument("--appetite", type=float)
    ap.add_argument("--factor", type=float)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    data = json.load(open(a.scenarios, encoding="utf-8"))
    r = run(data, a.appetite, a.factor)
    if a.json:
        json.dump(r, sys.stdout, indent=1); sys.exit(0)
    print(f"Appetite {r['appetite']:.2f}  Factor {r['factor']:,.0f}")
    print(f"{'ID':5}{'CVSS':>6}{'Estimated':>11}{'Tolerated':>11}{'Residual':>10}{'Res/Tol':>9}  Status")
    for x in r["results"]:
        print(f"{x['id']:5}{x['cvss']:6.1f}{x['est']:11,.0f}{x['tol']:11,.0f}{x['res']:10,.0f}{x['ratio']:9.2f}  {x['cls']}")
    t = r["totals"]
    print(f"Total estimated {t['est']:,.0f}  residual {t['res']:,.0f}  reduction {1 - t['res'] / t['est']:.0%}")
