#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Marc-André Léger
"""CyberRiskGuardian calculation engine - Community Edition (Core plugin).

Implements the CyberRiskGuardian risk-index formulas without modification (book v2.51b ch06 "The risk
index and underlying formulas"; Excel Guide v1.0c s.11): estimated, tolerated, mitigated and residual
risk, the risk-to-tolerance ratio and its bands, cost-effectiveness, CVSS v4.0 base scoring (via the
`cvss` package), the appetite-consistent budget target and the case-style normalized cross-check.
Sensitivity analysis and the live-formula workbook are part of CRG Pro Quantitative.

Every input is validated (book ch06 "The variables and their valid ranges"):
  Pb(A), Pb(psi,A), de, dm, mu(E), red_p, red_i in [0, 1]; theta in (0, 1]; dm >= de;
  appetite strictly in (0, 1) (warning outside 0.1-0.9); factor > 0 (warning when not 1,000);
  CVSS required, 0-10 (a blank is an error, never 0). When a scenario carries both a vector and a
  stored score, the score is recomputed from the vector, the vector wins and a disagreement is flagged.

Unified engine rules (CRG 1.9.2, X-1 / X-2; the Threat Live-feed plugin's crg_calc.py is the reference
implementation, this engine follows the same rules):
  - CVSS-B only: the eleven Base metrics. A vector with a Threat metric (E) is refused with the instruction
    to route exploitation evidence (KEV, EPSS, proof of concept) to Pb(psi,A); Environmental and Supplemental
    metrics are refused too. CVSS-BT / CVSS-BE are never reported as the base score.
  - Zero tolerated risk (Pb(A), Pb(psi,A), mu(E) or CVSS = 0, all admissible): estimated = tolerated = 0, the
    ratios are None and the verdict is "Not applicable (tolerated risk 0)" (never "Above tolerance"), with a
    warning asking the analyst to confirm the zero is deliberate.
  - A missing or null reduction means 0; numeric strings, booleans, NaN and infinity are rejected.
  - A missing APPETITE falls back to the default and is reported as an Analytical estimate.
  - Budget conventions: gap = planned - target (negative = below target); implied appetite None below the 4%
    point target (never clamped to 0.70); the TI position / status wording.

Usage:
    python3 crg_calc.py scenarios.json [--appetite 0.30] [--factor 1000] [--json]
                        [--it-budget 50000000 [--cyber-spend 3000000]]
    python3 crg_calc.py - < scenarios.json      # or pipe the data model on standard input
    python3 crg_calc.py --budget-target 0.35

The input JSON follows assets/scenario-template.json (keys: APPETITE, FACTOR, SCEN[...]; optional
per-scenario `cost` or `cost_y1` for cost-effectiveness).
"""
import argparse, json, math, os, sys

BAND_LOW, BAND_HIGH = 0.90, 1.10          # "approximately at tolerance" band on the risk-to-tolerance ratio
CVSS_TOLERANCE = 0.05                      # stored score vs vector score: larger differences are flagged
NOT_APPLICABLE = "Not applicable (tolerated risk 0)"
CVSS_BASE_METRICS = ("AV", "AC", "AT", "PR", "UI", "VC", "VI", "VA", "SC", "SI", "SA")
CVSS_THREAT_METRICS = ("E",)
CVSS_ENVIRONMENTAL_METRICS = ("CR", "IR", "AR", "MAV", "MAC", "MAT", "MPR", "MUI", "MVC", "MVI", "MVA", "MSC", "MSI", "MSA")
CVSS_SUPPLEMENTAL_METRICS = ("S", "AU", "R", "V", "RE", "U")

DEFAULT_APPETITE = float(os.environ.get("CRG_DEFAULT_APPETITE", "0.30"))
DEFAULT_FACTOR = float(os.environ.get("CRG_FACTOR", "1000"))
STD_FACTOR = 1000.0

# Budget best practice (book v2.51b ch04, ch06 "Calibrating the budget to risk appetite"):
# % of total IT budget including salaries, point targets with linear interpolation, clamped outside.
BUDGET_POINTS = ((0.30, 0.12), (0.50, 0.078), (0.70, 0.04))

PARAMS = (("PbA", "Pb(A)"), ("Pbx", "Pb(psi,A)"), ("De", "de (expected damage)"),
          ("Dm", "dm (maximum damage)"), ("Th", "theta (resilience)"), ("Mu", "mu(E) (organizational criticality)"))
BUDGET_BAND = {"min": 0.04, "median": 0.078, "max": 0.12}   # the three point targets, as shares


class CVSSBaseOnlyError(ValueError):
    """The vector carries Threat, Environmental or Supplemental metrics: CRG scores CVSS-B only."""


def check_base_only(vector):
    """Raise CVSSBaseOnlyError when the vector carries metrics outside the eleven Base metrics (X-1 / NV-P03).
    CRG multiplies CVSS by Pb(psi,A): exploitation evidence counted in E and again in Pb(psi,A) would compound, and
    Environmental modifiers duplicate mu(E), de/dm and theta."""
    if not isinstance(vector, str):
        return
    keys = [part.split(":", 1)[0] for part in vector.strip().split("/")[1:] if part]
    threat = [k for k in keys if k in CVSS_THREAT_METRICS]
    env = [k for k in keys if k in CVSS_ENVIRONMENTAL_METRICS]
    supp = [k for k in keys if k in CVSS_SUPPLEMENTAL_METRICS]
    if threat:
        raise CVSSBaseOnlyError(f"CVSS-B only: Threat metric {', '.join(threat)} is not accepted - route exploitation "
                                "evidence (CISA KEV, EPSS, proof of concept) to Pb(psi,A) and give the Base vector")
    if env:
        raise CVSSBaseOnlyError(f"CVSS-B only: Environmental metric {', '.join(env)} is not accepted - organizational "
                                "context enters through mu(E), de, dm and theta; give the Base vector")
    if supp:
        raise CVSSBaseOnlyError(f"CVSS-B only: Supplemental metric {', '.join(supp)} is not accepted (it does not change "
                                "the score) - give the Base vector")


def cvss4(vector):
    """Return (base score, severity) for a CVSS v4.0 vector string. Base metrics only (CVSS-B): a vector with Threat,
    Environmental or Supplemental metrics raises CVSSBaseOnlyError instead of reporting CVSS-BT / CVSS-BE."""
    from cvss import CVSS4
    check_base_only(vector)
    v = CVSS4(vector)
    return float(v.base_score), v.severities()[0]


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x) and not math.isinf(x)


def check_appetite(appetite):
    """Raise on an appetite outside (0, 1); return a list of warnings."""
    if not _num(appetite) or not (0 < appetite < 1):
        raise ValueError(f"appetite must lie strictly between 0 and 1 (got {appetite!r})")
    if not (0.1 <= appetite <= 0.9):
        return [f"appetite {appetite:.2f} is outside the practical range 0.1-0.9; justify it with strong evidence"]
    return []


def check_factor(factor):
    if not _num(factor) or factor <= 0:
        raise ValueError(f"factor must be a positive constant (got {factor!r})")
    if factor != STD_FACTOR:
        return [f"factor {factor:,.0f} differs from the CRG constant 1,000; it scales indices, never ratios, "
                "and must be the same for every scenario and every compared assessment"]
    return []


def resolve_cvss(s):
    """Return (score, severity or None, warnings). The vector is authoritative when present."""
    sid = s.get("id", "?")
    vec, stored = s.get("cvss"), s.get("cvss_score")
    warn = []
    if stored is not None and stored != "" and not _num(stored):
        raise ValueError(f"{sid}: CVSS score must be a number from 0 to 10 (got {stored!r})")
    if stored == "":
        stored = None
    if vec:
        try:
            score, sev = cvss4(vec)
        except CVSSBaseOnlyError as e:
            raise ValueError(f"{sid}: {e} (vector {vec!r})")
        except Exception as e:
            raise ValueError(f"{sid}: invalid CVSS v4.0 vector {vec!r} ({e})")
        if stored is not None and abs(float(stored) - score) > CVSS_TOLERANCE:
            warn.append(f"{sid}: stored cvss_score {stored} disagrees with the score computed from its vector ({score}); "
                        "the vector score is used - correct the stored score")
        return score, sev, warn
    if stored is None:
        raise ValueError(f"{sid}: CVSS required - give the CVSS v4.0 vector (preferred) or a score from 0 to 10; a blank is never 0")
    if not (0 <= stored <= 10):
        raise ValueError(f"{sid}: CVSS score must be within 0-10 (got {stored})")
    warn.append(f"{sid}: CVSS score given without its vector; keep the vector next to the score so it can be recomputed")
    return float(stored), None, warn


def validate(s):
    """Validate one scenario dict. Return (params, cvss, severity, warnings); raise ValueError listing every error."""
    sid = s.get("id", "?")
    raw = s.get("params") or {}
    p, errs = {}, []
    for k, label in PARAMS:
        v = raw.get(k)
        v = v.get("v") if isinstance(v, dict) else v
        if not _num(v):
            errs.append(f"{label} ({k}) is required and must be a number")
            continue
        p[k] = float(v)
        if k == "Th":
            if not (0 < v <= 1):
                errs.append(f"{label} must be in (0, 1] (got {v}); theta divides the index")
        elif not (0 <= v <= 1):
            errs.append(f"{label} must be within 0-1 (got {v})")
    if "De" in p and "Dm" in p and p["Dm"] < p["De"]:
        errs.append(f"dm ({p['Dm']}) must be at least de ({p['De']}): the plausible worst case cannot be milder than the expected case")
    for k in ("red_p", "red_i"):
        v = s.get(k, 0.0)
        if v is None:                       # a missing or null reduction means 0 (no treatment), unified rule
            v = 0.0
        if not _num(v) or not (0 <= v <= 1):
            errs.append(f"{k} must be within 0-1 (got {v!r}); a reduction above 1 is invalid - combine measures with 1 - prod(1 - r_i)")
    cost = s.get("cost", s.get("cost_y1"))
    if cost is not None and (not _num(cost) or cost <= 0):
        errs.append(f"cost must be a positive amount when given (got {cost!r})")
    try:
        c, sev, warn = resolve_cvss(s)
    except ValueError as e:
        errs.append(str(e).split(": ", 1)[-1] if str(e).startswith(f"{sid}: ") else str(e))
        c, sev, warn = None, None, []
    if errs:
        raise ValueError(f"{sid}: " + "; ".join(errs))
    return p, c, sev, warn


def calc(s, appetite=DEFAULT_APPETITE, factor=DEFAULT_FACTOR):
    """Estimated, tolerated, mitigated and residual risk, ratios and cost-effectiveness for one scenario dict."""
    warn = check_appetite(appetite) + check_factor(factor)
    p, c, sev, w = validate(s)
    warn += w
    red_p, red_i = float(s.get("red_p") or 0.0), float(s.get("red_i") or 0.0)
    base = p["PbA"] * p["Pbx"] * c * p["Mu"] / p["Th"] * factor
    est = base * (p["De"] + p["Dm"]) / 2
    tol = base * appetite
    mit = est * red_p * red_i
    res = max(0.0, est - mit)              # guard: never negative (reductions are validated to 0-1)
    if tol:
        ratio, pre = res / tol, est / tol
    else:                                  # zero tolerated risk: the ratio is not applicable (never "Above tolerance")
        ratio = pre = None
        zero = [label for k, label in PARAMS if k in ("PbA", "Pbx", "Mu") and p[k] == 0] + (["CVSS"] if c == 0 else [])
        warn.append(f"{s.get('id', '?')}: tolerated risk is 0 ({', '.join(zero)} = 0): the risk-to-tolerance ratio is "
                    "not applicable - confirm the zero is deliberate (the scenario is then not a risk in the period) "
                    "rather than a placeholder")
    out = dict(cvss=c, cvss_severity=sev, est=est, tol=tol, mit=mit, res=res, ratio=ratio, pre_ratio=pre, warnings=warn)
    cost = s.get("cost", s.get("cost_y1"))
    if cost:
        out["cost"] = float(cost)
        out["ce"] = cost_effectiveness(est, res, cost)
    return out


def cost_effectiveness(est, res, cost):
    """Risk units removed per $1,000: (Re - Residual) / cost x 1,000."""
    if not _num(cost) or cost <= 0:
        raise ValueError("cost must be positive for cost-effectiveness")
    return (est - res) / cost * 1000


def budget_target(appetite):
    """Appetite-consistent cybersecurity budget as a share of total IT budget incl. salaries.
    12% at appetite <= 0.30, 7.8% at 0.50, 4% at >= 0.70, linear in between (best practice, not a formula
    that dictates the budget)."""
    check_appetite(appetite)
    pts = BUDGET_POINTS
    if appetite <= pts[0][0]:
        return pts[0][1]
    for (a0, b0), (a1, b1) in zip(pts, pts[1:]):
        if appetite <= a1:
            return b0 + (appetite - a0) / (a1 - a0) * (b1 - b0)
    return pts[-1][1]


def implied_appetite(share):
    """Inverse of budget_target: the appetite implied by a cybersecurity spend share. None below the 4% point
    target, where the funding implies an appetite above 0.70, outside the guideline (unified convention, NV-094:
    never clamped to 0.70)."""
    pts = BUDGET_POINTS
    if share >= pts[0][1]:
        return pts[0][0]
    for (a0, b0), (a1, b1) in zip(pts, pts[1:]):
        if share >= b1:
            return a0 + (b0 - share) / (b0 - b1) * (a1 - a0)
    return None


def band_status(pct):
    """Position of a budget share relative to the three point targets (wording shared by every CRG product)."""
    if pct < BUDGET_BAND["min"]:
        return "Below the 4% point target (funding implies an appetite above 0.70, outside the guideline)"
    if pct > BUDGET_BAND["max"]:
        return "Above the 12% point target (more than a risk-averse organization's best-practice target)"
    return "Between the 4% and 12% point targets"


def target_status(pct, appetite):
    """Compare a budget share with the appetite-consistent point target (gap = planned - target)."""
    t = budget_target(appetite)
    gap = pct - t
    status = "Below the appetite-consistent target" if gap < 0 else "At or above the appetite-consistent target"
    return {"target_pct": t, "gap_pct": gap, "status": status}


def budget_position(appetite, it_budget, cyber_spend=None, portfolio_cost=0.0):
    """Target, planned spend, gap to target and implied appetite (book ch06 step 18).
    Unified conventions (NV-094): gap_amount / gap_pct = planned - target (negative = below target); implied_appetite
    is None below the 4% point target; position and status use the shared wording."""
    errs = []
    if not _num(it_budget) or it_budget <= 0:
        errs.append(f"IT budget must be a positive amount (got {it_budget!r})")
    for name, v in (("current spending", cyber_spend), ("portfolio cost", portfolio_cost)):
        if v is not None and (not _num(v) or v < 0):
            errs.append(f"{name} must be a non-negative amount (got {v!r})")
    if errs:
        raise ValueError("; ".join(errs))
    tgt = budget_target(appetite)
    out = dict(appetite=appetite, target_pct=tgt, target_amount=tgt * it_budget, it_budget=it_budget)
    if cyber_spend is not None:
        planned = cyber_spend + (portfolio_cost or 0.0)
        share = planned / it_budget
        ts = target_status(share, appetite)
        out.update(current_spend=cyber_spend, portfolio_cost=portfolio_cost or 0.0, planned_total=planned, planned_pct=share,
                   gap_amount=planned - tgt * it_budget, gap_pct=ts["gap_pct"], implied_appetite=implied_appetite(share),
                   status=ts["status"], position=band_status(share))
    return out


def classify(ratio):
    """Tolerance verdict from the risk-to-tolerance ratio. None or NaN (tolerated risk 0) is not a verdict."""
    if ratio is None or not _num(ratio):
        return NOT_APPLICABLE
    if ratio < BAND_LOW:
        return "Below tolerance"
    if ratio <= BAND_HIGH:
        return "Approximately at tolerance"
    return "Above tolerance"


def case_normalized(s):
    """Secondary cross-check: T x E x I x (1 - C), C proxied by resilience; post = cur x (1 - P x Q)."""
    p = {k: v["v"] if isinstance(v, dict) else v for k, v in s["params"].items()}
    cur = p["PbA"] * p["Pbx"] * (p["De"] + p["Dm"]) / 2 * (1 - p["Th"])
    return cur, cur * (1 - s.get("red_p", 0.0) * s.get("red_i", 0.0))


def run(data, appetite=None, factor=None):
    warnings = []
    if appetite is None:
        appetite = data.get("APPETITE")
    if appetite is None:
        appetite = DEFAULT_APPETITE
        warnings.append(f"APPETITE missing from the data model: defaulted to {DEFAULT_APPETITE:.2f} - "
                        "Analytical estimate — validation required. Record the approved or estimated appetite with its rationale.")
    if factor is None:
        factor = data.get("FACTOR", DEFAULT_FACTOR)
    warnings += check_appetite(appetite) + check_factor(factor)
    out, errors = [], []
    for s in data["SCEN"]:
        try:
            r = calc(s, appetite, factor)
        except ValueError as e:
            errors.append(str(e)); continue
        w = [x for x in r.pop("warnings") if x not in warnings]
        warnings += w
        cur, post = case_normalized(s)
        out.append(dict(id=s["id"], name=s.get("name", ""), **r, cls=classify(r["ratio"]), norm_cur=cur, norm_post=post))
    if errors:
        raise ValueError("invalid input:\n  " + "\n  ".join(errors))
    totals = dict(est=sum(x["est"] for x in out), res=sum(x["res"] for x in out))
    return dict(appetite=appetite, factor=factor, budget_target_pct=budget_target(appetite),
                results=out, totals=totals, warnings=warnings)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="?")
    ap.add_argument("--appetite", type=float)
    ap.add_argument("--factor", type=float)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--budget-target", type=float, metavar="APPETITE", help="print the appetite-consistent budget target and exit")
    ap.add_argument("--it-budget", type=float, help="total IT budget incl. salaries, for the budget position")
    ap.add_argument("--cyber-spend", type=float, help="current cybersecurity spending, for the gap to target")
    a = ap.parse_args(argv)
    try:
        if a.budget_target is not None:
            print(f"Appetite {a.budget_target:.2f}: budget target {budget_target(a.budget_target):.2%} of total IT budget incl. salaries")
            return 0
        if not a.scenarios:
            ap.error("give a scenarios JSON file ('-' reads it from standard input) or --budget-target")
        data = json.load(sys.stdin) if a.scenarios == "-" else json.load(open(a.scenarios, encoding="utf-8"))
        r = run(data, a.appetite, a.factor)
        if a.it_budget:
            cost = sum(x.get("cost", 0.0) for x in r["results"])
            r["budget"] = budget_position(r["appetite"], a.it_budget, a.cyber_spend, cost)
    except ValueError as e:
        print(f"CyberRiskGuardian calculator: {e}", file=sys.stderr)
        return 2
    if a.json:
        json.dump(r, sys.stdout, indent=1); return 0
    for w in r["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"Appetite {r['appetite']:.2f}  Factor {r['factor']:,.0f}  Budget target {r['budget_target_pct']:.2%} of IT budget")
    has_ce = any("ce" in x for x in r["results"])
    print(f"{'ID':5}{'CVSS':>6}{'Estimated':>11}{'Tolerated':>11}{'Residual':>10}{'Ratio':>7}" + (f"{'CE':>8}" if has_ce else "") + "  Status")
    for x in r["results"]:
        ce = (f"{x['ce']:8.2f}" if "ce" in x else f"{'':8}") if has_ce else ""
        ratio = "n/a" if x["ratio"] is None else f"{x['ratio']:.2f}"
        print(f"{x['id']:5}{x['cvss']:6.1f}{x['est']:11,.0f}{x['tol']:11,.0f}{x['res']:10,.0f}{ratio:>7}{ce}  {x['cls']}")
    t = r["totals"]
    reduction = f"{1 - t['res'] / t['est']:.0%}" if t["est"] else "n/a"
    print(f"Total estimated {t['est']:,.0f}  residual {t['res']:,.0f}  reduction {reduction}")
    b = r.get("budget")
    if b:
        line = f"Budget target {b['target_pct']:.2%} = {b['target_amount']:,.0f}"
        if "planned_total" in b:
            ia = b["implied_appetite"]
            line += (f"; planned {b['planned_total']:,.0f} ({b['planned_pct']:.2%}); gap {b['gap_amount']:+,.0f} "
                     f"({b['gap_pct'] * 100:+.2f} points, negative = below target); {b['status']}; "
                     f"implied appetite {'outside the guideline (below 4%)' if ia is None else f'{ia:.2f}'}")
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
