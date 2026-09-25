# -*- coding: utf-8 -*-
"""Cybersecurity budget model linked to risk appetite and the 30-scenario assessment."""
import copy, json
import data30 as M
from data import classify, calc

IT_BUDGET = 100_000_000          # E3 - case: total IT budget incl. salaries (evidence gap - validate)
BAND = {"min": 0.04, "median": 0.078, "max": 0.12}   # user-supplied guideline
FTE = 1950 * 80                  # workbook FTE convention (CAD 156,000/yr)

def target_pct(appetite):
    """Piecewise-linear map of appetite (0-1) to % of IT budget.
    0.30 (risk averse) -> 12%;  0.50 (neutral) -> 7.8%;  0.70 (risk seeking) -> 4%. Clamped outside."""
    a = appetite
    if a <= 0.30: return BAND["max"]
    if a <= 0.50: return BAND["max"] + (a - 0.30) / 0.20 * (BAND["median"] - BAND["max"])
    if a <= 0.70: return BAND["median"] + (a - 0.50) / 0.20 * (BAND["min"] - BAND["median"])
    return BAND["min"]

# cyber share of each portfolio initiative (rest is funded from the IT project budget)
CYBER_SHARE = {"I01": 1.0, "I02": 1.0, "I03": 1.0, "I04": 0.5, "I05": 0.75, "I06": 0.5, "I07": 0.7, "I08": 1.0,
               "I09": 1.0, "I10": 1.0, "I11": 0.7, "I12": 1.0, "I13": 0.5, "I14": 0.8, "I15": 1.0, "I16": 1.0,
               "I17": 0.3, "I18": 0.0, "I19": 0.5, "I20": 0.5, "I21": 0.5, "I22": 0.3, "I23": 1.0}
SHARE_WHY = {
    "I04": "Backup platform is IT disaster recovery; immutability and isolation are security.",
    "I05": "Scanning, SLAs and the analyst FTE are security; patch deployment effort sits in IT operations.",
    "I06": "Network re-architecture is IT; security zoning and NAC policy are security.",
    "I07": "Device inventory and monitoring are security; biomedical engineering effort is clinical/IT.",
    "I11": "IR plan, retainer and exercises are security; clinical downtime procedures are operations.",
    "I13": "Integrity monitoring is security; EHR reconciliation work is application support.",
    "I14": "Security tooling and pen tests are security; developer remediation time is IT.",
    "I17": "MDM and endpoint encryption are IT endpoint management; disposal certification is security.",
    "I18": "Carrier redundancy, SD-WAN and failover are IT infrastructure resilience.",
    "I19": "Data discovery is security/privacy; consolidation and referral channel are IT.",
    "I20": "Policy and web controls are security; the sanctioned AI tool licence is IT.",
    "I21": "Verification procedure is security; help-desk tooling and time are IT.",
    "I22": "TLS on interfaces and platform hardening split with integration and telehealth delivery.",
}

BASELINE = [  # existing run cost, before this plan - Analyst estimate, validation required
    ("B1", "Existing cybersecurity staff (2 FTE)", 2 * FTE, "E3: two dedicated staff; FTE valued at workbook convention 1,950 h x $80."),
    ("B2", "Existing security tooling and support (firewalls, IDS/IPS, antimalware/EDR, e-mail filtering)", 600_000, "E2 lists these controls; cost is an analyst estimate."),
]

STAFF = [  # new security roles (cyber 100%); month of hire
    ("H1", "IAM / PAM engineer", 3, "S1, S5, S20, S23, S27, S28"),
    ("H2", "Security architect (cloud and network)", 6, "S4, S6, S11, S12, S16, S20"),
    ("H3", "Governance, risk and privacy-security analyst", 6, "S5, S9, S19, S21, S26 and KRI reporting"),
    ("H4", "Application security engineer", 12, "S11, S18, S19"),
    ("H5", "Medical device / OT security specialist", 12, "S7, S24, S29"),
]

# second-wave treatments (enhancements) - id, name, scenarios, initial, recurring, start, end, cyber share, description
SECOND = [
    ("E01", "Zero-trust network access replacing VPN", ["S1", "S6", "S9", "S20", "S23", "S27"], 400_000, 150_000, 12, 24, 1.0, "Identity- and device-aware access per application; removes exposed VPN appliance and brokers vendor access."),
    ("E02", "Identity governance and administration (IGA)", ["S5", "S10", "S27", "S28"], 300_000, 100_000, 9, 18, 1.0, "Automated joiner-mover-leaver, access certification and separation-of-duties rules."),
    ("E03", "Data security posture management and advanced DLP", ["S4", "S11", "S17", "S21", "S26"], 150_000, 120_000, 9, 15, 1.0, "Continuous discovery and classification of PHI across cloud, clinic storage and SaaS."),
    ("E04", "Secondary site / cloud disaster recovery for EHR and core services", ["S2", "S3", "S16", "S28"], 1_200_000, 300_000, 12, 30, 0.2, "Warm standby for EHR, identity and imaging with tested failover."),
    ("E05", "Legacy medical-device replacement and isolation fund", ["S7", "S24", "S29"], 900_000, 0, 12, 36, 0.2, "Replace or isolate devices that cannot be patched or monitored."),
    ("E06", "Independent assurance programme", ["S8", "S18", "S19", "S28"], 0, 250_000, 3, 3, 1.0, "Annual red-team / penetration test, cyber audit by internal audit, compromise assessment."),
    ("E07", "Software supply-chain security", ["S11", "S18", "S19"], 80_000, 50_000, 6, 12, 1.0, "SBOM collection, software composition analysis, staged update environment for critical software."),
    ("E08", "SaaS and cloud backup", ["S2", "S20"], 40_000, 60_000, 3, 6, 0.5, "Independent backup of cloud mail, files and tenant configuration."),
    ("E09", "Cyber insurance (risk transfer)", ["S2", "S9", "S19", "S28"], 0, 350_000, 0, 0, 1.0, "Placeholder premium - broker quotation required (Evidence gap)."),
]
RESERVE_PCT = 0.10  # contingency and incident reserve, % of cyber spend before reserve

# improved reductions when second wave is funded (Analyst estimate - validation required)
ENHANCED = {
    "S1": (0.85, 0.80), "S2": (0.80, 0.80), "S3": (0.85, 0.90), "S4": (0.90, 0.75), "S5": (0.70, 0.70),
    "S6": (0.90, 0.70), "S7": (0.80, 0.72), "S8": (0.75, 0.80), "S9": (0.60, 0.75), "S10": (0.75, 0.80),
    "S11": (0.90, 0.70), "S16": (0.85, 0.80), "S17": (0.80, 0.75), "S18": (0.85, 0.70), "S19": (0.50, 0.70),
    "S20": (0.85, 0.75), "S21": (0.70, 0.60), "S23": (0.90, 0.70), "S24": (0.80, 0.70), "S26": (0.75, 0.65),
    "S27": (0.85, 0.70), "S28": (0.65, 0.70), "S29": (0.80, 0.75),
}

MONTHS = 36
def mth(s): return int(s.split()[1])

def phase(initial, recurring, start, end, share=1.0):
    """Monthly cash flow: initial spread evenly over [start, end) (or month `start` if equal); recurring from `end`."""
    out = [0.0] * MONTHS
    span = max(1, end - start)
    for m in range(start, min(start + span, MONTHS)):
        out[m] += initial / span
    for m in range(end, MONTHS):
        out[m] += recurring / 12
    return [x * share for x in out]

def years(v): return [sum(v[0:12]), sum(v[12:24]), sum(v[24:36])]

lines = []   # (group, id, name, cyber_y, it_y, kind)  kind: run / project
def add(group, iid, name, cyber, it, kind):
    lines.append(dict(group=group, id=iid, name=name, cyber=years(cyber), it=years(it), kind=kind))

for b in BASELINE:
    add("A. Existing baseline", b[0], b[1], [b[2] / 12] * MONTHS, [0] * MONTHS, "run")
for i in M.INITIATIVES:
    iid, name = i[0], i[1]
    st, en = mth(i[8]), mth(i[9])
    sh = CYBER_SHARE[iid]
    ini_c = phase(i[6], 0, st, en, sh); ini_i = phase(i[6], 0, st, en, 1 - sh)
    rec_c = phase(0, i[7], st, en, sh); rec_i = phase(0, i[7], st, en, 1 - sh)
    add("B. Risk-treatment portfolio - projects", iid, name, ini_c, ini_i, "project")
    add("C. Risk-treatment portfolio - run", iid, name, rec_c, rec_i, "run")
for h in STAFF:
    add("D. Security team expansion", h[0], h[1], phase(0, FTE, h[2], h[2]), [0] * MONTHS, "run")
for e in SECOND:
    iid, name, scs, ini, rec, st, en, sh, desc = e
    add("E. Second-wave treatment - projects", iid, name, phase(ini, 0, st, en, sh), phase(ini, 0, st, en, 1 - sh), "project")
    add("F. Second-wave treatment - run", iid, name, phase(0, rec, st, en, sh), phase(0, rec, st, en, 1 - sh), "run")

def total(groups, field="cyber"):
    t = [0, 0, 0]
    for l in lines:
        if l["group"][0] in groups:
            t = [a + b for a, b in zip(t, l[field])]
    return t

OPTIONS = {
    "A": ("Baseline only (status quo)", "A"),
    "B": ("Baseline + 30-scenario portfolio", "ABC"),
    "C": ("B + security team expansion", "ABCD"),
    "D": ("C + second-wave treatment (risk-driven budget)", "ABCDEF"),
}
res = {}
for k, (label, g) in OPTIONS.items():
    pre = total(g)
    reserve = [x * RESERVE_PCT for x in pre] if k in ("C", "D") else [0, 0, 0]
    cyber = [a + b for a, b in zip(pre, reserve)]
    itproj = total(g, "it")
    res[k] = dict(label=label, groups=g, pre=pre, reserve=reserve, cyber=cyber, pct=[c / IT_BUDGET for c in cyber], it=itproj)

# project budgets (one-time) for recommended option
proj_cyber = total("BE"); proj_it = total("BE", "it")
run_cyber = total("ACDF")

# residual risk by option
def scen_results(enhanced):
    out = {}
    for s in M.SCEN:
        t = copy.deepcopy(s)
        if enhanced and s["id"] in ENHANCED:
            t["red_p"], t["red_i"] = ENHANCED[s["id"]]
        r = calc(t)
        out[s["id"]] = dict(res=r["res"], ratio=r["ratio"], cls=classify(r["ratio"]), p=t["red_p"], q=t["red_i"])
    return out
BASE_RES = scen_results(False)
ENH_RES = scen_results(True)
NOTREAT = {s["id"]: dict(res=s["est"], ratio=s["pre_ratio"], cls=classify(s["pre_ratio"])) for s in M.SCEN}

def counts(r):
    from collections import Counter
    c = Counter(v["cls"] for v in r.values())
    return [c.get("Above tolerance", 0), c.get("Approximately at tolerance", 0), c.get("Below tolerance", 0)]

# ---------------------------------------------------------------- appetite-alignment tranche (Tranche 2)
RAMP = [0.07, 0.095, 0.12]          # recommended % of IT budget by year (ramp limited by delivery capacity)
T2_ROLES = [("H6", "Detection engineer (SOC)", 12), ("H7", "Detection engineer (SOC)", 18), ("H8", "Threat intelligence and vulnerability analyst", 12),
            ("H9", "Security project manager", 6), ("H10", "Security awareness and culture lead", 12), ("H11", "Privacy-security engineer", 18)]
T2_FIXED = [  # id, name, scenarios, [y1, y2, y3] cyber CAD, rationale
    ("T2-2", "Security-by-design for clinical innovation and modernization", "All new technology; S7, S18, S26, S29", [150_000, 400_000, 400_000], "Architecture review, threat and privacy impact assessments, security requirements for 2024-2030 EHR, imaging and AI projects (moderate appetite for governed innovation, E5)."),
    ("T2-3", "Medical-device and clinical-system modernization - security share", "S7, S24, S29", [0, 750_000, 1_500_000], "Accelerates E05 beyond the initial fund: replace or wrap devices that cannot meet the security baseline."),
    ("T2-4", "Resilience deepening", "S2, S3, S16, S28", [200_000, 600_000, 900_000], "Immutable backup extended to all clinic storage, full-scale crisis exercises with clinics, EHR high-availability security share."),
]
def t2_roles_years():
    t = [0, 0, 0]
    for h in T2_ROLES:
        y = years(phase(0, FTE, h[2], h[2]))
        t = [a + b for a, b in zip(t, y)]
    return t
T2_ROLES_Y = t2_roles_years()
T2_FIXED_Y = [sum(x[3][i] for x in T2_FIXED) for i in range(3)]
TARGET_Y = [r * IT_BUDGET for r in RAMP]
T2_RESERVE_Y = [TARGET_Y[i] - res["D"]["cyber"][i] - T2_ROLES_Y[i] - T2_FIXED_Y[i] for i in range(3)]
assert all(x >= 0 for x in T2_RESERVE_Y), T2_RESERVE_Y
REC_Y = [res["D"]["cyber"][i] + T2_ROLES_Y[i] + T2_FIXED_Y[i] + T2_RESERVE_Y[i] for i in range(3)]

def implied_appetite(pct):
    if pct >= BAND["max"]: return 0.30
    if pct >= BAND["median"]: return 0.30 + (BAND["max"] - pct) / (BAND["max"] - BAND["median"]) * 0.20
    if pct >= BAND["min"]: return 0.50 + (BAND["median"] - pct) / (BAND["median"] - BAND["min"]) * 0.20
    return None  # below band floor

# pessimistic-case check: all 30 scenarios shifted +0.10 (as in s.9), with and without second wave
import data as _b
def pess(enh):
    n = 0
    for s in M.SCEN:
        t = copy.deepcopy(s)
        if enh and s["id"] in ENHANCED: t["red_p"], t["red_i"] = ENHANCED[s["id"]]
        t = _b.shift(t, 0.10); r = calc(t)
        n += r["res"] / r["tol"] > 1.10
    return n
PESS_ABOVE = {"B": pess(False), "D": pess(True)}

# appetite sensitivity of the target
APP_TABLE = [(a, target_pct(a), target_pct(a) * IT_BUDGET) for a in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]

# IT budget sensitivity (evidence gap)
IT_SENS = []
for itb in (100e6, 50e6, 25e6, 10e6):
    rec = res["D"]["cyber"]
    IT_SENS.append((itb, [c / itb for c in rec], [target_pct(M.APPETITE) * itb] * 3))

if __name__ == "__main__":
    print("target @0.30", target_pct(0.30), "@0.40", target_pct(0.40))
    for k, r in res.items():
        print(k, r["label"], [round(x / 1e6, 2) for x in r["cyber"]], [f"{p:.1%}" for p in r["pct"]], "IT proj", [round(x / 1e6, 2) for x in r["it"]])
    print("proj cyber", [round(x/1e6,2) for x in proj_cyber], "proj IT", [round(x/1e6,2) for x in proj_it], "run cyber", [round(x/1e6,2) for x in run_cyber])
    print("counts none/base/enh", counts(NOTREAT), counts(BASE_RES), counts(ENH_RES))
    print("above enh:", [k for k, v in ENH_RES.items() if v["cls"].startswith("Above")])
    print("tot res base", sum(v["res"] for v in BASE_RES.values()), "enh", sum(v["res"] for v in ENH_RES.values()), "est", sum(s["est"] for s in M.SCEN))
    for s in IT_SENS: print(s[0], [f"{p:.1%}" for p in s[1]])
    print("T2 roles", T2_ROLES_Y, "fixed", T2_FIXED_Y, "reserve", T2_RESERVE_Y, "REC", REC_Y)
    print("implied appetite D", [round(implied_appetite(p),2) for p in res["D"]["pct"]], "status quo", implied_appetite(res["A"]["pct"][0]))
    print("pessimistic above", PESS_ABOVE)

# ---------------------------------------------------------------- export
BUDGET_KRIS = [
    ("KRI-21", "Cybersecurity budget as % of total IT budget (incl. salaries)", "All", "Approved cyber budget / IT budget", "Finance, budget", "CFO / CIO", "Annual", "12% (appetite 0.30)", "< 7.8%", "< 4% (guideline floor)"),
    ("KRI-22", "% of cyber budget spent on scenario-linked treatments", "All", "Spend mapped to S1-S30 / cyber spend", "Budget tracking", "Cybersecurity lead", "Quarterly", "> 80%", "< 70%", "< 50%"),
    ("KRI-23", "Risk-reduction reserve released against approved business cases", "All", "Released / available reserve, with scenario IDs", "Steering Committee minutes", "Executive sponsor", "Quarterly", "Every release tied to a scenario", "Release without scenario link", "Reserve used for non-risk items"),
]
if __name__ == "__main__":
    out = dict(IT_BUDGET=IT_BUDGET, BAND=BAND, FTE=FTE, TARGET_030=target_pct(0.30), TARGET_040=target_pct(0.40),
               CYBER_SHARE=CYBER_SHARE, SHARE_WHY=SHARE_WHY, BASELINE=BASELINE, STAFF=STAFF, SECOND=SECOND, RESERVE_PCT=RESERVE_PCT,
               ENHANCED=ENHANCED, lines=lines, OPTIONS=res, proj_cyber=proj_cyber, proj_it=proj_it, run_cyber=run_cyber,
               BASE_RES=BASE_RES, ENH_RES=ENH_RES, COUNTS=dict(none=counts(NOTREAT), base=counts(BASE_RES), enh=counts(ENH_RES)),
               APP_TABLE=APP_TABLE, IT_SENS=IT_SENS, RAMP=RAMP, T2_ROLES=T2_ROLES, T2_FIXED=T2_FIXED, T2_ROLES_Y=T2_ROLES_Y,
               T2_FIXED_Y=T2_FIXED_Y, T2_RESERVE_Y=T2_RESERVE_Y, TARGET_Y=TARGET_Y, REC_Y=REC_Y,
               IMPLIED_D=[implied_appetite(p) for p in res["D"]["pct"]], PESS_ABOVE=PESS_ABOVE, BUDGET_KRIS=BUDGET_KRIS,
               TOT_EST=sum(s["est"] for s in M.SCEN), TOT_RES_BASE=sum(v["res"] for v in BASE_RES.values()), TOT_RES_ENH=sum(v["res"] for v in ENH_RES.values()))
    json.dump(out, open("budget.json", "w"), ensure_ascii=False, indent=1)
    print("exported")
