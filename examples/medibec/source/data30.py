# -*- coding: utf-8 -*-
"""30-scenario model: S1-S10 unchanged + S11-S30."""
import copy, json
import data as base
from new_scen import NEW, NEW_INITS, NEW_KRIS, NEW_CANDIDATES, UPDATED_DECISIONS
from cvss import CVSS4

APPETITE, FACTOR, CASE_APPETITE = base.APPETITE, base.FACTOR, base.CASE_APPETITE
calc, classify, case_norm = base.calc, base.classify, base.case_norm
EVIDENCE, CROWN = base.EVIDENCE, base.CROWN

SCEN = [copy.deepcopy(s) for s in base.SCEN] + [copy.deepcopy(s) for s in NEW]
assert len(SCEN) == 30 and len({s["id"] for s in SCEN}) == 30

# initiatives: base (13 fields incl. scenario list) + new (without list); derive lists from scenario inits
raw = [(i[0], i[1], i[2], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11], i[12]) for i in base.INITIATIVES] + NEW_INITS
INITIATIVES = []
for (iid, name, desc, owner, prio, ini, rec, st, en, succ, dep, typ) in raw:
    scs = [s["id"] for s in SCEN if iid in s["inits"]]
    assert scs, iid
    INITIATIVES.append((iid, name, desc, scs, owner, prio, ini, rec, st, en, succ, dep, typ))
# base initiatives must still cover their original scenarios
for bi in base.INITIATIVES:
    ni = next(i for i in INITIATIVES if i[0] == bi[0])
    assert set(bi[3]) <= set(ni[3]), bi[0]
used = {x for s in SCEN for x in s["inits"]}
assert used == {i[0] for i in INITIATIVES}, used ^ {i[0] for i in INITIATIVES}

CANDIDATES = []
for c in base.CANDIDATES:
    c = list(c)
    if c[0] in UPDATED_DECISIONS:
        c[7], c[8] = UPDATED_DECISIONS[c[0]]
    CANDIDATES.append(tuple(c))
CANDIDATES += NEW_CANDIDATES
assert len(CANDIDATES) == 45
sel = [c for c in CANDIDATES if c[7].startswith("Selected")]
assert len(sel) == 30, len(sel)

KRIS = base.KRIS + NEW_KRIS

for s in SCEN:
    v = CVSS4(s["cvss"])
    s["cvss_score"], s["cvss_sev"] = v.base_score, v.severities()[0]
    s["alloc_initial"] = s["alloc_recurring"] = 0.0
for i in INITIATIVES:
    n = len(i[3])
    for sid in i[3]:
        s = next(x for x in SCEN if x["id"] == sid)
        s["alloc_initial"] += i[6] / n
        s["alloc_recurring"] += i[7] / n
for s in SCEN:
    s.update(calc(s))
    s["cost_y1"] = s["alloc_initial"] + s["alloc_recurring"]
    s["ce"] = (s["est"] - s["res"]) / s["cost_y1"] * 1000
    s["class"] = classify(s["ratio"])
    s["norm_cur"], s["norm_post"] = case_norm(s)
    s["ratio_case"] = calc(s, appetite=CASE_APPETITE)["ratio"]
    s["class_case"] = classify(s["ratio_case"])

SENS_IDS = ["S2", "S8", "S1", "S7", "S9", "S16", "S19", "S20", "S27", "S28"]
SENS = []
for sid in SENS_IDS:
    s = next(x for x in SCEN if x["id"] == sid)
    row = {"id": sid, "name": s["name"]}
    for lab, d in (("lower", -0.10), ("central", 0.0), ("higher", 0.10)):
        t = base.shift(s, d); r = calc(t)
        row[lab] = dict(est=r["est"], res=r["res"], ratio=r["res"] / r["tol"], cls=classify(r["res"] / r["tol"]))
    SENS.append(row)

TOT_INITIAL = sum(i[6] for i in INITIATIVES)
TOT_RECUR = sum(i[7] for i in INITIATIVES)

if __name__ == "__main__":
    for s in sorted(SCEN, key=lambda s: -s["est"]):
        print(f'{s["id"]:4} CVSS {s["cvss_score"]:4} est {s["est"]:8.0f} res {s["res"]:7.0f} ratio {s["ratio"]:.2f} {s["class"][:6]} @0.4 {s["ratio_case"]:.2f} {s["class_case"][:6]} norm {s["norm_cur"]:.3f}->{s["norm_post"]:.3f} cost {s["cost_y1"]:8.0f} CE {s["ce"]:.2f}')
    print("initial", TOT_INITIAL, "recurring", TOT_RECUR, "Y1", TOT_INITIAL + TOT_RECUR)
    te = sum(s["est"] for s in SCEN); tr = sum(s["res"] for s in SCEN); print("tot", te, tr, 1 - tr / te)
    for r in SENS:
        print(r["id"], {k: (round(r[k]["ratio"], 2), r[k]["cls"][:6]) for k in ("lower", "central", "higher")})
    from collections import Counter
    print(Counter(s["class"] for s in SCEN), Counter(s["class_case"] for s in SCEN))
    json.dump(dict(APPETITE=APPETITE, FACTOR=FACTOR, CASE_APPETITE=CASE_APPETITE, EVIDENCE=EVIDENCE, CROWN=CROWN,
                   CANDIDATES=CANDIDATES, INITIATIVES=INITIATIVES, SCEN=SCEN, KRIS=KRIS, SENS=SENS,
                   TOT_INITIAL=TOT_INITIAL, TOT_RECUR=TOT_RECUR), open("data30.json", "w"), ensure_ascii=False, indent=1)
