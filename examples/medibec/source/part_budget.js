
// ================================================================ 11. BUDGET
add(h1("11. Cybersecurity budget and funding strategy"));
{
  const pc = (x) => (x * 100).toFixed(1) + "%";
  const m1 = (x) => "$" + (x / 1e6).toFixed(2) + "M";
  const grp = (letter, field = "cyber") => [0, 1, 2].map((y) => BJ.lines.filter((l) => l.group[0] === letter).reduce((a, l) => a + l[field][y], 0));
  const sum3 = (arr) => arr.reduce((a, b) => a + b, 0);
  const O = BJ.OPTIONS;
  const contD = O.D.reserve;
  const t2roles = BJ.T2_ROLES_Y, t2fix = BJ.T2_FIXED_Y, t2res = BJ.T2_RESERVE_Y;

  add(p("MediBec's cybersecurity budget is set with the same logic as its risk decisions: the risk appetite fixes where the budget should sit within the guideline band, and the risk assessment shows what the money must buy. This section derives the appetite-consistent budget, compares it with the bottom-up cost of the treatments identified in Sections 7-10, reconciles the two, and sets out the cybersecurity and IT project budgets."));

  add(h2("11.1 Guideline and appetite-consistent target"));
  add(table(["Position in guideline band", "% of total IT budget (incl. salaries)", "Appetite anchor", "CAD at IT budget of " + money(BJ.IT_BUDGET)], [
    ["Floor - risk seeking", pc(BJ.BAND.min), "0.70 (relatively high appetite)", money(BJ.BAND.min * BJ.IT_BUDGET)],
    ["Median - risk neutral", pc(BJ.BAND.median), "0.50 (risk neutral)", money(BJ.BAND.median * BJ.IT_BUDGET)],
    ["Ceiling - risk averse", pc(BJ.BAND.max), "0.30 (risk averse)", money(BJ.BAND.max * BJ.IT_BUDGET)],
  ], [30, 22, 24, 24]));
  add(spacer());
  add(p(`The appetite scale and the guideline are aligned by linear interpolation between the anchors (0.30 -> 12%, 0.50 -> 7.8%, 0.70 -> 4%), clamped at the band limits. With the estimated appetite of **${f2(D.APPETITE)}**, the **appetite-consistent cybersecurity budget is ${pc(BJ.TARGET_030)} of the IT budget - ${money(BJ.TARGET_030 * BJ.IT_BUDGET)} per year**. At the case's 0.40 it would be ${pc(BJ.TARGET_040)} (${money(BJ.TARGET_040 * BJ.IT_BUDGET)}). The IT budget of ${money(BJ.IT_BUDGET)} is the case figure (E3) and is an **Evidence gap - organizational input required** (see 11.5).`));
  add(table(["Appetite", "Target % of IT budget", "Target CAD"], BJ.APP_TABLE.map((r) => [f2(r[0]), pc(r[1]), money(r[2])]), [30, 35, 35]));

  add(h2("11.2 Current position"));
  add(p(`The case documents two cybersecurity staff and perimeter, antimalware, EDR and e-mail controls, but no cybersecurity budget. Valuing the two staff at the workbook convention and the existing tooling at an analyst estimate gives a current run cost of about **${money(O.A.cyber[0])} - ${pc(O.A.pct[0])} of the IT budget, below the 4% guideline floor**. Even if the true figure were several times higher, MediBec is likely under-invested relative to both the guideline and its stated appetite. **Evidence gap - organizational input required**: actual security spend, including security embedded in IT contracts and licences.`));

  add(h2("11.3 Bottom-up (risk-driven) budget"));
  add(p("The treatments identified in this assessment were phased by fiscal year (October-September; Y1 = 2026-27): one-time costs are spread evenly between each initiative's start and end month and recurring costs start at completion. Each initiative is split between the cybersecurity budget and the IT budget according to its nature (11.7). This phased view differs from the simplified 'Year-1 cost' used for cost-effectiveness in Section 10, which charges all initial and a full year of recurring cost to Year 1."));
  const build = [
    ["A. Existing baseline (2 FTE + current tooling)", grp("A")],
    ["B. 30-scenario portfolio - projects (cyber share)", grp("B")],
    ["C. 30-scenario portfolio - run (cyber share)", grp("C")],
    ["D. Security team expansion (5 roles)", grp("D")],
    ["E. Second-wave treatment - projects (cyber share)", grp("E")],
    ["F. Second-wave treatment - run, incl. cyber insurance placeholder", grp("F")],
    ["Contingency and incident reserve (10%)", contD],
  ];
  add(table(["Component", "Y1", "Y2", "Y3", "3-year"], [...build.map(([n, v]) => [n, m1(v[0]), m1(v[1]), m1(v[2]), m1(sum3(v))]),
    ["**Risk-driven cybersecurity budget (option D)**", `**${m1(O.D.cyber[0])}**`, `**${m1(O.D.cyber[1])}**`, `**${m1(O.D.cyber[2])}**`, `**${m1(sum3(O.D.cyber))}**`],
    ["% of IT budget", pc(O.D.pct[0]), pc(O.D.pct[1]), pc(O.D.pct[2]), ""]], [46, 13, 13, 13, 15], { size: 15 }));
  add(spacer());
  add(p("**New roles (group D):** " + BJ.STAFF.map((h) => `${h[1]} (month ${h[2]}; ${h[3]})`).join("; ") + ". The vulnerability-management FTE is already in I05."));
  add(p("**Second-wave treatments (groups E-F)** were added because the 30-scenario portfolio leaves 12 scenarios above tolerance. Each is tied to those scenarios:"));
  add(table(["ID", "Treatment", "Scenarios", "Initial", "Recurring", "Months", "Cyber share", "Purpose"],
    BJ.SECOND.map((e) => [e[0], e[1], e[2].join(", "), k(e[3]), k(e[4]), `${e[5]}-${e[6]}`, pc(e[7]).replace(".0", ""), e[8]]), [5, 20, 13, 8, 9, 7, 8, 30], { size: 14 }));
  add(spacer());
  add(p("**Budget options compared with the guideline band:**"));
  const imp = (pctY) => { const x = pctY; if (x < BJ.BAND.min) return "below floor"; if (x >= BJ.BAND.max) return "0.30"; if (x >= BJ.BAND.median) return f2(0.3 + (BJ.BAND.max - x) / (BJ.BAND.max - BJ.BAND.median) * 0.2); return f2(0.5 + (BJ.BAND.median - x) / (BJ.BAND.median - BJ.BAND.min) * 0.2); };
  const band = (arr) => (Math.min(...arr) < BJ.BAND.min ? "Below floor" : Math.max(...arr) > BJ.BAND.max ? "Above ceiling" : "Within band");
  const recPct = BJ.REC_Y.map((x) => x / BJ.IT_BUDGET);
  const optRows = ["A", "B", "C", "D"].map((kk) => [O[kk].label, ...O[kk].cyber.map(m1), O[kk].pct.map(pc).join(" / "), band(O[kk].pct), imp(O[kk].pct[2])]);
  optRows.push(["**Recommended: D + Tranche 2 + risk-reduction reserve**", ...BJ.REC_Y.map(m1), recPct.map(pc).join(" / "), band(recPct), imp(recPct[2])]);
  add(table(["Option", "Y1", "Y2", "Y3", "% IT budget Y1 / Y2 / Y3", "Guideline band", "Implied appetite (Y3)"], optRows, [30, 9, 9, 9, 17, 12, 14],
    { size: 14, fillFn: (r, i) => (i === 5 ? (r[5] === "Within band" ? "D9EAD3" : "F4CCCC") : undefined) }));

  add(h2("11.4 What each funding level buys in risk reduction"));
  const C = BJ.COUNTS;
  add(table(["Funding level", "Above tolerance", "Approximately at", "Below tolerance", "Total residual risk"], [
    ["No treatment", C.none[0], C.none[1], C.none[2], fmt0(BJ.TOT_EST)],
    ["30-scenario portfolio (options B/C)", C.base[0], C.base[1], C.base[2], fmt0(BJ.TOT_RES_BASE)],
    ["Portfolio + second wave (option D and recommended)", C.enh[0], C.enh[1], C.enh[2], fmt0(BJ.TOT_RES_ENH)],
  ], [40, 15, 15, 15, 15]));
  add(spacer());
  const moved = S.filter((s) => BJ.BASE_RES[s.id].cls !== BJ.ENH_RES[s.id].cls);
  add(table(["ID", "Scenario", "P / Q portfolio", "P / Q with second wave", "Ratio portfolio", "Ratio second wave", "Status change"],
    S.filter((s) => BJ.ENHANCED[s.id]).map((s) => [s.id, s.name, `${f2(s.red_p)} / ${f2(s.red_i)}`, `${f2(BJ.ENHANCED[s.id][0])} / ${f2(BJ.ENHANCED[s.id][1])}`,
      f2(BJ.BASE_RES[s.id].ratio), f2(BJ.ENH_RES[s.id].ratio), BJ.BASE_RES[s.id].cls === BJ.ENH_RES[s.id].cls ? "-" : `${BJ.BASE_RES[s.id].cls.split(" ")[0]} -> ${BJ.ENH_RES[s.id].cls.split(" ")[0]}`]),
    [5, 33, 12, 13, 10, 10, 17], { size: 13, fillFn: (r, i) => (i === 5 ? clsFill(BJ.ENH_RES[r[0]].cls) : undefined) }));
  add(spacer());
  const stillAbove = S.filter((s) => BJ.ENH_RES[s.id].cls.startsWith("Above")).map((s) => s.id);
  add(p(`With second-wave funding, ${moved.length} scenarios change status and only **${stillAbove.join(", ")}** remain above tolerance - the vendor, software-supply-chain and privileged-insider scenarios that spending by MediBec alone cannot fix. They are addressed by transfer (insurance placeholder E09, contract terms) and formal acceptance, not by more budget. Second-wave reductions are **Analyst estimates - validation required**. Beyond option D the model shows sharply diminishing returns: further money does not move any remaining scenario within tolerance.`));
  add(p(`**Uncertainty argument for a risk-averse budget.** Under the pessimistic sensitivity case (all parameters shifted 0.10 against MediBec, Section 9), ${BJ.PESS_ABOVE.D} of 30 scenarios would be above tolerance even with second-wave funding. A risk-averse organization funds capacity to respond if estimates prove optimistic; that is the purpose of the Tranche 2 reserve below.`));

  add(h2("11.5 Reconciling the budget with the risk appetite"));
  add(p(`The risk-driven budget (option D) is **within the guideline band** (${O.D.pct.map(pc).join(", ")}), but at the risk-seeking end: its implied appetite is about **${BJ.IMPLIED_D.map((x) => f2(x)).join(" / ")}**, far from the 0.30 MediBec's clinical and privacy profile calls for. Budget and appetite are therefore inconsistent, and management has three coherent choices:`));
  add(nl("**Fund to the appetite (recommended).** Keep appetite at 0.30 and ramp the cybersecurity budget to 12% over three years, adding a Tranche 2 of capacity, security-by-design, device modernization, resilience and a gated risk-reduction reserve (11.6).", "num2"));
  add(nl("**Keep the risk-driven budget and revise the appetite.** Fund option D only and formally adopt an appetite near 0.60 - hard to defend for health data and patient safety, and inconsistent with the case statement of low appetite for these risks (E5).", "num2"));
  add(nl("**Validate the IT budget first.** If the true IT budget is nearer CAD 45-50M - plausible for 15 IT staff - option D alone is 10-12% and already appetite-consistent, and Tranche 2 is unnecessary.", "num2"));
  add(table(["IT budget (CAD)", "Option D % Y1", "Option D % Y2", "Option D % Y3", "Guideline status", "Appetite-consistent target (12%)"],
    BJ.IT_SENS.map((r) => [money(r[0]), pc(r[1][0]), pc(r[1][1]), pc(r[1][2]), band(r[1]) === "Above ceiling" ? "Above ceiling - phase or cut" : band(r[1]), money(r[2][0])]),
    [18, 14, 14, 14, 22, 18], { size: 15 }));
  add(spacer());
  add(p("Below an IT budget of about CAD 46M the risk-driven plan exceeds the 12% ceiling and must be phased or trimmed, starting with the lowest cost-effectiveness items in Tier 3 (Section 10). **Management decision required** once the IT budget is confirmed."));

  add(h2("11.6 Recommended cybersecurity budget"));
  add(callout(`**Recommended: ${pc(recPct[0])} / ${pc(recPct[1])} / ${pc(recPct[2])} of the IT budget - ${m1(BJ.REC_Y[0])} in Y1, ${m1(BJ.REC_Y[1])} in Y2, ${m1(BJ.REC_Y[2])} in Y3 - reaching the appetite-consistent 12% in Year 3 and holding it for the rest of the 2024-2030 horizon, subject to annual review.** The ramp reflects delivery capacity: a two-person team cannot absorb 12% in Year 1.`, "DEEAF6"));
  add(spacer());
  const recRows = [
    ...build.map(([n, v]) => [n, m1(v[0]), m1(v[1]), m1(v[2])]),
    ["G1. Tranche 2 - additional security roles (" + BJ.T2_ROLES.map((h) => h[1]).join(", ") + ")", m1(t2roles[0]), m1(t2roles[1]), m1(t2roles[2])],
    ...BJ.T2_FIXED.map((t) => [`G. ${t[0]} ${t[1]}`, m1(t[3][0]), m1(t[3][1]), m1(t[3][2])]),
    ["I. Risk-reduction investment reserve (gated)", m1(t2res[0]), m1(t2res[1]), m1(t2res[2])],
    ["**Total cybersecurity budget**", `**${m1(BJ.REC_Y[0])}**`, `**${m1(BJ.REC_Y[1])}**`, `**${m1(BJ.REC_Y[2])}**`],
    ["% of IT budget / guideline", pc(recPct[0]), pc(recPct[1]), pc(recPct[2]) + " (ceiling)"],
  ];
  add(table(["Line", "Y1", "Y2", "Y3"], recRows, [58, 14, 14, 14], { size: 14 }));
  add(spacer());
  add(p("**Tranche 2 content.** " + BJ.T2_FIXED.map((t) => `**${t[1]}** (${t[2]}): ${t[3 + 1]}`).join(" ")));
  add(p("**Governance of the reserve.** The risk-reduction reserve is not a spending target. It is released by the Cybersecurity and Privacy Steering Committee only against a business case that names the scenario IDs addressed, the expected change in probability and impact reduction, and the KRI that will show the effect (KRI-22, KRI-23). Unreleased reserve returns to the IT budget at year end. Priority uses: findings of the SME validation workshop, actual device-replacement quotations, the outcome of the IT-budget validation, and incident-driven needs."));

  add(h2("11.7 Budget for IT and cybersecurity projects"));
  const projC = [0, 1, 2].map((y) => BJ.proj_cyber[y] + BJ.T2_FIXED.filter((t) => t[0] !== "T2-2").reduce((a, t) => a + t[3][y], 0));
  const projI = BJ.proj_it;
  add(table(["Project budget (one-time spend)", "Y1", "Y2", "Y3", "3-year"], [
    ["Cybersecurity projects - committed (portfolio, second wave and Tranche 2 programmes, cyber share)", m1(projC[0]), m1(projC[1]), m1(projC[2]), m1(sum3(projC))],
    ["Cybersecurity projects - risk-reduction reserve (gated)", m1(t2res[0]), m1(t2res[1]), m1(t2res[2]), m1(sum3(t2res))],
    ["Security-relevant IT projects (IT share of I04, I06, I07, I11, I13, I14, I17-I22, E04, E05, E08)", m1(projI[0]), m1(projI[1]), m1(projI[2]), m1(sum3(projI))],
    ["**Total IT and cybersecurity projects**", `**${m1(projC[0] + t2res[0] + projI[0])}**`, `**${m1(projC[1] + t2res[1] + projI[1])}**`, `**${m1(projC[2] + t2res[2] + projI[2])}**`, `**${m1(sum3(projC) + sum3(t2res) + sum3(projI))}**`],
  ], [52, 12, 12, 12, 12], { size: 15 }));
  add(spacer());
  add(p("IT projects funded from the IT budget do not count toward the cybersecurity percentage but are prerequisites for several treatments (network resilience I18, secondary site E04, device replacement E05). The IT project figures cover only the security-relevant projects identified here; MediBec's wider IT project portfolio is outside this assessment. Split of shared initiatives (analyst convention, used for both budgets without double-counting):"));
  add(table(["Initiative", "Cyber share", "Rationale"], Object.entries(BJ.SHARE_WHY).map(([iid, why]) => [`${iid} ${INIT.find((i) => i[0] === iid)[1]}`, pc(BJ.CYBER_SHARE[iid]).replace(".0", ""), why]), [34, 10, 56], { size: 14 }));

  add(h2("11.8 Budget calibration as part of the risk-management process"));
  add(p("The budget is now a standing step of the CyberRiskGuardian cycle, repeated with each annual assessment:"));
  [
    "**Set the appetite** (Board) - 0.30 proposed.",
    "**Derive the appetite-consistent budget** - position in the 4%-12% band of total IT budget including salaries (0.30 -> 12%, 0.50 -> 7.8%, 0.70 -> 4%).",
    "**Cost the treatments** from the scenario assessment - phased, split between cyber and IT budgets, shared controls counted once.",
    "**Compare and compute the implied appetite** of the risk-driven budget; flag if below the 4% floor or above the 12% ceiling.",
    "**Reconcile** - fund to appetite, revise appetite, or accept residual risk formally; record the decision.",
    "**Release reserve only against scenario-linked business cases**; update P/Q in the workbook when funded.",
    "**Monitor** with KRI-21 to KRI-23 alongside the risk KRIs; report quarterly to the Steering Committee.",
    "**Re-run annually** - assessment, residual risk and budget together, aligned with the 2024-2030 roadmap.",
  ].forEach((x) => add(nl(x, "num3")));
}
