
// ================================================================ 4. APPETITE
add(h1("4. Risk appetite"));
add(callout(`**Estimated Risk Appetite: ${f2(D.APPETITE)}** (risk averse). Analyst estimate - validation required. Unchanged from the 10-scenario assessment.`, "DEEAF6"));
add(spacer());
add(p("**Rationale.**"));
add(bl("**Sector and mission** - healthcare delivery where cyber events can affect clinical decisions; MediBec states a **low** appetite for risks to patient safety, health-data confidentiality, compliance and public trust (E5)."));
add(bl("**Information sensitivity** - EHR, imaging, lab, pharmacy and billing data are among the most sensitive personal information."));
add(bl("**Regulatory exposure** - privacy-incident obligations and professional duties amplify the consequences of any breach."));
add(bl("**Operational dependence** - all 10 clinics depend on central services."));
add(bl("**Capacity** - a two-person team limits MediBec's ability to absorb and respond to incidents."));
add(bl("**Moderating factor** - moderate appetite for governed innovation (E5) argues against an extreme value."));
add(p(`**Relationship to the case value (${f2(D.CASE_APPETITE)}).** The case's value blends low appetite for clinical and privacy risk with moderate appetite for innovation and was designed for a different scale. A few of the new scenarios are financial or reputational rather than clinical (S13, S14, S30); a differentiated appetite by risk category could be considered. **Management decision required.**`));
add(table(["Scenario", "Ratio @ 0.20", "Ratio @ 0.30 (base)", "Ratio @ 0.40 (case)"],
  S.map((s) => { const b = s.res / s.tol * D.APPETITE; return [`${s.id} ${s.name}`, f2(b / 0.2), f2(b / 0.3), f2(b / 0.4)]; }),
  [55, 15, 15, 15], { size: 14, fillFn: (r, i) => (i > 0 ? (parseFloat(r[i]) > 1.1 ? "F4CCCC" : parseFloat(r[i]) >= 0.9 ? "FFF2CC" : "D9EAD3") : undefined) }));
add(spacer());
add(p(`At 0.40, ${case40.above.length} scenarios stay above tolerance (${listIds(case40.above)}) and ${case40.approx.length} sit approximately at it (${listIds(case40.approx)}).`));

// ================================================================ 5. SCENARIO DEVELOPMENT
add(h1("5. Scenario development and selection"));
add(p("Forty-five candidates were generated from seven perspectives - top-down (clinical continuity, privacy), bottom-up (documented weaknesses and audit universe), threat-driven, asset-driven (crown jewels), third-party, human and resilience - using the case scenario library (MED-CYR-01 to 15), the exercise evidence and the extended catalogue (Annex B). C01-C20 are the original candidates with updated decisions; C21-C45 are new."));
add(table(["ID", "Candidate", "Threat source", "Vulnerability / condition", "Asset", "Outcome", "Ref", "Decision", "Rationale"],
  D.CANDIDATES, [4, 15, 9, 14, 7, 10, 8, 9, 24], { size: 13, fillFn: (r, i) => (i === 7 ? (r[7].startsWith("Selected") ? "D9EAD3" : r[7].startsWith("Merged") ? "FFF2CC" : "F2F2F2") : undefined) }));
add(spacer());
const nMerged = D.CANDIDATES.filter((c) => c[7].startsWith("Merged")).length, nNot = D.CANDIDATES.filter((c) => c[7].startsWith("Not")).length;
add(p(`**Screening:** 30 selected, ${nMerged} merged into a selected scenario with the same causal chain, ${nNot} not selected (low consequence, out of scope or no evidence of exposure). Criteria: plausibility over 12 months, consequence severity (patient safety and privacy weighted highest), evidence of exposure, uniqueness of the causal chain, regulatory and operational significance. CVSS played no role in selection.`));
add(p("**Distinctness of related scenarios.** Several new scenarios share an asset with an original one but differ in threat path or weakness: S20 (cloud control plane) vs. S4 (storage exposure); S27 (help-desk reset) and S23 (exposed admin interfaces) vs. S1 (phishing); S19 (trojanized update) vs. S9 (vendor remote access); S29 (interface tampering) vs. S10 (record changes); S28 (destructive insider) vs. S5 (data-theft insider); S12 (Wi-Fi entry) vs. S7 (device foothold). This separation matters because each needs a different control."));

// ================================================================ 6. PORTFOLIO TABLE
add(h1("6. Risk scenario portfolio"));
add(table(["ID", "Scenario", "CVSS-B", "Estimated", "Tolerated", "Residual", "Res / Tol", "Status", "Key treatment"],
  S.map((s) => [s.id, s.name, s.cvss_score.toFixed(1), fmt0(s.est), fmt0(s.tol), fmt0(s.res), f2(s.ratio), s.class, s.inits.join(", ")]),
  [4, 30, 6, 8, 8, 8, 7, 13, 16], { size: 13, fillFn: (r, i) => (i === 7 ? clsFill(r[7]) : undefined) }));
add(spacer());
add(p(`Totals: estimated ${fmt0(totE)}, tolerated ${fmt0(S.reduce((a, s) => a + s.tol, 0))}, residual ${fmt0(totR)} risk units. Initiative names are in Section 10.`));
