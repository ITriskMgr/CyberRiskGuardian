
// ================================================================ 8. CROSS-SCENARIO
add(h1("8. Cross-scenario analysis"));
add(h2("8.1 Recurring root causes"));
add(table(["Root cause", "Scenarios", "Observation"], [
  ["Identity that can be phished, reset or abused", "S1, S5, S20, S23, S27, S28, S9, S10", "Push MFA, standing privileges, un-vaulted service accounts, reusable cloud tokens, weak help-desk verification and exposed admin interfaces provide the main entry and escalation paths."],
  ["Limited detection and response capacity", "All (S8 amplifier)", "Two staff cannot provide 24/7 triage; weak signals in E6 were not correlated. Raises damage everywhere."],
  ["Unproven recoverability and single hub", "S2, S3, S10, S16, S28", "Backups, restore tests, integrity checks and hub redundancy are not evidenced."],
  ["Flat, heterogeneous clinic networks", "S2, S6, S7, S12, S24, S29", "Workstations, devices, Wi-Fi and cleartext clinical interfaces share networks."],
  ["Unknown attack surface", "S6, S11, S18, S23", "No inventory of internet-exposed services, secrets or application flaws."],
  ["Data outside central control", "S4, S15, S17, S21, S25, S26", "Cloud exports, local storage, mobile devices, disposed assets and AI tools hold PHI outside monitored stores."],
  ["Dependence on third parties", "S9, S19, S13, S30", "Vendors, software suppliers, ISPs and the public brand channel are outside MediBec's direct control."],
  ["Governance operating on paper", "All", "No KRI reporting, access recertification, tested IR, AI policy or payment verification evidenced."],
], [24, 20, 56]));
add(h2("8.2 Concentration and systemic dependencies"));
add(bl("**Montreal hub** - EHR, identity, backups and WAN converge on one site (S16); any enterprise-wide event affects all 10 clinics."));
add(bl("**Identity platform** - one directory, one MFA method and one cloud tenant gate every crown jewel (S1, S20, S27)."));
add(bl("**EHR and software suppliers** - concentration risk via both remote access (S9) and software updates (S19); supplier identity and contracts are an **Evidence gap**."));
add(bl("**Scenario chaining** - entry (S1, S6, S12, S20, S23, S24, S27) + undetected (S8) + no recovery (S3) = S2 at maximum damage. Treating I01, I03 and I04 breaks most chains."));
add(h2("8.3 Shared controls"));
add(table(["Initiative", "Scenarios", "Why it is shared"],
  INIT.filter((i) => i[3].length >= 3).map((i) => [`${i[0]} ${i[1]}`, i[3].join(", "), i[2]]), [26, 20, 54], { size: 14 }));
add(spacer());
add(p(`Shared initiatives are costed once; their Year-1 cost is split equally among the scenarios they address (Workbook, Portfolio sheet). I03 (MDR) now serves ${INIT.find((i) => i[0] === "I03")[3].length} scenarios and I12 (awareness and governance) ${INIT.find((i) => i[0] === "I12")[3].length}, which is why their allocated share per scenario is small. Dependencies: I02 needs the account inventory from I01; I06 needs asset and device inventories from I05 and I07; I13 needs logging from I03; I21 depends on I01 authenticators; I23 depends on DMARC from I16.`));

// ================================================================ 9. SENSITIVITY
add(h1("9. Sensitivity analysis"));
add(p("For ten key scenarios, all uncertain parameters were shifted together by 0.10: the lower-risk case lowers threat, exploitation and damage and raises resilience and treatment effectiveness; the higher-risk case does the reverse."));
add(table(["ID", "Scenario", "Lower: residual / ratio", "Central: residual / ratio", "Higher: residual / ratio", "Recommendation robust?"],
  D.SENS.map((r) => [r.id, r.name, `${fmt0(r.lower.res)} / ${f2(r.lower.ratio)}`, `${fmt0(r.central.res)} / ${f2(r.central.ratio)}`, `${fmt0(r.higher.res)} / ${f2(r.higher.ratio)}`,
    r.lower.cls === r.higher.cls ? `Yes - ${r.lower.cls.toLowerCase()} in all cases` : `No - ${r.lower.cls.toLowerCase()} to ${r.higher.cls.toLowerCase()}`]),
  [5, 30, 15, 15, 15, 20], { size: 14 }));
add(spacer());
add(p("**Findings.**"));
add(bl(`**${robustAbove.join(" and ")} stay above tolerance even in the lower-risk case.** Treatment within MediBec's control cannot bring them within appetite; transfer (insurance, contract terms) and formal Board acceptance are the robust recommendation.`));
add(bl("**S7 and S9** reach approximately-at tolerance only in the lower case: outcome depends on manufacturer and vendor cooperation."));
add(bl("**S1, S2, S8, S16, S20 and S27** flip between below and above tolerance: investment is justified in every case (estimated risk stays high), but acceptance of the residual depends on treatment effectiveness. Track with KRI-01, KRI-05, KRI-06, KRI-07, KRI-16 and KRI-18."));
add(bl("Appetite is as influential as any parameter: at 0.40 most scenarios fall below or near tolerance (Section 4)."));

// ================================================================ 10. PRIORITIZATION
add(h1("10. Prioritization and treatment portfolio"));
add(h2("10.1 Management attention list"));
add(p("Ranking combines estimated and residual risk, tolerance ratio, maximum consequence, patient-safety and privacy significance, velocity, cost-effectiveness, shared benefit and time to benefit. Tiers support discussion; order within a tier is indicative."));
const WHY = {
  S2: "Highest combined consequence; above tolerance; fast-moving", S8: "Highest estimated risk; best cost-effectiveness; amplifies all scenarios",
  S1: "Most likely entry path; high cost-effectiveness", S27: "Bypasses MFA investment; cheap process fix; above tolerance",
  S23: "High velocity; exposure unknown - discover and remove quickly", S3: "Determines recoverability from S2 and S28",
  S20: "Cloud control-plane takeover; above tolerance; shares I01/I08", S6: "High estimated risk; frequent real-world vector",
  S7: "Patient-safety pathway; slow to treat (certification, vendors)", S9: "Above tolerance; largely outside MediBec's control - transfer",
  S5: "Above tolerance; high regulatory significance", S16: "Network-wide availability dependency; major capital item",
  S10: "Patient-safety consequence; above tolerance", S18: "Public-facing breach pathway; above tolerance",
  S19: "Above tolerance in all cases - transfer and accept", S28: "Above tolerance in all cases - PAM and offboarding, accept remainder",
  S4: "Approximately at tolerance; fast fix", S26: "Very likely; quick policy and tooling fix", S11: "Approximately at tolerance; depends on development footprint",
  S29: "Patient-safety integrity; low likelihood", S24: "Imaging downtime; shares S7 treatment", S17: "Data sprawl; approximately at tolerance",
  S12: "Entry path; handled by segmentation", S30: "Patient harm outside MediBec systems; low cost",
  S13: "Below tolerance after low-cost protection", S14: "Below tolerance after process control", S21: "Frequent but low impact",
  S22: "Below tolerance after platform configuration", S15: "Below tolerance with MDM", S25: "Below tolerance; cheap fix",
};
const TIERS = [
  ["Tier 1 - act now (0-90 days)", ["S2", "S8", "S1", "S27", "S23", "S3", "S20", "S6"]],
  ["Tier 2 - fund and decide (3-12 months)", ["S7", "S9", "S5", "S16", "S10", "S18", "S19", "S28", "S4", "S26", "S11"]],
  ["Tier 3 - manage through standard controls", ["S29", "S24", "S17", "S12", "S30", "S13", "S14", "S21", "S22", "S15", "S25"]],
];
const tierRows = [];
TIERS.forEach(([t, ids]) => ids.forEach((id, j) => { const s = byId[id]; tierRows.push([j === 0 ? t : "", id, s.name, fmt0(s.est), f2(s.ratio), WHY[id]]); }));
if (tierRows.length !== 30) throw new Error("tiers " + tierRows.length);
add(table(["Tier", "ID", "Scenario", "Estimated", "Res / Tol", "Basis"], tierRows, [13, 5, 32, 9, 8, 33], { size: 13 }));
add(h2("10.2 Recommended investment portfolio"));
add(p("Costs are **indicative analyst estimates in CAD**, not quotations or approved budgets. **Initial** = one-time implementation; **recurring** = annual operating cost; **Year 1** = initial + first-year recurring. Cyber-insurance premiums are excluded (**Evidence gap**)."));
add(table(["ID", "Initiative", "Scenarios", "Owner", "Priority", "Initial", "Recurring / yr", "Start - end", "Success indicator"],
  INIT.map((i) => [i[0], `**${i[1]}**\n${i[2]}`, i[3].join(", "), i[4], i[5], k(i[6]), k(i[7]), `${i[8]} - ${i[9]}`, i[10]]),
  [4, 30, 10, 11, 8, 7, 8, 9, 13], { size: 13 }));
add(spacer());
add(table(["Portfolio total", "Amount"], [
  ["Initial implementation", money(D.TOT_INITIAL)], ["Annual recurring", money(D.TOT_RECUR)], ["Year-1 cost", money(Y1)],
  ["Indicative 3-year total cost of ownership", money(D.TOT_INITIAL + 3 * D.TOT_RECUR)],
  ["Of which new initiatives I14-I23 (Year 1)", money(D.TOT_INITIAL - 2410000 + D.TOT_RECUR - 1456000)],
  ["Aggregate risk reduction", `${fmt0(totE)} -> ${fmt0(totR)} risk units (${Math.round((1 - totR / totE) * 100)}%)`],
], [60, 40]));
add(spacer());
add(p("Capacity: MDR (I03) offloads monitoring and one vulnerability FTE is costed in I05 (1,950 h x $80). The additional ten initiatives increase project load; phasing in Section 11 keeps no more than five major initiatives in flight at once. Change windows must be agreed with clinical operations."));

// ================================================================ 11. ROADMAP
add(h1("11. Implementation roadmap"));
add(table(["Horizon", "Actions", "Scenarios"], [
  ["0-90 days\nUrgent / quick wins", "I01 phase 1: FIDO2 for privileged, remote-access, cloud-admin and EHR-admin accounts; number matching; block internet macros\nI21: help-desk verification for password and MFA resets\nI03: onboard MDR; connect identity, cloud, EHR audit and backup logs\nI04 phase 1: separate backup credentials, offline copy, first restore test\nI05 quick win: external attack-surface scan; close exposed admin interfaces\nI08: block public cloud storage; admin-consent workflow\nI16: payment call-back and DMARC enforcement\nI20: AI acceptable-use policy and sanctioned tool\nI23: look-alike domain monitoring and patient notice\nI15: DDoS/CDN/WAF for public services\nI11: IR plan, retainer, first tabletop; I12: approve appetite, KRI dashboard\nIf any case-appendix indicator reflects real activity: compromise assessment immediately", "S1, S2, S3, S4, S6, S8, S13, S14, S20, S23, S26, S27, S30"],
  ["3-6 months\nFoundations", "I02: PAM, tiered administration, JIT cloud roles, admin session recording\nI05: inventory, authenticated scanning, SLAs, exceptions\nI09: EHR role redesign, recertification, EHR access analytics, outbound DLP\nI10: vendor tiering, contract clauses, staged updates for critical software\nI14: secret scanning, vault, portal penetration test\nI17: MDM enrolment, device encryption, certified disposal\nI22: approved telehealth/messaging platform", "S5, S6, S9, S11, S15, S18, S19, S21, S22, S25, S28"],
  ["6-12 months\nMajor capabilities", "I04: immutable backup and isolated backup network\nI06 phase 1: segment devices, guest Wi-Fi, backups, imaging\nI07: medical device programme and passive monitoring\nI13: integrity monitoring and reconciliation\nI19: clinic data discovery, consolidation and retention\nI22: TLS for HL7/DICOM interfaces\nI18 phase 1: dual carriers / SD-WAN, firewall HA", "S2, S3, S7, S10, S12, S16, S17, S24, S29"],
  ["12-24 months\nStrategic maturity", "I06 complete: NAC in all clinics\nI18 complete: secondary-site EHR read access and failover test\nZero-trust remote access\nFull DR exercise including clinical downtime\nIndependent reassessment against this baseline; insurance renewal aligned with residual risks", "S2, S6, S7, S9, S16, S19, S28"],
], [16, 64, 20], { size: 14 }));
