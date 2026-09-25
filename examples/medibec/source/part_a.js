
// ---------- 30-scenario helpers
const preCls = S.map((s) => ({ id: s.id, c: s.pre_ratio > 1.1 ? "Above" : s.pre_ratio >= 0.9 ? "Approx" : "Below" }));
const preAbove = preCls.filter((x) => x.c === "Above").length;
const origIds = S.slice(0, 10).map((s) => s.id);
const newS = S.slice(10);
const case40 = { above: S.filter((s) => s.class_case.startsWith("Above")), approx: S.filter((s) => s.class_case.startsWith("Approx")) };
const listIds = (arr) => arr.map((s) => s.id).join(", ");
const robustAbove = D.SENS.filter((r) => r.lower.cls.startsWith("Above")).map((r) => r.id);

// ================================================================ TITLE
add(
  new Paragraph({ spacing: { before: 2400, after: 200 }, children: [new TextRun({ text: "MediBec", font: FONT, size: 56, bold: true, color: NAVY })] }),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "Scenario-Based Cybersecurity Risk Assessment - 30 Scenarios", font: FONT, size: 36, color: NAVY })] }),
  new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: "Thirty material risk scenarios, quantified risk indicators, treatment portfolio and 24-month roadmap", font: FONT, size: 24, color: "595959" })] }),
  p("**Prepared for:** Cybersecurity and Privacy Steering Committee, Executive Leadership, Board / ownership group"),
  p("**Prepared by:** CyberRiskGuardian analysis (AI-assisted, analyst validation required)"),
  p("**Date:** 25 September 2026   **Assessment period:** next 12 months (October 2026 - September 2027)"),
  p("**Methodology:** CyberRiskGuardian (Excel Guide v1.0c formulas), CVSS v4.0, multiplication factor 1,000, risk appetite 0.30"),
  p("**Companion file:** MediBec_CyberRiskGuardian_Workbook_30.xlsx (live calculations)"),
  p("**Relationship to earlier work:** extends the 10-scenario assessment of the same date; S1-S10 are carried over unchanged."),
  spacer(),
  callout("**Source and status.** Based solely on the MediBec Business Case v2.0b, a fictional teaching case. All probabilities, damages, CVSS vectors, costs and effectiveness values are **analyst estimates - validation required**. Scores are relative decision-support indicators, not predictions of incidents or dollar losses. Risk acceptance decisions remain with MediBec management."),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ children: [new TextRun({ text: "Contents", font: FONT, size: 32, bold: true, color: NAVY })], spacing: { after: 200 } }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
);

// ================================================================ 1. EXECUTIVE SUMMARY
add(h1("1. Executive summary"));
add(p(`MediBec operates 10 clinics that depend on a central Montreal data centre, an integrated EHR, cloud-based remote access and interconnected medical equipment, protected by a two-person cybersecurity team. Attack attempts have increased and no breach is confirmed, but the case evidence shows the weak points a real intrusion would use: push-based MFA, alerts closed without correlation, untested recovery, and little visibility over cloud, service accounts, devices, vendors and data held in clinics.`));
add(p(`This assessment widens the earlier ten-scenario view to **30 scenarios** selected from **45 candidates**. The ten original scenarios are unchanged; twenty new ones add application and cloud control-plane attacks, social engineering of the help desk, insider sabotage, hub resilience, data lifecycle, generative-AI use, clinical interface integrity and patient-targeted fraud. Against an estimated risk appetite of **${f2(D.APPETITE)}**, ${preAbove} of 30 scenarios are above tolerance today. After the recommended portfolio:`));
add(bl(`**Above tolerance (${above.length}):** ${above.map((s) => `${s.id} (${f2(s.ratio)})`).join(", ")}. These need Board / ownership-group acceptance, further treatment or transfer.`));
add(bl(`**Approximately at tolerance (${approx.length}):** ${approx.map((s) => `${s.id} (${f2(s.ratio)})`).join(", ")}.`));
add(bl(`**Below tolerance (${below.length}):** ${below.map((s) => `${s.id} (${f2(s.ratio)})`).join(", ")}.`));
add(p(`The portfolio of **${INIT.length} initiatives** (13 carried over, extended in scope, and 10 new) reduces aggregate estimated risk by about **${Math.round((1 - totR / totE) * 100)}%** (${fmt0(totE)} to ${fmt0(totR)} risk units). Indicative cost: **${money(D.TOT_INITIAL)} initial** plus **${money(D.TOT_RECUR)} per year**, i.e. **${money(Y1)} in Year 1**. The ten new initiatives add ${money(D.TOT_INITIAL - 2410000)} initial and ${money(D.TOT_RECUR - 1456000)} recurring; most new scenarios are treated largely by controls already in the portfolio.`));
add(h3("Most material scenarios (by estimated risk)"));
add(table(["ID", "Scenario", "Estimated", "Residual", "Residual / Tolerated", "Status"],
  [...S].sort((a, b) => b.est - a.est).slice(0, 10).map((s) => [s.id, s.name, fmt0(s.est), fmt0(s.res), f2(s.ratio), s.class]),
  [5, 45, 10, 10, 10, 20], { fillFn: (r, i) => (i === 5 ? clsFill(r[5]) : undefined) }));
add(spacer());
add(h3("What the wider view adds"));
add(bl("**Identity is attacked from several sides** - phishing (S1), help-desk resets (S27), token theft in the cloud (S20) and exposed admin interfaces (S23). MFA alone does not close them; verification processes, token binding and PAM are needed."));
add(bl(`**Some risks cannot be brought within tolerance by MediBec alone** - software supply chain (S19), privileged-insider sabotage (S28) and vendors (S9) stay above tolerance even under favourable assumptions (${robustAbove.join(", ")} in the lower-risk case). Transfer and formal acceptance are required.`));
add(bl("**Resilience is concentrated** - the Montreal hub (S16) is a single point of failure for all clinics, independent of any attacker."));
add(bl("**Low-cost fixes remove several below-tolerance risks** - payment verification (S14), device disposal (S25), MDM (S15), DDoS protection (S13) and an AI-use policy (S26) cost little and act quickly."));
add(bl("**Patient-safety integrity risks** - record manipulation (S10), interface tampering (S29) and medical devices (S7) require clinical sign-off regardless of score."));
add(h3("Decisions requested"));
add(nl(`**Management decision required** - Confirm the risk appetite (0.30 estimated vs. 0.40 in the case). At 0.40, ${case40.above.length} scenarios remain above tolerance (${listIds(case40.above)}) and ${case40.approx.length} are approximately at it.`));
add(nl(`**Management decision required** - Approve the portfolio (${money(Y1)} Year 1) and the 0-90-day actions in Section 11.`));
add(nl(`**Management decision required** - Board / ownership-group decision on the ${above.length} scenarios above tolerance; transfer (insurance, contracts) and formal acceptance for ${robustAbove.join(", ")}.`));
add(nl("**Management decision required** - Clinical leadership sign-off on S7, S10, S16 and S29 (patient-safety clause of the appetite statement)."));
add(nl("**Management decision required** - Capacity: 23 initiatives exceed what a two-person team can deliver; approve the costed vulnerability FTE and assess an IAM/application-security role (not costed)."));

// ================================================================ 2. SCOPE & METHOD
add(h1("2. Assessment scope and methodology"));
add(h2("2.1 Scope"));
add(table(["Element", "Definition"], [
  ["Organization", "MediBec private medical centre: Montreal head office and data centre, 10 clinics (Montreal, Quebec City, Gatineau, Saguenay)."],
  ["In scope", "EHR and clinical systems (radiology, laboratory, pharmacy, interfaces), scheduling, patient portal and online services, billing and payments, identity, help desk and remote access, cloud tenant and storage, e-mail and communications, data centre, WAN, backups, clinic networks, Wi-Fi and local storage, endpoints and mobile devices, connected medical equipment, software supply chain and critical vendors, asset disposal, staff use of generative AI, patient-targeted impersonation."],
  ["Period", "Next 12 months (October 2026 - September 2027); roadmap extends to 24 months and aligns with the 2024-2030 planning horizon."],
  ["Risk perspective", "Confidentiality, integrity, availability, privacy, patient safety, operations, finance, legal/regulatory compliance, reputation, strategy."],
  ["Changes from the 10-scenario scope", "Non-malicious hub/WAN outage (S16) and financial fraud (S14) are now in scope as resilience and business-impact scenarios."],
  ["Exclusions", "Physical security of premises (C39), payment-card processing (C38 - not evidenced), detailed legal analysis."],
  ["Evidence limits", "No asset inventory, architecture diagrams, scan results, audit reports, contracts or incident records were supplied. Exercise injects (case Appendices A-B) are treated as indicators to validate, not confirmed facts."],
], [18, 82]));
add(h2("2.2 Method"));
add(p("CyberRiskGuardian method: context and crown jewels, risk appetite, 45 candidate scenarios screened to 30, causal-chain scenario development, six parameters on a 0-1 scale, CVSS v4.0 base score for a representative weakness, KRI calculation, treatment portfolio with shared-control allocation, residual-risk recalculation, sensitivity analysis and multi-criteria prioritization. The same scoring conventions as the 10-scenario assessment are used so results are comparable."));
add(p("**Formulas (Excel Guide v1.0c, s.11 - used without modification):**"));
add(table(["Indicator", "Formula"], [
  ["Estimated risk (Re)", "Pb(A) x Pb(psi,A) x CVSS x ((de + dm) / 2) x mu(E) / theta x Factor"],
  ["Tolerated risk (Rt)", "Pb(A) x Pb(psi,A) x CVSS x Appetite x mu(E) / theta x Factor"],
  ["Mitigated amount", "Re x Probability reduction x Impact reduction"],
  ["Residual risk", "Re - Mitigated (equivalent to the template's ABS() because reductions stay within 0-1)"],
  ["Risk-to-tolerance ratio", "Residual / Tolerated.  < 0.90 below;  0.90-1.10 approximately at;  > 1.10 above tolerance"],
  ["Cost-effectiveness", "(Re - Residual) / allocated Year-1 cost x 1,000"],
], [25, 75]));
add(spacer());
add(p("Factor = 1,000 for all 30 scenarios. The original CyberRiskGuardian workbook template holds ten scenarios; the companion workbook uses the same formulas extended to 30 rows."));
add(h2("2.3 Evidence register"));
add(table(["ID", "Source", "Basis", "Content used"], D.EVIDENCE, [6, 30, 17, 47], { size: 15 }));
add(spacer());
add(p("Basis tags: **(E#)** documented case evidence; **Exercise evidence** simulated injects - validate; **Analyst assumption** / **Analyst estimate - validation required**; **Evidence gap - organizational input required**. Many new scenarios (S11-S30) rest on fewer case facts than S1-S10; their confidence ratings are correspondingly lower."));
add(h2("2.4 Changes from the 10-scenario assessment"));
add(bl("S1-S10: statements, parameters, CVSS vectors and reductions unchanged; estimated, tolerated and residual risk are identical."));
add(bl("Shared initiatives now serve more scenarios, so each scenario's allocated cost is lower and its cost-effectiveness higher; portfolio totals, not scenario allocations, should drive budget decisions."));
add(bl("Seven candidates previously not selected or merged (C05, C06, C10, C16-C19) are now retained as S11-S17; 25 new candidates were generated (C21-C45)."));
add(bl("Ten initiatives added (I14-I23); existing initiatives extended to new scenarios (e.g. I01 now also covers S20, S23, S27)."));
