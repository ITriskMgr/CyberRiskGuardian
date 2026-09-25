const fs = require("fs");
const D = JSON.parse(fs.readFileSync("data30.json", "utf8"));
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  HeadingLevel, AlignmentType, LevelFormat, Header, Footer, PageNumber, PageBreak, BorderStyle,
  TableOfContents, PageOrientation,
} = require("docx");

const FONT = "Arial";
const W = 9360; // content width (Letter, 1" margins)
const NAVY = "1F3864";
const fmt0 = (n) => Math.round(n).toLocaleString("en-US");
const f2 = (n) => n.toFixed(2);
const money = (n) => "$" + Math.round(n).toLocaleString("en-US");
const k = (n) => "$" + Math.round(n / 1000).toLocaleString("en-US") + "k";

// ---------- inline markup: **bold**, _italic_
function runs(text, opts = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|_[^_]+_)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...opts }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...opts }));
  return out;
}
const p = (text, o = {}) => new Paragraph({ children: runs(text, o.run || {}), spacing: { after: 120 }, alignment: o.align, ...(o.para || {}) });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })], pageBreakBefore: true });
const h1n = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })] });
const bl = (t, lvl = 0) => new Paragraph({ numbering: { reference: "bul", level: lvl }, children: runs(t), spacing: { after: 60 } });
const nl = (t, ref = "num") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60 } });
const callout = (t, fill = "FFF2CC") => new Table({
  width: { size: W, type: WidthType.DXA }, columnWidths: [W],
  rows: [new TableRow({ children: [new TableCell({ width: { size: W, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill, color: "auto" },
    margins: { top: 100, bottom: 100, left: 140, right: 140 }, children: [new Paragraph({ children: runs(t, { size: 19 }) })] })] })],
});
const spacer = () => new Paragraph({ children: [], spacing: { after: 80 } });

function table(headers, rows, widths, o = {}) {
  const tot = widths.reduce((a, b) => a + b, 0);
  const scale = W / tot;
  const w = widths.map((x) => Math.floor(x * scale));
  w[w.length - 1] += W - w.reduce((a, b) => a + b, 0);
  const sz = o.size || 16;
  const cell = (txt, i, head, fill) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: NAVY, color: "auto" } : fill ? { type: ShadingType.CLEAR, fill, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: String(txt).split("\n").map((line) => new Paragraph({ children: runs(line, { size: sz, color: head ? "FFFFFF" : undefined, bold: head ? true : undefined }) })),
  });
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: w,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, false, o.fillFn ? o.fillFn(r, i, ri) : (ri % 2 ? "F2F2F2" : undefined))) })),
    ],
  });
}
const clsFill = (c) => (c.startsWith("Above") ? "F4CCCC" : c.startsWith("Approx") ? "FFF2CC" : "D9EAD3");
const S = D.SCEN;
const byId = Object.fromEntries(S.map((s) => [s.id, s]));
const INIT = D.INITIATIVES;
const initName = Object.fromEntries(INIT.map((i) => [i[0], i[1]]));

const above = S.filter((s) => s.class.startsWith("Above"));
const approx = S.filter((s) => s.class.startsWith("Approx"));
const below = S.filter((s) => s.class.startsWith("Below"));
const totE = S.reduce((a, s) => a + s.est, 0), totR = S.reduce((a, s) => a + s.res, 0);
const Y1 = D.TOT_INITIAL + D.TOT_RECUR;

const body = [];
const add = (...x) => body.push(...x);


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
// ================================================================ 3. CONTEXT
add(h1("3. Organizational and technology context"));
add(p("MediBec's mission is to deliver high-quality healthcare supported by modern technology, efficient clinical workflows and secure access to patient information (E1). Its risk is therefore not only data theft: an outage or integrity failure can delay diagnosis, disrupt medication management and cancel appointments across four regions."));
add(table(["Area", "Documented facts (E1-E4)", "Implication for risk"], [
  ["Network", "Montreal hub, WAN fibre to clinics, firewalls, IDS/IPS, antivirus", "Perimeter controls exist; segmentation, east-west monitoring and Wi-Fi security unknown"],
  ["Data centre and storage", "Montreal DC, local clinic storage, on-site and off-site backups, encrypted storage", "Concentration on one hub; backup isolation and restore testing not evidenced"],
  ["Cloud and remote access", "Cloud-based remote data access for staff and clinicians", "Identity is the new perimeter; cloud configuration and logging unknown"],
  ["Endpoints and devices", "Workstations, tablets, handhelds, interconnected medical equipment", "Device inventory and patch status unknown; certification limits patching"],
  ["Applications", "Integrated EHR, radiology, lab, pharmacy, billing, communications", "Highly interdependent; one compromise spreads across clinical workflows"],
  ["People and budget", "15 IT staff, 2 cybersecurity; IT budget stated as CAD 100M incl. salaries", "Capacity is the binding constraint. **Evidence gap** - the stated budget appears high relative to 15 staff; cybersecurity budget share not stated"],
  ["Governance", "Steering Committee (teaching assumption); Board accepts material residual risk", "Structure exists on paper; operation and reporting unproven"],
  ["Regulatory", "Quebec and Canadian privacy expectations, health information confidentiality, professional obligations", "Confidentiality incidents involving health data are likely to trigger register, assessment and notification duties - **legal counsel to confirm applicable regime**"],
], [16, 42, 42]));
add(h2("3.1 Crown-jewel assets and services"));
add(table(["ID", "Asset / service", "Business purpose", "Owner", "C", "I", "A", "Technical dependencies", "Third parties"],
  D.CROWN, [5, 15, 15, 14, 7, 7, 7, 17, 13], { size: 14 }));
add(spacer());
add(p("Owners are proposed roles, not documented appointments - **Evidence gap - organizational input required**."));


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
// ================================================================ 7. DETAILED
add(h1("7. Detailed risk scenarios"));
add(p("Each scenario follows the causal chain threat source -> vulnerability -> asset/process -> cybersecurity event -> business consequence. Existing controls are listed only where the case documents them."));
const PN = { PbA: "Pb(A) threat presence", Pbx: "Pb(psi,A) exploitation", De: "de expected damage", Dm: "dm maximum damage", Th: "theta resilience", Mu: "mu(E) criticality" };
S.forEach((s, idx) => {
  add(idx === 0 ? h2(`${s.id} - ${s.name}`) : new Paragraph({ heading: HeadingLevel.HEADING_2, pageBreakBefore: true, children: [new TextRun({ text: `${s.id} - ${s.name}`, font: FONT })] }));
  add(callout(`**Scenario statement.** ${s.statement}`, "DEEAF6"));
  add(spacer());
  add(table(["Element", "Description"], [
    ["Library reference", s.ref], ["Stakeholders", s.stakeholders], ["Background", s.background],
    ["Threat source", s.threat_source], ["Threat event", s.threat_event],
    ["Vulnerabilities / conditions", s.vulns.join("\n")], ["Assets at risk", s.assets], ["Processes affected", s.processes],
    ["Existing controls (evidenced)", s.controls.join("\n")], ["Risk owner (proposed)", s.owner],
  ], [22, 78], { size: 16 }));
  add(h3("Narrative and event sequence"));
  add(p(s.narrative));
  s.sequence.forEach((x) => add(bl(x)));
  add(h3("Consequences"));
  add(table(["Dimension", "Consequence"], Object.entries(s.consequences), [22, 78], { size: 16 }));
  add(h3("Quantitative parameters"));
  add(table(["Parameter", "Value", "Reading", "Rationale", "Evidence", "Confidence"],
    Object.entries(s.params).map(([key, v]) => [PN[key], f2(v.v), v.qual, v.rat, v.ev, v.conf]), [18, 7, 12, 43, 9, 11], { size: 15 }));
  add(spacer());
  add(p(`All values: **Analyst estimate - validation required.**`, { run: { size: 18 } }));
  add(h3("CVSS v4.0"));
  add(p(`**Vector:** ${s.cvss}   **CVSS-B:** ${s.cvss_score.toFixed(1)} (${s.cvss_sev})`));
  add(p(s.cvss_note));
  add(h3("Calculation"));
  const pr = s.params;
  add(table(["Step", "Computation", "Result"], [
    ["Estimated risk", `${f2(pr.PbA.v)} x ${f2(pr.Pbx.v)} x ${s.cvss_score.toFixed(1)} x ((${f2(pr.De.v)} + ${f2(pr.Dm.v)}) / 2) x ${f2(pr.Mu.v)} / ${f2(pr.Th.v)} x 1,000`, fmt0(s.est)],
    ["Tolerated risk", `${f2(pr.PbA.v)} x ${f2(pr.Pbx.v)} x ${s.cvss_score.toFixed(1)} x ${f2(D.APPETITE)} x ${f2(pr.Mu.v)} / ${f2(pr.Th.v)} x 1,000`, fmt0(s.tol)],
    ["Mitigated", `${fmt0(s.est)} x ${f2(s.red_p)} (probability) x ${f2(s.red_i)} (impact)`, fmt0(s.mit)],
    ["Residual risk", `${fmt0(s.est)} - ${fmt0(s.mit)}`, fmt0(s.res)],
    ["Residual / Tolerated", `${fmt0(s.res)} / ${fmt0(s.tol)}  (pre-treatment ratio ${f2(s.pre_ratio)})`, f2(s.ratio)],
    ["Cost-effectiveness", `(${fmt0(s.est)} - ${fmt0(s.res)}) / ${money(s.cost_y1)} x 1,000`, f2(s.ce)],
  ], [18, 67, 15], { size: 15 }));
  add(spacer());
  add(h3("Treatment and result"));
  add(p(s.treatment));
  add(p(`Initiatives: ${s.inits.map((i) => `${i} ${initName[i]}`).join("; ")}. Allocated Year-1 cost ${money(s.cost_y1)} (share of shared initiatives). Horizon: ${s.horizon}.`));
  const interp = s.class.startsWith("Below") ? "Residual risk falls below tolerance if the package is delivered as designed; the CIO can accept it within normal governance, subject to KRI monitoring."
    : s.class.startsWith("Approx") ? "Residual risk is approximately at tolerance; the result depends on the treatment actually achieving the assumed effectiveness. Executive review and KRI tracking are recommended."
    : "Residual risk remains above tolerance after treatment. **Management decision required**: accept formally at Board / ownership-group level, fund additional treatment, transfer part of the risk, or change the process.";
  add(callout(`**Result: ${s.class} (${f2(s.ratio)}).** ${interp}`, clsFill(s.class)));
});


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
// ================================================================ 12. KRIs
add(h1("12. Key risk indicators"));
add(table(["ID", "KRI", "Scenarios", "Measurement", "Data source", "Owner", "Frequency", "Target", "Warning", "Critical"],
  D.KRIS, [6, 18, 7, 14, 11, 10, 8, 8, 8, 10], { size: 13 }));
add(spacer());
add(p("Report quarterly to the Steering Committee; any KRI at critical threshold escalates to executive leadership within one week."));


// ================================================================ 13. CASE VIEW
add(h1("13. Cross-check with the case escalation method"));
add(p("The case (s.8) defines a normalized score (T x E x I, adjusted for control maturity) with escalation thresholds: executive review above 0.25, Board acceptance above 0.40. Mapping T = Pb(A), E = Pb(psi,A), I = (de + dm)/2 and C = theta (analyst mapping - validate):"));
const lvl = (x) => (x <= 0.07 ? "Low" : x <= 0.15 ? "Moderate" : x <= 0.25 ? "High" : "Critical");
add(table(["ID", "Current residual", "Level", "Post-treatment residual", "Level", "CRG status (appetite 0.30)"],
  S.map((s) => [s.id, s.norm_cur.toFixed(3), lvl(s.norm_cur), s.norm_post.toFixed(3), lvl(s.norm_post), s.class]), [8, 16, 14, 18, 14, 30], { size: 14 }));
add(spacer());
const curHigh = S.filter((s) => s.norm_cur > 0.15);
const maxPost = Math.max(...S.map((s) => s.norm_post));
add(p(`**Reading.** Under the case method, ${curHigh.map((s) => `${s.id} (${lvl(s.norm_cur)})`).join(", ")} are High or Critical today; after treatment every scenario is Moderate or Low (maximum ${maxPost.toFixed(3)}), within CIO acceptance authority. The CyberRiskGuardian ratio is more demanding because it compares damage directly with appetite and ignores likelihood in the tolerance test. The two lenses should be reconciled before governance use. Regardless of score, the patient-safety clause of the appetite statement requires clinical-leadership acceptance for S7, S10, S16 and S29.`));

// ================================================================ 14. ASSUMPTIONS & LIMITATIONS
add(h1("14. Assumptions, limitations and methodological notes"));
add(h2("14.1 Key assumptions"));
[
  "Exercise evidence (phishing lure and look-alike domain, closed PowerShell alert, MFA push approval, cloud download, service-account queries, backup failures) indicates control weaknesses; it is not treated as a confirmed incident.",
  "Controls not mentioned in the case (SIEM, PAM, segmentation, immutable backup, DLP, MDM, WAF/DDoS, secret management, vendor programme, AI policy, help-desk verification, payment verification, hub redundancy) are assumed absent or immature.",
  "Several new scenarios assume practices common in comparable organizations but not documented at MediBec (patient portal, in-house code repositories, teleconsultation, fax/e-mail referrals, leased equipment, generative-AI use). Each is labelled Analyst assumption in the scenario detail.",
  "Costs are Canadian mid-market indicative estimates; no quotations were obtained.",
  "Treatment effectiveness assumes full, well-configured implementation and adequate operating staff.",
  "Equal split of shared-initiative costs across scenarios is a convention for cost-effectiveness only; it does not change portfolio totals.",
].forEach((x) => add(bl(x)));
add(h2("14.2 Information gaps - organizational input required"));
[
  "Asset, medical-device and internet-exposure inventories; network diagrams and segmentation.",
  "Identity architecture: MFA coverage and method, privileged, service and cloud admin accounts, help-desk reset procedure.",
  "Backup architecture, RTO/RPO, restore tests; hub and WAN redundancy.",
  "Cloud tenant configuration, app-consent settings, sharing practices and logging.",
  "Application portfolio: patient portal, in-house development, repositories, pen-test history.",
  "Vulnerability scan results, patch SLAs and end-of-life systems.",
  "Critical vendor and software supplier list, contracts, remote-access and update mechanisms.",
  "Data held on clinic storage, mobile devices and disposed assets; retention schedule.",
  "Telehealth and messaging platforms; staff use of generative AI.",
  "Finance payment-change procedure and e-mail authentication (DMARC).",
  "Incident history since 2020; log retention; cyber-insurance coverage; cybersecurity budget (the stated CAD 100M IT budget appears inconsistent with a 15-person IT team).",
  "Applicable privacy regime and notification obligations (legal counsel).",
].forEach((x) => add(bl(x)));
add(h2("14.3 Methodological observations"));
add(bl("**The tolerance test depends only on damage, treatment and appetite.** Residual / Tolerated simplifies to ((de + dm)/2) x (1 - P x Q) / Appetite; probabilities, CVSS, resilience and criticality cancel. Rare scenarios such as S28 (Pb(A) 0.15) are therefore 'above tolerance' as strongly as frequent ones - read the ratio together with estimated risk."));
add(bl(`**CVSS is a required multiplier even where it does not fit.** Representative vectors were scored for non-technical scenarios (${S.filter((s) => s.cvss_note.startsWith("Limited applicability")).map((s) => s.id).join(", ")}) and flagged as limited applicability. Low CVSS values for fraud and human-error scenarios (e.g. S14, S21, S30) depress their estimated risk; their business impact is carried by the damage estimates.`));;
add(bl("**Mitigation is multiplicative** (Mitigated = Estimated x P x Q), a conservative convention."));
add(bl("**The template holds ten scenarios.** The companion workbook extends the same formulas to 30 rows; the ISO/NIST/CIS control catalogue sheets (M1-M10) were not used - initiatives are costed directly."));
add(bl("**Two scoring lenses disagree** (Section 13)."));
add(bl("**Limitations.** Desk-based assessment of a fictional case; no interviews, testing or technical data; AI-assisted estimates. Confidence is lower for S11-S30 than for S1-S10 because fewer case facts support them."));

// ================================================================ 15. DECISIONS
add(h1("15. Management decisions required"));
add(table(["#", "Decision", "Options", "Recommended", "Authority"], [
  ["D1", "Confirm cybersecurity risk appetite", "0.30 (analyst) / 0.40 (case) / differentiated by category", "Adopt 0.30 for patient-safety and health-data risks; consider a higher value for purely financial or reputational scenarios (S13, S14, S30)", "Board / ownership group"],
  ["D2", "Approve treatment portfolio", `Full (${money(Y1)} Y1) / phased / partial`, "Approve 0-90-day actions immediately; release remaining funding by phase against KRIs", "Executive leadership, Board"],
  ["D3", `Residual above tolerance (${listIds(above)})`, "Accept / further mitigate / transfer / avoid", `Transfer and formally accept ${robustAbove.join(", ")} and S9; for the others, accept with 6-month review conditional on KRI evidence of treatment effectiveness`, "Board / ownership group"],
  ["D4", "Patient-safety residual (S7, S10, S16, S29)", "Accept with compensating controls / restrict connectivity / fund redundancy", "CMO sign-off on compensating controls, downtime procedures and the I18 business case", "Clinical leadership + Executive"],
  ["D5", "Capacity and staffing", "Status quo / +1 FTE / +2 FTE / managed services", "+1 vulnerability FTE (costed), MDR, and assess an IAM/application-security role (not costed)", "CIO, Executive"],
  ["D6", "Cyber-insurance review", "Maintain / increase / obtain", "Broker review aligned with S2, S9, S19 and S28 residuals", "CFO"],
  ["D7", "Validate estimates and close evidence gaps", "-", "SME workshop within 60 days, prioritizing S11-S30 assumptions; re-run workbook", "Cybersecurity lead"],
  ["D8", "Assign risk owners", "-", "Confirm the owners proposed in Section 7 for all 30 scenarios", "Executive leadership"],
], [4, 20, 20, 36, 20], { size: 14 }));
add(spacer());
add(callout("CyberRiskGuardian outputs are decision support. Scenarios, probabilities, CVSS scores, costs and effectiveness values are initial analytical hypotheses. Final decisions to accept, mitigate, transfer or avoid risk remain the responsibility of MediBec management and its Board / ownership group."));
// ================================================================ DOCUMENT
const doc = new Document({
  features: { updateFields: true },
  creator: "CyberRiskGuardian",
  title: "MediBec Cybersecurity Risk Assessment",
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 32, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 240, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 22, bold: true, font: FONT, color: "2F5496" }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }, { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } }] },
      { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "MediBec - Cybersecurity Risk Assessment - Analyst draft for validation", font: FONT, size: 16, color: "7F7F7F" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Page ", font: FONT, size: 16, color: "7F7F7F" }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: "7F7F7F" })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync("../output/MediBec_Cybersecurity_Risk_Assessment_30.docx", b); console.log("ok"); });
