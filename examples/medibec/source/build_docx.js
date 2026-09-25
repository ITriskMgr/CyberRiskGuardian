const fs = require("fs");
const D = JSON.parse(fs.readFileSync("data.json", "utf8"));
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

// ================================================================ TITLE
add(
  new Paragraph({ spacing: { before: 2400, after: 200 }, children: [new TextRun({ text: "MediBec", font: FONT, size: 56, bold: true, color: NAVY })] }),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "Scenario-Based Cybersecurity Risk Assessment", font: FONT, size: 36, color: NAVY })] }),
  new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: "Ten material risk scenarios, quantified risk indicators, treatment portfolio and 24-month roadmap", font: FONT, size: 24, color: "595959" })] }),
  p("**Prepared for:** Cybersecurity and Privacy Steering Committee, Executive Leadership, Board / ownership group"),
  p("**Prepared by:** CyberRiskGuardian analysis (AI-assisted, analyst validation required)"),
  p("**Date:** 25 September 2026   **Assessment period:** next 12 months (October 2026 - September 2027)"),
  p("**Methodology:** CyberRiskGuardian (Excel Guide v1.0c formulas), CVSS v4.0, multiplication factor 1,000"),
  p("**Companion file:** MediBec_CyberRiskGuardian_Workbook.xlsx (live calculations)"),
  spacer(),
  callout("**Source and status.** Based solely on the MediBec Business Case v2.0b, a fictional teaching case. All probabilities, damages, CVSS vectors, costs and effectiveness values are **analyst estimates - validation required**. Scores are relative decision-support indicators, not predictions of incidents or dollar losses. Risk acceptance decisions remain with MediBec management."),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ children: [new TextRun({ text: "Contents", font: FONT, size: 32, bold: true, color: NAVY })], spacing: { after: 200 } }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
);

// ================================================================ 1. EXECUTIVE SUMMARY
add(h1("1. Executive summary"));
add(p(`MediBec operates 10 clinics that depend on a central Montreal data centre, an integrated EHR, cloud-based remote access and interconnected medical equipment, protected by a two-person cybersecurity team. Attack attempts have increased, and although no breach is confirmed, the case evidence shows the weak points a real intrusion would use: push-based MFA that can be fatigued, alerts closed without correlation, untested backups and recovery, and little visibility over cloud storage, service accounts, medical devices and vendors.`));
add(p(`From 20 candidate scenarios, ten were retained. Against an estimated risk appetite of **${f2(D.APPETITE)}**, **every scenario is currently above tolerance** (pre-treatment estimated-to-tolerated ratios of ${f2(Math.min(...S.map((s) => s.pre_ratio)))}-${f2(Math.max(...S.map((s) => s.pre_ratio)))}). After the recommended portfolio, the picture is:`));
add(bl(`**Above tolerance (${above.length}):** ${above.map((s) => `${s.id} (${f2(s.ratio)})`).join(", ")} - ransomware, insider/privileged misuse, medical devices, vendors and record integrity. These need Board / ownership-group acceptance, further treatment or transfer.`));
add(bl(`**Approximately at tolerance (${approx.length}):** ${approx.map((s) => `${s.id} (${f2(s.ratio)})`).join(", ")}.`));
add(bl(`**Below tolerance (${below.length}):** ${below.map((s) => `${s.id} (${f2(s.ratio)})`).join(", ")} - backup compromise, provided immutable and isolated backups are delivered.`));
add(p(`The portfolio of **13 initiatives** reduces aggregate estimated risk by about **${Math.round((1 - totR / totE) * 100)}%** (${fmt0(totE)} to ${fmt0(totR)} risk units). Indicative cost: **${money(D.TOT_INITIAL)} initial** plus **${money(D.TOT_RECUR)} per year** recurring, i.e. **${money(Y1)} in Year 1** (implementation plus first-year operation; no quotations obtained).`));
add(h3("Most material scenarios"));
add(table(["ID", "Scenario", "Estimated", "Residual", "Residual / Tolerated", "Status"],
  [...S].sort((a, b) => b.est - a.est).slice(0, 5).map((s) => [s.id, s.name, fmt0(s.est), fmt0(s.res), f2(s.ratio), s.class]),
  [5, 45, 10, 10, 10, 20], { fillFn: (r, i) => (i === 5 ? clsFill(r[5]) : undefined) }));
add(spacer());
add(h3("Systemic weaknesses"));
add(bl("**Identity** - push MFA without phishing resistance, no evidence of conditional access or privileged-access management; over-privileged service account (ehr-sync-svc)."));
add(bl("**Detection and response capacity** - two staff, no 24/7 monitoring, weak alert triage, untested incident response across clinics."));
add(bl("**Recoverability** - backups exist but isolation, immutability and restore testing are not evidenced."));
add(bl("**Flat, device-rich clinic networks** - medical devices, Wi-Fi and workstations not demonstrably segmented; patching constrained."));
add(bl("**Third-party and cloud visibility** - no vendor assurance, contract clauses or cloud posture management evidenced."));
add(h3("Decisions requested"));
add(nl("**Management decision required** - Confirm the cybersecurity risk appetite (analyst estimate 0.30 vs. 0.40 stated in the case). At 0.40, only S9 remains above tolerance and S7 sits at the upper edge."));
add(nl(`**Management decision required** - Approve the treatment portfolio (${money(Y1)} Year 1) and immediate 0-90-day actions: phishing-resistant MFA for privileged and remote access, 24/7 MDR, backup isolation, cloud public-access block, service-account lockdown, tabletop exercise.`));
add(nl("**Management decision required** - Board / ownership-group decision on residual risk above tolerance for S2, S5, S7, S9, S10: accept formally, fund further treatment, or transfer (cyber insurance, vendor contracts)."));
add(nl("**Management decision required** - Clinical leadership sign-off on S7 and S10: the appetite statement allows no residual risk creating a credible immediate threat to patient care."));
add(nl("**Management decision required** - Approve one dedicated vulnerability-management FTE (costed) and assess a second IAM/PAM role (not costed)."));

// ================================================================ 2. SCOPE & METHOD
add(h1("2. Assessment scope and methodology"));
add(h2("2.1 Scope"));
add(table(["Element", "Definition"], [
  ["Organization", "MediBec private medical centre: Montreal head office and data centre, 10 clinics (Montreal, Quebec City, Gatineau, Saguenay)."],
  ["In scope", "EHR and clinical systems (radiology, laboratory, pharmacy), scheduling and online services, billing and administration, identity and remote access, cloud storage, data centre, WAN, backups, clinic networks and Wi-Fi, connected medical equipment, critical vendors."],
  ["Period", "Next 12 months (October 2026 - September 2027); roadmap extends to 24 months and aligns with the 2024-2030 planning horizon."],
  ["Risk perspective", "Confidentiality, integrity, availability, privacy, patient safety, operations, finance, legal/regulatory compliance, reputation, strategy."],
  ["Exclusions", "Physical security, non-malicious data-centre outages (candidate C16 - referred to BCP/DR review), payment-card scope, detailed legal analysis."],
  ["Evidence limits", "No asset inventory, architecture diagrams, scan results, audit reports, contracts or incident records were supplied. Incident injects (Appendices A-B of the case) are treated as illustrative indicators to validate, not confirmed facts."],
], [18, 82]));
add(h2("2.2 Method"));
add(p("The assessment follows the CyberRiskGuardian method: context and crown jewels, risk appetite, 20 candidate scenarios screened to 10, detailed causal-chain scenarios, six parameters on a 0-1 scale, CVSS v4.0 base score (CVSS-B) for a representative weakness, KRI calculation, treatment portfolio with shared-control allocation, residual-risk recalculation, sensitivity analysis and multi-criteria prioritization."));
add(p("**Formulas (Excel Guide v1.0c, s.11 - used without modification):**"));
add(table(["Indicator", "Formula"], [
  ["Estimated risk (Re)", "Pb(A) x Pb(psi,A) x CVSS x ((de + dm) / 2) x mu(E) / theta x Factor"],
  ["Tolerated risk (Rt)", "Pb(A) x Pb(psi,A) x CVSS x Appetite x mu(E) / theta x Factor"],
  ["Mitigated amount", "Re x Probability reduction x Impact reduction"],
  ["Residual risk", "Re - Mitigated (the template's ABS() is equivalent because reductions are kept within 0-1)"],
  ["Risk-to-tolerance ratio", "Residual / Tolerated.  < 0.90 below;  0.90-1.10 approximately at;  > 1.10 above tolerance"],
  ["Cost-effectiveness", "(Re - Residual) / allocated Year-1 cost x 1,000 (risk units removed per $1,000)"],
], [25, 75]));
add(spacer());
add(p("Factor = 1,000 for every scenario. Parameters: Pb(A) threat presence; Pb(psi,A) exploitation; de expected damage; dm maximum damage; theta organizational resilience (higher = stronger); mu(E) criticality of the affected service. Confidence is rated High / Medium / Low for each value. The 'approximately at' band of +/-10% is an analyst convention reflecting estimate precision."));
add(h2("2.3 Evidence register"));
add(table(["ID", "Source", "Basis", "Content used"], D.EVIDENCE, [6, 30, 17, 47], { size: 15 }));
add(spacer());
add(p("Basis tags used in this report: **(E#)** documented case evidence; **Exercise evidence** simulated injects - validate; **Analyst assumption** / **Analyst estimate - validation required**; **Evidence gap - organizational input required**. Instructor Annex A values were used only as a calibration cross-check: the estimates here fall in a comparable range (e.g. resilience 0.30-0.40 vs. 0.38-0.55 in the annex), but were derived independently."));

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
add(callout(`**Estimated Risk Appetite: ${f2(D.APPETITE)}** (risk averse). Analyst estimate - validation required.`, "DEEAF6"));
add(spacer());
add(p("**Rationale.**"));
add(bl("**Sector and mission** - healthcare delivery where cyber events can affect clinical decisions; MediBec states a **low** appetite for risks to patient safety, health-data confidentiality, compliance and public trust (E5)."));
add(bl("**Information sensitivity** - EHR, imaging, lab, pharmacy and billing data are among the most sensitive personal information."));
add(bl("**Regulatory exposure** - privacy-incident obligations and professional duties amplify the consequences of any breach."));
add(bl("**Operational dependence** - all 10 clinics depend on central services; a single failure has network-wide effects."));
add(bl("**Capacity** - a two-person team limits MediBec's ability to absorb and respond to incidents, arguing against a neutral appetite."));
add(bl("**Moderating factor** - MediBec accepts moderate risk for governed innovation (E5), which argues against an extreme (0.1-0.2) value."));
add(p(`**Relationship to the case value.** The case gives a normalized tolerance of ${f2(D.CASE_APPETITE)}, blending low appetite for clinical and privacy risk with moderate appetite for innovation, and designed for a different scoring scale (T x E x I). Every scenario in this assessment involves health data or patient care, so the stricter 0.30 is proposed. Because the choice changes outcomes, it is shown below as a sensitivity. **Management decision required.**`));
add(table(["Scenario", "Ratio @ 0.20", "Ratio @ 0.30 (base)", "Ratio @ 0.40 (case)"],
  S.map((s) => { const b = s.res / s.tol * D.APPETITE; return [`${s.id} ${s.name}`, f2(b / 0.2), f2(b / 0.3), f2(b / 0.4)]; }),
  [55, 15, 15, 15], { fillFn: (r, i) => (i > 0 ? (parseFloat(r[i]) > 1.1 ? "F4CCCC" : parseFloat(r[i]) >= 0.9 ? "FFF2CC" : "D9EAD3") : undefined) }));
add(spacer());
add(p("Residual risk / tolerated risk after treatment. At 0.40, only S9 (vendors) stays above tolerance and S7 (medical devices) sits at the upper edge (1.09); at 0.20 all ten would be above."));

// ================================================================ 5. SCENARIO DEVELOPMENT
add(h1("5. Scenario development and selection"));
add(p("Twenty candidates were generated from six perspectives - top-down (clinical continuity, privacy), bottom-up (documented weaknesses), threat-driven, asset-driven (crown jewels), third-party and human - using the case scenario library (MED-CYR-01 to 15), the exercise evidence and the extended catalogue. Each is expressed as threat source, condition, asset and outcome."));
add(table(["ID", "Candidate", "Threat source", "Vulnerability / condition", "Asset", "Outcome", "Ref", "Decision", "Rationale"],
  D.CANDIDATES, [4, 15, 9, 14, 7, 10, 8, 9, 24], { size: 13, fillFn: (r, i) => (i === 7 ? (r[7].startsWith("Selected") ? "D9EAD3" : r[7].startsWith("Merged") ? "FFF2CC" : "F2F2F2") : undefined) }));
add(spacer());
add(p("**Screening criteria:** plausibility over 12 months, consequence severity (patient safety and privacy weighted highest), evidence of exposure in the case, uniqueness of the causal chain, regulatory and operational significance. Five candidates were merged into selected scenarios to avoid duplication; five were not selected because of lower consequence, thin evidence or a non-malicious cause. CVSS played no role in selection."));

// ================================================================ 6. PORTFOLIO TABLE
add(h1("6. Risk scenario portfolio"));
add(table(["ID", "Scenario", "CVSS-B", "Estimated", "Tolerated", "Residual", "Res / Tol", "Status", "Key treatment"],
  S.map((s) => [s.id, s.name, s.cvss_score.toFixed(1), fmt0(s.est), fmt0(s.tol), fmt0(s.res), f2(s.ratio), s.class, s.inits.join(", ")]),
  [4, 30, 6, 8, 8, 8, 7, 13, 16], { size: 14, fillFn: (r, i) => (i === 7 ? clsFill(r[7]) : undefined) }));
add(spacer());
add(p(`Totals: estimated ${fmt0(totE)}, tolerated ${fmt0(S.reduce((a, s) => a + s.tol, 0))}, residual ${fmt0(totR)} risk units. Initiative names are listed in Section 10.`));

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
  ["Identity that can be phished or abused", "S1, S2, S5, S8, S9, S10", "Push MFA, standing privileges, un-vaulted service accounts and vendor access provide the main entry and escalation paths."],
  ["Limited detection and response capacity", "All (S8 amplifier)", "Two staff cannot provide 24/7 triage; weak signals in E6 were not correlated. This raises damage in every scenario."],
  ["Unproven recoverability", "S2, S3, S10", "Backups exist but isolation, immutability, restore tests and integrity checks are not evidenced."],
  ["Flat, heterogeneous clinic networks", "S2, S6, S7", "Workstations, medical devices and Wi-Fi share networks; one foothold reaches clinical systems."],
  ["Limited visibility of data stores and third parties", "S4, S5, S9", "Cloud storage, EHR access patterns and vendor sessions are not monitored."],
  ["Governance operating on paper", "All", "Roles are defined, but no KRI reporting, access recertification or tested IR is evidenced."],
], [24, 18, 58]));
add(h2("8.2 Concentration and systemic dependencies"));
add(bl("**Montreal hub** - EHR, identity, backups and WAN converge on one site; any enterprise-wide compromise affects all 10 clinics simultaneously."));
add(bl("**Identity platform** - a single directory and MFA service gate every crown jewel; its compromise defeats most other controls."));
add(bl("**EHR vendor** - a single integrated EHR supplier represents concentration risk (S9); vendor identity and contract terms are an **Evidence gap**."));
add(bl("**Scenario chaining** - S1 or S6 (entry) + S8 (undetected) + S3 (no recovery) = S2 at maximum damage. Treating the chain's weakest links (I01, I03, I04) reduces several scenarios at once."));
add(h2("8.3 Shared controls"));
add(table(["Initiative", "Scenarios", "Why it is shared"],
  INIT.filter((i) => i[3].length >= 3).map((i) => [`${i[0]} ${i[1]}`, i[3].join(", "), i[2]]), [26, 18, 56], { size: 15 }));
add(spacer());
add(p("Shared initiatives are costed once in the portfolio and their Year-1 cost is split equally among the scenarios they address (Workbook, Portfolio sheet). Scenario-level cost-effectiveness therefore reflects a fair share, not the full enterprise cost. Key dependencies: I02 (PAM) needs the account inventory from I01; I06 (segmentation) needs the device and asset inventories from I05 and I07; I13 needs the logging platform from I03."));

// ================================================================ 9. SENSITIVITY
add(h1("9. Sensitivity analysis"));
add(p("For five key scenarios, all uncertain parameters were shifted together by 0.10: the lower-risk case lowers threat, exploitation and damage values and raises resilience and treatment effectiveness; the higher-risk case does the reverse. This is a deliberately wide band given Low-to-Medium confidence in most inputs."));
add(table(["ID", "Scenario", "Lower: residual / ratio", "Central: residual / ratio", "Higher: residual / ratio", "Recommendation robust?"],
  D.SENS.map((r) => [r.id, r.name, `${fmt0(r.lower.res)} / ${f2(r.lower.ratio)}`, `${fmt0(r.central.res)} / ${f2(r.central.ratio)}`, `${fmt0(r.higher.res)} / ${f2(r.higher.ratio)}`,
    r.lower.cls === r.higher.cls ? "Yes - same classification" : `No - ${r.lower.cls.toLowerCase()} to ${r.higher.cls.toLowerCase()}`]),
  [5, 30, 15, 15, 15, 20], { size: 15 }));
add(spacer());
add(p("**Findings.**"));
add(bl("**S9 (vendors)** is above tolerance in the central and higher cases and only reaches approximately-at in the lower case (1.04): treatment within MediBec's control is insufficient on its own, so contractual and insurance transfer plus formal acceptance is the robust recommendation."));
add(bl("**S7 (medical devices)** is at tolerance only in the lower case: its outcome depends on manufacturer cooperation and segmentation effectiveness - both need validation before committing to acceptance."));
add(bl("**S1, S2 and S8** flip between below and above tolerance: the decision to invest is robust (estimated risk is high in all cases), but whether residual risk can be accepted depends on the treatment achieving its assumed effectiveness. Track with KRI-01, KRI-05, KRI-06, KRI-07."));
add(bl("Appetite is as influential as any parameter (Section 4): moving from 0.30 to 0.40 changes the classification of eight of the ten scenarios."));

// ================================================================ 10. PRIORITIZATION
add(h1("10. Prioritization and treatment portfolio"));
add(h2("10.1 Management attention list"));
add(p("Ranking combines estimated and residual risk, tolerance ratio, maximum consequence, patient-safety and privacy significance, risk velocity, cost-effectiveness, shared benefit and time to benefit. It is a judgement to support discussion, not a mathematical order."));
const prio = [
  ["1", "S2", "Ransomware", "Highest combined clinical, privacy and operational consequence (dm 0.95); above tolerance; fast-moving", "Fund layered package; Board acceptance of residual; insurance review"],
  ["2", "S8", "Detection and response", "Highest estimated risk; best cost-effectiveness (25.4); amplifies every other scenario", "Immediate MDR and IR readiness"],
  ["3", "S1", "Credential compromise", "Most likely entry path; strong cost-effectiveness (17.3); quick to treat", "Phishing-resistant MFA within 90 days for privileged and remote access"],
  ["4", "S7", "Medical devices", "Direct patient-safety pathway; ratio 1.45; slow to treat (certification, vendors)", "Start device programme and segmentation; CMO sign-off on residual"],
  ["5", "S3", "Backup compromise", "Determines whether S2 is recoverable; falls below tolerance once treated", "Immediate backup isolation and restore test"],
  ["6", "S6", "Unpatched systems", "High estimated risk; frequent real-world vector; good cost-effectiveness (11.8)", "Vulnerability programme with SLAs"],
  ["7", "S9", "Vendor compromise", "Highest residual ratio (1.54); largely outside MediBec's control", "Contract clauses, brokered vendor access, transfer"],
  ["8", "S5", "Insider / privileged misuse", "Above tolerance; high regulatory significance", "Access governance and EHR monitoring"],
  ["9", "S10", "Record integrity", "Lowest likelihood but patient-safety consequence; above tolerance", "Integrity controls; clinical sign-off"],
  ["10", "S4", "Cloud image storage", "Approximately at tolerance; cheap, fast fix", "Block public access now"],
];
add(table(["#", "ID", "Scenario", "Basis", "Action"], prio, [4, 5, 16, 45, 30], { size: 15 }));
add(h2("10.2 Recommended investment portfolio"));
add(p("Costs are **indicative analyst estimates in CAD**, not quotations or approved budgets. Basis: **initial** = one-time implementation; **recurring** = annual operating cost; **Year 1** = initial + first-year recurring. Cyber-insurance premiums are excluded (**Evidence gap** - broker quotation required)."));
add(table(["ID", "Initiative", "Scenarios", "Owner", "Priority", "Initial", "Recurring / yr", "Start - end", "Success indicator"],
  INIT.map((i) => [i[0], `**${i[1]}**\n${i[2]}`, i[3].join(", "), i[4], i[5], k(i[6]), k(i[7]), `${i[8]} - ${i[9]}`, i[10]]),
  [4, 30, 9, 11, 8, 7, 8, 9, 14], { size: 13 }));
add(spacer());
add(table(["Portfolio total", "Amount"], [
  ["Initial implementation", money(D.TOT_INITIAL)], ["Annual recurring", money(D.TOT_RECUR)], ["Year-1 cost", money(Y1)],
  ["Indicative 3-year total cost of ownership", money(D.TOT_INITIAL + 3 * D.TOT_RECUR)],
  ["Aggregate risk reduction", `${fmt0(totE)} -> ${fmt0(totR)} risk units (${Math.round((1 - totR / totE) * 100)}%)`],
], [60, 40]));
add(spacer());
add(p("Dependencies and capacity: the two-person team cannot deliver this portfolio alone. MDR (I03) offloads monitoring; the vulnerability FTE in I05 is costed at the workbook convention of 1,950 h x $80; implementation partners are assumed within initial costs. Change windows must be agreed with clinical operations."));

// ================================================================ 11. ROADMAP
add(h1("11. Implementation roadmap"));
add(table(["Horizon", "Actions", "Scenarios"], [
  ["0-90 days\nUrgent / quick wins", "I01 phase 1: FIDO2 for privileged, remote-access and EHR admin accounts; number matching for all; block internet macros\nI03: onboard MDR, full EDR coverage, connect identity, remote access, EHR audit, cloud and backup logs\nI04 phase 1: separate backup admin credentials, offline copy, first crown-jewel restore test\nI08: block public access on all cloud storage; review existing shares and links\nI02 quick win: rotate and restrict ehr-sync-svc; inventory privileged accounts\nI11: update IR plan, engage IR/forensic retainer, first executive-and-clinic tabletop\nI12: approve appetite, launch KRI dashboard and risk register\nIf any indicator in the case appendices reflects real activity: commission a compromise assessment immediately", "S1, S2, S3, S4, S5, S8"],
  ["3-6 months\nFoundations", "I02: PAM vault, tiered administration, just-in-time elevation\nI05: asset inventory, authenticated scanning, SLAs, exception process\nI09: EHR role redesign, first access recertification, EHR access analytics\nI10: vendor tiering; contract clauses for critical vendors\nI01: complete rollout and conditional access", "S1, S5, S6, S9, S10"],
  ["6-12 months\nMajor capabilities", "I04: immutable backup and isolated backup network complete; quarterly restore tests\nI06 phase 1: segment medical devices, guest Wi-Fi (WPA3), backup and server zones\nI07: medical device inventory, MDS2, passive IoMT monitoring\nI13: integrity monitoring and reconciliation for medication, allergy, lab, billing\nI09: DLP for health data", "S2, S3, S7, S10"],
  ["12-24 months\nStrategic maturity", "I06 complete: NAC in all clinics, east-west enforcement\nZero-trust remote access replacing network-level VPN\nFull disaster-recovery exercise including clinical downtime\nIndependent reassessment against this baseline; audit of remediation; insurance renewal", "S2, S6, S7, S9"],
], [16, 68, 16], { size: 15 }));

// ================================================================ 12. KRIs
add(h1("12. Key risk indicators"));
add(table(["ID", "KRI", "Scenarios", "Measurement", "Data source", "Owner", "Frequency", "Target", "Warning", "Critical"],
  D.KRIS, [6, 18, 7, 14, 11, 10, 8, 8, 8, 10], { size: 13 }));
add(spacer());
add(p("Report quarterly to the Steering Committee; any KRI at critical threshold escalates to executive leadership within one week."));

// ================================================================ 13. CASE VIEW
add(h1("13. Cross-check with the case escalation method"));
add(p("The case (s.8) defines a simpler normalized score (T x E x I, adjusted for control maturity) with escalation thresholds: executive review above 0.25, Board acceptance above 0.40. Mapping T = Pb(A), E = Pb(psi,A), I = (de + dm)/2 and control maturity C = theta (an analyst mapping - validate) gives:"));
const lvl = (x) => (x <= 0.07 ? "Low" : x <= 0.15 ? "Moderate" : x <= 0.25 ? "High" : "Critical");
add(table(["ID", "Current residual", "Level", "Post-treatment residual", "Level", "CRG status (appetite 0.30)"],
  S.map((s) => [s.id, s.norm_cur.toFixed(3), lvl(s.norm_cur), s.norm_post.toFixed(3), lvl(s.norm_post), s.class]), [8, 16, 14, 18, 14, 30], { size: 15 }));
add(spacer());
add(p("**Reading.** Under the case method, S8 is Critical today and S1, S2, S6 are High; after treatment all fall to Moderate or Low and within CIO acceptance authority. The CyberRiskGuardian ratio is more demanding because it compares damage directly with appetite. The two lenses should be reconciled before they are used for governance. In any case, the patient-safety clause of the appetite statement means S7 and S10 require clinical-leadership acceptance whatever their score."));

// ================================================================ 14. ASSUMPTIONS & LIMITATIONS
add(h1("14. Assumptions, limitations and methodological notes"));
add(h2("14.1 Key assumptions"));
[
  "Exercise evidence (phishing lure, closed PowerShell alert, MFA push approval, cloud download, service-account queries, backup failures) is treated as indicative of control weaknesses, not as a confirmed incident.",
  "Controls not mentioned in the case (SIEM, PAM, segmentation, immutable backup, DLP, vendor programme, DDoS protection) are assumed absent or immature.",
  "Push-based MFA is assumed to protect remote access; coverage of EHR, cloud and privileged accounts is unknown.",
  "Costs are Canadian mid-market indicative estimates for an organization of this size; no quotations were obtained.",
  "Treatment effectiveness assumes full, well-configured implementation and adequate operating staff.",
  "Equal split of shared-initiative costs across scenarios is a convention for cost-effectiveness only; it does not change portfolio totals.",
].forEach((x) => add(bl(x)));
add(h2("14.2 Information gaps - organizational input required"));
[
  "Asset and medical-device inventories; network diagrams and segmentation; internet-facing services.",
  "Identity architecture: MFA coverage and method, privileged and service accounts, conditional access.",
  "Backup architecture, retention, immutability, RTO/RPO and restore-test results.",
  "Cloud tenant configuration, sharing practices and logging.",
  "Vulnerability scan results, patch SLAs and end-of-life systems.",
  "Critical vendor list, contracts, remote-access methods and assurance reports.",
  "Incident history since 2020 and how attempts were detected; log retention.",
  "Cybersecurity budget (the stated CAD 100M IT budget appears inconsistent with a 15-person IT team) and cyber-insurance coverage.",
  "Applicable privacy regime and notification obligations (legal counsel).",
].forEach((x) => add(bl(x)));
add(h2("14.3 Methodological observations"));
add(bl("**The tolerance test depends only on damage, treatment and appetite.** Residual / Tolerated simplifies to ((de + dm)/2) x (1 - P x Q) / Appetite: probabilities, CVSS, resilience and criticality cancel out. They drive the magnitude used for ranking, but a rare scenario and a frequent one with the same damage and treatment receive the same tolerance verdict. Read the ratio together with estimated risk."));
add(bl("**CVSS is a required multiplier even where it does not fit.** For S5, S8 and S9 a representative weakness was scored so the workbook produces a value; these scores are flagged as limited applicability and must not be read as the severity of insider, detection or supplier risk."));
add(bl("**Mitigation is multiplicative.** Mitigated = Estimated x P x Q, so 80% probability and 80% impact reduction remove 64% of risk - a conservative convention that makes 'below tolerance' hard to reach for high-damage scenarios."));
add(bl("**Two scoring lenses disagree** (Section 13). Governance thresholds should be tied to one method."));
add(bl("**Limitations.** Desk-based assessment of a fictional case; no interviews, testing or technical data; AI-assisted estimates. Results are initial analytical hypotheses to be validated with IT, cybersecurity, privacy, clinical leadership and vendors."));

// ================================================================ 15. DECISIONS
add(h1("15. Management decisions required"));
add(table(["#", "Decision", "Options", "Recommended", "Authority"], [
  ["D1", "Confirm cybersecurity risk appetite", "0.30 (analyst) / 0.40 (case) / other", "Adopt 0.30 for patient-safety and health-data risks; document separately any higher appetite for innovation", "Board / ownership group"],
  ["D2", "Approve treatment portfolio", `Full (${money(Y1)} Y1) / phased / partial`, "Approve 0-90-day actions immediately; release remaining funding by phase against KRIs", "Executive leadership, Board"],
  ["D3", "Residual risk above tolerance (S2, S5, S7, S9, S10)", "Accept / further mitigate / transfer / avoid", "Formal acceptance with review in 6 months; transfer part of S2 and S9 via insurance and contracts", "Board / ownership group"],
  ["D4", "Patient-safety residual (S7, S10)", "Accept with compensating controls / restrict device connectivity", "CMO sign-off on compensating controls and clinical downtime procedures", "Clinical leadership + Executive"],
  ["D5", "Staffing", "Status quo / +1 FTE / +2 FTE / managed services", "+1 vulnerability FTE (costed) and MDR; assess IAM/PAM engineer", "CIO, Executive"],
  ["D6", "Cyber-insurance review", "Maintain / increase / obtain", "Obtain broker review aligned with S2 and S9 residuals", "CFO"],
  ["D7", "Validate estimates and close evidence gaps", "-", "SME workshop within 60 days; re-run workbook", "Cybersecurity lead"],
  ["D8", "Assign risk owners", "-", "Confirm owners proposed in Section 7", "Executive leadership"],
], [4, 20, 20, 36, 20], { size: 15 }));
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
Packer.toBuffer(doc).then((b) => { fs.writeFileSync("../output/MediBec_Cybersecurity_Risk_Assessment.docx", b); console.log("ok"); });
