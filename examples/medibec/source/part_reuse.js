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

