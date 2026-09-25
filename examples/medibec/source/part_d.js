
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
