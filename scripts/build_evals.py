#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Marc-André Léger
"""Generate the CyberRiskGuardian eval suites (case.yaml files) and copy shared fixtures into each case.

Run from the repository root after changing a case or a fixture:
    python3 scripts/build_evals.py
Then run a suite, for example:
    claude plugin eval plugins/cyberriskguardian --allow-tools Bash Write --tag smoke
"""
import json, os, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EST = r"Analyst estimate|Analytical estimate|Estimation de l.analyste|Estimation analytique"
GAP = r"Evidence gap|Information gap|Lacune de preuve|Lacune d.information"
DEC = r"Management decision required|D[ée]cision de la direction requise"
FIND = "The fixture file {f} is in a directory added to this session; locate it with Glob before reading it."

CORE = {
 "context-intake-medibec": dict(
  tags=["smoke", "intake"], runs=2, fixtures=["medibec_profile.md"], max_turns=15, tools=["Read", "Glob", "Grep", "Skill"],
  prompt=FIND.format(f="medibec_profile.md") + " Using only that profile, build the evidence register, the crown-jewel table and the scope statement for a CyberRiskGuardian risk assessment of MediBec. Do not ask me questions; state assumptions instead.",
  graders=[
   dict(name="evidence-ids", type="regex", pattern=r"\bE1\b"),
   dict(name="gap-marker", type="regex", pattern=GAP, flags="i"),
   dict(name="intake-skill", type="tool_used", tool="Skill", input_match="crg-context-intake|cyber-risk-assessment"),
   dict(name="grounded", type="llm", criteria="The reply contains an evidence register with IDs, a crown-jewel table that includes the electronic health record and clinical systems, and a scope with a 12-month horizon. Facts are traceable to the MediBec profile; the reply does not invent facts absent from it (such as named vendors, headcounts or budgets) and labels missing information as an evidence gap."),
  ]),
 "context-intake-interview": dict(
  tags=["smoke", "intake"], runs=2, max_turns=6, tools=["Skill"],
  prompt="I want a cybersecurity risk assessment for my small accounting firm. I don't have any documents.",
  graders=[
   dict(name="interview", type="llm", criteria="The assistant starts an interview: it asks a first round of at most five numbered questions about the firm (a numbered question may have short parts or examples), covering topics such as mission, clients, critical services, data, systems or obligations. It does not invent facts about the firm and does not produce scenarios or risk scores yet."),
  ]),
 "risk-appetite-healthcare": dict(
  tags=["smoke", "appetite"], runs=2, fixtures=["medibec_profile.md"], max_turns=12, tools=["Read", "Glob", "Skill"],
  prompt=FIND.format(f="medibec_profile.md") + " No risk appetite has been approved for MediBec. Estimate one for the CyberRiskGuardian assessment and give the materiality criteria.",
  graders=[
   dict(name="appetite-line", type="regex", pattern=r"Estimated Risk Appetite:\s*\**\s*0[.,](?:[12]\d?|3\d?)\b"),
   dict(name="estimate-marker", type="regex", pattern=EST, flags="i"),
   dict(name="decision-marker", type="regex", pattern=DEC, flags="i"),
   dict(name="rationale", type="llm", criteria="The appetite is justified with factors from the profile (patient data sensitivity, clinical continuity, privacy and health-sector obligations) and the reply explains what a lower or higher value would change. It translates the appetite into materiality criteria for scenario selection."),
  ]),
 "scenario-builder-20-to-10": dict(
  tags=["scenarios"], runs=1, fixtures=["medibec_profile.md"], max_turns=25, timeout=900, tools=["Read", "Glob", "Grep", "Skill"],
  prompt=FIND.format(f="medibec_profile.md") + " Generate 20 candidate cyber risk scenarios for MediBec and select the 10 most material. Keep each candidate to one line and give a one-line rationale for each selected scenario. No workbook, reply in the conversation.",
  graders=[
   dict(name="builder-skill", type="tool_used", tool="Skill", input_match="crg-scenario-builder|cyber-risk-assessment"),
   dict(name="causal-chains", type="llm", criteria="There are 20 candidates and 10 selected scenarios. Each selected scenario is a causal chain (threat source, weakness, affected asset or process, event, organizational consequence), not a threat name such as 'ransomware' or 'phishing'. The selection is not dominated by one technique, includes at least one non-technical or third-party scenario, and each selection has a rationale tied to MediBec."),
   dict(name="no-cvss-ranking", type="llm", criteria="The reply does not rank or select scenarios on CVSS scores."),
  ]),
 "quantify-worked-example": dict(
  tags=["smoke", "regression", "quantify"], runs=3, max_turns=10, tools=["Bash", "Read", "Skill", "Write"],
  prompt="Compute the CyberRiskGuardian KRIs for one scenario with the calculator, not by hand: Pb(A) 0.70, Pb(ψ,A) 0.60, CVSS-B 9.4, δe 0.60, δm 0.80, θ 0.50, μ(E) 0.65, appetite 0.35, factor 1,000, probability reduction 0.65, impact reduction 0.70. Give estimated, tolerated, mitigated and residual risk, the ratio and the classification.",
  graders=[
   dict(name="estimated", type="regex", pattern=r"3[,\s ]?592"),
   dict(name="tolerated", type="regex", pattern=r"1[,\s ]?796"),
   dict(name="residual", type="regex", pattern=r"1[,\s ]?958"),
   dict(name="ratio", type="regex", pattern=r"1[.,]09"),
   dict(name="calculator-used", type="tool_used", tool="Bash", input_match="crg_calc"),
   dict(name="classification", type="llm", criteria="The scenario is classified as approximately at tolerance (ratio between 0.90 and 1.10)."),
  ]),
 "quantify-cvss-vector": dict(
  tags=["smoke", "quantify"], runs=3, max_turns=8, tools=["Bash", "Read", "Skill"],
  prompt="Score this CVSS v4.0 vector with the CyberRiskGuardian calculator and tell me the severity: CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:P/VC:H/VI:H/VA:N/SC:N/SI:N/SA:N",
  graders=[
   dict(name="score", type="regex", pattern=r"\b8[.,]6\b"),
   dict(name="severity", type="regex", pattern=r"High|[ÉE]lev[ée]", flags="i"),
   dict(name="tool", type="tool_used", tool="Bash", input_match="crg_calc|cvss"),
  ]),
 "quantify-medibec30-regression": dict(
  tags=["regression", "quantify"], runs=1, fixtures=["medibec30.json"], max_turns=10, tools=["Bash", "Read", "Glob", "Skill"],
  prompt=FIND.format(f="medibec30.json") + " It is a CyberRiskGuardian data model for the 30-scenario MediBec assessment. Run the CyberRiskGuardian calculator on it and report the total estimated risk, the total residual risk and the counts above, approximately at and below tolerance.",
  graders=[
   dict(name="total-estimated", type="regex", pattern=r"84[,\s ]?491"),
   dict(name="total-residual", type="regex", pattern=r"41[,\s ]?104"),
   dict(name="counts", type="regex", pattern=r"12\b[^\n]{0,60}above|above[^\n]{0,60}\b12\b", flags="i"),
   dict(name="tool", type="tool_used", tool="Bash", input_match="crg_calc"),
  ]),
 "summary-report-markers": dict(
  tags=["smoke", "report"], runs=2, max_turns=8, tools=["Skill"],
  prompt="Write the CyberRiskGuardian management summary for Clinique Nord from these calculator results (appetite 0.30, estimated by the analyst, not approved; supplier list not provided). S1 Admin credential compromise: estimated 5,428, tolerated 2,412, ratio 2.25, above tolerance. S2 Ransomware on scheduling servers: estimated 4,100, tolerated 2,050, ratio 2.00, above tolerance. S3 Supplier portal data leak: estimated 1,200, tolerated 1,150, ratio 1.04, approximately at tolerance. Reply in the conversation.",
  graders=[
   dict(name="decision-marker", type="regex", pattern=DEC, flags="i"),
   dict(name="estimate-marker", type="regex", pattern=EST, flags="i"),
   dict(name="gap-marker", type="regex", pattern=GAP, flags="i"),
   dict(name="conclusion-first", type="llm", criteria="The summary opens with the conclusion (two scenarios above tolerance, one approximately at tolerance, and what management must decide), includes a portfolio table with the given numbers unchanged, marks treatment directions as indicative and uncosted, and lists next steps that need CRG Pro (costed treatment, sensitivity, budget) without inventing costs."),
  ]),
 "assess-french": dict(
  tags=["language"], runs=1, fixtures=["medibec_profile.md"], max_turns=30, timeout=1200, tools=["Read", "Glob", "Grep", "Skill", "Bash", "Write"],
  prompt="Le fichier medibec_profile.md se trouve dans un répertoire ajouté à cette session; trouvez-le avec Glob. Fais un balayage rapide CyberRiskGuardian pour MediBec : 20 candidats, 10 scénarios, verdicts avant traitement. Réponds dans la conversation.",
  graders=[
   dict(name="french-markers", type="regex", pattern=r"Estimation de l.analyste|D[ée]cision de la direction requise", flags="i"),
   dict(name="french-reply", type="llm", criteria="The final reply is written in French and contains a table of about 10 scenarios with estimated risk, tolerated risk, ratio and a classification, followed by priorities and decisions for management."),
  ]),
}

TEXTBOOK = {
 "three-lines-model": dict(
  tags=["smoke", "textbook"], runs=2, max_turns=10, tools=["Read", "Grep", "Glob", "Skill"],
  prompt="Explain the three lines model as the CyberRiskGuardian governance textbook teaches it, and cite the chapter.",
  graders=[
   dict(name="citation", type="regex", pattern=r"Chap(?:ter|itre)\s+\d+", flags="i"),
   dict(name="read-book", type="tool_used", tool="Read", input_match=r"references/ch\d+"),
   dict(name="faithful", type="llm", criteria="The explanation describes the governing body, management (first and second line roles) and independent assurance (third line), cites a chapter of Léger's Introduction to Cybersecurity Governance, and does not present invented quotations."),
  ]),
 "appetite-vs-tolerance": dict(
  tags=["textbook"], runs=2, max_turns=10, tools=["Read", "Grep", "Glob", "Skill"],
  prompt="Selon le manuel de gouvernance de CyberRiskGuardian, quelle est la différence entre l'appétit pour le risque et la tolérance au risque ? Cite le chapitre.",
  graders=[
   dict(name="citation", type="regex", pattern=r"Chap(?:ter|itre)\s+\d+", flags="i"),
   dict(name="answer", type="llm", criteria="The reply is in French, distinguishes risk appetite (the amount and type of risk the organization is willing to pursue or retain) from risk tolerance (acceptable variation or limits around objectives), and cites a chapter of the textbook."),
  ]),
}


def yaml_scalar(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    return json.dumps(v, ensure_ascii=False)


def write_case(plugin_dir, name, c):
    d = os.path.join(plugin_dir, "evals", name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    lines = ['schema_version: "1.1"', f"name: {name}", f"tags: {json.dumps(c['tags'])}", f"runs: {c['runs']}", "execution:",
             f"  prompt: {yaml_scalar(c['prompt'])}", f"  max_turns: {c['max_turns']}",
             f"  timeout_seconds: {c.get('timeout', 600)}", f"  allowed_tools: {json.dumps(c['tools'])}"]
    if c.get("fixtures"):
        os.makedirs(os.path.join(d, "fixtures"))
        for f in c["fixtures"]:
            shutil.copy(os.path.join(plugin_dir, "evals", "_shared", f), os.path.join(d, "fixtures", f))
        lines += ["context:", "  add_dirs: [fixtures]"]
    lines.append("graders:")
    for g in c["graders"]:
        lines.append(f"  - name: {g['name']}")
        for k, v in g.items():
            if k != "name":
                lines.append(f"    {k}: {yaml_scalar(v)}")
    with open(os.path.join(d, "case.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build(plugin_dir, cases):
    for name, c in cases.items():
        write_case(plugin_dir, name, c)
    print(f"{plugin_dir}: {len(cases)} cases")


if __name__ == "__main__":
    build(os.path.join(ROOT, "plugins", "cyberriskguardian"), CORE)
    tb = os.path.join(ROOT, "plugins", "crg-governance-textbook")
    build(tb, TEXTBOOK)
