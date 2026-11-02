#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Marc-André Léger
"""CyberRiskGuardian label check (Core, Community Edition; ported from the Threat Live-feed plugin).

Checks that risk-assessment outputs carry the three standard markers (book v2.51b ch06 "Evidence discipline
and the three markers"), in English or French:
  - "Analytical estimate — validation required"              when quantitative estimates appear
      FR "Estimation analytique — validation requise"
  - "Information gap — organizational validation required"   when the text admits missing information
      FR "Lacune d’information — validation organisationnelle requise"
  - "Management decision required"                           when a report or summary presents results
      FR "Décision de la direction requise"
and flags OUTDATED wording: the retired markers ("Analyst estimate — validation required", "Evidence gap —
organizational input required", "Lacune de preuve — renseignement organisationnel requis", "Estimation de
l'analyste…"), which are detected but never accepted as the current marker, and retired method wording
(computed indices called KRIs, "within 4–12%" budgets, ABS() residuals, a 10,000 factor, the 1.25 threshold).

Modes
  hook  (default when stdin is JSON): PostToolUse on Write/Edit, or Stop. Warns, never blocks a tool;
        on Stop it asks Claude once to add a missing estimate marker (stop_hook_active prevents loops).
  cli   python3 check_labels.py FILE [FILE...]  -> report; exit 1 if a marker is missing or wording is outdated.

Reads only the files and messages it is given. No network access, no storage.
"""
import json, os, re, sys, unicodedata, zipfile

DASH = r"\s*[—–\-:]\s*"
CURRENT = {
    "estimate": [rf"analytical estimates?{DASH}validation required", rf"estimations? analytiques?{DASH}validation requise"],
    "gap": [rf"information gaps?{DASH}organi[sz]ational validation required",
            rf"lacunes? d.information{DASH}validation organisationnelle requise"],
    "decision": [r"management decisions? required", r"d[ée]cisions? de la direction requises?"],
}
OUTDATED_MARKERS = [
    (rf"analyst(?:'s)? estimates?{DASH}validation required", "Analytical estimate — validation required"),
    (rf"evidence gaps?{DASH}organi[sz]ational (?:input|validation) required", "Information gap — organizational validation required"),
    (rf"lacune de preuve{DASH}(?:renseignement|information) organisationnel(?:le)? requis(?:e)?",
     "Lacune d’information — validation organisationnelle requise"),
    (rf"estimation de l.analyste{DASH}validation requise", "Estimation analytique — validation requise"),
]
OUTDATED_METHOD = [
    (r"\b(?:calculated|computed|compute|calculate)\s+(?:the\s+)?(?:CyberRiskGuardian\s+|CRG\s+)?KRIs?\b|\bKRI formulas\b",
     "computed values are risk indices (Re, Rt, residual, ratio), not KRIs; KRIs are measured indicators with thresholds"),
    (r"\b(?:within|in)\s+(?:the\s+)?4\s*(?:%\s*)?[-–]\s*12\s*%|\b4\s*[-–]\s*12\s*%\s+(?:band|guideline)",
     "budget rule = point targets 12% / 7.8% / 4% of total IT budget incl. salaries for appetite 0.30 / 0.50 / 0.70, linear in between (best practice), not a 4–12% band"),
    (r"\bABS\s*\(\s*L\s*[-−]\s*R|\bresidual\b[^.\n]{0,40}\bABS\s*\(",
     "residual = Re − Mitigated with the MAX(0, …) guard and reductions validated to 0–1, never ABS()"),
    (r"\bfactor\b[^.\n]{0,20}\b10[ ,.]?000\b", "the multiplication factor is 1,000"),
    (r"\b1\.25\b[^.\n]{0,40}\b(?:threshold|ratio|tolerance)", "the ratio bands are < 0.90 / 0.90–1.10 / > 1.10; the 1.25 threshold is retired"),
]
LABEL_TEXT = {
    "estimate": "Analytical estimate — validation required (FR: Estimation analytique — validation requise)",
    "gap": "Information gap — organizational validation required (FR: Lacune d’information — validation organisationnelle requise)",
    "decision": "Management decision required (FR: Décision de la direction requise)",
}
QUANT = re.compile(r"Pb\s*\(\s*A\s*\)|Pb\s*\(\s*[ψψ]|δ\s*[em]\b|θ\s*\(|μ\s*\(\s*E|\bPbA\b|\bPbx\b|estimated risk|risque estim|tolerated risk|risque tol[ée]r", re.I)
NUMBER_NEAR = re.compile(r"\b0[.,]\d{1,2}\b")
REPORT = re.compile(r"executive summary|management summary|summary report|risk assessment report|quick scan|r[ée]sum[ée] (?:ex[ée]cutif|pour la direction)|rapport d.[ée]valuation|recommendations?|recommandations?", re.I)
GAPWORDS = re.compile(r"\bnot (?:provided|supplied|documented|available)\b|\bunknown\b|\bto be confirmed\b|\bTBD\b|\bmissing information\b|non fourni|inconnu|à confirmer|information manquante", re.I)
TEXT_EXT = {".md", ".markdown", ".txt", ".html", ".htm", ".csv", ".json"}
# Methodology sources quote both current and retired wording on purpose: never check them in hook mode.
SKIP_PARTS = ("/skills/", "/references/", "/hooks/", "/agents/", "/commands/", "/output-styles/", "/rules/",
              "/evals/", "/.claude/", "/plugins/", "/node_modules/", "/.git/")


def norm(t):
    return unicodedata.normalize("NFC", t).replace("’", "'")


def has(text, key):
    return any(re.search(p, text, re.I) for p in CURRENT[key])


NEGATION = re.compile(r"\b(?:never|not|no|instead of|without|retired|replaces?|rather than|ni|jamais|pas|au lieu)\b", re.I)


def outdated(text):
    """List of (found wording, replacement) for retired markers and method wording.
    Method wording that is explicitly negated nearby ("never ABS()", "not a 4–12% band") is not reported."""
    t = norm(text)
    out = []
    for pat, rep in OUTDATED_MARKERS:
        m = re.search(pat, t, re.I)
        if m:
            out.append((m.group(0).strip(), rep))
    for pat, rep in OUTDATED_METHOD:
        for m in re.finditer(pat, t, re.I):
            window = t[max(0, m.start() - 40):m.end()]
            if not NEGATION.search(window):
                out.append((m.group(0).strip(), rep))
                break
    return out


def required(text, name=""):
    """Return the marker keys this text needs but lacks (a retired marker does not count)."""
    t = norm(text)
    need = []
    quant = QUANT.search(t) and len(NUMBER_NEAR.findall(t)) >= 3
    if quant and not has(t, "estimate"):
        need.append("estimate")
    if (REPORT.search(t) or re.search(r"summary|report|rapport|sommaire|scan", name, re.I)) and quant and not has(t, "decision"):
        need.append("decision")
    if GAPWORDS.search(t) and not has(t, "gap"):
        need.append("gap")
    return need


def read_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "ignore")
        return re.sub(r"<[^>]+>", " ", xml.replace("</w:p>", "\n"))
    if ext in TEXT_EXT:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    return None


def message(need, old, where):
    lines = []
    if need:
        lines.append(f"CyberRiskGuardian label check: {where} is missing required markers:")
        lines += [f"- {LABEL_TEXT[k]}" for k in need]
        lines.append("Add each marker where it applies (estimated parameters, missing facts, decisions for management).")
    if old:
        lines.append(f"CyberRiskGuardian label check: {where} uses outdated wording:")
        lines += [f"- \"{found}\" → {rep}" for found, rep in old]
    return "\n".join(lines)


def hook(data):
    event = data.get("hook_event_name", "")
    if event == "PostToolUse":
        path = (data.get("tool_input") or {}).get("file_path") or ""
        if not path or any(p in path.replace("\\", "/") for p in SKIP_PARTS):
            return 0
        try:
            text = read_file(path)
        except Exception:
            return 0
        if not text:
            return 0
        need, old = required(text, os.path.basename(path)), outdated(text)
        if need or old:
            msg = message(need, old, os.path.basename(path))
            print(json.dumps({"systemMessage": msg,
                              "hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": msg}}))
        return 0
    if event == "Stop":
        if data.get("stop_hook_active"):
            return 0
        text = data.get("last_assistant_message") or ""
        if isinstance(text, list):
            text = " ".join(x.get("text", "") for x in text if isinstance(x, dict))
        need = [k for k in required(str(text)) if k == "estimate"]
        if need:
            print(json.dumps({"decision": "block",
                              "reason": message(need, [], "your last reply") + " Restate the reply with the markers; change nothing else."}))
            return 0
        old = outdated(str(text))      # outdated wording alone never blocks a reply: warn the user only
        if old:
            print(json.dumps({"systemMessage": message([], old, "the last reply")}))
        return 0
    return 0


def cli(paths):
    bad = 0
    for p in paths:
        try:
            text = read_file(p)
        except Exception as e:
            print(f"{p}: cannot read ({e})"); bad += 1; continue
        if text is None:
            print(f"{p}: skipped (unsupported type)"); continue
        need, old = required(text, os.path.basename(p)), outdated(text)
        if need or old:
            bad += 1
            if need:
                print(f"{p}: MISSING " + "; ".join(LABEL_TEXT[k] for k in need))
            for found, rep in old:
                print(f"{p}: OUTDATED \"{found}\" → {rep}")
        else:
            print(f"{p}: OK")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(cli(sys.argv[1:]))
    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        sys.exit(0)
    sys.exit(hook(data))
