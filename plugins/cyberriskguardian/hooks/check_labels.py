#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Marc-André Léger
"""CyberRiskGuardian methodology label check.

Checks that risk-assessment outputs carry the standard markers:
  - "Analyst estimate — validation required"         when quantitative estimates appear
  - "Management decision required"                    when the output is a report or summary
  - "Evidence gap — organizational input required"    when the text admits missing information

French equivalents and the earlier "Analytical estimate" / "Information gap" wording are accepted.

Modes
  hook  (default when stdin is JSON): PostToolUse on Write/Edit, or Stop. Warns, never blocks a tool;
        on Stop it asks Claude once to add missing markers (stop_hook_active prevents loops).
  cli   python3 check_labels.py FILE [FILE...]  -> report; exit 1 if a required marker is missing.

Reads only the files and messages it is given. No network access, no storage.
"""
import json, os, re, sys, unicodedata, zipfile

DASH = r"\s*[—–\-:]\s*"
MARKERS = {
    "estimate": [rf"analyst estimate{DASH}validation required", rf"analytical estimate{DASH}validation required",
                 rf"estimation de l.analyste{DASH}validation requise", rf"estimation analytique{DASH}validation requise"],
    "gap": [rf"evidence gap{DASH}organi[sz]ational input required", rf"information gap{DASH}organi[sz]ational validation required",
            rf"lacune de preuve{DASH}information organisationnelle requise", rf"lacune d.information{DASH}validation organisationnelle requise"],
    "decision": [r"management decision required", r"d[ée]cision de la direction requise"],
}
LABEL_TEXT = {
    "estimate": "Analyst estimate — validation required (FR: Estimation de l'analyste — validation requise)",
    "gap": "Evidence gap — organizational input required (FR: Lacune de preuve — information organisationnelle requise)",
    "decision": "Management decision required (FR: Décision de la direction requise)",
}
QUANT = re.compile(r"Pb\s*\(\s*A\s*\)|Pb\s*\(\s*[ψψ]|δ\s*[em]\b|θ\s*\(|μ\s*\(\s*E|\bPbA\b|\bPbx\b|estimated risk|risque estim|tolerated risk|risque tol[ée]r", re.I)
NUMBER_NEAR = re.compile(r"\b0[.,]\d{1,2}\b")
REPORT = re.compile(r"executive summary|management summary|summary report|risk assessment report|r[ée]sum[ée] (?:ex[ée]cutif|pour la direction)|rapport d.[ée]valuation|recommendations?|recommandations?", re.I)
GAPWORDS = re.compile(r"\bnot (?:provided|supplied|documented|available)\b|\bunknown\b|\bto be confirmed\b|\bTBD\b|\bmissing information\b|non fourni|inconnu|à confirmer|information manquante", re.I)
TEXT_EXT = {".md", ".markdown", ".txt", ".html", ".htm", ".csv", ".json"}
SKIP_PARTS = ("/skills/", "/references/", "/hooks/", "/agents/", "/commands/", "/output-styles/", "/evals/", "/.claude/", "/node_modules/")


def norm(t):
    return unicodedata.normalize("NFC", t).replace("’", "'")


def has(text, key):
    return any(re.search(p, text, re.I) for p in MARKERS[key])


def required(text, name=""):
    """Return the marker keys this text needs but lacks."""
    t = norm(text)
    need = []
    quant = QUANT.search(t) and len(NUMBER_NEAR.findall(t)) >= 3
    if quant and not has(t, "estimate"):
        need.append("estimate")
    if (REPORT.search(t) or re.search(r"summary|report|rapport|sommaire", name, re.I)) and quant and not has(t, "decision"):
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


def message(need, where):
    lines = [f"CyberRiskGuardian label check: {where} is missing required markers:"]
    lines += [f"- {LABEL_TEXT[k]}" for k in need]
    lines.append("Add each marker where it applies (estimated parameters, missing facts, decisions for management).")
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
        need = required(text, os.path.basename(path))
        if need:
            msg = message(need, os.path.basename(path))
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
                              "reason": message(need, "your last reply") + " Restate the reply with the markers; change nothing else."}))
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
        need = required(text, os.path.basename(p))
        if need:
            bad += 1
            print(f"{p}: MISSING " + "; ".join(LABEL_TEXT[k] for k in need))
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
