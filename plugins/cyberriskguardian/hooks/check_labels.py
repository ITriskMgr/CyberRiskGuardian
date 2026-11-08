#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Marc-André Léger
"""CyberRiskGuardian label check (Core, Community Edition; the same matcher as the Threat Live-feed plugin).

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
        Reads Markdown, text, HTML, CSV, JSON, Word (.docx) and Excel (.xlsx: shared strings, inline strings,
        comments and text literals in formulas).

Reads only the files and messages it is given. No network access, no storage.
"""
import html, json, os, re, sys, unicodedata, zipfile

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
QUANT_WORDS = r"Pb\s*\(\s*A\s*\)|Pb\s*\(\s*[ψψ]|δ\s*[em]\b|θ\s*\(|μ\s*\(\s*E|\bPbA\b|\bPbx\b|estimated risk|risque estim|tolerated risk|risque tol[ée]r"
QUANT = re.compile(QUANT_WORDS, re.I)
NUMBER_NEAR = re.compile(r"\b0[.,]\d{1,2}\b")
# A parameter name followed by its value ("Pb(A) 0.70", "Pb(ψ,A) = 0.6", "δe: 0,60") is an estimate on its own:
# two such pairs are enough, where the older rule waited for three loose numbers (NV-P11 false negative).
PARAM_VALUE = re.compile(r"(?:" + QUANT_WORDS + r")[^\n0-9]{0,12}\b[01][.,]\d{1,2}\b", re.I)
REPORT = re.compile(r"executive summary|management summary|summary report|risk assessment report|quick scan|r[ée]sum[ée] (?:ex[ée]cutif|pour la direction)|rapport d.[ée]valuation|recommendations?|recommandations?", re.I)
# "unknown" names a gap only when it is not the attribution of a threat source ("an unknown external actor",
# "unknown attacker", "source inconnue" are scenario wording, not missing organizational facts) (NV-P11).
GAPWORDS = re.compile(r"\bnot (?:provided|supplied|documented|available)\b"
                      r"|\bunknown\b(?!\s+(?:(?:external|internal|remote|threat|malicious|criminal|state)\s+)?(?:actors?|attackers?|adversar(?:y|ies)|groups?|part(?:y|ies)|sources?|threats?)\b)"
                      r"|\bto be confirmed\b|\bTBD\b|\bmissing information\b|non fourni"
                      r"|(?<!acteur )(?<!attaquant )(?<!source )(?<!menace )\binconnue?s?\b|à confirmer|information manquante", re.I)
TEXT_EXT = {".md", ".markdown", ".txt", ".html", ".htm", ".csv", ".json"}
OFFICE_EXT = {".docx", ".xlsx"}
# Code files carry user-facing templates (prompts, exports): only their wording is checked — a module is not a
# deliverable, so it is never asked to carry the markers itself.
CODE_EXT = {".js", ".mjs", ".py"}
# Methodology sources quote both current and retired wording on purpose: never check them in hook mode. The skip
# list is anchored to the plugin's own folder, the project's .claude/ folder and the repository's plugins/ folder,
# so a client deliverable that merely sits under a path containing "/plugins/" or "/references/" is still checked
# (NV-P11). Tooling folders (/node_modules/, /.git/) are skipped anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
SKIP_ANYWHERE = ("/node_modules/", "/.git/")


def _skip_roots():
    roots = [os.path.abspath(os.path.join(HERE, ".."))]                  # this plugin
    for env in ("CLAUDE_PLUGIN_ROOT",):
        if os.environ.get(env):
            roots.append(os.path.abspath(os.environ[env]))
    proj = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    roots += [os.path.join(os.path.abspath(proj), ".claude"), os.path.join(os.path.abspath(proj), "plugins")]
    return roots


def skipped(path):
    """True for files inside the plugin, the project's .claude/ or plugins/ folders, or tooling folders."""
    p = os.path.abspath(path).replace("\\", "/")
    if any(part in p + "/" for part in SKIP_ANYWHERE):
        return True
    for root in _skip_roots():
        root = root.replace("\\", "/").rstrip("/") + "/"
        if p.startswith(root):
            return True
    return False


def norm(t):
    return unicodedata.normalize("NFC", t).replace("’", "'")


def has(text, key):
    return any(re.search(p, text, re.I) for p in CURRENT[key])


NEGATION = re.compile(r"\b(?:never|not|no|instead of|without|retired|replaces?|rather than|ni|jamais|pas|au lieu)\b", re.I)
SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")


def sentence_start(t, pos):
    """Start of the sentence containing `pos`: a negation only counts in the same sentence. A "never" at the
    end of the previous sentence must not hide an ABS() residual in the next one."""
    start = 0
    for m in SENTENCE_END.finditer(t, 0, pos):
        start = m.end()
    return start


def sentence_end(t, pos):
    """End of the sentence containing `pos` (so a negation after the match - "the 1.25 threshold is retired" -
    counts too, NV-P11)."""
    m = SENTENCE_END.search(t, pos)
    return m.start() if m else len(t)


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
            window = t[max(sentence_start(t, m.start()), m.start() - 80):min(sentence_end(t, m.end()), m.end() + 80)]
            if not NEGATION.search(window):
                out.append((m.group(0).strip(), rep))
                break
    return out


def required(text, name=""):
    """Return the marker keys this text needs but lacks (a retired marker does not count)."""
    t = norm(text)
    need = []
    quant = bool(QUANT.search(t)) and (len(NUMBER_NEAR.findall(t)) >= 3 or len(PARAM_VALUE.findall(t)) >= 2)
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
        return html.unescape(re.sub(r"<[^>]+>", " ", xml.replace("</w:p>", "\n")))
    if ext == ".xlsx":                 # shared strings, inline strings and text literals in formulas (NV-P11)
        parts = []
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name == "xl/sharedStrings.xml" or (name.startswith("xl/worksheets/") and name.endswith(".xml")) \
                        or name == "xl/comments1.xml" or (name.startswith("xl/comments") and name.endswith(".xml")):
                    xml = z.read(name).decode("utf-8", "ignore")
                    parts.append(re.sub(r"<[^>]+>", " ", xml.replace("</si>", "\n").replace("</row>", "\n")))
        return html.unescape("\n".join(parts))
    if ext in TEXT_EXT or ext in CODE_EXT:
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
        if not path or skipped(path):
            return 0
        try:
            text = read_file(path)
        except Exception:
            return 0
        if not text:
            return 0
        code = os.path.splitext(path)[1].lower() in CODE_EXT
        need, old = ([] if code else required(text, os.path.basename(path))), outdated(text)
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
        code = os.path.splitext(p)[1].lower() in CODE_EXT
        need, old = ([] if code else required(text, os.path.basename(p))), outdated(text)
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
