---
description: Check deliverables for the three CyberRiskGuardian markers and outdated wording
argument-hint: "[file(s) - .md .txt .html .csv .json .docx; default: the deliverables just produced]"
allowed-tools: Bash, Read, Glob
---

Run the CyberRiskGuardian label check (book v2.51b, Chapter 6 "Evidence discipline and the three markers"; CRG appendix) on: $ARGUMENTS

```
python3 "${CLAUDE_PLUGIN_ROOT}/hooks/check_labels.py" <file> [<file> ...]
```

(In the repository without the plugin installed: `python3 plugins/cyberriskguardian/hooks/check_labels.py <file> ...`.) With no argument, check the deliverables written in this session.

The check is read-only and offline. It reports, per file:
- **MISSING** — a required marker is absent: **Analytical estimate — validation required** where quantitative estimates appear, **Information gap — organizational validation required** where the text admits missing facts, **Management decision required** in a report, summary or quick scan that presents results (French equivalents accepted: *Estimation analytique — validation requise*, *Lacune d’information — validation organisationnelle requise*, *Décision de la direction requise*);
- **OUTDATED** — retired wording: "Analyst estimate — validation required", "Evidence gap — organizational input required", "Lacune de preuve — renseignement organisationnel requis", "Estimation de l'analyste", computed indices called KRIs, a "4–12%" budget band, an ABS() residual, a 10,000 factor or the 1.25 threshold.

Summarize the findings in a table (file, finding, suggested fix). Do not edit the files unless asked; when asked, change only the wording, never a number. The check is a heuristic aid for the quality review, not a substitute for it (`/qc-review`).
