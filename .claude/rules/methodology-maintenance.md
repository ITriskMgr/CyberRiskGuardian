# Methodology maintenance

- `plugins/cyberriskguardian/` (Core) and `plugins/crg-governance-textbook/` are the single sources of truth. Edit there, then run `bash scripts/sync-project-config.sh` to refresh `.claude/`.
- Paid content (CRG Pro plugins, Compliance and Sector Packs, Educator Edition, CRGO ontology data, desktop builds) lives in the private `crg-pro-marketplace` repository and never in this public repository.
- Bump `version` in each changed plugin's `.claude-plugin/plugin.json` and in `.claude-plugin/marketplace.json` for any behaviour change (semver).
- Validate before release: `claude plugin validate plugins/cyberriskguardian/.claude-plugin/plugin.json`, `claude plugin validate plugins/crg-governance-textbook/.claude-plugin/plugin.json` and `claude plugin validate .`.
- Regenerate and run the evals: `python3 scripts/build_evals.py`, then `claude plugin eval plugins/cyberriskguardian --allow-tools Bash Write` and `claude plugin eval plugins/crg-governance-textbook`.
- Regression check: `python3 plugins/cyberriskguardian/skills/crg-quantify/scripts/crg_calc.py examples/medibec/source/medibec30.json` must reproduce total estimated 84,305 and residual 41,023.
- Never change a CRG formula silently; record methodology decisions in `master-task-spec.md` ("Resolved methodology choices").
