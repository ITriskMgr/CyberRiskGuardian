# Methodology maintenance

- `plugins/cyberriskguardian/` (Core) and `plugins/crg-governance-textbook/` are the single sources of truth. Edit there, then run `bash scripts/sync-project-config.sh` to refresh `.claude/`.
- Paid content (CRG Pro plugins, Compliance and Sector Packs, Educator Edition, CRGO ontology data, desktop builds) lives in the private `crg-pro-marketplace` repository and never in this public repository.
- Bump `version` in each changed plugin's `.claude-plugin/plugin.json` and in `.claude-plugin/marketplace.json` for any behaviour change (semver).
- Validate before release: `claude plugin validate plugins/cyberriskguardian/.claude-plugin/plugin.json`, `claude plugin validate plugins/crg-governance-textbook/.claude-plugin/plugin.json` and `claude plugin validate .`.
- Regenerate and run the evals: `python3 scripts/build_evals.py`, then `claude plugin eval plugins/cyberriskguardian --allow-tools Bash Write` and `claude plugin eval plugins/crg-governance-textbook`.
- Regression check: `python3 plugins/cyberriskguardian/skills/crg-quantify/scripts/crg_calc.py examples/medibec/source/medibec30.json` must reproduce total estimated 84,305 and residual 41,023.
- Cross-engine conformance (X-1 / X-2): `python3 plugins/cyberriskguardian/skills/crg-quantify/scripts/test_cross_engine.py` must pass (MediBec 84,305 / 41,023 and CanaRetail 27,526 / 16,049; add `--engine` / `--mcp` to test the Threat Live-feed and Pro engines from their checkouts).
- Never change a CRG formula silently; record methodology decisions in `master-task-spec.md` ("Resolved methodology choices").
