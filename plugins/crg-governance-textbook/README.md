# CRG Governance Textbook

*Introduction to Cybersecurity Governance for Business Technology Management*, 3rd edition, version 2.1g (August 2026), by Marc-André Léger, as a Claude skill. Use it to explain governance, risk, compliance and resilience concepts with chapter citations, and as the reference text behind the CyberRiskGuardian method.

## Install

```
/plugin install crg-governance-textbook --marketplace ITriskMgr/CyberRiskGuardian
```

On older Claude Code versions:

```
/plugin marketplace add ITriskMgr/CyberRiskGuardian
/plugin install crg-governance-textbook@cyberriskguardian-marketplace
```

## Components

| Type | Name | Purpose |
|---|---|---|
| Skill | `cybersecurity-governance` | Chapter index and one reference file per chapter; answers cite the chapter |
| Evals | `evals/` | Concept questions checked for chapter citations |

## What the plugin runs and sends

Nothing runs. The plugin contains text only: no hooks, scripts, MCP servers or network access.

## Licence

© 2026 Marc-André Léger. Licensed under CC BY-NC 4.0; commercial use requires the CyberRiskGuardian Commercial Licence. Third-party standards and frameworks quoted in the book remain under their owners' terms. See `LICENSE` and `NOTICE.md`.
