# Security policy

## Scope
- **Local-only execution:** the plugin runs locally.
  - The `crg-calculator` MCP server is a local Python process (stdio) that makes no network calls.
  - The scripts only read and write local files.
- **No credentials:** no credentials, tokens or external services are required.
- **Hooks:** the only hook prints a static text file at session start.
- **Permissions:** `.claude/settings.json` sets them.
  - **Allowed:** read-only access plus the bundled scripts.
  - **Ask first:** web access and package installation.
  - **Denied:** `.env`, key files and local settings.

## Reporting a vulnerability
Open a private security advisory on the GitHub repository, or contact the maintainer through the repository. Please do not open a public issue for security problems.

## Responsible use
The assessments the agent produces are decision support. Do not submit confidential, personal or regulated organizational information to any external AI service without authorization under your organization's policies.
