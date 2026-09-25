# Publier CyberRiskGuardian

Deux voies complémentaires :
- **Dépôt GitHub public** : n'importe qui peut installer l'agent dès que le dépôt est en ligne.
- **Répertoire communautaire d'Anthropic** : l'agent apparaît dans les catalogues de Claude Code et Cowork, après révision.

## 1. Dépôt GitHub public

1. Sur GitHub, créez un dépôt **public** vide nommé `cyberriskguardian-agent`. N'y ajoutez ni README ni licence : ils sont déjà dans le paquet.
2. Dans le dossier décompressé (il contient déjà un dépôt git avec un premier commit) :
   ```bash
   git remote add origin https://github.com/OWNER/cyberriskguardian-agent.git
   git branch -M main
   git push -u origin main
   ```
3. Remplacez `OWNER` par votre compte dans `README.md` (section *Public install*), puis faites un commit et un push.
4. Facultatif : dans les paramètres du dépôt, ajoutez une description et les sujets `claude-code`, `claude-plugin`, `cybersecurity`, `risk-assessment`, et activez les *private vulnerability reports* (voir `SECURITY.md`).
5. Pour publier une version, créez une étiquette et poussez-la :
   ```bash
   git tag v1.1.0
   git push --tags
   ```

Installation par n'importe qui :
```bash
claude plugin marketplace add OWNER/cyberriskguardian-agent
claude plugin install cyberriskguardian@cyberriskguardian-marketplace
```

## 2. Répertoire communautaire Anthropic (Claude Code et Cowork)

- Soumettez le plugin avec le formulaire https://clau.de/plugin-directory-submission, en indiquant l'URL du dépôt public et le chemin `plugins/cyberriskguardian`.
- Les pull requests ouvertes directement sur `anthropics/claude-plugins-community` sont fermées automatiquement.
- Chaque plugin passe une analyse de sécurité automatisée et une révision avant d'être listé.
- Une fois accepté, le plugin s'installe ainsi :
  ```bash
  claude plugin marketplace add anthropics/claude-plugins-community
  claude plugin install cyberriskguardian@claude-community
  ```
  Dans Cowork, il s'installe depuis claude.com/plugins.

## Avant chaque publication
```bash
claude plugin validate plugins/cyberriskguardian/.claude-plugin/plugin.json
claude plugin validate .
python3 plugins/cyberriskguardian/skills/cyber-risk-assessment/scripts/crg_calc.py examples/medibec/source/medibec30.json   # doit afficher 84,491 / 41,104
bash scripts/sync-project-config.sh
```
Incrémentez la version (semver) dans `plugins/cyberriskguardian/.claude-plugin/plugin.json` et dans `.claude-plugin/marketplace.json`.

## Rappels
- **Licence** : CC BY-NC 4.0. L'usage commercial n'est pas permis sans votre accord. Les notes de licence se trouvent dans `LICENSE` et `NOTICE.md`.
- **Contenu public** : le paquet contient votre manuel, le cas MediBec et ses livrables. Tout ce qui est poussé sur un dépôt public est visible et copiable par tous. Retirez ce que vous ne voulez pas diffuser avant le premier push, car l'historique git le conserverait.
