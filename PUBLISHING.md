# Publier CyberRiskGuardian

> **Version 2.1** : le dépôt est un un marketplace de plusieurs plugiciels (Core et manuel). Pour publier une nouvelle version, suivez la section « Maintain and release » du README. La version 1.1.1 reste disponible sur la branche `release/1.x`.

Deux voies complémentaires :
- **Dépôt GitHub public** : n'importe qui peut installer l'agent dès que le dépôt est en ligne.
- **Répertoire communautaire d'Anthropic** : l'agent apparaît dans les catalogues de Claude Code et Cowork, après révision.

## 1. Dépôt GitHub public

1. Sur GitHub, créez un dépôt **public** vide nommé `CyberRiskGuardian` (compte `ITriskMgr`). N'y ajoutez ni README ni licence : ils sont déjà dans le paquet.
2. Dans le dossier décompressé (il contient déjà un dépôt git avec un premier commit) :
   ```bash
   git remote add origin https://github.com/ITriskMgr/CyberRiskGuardian.git
   git branch -M main
   git push -u origin main
   ```
3. Facultatif : dans les paramètres du dépôt, ajoutez une description et les sujets `claude-code`, `claude-plugin`, `cybersecurity`, `risk-assessment`, et activez les *private vulnerability reports* (voir `SECURITY.md`).
4. Pour publier une version, créez une étiquette et poussez-la :
   ```bash
   git tag v2.1.0
   git push --tags
   ```

Installation par n'importe qui :
```bash
claude plugin marketplace add ITriskMgr/CyberRiskGuardian
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
claude plugin validate plugins/crg-governance-textbook/.claude-plugin/plugin.json
python3 plugins/cyberriskguardian/skills/crg-quantify/scripts/crg_calc.py examples/medibec/source/medibec30.json   # doit afficher 84,305 / 41,023
bash scripts/sync-project-config.sh
python3 scripts/build_evals.py
```
Incrémentez la version (semver) dans le `plugin.json` de chaque plugiciel modifié et dans `.claude-plugin/marketplace.json`.

Les 21 jeux de diapositives des chapitres du manuel sont publiés comme fichiers joints de la version GitHub `education-slides-v2.51` (Releases), et non dans l'historique git.

## Rappels
- **Licence** : ce dépôt est l'Édition communautaire, sous CC BY-NC 4.0. L'usage commercial exige une édition professionnelle (Licence commerciale CyberRiskGuardian); les collèges, cégeps et universités qui veulent l'édition éducative ont besoin de la Licence éducative. Ne publiez jamais de contenu des éditions éducative ou professionnelles (dont l'Édition Éducateur et le guide de ressources de l'enseignant) dans ce dépôt public. Renseignements : https://cyberriskguardian.ca · sales@cyberriskguardian.ca. Les notes de licence se trouvent dans `LICENSE` et `NOTICE.md`.
- **Contenu public** : le paquet contient votre manuel, le cas MediBec et ses livrables. Tout ce qui est poussé sur un dépôt public est visible et copiable par tous. Retirez ce que vous ne voulez pas diffuser avant le premier push, car l'historique git le conserverait.
