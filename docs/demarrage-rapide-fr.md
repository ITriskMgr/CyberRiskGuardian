# Démarrage rapide CyberRiskGuardian

*Version 2.0 · English version: `quickstart-en.md`*

CyberRiskGuardian vous aide à mener avec Claude une évaluation structurée des risques de cybersécurité : du contexte de votre organisation jusqu'à un ensemble de scénarios de risque classés et aux décisions que la direction doit prendre. C'est un outil d'aide à la décision. Vous validez ses estimations et la direction prend les décisions.

## 1. Installer (5 minutes)

**Claude Code**

```
/plugin install cyberriskguardian --marketplace ITriskMgr/CyberRiskGuardian
/plugin install crg-governance-textbook --marketplace ITriskMgr/CyberRiskGuardian
```

Installez ensuite une fois les dépendances du calculateur : `pip install cvss "mcp>=1.2"`. Redémarrez Claude Code et approuvez le serveur `crg-calculator` lorsqu'il vous le demande.

**Claude Cowork** : dans le processus d'installation des plugiciels, téléversez `cyberriskguardian.plugin` et `crg-governance-textbook.plugin` à partir de la version publiée sur GitHub.

**claude.ai** : les compétences y fonctionnent aussi; les calculs utilisent alors le script fourni plutôt que le serveur de calcul local.

## 2. Votre première évaluation (20 minutes)

1. Rassemblez ce que vous avez sur l'organisation : un profil, une vue d'ensemble des TI, une liste de fournisseurs, des notes d'audit ou d'incident. Aucun document? Dites-le, et Claude vous interrogera, cinq questions à la fois.
2. Tapez `/crg-quick-scan` suivi d'une courte description, ou joignez les documents. Exemple :
   `/crg-quick-scan Réseau de cliniques régionales, 10 sites, dossiers médicaux électroniques, accès à distance pour les cliniciens.`
3. Répondez aux questions. Claude estime l'appétit pour le risque, construit 20 scénarios candidats, retient les 10 plus importants et calcule le risque estimé et le risque toléré avec le calculateur.
4. Lisez le tableau des résultats : chaque scénario est **au-dessus**, **à peu près au niveau** ou **en dessous** de la tolérance avant traitement.

Pour la démarche Core complète avec un résumé écrit pour la direction, utilisez `/crg-assess`.

## 3. Lire les marqueurs

Chaque livrable porte trois marqueurs. Ils indiquent ce qu'il reste à faire :

| Marqueur | Signification | Que faire |
|---|---|---|
| **Estimation analytique — validation requise** | Une valeur estimée par Claude | La confirmer avec les personnes qui connaissent le système ou le processus |
| **Lacune d'information — validation organisationnelle requise** | Un fait que personne n'a fourni | Trouver le document ou demander au responsable |
| **Décision de la direction requise** | Un choix qui revient à la direction | L'inscrire à l'ordre du jour |

Une vérification intégrée vous avertit lorsqu'un livrable n'a pas un marqueur requis.

## 4. Commandes utiles

| Commande | Usage |
|---|---|
| `/crg-quick-scan` | 20 candidats, 10 scénarios, un tableau |
| `/crg-assess` | Démarche Core complète et résumé pour la direction |
| `/select-scenarios` | Tri et sélection des scénarios seulement (sections A à E) |
| `/crg-calc` | Indicateurs d'un scénario à partir de ses paramètres |
| `/qc-review` | Revue indépendante d'un rapport avant de l'envoyer |

Posez vos questions de gouvernance en langage courant, par exemple : « Quelle est la différence entre l'appétit pour le risque et la tolérance au risque? » Le plugiciel du manuel répond en citant le chapitre.

## 5. Aller plus loin : CRG Pro

CRG Pro ajoute le portefeuille de traitement chiffré, l'analyse de sensibilité, le budget de cybersécurité aligné sur l'appétit pour le risque, la conception des indicateurs de risque clés, le classeur Excel à formules actives et le rapport complet pour le conseil en Word. Il est offert sous la Licence commerciale CyberRiskGuardian ou, pour les collèges et universités, sous la Licence éducative. Détails : www.leger.ca.

## Licence

L'Édition communautaire est gratuite pour les étudiants, les organismes sans but lucratif et tout usage non commercial (CC BY-NC 4.0). L'utiliser pour fournir des services rémunérés, ou au sein d'une entreprise ou d'un organisme public, exige une licence commerciale.
