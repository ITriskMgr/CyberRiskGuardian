# Ensemble pédagogique CyberRiskGuardian 2.1.2

*English version: `README.md`*

Matériel pédagogique pour les enseignants et les étudiants qui utilisent CyberRiskGuardian dans un cours de gestion des risques en cybersécurité, de gouvernance, d'audit des TI ou de gestion des technologies d'affaires. Chaque document existe en français (`fr/`) et en anglais (`en/`), sous forme de sources Markdown et de fichiers Word et PDF finalisés.

## Contenu

| Document | Pour | Ce qu'il vous apporte |
|---|---|---|
| `01_Guide_enseignant` — Guide de l'enseignant | Enseignants, auxiliaires d'enseignement | La méthode en une page, les résultats d'apprentissage, les parcours d'outil, quatre modèles d'intégration (d'un seul atelier à une session complète), la liste de vérification de configuration, les notes d'animation des laboratoires, les concepts difficiles à enseigner, la politique sur l'IA générative, le dépannage |
| `02_Cahier_laboratoires` — Cahier de laboratoires de l'étudiant | Étudiants | Treize laboratoires pas à pas sur le cas pédagogique MediBec, un cas d'exercice autonome (CanaRetail), aide-mémoire des formules, modèles, réponses d'autovérification, glossaire |
| `03_Trousse_evaluation` — Trousse d'évaluation | Étudiants et enseignants | Échelle de notation de 0 à 10, énoncés et grilles d'évaluation pour l'évaluation des risques (A1), la politique GIA (A2) et le plan d'audit (A3), quatre formats plus courts, autovérifications, formulaire d'évaluation par les pairs, déclaration d'utilisation de l'IA |
| `04_Bibliotheque_cas` — Bibliothèque de cas | Étudiants et enseignants | Cinq études de cas fictives et la façon d'en choisir une; les versions étudiantes des cas se trouvent dans `cases/` |
| `slides/CRG_Intro_Deck_FR` | Enseignants | Un cours d'introduction à la méthode CRG, avec des captures d'écran de l'application de bureau 1.9.1 (PowerPoint; les écrans de la 1.9.2 ne diffèrent que par le libellé) |
| `templates/CyberRiskGuardian_Excel_Model_Blank_v2-6.xlsx` | Étudiants et enseignants | Le modèle Excel CyberRiskGuardian vierge (Analyse, S1–S10, M1–M10) utilisé pour le travail d'évaluation des risques — gratuit. Saisies validées, score et vecteur CVSS, règle du produit pour les réductions combinées, calibrage du budget avec les dépenses actuelles, consignation du transfert; suit la procédure en 8 étapes du chapitre 6 du manuel |
| Présentations des chapitres (fichiers joints aux versions) | Enseignants | 21 présentations PowerPoint du manuel *Introduction to Cybersecurity Governance* (3e éd., v2.51), une par chapitre, en anglais — gratuites; voir ci-dessous |

```
education/
├── README.md, LISEZMOI.md
├── fr/            Sources Markdown, .docx, .pdf
├── en/            Markdown sources, .docx, .pdf
├── slides/        CRG_Intro_Deck_EN.pptx, CRG_Intro_Deck_FR.pptx (présentations des chapitres : fichiers joints aux versions)
├── templates/     Le modèle Excel CyberRiskGuardian vierge (.xlsx)
└── cases/         Versions étudiantes des cinq cas, v2.0c (.docx)
```

## Nouveautés de la version 2.1.2 (novembre 2026)

Pour **CyberRiskGuardian Desktop 1.9.2**, à la suite de la campagne de tests du 8 novembre 2026. Changements de contenu seulement; aucune formule ni valeur de cas n'a changé.

- **Laboratoire 12, étape facultative [B]** : rédiger la politique de GIA avec le **Rédacteur de politiques de sécurité** de l'application de bureau à partir du cas seulement, puis comparer le brouillon généré à ses propres énoncés et expliquer ses espaces réservés `[DÉCISION : …]` (anglais et français).
- **Guide de l'enseignant** : le tableau de dépannage marque le problème du MediBec en double *corrigé dans Desktop 1.9.2* et ajoute un tableau *Corrigé dans Desktop 1.9.2* (extraction par règles `[object Object]` au laboratoire 1, promotion depuis le tri au laboratoire 3, score CVSS et vecteur au laboratoire 5, QC-02 et approbateurs aux laboratoires 10 et 11, deux fenêtres, espaces de travail en double) pour les classes encore en 1.9.1.
- **Laboratoire 0** : le bon chemin vers `medibec30.json` pour le volet Claude; **laboratoire 8** : où se trouve le bouton *Facteurs déterminants* (sur chaque ligne Monte-Carlo).
- **Documents français** : la terminologie de Desktop 1.9.2 — « ratio risque résiduel / risque toléré », « probabilité », « éléments probants » — et l'apostrophe typographique dans les libellés de l'interface.
- **Modèle Excel vierge** : un avis bilingue sur les CIS Controls v8.1 dans les feuilles M1 à M10 et Analyse (identifiants et titres seulement, CC BY-NC-ND 4.0); formules inchangées.
- `scripts/build_education.py` : port LibreOffice configurable et dépendances documentées.

L'ensemble complet est aussi publié en un seul ZIP dans la version [**education-v2.1.2**](https://github.com/ITriskMgr/CyberRiskGuardian/releases/tag/education-v2.1.2) (`CyberRiskGuardian_Education_Package_2.1.2.zip` et son `.sha256`).

## Nouveautés de la version 2.1.1 (novembre 2026)

Pour **CyberRiskGuardian Desktop 1.9.1**, le plugiciel Core 2.2.0 et le manuel **v2.51b** (en anglais et en français). Le numéro de version suit celui du plugiciel du manuel (2.1.1) qui porte la v2.51b.

- **Laboratoire 13 — Posture des identités, des accès et de la configuration** : les modules facultatifs *Gestion des identités et des accès (GIA/GAP)* et *Gestion de la configuration* de l'application de bureau, appliqués aux registres d'exemple de MediBec — recalculer un indicateur, transformer un constat en élément probant et en proposition de paramètre, et reporter la posture dans la politique GIA du travail A2.
- **Cas d'exercice CanaRetail** : la solution du module D du cas est fournie dans l'application de bureau (anglais et français, totaux 27 526 / 16 049); CanaRetail devient donc un cas d'exercice autonome pour les laboratoires 4 à 7 et n'est plus recommandé pour le travail noté A1.
- **L'interface 1.9** : chemins de menu, carte de prochaine étape, niveaux de détail, bouton Aide de la barre supérieure, aide reliée aux chapitres et pied de page épinglé (*✓ Moteur vérifié*); le laboratoire 0 et la liste de préparation renvoient aux guides de première installation et de première configuration de chaque version.
- **Consultant virtuel** : l'enseignant décide pour chaque travail s'il peut être utilisé; les énoncés A1 à A3 prévoient une ligne à cet effet et la déclaration d'utilisation de l'IA le couvre.
- **Manuel v2.51b en anglais et en français** : les tableaux « Utiliser le manuel » suivent la v2.51b et ajoutent les chapitres 10 et 14 pour le laboratoire 13; les documents français citent l'édition française (*Introduction à la gouvernance de la cybersécurité*) et en reprennent les termes (*appétence pour le risque*, *joyaux de la couronne*, *registre des éléments probants*, *ratio risque résiduel / risque toléré*).
- **Présentation d'introduction** (anglais et français) : texte corrigé et nouvelles diapositives montrant l'application de bureau 1.9.1.

L'ensemble complet est aussi publié en un seul ZIP dans la version [**education-v2.1.1**](https://github.com/ITriskMgr/CyberRiskGuardian/releases/tag/education-v2.1.1).

## Présentations des chapitres (manuel v2.51)

Les 21 présentations des chapitres sont trop volumineuses pour l'historique du dépôt (environ 105 Mo); elles sont publiées comme fichiers joints d'une version GitHub : [**education-slides-v2.51**](https://github.com/ITriskMgr/CyberRiskGuardian/releases/tag/education-slides-v2.51). Elles sont gratuites sous licence CC BY-NC 4.0, en anglais.

| Ch. | Chapitre | Fichier |
|---|---|---|
| 1 | Basic Concepts of Cybersecurity | `Ch01_Basic_Concepts_of_Cybersecurity_v2-51.pptx` |
| 2 | Cybersecurity Governance | `Ch02_Cybersecurity_Governance_v2-51.pptx` |
| 3 | Cybersecurity Compliance | `Ch03_Cybersecurity_Compliance_v2-51.pptx` |
| 4 | Cybersecurity Risk Management | `Ch04_Cybersecurity_Risk_Management_v2-51.pptx` |
| 5 | Creating Cybersecurity Risk Scenarios | `Ch05_Creating_Cybersecurity_Risk_Scenarios_v2-51.pptx` |
| 6 | Performing a Cybersecurity Risk Assessment | `Ch06_Performing_a_Cybersecurity_Risk_Assessment_v2-51.pptx` |
| 7 | Cybersecurity Operations | `Ch07_Cybersecurity_Operations_v2-51.pptx` |
| 8 | Cybersecurity Detection, Orchestration, Vulnerabilities, Response, and Testing | `Ch08_Cybersecurity_Detection_Orchestration_Vulnerabilities_Response_and_Testing_v2-51.pptx` |
| 9 | Cybersecurity Control Frameworks | `Ch09_Cybersecurity_Control_Frameworks_v2-51.pptx` |
| 10 | Identity and Access Management | `Ch10_Identity_and_Access_Management_v2-51.pptx` |
| 11 | OT, IoT, and Cyber-Physical Security | `Ch11_OT_IoT_and_Cyber_Physical_Security_v2-51.pptx` |
| 12 | Incident Management, Business Continuity, and Resilience | `Ch12_Incident_Management_Business_Continuity_and_Resilience_v2-51.pptx` |
| 13 | Security Architecture and Security Engineering | `Ch13_Security_Architecture_and_Security_Engineering_v2-51.pptx` |
| 14 | Security Baselines and Configuration Governance | `Ch14_Security_Baselines_and_Configuration_Governance_v2-51.pptx` |
| 15 | Cloud Cybersecurity | `Ch15_Cloud_Cybersecurity_v2-51.pptx` |
| 16 | Data Protection and Privacy | `Ch16_Data_Protection_and_Privacy_v2-51.pptx` |
| 17 | Cybersecurity Economics and Investment Governance | `Ch17_Cybersecurity_Economics_and_Investment_Governance_v2-51.pptx` |
| 18 | AI for Cybersecurity and the Cybersecurity of AI | `Ch18_AI_for_Cybersecurity_and_the_Cybersecurity_of_AI_v2-51.pptx` |
| 19 | Third-Party and Supply-Chain Cybersecurity Governance | `Ch19_Third_Party_and_Supply_Chain_Cybersecurity_Governance_v2-51.pptx` |
| 20 | Cryptography and Key Management in Managerial Terms | `Ch20_Cryptography_and_Key_Management_in_Managerial_Terms_v2-51.pptx` |
| 21 | People, Organizational Design, and Accountability Mechanisms | `Ch21_People_Organizational_Design_and_Accountability_Mechanisms_v2-51.pptx` |

Les présentations ont été produites à partir de la v2.51. Le texte est maintenant la **v2.51b** (la v2.51 corrigée, dans le plugiciel `crg-governance-textbook` 2.1.1) : totaux MediBec 84 305 / 41 023, réductions d'un ensemble de mesures combinées selon la règle du produit 1 − ∏(1 − rᵢ), au moins vingt scénarios candidats (habituellement de 25 à 40) et triangle du risque formé de la menace, de la vulnérabilité et de l'exposition. En cas d'écart, le texte v2.51b prévaut.

La présentation d'introduction à CRG demeure dans `slides/` (anglais et français).

## Édition Éducateur

Les corrigés, une banque de questions de quiz notées, le calibrage des grilles d'évaluation et les commentaires de rétroaction, les notes pédagogiques et les errata des cas, les annexes de l'enseignant des cas le cas de l'examen final et son plan ainsi que le **guide de ressources de l'enseignant** du manuel (justification des réponses aux questions à choix multiples et indications de correction des questions à développement) forment l'**Édition Éducateur**. Elle ne se trouve pas dans ce dépôt : elle est fournie avec l'**édition éducative** aux établissements qui détiennent la licence éducative CyberRiskGuardian, afin que les réponses demeurent utiles d'une session à l'autre.

## Qui peut l'utiliser

- Les **étudiants** utilisent CyberRiskGuardian et ce matériel gratuitement pour leurs propres travaux de cours, y compris les travaux notés (Édition communautaire).
- Les **enseignants** peuvent adapter et réutiliser le matériel à des fins d'enseignement non commerciales, avec attribution.
- Les **collèges, cégeps et universités, en tant qu'établissements,** doivent détenir l'**édition éducative** (licence éducative CyberRiskGuardian) pour offrir un cours fondé sur CRG : prix considérablement réduit; installation, personnalisation, l'application de bureau courante, CRG Pro Quantitative et l'Édition Éducateur inclus.
- Les **organismes sans but lucratif** utilisent CyberRiskGuardian gratuitement pour évaluer leurs propres risques.

Demandes de licence : sales@cyberriskguardian.ca · cyberriskguardian.ca

## Pour commencer

1. Enseignants : lisez `01_Guide_enseignant` (Guide de l'enseignant), sections 2 (la méthode en une page) et 5 (modèles d'intégration).
2. Choisissez un parcours d'outil : plugiciel Claude, application de bureau ou modèle Excel.
3. Faites vous-même le laboratoire 0 et confirmez la vérification du moteur **84 305 / 41 023** (application de bureau : le pied de page affiche *✓ Moteur vérifié*; dans Excel : l'exemple du chapitre 6, 3 554,46 / 1 937,18).
4. Utilisez MediBec pour les laboratoires et CanaRetail pour l'exercice autonome; attribuez Chez Pierre, ACRB ou CNIC pour les travaux notés.
5. Volet Claude (plugiciel Core 2.2.0) : `/crg-quick-scan` produit au moins vingt candidats, dix scénarios et leurs verdicts avant traitement dans un seul tableau; `/label-check <fichier>` vérifie qu'un livrable porte les trois mentions — **Estimation analytique — validation requise**, **Lacune d’information — validation organisationnelle requise**, **Décision de la direction requise** — et signale la formulation retirée.

Responsables du dépôt : les fichiers .docx et .pdf sont produits à partir des sources Markdown avec `python3 scripts/build_education.py` (exige pandoc, python-docx et LibreOffice avec son pont Python UNO; `--port` ou `CRG_LO_PORT` fixe le port de LibreOffice, sinon un port libre est utilisé; voir la documentation du script). L'avis CIS du modèle Excel vierge est écrit par `scripts/add_cis_notice.py` (Core).

## Licence

© 2026 Marc-André Léger. Sous licence Creative Commons Attribution - Pas d'Utilisation Commerciale 4.0 International (CC BY-NC 4.0). Les cas sont fictifs; toute ressemblance avec des organisations réelles est fortuite. CyberRiskGuardian est un outil d'aide à la décision : ses estimations sont des hypothèses analytiques initiales, et les décisions relatives au risque demeurent la responsabilité de la direction.

Attribution : *« Ensemble pédagogique CyberRiskGuardian 2.1.2, © 2026 Marc-André Léger, CC BY-NC 4.0. »*
