---
title: "Cahier de laboratoires de l'étudiant CyberRiskGuardian"
subtitle: "Treize laboratoires sur le cas pédagogique MediBec — Ensemble pédagogique 2.1.1"
author: "Marc-André Léger, Ph. D."
date: "novembre 2026"
toc-title: "Table des matières"
---

# Comment utiliser ce cahier

Ces laboratoires vous apprennent à réaliser, étape par étape, une évaluation des risques de cybersécurité fondée sur des scénarios de risque avec CyberRiskGuardian (CRG), à partir d'une seule organisation fictive : **MediBec**, un centre médical privé comptant dix cliniques au Québec. L'évaluation complète de MediBec est publiée avec CyberRiskGuardian; vous pouvez donc comparer votre travail avec celle-ci. Votre travail noté portera sur un cas *différent*.

Chaque laboratoire présente ses objectifs, sa durée, le matériel requis, les étapes, le livrable et trois questions pour vérifier votre compréhension. Les réponses à ces questions se trouvent à l'annexe D. Travaillez dans le parcours d'outils choisi par votre enseignant :

- **Parcours A — Claude** : le plugiciel ou le projet CyberRiskGuardian dans Claude.
- **Parcours B — Application de bureau** : l'application de bureau CyberRiskGuardian (1.9.1 ou plus récente) dans votre navigateur.
- **Parcours C — Excel** : le classeur CyberRiskGuardian (feuilles Analyse, S1–S10, M1–M10).

Lorsqu'une étape diffère selon le parcours, elle est marquée **[A]**, **[B]** ou **[C]**. Après le laboratoire 7, vous pouvez vous exercer sur un deuxième cas, CanaRetail, à l'aide de sa solution publiée (voir *Cas d'exercice — CanaRetail*).

## Cinq règles pour chaque laboratoire

1. **Utilisez le cas comme source des faits.** Si vous avez besoin d'un fait que le cas ne fournit pas, indiquez-le, formulez une hypothèse raisonnable et expliquez ce qui changerait si elle était fausse.
2. **Étiquetez ce que vous estimez.** Chaque nombre estimé par vous ou par un outil d'IA porte la mention **Estimation analytique — validation requise**. Chaque fait manquant porte la mention **Lacune d'information — validation organisationnelle requise**. Chaque choix qui relève de la direction porte la mention **Décision de la direction requise**.
3. **Laissez le moteur calculer.** Les résultats finaux proviennent du calculateur vérifié (application de bureau, `crg_calc.py` ou classeur recalculé), jamais de l'arithmétique d'une IA.
4. **Vérifiez les résultats de l'IA.** Les outils d'IA inventent des normes, des identifiants de contrôles, des lois, des statistiques et des vecteurs CVSS qui semblent exacts. Vérifiez chacun d'eux auprès d'une source primaire avant de l'utiliser.
5. **Protégez les données.** Ne collez pas de documents confidentiels dans un service d'IA à moins d'y être autorisé. MediBec est fictive et peut être utilisée sans risque.

# Laboratoire 0 — Prise en main

**Objectifs :** installer ou ouvrir votre parcours; confirmer que le moteur de calcul est exact; apprendre les trois mentions.
**Durée :** 30 minutes.

## Étapes

1. Installez ou ouvrez votre parcours.
   - **[A]** Dans Claude Code : `/plugin install cyberriskguardian --marketplace ITriskMgr/CyberRiskGuardian`, puis `/plugin install crg-governance-textbook --marketplace ITriskMgr/CyberRiskGuardian`. Exécutez une fois `pip install cvss "mcp>=1.2"`, redémarrez et approuvez le serveur `crg-calculator`.
   - **[B]** Suivez le guide **Première installation** de la version (un PDF sur la page de la version, et `FIRST-INSTALL.md` dans le ZIP). En bref : décompressez l'application de bureau, puis double-cliquez sur `start.command` (macOS, Linux) ou `start.bat` (Windows). Laissez la fenêtre du terminal ouverte. L'application s'ouvre à `http://localhost:8099`. Utilisez toujours cette même adresse, dans une seule fenêtre : votre travail est conservé par votre navigateur pour cette adresse exacte. Sur la page Processus, la carte *Configurer l'application pour votre pratique* propose **Répondre aux questions de configuration** : une question sur votre expérience règle le niveau de détail (répondez *Nouveau ou nouvelle — montrez-moi pourquoi chaque écran compte*, ce qui donne **Détaillé**), et les questions sur les modules peuvent garder leurs valeurs par défaut (votre enseignant vous dira si un laboratoire exige un module). Vous pouvez aussi passer cette étape et changer le niveau plus tard dans Paramètres → Langue et détail. Le guide **Première configuration** explique chaque écran d'une première évaluation.
   - **[C]** Ouvrez le modèle vierge `templates/CyberRiskGuardian_Excel_Model_Blank_v2-6.xlsx` (ou la copie que votre enseignant vous a remise) et enregistrez-en une copie au nom de votre équipe. Il s'ouvre en mode de calcul manuel : appuyez sur **F9** (Formules → Calculer maintenant) avant de lire quelque nombre que ce soit.
2. Vérifiez le moteur.
   - **[A]** Exécutez `python3 crg_calc.py medibec30.json` à partir du dossier `crg-quantify/scripts` du plugiciel sur l'exemple MediBec. Il doit afficher **risque estimé total 84 305** et **résiduel 41 023**.
   - **[B]** Le pied de page épinglé du menu latéral doit afficher **✓ Moteur vérifié**. Pointez dessus (ou ouvrez Aide → À propos et versions) pour voir les totaux **84 305 / 41 023**.
   - **[C]** Saisissez l'exemple du chapitre 6 du manuel dans la feuille S1 et à la ligne 3 de la feuille Analyse : Pb(A) 0,70, Pb(ψ,A) 0,60, δe 0,60, δm 0,80, θ 0,50, μ(E) 0,65, CVSS 9,3 en G3 avec le vecteur `CVSS:4.0/AV:L/AC:L/AT:N/PR:N/UI:P/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H` en H3, appétence 0,35 en E14; dans M1, une ligne « Other / autre » retenue avec un coût de 62 000, une réduction d'impact de 0,70 et une réduction de probabilité de 0,65. Après « Calculer maintenant », vous devez lire un risque estimé de 3 554,46, un risque toléré de 1 777,23, un risque résiduel de 1 937,18 et un ratio de 1,09. Effacez ensuite ces données (ou repartez du modèle vierge).
3. **[B]** Ouvrez l'espace de travail MediBec, puis Démarrer → **Processus**. Lisez les dix-neuf étapes d'une évaluation complète. La **carte de prochaine étape**, en haut, nomme toujours la prochaine étape, sa liste de vérification et l'écran qui s'en charge. Sur n'importe quel écran, le bouton **Aide** de la barre supérieure ouvre *Aide sur cet écran*, avec un lien *Pour en savoir plus : chapitre …* vers le passage du manuel correspondant.
4. Relisez les trois mentions dans les « Cinq règles » ci-dessus. Trouvez un exemple de chacune dans les livrables d'exemple ou l'espace de travail MediBec.

**Livrable :** une capture d'écran du moteur vérifié (ou du résultat de votre calculateur), avec votre nom et la date.

## Vérifiez votre compréhension


0.1. Pourquoi ne devriez-vous vous fier à aucun nombre tant que le moteur n'a pas été vérifié?

0.2. Dans l'application de bureau, pourquoi vos espaces de travail pourraient-ils sembler disparaître si vous ouvrez l'application à `127.0.0.1:8099` au lieu de `localhost:8099`?

0.3. Quelle mention utiliseriez-vous pour « Le fournisseur du DME de MediBec n'est pas nommé dans le cas »?

# Laboratoire 1 — Contexte organisationnel et joyaux de la couronne

**Objectifs :** décrire, à partir du cas, la mission, les processus critiques et les dépendances de MediBec; déterminer ses joyaux de la couronne (actifs critiques); distinguer les faits des hypothèses.
**Durée :** 60 minutes. **Intrant :** étude de cas MediBec, sections 3 à 5.

## Mise en contexte

Une évaluation des risques part de l'organisation, et non d'une liste de menaces. Vous devez savoir ce que fait MediBec, quels processus importent le plus, de quels actifs ces processus dépendent et ce que vous ne savez *pas* encore.

## Étapes

1. Lisez les sections 3 à 5 du cas. Dressez la liste des faits sur lesquels vous vous appuierez, chacun avec son numéro de section. Exemple : *« 10 cliniques à Montréal, Québec, Gatineau et Saguenay (s. 3). »*
2. Remplissez le tableau de contexte ci-dessous. Limitez chaque cellule à 40 mots.

| Élément de contexte | MediBec (avec source) |
|---|---|
| Mission | |
| Secteur et taille | |
| Processus d'affaires critiques | |
| Systèmes et applications critiques | |
| Infrastructure et infonuagique | |
| Informations sensibles | |
| Fournisseurs et partenaires | |
| Contexte juridique et réglementaire (tel que le cas l'énonce) | |
| Exigences de disponibilité | |
| Contrôles existants | |
| Historique des incidents | |
| Capacité en cybersécurité | |

3. Retracez une chaîne : **Objectif → Processus → Actif principal → Actifs de soutien → Dépendances → Conséquence en cas de perte.** Exemple de départ : *Continuité des soins → documentation clinique → DME → identité, centre de données de Montréal, réseau étendu (WAN) → …*
4. Déterminez **cinq ou six joyaux de la couronne** : les actifs ou services dont la perte nuirait le plus à la mission. Pour chacun, indiquez un responsable métier, l'importance de la confidentialité, de l'intégrité et de la disponibilité (Élevée / Très élevée) et ses principales dépendances.
5. Énumérez au moins trois **lacunes d'information** (faits dont vous auriez besoin mais que le cas ne fournit pas) et deux **hypothèses** que vous formulez, chacune avec sa mention.
   - **[A]** Demandez à l'assistant d'effectuer la collecte du contexte à partir du document du cas. Comparez ses joyaux de la couronne avec les vôtres; ne conservez que ce que le cas appuie.
   - **[B]** Organisation → Espaces de travail → *Nouvel espace de travail*, *Partir de* *Évaluation vierge*, type *Cas d'affaires pour la classe*; déposez le document du cas, cochez *Extraire le profil et l'appétence de ces documents après la création* et *Puis extraire les joyaux de la couronne*, choisissez **Règles seulement (sans IA, hors ligne)** et cliquez sur *Créer l'espace de travail*. Examinez chaque valeur proposée et son étiquette (FACT, INFERENCE, ASSUMPTION, EXTERNAL, UNKNOWN). Résolvez à partir du cas les valeurs UNKNOWN que vous pouvez.

**Livrable :** le tableau de contexte, une chaîne de dépendances, la liste des joyaux de la couronne et la liste des lacunes et hypothèses (deux pages au maximum).

## Vérifiez votre compréhension


1.1. Pourquoi « l'identité et l'accès à distance » constitue-t-il un joyau de la couronne même s'il ne contient pas lui-même de données sur les patients?

1.2. Le cas indique que MediBec a subi « de multiples tentatives d'attaque infructueuses depuis 2020 ». S'agit-il d'un fait, d'une inférence ou d'une hypothèse? Que ne pouvez-vous *pas* en conclure?

1.3. Nommez une dépendance qui constitue un point de défaillance unique pour plusieurs joyaux de la couronne.

# Laboratoire 2 — Appétence pour le risque

**Objectifs :** estimer et justifier une appétence pour le risque de cybersécurité sur une échelle de 0 à 1; distinguer l'appétence, la tolérance et le score de risque.
**Durée :** 45 minutes. **Intrant :** section 8 du cas (énoncé d'appétence pour le risque); votre contexte du laboratoire 1.

## Mise en contexte

L'appétence exprime le niveau de risque de cybersécurité que la direction est prête à rechercher ou à conserver. CRG utilise les repères suivants :

| Valeur | Signification |
|---|---|
| 0,1–0,2 | Très forte aversion au risque — très faible appétence |
| 0,3 | Faible appétence |
| 0,5 | Neutralité face au risque, appétence modérée |
| 0,7 | Appétence relativement élevée |
| 0,8–0,9 | Très forte appétence |

Les valeurs de 0,1 et de 0,9 sont des extrêmes qui ne devraient normalement pas être utilisés.

## Étapes

1. Repérez ce que le cas énonce. L'énoncé d'appétence de MediBec indique une **tolérance normalisée de 0,40**, une faible appétence pour ce qui touche la sécurité des patients, la confidentialité des données de santé, la conformité et la confiance du public, ainsi que des seuils de remontée (risque résiduel supérieur à 0,25 pour un examen par la haute direction, supérieur à 0,40 pour une acceptation par le conseil d'administration).
2. Tenez compte des facteurs : secteur, mission, sensibilité des informations, exigences réglementaires, impact potentiel sur les personnes (patients), dépendance à la technologie, capacité financière, tolérance aux perturbations, réputation.
3. Choisissez une valeur. Vous pouvez conserver la valeur de 0,40 du cas ou en proposer une autre; dans les deux cas, justifiez-la en cinq à huit phrases. L'évaluation publiée de MediBec a utilisé **0,30** (une estimation analytique inférieure à la valeur énoncée de 0,40, en raison de la sécurité des patients et de la sensibilité des données de santé). Expliquez si vous êtes d'accord.
4. Consignez la valeur avec sa mention : si elle diffère de celle du cas, il s'agit d'une **Estimation analytique — validation requise**, et le choix final relève d'une **Décision de la direction requise**.
   - **[A]** Demandez : *« Estime l'appétence pour le risque de cybersécurité de MediBec sur une échelle de 0 à 1, avec la justification. »* Comparez la réponse avec l'énoncé du cas; n'acceptez pas une valeur que vous ne pouvez pas justifier vous-même.
   - **[B]** Organisation → Profil et appétence : saisissez l'appétence et sa justification. L'*Estimateur d'appétence* au bas de la page en propose une à partir des facteurs cochés.
   - **[C]** Saisissez l'appétence dans la feuille Analyse (cellule E14 du modèle standard).

**Livrable :** la valeur d'appétence et sa justification (une demi-page).

## Vérifiez votre compréhension


2.1. Une appétence de 0,30 est-elle « plus prudente » ou « moins prudente » qu'une appétence de 0,40? Qu'arrive-t-il au risque toléré lorsque l'appétence diminue?

2.2. Pourquoi est-ce une erreur de laisser l'appétence par défaut du classeur sans justification?

2.3. L'appétence est de 0,30 et le risque estimé d'un scénario de risque est de 6 467. Pouvez-vous dire si le scénario est acceptable à partir de ces deux seuls nombres? Pourquoi pas?

# Laboratoire 3 — Rédaction de scénarios de risque

**Objectifs :** rédiger des scénarios de risque sous forme de chaînes causales complètes; générer 20 candidats (version rapide des laboratoires; une évaluation complète en utilise environ 25 à 40, jamais moins de 20) et sélectionner les 10 plus significatifs; éviter les doublons et les simples noms de menaces.
**Durée :** 90 minutes. **Intrant :** contexte et joyaux de la couronne du laboratoire 1; section 9 du cas (bibliothèque de scénarios), à titre d'inspiration seulement.

## Mise en contexte

Un scénario de risque n'est pas un nom de menace. C'est une chaîne causale :

> **Source ou événement de menace → vulnérabilité ou faiblesse → actif ou processus touché → événement de cybersécurité → conséquence organisationnelle.**

Utilisez la structure de phrase suivante :

> *Parce que* [source ou événement de menace] *peut exploiter* [faiblesse] *touchant* [actif ou processus], [événement indésirable] *peut survenir, entraînant* [conséquences] *et compromettant* [objectif].

**Faible :** « Rançongiciel. » **Solide :** « Un affilié de rançongiciel utilise des identifiants VPN volés et l'absence d'authentification multifacteur (MFA) sur l'accès à distance pour chiffrer les serveurs de prise de rendez-vous cliniques, ce qui interrompt les rendez-vous dans tous les sites pendant plusieurs jours. »

## Étapes

1. Générez **20 scénarios de risque candidats** pour MediBec. Répartissez-les entre : identité, rançongiciels, infonuagique, tiers, initiés, applications et API, infrastructure, postes de travail, dispositifs médicaux, protection des renseignements personnels, fraude, résilience et reprise. Utilisez au moins trois stratégies de génération : à partir des joyaux de la couronne (descendante), à partir des faiblesses que révèle le cas (ascendante), à partir des menaces actives contre le secteur de la santé (axée sur la menace).
2. Pour chaque candidat, rédigez une phrase selon la structure causale et indiquez la dimension CID (confidentialité, intégrité, disponibilité) principalement touchée.
3. **Filtrez** les candidats. Notez chacun de 1 à 5 selon quatre critères : criticité pour l'organisation, pertinence de la menace, exposition, impact potentiel. Ajoutez un **indicateur de remontée** lorsque le scénario pourrait toucher la sécurité des patients, détruire les sauvegardes ou causer une perte de données catastrophique — de tels scénarios ne peuvent pas être écartés en raison de leur seule faible probabilité.
4. **Regroupez.** Fusionnez les variantes d'une même exposition (par exemple trois scénarios d'hameçonnage qui ne diffèrent que par le service visé).
5. **Sélectionnez-en dix.** Ne prenez pas simplement les dix totaux les plus élevés : vérifiez que le portefeuille couvre des actifs, des acteurs et des conséquences variés, et qu'aucune technique unique ne domine. Pour chaque scénario retenu, répondez à la question : *Pourquoi celui-ci, et non le plus solide des scénarios écartés?*
6. Développez **un** scénario retenu de façon complète : nom, parties prenantes, source de la menace, événement déclencheur, vulnérabilités, actifs et processus touchés, contrôles existants (uniquement ceux dont le cas fait état), séquence d'événements (cinq à huit étapes), conséquences (confidentialité, intégrité, disponibilité, opérations, finances, réglementation, réputation, vie privée, sécurité des personnes), hypothèses.
   - **[A]** `/select-scenarios` exécute la procédure complète de filtrage (sections A à E). Vous pouvez aussi demander 20 candidats, puis remettre chacun en question : *« Quelle étape de cette chaîne le cas appuie-t-il? »*
   - **[B]** Scénarios en lot → *Modèles de menace × actifs* → cochez les modèles, *Cocher les joyaux de la couronne + les 5 plus critiques*, réglez le *Nombre de candidats* à 20, **Générer des candidats**. Modifiez les noms et les énoncés dans la grille, cochez les vingt et **Ajouter les lignes cochées comme candidats au filtrage** : l'étape 5 du Processus compte les candidats dans Filtrage des scénarios → *Candidats et notes*, où vous les notez. Après votre sélection, cochez vos dix scénarios dans Scénarios en lot et **Ajouter les … sélectionnés au registre**. Facultatif : Organisation → Ontologie · CRGO → *Suggérer des scénarios* pour voir les scénarios connexes de la bibliothèque.

**Livrable :** les 20 candidats avec leurs notes, vos dix scénarios retenus avec une justification d'une ligne chacun, et un scénario entièrement développé (modèle de l'annexe C).

## Vérifiez votre compréhension


3.1. Réécrivez « Fuite de données chez MediBec » sous forme de scénario de risque à chaîne causale.

3.2. Pourquoi un scénario de faible probabilité qui pourrait détruire toutes les sauvegardes doit-il demeurer sur la liste?

3.3. Deux de vos scénarios sont « hameçonnage du personnel des cliniques » et « hameçonnage du personnel de la facturation ». Devriez-vous conserver les deux? Qu'est-ce qui justifierait de les garder distincts?

# Laboratoire 4 — Quantification d'un scénario de risque

**Objectifs :** estimer les six paramètres CRG d'un scénario de risque, chacun avec une valeur, une justification, des éléments probants et un niveau de confiance.
**Durée :** 75 minutes. **Intrant :** votre scénario développé au laboratoire 3 (ou le scénario S1 de MediBec, ci-dessous).

## Les six paramètres

| Symbole | Nom | Ce qu'il mesure | Faible (≈ 0,2–0,35) | Élevé (≈ 0,65–0,85+) |
|---|---|---|---|---|
| Pb(A) | Présence de la menace | Cette menace agira-t-elle contre MediBec au cours des 12 prochains mois? | Acteur rare, nécessite une position particulière | Observée ou constante (hameçonnage, balayage) |
| Pb(ψ,A) | Probabilité d'exploitation | Si elle agit, franchira-t-elle cette faiblesse, compte tenu des contrôles existants? | Contrôles solides et démontrés | Faiblesse démontrée (notification MFA approuvée, alerte fermée) |
| δe | Dommage estimé | Préjudice raisonnablement attendu | Dossiers limités, un seul site | À l'échelle du réseau, plusieurs sites |
| δm | Dommage maximal | Préjudice plausible dans le pire des cas | — | Impact sur la sécurité des personnes, perte de données catastrophique |
| θ | Résilience organisationnelle | Capacité de prévenir, détecter, contenir, intervenir, rétablir et poursuivre les activités (plus élevé est meilleur) | Aucune reprise testée, petite équipe | Reprise testée, détection 24 h/24, 7 j/7 |
| μ(E) | Criticité | Importance du service touché pour la mission | Processus de soutien | Essentiel à la mission (prestation des soins, identité) |

**Règles :** chaque valeur se situe entre 0 et 1; δm est au moins égal à δe; θ n'est jamais nul; Pb(A) et Pb(ψ,A) sont estimés séparément; μ(E) n'est pas une seconde estimation du dommage.

## Exemple résolu — MediBec S1

*S1 : Compromission des comptes d'administrateurs cliniques par hameçonnage et fatigue MFA.* Un acteur motivé par l'appât du gain hameçonne les administrateurs des cliniques au moyen d'un leurre sur le thème du DME, capture leurs identifiants et obtient l'approbation d'une notification MFA, ce qui lui donne un accès à distance au DME, à la facturation et aux systèmes administratifs.

| Paramètre | Valeur | Justification (abrégée) | Éléments probants | Confiance |
|---|---|---|---|---|
| Pb(A) | 0,85 | L'hameçonnage est constant et les tentatives augmentent; des leurres ont déjà été observés | Cas, s. 5; éléments probants de l'exercice | Moyenne |
| Pb(ψ,A) | 0,55 | La MFA par notification peut être contournée par la fatigue; aucun accès conditionnel ni blocage des macros n'est démontré | Éléments probants de l'exercice | Faible |
| δe | 0,55 | Accès à un sous-ensemble de dossiers; confinement possible si l'attaque est détectée | Cas, s. 3 | Moyenne |
| δm | 0,85 | Accès complet au DME, ou pivot vers un rançongiciel | Cas, s. 3 | Moyenne |
| θ | 0,40 | Équipe de sécurité de deux personnes; triage des alertes faible; intervention en cas d'incident non testée | Cas, s. 4 | Moyenne |
| μ(E) | 0,85 | L'identité est la porte d'entrée vers chacun des joyaux de la couronne | Cas, s. 4 | Élevée |

Les six valeurs sont des **Estimations analytiques — validation requise.**

## Étapes

1. Reprenez votre scénario développé au laboratoire 3.
2. Estimez chacun des six paramètres. Pour chacun, rédigez une ou deux phrases de justification, citez la section du cas (ou signalez la lacune) et indiquez un niveau de confiance : Élevé, Moyen ou Faible.
3. Nommez les **deux ou trois valeurs dont vous êtes le moins sûr**, ainsi que les éléments probants qui permettraient de trancher (une entrevue, un examen des journaux, un test de restauration, un contrat de fournisseur…).
   - **[A]** Demandez à l'assistant une *critique des paramètres* de vos valeurs. N'acceptez une modification que si vous pouvez la défendre vous-même.
   - **[B]** Éditeur de scénarios → *Quantification* : saisissez la valeur, la justification, les éléments probants et le niveau de confiance pour chaque paramètre.
   - **[C]** Saisissez les valeurs dans la feuille S du scénario; recalculez (F9).

**Livrable :** le tableau des paramètres de votre scénario, avec les valeurs les plus incertaines indiquées.

## Vérifiez votre compréhension


4.1. Pourquoi le fait de fusionner Pb(A) et Pb(ψ,A) en une seule « vraisemblance » entraînerait-il un double comptage?

4.2. Un coéquipier fixe θ = 0,90 « parce que MediBec a des pare-feu et un antivirus ». Qu'est-ce qui cloche dans ce raisonnement?

4.3. Pourquoi δm doit-il être au moins égal à δe?

# Laboratoire 5 — CVSS v4.0 pour un scénario de risque

**Objectifs :** coter une vulnérabilité définie avec le score de base CVSS v4.0; comprendre ce que le CVSS mesure et ne mesure pas.
**Durée :** 60 minutes.

## Mise en contexte

Le CVSS (Common Vulnerability Scoring System) mesure la **gravité technique d'une vulnérabilité**. CRG utilise le score de base CVSS v4.0 comme multiplicateur dans la formule de risque. Ce n'est pas une mesure du risque organisationnel : un CVSS élevé ne signifie pas le risque d'affaires le plus élevé.

Pour les scénarios de risque non techniques (hameçonnage, utilisation abusive par un initié, fraude, lacunes de gouvernance), cotez une **faiblesse représentative** (par exemple « MFA par notification sur un seul canal ») et qualifiez le score d'*applicabilité limitée* — ou expliquez pourquoi le CVSS ne s'applique pas.

## Les métriques de base

| Groupe | Métrique | Valeurs |
|---|---|---|
| Exploitabilité | Vecteur d'attaque (AV) | Réseau, Adjacent, Local, Physique |
| | Complexité de l'attaque (AC) | Faible, Élevée |
| | Exigences de l'attaque (AT) | Aucune, Présentes |
| | Privilèges requis (PR) | Aucun, Faibles, Élevés |
| | Interaction de l'utilisateur (UI) | Aucune, Passive, Active |
| Impact sur le système vulnérable | Confidentialité, intégrité, disponibilité (VC, VI, VA) | Élevé, Faible, Aucun |
| Impact sur les systèmes subséquents | Confidentialité, intégrité, disponibilité (SC, SI, SA) | Élevé, Faible, Aucun |

## Exemple résolu — MediBec S1

Faiblesse représentative : MFA par notification sans correspondance de nombres sur l'accès à distance.

`CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:A/VC:H/VI:H/VA:N/SC:H/SI:L/SA:L` → **9,3 (Critique)**

| Métrique | Valeur | Justification |
|---|---|---|
| AV | Réseau | Le portail d'accès à distance est accessible par Internet |
| AC | Faible | Aucune condition particulière au-delà de la trousse d'hameçonnage |
| AT | Aucune | Aucun prérequis propre au déploiement |
| PR | Aucun | L'attaquant part sans compte |
| UI | Active | Un utilisateur doit saisir ses identifiants et approuver la notification |
| VC / VI | Élevé / Élevé | Contrôle complet de la session compromise |
| VA | Aucun | La disponibilité de la passerelle n'est pas touchée |
| SC | Élevé | L'accès s'étend aux dossiers des patients dans le DME |
| SI / SA | Faible / Faible | Effet limité en aval sur l'intégrité et la disponibilité |

## Étapes

1. Définissez, en une phrase, la vulnérabilité unique dont dépend votre scénario de risque.
2. Choisissez chaque métrique de base et justifiez-la en une ligne.
3. Calculez le score avec un calculateur — ne l'estimez jamais à l'œil.
   - **[A]** `/crg-calc` ou l'outil `cvss4_score` du serveur `crg-calculator`.
   - **[B]** Outils → *CVSS v4.0* : collez ou construisez le vecteur, puis enregistrez-le dans le scénario.
   - **[C]** Utilisez le calculateur CVSS v4.0 du NVD (nvd.nist.gov/vuln-metrics/cvss/v4-calculator) ou celui de FIRST, puis saisissez le score dans la colonne G de la feuille Analyse et la chaîne complète du vecteur à côté, dans la colonne H; la colonne de contrôles signale un vecteur manquant ou mal formé.
4. Rédigez une phrase sur les limites du score pour votre scénario.

**Livrable :** le vecteur, le score et la justification des métriques.

## Vérifiez votre compréhension


5.1. Faites passer l'interaction de l'utilisateur de S1 d'Active à Aucune. Vous attendez-vous à ce que le score augmente ou diminue? Pourquoi est-ce logique?

5.2. Pourquoi devez-vous conserver la chaîne complète du vecteur, et non seulement le score?

5.3. Donnez un scénario de risque de MediBec pour lequel le CVSS n'a qu'une applicabilité limitée, et expliquez.

# Laboratoire 6 — Calcul et interprétation du risque

**Objectifs :** calculer le risque estimé et le risque toléré; interpréter le résultat par rapport à la tolérance; expliquer ce qui détermine le verdict.
**Durée :** 45 minutes.

## Les formules

```
Risque estimé  Re = Pb(A) × Pb(ψ,A) × CVSS × ((δe + δm) / 2) × μ(E) ÷ θ × Facteur
Risque toléré  Rt = Pb(A) × Pb(ψ,A) × CVSS × Appétence × μ(E) ÷ θ × Facteur
```

Le facteur (1 000 dans l'évaluation de MediBec) ne sert qu'à mettre les nombres à l'échelle. Les scores de risque sont des **indicateurs relatifs** servant à la comparaison et à la priorisation; ce ne sont ni des dollars ni des prédictions de pertes.

## Exemple résolu — MediBec S1 (appétence 0,30, facteur 1 000)

```
Re = 0,85 × 0,55 × 9,3 × ((0,55 + 0,85) / 2) × 0,85 ÷ 0,40 × 1 000 = 6 467
Rt = 0,85 × 0,55 × 9,3 × 0,30 × 0,85 ÷ 0,40 × 1 000          = 2 772
Ratio avant traitement = Re ÷ Rt = 2,33  →  au-dessus de la tolérance avant traitement
```

Remarquez que le ratio avant traitement correspond simplement à **dommage moyen ÷ appétence** = 0,70 ÷ 0,30 = 2,33. Tout le reste s'annule.

## Étapes

1. Calculez Re et Rt pour votre scénario de risque à la main (montrez chaque multiplication), puis avec le moteur. Les deux résultats doivent concorder à l'unité près.
   - **[A]** `/crg-calc` avec les six paramètres, le vecteur CVSS, l'appétence et le facteur.
   - **[B]** Calculateur de risque → cliquez sur la ligne de votre scénario pour voir chaque étape de la formule.
   - **[C]** Recalculez (F9); lisez les colonnes L (risque estimé) et M (risque toléré) de la feuille Analyse. **Ne vous fiez pas à la couleur rouge de la cellule** : comparez L et M numériquement.
2. Calculez le ratio avant traitement et vérifiez qu'il est égal à dommage moyen ÷ appétence.
3. Refaites l'exercice avec l'appétence énoncée dans le cas (0,40). Qu'est-ce qui change et qu'est-ce qui ne change pas?
4. **[B]** Examinez l'ensemble du registre de MediBec (30 scénarios). Classez-les selon le risque estimé, puis selon le ratio avant traitement. Pourquoi les deux classements diffèrent-ils?

**Livrable :** votre calcul présenté étape par étape, le résultat du moteur et une interprétation en trois phrases.

## Vérifiez votre compréhension


6.1. Le scénario X a Pb(A) = 0,9 et le scénario Y a Pb(A) = 0,2. Tous leurs autres paramètres sont égaux. Lequel a le risque estimé le plus élevé? Lequel a le ratio avant traitement le plus élevé?

6.2. Qu'arrive-t-il à Re et à Rt si vous faites passer le facteur de 1 000 à 10 000? Et au ratio?

6.3. Pourquoi « risque estimé = 6 467 » n'est-il pas un montant en dollars?

# Laboratoire 7 — Traitement, contrôles partagés et risque résiduel

**Objectifs :** concevoir un ensemble de mesures de traitement lié à la chaîne causale; estimer les réductions de probabilité et d'impact; calculer le risque résiduel; éviter le double comptage des contrôles partagés.
**Durée :** 90 minutes.

## Les formules

```
Risque atténué  = Re × Réduction de la probabilité × Réduction de l'impact
Risque résiduel = Re − Risque atténué
Ratio           = Résiduel ÷ Toléré
```

Interprétez le ratio ainsi : **inférieur à 0,90** sous la tolérance · **0,90–1,10** approximativement à la tolérance · **supérieur à 1,10** au-dessus de la tolérance.

L'atténuation est **multiplicative** : une réduction de 80 % de la probabilité et une réduction de 80 % de l'impact éliminent 64 % du risque. Un ensemble de mesures qui ne réduit que la probabilité, ou que l'impact, n'atténue rien selon cette convention. Pour que le risque résiduel ne dépasse pas le risque toléré (ratio ≤ 1,0), il faut P × Q ≥ 1 − appétence ÷ dommage moyen; pour le ramener **sous la tolérance** (ratio < 0,90), il faut P × Q > 1 − 0,90 × appétence ÷ dommage moyen.

## Exemple résolu — MediBec S1

Ensemble de mesures : MFA résistante à l'hameçonnage et accès conditionnel (I01), gestion des accès privilégiés (I02), détection et intervention gérées 24 h/24, 7 j/7 (I03), sensibilisation (I12). Réduction de la probabilité 0,80, réduction de l'impact 0,75.

```
Atténué  = 6 467 × 0,80 × 0,75 = 3 880
Résiduel = 6 467 − 3 880       = 2 587
Ratio    = 2 587 ÷ 2 772        = 0,93  →  approximativement à la tolérance
```

Vérification : 1 − 0,30 ÷ 0,70 = 0,571 et 1 − 0,90 × 0,30 ÷ 0,70 = 0,614; P × Q = 0,60 se situe entre les deux, de sorte que le risque résiduel ne dépasse pas le toléré sans passer sous la bande de 0,90 : approximativement à la tolérance.

## Étapes

1. Pour votre scénario de risque, énumérez **trois à cinq mesures**. Pour chacune, précisez : quelle étape de la chaîne causale elle interrompt; ce qu'elle ajoute aux contrôles déjà en place; son responsable; son coût ponctuel et son coût récurrent (en dollars canadiens, sur un horizon unique de 12 mois); son délai avant bénéfice.
2. Estimez la réduction de la probabilité et la réduction de l'impact de l'ensemble de mesures (chacune entre 0 et 1) et justifiez-les.
3. Calculez le risque atténué, le risque résiduel et le ratio.
4. **Contrôles partagés.** Déterminez lesquelles de vos mesures servent aussi d'autres scénarios. Répartissez leur coût entre ces scénarios (une répartition égale est acceptable) et notez que le total du portefeuille compte chaque initiative une seule fois.
5. Si le scénario demeure au-dessus de la tolérance : énoncez les options — atténuer davantage, **transférer** (assurance, contrat), **éviter** (cesser l'activité) ou **accepter** formellement — et qui doit décider.
   - **[B]** Atténuation des risques → *Propositions* ou *Catalogue de contrôles*; fixez les coûts et rp / ri; **Appliquer**. Recommandations → *Besoin de traitement* affiche le verdict pour chaque scénario.
   - **[C]** Sélectionnez les contrôles dans la feuille M (1 dans la colonne A). La feuille combine les coefficients retenus selon la règle du produit, réduction du bouquet = 1 − ∏(1 − rᵢ), séparément pour l'impact et la probabilité, de sorte qu'un bouquet ne dépasse jamais 1; un avertissement s'affiche si un coefficient est hors de 0 à 1. Les coûts par défaut sont des coûts indicatifs de première année pour une organisation de taille moyenne, selon un profil d'effort (le tableau de base des coûts est sous le catalogue), et les coefficients par défaut sont génériques : remplacez les deux par les chiffres du cas, privilégiez quelques contrôles justifiés, ou saisissez une estimation globale dans une ligne « Other / autre ».

**Livrable :** le tableau de traitement, le calcul du risque résiduel et la décision requise.

## Vérifiez votre compréhension


7.1. Un ensemble de mesures a une réduction de la probabilité de 0,90 et une réduction de l'impact de 0,00. Quel est le risque atténué? Qu'est-ce que cela vous apprend sur la conception de l'ensemble?

7.2. La MFA sert sept scénarios de MediBec et coûte 180 000 $ une seule fois. Comment son coût devrait-il apparaître dans chaque scénario et dans le portefeuille?

7.3. Pourquoi le fait de recommander un contrôle « que l'organisation possède déjà » ne peut-il jamais réduire le risque?

# Laboratoire 8 — Sensibilité et incertitude

**Objectifs :** vérifier comment la conclusion change lorsque vos estimations changent; déterminer les décisions qui dépendent de valeurs incertaines.
**Durée :** 45 minutes.

## Mise en contexte

Vos paramètres sont des estimations. La question n'est pas « quel est le risque exact? », mais « la décision change-t-elle si je me trompe? ». La convention de CRG construit trois cas : faire varier Pb(A), Pb(ψ,A), δe et δm de ∓0,10 / ±0,10, θ de ±0,10 / ∓0,10, et les deux réductions de ±0,10 / ∓0,10, en maintenant chaque valeur entre 0,05 et 0,99.

## Exemple résolu — MediBec S1

| Cas | Estimé | Toléré | Résiduel | Ratio | Verdict |
|---|---|---|---|---|---|
| Inférieur (optimiste) | 3 202 | 1 601 | 752 | 0,47 | Sous la tolérance |
| Central | 6 467 | 2 772 | 2 587 | 0,93 | Approximativement à la tolérance |
| Supérieur (pessimiste) | 13 017 | 4 881 | 7 094 | 1,45 | Au-dessus de la tolérance |

Le verdict change d'un cas à l'autre : S1 est un scénario de risque pour lequel **de meilleurs éléments probants valent la peine d'être payés** (par exemple, vérifier si les utilisateurs approuvent les notifications MFA) avant l'investissement ou en parallèle.

## Étapes

1. Construisez les cas inférieur, central et supérieur pour votre scénario.
2. Le verdict change-t-il? S'il est « au-dessus de la tolérance » même dans le cas inférieur, le scénario nécessite un transfert ou une acceptation formelle, et non seulement davantage de dépenses.
3. Déterminez le paramètre dont la variation modifie le plus le résultat. **[B]** Outils → Prévisions → *Facteurs déterminants* (diagramme en tornade); *Monte-Carlo* convertit vos niveaux de confiance en plages.
4. Rédigez deux phrases à l'intention de la direction : ce dont vous êtes certain et quels éléments probants changeraient la recommandation.

**Livrable :** le tableau des trois cas et l'énoncé de deux phrases.

## Vérifiez votre compréhension


8.1. Dans l'exemple de S1, quelle décision de la direction changerait entre le cas inférieur et le cas supérieur?

8.2. Une prévision donne un intervalle étroit. Cela signifie-t-il que vos estimations sont exactes?

8.3. Pourquoi une diminution de θ fait-elle augmenter à la fois le risque estimé et le risque toléré, mais laisse-t-elle le ratio inchangé?

# Laboratoire 9 — Indicateurs clés de risque

**Objectifs :** définir des indicateurs clés de risque (ICR) mesurables qui indiquent à la direction si un risque évolue.
**Durée :** 45 minutes.

## Mise en contexte

Un ICR n'est utile que si quelqu'un peut le mesurer, à partir d'une source nommée, selon un calendrier, par rapport à des seuils qui déclenchent une action.

## Exemple résolu — MediBec

| ICR | Risque visé | Mesure | Source | Responsable | Fréquence | Cible | Avertissement | Critique |
|---|---|---|---|---|---|---|---|---|
| ICR-01 % des comptes privilégiés et d'accès à distance protégés par une MFA résistante à l'hameçonnage | S1, S8, S9 | Comptes FIDO2 ÷ comptes visés | Rapport du fournisseur d'identité | Responsable de la cybersécurité | Mensuelle | 100 % | < 95 % | < 85 % |
| ICR-05 Réussite des tests de restauration des sauvegardes des joyaux de la couronne dans le RTO | S2, S3, S10 | Tests réussis ÷ tests effectués | Plateforme de sauvegarde, registres de reprise après sinistre | DPI (Infrastructure) | Trimestrielle | 100 % | < 95 % | < 80 % |

## Étapes

1. Définissez **trois ICR** pour vos principaux scénarios de risque, en remplissant chaque colonne du tableau ci-dessus.
2. Pour chaque ICR, indiquez ce que la direction devrait faire lorsque le seuil d'avertissement est franchi.
3. **[B]** Évaluation des risques → Tableau de bord des ICR → *Enregistrer une mesure*. Après quatre relevés, Outils → Prévisions → *Projections des ICR* indique si un seuil sera franchi.

**Livrable :** trois ICR complets.

## Vérifiez votre compréhension


9.1. Qu'est-ce qui cloche avec l'ICR « la sécurité s'améliore »?

9.2. Pourquoi un indicateur pour lequel « une valeur plus basse est meilleure » doit-il avoir des seuils formulés avec « supérieur à »?

9.3. Combien de relevés l'application de bureau exige-t-elle avant de projeter la tendance d'un ICR, et pourquoi refuse-t-elle de le faire avec moins de relevés?

# Laboratoire 10 — La note de service à la direction

**Objectifs :** rédiger une note de service d'une page, axée sur la décision, à l'intention de la haute direction.
**Durée :** 60 minutes.

## Ce que fait une note de service solide

- Elle **commence par la décision** que la direction doit prendre.
- Elle indique combien de scénarios de risque sont au-dessus de la tolérance **avant et après** le traitement.
- Elle présente le **coût du portefeuille recommandé** par rapport au contexte budgétaire.
- Elle précise le **risque résiduel** qui demeure et qui doit l'accepter formellement.
- Elle nomme les **deux ou trois principales incertitudes** et les prochaines étapes.
- Elle **ne reprend pas** les tableaux. Elle concorde avec le classeur à l'unité près.

## Points de référence de MediBec

L'évaluation publiée de 30 scénarios a établi ce qui suit : après traitement, 12 scénarios au-dessus de la tolérance, 11 approximativement à la tolérance et 7 sous la tolérance; risque estimé total de 84 305 contre un risque résiduel de 41 023 (une réduction de 51 %); dépenses actuelles en cybersécurité d'environ 0,9 % du budget des TI, sous le plancher de 4 % de la ligne directrice budgétaire. *Ne copiez pas ces chiffres dans votre propre cas.*

## Étapes

1. Rédigez la note de service à l'aide du modèle de l'annexe C.
2. Échangez vos notes avec une autre équipe. Chaque réviseur répond aux questions suivantes : *Qu'est-ce qu'on me demande exactement d'approuver? Le coût est-il clair? Quel chiffre contesterais-je?*
3. Révisez.
   - **[A]** `/qc-review` effectue une revue de qualité indépendante de votre note; `check_labels.py` énumère les mentions manquantes.
   - **[B]** Gouvernance → *Revue de qualité* avant d'exporter quoi que ce soit.

**Livrable :** une note de service d'une page.

## Vérifiez votre compréhension


10.1. Pourquoi le premier paragraphe de la note de service devrait-il énoncer la décision plutôt que la méthode?

10.2. Votre note indique « risque résiduel 41 023 » et votre classeur indique 41 032. Que devriez-vous faire?

10.3. Qui accepte formellement un risque résiduel qui demeure au-dessus de la tolérance?

# Laboratoire 11 — Éléments probants des opérations de sécurité (facultatif, parcours application de bureau)

**Objectifs :** relier les éléments probants des opérations de sécurité aux scénarios de risque sans transformer les alertes en incidents ni les niveaux de confiance en probabilités.
**Durée :** 120 minutes. **Prérequis :** l'application de bureau avec la trousse de simulation MediBec chargée par votre enseignant.

## Étapes

1. Opérations de sécurité → SIEM → *Vue d'ensemble* : réglez la date de référence au **9 octobre 2026**.
2. Effectuez les dix exercices que votre enseignant vous assigne (EX-01 à EX-10) dans le SIEM et le SOAR : ordre de triage, un vrai positif fermé comme bénin, lacunes de couverture, télémétrie muette, restrictions TLP, un indicateur avec une confiance de 95, un dossier qui peut être ou ne pas être un incident, des actions sans approbation valide, état de préparation des playbooks, lacunes de détection.
3. Scénarios en lot → *Des opérations de sécurité (SIEM et SOAR)*, en date du 9 octobre. **Consigner la preuve dans S2.** Ouvrez S2 dans l'éditeur de scénarios : la fiche *Preuves des opérations de sécurité* l'affiche, et **aucun paramètre n'a changé**.
4. Décidez, en équipe, si les éléments probants justifient une modification de Pb(A), de Pb(ψ,A) ou de θ pour S2. Rédigez la justification. Lancez ensuite Gouvernance → *Revue de qualité*.

**Livrable :** les réponses aux exercices assignés et la justification relative à S2.

## Vérifiez votre compréhension


11.1. Un indicateur porte une confiance de source de 95. Pouvez-vous fixer Pb(A) = 0,95? Pourquoi pas?

11.2. Qu'est-ce qui transforme un dossier SOAR en incident?

11.3. Pourquoi une règle de détection dont la télémétrie requise est muette ne compte-t-elle pas comme couverture?

# Laboratoire 12 — Du risque à la politique et à l'audit

**Objectifs :** utiliser vos résultats d'analyse de risque pour justifier une politique de sécurité et pour prioriser un plan d'audit axé sur les risques.
**Durée :** 90 minutes (deux parties).

## Partie 1 — Du risque à l'audit

1. Prenez les cinq scénarios de risque de MediBec dont le risque résiduel est le plus élevé. Pour chacun, nommez l'**entité auditable** (un processus, un système ou un fournisseur) et le **contrôle clé** dont dépend le traitement.
2. Pour un contrôle clé, concevez deux tests : un **test de conception** (le contrôle est-il conçu pour fonctionner?) et un **test d'efficacité opérationnelle** sur une période (a-t-il fonctionné?). Précisez la population, la période, l'unité d'échantillonnage, la méthode et la taille de l'échantillon.
3. Proposez une analyse de données (par exemple : comparaison des employés ayant quitté l'organisation avec les comptes actifs), avec les données, la logique de la requête et les étapes de validation des données.

## Partie 2 — Du risque à la politique

1. Rédigez la phrase : *« MediBec a besoin d'une politique de gestion des identités et des accès parce que \_\_\_, et cette politique sera réussie lorsque \_\_\_. »* Utilisez S1 et vos résultats.
2. Rédigez cinq énoncés de politique numérotés. Chacun nomme un acteur, utilise **doit** ou **ne doit pas** et énonce un résultat vérifiable. Placez les valeurs techniques (longueur des mots de passe, intervalles de revue) dans une *norme* nommée, et non dans la politique.
3. Définissez un indicateur clé de performance (ICP) pour la politique, avec une cible, un responsable, une source de données et une fréquence.

**Livrable :** la conception du test d'audit et les cinq énoncés de politique avec un ICP. Si votre cours utilise le laboratoire 13, fondez la partie 2 sur ses registres.

## Vérifiez votre compréhension


12.1. Pourquoi « l'auditeur a confirmé qu'une politique de revue des accès existe » ne constitue-t-il pas un élément probant que les revues des accès fonctionnent?

12.2. Réécrivez sous forme d'énoncé vérifiable : « Des contrôles de sécurité appropriés devraient être utilisés pour l'accès à distance. »

12.3. Où devrait-on inscrire « les mots de passe doivent comporter au moins 14 caractères », et pourquoi?

# Laboratoire 13 — Posture des identités, des accès et de la configuration (parcours application de bureau)

**Objectifs :** lire la posture d'une organisation en matière d'identités, d'accès et de configuration à partir de registres au niveau des populations; recalculer un indicateur à partir de son registre; transformer un constat en élément probant et en proposition de paramètre sans laisser l'outil décider; relier la posture à la politique de gestion des identités et des accès du travail A2.
**Durée :** 90 minutes. **Prérequis :** l'application de bureau (1.9.1 ou plus récente) et une copie de l'espace de travail MediBec. Les parcours A et C font les étapes 2 et 7, ainsi que le recalcul de CRG-22 de l'étape 4, sur papier avec le tableau 13.1. **À lire d'abord :** chapitre 10 (gestion des identités et des accès privilégiés) et chapitre 14 (configurations de référence et dérives) du manuel.

## Mise en contexte

Deux modules facultatifs de l'application de bureau consignent la qualité de la gouvernance des identités, des accès privilégiés et des configurations : **Gestion des identités et des accès (GIA/GAP)** et **Gestion de la configuration**. Ce sont des registres stratégiques de posture, et non des outils d'administration :

- ils contiennent des **décomptes par population** (un groupe de comptes, un niveau de privilèges, une campagne de revue, un périmètre de configuration), jamais un nom, un nom d'utilisateur, une adresse courriel ni un secret;
- un **décompte vide est inconnu, et non nul** (« Laissez vide si inconnu — 0 signifie aucun »), et le demeure dans tous les calculs;
- leurs indicateurs (CRG-17 à CRG-25) sont des **ICR mesurés**, assortis de seuils, et non des indices de risque;
- un **constat** proposé par une règle ne donne qu'un sens (Pb(ψ,A) ↑ ou θ ↓). C'est vous qui saisissez la nouvelle valeur; l'application montre le ratio avant et après; **Accepter** est un acte distinct, consigné dans le fondement du scénario et dans le journal des révisions. Rien n'est jamais appliqué automatiquement.

L'espace de travail MediBec contient un exemple fictif pour les deux modules, daté du 1er novembre 2026 : cinq populations de comptes, trois niveaux de privilèges, deux ensembles de comptes d'urgence, deux campagnes de revue des accès, deux règles de séparation des tâches, trois configurations de référence, trois périmètres de configuration, trois écarts et un trimestre de changements de configuration. Cinq constats sont déjà consignés (IF-01 à IF-05) et reliés comme éléments probants (E13 à E15). Aucune proposition n'a été acceptée : les totaux restent 84 305 / 41 023.

## Tableau 13.1 — Décomptes de l'exemple MediBec (pour les parcours A et C)

Groupes de rôles privilégiés : PR-01 administrateurs IdP, domaine et locataire infonuagique; PR-02 administrateurs DME, bases de données et serveurs; PR-03 administrateurs d'applications et soutien.

| Groupe | Niveau | Titulaires | Permanents | JAT | AMF | Partagés | Dernière revue |
|---|---|---|---|---|---|---|---|
| PR-01 | 0 | 9 | 9 | 0 | 9 | 1 | 2026-03-10 |
| PR-02 | 1 | 27 | 21 | 6 | 24 | 0 | 2025-10-02 |
| PR-03 | 2 | 58 | 58 | 0 | 41 | 0 | (vide) |

JAT : titulaires juste-à-temps. AMF : titulaires avec AMF. Partagés : comptes d'administration partagés.

| Périmètre de configuration | Dans le périmètre | Avec une configuration de référence | Évalués | Conformes |
|---|---|---|---|---|
| CS-01 Serveurs Windows | 96 | 88 | 70 | 58 |
| CS-02 Postes cliniques | 1 020 | 1 020 | 912 | 801 |
| CS-03 Dispositifs médicaux connectés | 340 | 0 | (vide) | (vide) |

## Étapes

1. **[B]** Paramètres → Modules : activez **Gestion des identités et des accès (GIA/GAP)** et **Gestion de la configuration** (groupe Gouvernance). Deux nouveaux écrans apparaissent sous Gouvernance.
2. **Lisez les indicateurs d'identité.** Gouvernance → Gestion des identités et des accès (GIA/GAP) → Vue d'ensemble. Recopiez les six tuiles d'indicateurs (valeur, numérateur et dénominateur, décomptes inconnus et périmés, statut). Recalculez ensuite vous-même **CRG-18 Couverture de l'AMF des accès privilégiés** et **CRG-19 Proportion de privilèges permanents** à partir de l'onglet Rôles privilégiés (ou du tableau 13.1). CRG-19 ne compte que les niveaux 0 et 1. Vos résultats doivent concorder avec les tuiles.
3. **Regardez les lignes derrière les indicateurs.** Dans les onglets Comptes et Rôles privilégiés, trouvez une ligne dont le statut est *Inféré* et une ligne marquée **collecte périmée**. Pourquoi les populations de comptes de service et d'agents d'IA n'ont-elles aucun décompte d'AMF?
4. **Lisez les indicateurs de configuration.** Gouvernance → Gestion de la configuration → Vue d'ensemble. Recopiez les trois tuiles et les données clés. Recalculez **CRG-22 Couverture d'affectation des configurations de référence** à partir des périmètres de configuration. Ouvrez ensuite *Écarts et exceptions* : pourquoi DV-02 est-il marqué *Expirée — non comptée*?
5. **D'un constat à un élément probant.** Gouvernance → Gestion des identités et des accès (GIA/GAP) → *Constats et propositions*. Parmi les constats proposés, choisissez **IR-03 Privilège permanent de niveau 0** (ligne PR-01). **Consignez-le** (il devient IF-06), puis **Consigner comme preuve** : notez le numéro E qu'il reçoit (E16 dans une copie neuve de MediBec). **Lier un scénario** : reliez-le à S1 (compromission d'identifiants de comptes d'administrateurs cliniques).
6. **D'un élément probant à une proposition.** Sur le constat consigné, **Proposer une modification de paramètre** pour la Pb(ψ,A) de S1 (actuellement 0,55). Saisissez la valeur que **vous** défendez, un niveau de confiance et une justification qui cite le constat et son numéro E (la boîte de dialogue préremplit la citation), puis lisez le ratio actuel et après, calculé par le moteur. Remarquez que le ratio ne bouge pas (0,93 → 0,93) : Pb(ψ,A) s'annule dans le ratio (laboratoire 6), tandis que les risques estimé et toléré augmentent tous deux. Un constat de posture change habituellement le rang d'un scénario, pas son verdict. **Enregistrer comme proposition.** **N'acceptez pas** la proposition dans l'espace de travail qui sert aux autres laboratoires : l'accepter modifie les totaux de MediBec, et les valeurs de référence des laboratoires 6 à 10 ne concorderaient plus.
7. **De la posture à la politique (A2).** Rédigez trois besoins de politique de gestion des identités et des accès qu'appuient les registres (par exemple l'AMF pour chaque titulaire de privilèges, l'examen de chaque utilisation d'un compte d'urgence, un propriétaire pour chaque compte de service). Pour chacun, donnez l'indicateur qui en assurerait le suivi, avec une cible, un seuil d'alerte et un seuil critique, ainsi que le registre qui en est la source de données.
8. Rédigez deux phrases pour la direction : ce que la posture des identités et de la configuration dit de S1 et de S6, et quels décomptes doivent être confirmés avant qu'on s'y fie.

**Livrable :** le tableau des indicateurs avec vos deux valeurs recalculées; une capture d'écran de la proposition enregistrée avec sa justification; les trois besoins de politique avec leurs indicateurs; les deux phrases.

## Vérifiez votre compréhension

13.1. La population des comptes de service (AP-03) n'a aucun décompte d'AMF. Sa couverture d'AMF est-elle de 0 %?

13.2. Le constat IR-02 affiche l'indication de sens « Pb(ψ,A) ↑ ». Pourquoi l'application ne calcule-t-elle pas elle-même la nouvelle valeur?

13.3. DV-02 est un écart *compensé*, et pourtant il est compté dans CRG-24 Exceptions expirées. Pourquoi?

# Cas d'exercice — CanaRetail (autonome, parcours application de bureau)

**Objectifs :** refaire les laboratoires 4 à 7 sur une deuxième organisation et vous vérifier à l'aide d'une solution complète et publiée.
**Durée :** 2 à 3 heures, seul. **Intrants :** l'étude de cas CanaRetail (`cases/CanaRetail_BusinessCase_v2-0c_Student.docx`) et l'espace de travail CanaRetail fourni.

CanaRetail est un détaillant national fictif : 100 magasins et une plateforme de commerce électronique. L'application de bureau fournit son évaluation du module D en anglais et en français : dix scénarios (les neuf fournis par le cas et un scénario original, S10), leurs paramètres, leurs vecteurs CVSS, dix initiatives de traitement et un sommaire de gestion, à l'appétence de 0,5 que le cas énonce. Comme la solution est publique, CanaRetail sert à l'exercice, **jamais à un travail noté**.

## Étapes

1. **[B]** Organisation → Espaces de travail → Nouvel espace de travail, *Partir de* **Exemple pédagogique : CanaRetail — commerce de détail national** (anglais ou français). Dans Évaluation des risques → Registre des scénarios, lisez **seulement les noms et les énoncés** des dix scénarios et choisissez-en trois. N'ouvrez pas encore leurs paramètres.
2. **Travaillez d'abord, regardez ensuite.** À partir du document du cas seulement, rédigez les six paramètres de chacun de vos trois scénarios avec leur justification (laboratoire 4) et son vecteur CVSS v4.0 (laboratoire 5). Ouvrez ensuite chacun d'eux dans l'éditeur de scénarios et comparez vos valeurs à celles de la solution. Pour chaque écart de plus de 0,15, rédigez une phrase : quelle preuve du cas appuie chaque valeur?
3. **Vérifiez le moteur.** Calculateur de risque : pour S1, vous devriez lire un risque estimé de 4 453, un risque toléré de 2 783 et un risque résiduel de 2 572, soit un ratio de 0,92 — environ à la tolérance. Les totaux sont **27 526 (estimé)** et **16 049 (résiduel)**.
4. **Lisez le portefeuille** (laboratoires 6 et 7). Après traitement, 2 scénarios sont au-dessus de la tolérance (S2, S9), 3 environ à la tolérance (S1, S6, S10) et 5 au-dessous. Pourquoi le traitement de S10 (I02, I03, I06, I08) le laisse-t-il environ à la tolérance plutôt qu'au-dessous?
5. **Lien avec les identités.** Si votre enseignant utilise le laboratoire 13, ouvrez Gouvernance → Gestion des identités et des accès (GIA/GAP) → *Constats et propositions* dans l'espace de travail CanaRetail : un constat (comptes de service sans propriétaire) porte une proposition **en attente** sur S10 (Pb(ψ,A) 0,40 → 0,45). Décidez si vous l'accepteriez, et pourquoi. Ne l'acceptez pas si vous voulez que les totaux ci-dessus demeurent valides.

**Livrable :** aucun à remettre; conservez vos notes de comparaison pour le travail A1.

# Annexe A — Aide-mémoire des formules

```
Risque estimé  Re = Pb(A) × Pb(ψ,A) × CVSS × ((δe + δm) / 2) × μ(E) ÷ θ × Facteur
Risque toléré  Rt = Pb(A) × Pb(ψ,A) × CVSS × Appétence × μ(E) ÷ θ × Facteur
Atténué            = Re × Réduction de la probabilité × Réduction de l'impact
Résiduel           = Re − Atténué
Ratio              = Résiduel ÷ Rt
Coût-efficacité    = (Re − Résiduel) ÷ Coût × 1 000   (unités de risque éliminées par tranche de 1 000 $)

Simplifications à retenir
Ratio avant traitement = ((δe + δm) / 2) ÷ Appétence
Ratio après traitement = ((δe + δm) / 2) × (1 − P × Q) ÷ Appétence
Ratio ≤ 1,0 exige          P × Q ≥ 1 − Appétence ÷ ((δe + δm) / 2)
Sous la tolérance (< 0,90)  P × Q > 1 − 0,90 × Appétence ÷ ((δe + δm) / 2)
Plusieurs mesures, un levier  r = 1 − ∏(1 − rᵢ)
```

Classification : ratio < 0,90 sous la tolérance · 0,90–1,10 approximativement à la tolérance · > 1,10 au-dessus de la tolérance.

Ligne directrice budgétaire (CRG) : dépenses en cybersécurité de **12 %** du budget total des TI (salaires compris) pour une appétence de 0,30 ou moins, de **7,8 %** à 0,50 et de **4 %** à 0,70 ou plus, avec une interpolation linéaire entre ces valeurs.

# Annexe B — Repères des paramètres et plages valides

| Paramètre | Plage valide | Faible | Modéré | Élevé |
|---|---|---|---|---|
| Pb(A) | 0–1 | Acteur ou événement rare | Plausible pour le secteur | Observé ou constant |
| Pb(ψ,A) | 0–1 | Contrôles solides et démontrés | Contrôles partiels ou inconnus | Faiblesse démontrée |
| δe, δm | 0–1, δm ≥ δe | Dossiers limités, un seul site | Important, plusieurs sites | À l'échelle du réseau, sécurité des personnes, perte catastrophique |
| θ | > 0 à 1 | Faible : aucune reprise testée | Partielle | Reprise testée, détection 24 h/24, 7 j/7 |
| μ(E) | 0–1 | Processus de soutien | Processus important | Essentiel à la mission |
| CVSS | 0–10 | — | — | — |
| Réductions | 0–1 chacune | — | — | — |

# Annexe C — Modèles

## C.1 Modèle de développement d'un scénario de risque

| Champ | Contenu |
|---|---|
| Identifiant et nom | |
| Énoncé causal (une ou deux phrases) | |
| Parties prenantes | |
| Source de la menace | |
| Événement déclencheur | |
| Vulnérabilités ou conditions prédisposantes | |
| Actifs touchés | |
| Processus d'affaires touchés | |
| Contrôles existants (avec éléments probants) | |
| Séquence d'événements (5 à 8 étapes) | |
| Conséquences : C, I, D, opérations, finances, réglementation, réputation, vie privée, sécurité des personnes | |
| Hypothèses et lacunes d'information (avec mentions) | |
| Pourquoi ce scénario? (pertinence, éléments probants, actif, faiblesse, conséquence, raisons pour lesquelles il l'emporte sur les scénarios non retenus) | |

## C.2 Modèle de note de service à la direction

> **Destinataire :** [Membre de la haute direction] · **Expéditeur :** [Équipe] · **Date :** · **Objet :** Décision requise — portefeuille de traitement des risques de cybersécurité
>
> **Décision demandée.** [Une ou deux phrases : ce qu'il faut approuver, à quel coût, pour quand.]
>
> **Pourquoi maintenant.** [Combien de scénarios de risque dépassent la tolérance avant traitement; les deux ou trois plus importants, en termes d'affaires.]
>
> **Portefeuille recommandé.** [Les initiatives, le coût ponctuel et récurrent total, comparé au contexte budgétaire.]
>
> **Résultat.** [Combien de scénarios sont sous la tolérance, à la tolérance et au-dessus de la tolérance après traitement; quel risque résiduel demeure.]
>
> **Risque résiduel à accepter.** [Quels scénarios, qui doit les accepter formellement — **Décision de la direction requise**.]
>
> **Principales incertitudes et prochaines étapes.** [Les estimations à valider; les éléments probants à recueillir; la date de la prochaine revue.]

## C.3 Déclaration d'utilisation de l'IA générative

> J'ai / Nous avons utilisé [nom et version de l'outil] pour [objectif : par exemple, générer des idées de scénarios de risque, proposer des justifications de paramètres, vérifier la grammaire]. J'ai / Nous avons vérifié tous les résultats par rapport au cas et à des sources primaires, recalculé chaque nombre avec le moteur CyberRiskGuardian, révisé le texte, et j'assume / nous assumons la responsabilité de l'ensemble du contenu.
> [Ou : Aucun outil d'IA générative n'a été utilisé dans ce travail.]

# Annexe D — Réponses aux questions d'autovérification

**Laboratoire 0.** 0.1 Un moteur non vérifié peut calculer différemment de l'implémentation de référence; tout nombre qu'il produit pourrait être erroné sans avertissement. 0.2 Le navigateur conserve les espaces de travail par adresse exacte; `127.0.0.1:8099` et `localhost:8099` sont des adresses différentes. 0.3 **Lacune d'information — validation organisationnelle requise.**

**Laboratoire 1.** 1.1 Tous les services cliniques et administratifs en dépendent; sa compromission ouvre l'accès à tous les autres joyaux de la couronne. 1.2 C'est un fait énoncé dans le cas. Vous ne pouvez pas conclure que les contrôles ont arrêté les tentatives, que les journaux suffisent à prouver qu'elles ont échoué, ni qu'aucune tentative n'a réussi sans être détectée. 1.3 Le centre de données de Montréal et le nœud central du réseau étendu (WAN), ou le service d'identité et d'annuaire.

**Laboratoire 2.** 2.1 Plus prudente. Le risque toléré diminue proportionnellement à l'appétence, de sorte qu'un plus grand nombre de scénarios le dépassent. 2.2 La valeur par défaut n'est pas l'appétence de l'organisation; la laisser sans explication signifie que le test de tolérance n'a aucun fondement. 2.3 Non. Il faut le risque toléré de ce scénario (qui dépend de ses propres paramètres) et, après traitement, le risque résiduel.

**Laboratoire 3.** 3.1 Exemple : « Un acteur externe exploite un emplacement de stockage infonuagique exposé publiquement, utilisé pour le transfert d'images médicales et dépourvu de restrictions d'accès et de journalisation, afin de télécharger des fichiers d'imagerie contenant des identifiants de patients, ce qui cause une atteinte à la vie privée devant être déclarée, des coûts de notification et une perte de confiance des patients. » 3.2 Parce que l'impact pourrait être catastrophique (aucune reprise possible après une attaque destructrice); les scénarios de faible probabilité et à conséquences graves portent un indicateur de remontée et peuvent nécessiter une investigation plutôt qu'une exclusion. 3.3 En général, il faut les fusionner. Ne les gardez distincts que si les actifs, les conséquences, les contrôles, les responsables ou les décisions de traitement diffèrent de façon importante (par exemple, le personnel de la facturation peut détourner des paiements — une conséquence de fraude que le personnel des cliniques ne crée pas).

**Laboratoire 4.** 4.1 Pb(A) indique déjà si la menace agit; si votre « vraisemblance » inclut aussi la réussite de l'attaque, la multiplier par Pb(ψ,A) compte deux fois la même incertitude. 4.2 Posséder des contrôles n'équivaut pas à de la résilience. θ mesure la capacité démontrée de prévenir, détecter, contenir, intervenir, rétablir et poursuivre les activités; le cas fait état d'une équipe de deux personnes, d'un triage faible et d'une reprise non testée. 4.3 Le dommage maximal correspond au pire cas plausible; le cas attendu ne peut pas le dépasser.

**Laboratoire 5.** 5.1 Il augmenterait, de 9,3 à 9,9, parce qu'une attaque qui ne nécessite aucune action de l'utilisateur est plus facile à réaliser. 5.2 Le vecteur montre *pourquoi* le score est ce qu'il est, permet à un réviseur de le recalculer et révèle des erreurs qu'un nombre seul dissimule. 5.3 Exemple : S21, divulgation accidentelle par le personnel — il n'y a aucune vulnérabilité technique à exploiter; tout score ne décrit qu'une faiblesse représentative.

**Laboratoire 6.** 6.1 X a le risque estimé le plus élevé (4,5 fois celui de Y); leurs ratios avant traitement sont égaux, parce que les probabilités s'annulent dans le ratio. 6.2 Les deux sont multipliés par 10; le ratio ne change pas. 6.3 La formule combine des probabilités sans dimension, une gravité de 0 à 10, un dommage et une criticité de 0 à 1 et un facteur d'échelle; c'est un indicateur relatif servant au classement et à la comparaison.

**Laboratoire 7.** 7.1 Zéro, parce que l'atténuation multiplie les deux réductions. L'ensemble de mesures doit aussi réduire l'impact (par exemple, reprise testée, segmentation, détection). 7.2 Attribuez une part à chacun des sept scénarios (par exemple 180 000 $ ÷ 7 ≈ 25 714 $ chacun, plus une part du coût récurrent) pour l'analyse par scénario; comptez 180 000 $ une seule fois dans le total du portefeuille. 7.3 Son effet est déjà intégré à l'état actuel (il influe sur Pb(ψ,A) et θ); en tenir compte de nouveau compte deux fois la même réduction.

**Laboratoire 8.** 8.1 La question de savoir si S1 peut être accepté avec l'ensemble de mesures prévu (cas inférieur et central) ou s'il nécessite un traitement plus rigoureux, un transfert ou une acceptation formelle par le conseil d'administration (cas supérieur). 8.2 Non. Cela signifie que vos estimations sont cohérentes entre elles, et non qu'elles sont exactes. 8.3 θ divise à la fois Re et Rt; il s'annule donc dans le ratio.

**Laboratoire 9.** 9.1 Il n'a ni mesure, ni source, ni responsable, ni fréquence, ni seuil; personne ne peut savoir quand agir. 9.2 Parce qu'un dépassement survient lorsque la valeur s'élève au-dessus du seuil (par exemple, un taux de clics supérieur à 8 %). 9.3 Quatre relevés; un nombre inférieur de points ne permet pas d'établir une tendance, et une droite tracée à partir de trois points donnerait une fausse impression de précision.

**Laboratoire 10.** 10.1 Les membres de la haute direction lisent le premier paragraphe pour décider s'ils liront le reste; la note de service existe pour obtenir une décision. 10.2 Recalculez le classeur, déterminez quel chiffre est exact et corrigez chaque document pour qu'ils concordent; des chiffres incohérents minent l'ensemble de la recommandation. 10.3 Le membre de la haute direction responsable ou le conseil d'administration, selon la délégation de pouvoirs (dans le cas de MediBec, un risque résiduel supérieur à 0,40 normalisé exige l'acceptation du conseil d'administration).

**Laboratoire 11.** 11.1 Non. Une confiance de source est l'affirmation du producteur au sujet de l'indicateur, et non une probabilité que la menace agisse contre MediBec. 11.2 Une confirmation par une personne nommée à une date donnée (*Confirmer comme incident*). 11.3 Une règle ne peut pas détecter des événements qu'elle ne reçoit jamais; la couverture n'est validée que par un test de détection concluant avec une télémétrie fonctionnelle.

**Laboratoire 12.** 12.1 Cela vérifie l'existence, et non le fonctionnement. L'efficacité opérationnelle exige des éléments probants démontrant que les revues ont été effectuées sur une période, sur un échantillon, et que les exceptions ont été résolues. 12.2 Exemple : « Les propriétaires de systèmes doivent imposer l'authentification multifacteur pour tout accès à distance aux systèmes qui traitent des renseignements sur les patients, conformément à la norme d'authentification. » 12.3 Dans la norme d'authentification (ou norme sur les mots de passe), parce que les valeurs techniques changent plus vite que la politique et doivent pouvoir être mises à jour sans qu'il faille approuver de nouveau la politique.

**Laboratoire 13.** 13.1 Non. Les décomptes d'AMF ne s'appliquent qu'aux personnes et aux comptes d'urgence; une identité de service ou d'agent d'IA n'a pas d'AMF, et la cellule reste vide. Une cellule vide est inconnue ou sans objet, jamais 0 %. La faiblesse des comptes de service apparaît ailleurs : 11 sur 74 n'ont pas de propriétaire (constat IR-01). 13.2 Une règle sait seulement qu'un registre révèle une faiblesse; elle ne peut pas savoir dans quelle mesure cette faiblesse modifie l'exploitation pour un scénario donné. La valeur est l'estimation de l'analyste (**Estimation analytique — validation requise**), argumentée à partir des preuves, et son acceptation est un acte distinct, consigné : aucun nombre ne change sans qu'une personne le décide. 13.3 Son exception EXC-90 a expiré le 30 septembre 2026. Une exception expirée ne compense rien : l'écart est marqué *Expirée — non comptée* et compte dans CRG-24 jusqu'à ce que l'exception soit renouvelée ou l'écart corrigé. *Valeurs de référence (1er novembre 2026) : CRG-17 98,27 % (1 479 / 1 505), CRG-18 78,72 % (74 / 94), CRG-19 83,33 % (30 / 36), CRG-20 85,06 %, CRG-21 0 %, CRG-23 80,6 %; CRG-22 76,1 % (1 108 / 1 456), CRG-24 1, CRG-25 86,67 %.*

# Annexe E — Glossaire

**Appétence pour le risque** — le niveau de risque de cybersécurité que la direction est disposée à rechercher ou à conserver, sur une échelle de 0 à 1.
**Chaîne causale** — la séquence source de la menace → faiblesse → actif → événement → conséquence qui définit un scénario de risque.
**Joyaux de la couronne (actifs critiques)** — les actifs et services dont la perte nuirait le plus à la mission.
**Score de base CVSS v4.0 (CVSS-B)** — le score de gravité technique d'une vulnérabilité, de 0 à 10.
**Risque estimé** — le risque d'un scénario de risque avant traitement, tel que calculé par la formule CRG.
**ICR** — indicateur clé de risque : une valeur mesurée assortie de seuils qui signalent une variation du risque.
**Risque atténué** — la part du risque estimé éliminée par le traitement.
**Risque résiduel** — le risque qui demeure après le traitement.
**Résilience organisationnelle (θ)** — la capacité démontrée de prévenir, détecter, contenir, intervenir, rétablir et maintenir les opérations critiques.
**Risque toléré** — l'appétence appliquée à un scénario de risque précis au moyen de la formule CRG.
**Contrôle partagé** — une initiative unique qui réduit plusieurs scénarios de risque; son coût est compté une seule fois dans le portefeuille.
**Compte d'urgence** — un compte réservé aux situations où l'accès normal fait défaut; chaque utilisation doit être examinée.
**Configuration de référence** — une configuration minimale sécuritaire, approuvée et versionnée, pour une plateforme; s'en écarter est un **écart**.
**Constat (identités, configuration)** — une faiblesse qu'une règle détecte dans un registre; il donne un sens à Pb(ψ,A) ou à θ, jamais une valeur.
**Privilège permanent** — un accès privilégié détenu en permanence, par opposition à un accès accordé juste-à-temps.
**Registre au niveau des populations** — un registre qui contient des décomptes par groupe (comptes, niveaux, périmètres), jamais de noms ni d'identifiants.

*© 2026 Marc-André Léger. Sous licence CC BY-NC 4.0. MediBec est une organisation fictive. Toutes les valeurs des paramètres sont des estimations analytiques fournies à titre illustratif à des fins pédagogiques.*
