# CyberRiskGuardian 2.0: 10-minute demo script

Recording plan for a screen-capture demo with voice-over. Record once in English and once in French, using the same takes on screen. Suggested setup: Claude Code in a terminal at 1280×800 or larger, font size 16, the MediBec profile (`plugins/cyberriskguardian/evals/_shared/medibec_profile.md`) ready in the working folder, and the plugins already installed except where the script installs them.

Total running time: about 10 minutes.

## 0:00–0:45 · The problem

**On screen:** title card "CyberRiskGuardian · scenario-driven cyber risk assessment".

**EN:** Most cyber risk assessments start with a list of threats and end with a heat map nobody acts on. CyberRiskGuardian starts with the organization, builds risk scenarios as causal chains, scores them against the risk the organization says it can tolerate, and ends with the decisions management has to take.

**FR :** La plupart des évaluations des risques commencent par une liste de menaces et se terminent par une carte de chaleur dont personne ne se sert. CyberRiskGuardian part de l'organisation, construit des scénarios sous forme de chaînes causales, les compare au risque que l'organisation accepte de tolérer, et se termine par les décisions que la direction doit prendre.

## 0:45–1:45 · Install

**On screen:** type the two install commands from the quick-start guide; restart; approve `crg-calculator`.

**EN:** Core and the governance textbook are free plugins. Two commands install them. The calculator runs locally; nothing leaves your machine.

**FR :** Core et le manuel de gouvernance sont des plugiciels gratuits. Deux commandes suffisent. Le calculateur fonctionne localement; rien ne quitte votre ordinateur.

## 1:45–3:00 · Context and appetite

**On screen:** `/crg-assess MediBec` with the profile attached. Scroll through the evidence register and the crown-jewel table. Pause on an **Information gap** marker.

**EN:** Claude reads the profile first and builds an evidence register. Every fact points to a source. Where the profile is silent, it says so instead of guessing: that's the information-gap marker. Then it proposes a risk appetite of 0.30 for a healthcare organization and explains why. It's marked as an estimate: management confirms it.

**FR :** Claude lit d'abord le profil et constitue un registre des preuves. Chaque fait renvoie à une source. Là où le profil est muet, il le dit au lieu de deviner : c'est le marqueur de lacune d'information. Il propose ensuite un appétit pour le risque de 0,30 pour un organisme de santé et explique pourquoi. C'est une estimation : la direction la confirme.

## 3:00–5:00 · Scenarios

**On screen:** the 20 candidates; zoom on one causal chain; then the 10 selected with rationale.

**EN:** A scenario is not "ransomware". It's a chain: who attacks, through which weakness, which asset, what event, and what it does to the organization. Claude screens 20 candidates on 12 criteria and keeps 10. It never ranks on CVSS, and it tells you which selections rest on judgement.

**FR :** Un scénario, ce n'est pas « rançongiciel ». C'est une chaîne : qui attaque, par quelle faiblesse, quel actif, quel événement, et ce que cela fait à l'organisation. Claude évalue 20 candidats selon 12 critères et en retient 10. Il ne classe jamais selon CVSS, et il indique quelles sélections reposent sur le jugement.

## 5:00–7:00 · Quantification

**On screen:** the parameter table for one scenario; the calculator call; the results table with classifications.

**EN:** Each scenario gets six parameters, each with a rationale and a confidence level. The numbers come from the calculator, not from the model's head. Estimated risk is compared with tolerated risk: above, at, or below tolerance. The ratio depends on damage, treatment and appetite; the probabilities change the ranking, not the verdict.

**FR :** Chaque scénario reçoit six paramètres, chacun avec une justification et un niveau de confiance. Les chiffres viennent du calculateur, pas de la mémoire du modèle. Le risque estimé est comparé au risque toléré : au-dessus, au niveau ou en dessous de la tolérance. Le ratio dépend des dommages, du traitement et de l'appétit; les probabilités changent le classement, pas le verdict.

## 7:00–8:15 · Summary and markers

**On screen:** the management summary; highlight the three markers; save the summary as a file and show the label-check warning on a version with a marker removed.

**EN:** The summary opens with the conclusion and ends with the decisions management must take. A built-in check makes sure estimates, gaps and decisions are always labelled. Remove a marker, and it warns you.

**FR :** Le résumé commence par la conclusion et se termine par les décisions de la direction. Une vérification intégrée s'assure que les estimations, les lacunes et les décisions sont toujours signalées. Retirez un marqueur, et elle vous avertit.

## 8:15–9:15 · Governance textbook

**On screen:** ask "What is the three lines model?"; show the chapter citation.

**EN:** The textbook plugin answers governance questions from the book, with the chapter, so students and analysts can go to the source.

**FR :** Le plugiciel du manuel répond aux questions de gouvernance à partir du livre, en citant le chapitre, pour que les étudiants et les analystes puissent remonter à la source.

## 9:15–10:00 · CRG Pro and licences

**On screen:** the licence table from www.leger.ca.

**EN:** Core is free for students and non-profits. CRG Pro adds the costed treatment plan, sensitivity analysis, the cybersecurity budget, KRIs, the Excel workbook and the board report, under a commercial licence or, for colleges and universities, an educational licence. Details at leger.ca.

**FR :** Core est gratuit pour les étudiants et les organismes sans but lucratif. CRG Pro ajoute le plan de traitement chiffré, l'analyse de sensibilité, le budget de cybersécurité, les indicateurs de risque clés, le classeur Excel et le rapport au conseil, sous licence commerciale ou, pour les collèges et universités, sous licence éducative. Détails sur leger.ca.

## Recording notes

- Use the fictional MediBec case only; never show real organizational data.
- Run the scenario steps once before recording so the takes are predictable; Claude's wording varies between runs.
- Cut waiting time between tool calls in editing; the narration timings assume edited footage.
