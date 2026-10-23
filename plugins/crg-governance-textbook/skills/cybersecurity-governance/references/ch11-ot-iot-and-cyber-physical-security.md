<!-- Léger, M.-A. (2026). Introduction to Cybersecurity Governance for Business Technology Management (3rd ed., v2.51). Chapter 11: OT, IoT, and Cyber-Physical Security. CC BY-NC 4.0 -->

# Chapter 11: OT, IoT, and Cyber-Physical Security

## Learning outcomes

After completing this chapter, you should be able to:

1.  Define operational technology, the Internet of Things, and cyber-physical systems, and apply a consequence-based scope test to decide which systems a governance program must include.

2.  Explain how OT and IoT risk differs from enterprise IT risk in priorities, lifecycles, ownership, threats, and physical consequences.

3.  Compare the main standards and regulatory regimes, including ISA/IEC 62443, NIST SP 800-82, the EU Cyber Resilience Act, NIS2, Canada's CCSPA, and medical device rules, and explain what each requires of managers.

4.  Design an operating model that assigns ownership and decision rights across operations, engineering, cybersecurity, procurement, and vendors before an incident occurs.

5.  Recommend procurement, architecture, and patch and change controls, including zones and conduits, least privilege, compensating controls, and exceptions with expiry.

6.  Evaluate how cybersecurity must protect the safety case and how detection, response, and recovery must be adapted to degraded physical operations.

7.  Assess the governance boundaries required before AI-enabled systems are allowed to influence physical operations.

8.  Apply key risk indicators and the scenario method to an OT or IoT environment and justify risk decisions against a stated risk appetite.

Operational technology, the Internet of Things, and cyber-physical systems take cybersecurity beyond the protection of information. They add the protection of physical operations, asset integrity, human safety, service continuity, and public trust. In these environments, a cybersecurity incident may do more than expose data or interrupt a digital service. It may stop a production line, damage equipment that took months to procure, degrade water or power distribution, disrupt a hospital, release a pollutant, or endanger the people who work near the process. The governance question changes accordingly: it is no longer only "what information could we lose?" but also "what could physically happen, to whom, and how quickly?"

In most organizations, OT and IoT environments coexist with enterprise IT but are purchased, maintained, staffed, and measured differently. That divergence is one of the main reasons these environments remain chronically vulnerable. They tend to fall between ownership boundaries. IT and cybersecurity teams understand threats and controls but often lack authority over industrial, facilities, laboratory, or embedded systems. Engineering, operations, and facilities teams understand the equipment intimately but may lack cybersecurity training, monitoring tools, or incentives. Vendors and integrators frequently control maintenance access. Devices are deployed at scale by business units without a reliable inventory, a patching plan, or any thought for end of life. Each group acts reasonably within its own mandate, and the organization as a whole ends up with risks that nobody owns.

For managers, the challenge is not to become experts in industrial protocols, ladder logic, embedded firmware, or sensor engineering. It is to direct and oversee a class of systems in which cyber events have physical consequences. A compromised email account is serious; a compromised chemical dosing controller, infusion pump fleet, or remote pumping station changes how the organization functions in the physical world. The risk extends from the confidentiality, integrity, and availability of information to the safety, reliability, controllability, and resilience of operations. That is why this chapter treats OT, IoT, and cyber-physical security as a **governance domain** rather than a technical specialty.

These environments also impose constraints that conventional IT rarely faces. Assets have lifecycles of fifteen to thirty years, limited computing capacity, proprietary architectures, vendor-controlled maintenance models, sparse logging, and little support for modern security controls. Some cannot be patched without stopping a process that runs around the clock. Others were designed for reliability in isolated settings and were later connected to corporate networks, cloud platforms, and remote-support tools without being redesigned for adversarial conditions. Improving security therefore requires managerial decisions about planned downtime, capital replacement, vendor renegotiation, segmentation projects, monitoring investment, maintenance windows, and compensating controls. Those decisions involve explicit trade-offs among uptime, safety, cost, compliance, and **risk appetite**.

This chapter builds on earlier material. The risk vocabulary of Chapter 4, the scenario method of Chapters 5 and 6, the operations and vulnerability practices of Chapters 7 and 8, the control frameworks of Chapter 9, and the identity disciplines of Chapter 10 all apply here, but each must be adapted to an environment where safety and availability usually outrank confidentiality. Incident management and continuity are treated in depth in Chapter 12, security architecture in Chapter 13, baselines and configuration governance in Chapter 14, supplier governance in Chapter 19, and the cybersecurity of AI in Chapter 18; this chapter points to them rather than repeating them.

The chapter proceeds as follows. It first defines what managers should include in scope and explains how OT and IoT risk differs from IT risk. It then surveys the standards and regulatory landscape that now shapes the field, from the ISA/IEC 62443 series and NIST SP 800-82 Rev. 3 to the EU Cyber Resilience Act, NIS2, Canada's new critical infrastructure legislation, and medical device requirements. The middle of the chapter addresses the governance building blocks: inventory and criticality, the operating model, procurement, architecture, patch and change governance, and the interplay of cybersecurity with safety engineering. It closes with detection, response and resilience, AI-enabled cyber-physical systems, metrics and an illustrative scenario set in a Québec water utility, and a synthesis of the chapter's lessons.

## Definitions and scope: what managers should include in OT, IoT, and CPS

Terminology varies across industries, vendors, and regulators, but a governance program needs clear boundaries. Managers need not master every technical distinction. They must, however, understand which systems fall within scope, why those systems carry different risks than ordinary enterprise IT, and who should be accountable for them. A scope that is too narrow leaves dangerous blind spots; a scope that is too broad treats a conference-room display with the same seriousness as a chlorine dosing system and wastes scarce attention.

### Operational technology

**Operational technology (OT)** refers to hardware and software that monitor, control, or automate physical processes, devices, and infrastructure. NIST SP 800-82 Rev. 3, *Guide to Operational Technology (OT) Security* (September 2023), deliberately broadened the earlier term "industrial control systems" to OT to reflect how widely such systems are now found. The category includes supervisory control and data acquisition (SCADA) systems that manage geographically dispersed assets such as pipelines and water networks; distributed control systems (DCS) used in continuous process plants; programmable logic controllers (PLCs) and remote terminal units; human–machine interfaces (HMIs) and engineering workstations; safety instrumented systems (SIS) that bring a process to a safe state when limits are exceeded; building automation and energy management systems; laboratory and test equipment; and physical access control systems.

OT is common in manufacturing, energy, water and wastewater, transportation, food processing, pharmaceuticals, mining, healthcare facilities, universities, and warehouses. Its primary objective is the reliable and safe operation of a physical process, not the efficient processing of information. That single difference explains much of what follows: an OT engineer who resists an "urgent" patch is usually not being obstructive but protecting a process whose failure can hurt people or equipment.

### The Internet of Things and the industrial IoT

**The Internet of Things (IoT)** refers to network-connected devices that combine at least one sensor or actuator with network connectivity, are often embedded, and are typically deployed in large numbers. Examples include IP cameras, badge readers, smart thermostats and lighting, environmental and occupancy sensors, connected medical devices, point-of-sale peripherals, fleet telematics, asset trackers, smart meters, and consumer-grade devices that employees bring into the workplace. The **industrial IoT (IIoT)** applies the same pattern to operations: wireless vibration sensors on pumps, cloud-connected compressors, and edge gateways that stream plant data to analytics platforms.

Some IoT devices are operationally critical; others are mainly conveniences. Both categories create risk because unmanaged devices become entry points, surveillance points, data-exposure points, or bridges into more sensitive networks. IoT devices also frequently collect personal information—video, location, occupancy, biometric, or health data—which brings privacy law into scope. In Québec, Law 25 requires a privacy impact assessment for any project to acquire, develop, or overhaul an information system or electronic service delivery system involving personal information, which squarely includes a building-wide camera or occupancy-sensor deployment (see Chapter 16).

### Cyber-physical systems

**Cyber-physical systems (CPS)** are integrated systems in which software, sensors, networks, and computing components govern physical behaviour, usually through feedback loops: sense, compute, act, and sense again. OT environments are, in most cases, a subset of CPS. The broader category also includes autonomous mobile robots, automated warehouses, connected vehicles and rail signalling, smart grids, surgical robots, drones, and AI-enabled physical systems. The defining feature is that digital compromise or failure can change physical action, physical conditions, or physical outcomes.

From a governance perspective, the label matters less than the risk profile. A pragmatic manager does not argue about whether a networked chiller is "OT" or "IoT"; the manager asks what happens if it stops, lies, or is commanded by someone else.

### Drawing the scope boundary

Managers should include in scope any device, system, platform, or integration whose compromise could disrupt operations, compromise safety, damage equipment, harm the environment, expose sensitive information, weaken physical security, or create an unmanaged path into the enterprise—including vendor-managed equipment, remote maintenance connections, and technologies deployed directly by facilities or business units.

A practical scope test asks a short series of questions:

- **Physical influence.** Does the system monitor or control a physical process, environment, or piece of equipment?

- **Consequence.** Could compromise or failure affect safety, service continuity, production, facilities, the environment, regulatory standing, or public trust?

- **Connectivity.** Is the device connected to the enterprise network, the Internet, a cellular modem, a vendor cloud, or a remote-access tool?

- **Ownership.** Is it managed by IT, engineering, facilities, a supplier, or by no clearly accountable owner?

- **Manageability.** Does it have known weaknesses in lifecycle support, patching, logging, authentication, or access control?

- **Data.** Does it collect or transmit personal, health, or commercially sensitive information?

A "yes" to the first two questions usually places a system in the high-consequence tier. A "yes" to the connectivity and ownership questions without clear answers usually signals the most urgent governance gap. The figure below summarizes the categories managers should include.

<span id="_Toc243635552" class="anchor"></span>Figure 53: What managers should include in OT, IoT, and CPS scope

The managerial lesson is that OT, IoT, and CPS governance begins with visibility. Organizations cannot govern what they have not identified, classified, assigned to an owner, and linked to business consequences. A mature program defines scope broadly enough to capture operational and physical consequences, yet precisely enough to assign ownership, prioritize controls, and avoid treating every connected device as equally important.

## How OT and IoT risk differs: priorities, constraints, and risk types

Traditional enterprise IT security emphasizes the confidentiality of information, the integrity of records, and the availability of business applications. Downtime is costly, but in most IT contexts recovery is measured in hours and the main consequences are financial loss, productivity disruption, privacy breaches, reputational harm, or regulatory exposure. OT and IoT environments change this profile in ways that have direct governance implications.

### Different priorities: safety and availability first

In operational environments, the safety of people, the availability of the process, and the integrity of control usually take precedence over confidentiality. Tolerable downtime may be measured in seconds or minutes, and consequences may include equipment damage, product spoilage, environmental releases, safety incidents, or interruption of essential public services. NIST SP 800-82 Rev. 3 stresses that OT systems have performance, reliability, and safety requirements that differ fundamentally from IT, and the joint international guidance *Principles of Operational Technology Cyber Security* (2024), endorsed by the Canadian Centre for Cyber Security, puts it bluntly in its first principle: **safety is paramount**.

These differences do not mean confidentiality is irrelevant. Process recipes, network diagrams, engineering drawings, and safety system configurations are valuable to attackers precisely because they reveal how to cause harm; the same joint guidance identifies OT data as something that "needs to be protected." But the ordering of priorities shapes almost every control decision, from how aggressively a network may be scanned to who can authorize isolating a system during an incident. The figure below contrasts the two perspectives.

<span id="_Toc243635553" class="anchor"></span>Figure 54: How IT and OT priorities differ

### Lifecycles, legacy, and the limits of patching

OT assets are bought as capital equipment, depreciated over decades, and often certified or validated for a specific configuration. A controller installed in 2008 may still run a critical process in 2026, on an operating system its vendor stopped supporting years ago. Patching may require a planned shutdown, vendor supervision, revalidation, or regulatory notification. Many devices cannot run endpoint agents, cannot tolerate active scanning, and produce few useful logs.

IoT adds scale and heterogeneity. A single organization may operate thousands of devices from dozens of manufacturers, across many firmware versions, with weak default configurations and inconsistent update mechanisms. Some devices are inexpensive and easy to deploy but difficult to manage over time; others quietly become operationally important even though they were purchased as conveniences. The result is a broad attack surface that ordinary IT asset management often does not see. "We cannot patch it" is a description of a constraint, not a governance answer; it must lead to an explicit risk decision, compensating controls, and a plan.

### Fragmented ownership and a different threat landscape

OT and IoT may be owned by engineering, manufacturing, facilities, biomedical engineering, physical security, laboratories, or vendors rather than by central IT. This fragmentation means that the people who understand the threat and the people who control the equipment are rarely the same, and governance must bridge them deliberately (see the operating model section below).

The threat landscape also differs. OT environments face opportunistic ransomware that spreads from IT and forces precautionary shutdowns, targeted sabotage by capable state-linked actors, insider misuse, compromise of vendors and integrators, and accidental misconfiguration. IoT environments face mass exploitation of default credentials, weak update mechanisms, insecure vendor cloud back ends, and exposed management interfaces. Canadian organizations are not exempt: in Alert AL25-016 (October 2025), the Canadian Centre for Cyber Security reported that hacktivists had tampered with Internet-accessible industrial control systems, including manipulating water pressure values at a water facility, an automated tank gauge at an oil and gas company, and temperature and humidity levels in a farm grain-drying silo. MITRE's ATT&CK for ICS knowledge base, updated in ATT&CK v20 (October 2026), catalogues adversary tactics and techniques specific to industrial environments and now includes campaign entries drawn from attacks on water utilities.

| In OT and IoT, cyber risk is measured in physical consequences—**harm to people, equipment, the environment, and essential services**—and not only in lost or exposed data. |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### A manager's taxonomy of cyber-physical risk

Although technical details vary by sector, a recurring set of risk types appears across OT, IoT, and CPS environments. Naming them helps managers link technical compromise to operational, safety, financial, regulatory, and reputational consequences, and it gives risk scenarios (Chapter 5) a consistent vocabulary.

- **Operational disruption and extortion.** Ransomware, destructive malware, or loss of control systems can halt production, disable building systems, disrupt logistics, or interrupt utilities. Recovery is not only about restoring servers; it means re-establishing trusted control of a physical process, which may require validating equipment states, checking control logic, and verifying safety functions before restart.

- **Integrity manipulation.** Attackers may alter setpoints, sensor readings, control logic, recipes, calibration data, or alarms. The system may keep running while producing unsafe, off-specification, or misleading results, so integrity attacks can remain hidden until physical damage, quality failures, or unsafe conditions appear.

- **Unsafe automation.** A malicious or erroneous command can disable protective functions, cause mechanical collisions, create overpressure or overheating, or trigger unsafe movement. This is where cybersecurity must meet safety engineering and hazard analysis.

- **Unmanaged device fleets.** Large IoT deployments with default passwords, outdated firmware, and reliance on vendor clouds may each look low risk but collectively create a large attack surface, a surveillance exposure, and lateral pathways into sensitive zones.

- **Supplier and remote-access risk.** Vendors, integrators, and maintenance partners often need remote access. Persistent, shared, weakly authenticated, or unmonitored access paths are among the most frequent and highest-impact entry points.

- **Data, analytics, and model abuse.** As predictive maintenance, optimization, digital twins, and AI decision support enter operations, poisoned data or manipulated models can generate unsafe recommendations or mask real problems.

- **Loss of visibility and accountability.** When inventories, logs, ownership, documentation, and configuration baselines are incomplete, management cannot prioritize what it cannot see or assure what it cannot evidence.

A manager-oriented program need not treat these risks equally. A food manufacturer may prioritize production continuity, recipe integrity, and vendor access. A hospital may prioritize medical device safety, environmental controls, and patient-care continuity. A property operator may prioritize access control, surveillance privacy, and vendor-managed platforms. In each case, the risk must be articulated in operational terms: not "can this device be hacked?" but "what happens to our people, services, assets, and obligations if this device, connection, data stream, or automated process can no longer be trusted?" Expressed through the risk triangle of Chapter 4, the impact side of the triangle now includes physical harm, and the vulnerability side is often dominated by lifecycle and ownership weaknesses rather than missing technical controls.

## The standards and regulatory landscape

Until recently, OT and IoT security was largely voluntary outside a few regulated sectors such as electricity. That has changed. A body of international standards, government guidance, and product and infrastructure legislation now shapes what customers, regulators, insurers, and auditors expect. Managers do not need to read these documents cover to cover, but they need to know which ones apply, what kind of obligation each creates, and how they fit with the ISMS and frameworks described in Chapter 9.

### ISA/IEC 62443: the reference series for industrial automation

The **ISA/IEC 62443 series**, *Security for industrial automation and control systems*, is the most widely used international reference for OT security. Its distinctive contribution is that it assigns responsibilities to the three main parties in the OT ecosystem—asset owners, service providers and integrators, and product suppliers—and defines requirements for each. The series is organized in four groups:

- **General (62443-1-x).** Concepts and models, including the ideas of zones, conduits, and security levels (IEC TS 62443-1-1, first published in 2009, is being revised).

- **Policies and procedures (62443-2-x).** IEC 62443-2-1:2024 (edition 2.0) sets security program requirements for asset owners—in effect, the OT counterpart of an ISMS. IEC 62443-2-4 (edition 2.0, 2023) sets requirements for service providers such as integrators and maintenance contractors, and IEC TR 62443-2-3:2015 addresses patch management in the OT environment.

- **System (62443-3-x).** IEC 62443-3-2:2020 describes security risk assessment and system design, including how to partition a system into zones and conduits and assign target security levels. IEC 62443-3-3:2013 specifies system security requirements against seven foundational requirements, from identification and authentication control to resource availability.

- **Component (62443-4-x).** IEC 62443-4-1:2018 sets secure product development lifecycle requirements for suppliers, and IEC 62443-4-2:2019 sets technical security requirements for components such as controllers, network devices, and host software.

The concept of **security levels** (SL 1 to SL 4) is especially useful to managers. It expresses protection in terms of the attacker the system must withstand, ranging from casual or accidental violation up to sophisticated, well-resourced attackers. Setting a target security level for each zone is therefore a risk-appetite decision, not a technical preference. Conformity schemes, including ISASecure and the IECEE scheme, allow products, development processes, and service providers to be certified against parts of the series, which gives procurement teams a source of evidence. Several parts are under revision at the time of writing, so managers should confirm current editions when writing contract clauses.

### NIST SP 800-82 and the NIST IoT guidance

NIST SP 800-82 Rev. 3 (September 2023) is the principal U.S. government guide to OT security. It explains OT architectures and threats, describes how to establish an OT security program, and provides an OT overlay of the NIST SP 800-53 Rev. 5 controls. In September 2026 NIST released an initial public draft of Revision 4 for comment. The draft reorganizes the guidance around NIST CSF 2.0 and its **Govern** function, ties OT risk management more closely to enterprise risk management, integrates zero trust principles adapted to OT, and widens the sectors covered to include building automation, water and wastewater, food and agriculture, and the industrial IoT. Until Revision 4 is final, Rev. 3 remains the current reference.

For IoT, NIST has developed two complementary perspectives. The **NIST IR 8259 series** addresses manufacturers: NIST IR 8259 Rev. 1, *Foundational Cybersecurity Activities for IoT Product Manufacturers* (April 2026), describes what manufacturers should do before and after sale, while NIST IR 8259A and 8259B define core baselines of device technical capabilities (such as device identity, secure update, and data protection) and non-technical supporting capabilities (such as documentation and vulnerability information). The **SP 800-213** publications address organizations that buy and deploy IoT: SP 800-213 (2021) and its requirement catalogue SP 800-213A guide federal agencies in setting device requirements, and both are being revised—a draft SP 800-213 Rev. 1 retitled around IoT *products* was released in June 2026. Even outside government, these documents give managers a ready-made vocabulary for procurement requirements.

### Product security law: the EU Cyber Resilience Act and consumer IoT

The most significant regulatory development for connected products is the **EU Cyber Resilience Act (CRA)**, Regulation (EU) 2024/2847. It applies to "products with digital elements" placed on the EU market, which covers most connected hardware and software, including many industrial and IoT products. It entered into force on December 10, 2024, and applies in stages: provisions on conformity assessment bodies from June 11, 2026; manufacturers' obligations to report actively exploited vulnerabilities and severe incidents from September 11, 2026; and the main obligations—essential cybersecurity requirements, vulnerability handling, a defined support period, technical documentation, and CE marking—from December 11, 2027. Reports go through ENISA's single reporting platform on a 24-hour early warning, 72-hour notification, and final report timeline. Products classed as "important" or "critical" face stricter conformity assessment.

Canadian manufacturers that sell into Europe must comply directly, while buyers benefit indirectly as suppliers make software bills of materials, declared support periods, and vulnerability handling standard practice; Chapter 19 discusses these supply-chain effects. Alongside the CRA, the EU's Radio Equipment Directive cybersecurity delegated act (Delegated Regulation (EU) 2022/30) has applied since August 1, 2025 to Internet-connected radio equipment, supported by the EN 18031 harmonised standards. For consumer IoT, **ETSI EN 303 645** (current version V3.1.3, 2024) sets baseline provisions such as no universal default passwords, a vulnerability disclosure policy, and keeping software updated; it underpins the UK's product security regime and many certification labels. Managers buying devices for workplaces can use it as a minimum floor even when the devices are marketed for consumers.

### Critical infrastructure regulation: NIS2, Canada's CCSPA, and sector rules

The EU **NIS2 Directive**, Directive (EU) 2022/2555, extends cybersecurity risk-management and incident-reporting obligations to essential and important entities in sectors that include energy, transport, health, drinking water, wastewater, manufacturing of medical devices and other products, and food. Member states were required to transpose it by October 17, 2024, although many did so late; in May 2025 the European Commission sent reasoned opinions to 19 member states for incomplete transposition. NIS2 is notable for making management bodies responsible for approving and overseeing cybersecurity measures, which reinforces the governance theme of this book.

In Canada, Bill C-8 received Royal Assent in June 2026, enacting the **Critical Cyber Systems Protection Act (CCSPA)** and amending the *Telecommunications Act*. The CCSPA will require designated operators in federally regulated sectors—finance, telecommunications, energy, and transportation—to establish cybersecurity programs, manage supply-chain and third-party risks, and report significant incidents. Its provisions are being brought into force in phases, so organizations should follow the regulations as they are published. Much OT, however, sits outside federal jurisdiction. Municipal water systems, food processors, and most manufacturers are not covered by the CCSPA, and the Cyber Centre's AL25-016 alert explicitly encouraged provinces and territories to coordinate with municipalities and organizations in sectors such as water, food, and manufacturing where cybersecurity is not covered by regulation. Sector rules also apply; bulk electric system operators, for example, are generally subject to the NERC Critical Infrastructure Protection (CIP) reliability standards as adopted by provincial regulators.

### Medical devices: FDA and Health Canada

Connected medical devices are cyber-physical systems in which compromise can affect patient safety directly. In the United States, section 524B of the Federal Food, Drug, and Cosmetic Act, in force since March 2023, requires manufacturers of "cyber devices" to include cybersecurity information in premarket submissions, including a plan for monitoring and addressing post-market vulnerabilities and a software bill of materials (SBOM). FDA's guidance *Cybersecurity in Medical Devices: Quality Management System Considerations and Content of Premarket Submissions* was most recently reissued on February 3, 2026, aligning it with the agency's new Quality Management System Regulation. In Canada, Health Canada's *Pre-market Requirements for Medical Device Cybersecurity* guidance (effective June 2019) expects manufacturers of software-containing devices to address secure design, risk management, testing, and a plan for the device's service life, and asks for a cybersecurity bill of materials in labelling.

For hospital managers, the practical implication is that procurement can and should demand this evidence. A hospital does not design devices, but it decides which devices it buys, how they are networked, and who is accountable for them once installed.

### Government guidance and threat knowledge

Government agencies have produced a substantial body of free, practical OT guidance, often jointly. The Canadian Centre for Cyber Security publishes *Protect your operational technology* (ITSAP.00.051) and has co-sealed international guidance including the *Principles of Operational Technology Cyber Security* (2024), *Foundations for OT Cybersecurity: Asset Inventory Guidance for Owners and Operators* (August 2025), and *Secure Connectivity Principles for Operational Technology* (January 2026). CISA and partners published *Secure by Demand: Priority Considerations for Operational Technology Owners and Operators when Selecting Digital Products* (January 2025), which lists security features OT buyers should prioritize when selecting products. MITRE ATT&CK for ICS supports threat-informed defence by mapping adversary techniques to detections and mitigations.

The table below summarizes how managers can use the main references.

| **Reference**                  | **Type**            | **Main audience**                    | **Managerial use**                          |
|--------------------------------|---------------------|--------------------------------------|---------------------------------------------|
| ISA/IEC 62443 series           | Consensus standard  | Asset owners, integrators, suppliers | Program, zone, and procurement requirements |
| NIST SP 800-82 Rev. 3          | Government guide    | OT asset owners                      | Program design and control overlay          |
| NIST IR 8259 / SP 800-213      | Government guide    | IoT makers and buyers                | Device capability requirements              |
| EU CRA and ETSI EN 303 645     | Law and standard    | Product manufacturers                | Supplier evidence and support periods       |
| NIS2 and Canada's CCSPA        | Legislation         | Critical infrastructure operators    | Board accountability and reporting          |
| FDA and Health Canada guidance | Regulatory guidance | Medical device makers                | Hospital procurement evidence               |
| CCCS and CISA joint guidance   | Government guidance | Owners and operators                 | Practical priorities and baselines          |

The managerial lesson is that standards and laws are not substitutes for risk judgment, but they define what "reasonable" now looks like. An organization that cannot show how its OT program relates to ISA/IEC 62443 or NIST SP 800-82, or that buys connected devices without asking for the evidence the CRA and FDA now require of manufacturers, will find its decisions harder to defend.

## Governance foundations: inventory, criticality, and what must not fail

The visibility principle is easy to state and hard to achieve. Many organizations lack a reliable inventory of their operational systems and connected devices, particularly when local teams buy equipment independently, facilities teams manage building technologies, vendors install devices as part of service contracts, or engineering teams maintain assets outside IT processes. This invisibility is a governance failure, not just a technical inconvenience, because risk cannot be owned, prioritized, monitored, or accepted if the relevant assets are unknown.

### Building an inventory of assets and dependencies

An effective program maintains an asset and dependency inventory tailored to OT, IoT, and CPS. The joint *Foundations for OT Cybersecurity: Asset Inventory Guidance* (2025) recommends going beyond a list of devices to build a structured, prioritized taxonomy of assets with the attributes that matter for risk decisions. For managers, the inventory should be able to answer the following:

- **What exists and where.** Device types, models, locations, and the process or service each one supports.

- **Who is accountable.** A business owner who accepts the risk and a technical custodian who maintains the asset.

- **How it is supported.** Vendor, support contract, firmware or software version where feasible, and announced end-of-support dates.

- **How it connects.** Network zone, communication paths, remote-access pathways, wireless and cellular links, and dependencies on cloud or vendor services.

- **What it depends on.** Upstream and downstream systems, utilities, spare parts, and specialist skills.

Building such an inventory in OT requires care. Aggressive active scanning can disrupt fragile devices, so organizations typically combine passive network monitoring, configuration file review, physical walk-downs, procurement and maintenance records, and vendor documentation. The inventory is never finished; it must be maintained through change management and procurement so that new devices enter it by default rather than by audit. Chapter 14 discusses configuration governance, which depends on this foundation.

### Criticality mapping and consequence

The next step is criticality mapping. In IT, criticality is often defined by data sensitivity or application importance. In OT and CPS, criticality is defined primarily by **consequence**: which systems, if disrupted, manipulated, or rendered untrustworthy, could cause safety incidents, environmental harm, production shutdowns, interruption of essential services, major financial loss, regulatory enforcement, or loss of public confidence. This consequence-driven approach starts with the worst credible outcomes and works backward to the systems that could produce them.

Criticality mapping directly informs governance. Systems that support safety functions, essential operations, regulated processes, or major revenue should receive stronger minimum controls, stricter change management, tighter vendor-access governance, more monitoring, and more rigorous recovery planning. Lower-criticality devices can be governed with proportionate controls, provided they cannot become pathways into higher-criticality zones. Criticality tiers also become the basis for procurement requirements, patching timelines, and the target security levels discussed later.

### Defining what must not fail

Managers should also define "what must not fail" in operational terms. This serves as the continuity anchor for the entire program. The organization should identify the processes, safety functions, control capabilities, manual procedures, and decision points that must remain reliable even under cyber stress. A manufacturer may need safe shutdown capability above all; a hospital may need environmental controls, medical gases, and access to critical clinical equipment; a logistics operation may need warehouse automation, fleet visibility, and access control; a water utility must keep treatment within regulatory limits and maintain distribution pressure.

This definition shapes architecture and investment. It informs segmentation priorities, backup and recovery strategies, manual fallback procedures, monitoring requirements, vendor support expectations, spare-parts planning, crisis communication, and escalation thresholds. It also helps executives understand that some systems require protection not because they hold sensitive data, but because the organization cannot operate safely or lawfully without them. **OT security cannot begin with tools; it begins with knowing what exists, who owns it, how it connects, how critical it is, and what must keep working under adverse conditions.**

## Operating model and accountability: bridging IT, engineering, facilities, and vendors

OT, IoT, and cyber-physical security programs frequently fail because accountability is unclear. OT engineers own uptime, reliability, and safety but may not see cybersecurity as part of their mandate. The CISO owns cybersecurity policy but may lack authority to change plant equipment, building systems, laboratory devices, or controllers. Facilities teams procure connected systems without cybersecurity review. Vendors retain privileged maintenance access without monitoring or contractual control. In this environment, policy cannot be executed unless the organization defines an explicit operating model.

### Four ownership layers

A mature operating model clarifies who owns risk, who sets requirements, who operates controls, who approves exceptions, and who provides assurance. IEC 62443-2-1:2024 frames these as elements of an asset owner's security program; the managerial point is that no single team can carry them alone. At minimum, the model should define four ownership layers:

- **Business and operational owners are accountable for operational risk in their domains.** They understand which processes are critical, what downtime means, which safety or service consequences are unacceptable, and which trade-offs are feasible. They must take part in risk acceptance, especially where cybersecurity controls could affect uptime, safety, production, patient care, or service continuity.

- **OT, engineering, and facilities leadership are accountable for operational integrity and safe operating conditions.** This includes the cyber-relevant aspects of control logic, configuration, maintenance practices, change windows, device deployment, physical access to equipment, and vendor activities. These teams understand the consequences of technical change better than central IT, so their participation is essential.

- **Cybersecurity and IT leadership define minimum requirements and provide expertise.** They set reference architectures, segmentation principles, identity and access rules, monitoring expectations, incident-response interfaces, and assurance activities tailored to OT constraints. The security function must act as a design partner rather than an external critic that arrives only to identify gaps.

- **Procurement and vendor management are accountable for contractual control.** They ensure that contracts and purchasing processes embed remote-access conditions, authentication, patching and upgrade expectations, logging, vulnerability and incident notification, end-of-life commitments, and audit or assurance rights.

Many organizations formalize this through an OT security lead or "OT CISO" role that sits between engineering and cybersecurity, with a dotted line to both. The title matters less than the mandate: someone must be able to speak the language of process safety and of threat intelligence, and must have the authority to convene both.

### Decision rights before the crisis

The operating model must be expressed through explicit **decision rights** that are agreed before an incident. The organization should be able to answer, in writing, questions such as the following:

- Who approves remote access for an OT vendor, and for how long?

- Who may authorize emergency isolation of an OT network or a cell of a plant during an attack, and who must be consulted first?

- Who decides whether a patch is deferred because of uptime or safety constraints, and who accepts the resulting residual risk?

- Who accepts the risk of an unsupported controller that cannot be replaced this budget year?

- Who is accountable when a business unit deploys IoT devices that create an intrusion path?

- Who determines when operations can safely resume after an incident?

Without explicit answers, organizations default to delay, informal negotiation, and diffusion of responsibility. In normal operations this creates unmanaged exceptions and inconsistent controls. During incidents it creates confusion about authority and acceptable operational impact: the security team wants to isolate a system, operations fears production loss, a vendor demands remote access that no one can verify, and executives demand rapid recovery while engineers insist on careful validation. A RACI matrix for these decisions, approved by the executive risk committee, is one of the most valuable and least expensive artifacts an OT program can produce.

### Governance routines and the three lines

A mature operating model also needs routines: an OT and IoT risk committee or working group, joint change-review procedures, periodic vendor-access reviews, reviews of critical-asset exceptions, cross-functional exercises, and regular reporting to the executive risk committee and board. In the three lines model, operational and engineering teams together with cybersecurity form the first line that owns and operates controls; enterprise risk management, safety, and compliance provide second-line oversight and challenge; and internal audit provides independent assurance, ideally with access to OT expertise. The purpose is not bureaucracy. It is to ensure that cybersecurity decisions affecting physical operations are made by people with the authority and the information to make them, and that those decisions leave evidence.

| Good OT governance makes ownership visible, decision rights explicit, exceptions reviewable, and accountability enforceable—**before** the first incident tests them. |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Policy and procurement: controlling risk before devices are deployed

In OT and IoT environments, governance leverage is highest before devices are purchased, connected, and embedded into operations. After deployment, options are constrained by vendor design, proprietary architectures, installation realities, production schedules, safety validation, and the cost of disruption. A device that cannot be patched, monitored, authenticated, segmented, or retired safely may create risk for twenty years. **Procurement is therefore not merely a purchasing function; it is a primary cybersecurity control.** The CISA-led *Secure by Demand* guidance for OT owners makes the same argument: buyers who ask for security features consistently change what suppliers build.

### Tiered minimum requirements

Managers should define minimum cybersecurity requirements for OT and IoT purchases, tiered by criticality. A safety-related controller, medical device, access-control system, or production-line robot should not be held to the same requirements as a low-risk convenience device. Criticality should determine the expected strength of authentication, logging, update capability, vendor support, segmentation compatibility, and assurance evidence. Existing references make this easier: IEC 62443-4-2 component requirements and target security levels for industrial products, NIST IR 8259A capabilities and SP 800-213A for IoT devices, ETSI EN 303 645 for consumer-grade devices, and FDA or Health Canada submissions for medical devices.

Typical minimum requirements include the following:

- **No default or shared credentials.** Unique credentials per device, the ability to change them, and integration with central identity where feasible.

- **Secure, supported updates.** Authenticated update mechanisms, a declared support period, and vendor commitments on vulnerability remediation and notice before end of support.

- **Hardening and segmentation compatibility.** The ability to disable unnecessary services, use secure protocols, and operate behind firewalls or in defined zones without breaking functionality.

- **Logging and visibility.** Event export or at least network behaviour that monitoring tools can observe.

- **Transparency.** An SBOM, a vulnerability disclosure policy, and, for industrial products, evidence of a secure development lifecycle such as IEC 62443-4-1 certification.

### Lifecycle manageability and total cost

Procurement should also evaluate lifecycle manageability. The organization should ask whether the device can be inventoried, configured securely, updated, monitored, backed up, recovered, and decommissioned without leaving data behind. A device that appears inexpensive at purchase can become costly if it requires manual administration, vendor-only maintenance, insecure remote access, or replacement before the end of its operational life because the vendor stops supporting it. Total cost of ownership should therefore include security maintenance, monitoring, patching, support, and eventual secure disposal. Chapter 17 discusses how to frame these costs in investment decisions.

Architecture review should be built into the purchasing workflow. Before a contract is signed, cybersecurity and OT engineering should confirm where the new system will sit in the zone model, what it must communicate with, and how vendors will support it. Decisions taken at this stage are cheap; after installation they are expensive and disruptive.

### Contracts and vendor obligations

Contracts are equally important. Vendor and integrator agreements should define vulnerability disclosure and notification obligations, remediation timelines, incident notification, end-of-life and end-of-support notice periods, logging and evidence expectations, data handling (including where vendor clouds store operational or personal data), remote-access conditions, subcontractor responsibilities, and audit or assurance rights. IEC 62443-2-4 provides a structured catalogue of security capabilities that asset owners can require from integrators and maintenance service providers, and certification against it offers evidence. Supply-chain governance more broadly is the subject of Chapter 19.

For OT vendors, remote access is often the central risk driver. Maintenance access may be necessary, but it must not become a permanent, invisible pathway into critical systems. Contracts and vendor-management processes should require that remote access be approved per session or per defined window, strongly authenticated with multi-factor authentication, routed through a controlled access point rather than a vendor-installed modem, restricted to defined assets and purposes, logged, and, for high-consequence systems, monitored or recorded in real time.

### Devices we cannot manage: exceptions with expiry

Policy must also define how to handle devices that cannot meet minimum requirements—legacy systems, specialized instruments, or products for which no compliant alternative exists. The practical posture should be: **devices we cannot manage require explicit approval.** Each exception should name the business justification, the accountable risk owner, the compensating controls (isolation, restricted vendor access, additional monitoring, physical protection, manual fallback), the expiry date, and the review trigger. Exceptions with expiry dates force the organization to revisit decisions as conditions change, consistent with the exception discipline described in Chapter 14.

An unmanaged device fleet is a governance choice. The worst outcome is discovering, after a compromise, devices that nobody owns, monitors, can patch, or knows how to remove safely.

## Architecture: zones, conduits, least privilege, and assume compromise

Managers need not design OT networks or configure industrial firewalls. They should, however, understand the architectural principles that determine how much damage a compromise can do, because those principles drive major investment decisions. Three themes are central: segmentation through zones and conduits, least privilege, and assume compromise. Together they shift governance away from the unrealistic ideal of perfect prevention and toward bounded impact and safe operation under stress. Chapter 13 develops security architecture more generally.

### Segmentation through zones and conduits

**Segmentation** is foundational. OT networks, building systems, laboratory equipment, physical-security systems, and IoT fleets should not sit on flat networks freely connected to corporate IT, guest Wi-Fi, vendor clouds, or the Internet. Segmentation limits the blast radius of compromise; it is what prevents a phishing email in the finance department from becoming a shutdown on the plant floor. The joint *Principles of OT Cyber Security* lists segmenting and segregating OT from all other networks as a core principle, and the Cyber Centre's AL25-016 alert begins its advice by asking operators to eliminate unnecessary Internet exposure of control systems.

ISA/IEC 62443-3-2 turns segmentation into a governance method through the concepts of **zones** and **conduits**. A zone is a grouping of assets that share common security requirements and consequences—for example, a safety instrumented system, a production cell, or a building automation network. A conduit is a controlled communication path between zones, with defined permitted traffic and protective measures. Each zone receives a target security level based on the risk assessment, and the gap between the target and the achieved level becomes a visible, governable item. The figure below summarizes the managerial sequence.

<span id="_Toc243635554" class="anchor"></span>Figure 55: Governing segmentation with zones and conduits

The governance value of this approach is that it converts an abstract instruction ("segment the network") into explicit decisions that managers can approve and audit: which assets belong together, which communications are permitted and why, what level of attacker each zone must resist, and who accepts the residual risk where the target cannot yet be met. It also allows differential protection. Controls that would be unacceptable in a safety zone, such as automatic patching or active scanning, may be perfectly appropriate in a supervisory or business-facing zone.

### Least privilege for people, vendors, and machines

**Least privilege** applies to people, vendors, service accounts, machines, and integrations. Remote-access accounts, vendor credentials, engineering workstations, management consoles, APIs, and cloud connectors should be limited to what an approved purpose requires. Privileged access to controllers, safety systems, building automation, medical devices, or production systems should be highly restricted, strongly authenticated, time-bound where feasible, and monitored. The identity disciplines of Chapter 10—privileged access management, joiner–mover–leaver processes, and periodic access review—apply here, though often through OT-specific tools.

In OT, least privilege protects more than data. It prevents unauthorized changes to physical processes, setpoints, control logic, safety parameters, and firmware. A vendor who needs to diagnose one drive does not need persistent access to the entire plant network. A data historian that collects readings does not need write access to controllers. An analytics platform that streams sensor data to the cloud should be architected so that nothing in the cloud can send commands back into the control zone unless that capability has been explicitly designed, approved, and protected.

### Assume compromise and the limits of IT-style zero trust

**Assume compromise** is the realism principle. Architectures should be designed with the expectation that some device, account, vendor pathway, workstation, or integration will eventually be compromised—through default credentials, phishing, vulnerable firmware, a compromised supplier, or lateral movement from IT. Governance should therefore require designs that limit the scope of compromise, detect abnormal behaviour, and preserve the ability to operate safely in a degraded mode. In legacy OT environments this may be the most realistic posture, because replacing every vulnerable asset is not feasible in the short term.

Zero trust principles, now being integrated into OT guidance by the NIST SP 800-82 Rev. 4 draft, support this posture but must be adapted. Continuous verification and fine-grained policy enforcement are valuable at the boundaries of OT and for remote and administrative access; inside a time-critical control loop, adding authentication steps or dependencies on external identity services can itself create safety and availability risk. The joint *Secure Connectivity Principles for OT* (2026) frame connectivity choices as desirable end states rather than minimum requirements, which is a sensible managerial stance: set a direction, prioritize by consequence, and record the reasons where the end state cannot yet be reached.

Managers should ask a few architectural questions regularly:

- Are operational networks segmented into zones based on consequence, with documented conduits and owners?

- Can any OT or IoT asset be reached directly from the Internet, and if so, who approved it and why?

- Is all vendor and administrative access forced through a controlled, logged access point?

- Can an affected zone be isolated quickly without creating an unsafe condition, and has that been tested?

- Could essential operations continue if parts of the environment became untrusted?

Segmentation limits spread, least privilege limits misuse, and assume compromise prepares the organization to detect, contain, and operate safely under adverse conditions.

## Patch and change governance under operational constraints

A central governance tension in OT is that patching and upgrades compete with uptime, safety, validation, and continuity. In IT, patching is disruptive but usually manageable. In OT, a patch may require a production shutdown, vendor supervision, safety testing, recertification, or validation that the change will not alter a physical process. Some systems have only one maintenance window per year. This does not mean patching can be ignored. It means **patching must be governed** rather than either imposed mechanically or deferred indefinitely.

### Risk-based vulnerability prioritization

A mature organization has a structured process for identifying vulnerabilities that affect its OT and IoT assets, assessing their operational relevance, prioritizing remediation, deciding between patching and compensating controls, documenting the decision, and reviewing residual risk over time. The question is not simply "can we apply the patch?" but "what is the risk of applying it, what is the risk of delaying it, and what reduces exposure while we decide?" IEC TR 62443-2-3 describes how asset owners and suppliers can cooperate on patch management, including vendor qualification of patches before they are applied.

Prioritization should combine several factors rather than a single score:

- **Exposure.** Is the asset reachable from the Internet, from IT, or from vendor connections, or only from within a well-controlled zone?

- **Exploitation.** Is there evidence of active exploitation, such as an entry in CISA's Known Exploited Vulnerabilities catalogue or a Cyber Centre advisory?

- **Consequence.** What could an attacker do with the asset, given its zone and the process it controls?

- **Severity.** What does the technical severity assessment, such as CVSS v4.0, indicate (see Chapter 8)?

- **Feasibility.** Is a tested, vendor-approved patch available, and when is the next safe window?

Under this approach, a vulnerability in an Internet-exposed remote-access gateway or engineering workstation may warrant emergency action, while a theoretically severe flaw in an isolated sensor deep inside a well-protected zone may reasonably wait for the next planned outage.

### Compensating controls and documented exceptions

When patches are delayed, compensating controls must be defined and implemented, not merely assumed. These may include tightening conduit rules, blocking specific protocols or ports, disabling unused services, restricting or suspending vendor access, increasing monitoring of the affected asset, applying virtual patching through intrusion prevention where it is safe to do so, or accelerating replacement. Delayed patching must not become informal risk acceptance. It should produce a documented decision that identifies the risk owner, the justification, the compensating controls, the review date, and the conditions that would trigger escalation, such as public exploitation or a change in exposure. These records form a key part of the evidence auditors and regulators will expect.

### Configuration integrity and controlled change

Change management is as important as patching, because changes themselves create risk. Modifications to control logic, setpoints, firmware, safety parameters, network rules, and engineering workstation builds can affect physical operations; a poorly tested change can halt production or create unsafe conditions, and a malicious change can do so deliberately. Many process industries already operate a **management of change (MOC)** procedure for physical modifications; extending it to cyber-relevant changes is usually more effective than creating a parallel IT-style change board.

Governance should require that OT changes be authorized, risk-assessed, documented, tested where feasible, communicated to affected teams, and reversible. Control logic, configuration files, firmware versions, and network settings should be baselined and version-controlled so that unauthorized or unexpected changes can be detected. Emergency changes should be permitted when necessary but reviewed afterward and incorporated into the record. **Configuration integrity** is the end goal: the organization should be able to determine at any time whether a critical system's running configuration matches the approved one, and to restore a known-good state with confidence. Without that capability, it may not know whether a system is behaving as intended, whether an incident altered control logic, or whether recovery has restored a safe state.

A mature organization does not choose blindly between security and availability; it governs the trade-off and revisits it as conditions change.

## Safety and security: one risk conversation

In many cyber-physical environments, a well-established safety discipline already exists, with its own standards, specialists, and regulators. Functional safety engineering, governed by IEC 61508 (*Functional safety of electrical/electronic/programmable electronic safety-related systems*, edition 2.0 from 2010, with a third edition in development at the time of writing) and sector derivatives such as IEC 61511 for the process industries, designs protective systems so that the probability of dangerous failure is reduced to a tolerable level. Safety engineers think in terms of hazards, safety integrity levels, and independent layers of protection. Cybersecurity professionals think in terms of threats, vulnerabilities, and controls. The two disciplines have historically worked apart, and that separation is itself a risk.

### Why safety and security must meet

Traditional safety analysis assumes that failures are random or arise from foreseeable human error. It rarely assumes an intelligent adversary who deliberately defeats several protective layers at once, for example by disabling a safety instrumented system while simultaneously driving the process beyond its limits and suppressing alarms. Attacks on safety systems have shown that this is not hypothetical. Conversely, security controls designed without safety input can create hazards: an automatic lockout that prevents an operator from reaching an HMI during an emergency, or an endpoint agent that delays a controller's response time.

IEC TR 63069:2019 offers a framework for integrating functional safety and security in industrial automation. Its central idea is that security protects the safety function: a **security environment** around safety-related systems must be established and maintained so that safety assumptions remain valid. IEC 61511 likewise expects a security risk assessment of the safety instrumented system. For managers, the implication is that cyber-physical risk assessment should be a joint exercise in which safety and security specialists examine the same hazard scenarios, and that cyber threats should be considered explicitly in process hazard analyses and safety cases.

### Practical interfaces between the disciplines

Integration does not require a new bureaucracy. It requires a few deliberate interfaces:

- **Joint hazard and threat analysis.** Process hazard analyses and cyber risk assessments examine the same high-consequence scenarios, with each discipline challenging the other's assumptions.

- **Protected safety functions.** Safety instrumented systems sit in their own zone with the highest target security level, minimal connectivity, and, where possible, non-programmable or hardwired protective layers as a final backstop.

- **Shared change control.** Cyber-relevant changes to safety-related systems pass through the same management-of-change process as physical changes.

- **Combined incident criteria.** Incident severity scales include safety and environmental consequences, and safety leaders are part of the cyber incident structure.

- **Common reporting.** Executives see safety and cybersecurity risk for critical processes in one view rather than in competing reports.

Medical devices illustrate the same principle. Device manufacturers apply ISO 14971 risk management for safety, and FDA and Health Canada now expect cybersecurity risk management to feed into it. Hospitals that bring biomedical engineering, clinical leadership, and cybersecurity together to govern connected devices apply the same idea on the user side.

| **Security protects the safety case.** If an adversary can defeat the assumptions on which safety engineering relies, the organization's safety performance is only as strong as its cybersecurity. |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Detection, response, and resilience when systems are degraded

Monitoring, detection, incident response, and continuity in OT and CPS cannot simply be copied from IT. Enterprise practices often rely on endpoint agents, active scanning, rapid containment, and frequent change. Many OT environments cannot tolerate these. Isolating a system may interrupt production, safety functions, or essential services, and recovery must establish that the physical process can be trusted again. Chapter 12 covers incident management and continuity in general; this section focuses on what is different in cyber-physical settings.

### Passive monitoring and what to watch

OT monitoring typically relies on passive network monitoring with tools that understand industrial protocols, logs from control systems and historians where available, close monitoring of engineering workstations, and focused oversight of remote-access pathways. The aim is to detect behaviour that may indicate compromise, unsafe manipulation, or unauthorized change: unexpected device communications, unusual commands such as controller program downloads, abnormal setpoint changes, unauthorized remote sessions, new devices, traffic crossing zones that should be separate, or modifications to control logic. MITRE ATT&CK for ICS helps prioritize which behaviours to detect, by mapping known adversary techniques to observable activity.

IoT monitoring faces similar challenges at greater scale. Many devices have limited logging and depend on vendor clouds, so monitoring often focuses on network behaviour, device identity, communication patterns, authentication events, firmware status, and connections to external services. The goal is not to collect every signal but to achieve enough visibility to spot unmanaged devices, abnormal activity, misuse of vendor access, and pathways into sensitive zones. The security operations centre (Chapter 7) needs OT context to interpret these signals; an alert about a controller download means little without knowing whether a planned change was under way.

### Response decisions and safety gates

Incident response in OT and CPS must involve safety, engineering, facilities, and operational leadership from the outset, because technical containment actions have physical consequences. Disconnecting a network segment, revoking vendor access, shutting down a controller, blocking a protocol, or isolating an engineering workstation may be necessary, but each can affect production, equipment, patient care, or public services. Governance must therefore predefine crisis decision rights, escalation paths, and **safety gates**—points at which a qualified operations or safety leader must confirm that a containment or recovery action will not create a hazard.

A mature response model specifies who can authorize containment, who determines whether operations can continue safely in a degraded mode, who communicates with vendors, regulators, and the public, and who decides to activate continuity procedures. In higher-risk environments, incidents are managed through a joint cyber-operations crisis structure that includes cybersecurity, OT engineering, operations, safety, legal, communications, and executive decision-makers. Many organizations also pre-decide **precautionary shutdown** criteria: the conditions under which they will stop a process because they can no longer trust its monitoring or control, even if the OT itself has not been proven compromised. Making that decision in advance, at the right level, avoids both reckless continuation and panicked shutdown.

### Safe recovery and minimum viable operations

A crucial managerial point is that OT incidents require **safe recovery**, not merely system restoration. In IT, recovery means reimaging systems, restoring data, resetting credentials, and confirming eradication. In OT, recovery also means validating that sensor readings are trustworthy, that control logic has not been altered, that safety systems function, that configurations match known-good baselines, that equipment is in a safe state, and that operators can confirm that restart will not create new risk. This takes time and must be planned before the crisis.

Resilience begins with defining **minimum viable operations**: the processes, safety functions, facilities, and services that must continue during cyber stress, and at what reduced level. Manual fallback procedures are part of this. Where feasible, trained staff should be able to operate safely without full automation through local control modes, manual sampling and inspection, alternate communications, paper procedures, and pre-approved shutdown steps. These procedures must be realistic. A manual process that exists only on paper, has never been staffed or practised, and depends on retired employees is not a resilience capability.

### Backups, spares, vendors, and exercises

Backup strategies must cover far more than business data. Organizations should protect controller logic, configuration files, firmware images, engineering workstation builds, network device configurations, safety settings, recipes, and calibration data, store them offline or immutably, test restoration periodically, and tie them to procedures for restoring a known-good state. For IoT fleets, resilience means being able to isolate classes of devices rapidly, rotate credentials across fleets, revoke compromised certificates, and replace devices at scale, which favours standardized models and centralized management. Critical spare parts, including spare controllers pre-loaded with known-good logic, can shorten recovery dramatically.

Vendor dependencies must be addressed before an incident. Many OT systems cannot be restored without vendor support, so contracts and plans should specify how vendors will be reached, what access they will be granted and through which channels, what evidence they must provide, and what happens if the vendor itself is unavailable or compromised. Finally, resilience must be exercised: tabletop exercises with operations and executives, technical recovery tests, safe shutdown drills, manual-operation rehearsals, and joint exercises with key vendors. **A resilient organization knows what must continue, what can be degraded, who has authority to decide, which configurations can be trusted, and when it is safe to return to normal.**

## AI-enabled cyber-physical environments: model abuse, data poisoning, and unsafe automation

As organizations integrate analytics and artificial intelligence into operations, the cyber-physical risk surface expands. AI supports predictive maintenance, anomaly detection, process optimization, scheduling, energy management, robotics coordination, warehouse automation, digital twins, quality inspection, and safety-related decision support. In these contexts, the governance concern is not only whether a model is accurate. It is whether an attacker, insider, vendor, or system failure could manipulate the model, its data, or its operating environment to produce harmful physical outcomes. Chapter 18 treats AI security comprehensively; this section focuses on what changes when AI touches the physical world.

### Data poisoning, model abuse, and unsafe automation

Three risks deserve particular attention:

- **Data poisoning is a credibility risk.** If an attacker can influence sensor streams, logs, maintenance records, training data, or feedback loops, the model may learn the wrong pattern. Predictive maintenance may miss a failing bearing; anomaly detection may learn to treat an attack as normal. In CPS, corrupted data becomes a safety, reliability, and continuity problem as well as a data-quality one.

- **Model abuse is an integrity risk.** If an attacker can manipulate model inputs, parameters, thresholds, prompts, model files, APIs, or deployment pipelines, the system may generate unsafe recommendations or suppress legitimate warnings while still appearing to function normally.

- **Unsafe automation is a safety risk.** When AI outputs are executed automatically, or human approval becomes perfunctory, the AI system becomes a high-leverage control point. The risk is not that AI is inherently unsafe; it is that automation can amplify errors or manipulation faster than humans can respond.

A further, quieter risk is **automation complacency**: operators who come to trust AI recommendations without independent verification gradually lose the skills and situational awareness they would need when the model is wrong. This connects directly to the manual-fallback discussion above.

### Bounding what AI is permitted to do

Governance should begin with explicit boundaries on AI authority. Managers should distinguish among four levels: systems that only **inform** decisions, systems that **recommend** actions, systems that **initiate** actions subject to human approval, and systems that **execute** actions automatically. The higher the potential physical consequence, the stronger the requirements for human approval, independent safety interlocks, validation, logging, testing under adverse conditions, and override capability. Critically, AI should not be placed in a position where it can override or bypass a safety instrumented function; protective layers designed under IEC 61508 and IEC 61511 must remain independent of the optimization layer.

AI-enabled operational systems also require data-integrity controls: validating critical sensor inputs against physical plausibility and redundant sources, monitoring for anomalous data patterns, protecting training and operational datasets, controlling access to data pipelines, and ensuring model outputs can be traced to trusted sources. Where models depend on vendor platforms, cloud analytics, or external data, those dependencies belong in the risk assessment and in third-party governance. The architectural rule from earlier applies with particular force: a cloud analytics service should not be able to send commands into a control zone unless that path has been explicitly designed, approved, and protected.

### Managing models as operational assets

Models, prompts, thresholds, feature sets, training datasets, APIs, and deployment pipelines should be treated as operational assets with owners, version control, access control, change approval, testing, monitoring, rollback procedures, and periodic review. A model that influences physical operations should not be retrained, re-tuned, or redeployed informally. In high-consequence settings, AI governance begins to resemble safety engineering: the organization defines predictable behaviour, safe operating limits, failure modes, human override procedures, and accountability. It should be clear who approved the model, who owns the operational risk, who monitors performance, who investigates abnormal outputs, who can suspend automated action, and what evidence is retained when AI-influenced decisions affect operations. **AI-enabled automation must be trustworthy, bounded, monitored, and accountable before it is permitted to influence physical operations.**

## Governing OT and IoT risk in practice: metrics, a worked scenario, and questions

The previous sections described the building blocks of OT, IoT, and CPS governance. This section shows how they come together in measurement, in a realistic decision process, and in the questions managers and boards should ask.

### Key risk indicators and evidence

OT and IoT programs need a small set of indicators that tell executives whether exposure is shrinking and whether the organization could cope with an incident. Indicators should be tied to risk appetite and tolerance statements (Chapter 4) and reported by criticality tier rather than as raw counts. Useful examples include:

- **Inventory coverage.** The proportion of high-consequence sites or zones with a validated, current asset inventory and named owners.

- **Internet exposure.** The number of OT or IoT assets directly reachable from the Internet, with a target of zero for control systems.

- **Remote access control.** The proportion of vendor and administrative remote access routed through the controlled access point with multi-factor authentication and session logging.

- **Exception health.** The number of open exceptions for high-consequence assets, the proportion past their expiry date, and the average age of exceptions.

- **Exploited vulnerability exposure.** Known exploited vulnerabilities on high-consequence assets without either a patch or documented compensating controls.

- **Recoverability.** The proportion of critical controllers and workstations with tested, offline known-good backups, and the date of the last successful restoration test.

- **Unsupported assets.** The number of critical assets past vendor end of support, with funded replacement plans.

- **Exercise readiness.** Whether a joint cyber-operations exercise was held in the past year and whether its findings were closed.

These indicators generate evidence as well as insight. Inventory records, zone and conduit diagrams, exception registers, access logs, backup test results, and exercise reports are the artifacts that internal audit, regulators, insurers, and certification bodies will request. Building the program so that it produces them routinely avoids compliance theatre—impressive documentation assembled before an audit that bears little relation to day-to-day control.

### Illustrative scenario: a Québec municipal water utility

The *Régie intermunicipale de l'eau des Deux-Rives* is a fictional intermunicipal water authority serving about 90,000 residents in three Québec municipalities. It operates one drinking water treatment plant, a wastewater treatment plant, eleven remote pumping stations and reservoirs, and a SCADA system installed in stages since 2006. The OT environment is maintained by a small in-house team of operators and a regional integrator, while the municipalities' shared IT service manages office systems. After the Cyber Centre's AL25-016 alert describing tampering with water pressure values at a Canadian facility, the board of the *Régie* asked the general manager a simple question: "Could that happen here, and how would we know?"

The general manager appointed the plant superintendent as the accountable risk owner for treatment and distribution OT, with the municipal IT director's security adviser as the cybersecurity lead and the integrator as a contracted participant. A six-week discovery combined a passive network sensor, a review of integrator documentation, and physical walk-downs of every station. The findings were sobering but typical. Two remote pumping stations had HMIs reachable from the Internet through cellular modems installed by a previous contractor; one still used the manufacturer's default password. The integrator maintained an always-on VPN account shared by several technicians. The SCADA servers ran an operating system past end of support, and controller logic backups existed only on a laptop at the integrator's office. Chlorine dosing at the treatment plant could be adjusted from the SCADA HMI within wide limits, although an independent hardwired high-level alarm remained in place.

The team then ran a scenario-based risk assessment using the method of Chapters 5 and 6. The highest-rated scenario combined a plausible threat (opportunistic actors scanning for exposed HMIs), a confirmed vulnerability (an Internet-reachable HMI with default credentials), and a serious impact (loss of distribution pressure in one sector, with a boil-water advisory, possible contamination risk from backflow, and regulatory and reputational consequences). A second scenario considered ransomware spreading from the municipal IT network into SCADA servers that shared a flat network segment, forcing a precautionary switch to manual operation. A third examined manipulation of dosing setpoints. Each scenario was expressed in operational terms—hours without remote visibility, residents affected, regulatory notifications triggered—rather than only in technical severity scores.

The board's risk appetite statement, adopted the previous year, said the *Régie* had "no appetite for cyber risks that could compromise drinking water safety and very low appetite for risks that could interrupt service for more than four hours." Against that statement, the first scenario was clearly intolerable. The general manager authorized immediate actions within existing budget: the cellular modems were removed and replaced with a single controlled remote-access gateway with multi-factor authentication; default credentials were changed everywhere; the shared integrator account was replaced with named, time-limited accounts approved per session; and controller logic and SCADA configurations were backed up to offline media held at the plant, with a first restoration test witnessed by the superintendent.

Larger decisions went to the board with options. The team proposed a zone and conduit design aligned with IEC 62443-3-2, separating the treatment plant control zone, the distribution SCADA zone, the safety-related dosing controls, and the municipal IT network, with a monitored demilitarized zone between them; a SCADA platform upgrade over two budget years; and passive OT monitoring integrated into the municipal security operations arrangement. Because the SCADA upgrade could not be completed for eighteen months, the superintendent formally accepted the residual risk of the unsupported servers with compensating controls—strict isolation, application allow-listing, and enhanced monitoring—recorded as an exception with a twelve-month expiry and quarterly review. The integrator's contract renewal added requirements drawn from IEC 62443-2-4, including remote-access conditions, vulnerability notification, and backup obligations.

Finally, the *Régie* rehearsed. A tabletop exercise with operators, the general manager, the communications officer, the integrator, and a representative of the regional public health authority walked through the ransomware scenario and revealed that nobody had authority to order a precautionary switch to manual operation at night. The decision-rights matrix was updated so that the on-call shift supervisor could make that call and inform the superintendent afterward. The board now receives a one-page quarterly report showing Internet exposure (now zero), the share of remote sessions through the controlled gateway (now all), open exceptions and their expiry dates, backup test status, and progress on the segmentation project. The utility is not invulnerable, but its risks are now visible, owned, prioritized, and defensible.

### Questions managers and boards should ask

The following questions help executives and directors test whether OT, IoT, and CPS risks are genuinely governed:

- Do we have a current inventory of our OT and IoT assets, and does every high-consequence asset have a named business owner and technical custodian?

- Have we defined, in operational terms, what must not fail, and do our investment and architecture priorities reflect it?

- Are any of our control systems or connected devices reachable from the Internet, and who approved each exception?

- How is vendor and integrator remote access approved, authenticated, limited, logged, and reviewed?

- Which recognized references—ISA/IEC 62443, NIST SP 800-82, NIST IR 8259, sector rules—underpin our program, and which legal regimes (NIS2, the CRA, Canada's CCSPA, medical device rules) apply to us or to our suppliers?

- Do our procurement standards and contracts require secure-by-design features, SBOMs, declared support periods, and vulnerability notification for connected products?

- How many exceptions are open for high-consequence assets, how many have expired, and who has accepted the residual risk?

- Are safety engineering and cybersecurity jointly assessing high-consequence scenarios, and are safety instrumented systems protected as the highest-priority zone?

- Who has authority to isolate or shut down an operational system during a cyber incident, including at night and on weekends?

- Could we operate safely in a degraded or manual mode, and when did we last practise it?

- Do we hold tested, offline, known-good backups of controller logic and configurations?

- What are AI and analytics systems permitted to do in our operations, and can any of them act on physical processes without human approval?

## Bringing it together: making cyber-physical risk governable

OT, IoT, and cyber-physical security are governance domains because they force organizations to confront risks that are not purely digital. A cyber event in these environments may disrupt operations, damage equipment, compromise safety, interrupt public services, harm the environment, or expose the organization's dependence on vendors, remote access, unmanaged devices, and fragile legacy systems. The question is not only whether systems are technically vulnerable, but whether the organization can identify, own, control, monitor, and recover from risks that affect how it operates in the physical world.

The chapter's lessons form a coherent sequence. **Visibility and criticality come first**: an organization cannot govern assets it has not identified, classified by consequence, linked to processes, and assigned to accountable owners, and it cannot set priorities without defining what must not fail. **Accountability must span functions**: OT risk persists when no single group owns the whole picture, so the operating model must define decision rights for vendor access, risk acceptance, emergency isolation, patch deferral, device deployment, and return to operations, and must exercise them before a crisis.

**Prevention starts at procurement and architecture.** Once devices are embedded in production lines, buildings, hospitals, or utility networks, security options become limited and costly. Tiered purchasing requirements, contract clauses, and lifecycle planning determine whether future teams inherit governable systems or permanent exceptions. Zones and conduits, least privilege, and assume-compromise design determine whether a single compromise stays contained or spreads to the systems that matter most. Patch and change governance then turns unavoidable constraints into documented, time-limited decisions supported by compensating controls and configuration integrity.

**Safety and security belong in one conversation.** Where functional safety engineering exists, cybersecurity must protect the assumptions on which it relies, and safety specialists must help shape security controls so that they do not create new hazards. Detection and response must be adapted to operational realities, with safety gates and pre-agreed shutdown criteria, and recovery must be safe rather than merely fast. Resilience—minimum viable operations, manual fallback, offline known-good backups, spares, vendor arrangements, and exercises—determines whether an incident becomes a crisis. As AI and advanced automation enter operations, the same discipline must extend to data, models, and the authority automated systems are granted.

The external environment reinforces these lessons: standards such as ISA/IEC 62443 and NIST SP 800-82, product rules such as the Cyber Resilience Act, and infrastructure laws such as NIS2 and the CCSPA increasingly define what a reasonable, defensible decision looks like.

For business technology managers, OT, IoT, and CPS security may be where cybersecurity is most consequential and most visible. It is where weak procurement creates decades of exposure, where unclear accountability delays response, where unmanaged access creates operational risk, and where poor resilience planning turns a technical incident into a public crisis. Managers do not need to configure every controller, sensor, robot, or smart device. They must ensure that the organization knows what it depends on, what must not fail, who has authority to act, how residual risk is accepted and reviewed, and how operations will remain safe and reliable under cyber stress. That is what it means to make cyber-physical risk governable.

## Chapter summary

Operational technology, IoT, and cyber-physical systems take cybersecurity beyond information into physical operations, safety, equipment, the environment, and essential services. The governance question shifts from what information could be lost to what could physically happen, to whom, and how quickly. These environments are chronically vulnerable because they fall between ownership boundaries, run for decades, resist patching, and depend on vendors. The chapter argues that managers need not master industrial protocols, but they must make cyber-physical risk governable: know what exists and what must not fail, assign accountable owners and decision rights, control risk at procurement and in architecture, protect the safety case, and prepare for safe recovery.

### Key points to retain

- Scope should include any system whose compromise could disrupt operations, compromise safety, damage equipment, harm the environment, expose data, or create an unmanaged path into the enterprise.

- In OT, safety, availability, and integrity of control usually outrank confidentiality, and cyber risk is measured in physical consequences, not only lost data.

- Long lifecycles, limited patchability, fragmented ownership, and vendor-controlled access mean that 'we cannot patch it' is a constraint requiring an explicit risk decision, not a governance answer.

- ISA/IEC 62443, NIST SP 800-82, the CRA, NIS2, Canada's CCSPA, and medical device rules do not replace risk judgment, but they define what reasonable and defensible now look like.

- Governance begins with a maintained inventory of assets and dependencies, consequence-based criticality tiers, and an operational definition of what must not fail.

- Four ownership layers and written decision rights, agreed before a crisis, prevent delay and diffusion of responsibility over vendor access, isolation, patch deferral, and restart.

- Procurement is a primary cybersecurity control: tiered minimum requirements, lifecycle cost, architecture review, and contract clauses determine decades of exposure.

- Zones and conduits turn 'segment the network' into approvable decisions, with target security levels set as risk-appetite choices and least privilege applied to people, vendors, and machines.

- Patching must be governed by exposure, exploitation, consequence, severity, and feasibility, with documented compensating controls and configuration integrity under management of change.

- Security protects the safety case, so safety and security specialists should analyse the same hazard scenarios and safety instrumented systems sit in the most protected zone.

- OT response needs safety gates, pre-agreed precautionary shutdown criteria, minimum viable operations, practised manual fallback, and offline known-good backups of controller logic.

- AI in operations must be bounded by explicit authority levels, must never bypass safety functions, and should be managed as an operational asset; KRIs and exercises show whether exposure is falling.

### Key terms

- **Operational technology (OT)** — Hardware and software that monitor, control, or automate physical processes, devices, and infrastructure, such as SCADA, DCS, PLCs, HMIs, and safety instrumented systems.

- **Internet of Things (IoT)** — Network-connected devices that combine at least one sensor or actuator with connectivity, are often embedded, and are typically deployed in large numbers.

- **Industrial IoT (IIoT)** — The application of the IoT pattern to operations, such as wireless pump sensors, cloud-connected compressors, and edge gateways streaming plant data.

- **Cyber-physical system (CPS)** — An integrated system in which software, sensors, networks, and computing govern physical behaviour through feedback loops, so digital compromise can change physical outcomes.

- **Safety instrumented system (SIS)** — A protective system that brings a process to a safe state when limits are exceeded, designed under functional safety standards such as IEC 61511.

- **Zone** — In ISA/IEC 62443, a grouping of assets that share common security requirements and consequences, assigned a target security level.

- **Conduit** — In ISA/IEC 62443, a controlled communication path between zones with defined permitted traffic and protective measures.

- **Security level (SL 1 to SL 4)** — An ISA/IEC 62443 expression of protection in terms of the attacker a zone must withstand, from casual violation to sophisticated, well-resourced attackers.

- **Criticality mapping** — Ranking systems by the consequences of their disruption or manipulation, starting from the worst credible outcomes and working back to the systems that could produce them.

- **Compensating controls** — Measures such as tighter conduit rules, restricted vendor access, extra monitoring, or virtual patching that reduce exposure when a patch must be delayed.

- **Management of change (MOC)** — The process-industry procedure for authorizing and assessing physical modifications, which should be extended to cyber-relevant changes to control logic and configurations.

- **Configuration integrity** — The ability to confirm that a critical system's running configuration matches the approved baseline and to restore a known-good state with confidence.

- **Safety gate** — A point in incident response at which a qualified operations or safety leader confirms that a containment or recovery action will not create a hazard.

- **Minimum viable operations** — The processes, safety functions, and services that must continue during cyber stress, and the reduced level at which they must operate.

- **Software bill of materials (SBOM)** — A list of the software components in a product, which procurement can demand from suppliers and regulators increasingly require of manufacturers.

## Review questions

### Essay questions

1.  The chapter states that OT and IoT environments are chronically vulnerable partly because 'each group acts reasonably within its own mandate, and the organization as a whole ends up with risks that nobody owns.' Analyze this problem and design an operating model, including decision rights, that would resolve it.

2.  Evaluate the claim that procurement is 'a primary cybersecurity control' in OT and IoT. Using the chapter, propose tiered minimum requirements and contract clauses for two very different purchases, such as a production-line robot and a building-wide occupancy-sensor network.

3.  A plant manager argues that an unsupported controller cannot be patched and that the issue should therefore be closed. Critique this position and explain how the chapter's patch and change governance, compensating controls, and exception discipline should be applied.

4.  Apply the chapter to an organization of your choice that operates OT or IoT (or to the Régie intermunicipale de l'eau des Deux-Rives). Define what must not fail, identify the two highest-consequence scenarios, and recommend immediate actions and board-level decisions consistent with a stated risk appetite.

5.  Assess the statement that 'security protects the safety case.' Explain how safety engineering and cybersecurity should work together, and how the same principle applies when AI systems are allowed to recommend or execute actions on physical processes.

### Multiple-choice questions

Choose the best answer. Answers are listed in the appendix "Answers to multiple-choice questions".

**1.** How did NIST SP 800-82 Rev. 3 change the terminology used for industrial control systems?

> A. It broadened 'industrial control systems' to 'operational technology' to reflect wider use
>
> B. It replaced 'operational technology' with 'cyber-physical systems' for all sectors
>
> C. It limited its scope to electricity and pipelines regulated by federal agencies
>
> D. It merged IoT and OT into a single category called the industrial IoT

**2.** According to the chapter's scope test, which pair of 'yes' answers usually places a system in the high-consequence tier?

> A. Connectivity and ownership
>
> B. Physical influence and consequence
>
> C. Manageability and data
>
> D. Data and connectivity

**3.** Which statement best reflects the chapter's view of priorities in operational environments?

> A. Confidentiality is irrelevant because OT data has no value to attackers
>
> B. Confidentiality always outranks availability because recipes are trade secrets
>
> C. Safety, process availability, and integrity of control usually take precedence over confidentiality
>
> D. Availability matters only for IoT devices, not for safety-related controllers

**4.** A plant engineer says a controller 'cannot be patched'. How does the chapter say this statement should be treated?

> A. As a final answer that closes the vulnerability record for that asset
>
> B. As evidence that the controller is safe because it is isolated by design
>
> C. As a reason to transfer the asset's ownership to the central IT department
>
> D. As a constraint requiring an explicit risk decision, compensating controls, and a plan

**5.** Which risk type from the chapter's taxonomy can let a system keep running while producing unsafe or misleading results?

> A. Operational disruption through ransomware encryption
>
> B. Loss of visibility caused by incomplete inventories
>
> C. Supplier remote-access risk through shared accounts
>
> D. Integrity manipulation of setpoints, readings, logic, or alarms

**6.** What is the distinctive contribution of the ISA/IEC 62443 series, according to the chapter?

> A. It sets mandatory reporting deadlines for incidents at critical infrastructure operators
>
> B. It replaces functional safety standards such as IEC 61508 for industrial processes
>
> C. It assigns requirements to asset owners, service providers and integrators, and product suppliers
>
> D. It applies only to consumer IoT devices sold on the European market

**7.** Why does the chapter describe setting a target security level (SL 1 to SL 4) for a zone as a managerial matter?

> A. It determines the product warranty period that suppliers must provide by law
>
> B. It defines the attacker the zone must withstand, which is a risk-appetite decision
>
> C. It sets the maximum number of devices that may be placed in each zone
>
> D. It is assigned automatically by certification bodies such as ISASecure

**8.** Under the EU Cyber Resilience Act as described in the chapter, when do the main obligations such as essential requirements, support periods, and CE marking apply?

> A. From December 11, 2027
>
> B. From December 10, 2024
>
> C. From September 11, 2026
>
> D. From October 17, 2024

**9.** A Québec municipal water utility asks whether Canada's Critical Cyber Systems Protection Act (CCSPA) applies to it. Based on the chapter, what is the most accurate answer?

> A. Yes, because the CCSPA covers every operator of drinking-water systems in Canada
>
> B. Probably not, because the CCSPA targets designated operators in federally regulated sectors
>
> C. Yes, because the CCSPA applies to all organizations using SCADA systems
>
> D. No, because the CCSPA has not yet received Royal Assent in Parliament

**10.** Why does the chapter recommend combining passive monitoring, configuration review, walk-downs, and records to build an OT inventory?

> A. Aggressive active scanning can disrupt fragile OT devices
>
> B. Passive methods are required by NIS2 for all essential entities
>
> C. Physical walk-downs replace the need for ongoing inventory maintenance
>
> D. Vendor documentation is always complete and current for legacy equipment

**11.** In OT and CPS, how does the chapter say criticality should primarily be defined?

> A. By the sensitivity of the data the system stores and processes
>
> B. By the purchase price and replacement cost of the equipment
>
> C. By the number of users who log in to the system each day
>
> D. By the consequences of disruption or manipulation, working back from the worst credible outcomes

**12.** During an OT incident, security wants to isolate a network, operations fears production loss, and a vendor demands unverified remote access. What does the chapter identify as the governance gap?

> A. The organization lacked enough endpoint agents on its controllers
>
> B. The vendor contract did not include a sufficiently high penalty clause
>
> C. Decision rights were not explicitly agreed and documented before the incident
>
> D. The SOC had not been given administrative rights to the OT network

**13.** Why does the chapter call procurement a primary cybersecurity control in OT and IoT?

> A. Procurement teams are the only function with authority to patch devices
>
> B. Suppliers are legally liable for all incidents once a contract is signed
>
> C. Governance leverage is highest before purchase, since devices may create risk for twenty years
>
> D. Purchasing rules automatically satisfy the requirements of ISA/IEC 62443

**14.** Which remote-access arrangement best matches the chapter's expectations for OT vendors?

> A. An always-on VPN account shared by the vendor's technicians for faster support
>
> B. A vendor-installed cellular modem so access does not depend on corporate networks
>
> C. Permanent access to the whole plant network to avoid delays during breakdowns
>
> D. Approved per session, MFA-protected, routed through a controlled access point, restricted, and logged

**15.** In ISA/IEC 62443-3-2 terms, what is a conduit?

> A. A controlled communication path between zones with defined permitted traffic and protective measures
>
> B. A grouping of assets that share common security requirements and consequences
>
> C. A hardwired safety layer that brings a process to a safe state on failure
>
> D. A certification scheme for suppliers' secure development lifecycles

**16.** Two vulnerabilities share a high CVSS score: one on an Internet-exposed remote-access gateway, one on an isolated sensor deep inside a well-protected zone. Applying the chapter, which prioritization is most defensible?

> A. Patch both immediately, because equal CVSS scores require equal treatment
>
> B. Act urgently on the gateway; the sensor may wait for the next planned outage
>
> C. Patch the sensor first, because it is closer to the physical process
>
> D. Defer both until the annual shutdown to avoid any production disruption

**17.** What is the central idea of IEC TR 63069:2019 as presented in the chapter?

> A. Security protects the safety function, so safety assumptions remain valid
>
> B. Safety systems should be connected to the Internet for real-time monitoring
>
> C. Security controls should replace independent layers of protection over time
>
> D. Safety and security should be managed by separate teams with separate reports

**18.** What is the purpose of pre-agreed precautionary shutdown criteria in OT incident response?

> A. To guarantee that production never stops during any cyber incident or outage
>
> B. To let the vendor decide alone whether the plant should keep operating normally
>
> C. To decide in advance when to stop a process that can no longer be trusted
>
> D. To replace the need for manual fallback procedures and exercises

**19.** An AI optimization system could adjust a process automatically in ways that conflict with a safety instrumented function. What does the chapter require?

> A. The AI may override the safety function if its accuracy exceeds that of operators
>
> B. AI must never override or bypass the safety function, and protective layers must stay independent
>
> C. The safety function should be retuned to follow the AI's optimized setpoints
>
> D. Human approval can be removed once the model has run safely for a year

**20.** In the water utility scenario, the board's appetite statement expressed 'very low appetite' for interruptions over four hours. Why was the Internet-exposed HMI scenario judged clearly intolerable?

> A. It had the highest CVSS score of all vulnerabilities found during discovery
>
> B. Its remediation would have cost more than the utility's annual budget
>
> C. The integrator refused to support the HMIs after the contract renewal
>
> D. It threatened drinking water safety, for which the board had stated no appetite
