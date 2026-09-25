// ================================================================ 3. CONTEXT
add(h1("3. Organizational and technology context"));
add(p("MediBec's mission is to deliver high-quality healthcare supported by modern technology, efficient clinical workflows and secure access to patient information (E1). Its risk is therefore not only data theft: an outage or integrity failure can delay diagnosis, disrupt medication management and cancel appointments across four regions."));
add(table(["Area", "Documented facts (E1-E4)", "Implication for risk"], [
  ["Network", "Montreal hub, WAN fibre to clinics, firewalls, IDS/IPS, antivirus", "Perimeter controls exist; segmentation, east-west monitoring and Wi-Fi security unknown"],
  ["Data centre and storage", "Montreal DC, local clinic storage, on-site and off-site backups, encrypted storage", "Concentration on one hub; backup isolation and restore testing not evidenced"],
  ["Cloud and remote access", "Cloud-based remote data access for staff and clinicians", "Identity is the new perimeter; cloud configuration and logging unknown"],
  ["Endpoints and devices", "Workstations, tablets, handhelds, interconnected medical equipment", "Device inventory and patch status unknown; certification limits patching"],
  ["Applications", "Integrated EHR, radiology, lab, pharmacy, billing, communications", "Highly interdependent; one compromise spreads across clinical workflows"],
  ["People and budget", "15 IT staff, 2 cybersecurity; IT budget stated as CAD 100M incl. salaries", "Capacity is the binding constraint. **Evidence gap** - the stated budget appears high relative to 15 staff; cybersecurity budget share not stated"],
  ["Governance", "Steering Committee (teaching assumption); Board accepts material residual risk", "Structure exists on paper; operation and reporting unproven"],
  ["Regulatory", "Quebec and Canadian privacy expectations, health information confidentiality, professional obligations", "Confidentiality incidents involving health data are likely to trigger register, assessment and notification duties - **legal counsel to confirm applicable regime**"],
], [16, 42, 42]));
add(h2("3.1 Crown-jewel assets and services"));
add(table(["ID", "Asset / service", "Business purpose", "Owner", "C", "I", "A", "Technical dependencies", "Third parties"],
  D.CROWN, [5, 15, 15, 14, 7, 7, 7, 17, 13], { size: 14 }));
add(spacer());
add(p("Owners are proposed roles, not documented appointments - **Evidence gap - organizational input required**."));

