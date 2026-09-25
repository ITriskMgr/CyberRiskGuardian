// ================================================================ 12. KRIs
add(h1("12. Key risk indicators"));
add(table(["ID", "KRI", "Scenarios", "Measurement", "Data source", "Owner", "Frequency", "Target", "Warning", "Critical"],
  D.KRIS, [6, 18, 7, 14, 11, 10, 8, 8, 8, 10], { size: 13 }));
add(spacer());
add(p("Report quarterly to the Steering Committee; any KRI at critical threshold escalates to executive leadership within one week."));

// ================================================================ 13. CASE VIEW
add(h1("13. Cross-check with the case escalation method"));
add(p("The case (s.8) defines a simpler normalized score (T x E x I, adjusted for control maturity) with escalation thresholds: executive review above 0.25, Board acceptance above 0.40. Mapping T = Pb(A), E = Pb(psi,A), I = (de + dm)/2 and control maturity C = theta (an analyst mapping - validate) gives:"));
const lvl = (x) => (x <= 0.07 ? "Low" : x <= 0.15 ? "Moderate" : x <= 0.25 ? "High" : "Critical");
add(table(["ID", "Current residual", "Level", "Post-treatment residual", "Level", "CRG status (appetite 0.30)"],
  S.map((s) => [s.id, s.norm_cur.toFixed(3), lvl(s.norm_cur), s.norm_post.toFixed(3), lvl(s.norm_post), s.class]), [8, 16, 14, 18, 14, 30], { size: 15 }));
add(spacer());
add(p("**Reading.** Under the case method, S8 is Critical today and S1, S2, S6 are High; after treatment all fall to Moderate or Low and within CIO acceptance authority. The CyberRiskGuardian ratio is more demanding because it compares damage directly with appetite. The two lenses should be reconciled before they are used for governance. In any case, the patient-safety clause of the appetite statement means S7 and S10 require clinical-leadership acceptance whatever their score."));

