#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Marc-André Léger
"""Cross-engine conformance test for the CyberRiskGuardian calculation engines (CRG 1.9.2, X-1 / X-2).

One test, the same copy in every repository (Threat Live-feed plugin, Core plugin, CRG Pro Quantitative), run
against every `crg_calc.py` engine and every MCP calculator server it can find, so the unified rules hold
everywhere:

  X-1  CVSS: the vector wins and a disagreeing stored score is flagged; only the eleven CVSS v4.0 Base metrics are
       accepted - a vector with Threat (/E:U) or Environmental (/MAV:L) metrics is refused with a message that
       routes exploitation evidence to Pb(psi,A), never scored as CVSS-BT / CVSS-BE.
  X-2  Zero tolerated risk (Pb(A), Pb(psi,A), mu(E) or CVSS = 0) -> estimated 0, tolerated 0, ratio None, verdict
       "Not applicable (tolerated risk 0)": never "Above tolerance", never a crash. Invalid inputs (ranges, dm < de,
       theta 0, appetite 0 or 1, reductions above 1, blank CVSS, score above 10, booleans, numeric strings) are
       rejected with a structured error; missing reductions mean 0.
  Regression: MediBec 30 scenarios 84,305 / 41,023 and CanaRetail 27,526 / 16,049 wherever the fixtures exist.
  Book worked example (ch06, AV:L vector 9.3): Re 3,554.46, ratio 1.09.

Usage
    python3 test_cross_engine.py                       # engines and servers of this repository
    python3 test_cross_engine.py --engine pro=/path/to/crg_calc.py --mcp pro=/path/to/crg_pro_server.py
    python3 test_cross_engine.py --no-mcp              # Python engines only (no `mcp` client package needed)

Exit 1 on any failure. Offline and deterministic.
"""
import argparse, asyncio, glob, importlib.util, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []
NA = "Not applicable (tolerated risk 0)"
VEC = "CVSS:4.0/AV:L/AC:L/AT:N/PR:N/UI:P/VC:H/VI:H/VA:H/SC:H/SI:H/SA:H"       # book ch06 worked example, 9.3
BASE = dict(pba=0.70, pbx=0.60, de=0.60, dm=0.80, theta=0.50, mu=0.65, red_p=0.65, red_i=0.70)


def check(label, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + label + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(label)


def repo_root():
    d = HERE
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "plugins")) or os.path.isfile(os.path.join(d, ".claude-plugin", "marketplace.json")):
            return d
        d = os.path.dirname(d)
    return HERE


def discover(root):
    engines = {}
    for p in glob.glob(os.path.join(root, "plugins", "*", "skills", "*", "scripts", "crg_calc.py")) + \
             glob.glob(os.path.join(root, "plugins", "*", "scripts", "crg_calc.py")):
        engines[p.split(os.sep + "plugins" + os.sep)[1].split(os.sep)[0]] = p
    servers = {}
    for p in glob.glob(os.path.join(root, "plugins", "*", "mcp", "*server*.py")):
        servers[p.split(os.sep + "plugins" + os.sep)[1].split(os.sep)[0]] = p
    return engines, servers


def fixtures(root):
    out = {}
    for name, est, res, pats in (
            ("MediBec 30", 84305, 41023, ["examples/medibec/source/medibec30.json", "plugins/*/evals/_shared/medibec30.json",
                                           "plugins/*/evals/*/fixtures/medibec30.json"]),
            ("CanaRetail 10", 27526, 16049, ["examples/canaretail/source/canaretail10.json", "plugins/*/skills/*/scripts/fixtures/canaretail10.json",
                                             "plugins/*/scripts/fixtures/canaretail10.json", "plugins/*/evals/_shared/canaretail10.json"])):
        for pat in pats:
            hits = glob.glob(os.path.join(root, pat))
            if hits:
                out[name] = (hits[0], est, res)
                break
    return out


def load(path):
    name = "crg_engine_" + str(abs(hash(path)))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(mod)
    return mod


def scenario(sid="X", vector=VEC, score=None, **over):
    v = dict(BASE, **over)
    s = {"id": sid, "params": {"PbA": {"v": v["pba"]}, "Pbx": {"v": v["pbx"]}, "De": {"v": v["de"]}, "Dm": {"v": v["dm"]},
                               "Th": {"v": v["theta"]}, "Mu": {"v": v["mu"]}}}
    if "red_p" in v:
        s["red_p"] = v["red_p"]
    if "red_i" in v:
        s["red_i"] = v["red_i"]
    if vector:
        s["cvss"] = vector
    if score is not None:
        s["cvss_score"] = score
    return s


def is_none(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


GREEK = str.maketrans({"ψ": "psi", "δ": "d", "θ": "theta", "μ": "mu", "–": "-", "—": "-"})


def ascii_msg(msg):
    """Engines may spell the parameters with Greek letters (desktop) or ASCII (plugins): compare on ASCII."""
    return str(msg).translate(GREEK)


# ----------------------------------------------------------------------------------------------- Python engines
def rejects(C, s, appetite=0.35, needle=None):
    try:
        C.calc(s, appetite, 1000)
    except ValueError as e:        # CRGInputError subclasses ValueError in the reference engine
        msg = "; ".join(getattr(e, "errors", None) or [str(e)])
        return (needle is None) or (needle.lower() in ascii_msg(msg).lower()), msg
    except Exception as e:         # a crash (ZeroDivisionError, KeyError, TypeError) is not a structured rejection
        return False, f"CRASH {type(e).__name__}: {e}"
    return False, "accepted"


def verdict(C, r):
    return C.classify(r["ratio"])


def test_engine(label, path, fx):
    print(f"\n== Python engine: {label} ({path})")
    C = load(path)
    # X-1 vector wins, disagreeing score flagged
    r = C.calc(scenario(), 0.35, 1000)
    check(f"[{label}] worked example: CVSS 9.3, Re 3,554.46, ratio 1.09",
          r["cvss"] == 9.3 and round(r["est"], 2) == 3554.46 and round(r["ratio"], 2) == 1.09, f"{r['cvss']} {r['est']:.2f}")
    r = C.calc(scenario(score=9.9), 0.35, 1000)
    check(f"[{label}] vector + stored 9.9: the vector's 9.3 is used and the disagreement is flagged",
          r["cvss"] == 9.3 and any("disagree" in w.lower() for w in r.get("warnings", [])), str(r.get("warnings"))[:120])
    ok, msg = rejects(C, scenario(vector=VEC + "/E:U"), needle="Pb(psi,A)")
    check(f"[{label}] /E:U (Threat metric) refused, exploitation evidence routed to Pb(psi,A)", ok, msg[:140])
    ok, msg = rejects(C, scenario(vector=VEC + "/MAV:L"), needle="CVSS-B only")
    check(f"[{label}] /MAV:L (Environmental metric) refused as CVSS-B only", ok, msg[:140])
    # X-2 zero tolerated risk
    for k in ("pba", "pbx", "mu"):
        try:
            r = C.calc(scenario(**{k: 0.0}), 0.35, 1000)
            check(f"[{label}] {k} = 0: estimated 0, tolerated 0, ratio None, verdict '{NA}'",
                  r["est"] == 0 and r["tol"] == 0 and is_none(r["ratio"]) and verdict(C, r) == NA, f"{r['ratio']!r} {verdict(C, r)}")
        except Exception as e:
            check(f"[{label}] {k} = 0 must not crash", False, f"{type(e).__name__}: {e}")
    try:
        r = C.calc(scenario(vector=None, score=0.0), 0.35, 1000)
        check(f"[{label}] CVSS score 0 (no vector): verdict '{NA}'", r["tol"] == 0 and verdict(C, r) == NA, verdict(C, r))
    except Exception as e:
        check(f"[{label}] CVSS 0 must not crash", False, f"{type(e).__name__}: {e}")
    # X-2 invalid inputs: structured rejection, never a crash
    for name, s, app, needle in (
            ("Pb(A) 1.2", scenario(pba=1.2), 0.35, "Pb(A)"),
            ("Pb(psi,A) -0.1", scenario(pbx=-0.1), 0.35, "Pb(psi,A)"),
            ("dm < de", scenario(de=0.8, dm=0.6), 0.35, "de"),
            ("theta 0", scenario(theta=0), 0.35, "theta"),
            ("theta 1.5", scenario(theta=1.5), 0.35, "theta"),
            ("mu(E) 1.1", scenario(mu=1.1), 0.35, "mu(E)"),
            ("red_p 1.2", scenario(red_p=1.2), 0.35, "0-1"),
            ("red_i -0.1", scenario(red_i=-0.1), 0.35, "0-1"),
            ("appetite 0", scenario(), 0.0, "appetite"),
            ("appetite 1", scenario(), 1.0, "appetite"),
            ("no vector and no score", scenario(vector=None), 0.35, "CVSS required"),
            ("blank score", scenario(vector=None, score=""), 0.35, "CVSS"),
            ("score 11", scenario(vector=None, score=11), 0.35, "10"),
            ("invalid vector", scenario(vector="CVSS:4.0/AV:X"), 0.35, "invalid"),
            ("boolean mu(E)", scenario(mu=True), 0.35, "number"),
            ("numeric string Pb(A)", scenario(pba="0.7"), 0.35, "number"),
            ("NaN Pb(A)", scenario(pba=float("nan")), 0.35, None)):
        ok, msg = rejects(C, s, app, needle)
        check(f"[{label}] rejects {name}", ok, msg[:120])
    try:
        s = scenario(); del s["red_p"]; del s["red_i"]
        r = C.calc(s, 0.35, 1000)
        check(f"[{label}] missing reductions mean 0: residual = estimated", round(r["res"], 2) == 3554.46 and r["res"] == r["est"], f"{r['res']:.2f}")
    except Exception as e:
        check(f"[{label}] missing reductions mean 0", False, f"{type(e).__name__}: {e}")
    # NV-045 sensitivity (engines that ship shift(): TI, Pro, desktop): central never shifted or clamped, directional clamp
    if hasattr(C, "shift"):
        edge = scenario(pba=1.0, pbx=0.02, theta=0.03, red_p=0.0, red_i=1.0)
        try:
            c0, hi, lo = C.shift(edge, 0), C.shift(edge, 0.10), C.shift(edge, -0.10)
            v = lambda t, k: t["params"][k]["v"] if isinstance(t["params"][k], dict) else t["params"][k]
            check(f"[{label}] sensitivity central case = unshifted scenario (Pb(A) 1.00, Pb(psi,A) 0.02, theta 0.03 kept)",
                  (v(c0, "PbA"), v(c0, "Pbx"), v(c0, "Th"), c0.get("red_p") or 0) == (1.0, 0.02, 0.03, 0.0))
            check(f"[{label}] higher case never milder than central (Pb(A) stays 1.00, red_p stays 0, theta stays 0.03)",
                  v(hi, "PbA") == 1.0 and (hi.get("red_p") or 0) == 0.0 and v(hi, "Th") == 0.03, str((v(hi, "PbA"), hi.get("red_p"), v(hi, "Th"))))
            check(f"[{label}] lower case never harsher than central (Pb(psi,A) stays 0.02, red_i stays 1.00)",
                  v(lo, "Pbx") == 0.02 and lo.get("red_i") == 1.0, str((v(lo, "Pbx"), lo.get("red_i"))))
        except Exception as e:
            check(f"[{label}] sensitivity shift must not crash", False, f"{type(e).__name__}: {e}")
    # regression fixtures
    for name, (path_, est, res) in fx.items():
        try:
            out = C.run(json.load(open(path_, encoding="utf-8")))
            t = out["totals"]
            check(f"[{label}] {name} regression {est:,} / {res:,}", round(t["est"]) == est and round(t["res"]) == res,
                  f"{t['est']:,.0f} / {t['res']:,.0f}")
        except Exception as e:
            check(f"[{label}] {name} regression", False, f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------------------------------- MCP servers
async def mcp_calls(cmd, calls):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    env = {"PYTHONDONTWRITEBYTECODE": "1", "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "CRG_DEFAULT_APPETITE": "0.30", "CRG_FACTOR": "1000"}
    params = StdioServerParameters(command=cmd[0], args=cmd[1:], env=env)
    res = []
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            names = [t.name for t in tools.tools]
            for tool, args in calls:
                if tool not in names:                      # optional tool (e.g. sensitivity on the Core server)
                    res.append({"isError": True, "exception": f"tool {tool} not offered"})
                    continue
                try:
                    out = await s.call_tool(tool, args)
                    txt = [x.text for x in out.content if getattr(x, "type", "") == "text"]
                    sc = getattr(out, "structured_content", None) or getattr(out, "structuredContent", None)
                    val = sc.get("result", sc) if isinstance(sc, dict) else None
                    if val is None and txt:
                        try:
                            val = json.loads(txt[0]) if len(txt) == 1 else [json.loads(t) for t in txt]
                        except Exception:
                            val = txt
                    res.append({"isError": bool(getattr(out, "is_error", getattr(out, "isError", False))), "value": val})
                except Exception as e:
                    res.append({"isError": True, "exception": repr(e)[:300]})
    return names, res


def structured_error(x, needle=None):
    """A unified rejection is a tool result (not a protocol error) carrying an `error` key, with a readable message."""
    v = x.get("value")
    if isinstance(v, list) and v and isinstance(v[0], dict):
        v = v[0]
    if not isinstance(v, dict) or "error" not in v:
        return False, (x.get("exception") or str(v))[:140]
    msg = json.dumps(v, ensure_ascii=False)
    return (needle is None) or (needle.lower() in ascii_msg(msg).lower()), msg[:140]


def test_mcp(label, path):
    print(f"\n== MCP server: {label} ({path})")
    try:
        import mcp  # noqa: F401
    except ImportError:
        print("SKIP: the `mcp` client package is not installed (pip install \"mcp>=1.2\")")
        return
    a = dict(BASE, appetite=0.35)
    calls = [
        ("crg_risk", dict(a, cvss_vector=VEC)),                                   # 0 worked example
        ("crg_risk", dict(a, cvss_vector=VEC, cvss_score=9.9)),                   # 1 disagreeing stored score
        ("crg_risk", dict(a, cvss_vector=VEC + "/E:U")),                          # 2 Threat metric
        ("crg_risk", dict(a, cvss_vector=VEC + "/MAV:L")),                        # 3 Environmental metric
        ("cvss4_score", {"vector": VEC + "/E:U"}),                                # 4
        ("cvss4_score", {"vector": VEC}),                                         # 5
        ("crg_risk", dict(a, pba=0.0, cvss_vector=VEC)),                          # 6 zero tolerated
        ("crg_risk", dict(a, cvss_score=0.0)),                                    # 7 CVSS 0
        ("crg_risk", dict(a, pba=1.2, cvss_vector=VEC)),                          # 8 invalid
        ("crg_risk", dict(a, pbx=-0.1, cvss_vector=VEC)),                         # 9
        ("crg_risk", dict(a, de=0.8, dm=0.6, cvss_vector=VEC)),                   # 10
        ("crg_risk", dict(a, theta=0.0, cvss_vector=VEC)),                        # 11
        ("crg_risk", dict(a, appetite=0.0, cvss_vector=VEC)),                     # 12
        ("crg_risk", dict(a, red_p=1.2, cvss_vector=VEC)),                        # 13
        ("crg_risk", dict(a)),                                                    # 14 no CVSS at all
        ("crg_risk", dict(a, cvss_score=11)),                                     # 15 score above 10
        ("cvss4_score", {"vector": "garbage"}),                                   # 16 invalid vector
        ("sensitivity", dict(a, pba=1.0, red_p=0.0, cvss_vector=VEC)),            # 17 NV-045 (servers that have the tool)
    ]
    try:
        names, res = asyncio.run(mcp_calls([sys.executable, path], calls))
    except Exception as e:
        check(f"[{label}] MCP server starts (mcp {getattr(sys.modules.get('mcp'), '__version__', '?')})", False, repr(e)[:200])
        return
    check(f"[{label}] MCP server starts and lists crg_risk and cvss4_score", "crg_risk" in names and "cvss4_score" in names, str(names))
    v = res[0].get("value") or {}
    check(f"[{label}] crg_risk worked example: CVSS 9.3, Re 3,554.46",
          isinstance(v, dict) and v.get("cvss") == 9.3 and round(v.get("est", 0), 2) == 3554.46, str(v)[:120])
    v = res[1].get("value") or {}
    check(f"[{label}] crg_risk vector + stored 9.9: 9.3 used, warning",
          isinstance(v, dict) and v.get("cvss") == 9.3 and any("disagree" in w.lower() for w in v.get("warnings", [])), str(v.get("warnings") if isinstance(v, dict) else v)[:120])
    ok, msg = structured_error(res[2], "Pb(psi,A)")
    check(f"[{label}] crg_risk /E:U refused with routing to Pb(psi,A) (structured error)", ok, msg)
    ok, msg = structured_error(res[3], "CVSS-B only")
    check(f"[{label}] crg_risk /MAV:L refused as CVSS-B only (structured error)", ok, msg)
    ok, msg = structured_error(res[4], "Pb(psi,A)")
    check(f"[{label}] cvss4_score /E:U refused (structured error)", ok, msg)
    v = res[5].get("value") or {}
    check(f"[{label}] cvss4_score base vector = 9.3", isinstance(v, dict) and v.get("base_score") == 9.3, str(v)[:100])
    for i, name in ((6, "Pb(A) = 0"), (7, "CVSS score 0")):
        v = res[i].get("value") or {}
        check(f"[{label}] crg_risk {name}: verdict '{NA}', no crash",
              isinstance(v, dict) and v.get("classification") == NA and is_none(v.get("ratio")), (res[i].get("exception") or str(v))[:120])
    if "sensitivity" in names:
        v = res[17].get("value") or {}
        c = v.get("central", {}) if isinstance(v, dict) else {}
        check(f"[{label}] sensitivity: central case not clamped (Pb(A) 1.00 kept: central estimated = crg_risk estimated) and higher >= central >= lower",
              isinstance(v, dict) and round(c.get("estimated", -1), 2) == round((res[0].get("value") or {}).get("est", 0) * (1.0 / BASE["pba"]), 2)
              and v["higher"]["estimated"] >= c["estimated"] >= v["lower"]["estimated"] and v["higher"]["residual"] >= c["residual"],
              str(v)[:160])
    for i, name, needle in ((8, "Pb(A) 1.2", "Pb(A)"), (9, "Pb(psi,A) -0.1", "Pb(psi,A)"), (10, "dm < de", "de"),
                            (11, "theta 0", "theta"), (12, "appetite 0", "appetite"), (13, "red_p 1.2", "0-1"),
                            (14, "no CVSS", "CVSS required"), (15, "score 11", "10"), (16, "invalid vector", "vector")):
        ok, msg = structured_error(res[i], needle)
        check(f"[{label}] structured error for {name}", ok, msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", action="append", default=[], metavar="LABEL=PATH", help="extra crg_calc.py to test")
    ap.add_argument("--mcp", action="append", default=[], metavar="LABEL=PATH", help="extra MCP server to test")
    ap.add_argument("--only-extra", action="store_true", help="skip this repository's own engines and servers")
    ap.add_argument("--no-mcp", action="store_true")
    ap.add_argument("--fixture", action="append", default=[], metavar="NAME=PATH:EST:RES", help="extra regression fixture")
    a = ap.parse_args()
    root = repo_root()
    engines, servers = ({}, {}) if a.only_extra else discover(root)
    for item in a.engine:
        k, _, v = item.partition("="); engines[k or v] = v
    for item in a.mcp:
        k, _, v = item.partition("="); servers[k or v] = v
    fx = fixtures(root)
    for item in a.fixture:
        k, _, v = item.partition("="); p, est, res = v.rsplit(":", 2); fx[k] = (p, int(est), int(res))
    print(f"repository root: {root}")
    print("engines: " + ", ".join(f"{k} -> {v}" for k, v in engines.items()))
    print("servers: " + ", ".join(f"{k} -> {v}" for k, v in servers.items()))
    print("fixtures: " + ", ".join(f"{k} ({v[1]:,} / {v[2]:,})" for k, v in fx.items()))
    if not engines:
        check("at least one crg_calc.py engine found", False, root)
    for k, v in engines.items():
        test_engine(k, v, fx)
    if not a.no_mcp:
        for k, v in servers.items():
            test_mcp(k, v)
    print(f"\n{'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
