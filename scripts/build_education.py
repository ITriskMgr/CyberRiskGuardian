#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2026 Marc-André Léger
"""Rebuild the education package .docx and .pdf files from their Markdown sources.

    python3 scripts/build_education.py [education/en/01_Instructor_Guide.md ...]   (default: all eight documents)
    python3 scripts/build_education.py --port 2199 ...                              (fixed LibreOffice port)
    CRG_LO_PORT=2199 python3 scripts/build_education.py ...                         (same, by environment)

Each document is converted with pandoc, using its current .docx as the style reference (title page, footer,
heading and table styles); tables get explicit DXA widths (9,600 twips, weighted by content), the navy header row
and compact cell spacing; LibreOffice (headless, through its Python UNO bridge) updates the table of contents and
exports the PDF.

The running footer (package version) is set from PACKAGE_VERSION below: bump it with the package.

Dependencies (Education 2.1.2, NV-E06):
  - pandoc 3.x on PATH (https://pandoc.org; macOS: brew install pandoc; Debian/Ubuntu: apt install pandoc);
  - Python 3.10+ with python-docx (pip install python-docx);
  - LibreOffice 7.x or 24.x with its Python UNO bridge: `soffice` on PATH and `import uno` available to the Python that
    runs this script. On Debian/Ubuntu: apt install libreoffice-writer python3-uno, run with the system python3.
    On macOS, run with LibreOffice's own Python: /Applications/LibreOffice.app/Contents/Resources/python scripts/build_education.py.
  The Node `docx` package of package.json is NOT used here (it serves the report builder of the Pro edition).

LibreOffice listens on a local TCP port for the UNO bridge. By default a free port is picked for each run (so two
builds, or another soffice instance, never collide); --port / CRG_LO_PORT fix it. The listener accepts connections
from this computer only (host=localhost).
"""
import os, re, shutil, socket, subprocess, sys, tempfile, time
import docx
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = {  # source -> (toc, numbered sections)
    "education/en/01_Instructor_Guide.md": (True, True), "education/fr/01_Guide_enseignant.md": (True, True),
    "education/en/02_Student_Lab_Workbook.md": (True, False), "education/fr/02_Cahier_laboratoires.md": (True, False),
    "education/en/03_Assessment_Kit.md": (True, True), "education/fr/03_Trousse_evaluation.md": (True, True),
    "education/en/04_Case_Library.md": (False, True), "education/fr/04_Bibliotheque_cas.md": (False, True),
}
TOTAL = 9600
PACKAGE_VERSION = "2.1.2"   # written into the running footer of every document (the footer lives in the .docx reference, not in Markdown)
VERSION_RX = re.compile(r"((?:Education Package|Ensemble pédagogique CyberRiskGuardian) )\d+\.\d+\.\d+")


def fix_footer(path):
    """The footer text comes from the style reference (the previous .docx), so its package version is set here (2.1.2)."""
    d = docx.Document(path)
    n = 0
    for sec in d.sections:
        for part in (sec.footer, sec.header, sec.first_page_footer, sec.first_page_header):
            for para in part.paragraphs:
                for run in para.runs:
                    new = VERSION_RX.sub(lambda m: m.group(1) + PACKAGE_VERSION, run.text)
                    if new != run.text:
                        run.text = new; n += 1
    if n:
        d.save(path)
    return n


def fix_tables(path):
    d = docx.Document(path)
    for t in d.tables:
        rows = t.rows; ncol = len(t.columns)
        lens = [[] for _ in range(ncol)]; words = [8] * ncol
        for r in rows:
            for j, c in enumerate(r.cells[:ncol]):
                tx = c.text.strip(); lens[j].append(len(tx))
                words[j] = max(words[j], max((len(x) for x in tx.split()), default=0) + 2)
        w = []
        for j, L in enumerate(lens):
            mean = sum(L) / max(1, len(L)); mx = max(L) if L else 1
            w.append(max(float(min(words[j], 30)), 0.5 * mean + 0.5 * min(mx, 90)))
        s = sum(w); dx = [int(round(TOTAL * x / s)) for x in w]
        dx[-1] += TOTAL - sum(dx)
        tbl = t._tbl; pr = tbl.tblPr
        tw = pr.find(qn("w:tblW"))
        if tw is None:
            tw = OxmlElement("w:tblW"); pr.append(tw)
        tw.set(qn("w:type"), "dxa"); tw.set(qn("w:w"), str(TOTAL))
        grid = tbl.tblGrid
        for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(dx[j]))
        for r in rows:
            for j, tc in enumerate(r._tr.findall(qn("w:tc"))):
                tcpr = tc.get_or_add_tcPr()
                cw = tcpr.find(qn("w:tcW"))
                if cw is None:
                    cw = OxmlElement("w:tcW"); tcpr.insert(0, cw)
                cw.set(qn("w:type"), "dxa"); cw.set(qn("w:w"), str(dx[j] if j < len(dx) else dx[-1]))
                header = r._tr.find(qn("w:trPr")) is not None and r._tr.find(qn("w:trPr")).find(qn("w:tblHeader")) is not None
                if header and tcpr.find(qn("w:shd")) is None:
                    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "123A5C"); tcpr.append(shd)
                for para in tc.findall(qn("w:p")):
                    ppr = para.find(qn("w:pPr"))
                    if ppr is None:
                        ppr = OxmlElement("w:pPr"); para.insert(0, ppr)
                    if ppr.find(qn("w:spacing")) is None:
                        sp = OxmlElement("w:spacing"); sp.set(qn("w:after"), "40"); sp.set(qn("w:before"), "20")
                        st = ppr.find(qn("w:pStyle"))
                        if st is not None: st.addnext(sp)
                        else: ppr.insert(0, sp)
                    if header:
                        for run in para.findall(qn("w:r")):
                            rpr = run.find(qn("w:rPr"))
                            if rpr is None:
                                rpr = OxmlElement("w:rPr"); run.insert(0, rpr)
                            if rpr.find(qn("w:b")) is None: rpr.append(OxmlElement("w:b"))
                            if rpr.find(qn("w:color")) is None:
                                c = OxmlElement("w:color"); c.set(qn("w:val"), "FFFFFF"); rpr.append(c)
    d.save(path)


def free_port():
    """A TCP port nobody listens on right now (the kernel picks it; LibreOffice binds it a moment later)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def to_pdf(pairs, port=None):
    try:
        import uno
        from com.sun.star.beans import PropertyValue
    except ImportError:
        sys.exit("LibreOffice's Python UNO bridge is not available to this Python (import uno failed). "
                 "Install python3-uno (Debian/Ubuntu) or run the script with LibreOffice's own Python; see the docstring.")
    if shutil.which("soffice") is None:
        sys.exit("soffice (LibreOffice) is not on PATH.")
    port = int(port or os.environ.get("CRG_LO_PORT") or free_port())
    def pv(n, v):
        p = PropertyValue(); p.Name = n; p.Value = v; return p
    prof = tempfile.mkdtemp(prefix="crg-lo-")
    proc = subprocess.Popen(["soffice", f"-env:UserInstallation=file://{prof}", "--headless", "--invisible", "--norestore",
                             f"--accept=socket,host=localhost,port={port};urp;"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ctx = None
    for _ in range(120):
        try:
            local = uno.getComponentContext()
            res = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
            ctx = res.resolve(f"uno:socket,host=localhost,port={port};urp;StarOffice.ComponentContext"); break
        except Exception:
            if proc.poll() is not None:
                break
            time.sleep(0.5)
    if ctx is None:
        proc.kill(); shutil.rmtree(prof, ignore_errors=True)
        sys.exit(f"LibreOffice did not answer on port {port} within 60 s (is another soffice using the port? try --port).")
    print(f"libreoffice listening on localhost:{port}")
    desk = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    try:
        for src, out in pairs:
            doc = desk.loadComponentFromURL(uno.systemPathToFileUrl(src), "_blank", 0, (pv("Hidden", True),))
            idx = doc.getDocumentIndexes()
            for k in range(idx.getCount()):
                idx.getByIndex(k).update()
            doc.refresh()
            doc.storeToURL(uno.systemPathToFileUrl(out), (pv("FilterName", "writer_pdf_Export"),))
            doc.close(True)
            print("pdf ", os.path.relpath(out, ROOT))
    finally:
        try:
            desk.terminate()
        except Exception:
            pass
        proc.wait(timeout=60)
        shutil.rmtree(prof, ignore_errors=True)


def main(args):
    port = None
    if "--port" in args:
        i = args.index("--port"); port = args[i + 1]; args = args[:i] + args[i + 2:]
    for tool in ("pandoc",):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} is not on PATH (see the docstring for the dependencies).")
    srcs = args or list(DOCS)
    tmp = tempfile.mkdtemp(prefix="crg-edu-")
    pairs = []
    for rel in srcs:
        rel = os.path.relpath(os.path.abspath(rel), ROOT)
        toc, num = DOCS[rel]
        md = os.path.join(ROOT, rel); base = md[:-3]
        fr = "/fr/" in rel
        ref = os.path.join(tmp, os.path.basename(base) + "_ref.docx")
        shutil.copy(base + ".docx", ref)
        cmd = ["pandoc", md, "-o", base + ".docx", f"--reference-doc={ref}", "--columns=9999",
               "-M", "lang=" + ("fr-CA" if fr else "en-CA")]
        if toc:
            cmd += ["--toc", "--toc-depth=2", "-M", "toc-title=" + ("Table des matières" if fr else "Contents")]
        if num:
            cmd.append("--number-sections")
        subprocess.run(cmd, check=True)
        fix_tables(base + ".docx")
        fix_footer(base + ".docx")
        print("docx", rel[:-3] + ".docx")
        pairs.append((base + ".docx", base + ".pdf"))
    to_pdf(pairs, port)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1:])
