const fs = require("fs");
const D = JSON.parse(fs.readFileSync("data30.json", "utf8"));
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  HeadingLevel, AlignmentType, LevelFormat, Header, Footer, PageNumber, PageBreak, BorderStyle,
  TableOfContents, PageOrientation,
} = require("docx");

const FONT = "Arial";
const W = 9360; // content width (Letter, 1" margins)
const NAVY = "1F3864";
const fmt0 = (n) => Math.round(n).toLocaleString("en-US");
const f2 = (n) => n.toFixed(2);
const money = (n) => "$" + Math.round(n).toLocaleString("en-US");
const k = (n) => "$" + Math.round(n / 1000).toLocaleString("en-US") + "k";

// ---------- inline markup: **bold**, _italic_
function runs(text, opts = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|_[^_]+_)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...opts }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...opts }));
  return out;
}
const p = (text, o = {}) => new Paragraph({ children: runs(text, o.run || {}), spacing: { after: 120 }, alignment: o.align, ...(o.para || {}) });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })], pageBreakBefore: true });
const h1n = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })] });
const bl = (t, lvl = 0) => new Paragraph({ numbering: { reference: "bul", level: lvl }, children: runs(t), spacing: { after: 60 } });
const nl = (t, ref = "num") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60 } });
const callout = (t, fill = "FFF2CC") => new Table({
  width: { size: W, type: WidthType.DXA }, columnWidths: [W],
  rows: [new TableRow({ children: [new TableCell({ width: { size: W, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill, color: "auto" },
    margins: { top: 100, bottom: 100, left: 140, right: 140 }, children: [new Paragraph({ children: runs(t, { size: 19 }) })] })] })],
});
const spacer = () => new Paragraph({ children: [], spacing: { after: 80 } });

function table(headers, rows, widths, o = {}) {
  const tot = widths.reduce((a, b) => a + b, 0);
  const scale = W / tot;
  const w = widths.map((x) => Math.floor(x * scale));
  w[w.length - 1] += W - w.reduce((a, b) => a + b, 0);
  const sz = o.size || 16;
  const cell = (txt, i, head, fill) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: NAVY, color: "auto" } : fill ? { type: ShadingType.CLEAR, fill, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: String(txt).split("\n").map((line) => new Paragraph({ children: runs(line, { size: sz, color: head ? "FFFFFF" : undefined, bold: head ? true : undefined }) })),
  });
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: w,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, false, o.fillFn ? o.fillFn(r, i, ri) : (ri % 2 ? "F2F2F2" : undefined))) })),
    ],
  });
}
const clsFill = (c) => (c.startsWith("Above") ? "F4CCCC" : c.startsWith("Approx") ? "FFF2CC" : "D9EAD3");
const S = D.SCEN;
const byId = Object.fromEntries(S.map((s) => [s.id, s]));
const INIT = D.INITIATIVES;
const initName = Object.fromEntries(INIT.map((i) => [i[0], i[1]]));

const above = S.filter((s) => s.class.startsWith("Above"));
const approx = S.filter((s) => s.class.startsWith("Approx"));
const below = S.filter((s) => s.class.startsWith("Below"));
const totE = S.reduce((a, s) => a + s.est, 0), totR = S.reduce((a, s) => a + s.res, 0);
const Y1 = D.TOT_INITIAL + D.TOT_RECUR;

const body = [];
const add = (...x) => body.push(...x);

