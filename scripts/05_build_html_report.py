#!/usr/bin/env python3
"""
05_build_html_report.py
=======================
Builds a self-contained offline HTML version of REPORT.md (../report.html)
with all 4 high-resolution PNG figures embedded as base64 data URIs and
clean print/screen academic styling (zero external network dependencies).
"""

import base64
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "figures")
MD_PATH = os.path.join(BASE_DIR, "REPORT.md")
HTML_PATH = os.path.join(BASE_DIR, "report.html")


def img_to_data_uri(rel_path):
    full = os.path.join(BASE_DIR, rel_path)
    with open(full, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def inline_fmt(text):
    # Escape HTML entities first (except already-allowed <br>)
    text = text.replace("<br>", "___BR___")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("___BR___", "<br>")
    # Simple LaTeX math prettifier for offline readability without MathJax CDN
    def fmt_math(m):
        s = m.group(1)
        repls = [
            (r"180^\circ", "180°"), (r"45^\circ", "45°"), (r"2^5", "2⁵"),
            (r"10^{-5}", "10⁻⁵"), (r"\times", "×"), (r"\approx", "≈"),
            (r"\le", "≤"), (r"\ge", "≥"), (r"\ll", "≪"), (r"\implies", "⇒"),
            (r"\to", "→"), (r"\pm", "±"), (r"^\circ", "°"), (r"\circ", "°"),
            (r"\chi", "χ"), (r"\rho", "ρ"), (r"\Delta", "Δ"), (r"\sigma", "σ"),
            (r"\min", "min"), (r"\,", " "),
            (r"\text{corr}", "corr"), (r"\text{corrigé}", "corrigé"),
            (r"\text{intra-W/V}", "intra-W/V"), (r"\text{bilat}", "bilat"),
            (r"\text{OA}", "OA"), (r"^2", "²"),
            (r"_{WV}", "<sub>WV</sub>"), (r"_{corr}", "<sub>corr</sub>"),
            (r"_{corrigé}", "<sub>corrigé</sub>"), (r"_{OA}", "<sub>OA</sub>"),
            (r"_{intra-W/V}", "<sub>intra-W/V</sub>"), (r"_{bilat}", "<sub>bilat</sub>"),
        ]
        for a, b in repls:
            s = s.replace(a, b)
        s = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", s)
        s = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", s)
        s = s.replace("{", "").replace("}", "")
        return f'<span class="math">{s}</span>'

    text = re.sub(r"\$([^$]+)\$", fmt_math, text)
    # Code spans
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Bold + italic (handle nested ***foo* bar** as well as ***foo***)
    text = re.sub(r"\*\*\*([^*]+)\*\*\*", r"<strong><em>\1</em></strong>", text)
    text = re.sub(r"\*\*\*([^*]+)\*([^*]+)\*\*", r"<strong><em>\1</em>\2</strong>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    return text


def md_to_html(md_text):
    lines = md_text.splitlines()
    out = []
    i = 0
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            close_lists()
            i += 1
            continue
        if line.strip() == "---":
            close_lists()
            out.append("<hr>")
            i += 1
            continue
        # Image + caption
        m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", line.strip())
        if m_img:
            close_lists()
            alt, rel = m_img.group(1), m_img.group(2)
            uri = img_to_data_uri(rel)
            cap = ""
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("*") and lines[i + 1].strip().endswith("*"):
                cap = inline_fmt(lines[i + 1].strip().strip("*"))
                i += 1
            out.append(f'<figure><img src="{uri}" alt="{alt}"><figcaption>{cap}</figcaption></figure>')
            i += 1
            continue
        # Headings
        m_h = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m_h:
            close_lists()
            lvl = len(m_h.group(1))
            out.append(f"<h{lvl}>{inline_fmt(m_h.group(2))}</h{lvl}>")
            i += 1
            continue
        # Blockquote
        if line.startswith("> "):
            close_lists()
            bq = [line[2:]]
            while i + 1 < len(lines) and lines[i + 1].startswith("> "):
                i += 1
                bq.append(lines[i][2:])
            out.append(f'<blockquote>{inline_fmt(" ".join(bq))}</blockquote>')
            i += 1
            continue
        # Markdown Table
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|") and "---" in lines[i + 1]:
            close_lists()
            headers = [c.strip() for c in line.strip("|").split("|")]
            aligns_raw = [c.strip() for c in lines[i + 1].strip("|").split("|")]
            aligns = []
            for a in aligns_raw:
                if a.startswith(":") and a.endswith(":"):
                    aligns.append("center")
                elif a.endswith(":"):
                    aligns.append("right")
                else:
                    aligns.append("left")
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            tbl = ['<div class="table-wrap"><table><thead><tr>']
            for h, al in zip(headers, aligns):
                tbl.append(f'<th style="text-align:{al}">{inline_fmt(h)}</th>')
            tbl.append("</tr></thead><tbody>")
            for r in rows:
                tbl.append("<tr>")
                for c, al in zip(r, aligns):
                    tbl.append(f'<td style="text-align:{al}">{inline_fmt(c)}</td>')
                tbl.append("</tr>")
            tbl.append("</tbody></table></div>")
            out.append("".join(tbl))
            continue
        # Unordered list
        m_ul = re.match(r"^(\s*)\*\s+(.*)$", line)
        if m_ul:
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{inline_fmt(m_ul.group(2))}</li>")
            i += 1
            continue
        # Ordered list
        m_ol = re.match(r"^\d+\.\s+(.*)$", line)
        if m_ol:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{inline_fmt(m_ol.group(1))}</li>")
            i += 1
            continue
        # Paragraph
        close_lists()
        para = [line]
        while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#{1,3}\s|\*\s|\d+\.\s|\||>|!\[|---)", lines[i + 1]):
            i += 1
            para.append(lines[i].rstrip())
        joined = "<br>".join(p[:-2] if p.endswith("  ") else p for p in para) if any(p.endswith("  ") for p in para) else " ".join(para)
        out.append(f"<p>{inline_fmt(joined)}</p>")
        i += 1
    close_lists()
    return "\n".join(out)


def main():
    with open(MD_PATH) as f:
        md = f.read()
    body = md_to_html(md)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Étude 2D &amp; Forensique du Chiffre de Dorabella (1897)</title>
<style>
  :root {{
    --navy: #1E3A5F;
    --teal: #0F766E;
    --crimson: #BE123C;
    --slate: #475569;
    --bg: #F8FAFC;
    --card: #FFFFFF;
    --border: #E2E8F0;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.65;
    color: #0F172A;
    background: var(--bg);
    margin: 0;
    padding: 28px 18px 60px;
  }}
  main {{
    max-width: 1080px;
    margin: 0 auto;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 42px 52px;
    box-shadow: 0 6px 24px rgba(15, 23, 42, 0.06);
  }}
  .kpi-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 22px 0 28px;
  }}
  .kpi {{
    background: #F8FAFC;
    border: 1px solid var(--border);
    border-top: 3px solid var(--navy);
    border-radius: 8px;
    padding: 14px 16px;
  }}
  .kpi.accent {{ border-top-color: var(--crimson); }}
  .kpi.teal {{ border-top-color: var(--teal); }}
  .kpi .val {{ font-size: 1.45rem; font-weight: 800; color: var(--navy); line-height: 1.2; }}
  .kpi.accent .val {{ color: var(--crimson); }}
  .kpi.teal .val {{ color: var(--teal); }}
  .kpi .lbl {{ font-size: 0.78rem; color: var(--slate); margin-top: 5px; }}
  h1 {{
    color: var(--navy);
    font-size: 1.85rem;
    line-height: 1.28;
    margin-top: 0;
    border-bottom: 2px solid var(--navy);
    padding-bottom: 14px;
  }}
  h2 {{
    color: var(--navy);
    font-size: 1.32rem;
    margin-top: 34px;
    border-left: 4px solid var(--teal);
    padding-left: 12px;
  }}
  h3 {{
    color: #1E293B;
    font-size: 1.07rem;
    margin-top: 24px;
  }}
  p {{ margin: 11px 0; text-align: justify; }}
  ul, ol {{ margin: 10px 0 14px; padding-left: 24px; }}
  li {{ margin: 6px 0; }}
  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.88em;
    background: #F1F5F9;
    color: #0F172A;
    padding: 1px 5px;
    border-radius: 4px;
    border: 1px solid #E2E8F0;
  }}
  .math {{
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic;
  }}
  blockquote {{
    margin: 16px 0;
    padding: 12px 18px;
    background: #EFF6FF;
    border-left: 4px solid var(--navy);
    border-radius: 0 6px 6px 0;
    font-size: 0.94rem;
    color: #1E293B;
  }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 26px 0; }}
  .table-wrap {{ overflow-x: auto; margin: 18px 0; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.87rem;
  }}
  thead th {{
    background: var(--navy);
    color: white;
    font-weight: 600;
    padding: 9px 10px;
    border: 1px solid var(--navy);
  }}
  tbody td {{
    padding: 8px 10px;
    border: 1px solid var(--border);
    vertical-align: top;
  }}
  tbody tr:nth-child(even) {{ background: #F8FAFC; }}
  figure {{
    margin: 24px 0;
    padding: 14px;
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 8px;
    text-align: center;
  }}
  figure img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
  }}
  figcaption {{
    margin-top: 10px;
    font-size: 0.85rem;
    color: var(--slate);
    text-align: left;
    line-height: 1.5;
  }}
  @media (max-width: 780px) {{
    main {{ padding: 22px 18px; }}
    .kpi-strip {{ grid-template-columns: repeat(2, 1fr); }}
  }}
  @media print {{
    body {{ background: white; padding: 0; }}
    main {{ box-shadow: none; border: none; padding: 12px; max-width: 100%; }}
  }}
</style>
</head>
<body>
<main>
  <div class="kpi-strip">
    <div class="kpi accent">
      <div class="val">Z = &minus;4,76</div>
      <div class="lbl">Ségrégation W/V (18/86 croisements ; 0/1 000 MASC EN)</div>
    </div>
    <div class="kpi accent">
      <div class="val">27 / 86 (31,4%)</div>
      <div class="lbl">Sauts 180° (13/27 même-arc = 48,1%, p = 0,0012 intra-W/V)</div>
    </div>
    <div class="kpi teal">
      <div class="val">#1 / 105 (p = 0,0095)</div>
      <div class="lbl">Symétrie d'arcs à 180° sur les 4 axes (cos = 0,9746)</div>
    </div>
    <div class="kpi">
      <div class="val">+12,03 nats</div>
      <div class="lbl">LOO-CV Posture W/V + Miroir (+4 p., RMS générative 1,09σ)</div>
    </div>
  </div>
  {body}
</main>
</body>
</html>
"""
    with open(HTML_PATH, "w") as f:
        f.write(html)
    sz = os.path.getsize(HTML_PATH) / 1024.0
    print(f"[05] Built self-contained offline HTML report: {HTML_PATH} ({sz:.1f} KB)")


if __name__ == "__main__":
    main()
