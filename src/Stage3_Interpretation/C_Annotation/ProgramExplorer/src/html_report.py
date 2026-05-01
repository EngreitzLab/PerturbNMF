"""
Library module for generating interactive HTML reports of gene program annotations.

Ported from 05_generate_html_report.py — contains only reusable functions,
no CLI / argparse / config-loading code.

Design A: Scientific Minimal - Professional scientific design with:
- Program stats at TOP
- 1:1 volcano plot with ALL program points
- PRIORITY_GENES labeling
- Full-text search
- Clean grayscale + teal accent

Functions:
    extract_program_stats   — extract stats from annotation markdown
    generate_report         — build the full HTML report from data files
    generate_design_a_html  — render the Design A HTML template
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from markdown import markdown
from .column_mapper import standardize_regulator_results

PRIORITY_GENES = [
    "Hif1a", "Epas1", "Arnt", "Fzd4", "Fzd6", "Idh2", "Mdh2", "Ogdh",
    "Hsp90ab1", "Hspa5", "Creb3l2", "Fkbp8", "Kdr", "Egln1", "Egln2", "Foxo1", "Foxo3"
]


def extract_program_stats(annotation_md: str) -> dict:
    """Extract stats from annotation markdown."""
    stats = {'top_loading': '', 'unique': '', 'celltype': ''}

    # Extract top-loading genes
    match = re.search(r'\*\*Top-loading genes?:\*\*\s*([^\n]+)', annotation_md, re.I)
    if match:
        stats['top_loading'] = match.group(1).strip()

    # Extract unique genes
    match = re.search(r'\*\*Unique genes?:\*\*\s*([^\n]+)', annotation_md, re.I)
    if match:
        stats['unique'] = match.group(1).strip()

    # Extract cell-type enrichment
    match = re.search(r'\*\*Cell-type enrichment:\*\*\s*([^\n]+)', annotation_md, re.I)
    if match:
        stats['celltype'] = match.group(1).strip()

    # Extract brief summary
    match = re.search(r'\*\*Brief Summary:\*\*\s*([^\n]+)', annotation_md, re.I)
    if match:
        stats['summary'] = match.group(1).strip()

    # Extract program label
    match = re.search(r'\*\*Program label:\*\*\s*([^\n]+)', annotation_md, re.I)
    if match:
        stats['label'] = match.group(1).strip()

    return stats


def generate_report(
    summary_csv: str,
    annotations_dir: str,
    enrichment_dir: str,
    volcano_csv: str | None,
    gene_loading_csv: str,
    output_html: str,
    regulator_significance_threshold: float = 0.05,
):
    """Generate Design A: Scientific Minimal HTML report."""

    # Load data
    summary_df = pd.read_csv(summary_csv)
    volcano_df = None
    if volcano_csv and os.path.exists(volcano_csv):
        volcano_df = standardize_regulator_results(
            pd.read_csv(volcano_csv),
            significance_threshold=regulator_significance_threshold,
        )

    # Process volcano data by program
    volcano_by_program = {}
    if volcano_df is not None:
        pvalue_col = (
            'adj_p_value'
            if 'adj_p_value' in volcano_df.columns and not volcano_df['adj_p_value'].isna().all()
            else 'p_value'
        )
        volcano_df['neg_log10_pvalue'] = -np.log10(volcano_df[pvalue_col].replace(0, 1e-300))
        volcano_df.loc[np.isinf(volcano_df['neg_log10_pvalue']), 'neg_log10_pvalue'] = 300

        for tid, group in volcano_df.groupby('program_id'):
            volcano_by_program[int(tid)] = [
                {
                    'g': row['grna_target'],
                    'fc': round(row['log_2_fold_change'], 3),
                    'p': round(row['neg_log10_pvalue'], 2),
                    's': bool(row['significant'])
                }
                for _, row in group.iterrows()
            ]

    # Build per-program data
    programs_data = []
    for _, row in summary_df.iterrows():
        topic_id = int(row['Topic'])
        topic_name = row['Name']

        # Read annotation
        ann_path = Path(annotations_dir) / f"topic_{topic_id}_annotation.md"
        annotation_md = ann_path.read_text(encoding='utf-8') if ann_path.exists() else ""

        # Extract stats
        stats = extract_program_stats(annotation_md)

        # Enrichment paths
        enr_rel = os.path.relpath(enrichment_dir, os.path.dirname(output_html))

        programs_data.append({
            'id': topic_id,
            'name': topic_name,
            'label': stats.get('label', topic_name),
            'summary': stats.get('summary', ''),
            'top_loading': stats.get('top_loading', ''),
            'unique': stats.get('unique', ''),
            'celltype': stats.get('celltype', ''),
            'annotation_html': markdown(annotation_md, extensions=['tables', 'fenced_code']),
            'annotation_text': annotation_md,  # For full-text search
            'kegg_fig': f"{enr_rel}/program_{topic_id}_kegg_enrichment.png",
            'process_fig': f"{enr_rel}/program_{topic_id}_process_enrichment.png",
            'volcano': volcano_by_program.get(topic_id, []),
        })

    generated_on = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Generate HTML
    html = generate_design_a_html(programs_data, len(programs_data), generated_on)

    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Design A (Scientific Minimal): {output_html}")
    print(f"  - {len(programs_data)} programs")


def generate_design_a_html(programs_data, num_programs, generated_on):
    """Generate Design A HTML."""

    # Build TOC
    toc_items = "\n".join([
        f'<a href="#" onclick="selectProgram({p["id"]}); return false;" data-id="{p["id"]}">'
        f'{p["id"]}. {p["name"]}</a>'
        for p in programs_data
    ])

    # Program selector options
    options = "\n".join([
        f'<option value="{p["id"]}">Program {p["id"]}: {p["name"]}</option>'
        for p in programs_data
    ])

    # Program data JS
    programs_js = f"window.PROGRAMS = {json.dumps({p['id']: p for p in programs_data})};\n"
    programs_js += f"window.PRIORITY_GENES = {json.dumps(PRIORITY_GENES)};\n"

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gene Program Annotations - Scientific Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg: #fafafa;
            --surface: #fff;
            --text: #1a1a1a;
            --muted: #666;
            --accent: #0d9488;
            --accent-light: #14b8a6;
            --border: #e5e5e5;
            --negative: #dc2626;
            --positive: #059669;
        }}
        [data-theme="dark"] {{
            --bg: #0a0a0a;
            --surface: #141414;
            --text: #e5e5e5;
            --muted: #888;
            --border: #2a2a2a;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            font-size: 14px;
        }}

        .header {{
            position: sticky;
            top: 0;
            z-index: 100;
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 8px 16px;
        }}
        .header-inner {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        .logo {{
            font-weight: 700;
            font-size: 14px;
            color: var(--accent);
        }}
        .program-select {{
            flex: 1;
            max-width: 450px;
            padding: 6px 10px;
            border: 1px solid var(--border);
            border-radius: 4px;
            background: var(--surface);
            color: var(--text);
            font-size: 13px;
        }}
        .search-input {{
            width: 200px;
            padding: 6px 10px;
            border: 1px solid var(--border);
            border-radius: 4px;
            background: var(--surface);
            color: var(--text);
            font-size: 13px;
        }}
        .btn {{
            padding: 6px 10px;
            border: 1px solid var(--border);
            border-radius: 4px;
            background: var(--surface);
            color: var(--text);
            cursor: pointer;
            font-size: 12px;
        }}
        .btn:hover {{ background: var(--border); }}
        .stats-label {{
            font-size: 11px;
            color: var(--muted);
        }}

        .layout {{
            display: grid;
            grid-template-columns: 280px 1fr;
            max-width: 1400px;
            margin: 0 auto;
        }}

        .sidebar {{
            position: sticky;
            top: 44px;
            height: calc(100vh - 44px);
            overflow-y: auto;
            padding: 8px;
            border-right: 1px solid var(--border);
            font-size: 11px;
        }}
        .toc a {{
            display: block;
            padding: 3px 6px;
            border-radius: 3px;
            color: var(--muted);
            text-decoration: none;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .toc a:hover {{ color: var(--text); background: var(--border); }}
        .toc a.active {{ color: var(--accent); font-weight: 600; }}

        .main {{
            padding: 20px 24px;
        }}

        /* Stats at TOP */
        .stats-panel {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }}
        .stat-item label {{
            display: block;
            font-size: 11px;
            font-weight: 600;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        .stat-item p {{
            font-size: 13px;
            line-height: 1.5;
        }}

        .program-header {{
            margin-bottom: 16px;
        }}
        .program-title {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .program-title span {{ color: var(--accent); }}
        .program-summary {{
            font-size: 15px;
            color: var(--muted);
            line-height: 1.6;
        }}

        .section {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }}
        .section h2 {{
            font-size: 18px;
            font-weight: 600;
            margin: 20px 0 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid var(--border);
        }}
        .section h2:first-child {{ margin-top: 0; }}
        .section h3 {{
            font-size: 15px;
            color: var(--accent);
            margin: 14px 0 6px;
        }}
        .section p {{ margin: 8px 0; line-height: 1.7; }}
        .section strong {{ color: var(--accent-light); }}
        .section hr {{ border: none; border-top: 1px solid var(--border); margin: 14px 0; }}
        .section ul, .section ol {{ margin: 8px 0; padding-left: 20px; }}
        .section li {{ margin: 4px 0; }}

        .figures {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}
        .figure {{
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
            background: #fff;
        }}
        .figure-title {{
            padding: 8px 12px;
            font-size: 12px;
            font-weight: 600;
            background: #f5f5f5;
            border-bottom: 1px solid var(--border);
            color: #333;
        }}
        .figure img {{ width: 100%; display: block; }}
        .figure .no-data {{
            padding: 40px;
            text-align: center;
            color: var(--muted);
            font-size: 12px;
        }}

        .volcano-section {{
            margin-top: 16px;
        }}
        .volcano-section h4 {{
            font-size: 14px;
            margin-bottom: 10px;
        }}
        #volcano-plot {{
            width: 500px;
            height: 500px;
            margin: 0 auto;
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 6px;
        }}

        @media (max-width: 900px) {{
            .layout {{ grid-template-columns: 1fr; }}
            .sidebar {{ display: none; }}
            .stats-grid {{ grid-template-columns: 1fr; }}
            .figures {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <header class="header">
        <div class="header-inner">
            <div class="logo">Gene Programs</div>
            <select class="program-select" id="program-select" onchange="selectProgram(this.value)">
                {options}
            </select>
            <input type="text" class="search-input" id="search" placeholder="Search full text..." onkeyup="filterPrograms()">
            <button class="btn" onclick="prev()">&#8592;</button>
            <button class="btn" onclick="next()">&#8594;</button>
            <button class="btn" onclick="toggleTheme()">Theme</button>
            <span class="stats-label">{num_programs} programs &middot; {generated_on}</span>
        </div>
    </header>

    <div class="layout">
        <aside class="sidebar">
            <nav class="toc" id="toc">{toc_items}</nav>
        </aside>
        <main class="main" id="main"></main>
    </div>

    <script>
    {programs_js}

    let current = null;
    const ids = Object.keys(PROGRAMS).map(Number).sort((a,b) => a-b);
    const programSearch = {{}};
    ids.forEach(id => {{
        const p = PROGRAMS[id] || {{}};
        const parts = [
            p.name,
            p.label,
            p.summary,
            p.top_loading,
            p.unique,
            p.celltype,
            p.annotation_text,
        ];
        const combined = parts.filter(Boolean).join(' ');
        programSearch[id] = combined.replace(/\\s+/g, ' ').toLowerCase();
    }});

    function selectProgram(id) {{
        id = parseInt(id);
        current = id;
        const p = PROGRAMS[id];
        if (!p) return;

        document.getElementById('program-select').value = id;
        document.querySelectorAll('.toc a').forEach(a => a.classList.toggle('active', parseInt(a.dataset.id) === id));
        history.replaceState(null, '', '#program-' + id);

        document.getElementById('main').innerHTML = `
            <div class="stats-panel">
                <div class="stats-grid">
                    <div class="stat-item">
                        <label>Top-Loading Genes</label>
                        <p>${{p.top_loading || 'N/A'}}</p>
                    </div>
                    <div class="stat-item">
                        <label>Unique Genes</label>
                        <p>${{p.unique || 'N/A'}}</p>
                    </div>
                    <div class="stat-item">
                        <label>Cell-Type Enrichment</label>
                        <p>${{p.celltype || 'N/A'}}</p>
                    </div>
                </div>
            </div>

            <div class="program-header">
                <h1 class="program-title"><span>Program ${{id}}</span> — ${{p.name}}</h1>
                <p class="program-summary">${{p.summary || ''}}</p>
            </div>

            <div class="section">
                ${{p.annotation_html}}
            </div>

            <div class="figures">
                <div class="figure">
                    <div class="figure-title">KEGG Pathway Enrichment</div>
                    <img src="${{p.kegg_fig}}" onerror="this.outerHTML='<div class=no-data>No data</div>'">
                </div>
                <div class="figure">
                    <div class="figure-title">Biological Process Enrichment</div>
                    <img src="${{p.process_fig}}" onerror="this.outerHTML='<div class=no-data>No data</div>'">
                </div>
            </div>

            <div class="volcano-section">
                <h4>Perturbation Effects (All Program Data)</h4>
                <div id="volcano-plot"></div>
            </div>
        `;

        renderVolcano(p.volcano);
    }}

    function renderVolcano(data) {{
        if (!data || data.length === 0) {{
            document.getElementById('volcano-plot').innerHTML = '<p style="text-align:center;padding:80px;color:#888">No perturbation data</p>';
            return;
        }}

        const sig = data.filter(d => d.s);
        const nonsig = data.filter(d => !d.s);

        // Labels: top 8 by p-value + priority genes
        const sorted = [...sig].sort((a,b) => b.p - a.p);
        const top8 = sorted.slice(0, 8).map(d => d.g);
        const toLabel = data.filter(d => top8.includes(d.g) || (PRIORITY_GENES.includes(d.g) && d.s));

        const traceSig = {{
            x: sig.map(d => d.fc),
            y: sig.map(d => d.p),
            mode: 'markers',
            name: 'Significant',
            marker: {{ color: sig.map(d => d.fc > 0 ? '#dc2626' : '#2563eb'), size: 7 }},
            text: sig.map(d => `<b>${{d.g}}</b><br>log2FC: ${{d.fc}}<br>-log10(p): ${{d.p}}`),
            hoverinfo: 'text'
        }};

        const traceNon = {{
            x: nonsig.map(d => d.fc),
            y: nonsig.map(d => d.p),
            mode: 'markers',
            name: 'Non-significant',
            marker: {{ color: '#ccc', size: 5, opacity: 0.6 }},
            text: nonsig.map(d => `<b>${{d.g}}</b><br>log2FC: ${{d.fc}}<br>-log10(p): ${{d.p}}`),
            hoverinfo: 'text'
        }};

        const annotations = toLabel.map(d => ({{
            x: d.fc, y: d.p, text: d.g, showarrow: false,
            font: {{ size: 10, color: '#000' }}, yshift: 10
        }}));

        Plotly.newPlot('volcano-plot', [traceNon, traceSig], {{
            width: 500, height: 500,
            margin: {{ t: 30, b: 50, l: 50, r: 30 }},
            xaxis: {{ title: 'log2 Fold Change', zeroline: true }},
            yaxis: {{ title: '-log10(p-value)' }},
            showlegend: true,
            legend: {{ orientation: 'h', y: -0.15, x: 0.5, xanchor: 'center' }},
            annotations: annotations,
            template: 'plotly_white'
        }}, {{responsive: true}});
    }}

    function filterPrograms() {{
        const q = document.getElementById('search').value.toLowerCase();
        document.querySelectorAll('.toc a').forEach(a => {{
            const id = a.dataset.id;
            const searchText = programSearch[id] || '';
            a.style.display = searchText.includes(q) ? '' : 'none';
        }});
    }}

    function toggleTheme() {{
        document.body.dataset.theme = document.body.dataset.theme === 'dark' ? '' : 'dark';
    }}
    function prev() {{ const i = ids.indexOf(current); if (i > 0) selectProgram(ids[i-1]); }}
    function next() {{ const i = ids.indexOf(current); if (i < ids.length-1) selectProgram(ids[i+1]); }}
    document.addEventListener('keydown', e => {{ if (e.key === 'ArrowLeft') prev(); if (e.key === 'ArrowRight') next(); }});

    (function() {{
        const m = location.hash.match(/program-(\\d+)/);
        selectProgram(m ? parseInt(m[1]) : ids[0]);
    }})();
    </script>
</body>
</html>'''

    return html
