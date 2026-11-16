"""Sphinx configuration for the PerturbNMF documentation site.

The user guide and tutorials are not duplicated here: at build time the stage
READMEs and Jupyter notebooks under ``src/`` and ``examples/`` are copied into
``docs/guide/`` and ``docs/tutorials/`` (gitignored). Relative links inside the READMEs are rewritten so they
point at sibling generated pages or at the file on GitHub.
"""

from __future__ import annotations

import posixpath
import re
import shutil
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent

GITHUB_REPO = "https://github.com/EngreitzLab/PerturbNMF"
GITHUB_BRANCH = "main"

# -- Project ----------------------------------------------------------------

project = "PerturbNMF"
author = "PerturbNMF contributors"
copyright = "2026, PerturbNMF contributors"
release = "0.1.1"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx.ext.githubpages",  # writes .nojekyll
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "requirements.txt"]

# -- MyST / notebooks -------------------------------------------------------

myst_enable_extensions = ["attrs_block", "colon_fence", "deflist", "substitution"]
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3
nb_execution_mode = "off"  # notebooks need GPUs / large data; render source only

# READMEs were written for GitHub, not Sphinx: they jump heading levels and
# link to anchors Sphinx cannot resolve. Those are cosmetic, not build errors.
suppress_warnings = ["myst.header", "myst.xref_missing", "misc.highlighting_failure"]

# -- HTML -------------------------------------------------------------------

# Laid out like a pkgdown site: top navbar with name + version, full-width
# articles, and a right sidebar of project links on the home page.
html_theme = "pydata_sphinx_theme"
html_title = "PerturbNMF"
html_baseurl = "https://engreitzlab.github.io/PerturbNMF/"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
templates_path = ["_templates"]
html_show_sourcelink = False
html_context = {"default_mode": "light", "github_repo_url": GITHUB_REPO}
html_theme_options = {
    "navbar_start": ["navbar-logo", "pkg-version"],
    "navbar_align": "left",
    "navbar_center": ["pkg-navbar"],
    "navbar_end": ["search-field", "navbar-icon-links"],
    "navbar_persistent": [],  # drop the default search button; search-field replaces it
    "search_bar_text": "Search for",
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "article_header_start": [],  # no breadcrumb strip (pkgdown has none)
    "article_header_end": [],
    "secondary_sidebar_items": {"**": ["page-toc"], "index": ["home-sidebar"]},  # last match wins
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
    "icon_links": [
        {"name": "GitHub", "url": GITHUB_REPO, "icon": "fa-brands fa-github"},
    ],
}
# Left section nav only inside the multi-page sections (guide/, tutorials/),
# without pydata's "Collapse Sidebar" button.
html_sidebars = {
    page: [] for page in ("index", "quickstart", "installation", "explorer", "changelog", "citation")
} | {"guide/**": ["sidebar-nav-bs"], "tutorials/**": ["sidebar-nav-bs"]}

# -- Generated content ------------------------------------------------------

# (README path relative to repo root, generated page name under guide/)
READMES = [
    ("src/Stage1_Inference/README.md", "inference"),
    ("src/Stage1_Inference/sk-cNMF/README.md", "sk_cnmf"),
    ("src/Stage1_Inference/torch-cNMF/README.md", "torch_cnmf"),
    ("src/Stage2_Evaluation/README.md", "evaluation"),
    ("src/Stage2_Evaluation/A_Metrics/README.md", "metrics"),
    ("src/Stage2_Evaluation/B_Calibration/README.md", "calibration"),
    (
        "src/Stage2_Evaluation/B_Calibration/Slurm_version/U-test_perturbation_calibration/README.md",
        "calibration_utest",
    ),
    ("src/Stage2_Evaluation/B_Calibration/Slurm_version/CRT/README.md", "calibration_crt"),
    (
        "src/Stage2_Evaluation/B_Calibration/Slurm_version/Matched_cell_programDE/README.md",
        "calibration_matched_de",
    ),
    ("src/Stage3_Interpretation/README.md", "interpretation"),
    ("src/Stage3_Interpretation/A_Plotting/README.md", "plotting"),
    ("src/Stage3_Interpretation/B_Summarization/README.md", "summarization"),
    ("src/Stage3_Interpretation/C_Annotation/README.md", "annotation"),
]

# (notebook path relative to repo root, generated notebook name under tutorials/)
NOTEBOOKS = [
    ("examples/Inference/torch_cnmf_inference_tutorial.ipynb", "pbmc3k_torch_cnmf"),
    (
        "src/Stage1_Inference/sk-cNMF/JupterNote_Version/sk-cNMF_inference_pipeline.ipynb",
        "sk_cnmf_inference",
    ),
    (
        "src/Stage1_Inference/torch-cNMF/JupterNote_Version/torch_cnmf_inference_pipeline.ipynb",
        "torch_cnmf_inference",
    ),
    (
        "src/Stage2_Evaluation/A_Metrics/JupterNote_Version/cNMF_evaluation_pipeline.ipynb",
        "evaluation",
    ),
    (
        "src/Stage2_Evaluation/B_Calibration/JupterNote_Version/U-test_perturbation_calibration.ipynb",
        "calibration_utest",
    ),
    ("src/Stage2_Evaluation/B_Calibration/JupterNote_Version/CRT.ipynb", "calibration_crt"),
    (
        "src/Stage3_Interpretation/A_Plotting/JupterNote_Version/cNMF_k_selection.ipynb",
        "k_selection",
    ),
    (
        "src/Stage3_Interpretation/A_Plotting/JupterNote_Version/cNMF_program_analysis.ipynb",
        "program_analysis",
    ),
    (
        "src/Stage3_Interpretation/A_Plotting/JupterNote_Version/cNMF_perturbed_gene_analysis.ipynb",
        "perturbed_gene_analysis",
    ),
    (
        "src/Stage3_Interpretation/B_Summarization/JupterNote_Version/cNMF_compile_excel_table.ipynb",
        "excel_summary",
    ),
]

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg")
LINK_RE = re.compile(r"(\]\()([^)\s]+)(\))")
HTML_SRC_RE = re.compile(r"""(src=["'])([^"']+)(["'])""")


def _rewrite_target(target: str, readme_dir: str, page_of: dict[str, str]) -> str:
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE) or target.startswith("#"):
        return target
    path, _, anchor = target.partition("#")
    resolved = posixpath.normpath(posixpath.join(readme_dir, path))
    if resolved in page_of:
        return page_of[resolved] + ".md" + (f"#{anchor}" if anchor else "")
    if resolved.lower().endswith(IMAGE_EXT):
        return f"{GITHUB_REPO}/raw/{GITHUB_BRANCH}/{resolved}"
    kind = "tree" if (REPO_ROOT / resolved).is_dir() else "blob"
    return f"{GITHUB_REPO}/{kind}/{GITHUB_BRANCH}/{resolved}" + (f"#{anchor}" if anchor else "")


def _generate(app) -> None:
    guide_dir = DOCS_DIR / "guide"
    nb_dir = DOCS_DIR / "tutorials"

    page_of = {src: name for src, name in READMES}
    for src, name in READMES:
        text = (REPO_ROOT / src).read_text()
        readme_dir = posixpath.dirname(src)

        def sub(m: re.Match) -> str:
            return m.group(1) + _rewrite_target(m.group(2), readme_dir, page_of) + m.group(3)

        text = LINK_RE.sub(sub, text)
        text = HTML_SRC_RE.sub(sub, text)
        source_note = f"\n\n---\n\n*Source: [`{src}`]({GITHUB_REPO}/blob/{GITHUB_BRANCH}/{src})*\n"
        (guide_dir / f"{name}.md").write_text(text + source_note)

    for src, name in NOTEBOOKS:
        shutil.copy(REPO_ROOT / src, nb_dir / f"{name}.ipynb")


def setup(app):
    app.connect("builder-inited", _generate)
