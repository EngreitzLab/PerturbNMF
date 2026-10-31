"""Publish a built program viewer into a static site folder (e.g. a GitHub Pages repo clone): the default output.

    python publish_viewer.py --viewer <run_dir>/Report/program_viewer.html --site_dir <site> --dataset_name "<name>"

writes

    <site_dir>/<slug>/viewer.html   the viewer (program_viewer.py output), with its "All projects" button pointing to
                                    --home_url (default ../index.html, the root index below)
    <site_dir>/<slug>/site.json     dataset name, programs, date, page ("viewer.html"), read by the root index
    <site_dir>/index.html           one card per dataset folder with a site.json, linking to its viewer

and removes the older per-program pages of that dataset folder (index.html, P<k>.html, from program_site.py) unless
--keep_pages. It does not commit or push.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from program_site import page, slugify  # noqa: E402


def viewer_programs(path):
    """P<k> ids in the viewer's embedded PROGRAMS object, in numeric order."""
    html = Path(path).read_text()
    m = re.search(r"const PROGRAMS\s*=\s*", html)
    if not m:
        raise ValueError(
            f"{path} has no PROGRAMS object: not a program_viewer.py output"
        )
    programs, _ = json.JSONDecoder().raw_decode(html[m.end() :])
    return [f"P{k}" for k in sorted(programs, key=int)]


HOME_BTN = re.compile(r'<a class="homebtn" id="homeBtn" href="[^"]*"([^>]*)>')


def set_home(html, url):
    """The viewer with its "All projects" button linking to url (shown), or hidden when url is empty."""
    m = HOME_BTN.search(html)
    if not m:
        raise ValueError(
            "the viewer has no All projects button: rebuild it with the current program_viewer.py"
        )
    attrs = m.group(1).replace(" hidden", "")
    tag = f'<a class="homebtn" id="homeBtn" href="{escape(url, quote=True)}"{attrs}{"" if url else " hidden"}>'
    return html[: m.start()] + tag + html[m.end() :]


def root_index(site_dir):
    """Root page: one card per dataset folder with a site.json, linking to its page (the viewer)."""
    cards = []
    for meta in sorted(Path(site_dir).glob("*/site.json")):
        d = json.loads(meta.read_text())
        href = f"{meta.parent.name}/{d.get('page') or 'viewer.html'}"
        cards.append(
            f'<a class="pcard" href="{escape(href)}"><div class="pname">{escape(d["dataset_name"])}</div>'
            f'<div class="pdesc">{len(d["programs"])} programs &middot; updated '
            f"{escape(d.get('updated', ''))}</div></a>"
        )
    return f'<section class="hero"><h1 class="title">Program reports</h1></section><div class="plist">{"".join(cards)}</div>'


def main():
    p = argparse.ArgumentParser(
        description="Publish a program viewer into a static site folder and rebuild its root index."
    )
    p.add_argument(
        "--viewer", required=True, help="Viewer HTML written by program_viewer.py."
    )
    p.add_argument(
        "--site_dir", required=True, help="Site root (e.g. a GitHub Pages repo clone)."
    )
    p.add_argument(
        "--dataset_name",
        required=True,
        help="Dataset label; its slug names the dataset folder.",
    )
    p.add_argument(
        "--home_url",
        default="../index.html",
        help="Link of the viewer's 'All projects' button; '' hides it. Default: the site's root index.",
    )
    p.add_argument(
        "--keep_pages",
        action="store_true",
        help="Keep older per-program pages (index.html, P<k>.html) in the dataset folder.",
    )
    args = p.parse_args()

    site = Path(args.site_dir)
    out = site / slugify(args.dataset_name)
    out.mkdir(parents=True, exist_ok=True)
    programs = viewer_programs(args.viewer)
    (out / "viewer.html").write_text(
        set_home(Path(args.viewer).read_text(encoding="utf-8"), args.home_url),
        encoding="utf-8",
    )
    today = datetime.now().strftime("%Y-%m-%d")
    (out / "site.json").write_text(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "programs": programs,
                "updated": today,
                "page": "viewer.html",
            },
            indent=2,
        )
        + "\n"
    )
    removed = []
    if not args.keep_pages:
        for f in [out / "index.html", *out.glob("P*.html")]:
            if f.exists():
                f.unlink()
                removed.append(f.name)
    (site / "index.html").write_text(
        page(
            "Program reports", "", "index.html", root_index(site), f"Generated {today}"
        )
    )
    print(
        f"[done] {len(programs)} program(s) -> {out / 'viewer.html'}; root index rebuilt"
        + (f"; removed {len(removed)} older page(s)" if removed else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
