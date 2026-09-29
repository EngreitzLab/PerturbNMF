"""Shared open-access PDF download layer for the search_*.py modules.

Ported from Tools/AGeneTic/src/search/pdf_download.py. Search functions pass whatever
identifiers they have; this module turns them into a PDF on disk using legal open-access
sources only, in this order:

  direct PDF urls from the search source (arXiv / OpenAlex / Crossref links)
  -> PMCID -> PMC open-access PDF (PMC Article Datasets on AWS, no auth)
  -> PMID -> (NCBI ID converter) -> PMCID -> PMC open-access PDF
  -> DOI -> Unpaywall best open-access PDF location

A paper with no open-access PDF is skipped with a reason, never raised. Downloads are
cached by filename in ``out_dir`` so re-runs are cheap.

Note: AGeneTic's Europe PMC ``/{pmcid}/fullTextPDF`` endpoint now returns 404 and the
europepmc.org / pmc.ncbi.nlm.nih.gov PDF pages block scripts (403 / JS challenge), so
PMC PDFs come from the PMC open-data S3 bucket instead (checked 2026-09-24).

Env: NCBI_EMAIL / PUBMED_EMAIL (NCBI courtesy), UNPAYWALL_EMAIL (required by Unpaywall;
falls back to NCBI_EMAIL / PUBMED_EMAIL, and Unpaywall is skipped if none is set).
"""
import os
import re
from pathlib import Path

import requests

ID_CONVERTER = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
PMC_S3 = "https://pmc-oa-opendata.s3.amazonaws.com"
UNPAYWALL = "https://api.unpaywall.org/v2/{doi}"

DEFAULT_TIMEOUT = 60
TOOL_NAME = "PerturbNMF-AGeneTic"


def _email():
    return os.environ.get("NCBI_EMAIL") or os.environ.get("PUBMED_EMAIL")


def _user_agent():
    email = _email()
    return f"{TOOL_NAME} (mailto:{email})" if email else TOOL_NAME


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")[:150]


def _is_pdf(content):
    return content[:5].startswith(b"%PDF")


def _download(url, dest, timeout=DEFAULT_TIMEOUT, params=None):
    """GET ``url`` and save to ``dest`` iff the body is a real PDF. Returns (path, reason)."""
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest.resolve()), "cached"
    try:
        resp = requests.get(url, params=params, timeout=timeout, allow_redirects=True,
                            headers={"User-Agent": _user_agent()})
    except requests.RequestException as e:
        return None, f"network error ({type(e).__name__})"
    if resp.status_code != 200 or not resp.content:
        return None, f"HTTP {resp.status_code}"
    if not _is_pdf(resp.content):
        return None, "response was not a PDF (likely paywall / landing page)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return str(dest.resolve()), "downloaded"


def pmid_to_ids(pmid, timeout=DEFAULT_TIMEOUT):
    """Resolve a PMID to {pmcid, doi} via the NCBI ID converter (empty dict on failure)."""
    params = {"ids": str(pmid), "format": "json", "tool": TOOL_NAME}
    if _email():
        params["email"] = _email()
    try:
        resp = requests.get(ID_CONVERTER, params=params, timeout=timeout)
        records = resp.json().get("records", []) if resp.status_code == 200 else []
    except (requests.RequestException, ValueError):
        return {}
    if not records:
        return {}
    rec = records[0]
    return {"pmcid": rec.get("pmcid"), "doi": rec.get("doi")}


def pmc_pdf_url(pmcid, timeout=DEFAULT_TIMEOUT):
    """Return (pdf_url, reason) for a PMCID from the PMC open-access S3 bucket."""
    pmcid = str(pmcid).upper()
    latest = None
    for version in (1, 2, 3):   # articles are versioned PMCxxxx.1, .2, ...; keep the newest
        try:
            resp = requests.get(f"{PMC_S3}/metadata/{pmcid}.{version}.json", timeout=timeout)
        except requests.RequestException as e:
            return None, f"PMC error ({type(e).__name__})"
        if resp.status_code != 200:
            break
        latest = resp.json()
    if latest is None:
        return None, "not in PMC open-access dataset"
    pdf_url = latest.get("pdf_url")
    if not pdf_url:
        return None, f"PMC has no open-access PDF (license: {latest.get('license_code')})"
    return pdf_url.replace("s3://pmc-oa-opendata", PMC_S3).split("?")[0], "pmc"


def unpaywall_pdf_url(doi, timeout=DEFAULT_TIMEOUT):
    """Return (pdf_url, reason) for a DOI's best open-access location on Unpaywall."""
    email = os.environ.get("UNPAYWALL_EMAIL") or _email()
    if not email:
        return None, "Unpaywall skipped (set UNPAYWALL_EMAIL)"
    try:
        resp = requests.get(UNPAYWALL.format(doi=doi), params={"email": email}, timeout=timeout)
        data = resp.json() if resp.status_code == 200 else {}
    except (requests.RequestException, ValueError) as e:
        return None, f"Unpaywall error ({type(e).__name__})"
    pdf_url = (data.get("best_oa_location") or {}).get("url_for_pdf")
    return (pdf_url, "unpaywall") if pdf_url else (None, "no open-access PDF on Unpaywall")


def download_paper(out_dir, name, pmid=None, pmcid=None, doi=None, pdf_urls=(),
                   timeout=DEFAULT_TIMEOUT):
    """Try every open-access route for one paper.

    Returns (pdf_path, message): pdf_path is None when nothing worked, and message then
    holds the reasons from each route tried. Never raises.
    """
    try:
        dest = Path(out_dir) / f"{_safe_name(name)}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            return str(dest.resolve()), "cached"
        reasons = []

        for url in [u for u in (pdf_urls or []) if u]:
            path, why = _download(url, dest, timeout)
            if path:
                return path, why
            reasons.append(f"direct url: {why}")

        if pmid and not pmcid:
            ids = pmid_to_ids(pmid, timeout)
            pmcid, doi = ids.get("pmcid"), doi or ids.get("doi")
        if pmcid:
            pdf_url, why = pmc_pdf_url(pmcid, timeout)
            if pdf_url:
                path, why = _download(pdf_url, dest, timeout)
                if path:
                    return path, why
            reasons.append(f"PMC {pmcid}: {why}")
        elif pmid:
            reasons.append("not in PMC")

        if doi:
            pdf_url, why = unpaywall_pdf_url(doi, timeout)
            if pdf_url:
                path, why = _download(pdf_url, dest, timeout)
                if path:
                    return path, why
            reasons.append(f"Unpaywall: {why}")

        return None, "; ".join(reasons) or "no identifier to download from"
    except Exception as e:  # one bad paper must never sink the search
        return None, f"unexpected error: {e}"


def report_downloads(source, query, results, out_dir):
    """Print searched / attempted / downloaded counts and the per-paper outcome.

    ``results`` is a list of (title, pdf_path, message) for every paper searched.
    """
    ok = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    print(f"[{source}] Searched: {len(results)} papers for {query!r}")
    print(f"[{source}] Download attempted: {len(results)} -> {out_dir}")
    for title, path, msg in ok:
        print(f"  [OK]   {title[:90]} -> {path} ({msg})")
    for title, _, msg in failed:
        print(f"  [FAIL] {title[:90]} ({msg})")
    print(f"[{source}] Successfully downloaded: {len(ok)}/{len(results)} PDFs")
