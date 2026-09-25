"""Gather citable support for every claim an annotation rests on, one candidate file per program.

The annotation pass offers the model one small literature pool per program (one PubTator search
of 30 genes, capped at 25 papers), so most genes that drive a label arrive with no citable source.
This is the retrieval half of a separate CITATION PASS, run after the label is fixed:

  claims    every gene in `label_evidence.genes`, every regulator in `label_evidence.regulators`,
            and every `regulators[]` hypothesis with high or medium confidence.
  database  each significant enrichment term of THIS program that contains the gene (STRING,
            plus optional Enrichr/GO tables), and for GO terms the gene's own GO annotations to
            that term with the PMID each annotation cites (QuickGO, via UniProt). Also the gene's
            NCBI/Harmonizome summary.
  literature  DISCOVERY-FIRST. The citation that matters is usually the study that DISCOVERED the
            gene's role, not a recent paper that restates it. Recent topic hits (PubTator) mostly
            restate; so the pool is built from channels that surface the original work, the way
            citation-graph tools do (Semantic Scholar/Connected Papers "prior works", PaperQA2's
            citation traversal, citation counts). Backend: Europe PMC (EMBL-EBI; no key needed):
              most_cited  (symbol OR aliases) AND (label words) in title/abstract, primary
                          articles, sorted by citations — foundational papers are the most cited
              earliest    the same search, oldest first — first reports that are not heavily cited
              co_cited    references shared by >= 2 of the topic papers (reviews included: their
                          reference lists concentrate the originals) that name the gene
              curated     UniProt FUNCTION evidence (ECO:0000269, experimental) and NCBI GeneRIFs
                          matching the label words — curators attach these to the reporting paper
              topic       the targeted PubTator search (gene AND label words), as before
            Each paper carries year, journal, total citations, co-citation count, its channels and
            (after an esummary pass) whether it is a review. Sentences are the title/abstract
            sentences that name the gene or an alias.

Searching with the label's words finds papers that FIT the label. A citation from this pass
means "a paper links this gene to this process", not "the label is right".

All network results are cached under --cache-dir, so a rerun costs nothing.

Network: www.ncbi.nlm.nih.gov (PubTator3), mygene.info, rest.uniprot.org,
www.ebi.ac.uk (QuickGO, Europe PMC), eutils.ncbi.nlm.nih.gov. No API keys are needed.

Usage:
    python build_citation_candidates.py --dispatch <annotation_dispatch> --arm v3 \
        --enrichment string_enrichment_filtered.csv [--enrichr go_enrichment.tsv ...] \
        --ncbi-context ncbi_context.json --excluded-pmids excluded_pool_pmids.json \
        --cache-dir citation_cache --output-dir citation_candidates
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from answer_io import load_answer
from http_cache import CachedHttp
from verify_cited_pmids import fetch_pubmed_summaries, is_retracted

NON_PRIMARY_TYPES = {"Review", "Systematic Review", "Meta-Analysis", "Editorial", "Comment", "Letter"}

PUBTATOR = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
QUICKGO = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
MYGENE = "https://mygene.info/v3/query"
MOST_CITED_PER_CLAIM = 8
REVIEWS_AS_REFERENCE_SOURCES = 3   # reviews are never cited, but their reference lists concentrate the originals
REFERENCE_SOURCES_MAX = 14         # papers whose reference lists feed the co-citation backbone
EARLIEST_PER_CLAIM = 4
CO_CITED_PER_CLAIM = 5
CURATED_PER_CLAIM = 6
# Per-channel quotas so no one channel (GeneRIFs are numerous) crowds out the others.
CHANNEL_QUOTA = {"curated": 4, "co_cited": 4, "most_cited": 4, "earliest": 2, "topic": 3}
ALIASES_MAX = 6

ENRICHMENT_FDR = 0.05
PAPERS_PER_QUERY = 10
SENTENCES_PER_CLAIM = 6
DATABASE_TERMS_PER_CLAIM = 6
GO_PMIDS_PER_TERM = 3
LABEL_WORDS_MAX = 6
# Experimental GO evidence first; computational/electronic annotations cite no primary paper.
EXPERIMENTAL_ECO = {
    "ECO:0000269": "EXP", "ECO:0000314": "IDA", "ECO:0000315": "IMP", "ECO:0000316": "IGI",
    "ECO:0000353": "IPI", "ECO:0000270": "IEP",
}
GENERIC_WORDS = {
    "and", "the", "with", "from", "signaling", "signalling", "pathway", "state", "identity",
    "program", "process", "cell", "cells", "cellular", "regulation", "response", "axis",
    "driven", "associated", "machinery", "content", "module", "cluster", "mediated", "core",
    "genes", "gene", "high", "low", "like", "type", "activity", "activation", "induction",
    "none", "defensible", "distinguisher", "generic", "sibling",
}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def bare_symbol(value: str) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", str(value)).strip()


def extract_claims(answer: dict) -> List[dict]:
    claims, seen = [], set()
    evidence = answer.get("label_evidence") or {}
    for gene in evidence.get("genes", []):
        symbol = bare_symbol(gene.get("symbol", ""))
        if symbol and ("gene", symbol) not in seen:
            seen.add(("gene", symbol))
            claims.append({"kind": "gene", "symbol": symbol, "context": gene.get("why", ""),
                           "loading_rank": gene.get("loading_rank")})
    hypotheses = {bare_symbol(r.get("symbol", "")): r for r in answer.get("regulators", [])}
    for regulator in evidence.get("regulators", []):
        symbol = bare_symbol(regulator.get("symbol", ""))
        if symbol and ("regulator", symbol) not in seen:
            seen.add(("regulator", symbol))
            hypothesis = hypotheses.get(symbol, {})
            claims.append({"kind": "regulator", "symbol": symbol,
                           "context": hypothesis.get("hypothesis") or regulator.get("why", ""),
                           "role": hypothesis.get("role", "")})
    for symbol, hypothesis in hypotheses.items():
        if hypothesis.get("confidence") in {"high", "medium"} and ("regulator", symbol) not in seen:
            seen.add(("regulator", symbol))
            claims.append({"kind": "regulator", "symbol": symbol,
                           "context": hypothesis.get("hypothesis", ""), "role": hypothesis.get("role", "")})
    for index, claim in enumerate(claims, start=1):
        claim["claim_id"] = f"{'G' if claim['kind'] == 'gene' else 'R'}{index}"
    return claims


def label_words(answer: dict) -> List[str]:
    """The label's own process words, for the targeted search. Gene symbols are dropped."""
    text = " ".join(str(answer.get(k, "")) for k in ("label", "label_family", "label_distinguisher"))
    words = []
    for token in re.split(r"[^A-Za-z0-9β-]+", text):
        if len(token) < 4 or token.lower() in GENERIC_WORDS:
            continue
        if re.fullmatch(r"[A-Z0-9-]{2,}", token) and not token.isalpha():
            continue  # gene-like symbol (letters+digits)
        lowered = token.lower()
        if lowered not in words:
            words.append(lowered)
    return words[:LABEL_WORDS_MAX]


def load_enrichment(string_paths: List[Path], enrichr_paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in string_paths:
        frame = pd.read_csv(path)
        frames.append(pd.DataFrame({
            "program_id": frame["program_id"].astype(int),
            "source": "STRING " + frame["category"].astype(str),
            "term": frame["description"].astype(str),
            "term_id": frame["term"].astype(str),
            "fdr": frame["fdr"].astype(float),
            "genes": frame["inputGenes"].astype(str).str.split("|"),
        }))
    for path in enrichr_paths:
        frame = pd.read_csv(Path(path).expanduser(), sep="\t")
        term_id = frame["Term"].astype(str).str.extract(r"\((GO:\d+)\)")[0].fillna("")
        frames.append(pd.DataFrame({
            "program_id": frame["program_name"].astype(int),
            "source": "Enrichr " + Path(path).stem.split("_", 1)[-1],
            "term": frame["Term"].astype(str),
            "term_id": term_id,
            "fdr": frame["Adjusted P-value"].astype(float),
            "genes": frame["Genes"].astype(str).str.split(";"),
        }))
    enrichment = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return enrichment[enrichment["fdr"] < ENRICHMENT_FDR] if len(enrichment) else enrichment


def europepmc_work(record: dict) -> dict:
    """A Europe PMC `core` record reduced to what the candidate list needs."""
    abstract = re.sub(r"<[^>]+>", " ", record.get("abstractText") or "")
    journal = ((record.get("journalInfo") or {}).get("journal") or {}).get("title") or ""
    return {
        "pmid": str(record.get("pmid") or ""),
        "title": re.sub(r"<[^>]+>", "", record.get("title") or "").strip(),
        "abstract": re.sub(r"\s+", " ", abstract).strip(),
        "year": str(record.get("pubYear") or ""),
        "cited_by": record.get("citedByCount"),
        "journal": journal,
        "pubtypes": (record.get("pubTypeList") or {}).get("pubType") or [],
    }


class SupportFinder:
    def __init__(self, http: CachedHttp, excluded_pmids: set):
        self.http = http
        self.excluded = excluded_pmids
        self.uniprot: Dict[str, Optional[str]] = {}
        self.aliases: Dict[str, List[str]] = {}

    # ---- discovery-first literature ----------------------------------------------------------
    def gene_aliases(self, symbol: str) -> List[str]:
        """Older names matter: the foundational KDR paper says Flk-1, ETV2's says ER71/Etsrp."""
        if symbol not in self.aliases:
            data = self.http.get_json(f"{MYGENE}?" + urllib.parse.urlencode(
                {"q": f"symbol:{symbol}", "species": "human", "fields": "alias,generif"})) or {}
            hit = (data.get("hits") or [{}])[0]
            aliases = hit.get("alias") or []
            aliases = [aliases] if isinstance(aliases, str) else aliases
            keep = [a for a in aliases if len(a) >= 3 and re.fullmatch(r"[A-Za-z][A-Za-z0-9-]+", a)
                    and not re.fullmatch(r"(C\d+orf\d+|FLJ\d+|KIAA\d+|DKFZ\w+|MGC\d+)", a)]
            # UniProt protein names carry the names papers actually use ("VE-cadherin", "PECAM-1").
            query = urllib.parse.urlencode({
                "query": f"gene_exact:{symbol} AND organism_id:9606 AND reviewed:true",
                "fields": "protein_name", "format": "json", "size": 1,
            })
            entry = ((self.http.get_json(f"{UNIPROT}?{query}") or {}).get("results") or [{}])[0]
            description = entry.get("proteinDescription") or {}
            names = []
            for block in [description.get("recommendedName") or {}] + (description.get("alternativeNames") or []):
                names += [x.get("value", "") for x in block.get("shortNames") or []]
                full = (block.get("fullName") or {}).get("value", "")
                if full and len(full) <= 30:
                    names.append(full)
            names = [n for n in names if n and n.upper() != symbol.upper() and re.search(r"[A-Za-z]", n)]
            self.aliases[symbol] = list(dict.fromkeys(names + keep))[:ALIASES_MAX]
            generif = hit.get("generif") or []
            self.generifs = getattr(self, "generifs", {})
            self.generifs[symbol] = [generif] if isinstance(generif, dict) else generif
        return self.aliases[symbol]

    def europepmc_search(self, names: List[str], words: List[str], sort: str, n: int,
                         reviews: bool = False) -> List[dict]:
        gene = " OR ".join(f'TITLE_ABS:"{x}"' for x in names)
        process = " OR ".join(f"TITLE_ABS:{w}" for w in words)
        kind = 'PUB_TYPE:"Review"' if reviews else 'NOT PUB_TYPE:"Review" NOT PUB_TYPE:"Book"'
        query = f"({gene}) AND ({process}) AND SRC:MED {'AND ' if reviews else ''}{kind}"
        data = self.http.get_json(f"{EUROPEPMC}/search?" + urllib.parse.urlencode(
            {"query": query, "format": "json", "resultType": "core", "pageSize": n, "sort": sort})) or {}
        return [europepmc_work(r) for r in (data.get("resultList") or {}).get("result") or [] if r.get("pmid")]

    def europepmc_by_pmids(self, pmids: List[str]) -> List[dict]:
        works = []
        for start in range(0, len(pmids), 20):
            query = " OR ".join(f"EXT_ID:{p}" for p in pmids[start:start + 20]) + " AND SRC:MED"
            data = self.http.get_json(f"{EUROPEPMC}/search?" + urllib.parse.urlencode(
                {"query": query, "format": "json", "resultType": "core", "pageSize": 25})) or {}
            works += [europepmc_work(r) for r in (data.get("resultList") or {}).get("result") or [] if r.get("pmid")]
        return works

    def europepmc_references(self, pmid: str) -> List[str]:
        data = self.http.get_json(f"{EUROPEPMC}/MED/{pmid}/references?format=json&pageSize=1000") or {}
        return [str(r["id"]) for r in (data.get("referenceList") or {}).get("reference") or []
                if r.get("source") == "MED" and r.get("id")]

    def curated_pmids(self, symbol: str, words: List[str]) -> List[str]:
        query = urllib.parse.urlencode({
            "query": f"gene_exact:{symbol} AND organism_id:9606 AND reviewed:true",
            "fields": "cc_function", "format": "json", "size": 1,
        })
        data = self.http.get_json(f"{UNIPROT}?{query}") or {}
        uniprot = [e["id"] for r in data.get("results") or [] for c in r.get("comments") or []
                   for t in c.get("texts") or [] for e in t.get("evidences") or []
                   if e.get("source") == "PubMed" and e.get("evidenceCode") == "ECO:0000269"]
        self.gene_aliases(symbol)
        rifs = [r for r in self.generifs.get(symbol, []) if isinstance(r, dict)]
        scored = sorted(((sum(w in str(r.get("text", "")).lower() for w in words), str(r.get("pubmed")))
                         for r in rifs), key=lambda x: (-x[0], x[1]))
        generif = [pmid for score, pmid in scored if score > 0]
        return list(dict.fromkeys(uniprot + generif))[:CURATED_PER_CLAIM]

    def discovery_literature(self, symbol: str, words: List[str], topic: List[dict]) -> List[dict]:
        """Candidate papers from all channels, each with its gene-naming sentences."""
        if not words:
            return topic
        names = [symbol] + self.gene_aliases(symbol)
        papers: Dict[str, dict] = {}

        def add(work: dict, channel: str, cocited: int = 0):
            pmid = work.get("pmid")
            if not pmid or pmid in self.excluded:
                return
            entry = papers.setdefault(pmid, {"work": work, "channels": [], "cocited": 0})
            if channel not in entry["channels"]:
                entry["channels"].append(channel)
            entry["cocited"] = max(entry["cocited"], cocited)

        # An OR over the label words is loose (SMAD4 AND "endothelial OR specification" returns
        # cancer papers), so over-fetch and keep the hits that match the most distinct label words.
        def label_overlap(work):
            text = (work["title"] + " " + work["abstract"]).lower()
            return sum(w in text for w in words)
        pool = self.europepmc_search(names, words, "CITED desc", 25)
        most_cited = sorted(pool, key=lambda w: (-label_overlap(w), -(w.get("cited_by") or 0)))[:MOST_CITED_PER_CLAIM]
        for work in most_cited:
            add(work, "most_cited")
        for work in self.europepmc_search(names, words, "PUB_YEAR asc", EARLIEST_PER_CLAIM):
            add(work, "earliest")
        # Co-citation backbone: references shared by the most-cited hits, the most-cited reviews
        # and the PubTator topic hits.
        reviews = self.europepmc_search(names, words, "CITED desc", REVIEWS_AS_REFERENCE_SOURCES, reviews=True)
        sources = list(dict.fromkeys([w["pmid"] for w in most_cited + reviews] + [t["pmid"] for t in topic]))
        counts: Dict[str, int] = {}
        for source in sources[:REFERENCE_SOURCES_MAX]:
            for ref in set(self.europepmc_references(source)):
                counts[ref] = counts.get(ref, 0) + 1
        shared = [ref for ref, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= 2][:40]
        name_pattern = re.compile(r"(?<![A-Za-z0-9-])(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9-])", re.I)
        backbone = [w for w in self.europepmc_by_pmids(shared)
                    if name_pattern.search(w["title"] + " " + w["abstract"])
                    and "Review" not in w["pubtypes"]]
        backbone.sort(key=lambda w: -counts.get(w["pmid"], 0))
        for work in backbone[:CO_CITED_PER_CLAIM]:
            add(work, "co_cited", counts.get(work["pmid"], 0))
        for work in self.europepmc_by_pmids(self.curated_pmids(symbol, words)):
            add(work, "curated")
        for t in topic:
            papers.setdefault(t["pmid"], {"work": None, "channels": [], "cocited": 0})
            if "topic" not in papers[t["pmid"]]["channels"]:
                papers[t["pmid"]]["channels"].append("topic")

        literature = []
        for pmid, entry in papers.items():
            work = entry["work"]
            if work is None:  # PubTator-only paper: keep its already extracted sentences
                sentences = [dict(t) for t in topic if t["pmid"] == pmid]
                for s in sentences:
                    s.update(channels=entry["channels"], cited_by=None, cocited=0, journal="")
                literature += sentences[:2]
                continue
            text = f"{work['title']} {work['abstract']}"
            hits = [x.strip() for x in SENTENCE_SPLIT.split(text) if name_pattern.search(x) and len(x) < 600]
            hits.sort(key=lambda x: -sum(w in x.lower() for w in words))
            for sentence in hits[:2]:
                literature.append({
                    "pmid": pmid, "sentence": sentence, "title": work["title"], "year": work["year"],
                    "journal": work["journal"], "cited_by": work.get("cited_by"), "cocited": entry["cocited"],
                    "channels": entry["channels"],
                })
        # Fill each channel's quota (a paper found by several channels counts once, for the first
        # channel with room), then present oldest first so the lineage of the finding reads in order.
        chosen, used = set(), {c: 0 for c in CHANNEL_QUOTA}
        by_paper = {}
        for e in literature:
            by_paper.setdefault(e["pmid"], []).append(e)
        for channel in ("curated", "co_cited", "most_cited", "earliest", "topic"):
            ranked = sorted((pm for pm, es in by_paper.items() if channel in es[0]["channels"] and pm not in chosen),
                            key=lambda pm: -(by_paper[pm][0].get("cocited") or 0) * 1000 - (by_paper[pm][0].get("cited_by") or 0))
            for pm in ranked[: CHANNEL_QUOTA[channel]]:
                chosen.add(pm)
        kept = [e for e in literature if e["pmid"] in chosen]
        kept.sort(key=lambda e: (int(e["year"]) if str(e.get("year", "")).isdigit() else 9999, e["pmid"]))
        return kept

    def uniprot_accession(self, symbol: str) -> Optional[str]:
        if symbol not in self.uniprot:
            query = urllib.parse.urlencode({
                "query": f"gene_exact:{symbol} AND organism_id:9606 AND reviewed:true",
                "fields": "accession", "format": "json", "size": 1,
            })
            data = self.http.get_json(f"{UNIPROT}?{query}") or {}
            results = data.get("results") or []
            self.uniprot[symbol] = results[0]["primaryAccession"] if results else None
        return self.uniprot[symbol]

    def go_annotation_pmids(self, symbol: str, go_id: str) -> List[dict]:
        accession = self.uniprot_accession(symbol)
        if not accession or not go_id:
            return []
        data = None
        # QuickGO returns HTTP 500 on descendant queries for heavily annotated genes (ACTB, the
        # protocadherins); the exact term is the fallback.
        for usage in ("descendants", "exact"):
            query = urllib.parse.urlencode({
                "geneProductId": f"UniProtKB:{accession}", "goId": go_id, "goUsage": usage,
                "taxonId": 9606, "limit": 100,
            })
            data = self.http.get_json(f"{QUICKGO}?{query}")
            if data is not None:
                break
        data = data or {}
        found = {}
        for annotation in data.get("results") or []:
            reference = str(annotation.get("reference", ""))
            code = EXPERIMENTAL_ECO.get(annotation.get("evidenceCode", ""))
            if reference.startswith("PMID:") and code:
                pmid = reference.split(":", 1)[1]
                if pmid not in self.excluded:
                    found.setdefault(pmid, {"pmid": pmid, "evidence_code": code,
                                            "annotated_go": annotation.get("goId")})
        return list(found.values())[:GO_PMIDS_PER_TERM]

    def database_support(self, symbol: str, program_terms: pd.DataFrame, summaries: dict) -> List[dict]:
        support = []
        if len(program_terms):
            rows = program_terms[program_terms["genes"].apply(lambda genes: symbol in genes)]
            rows = rows.sort_values("fdr").drop_duplicates("term").head(DATABASE_TERMS_PER_CLAIM)
        else:  # a program with no significant term at all
            rows = pd.DataFrame()
        for _, row in rows.iterrows():
            go_id = row["term_id"] if str(row["term_id"]).startswith("GO:") else ""
            support.append({
                "source": row["source"], "term": row["term"], "term_id": row["term_id"],
                "fdr": float(row["fdr"]), "pmids": self.go_annotation_pmids(symbol, go_id),
            })
        if summaries.get(symbol):
            support.append({"source": "NCBI Gene summary", "term": str(summaries[symbol])[:400],
                            "term_id": "", "fdr": None, "pmids": []})
        return support

    def literature(self, symbol: str, words: List[str]) -> List[dict]:
        if not words:
            return []
        query = f"{symbol} AND ({' OR '.join(words)})"
        data = self.http.get_json(
            f"{PUBTATOR}/search/?" + urllib.parse.urlencode({"text": query, "size": PAPERS_PER_QUERY})
        ) or {}
        pmids = [str(r.get("pmid") or r.get("_id")) for r in data.get("results") or []]
        pmids = [p for p in pmids if p and p != "None" and p not in self.excluded]
        if not pmids:
            return []
        docs = self.http.get_json(f"{PUBTATOR}/publications/export/biocjson",
                                  body={"pmids": [int(p) for p in pmids]}) or {}
        docs = docs.get("PubTator3", docs) if isinstance(docs, dict) else docs
        sentences = []
        gene_pattern = re.compile(rf"(?<![A-Za-z0-9-]){re.escape(symbol)}(?![A-Za-z0-9-])", re.IGNORECASE)
        for doc in docs if isinstance(docs, list) else []:
            pmid = str(doc.get("pmid") or doc.get("id"))
            title, text_parts = "", []
            for passage in doc.get("passages", []):
                kind = (passage.get("infons") or {}).get("type", "")
                if kind == "title":
                    title = passage.get("text", "")
                if kind in ("title", "abstract"):
                    text_parts.append(passage.get("text", ""))
            year = str((doc.get("date") or doc.get("year") or ""))[:4]
            for sentence in SENTENCE_SPLIT.split(" ".join(text_parts)):
                sentence = sentence.strip()
                if not gene_pattern.search(sentence) or len(sentence) > 600:
                    continue
                score = 1 + sum(w in sentence.lower() for w in words)
                sentences.append({"pmid": pmid, "sentence": sentence, "title": title,
                                  "year": year, "score": score})
        sentences.sort(key=lambda s: -s["score"])
        picked, per_paper = [], {}
        for s in sentences:  # at most 2 sentences per paper, so one review does not fill the list
            if per_paper.get(s["pmid"], 0) >= 2:
                continue
            per_paper[s["pmid"]] = per_paper.get(s["pmid"], 0) + 1
            picked.append({k: v for k, v in s.items() if k != "score"})
            if len(picked) >= SENTENCES_PER_CLAIM:
                break
        return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--enrichment", action="append", default=[], type=Path)
    parser.add_argument("--enrichr", action="append", default=[], type=Path)
    parser.add_argument("--ncbi-context", required=True, type=Path)
    parser.add_argument("--excluded-pmids", type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--programs", help="comma list; default all answered")
    args = parser.parse_args()

    excluded = set()
    if args.excluded_pmids and args.excluded_pmids.exists():
        payload = json.loads(args.excluded_pmids.read_text())
        excluded = set(payload.get("retracted", [])) | set(payload.get("unresolved", []))
    enrichment = load_enrichment(args.enrichment, args.enrichr)
    context = json.loads(args.ncbi_context.read_text())
    http = CachedHttp(args.cache_dir)
    finder = SupportFinder(http, excluded)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    directories = sorted(args.dispatch.glob(f"{args.arm}_p*"), key=lambda d: int(d.name.split("_p")[-1]))
    wanted = {int(p) for p in args.programs.split(",")} if args.programs else None
    for directory in directories:
        pid = int(directory.name.split("_p")[-1])
        if (wanted and pid not in wanted) or not (directory / "answer.json").exists():
            continue
        answer = load_answer(directory / "answer.json")
        words = label_words(answer)
        claims = extract_claims(answer)
        program_terms = enrichment[enrichment["program_id"] == pid] if len(enrichment) else enrichment
        summaries = (context.get(str(pid)) or {}).get("gene_summaries") or {}
        for claim in claims:
            claim["database"] = finder.database_support(claim["symbol"], program_terms, summaries)
            topic = finder.literature(claim["symbol"], words)
            claim["literature"] = finder.discovery_literature(claim["symbol"], words, topic)
        # Publication type from PubMed: reviews are pointers to the originals, never the discovery
        # citation; retracted papers and notices are dropped outright.
        pmids = sorted({e["pmid"] for c in claims for e in c["literature"]})
        records = {}
        try:
            records = fetch_pubmed_summaries(pmids)
        except Exception as exc:
            print(f"  P{pid}: esummary failed ({exc}); review flags left unknown")
        for claim in claims:
            kept = []
            for e in claim["literature"]:
                record = records.get(e["pmid"]) or {}
                if record and is_retracted(record):
                    continue
                types = set(record.get("pubtype", [])) if record else set()
                e["pubtypes"] = sorted(types)
                e["is_review"] = bool(types & NON_PRIMARY_TYPES) if record else None
                e.setdefault("journal", record.get("source", ""))
                if not e.get("journal"):
                    e["journal"] = record.get("source", "")
                kept.append(e)
            claim["literature"] = kept
        out = {
            "program_id": pid, "label": answer.get("label", ""), "label_family": answer.get("label_family", ""),
            "label_distinguisher": answer.get("label_distinguisher", ""), "label_words": words, "claims": claims,
        }
        (args.output_dir / f"program_{pid}.json").write_text(json.dumps(out, indent=1))
        n_lit = sum(bool(c["literature"]) for c in claims)
        n_db = sum(any(d["source"] != "NCBI Gene summary" for d in c["database"]) for c in claims)
        n_go = sum(any(d["pmids"] for d in c["database"]) for c in claims)
        print(f"P{pid}: {len(claims)} claims; literature {n_lit}, enrichment term {n_db}, "
              f"GO-annotation PMID {n_go}; words={words}", flush=True)
    http.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
