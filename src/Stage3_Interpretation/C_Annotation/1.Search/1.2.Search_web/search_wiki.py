import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 1.Search: llm_summary_agent
from llm_summary_agent import Summarize_Agent  # noqa: E402

# Wikipedia API (MediaWiki action API; no langchain_community needed)
WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "PerturbNMF-AGeneTic/0.1 (literature search)"
MAX_CHARS = 4000   # page text kept per article (the intro and first sections)


def _wiki_get(params):
    resp = requests.get(WIKI_API, params={**params, "format": "json", "formatversion": 2},
                        headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def search_Wiki(
    query: str = "",
    max_results: int = 3,
    mini_handler = None
) -> str:
    """Search Wikipedia for articles related to the query

    Args:
        query (str): The query to search for
        max_results (int): The maximum number of results to return, default is 3
        mini_handler: The LLM handler for summarizing page text; None keeps the text
    """
    try:
        # Search for article titles related to the query
        hits = _wiki_get({"action": "query", "list": "search", "srsearch": query,
                          "srlimit": max_results})["query"]["search"]
        if not hits:
            return ''
        # Fetch plain-text page content for the titles
        pages = _wiki_get({"action": "query", "prop": "extracts", "explaintext": 1,
                           "titles": "|".join(h["title"] for h in hits)})["query"]["pages"]
    except Exception as e:
        return f"Error during Wikipedia search: {e}"
    text = {p["title"]: p.get("extract") or "" for p in pages}

    # Extract the relevant information from the search results, in search order
    data_list = []
    for hit in hits:
        title = hit["title"]
        url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        page = text.get(title, "")[:MAX_CHARS]
        content = Summarize_Agent(page, mini_handler) if page else 'No content available.'
        data_list.append('Title: ' + title + '\n' + \
            '             Url: ' + url + '\n' +
                        'Summary: ' + content)

    return '\n'.join(data_list) + '\n' + "Data source: Wikipedia"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run search_Wiki on one or more queries.")
    parser.add_argument("--query", nargs="+", required=True, help="One or more queries, each quoted.")
    parser.add_argument("--max_results", type=int, default=3, help="Max articles per query.")
    args = parser.parse_args()
    for q in args.query:
        print(f"===== {q}")
        print(search_Wiki(query=q, max_results=args.max_results))
