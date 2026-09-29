from langchain_community.retrievers import PubMedRetriever
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # llm_summary_agent / pdf_download
from llm_summary_agent import Summarize_Agent
from pdf_download import download_paper, report_downloads

# PubMed API
# @tool
def search_PubMed(
    query: str = "", 
    max_results: int = 3,
    mini_handler = None,
    out_dir: str = None
) -> list[dict]:
    """Search PubMed for articles related to the query

    When searching, you should consider:

    1. Search Fields:
    - Title [ti]
    - Abstract [ab]
    - Author [au]
    - Journal [jour]
    - MeSH Terms [mesh]
    - Publication Date [dp]
    - DOI [doi]

    2. Boolean Operators:
    - AND
    - OR 
    - NOT
    - Use parentheses () for complex queries

    3. Search Syntax Examples:
    - Basic: diabetes mellitus
    - Field-specific: glucose[ti] AND insulin[mesh]
    - Date range: (2020[dp]:2024[dp])
    - Author search: Smith J[au]
    - Complex: ((diabetes mellitus[mesh] OR hyperglycemia[ti]) AND treatment[ti]) NOT type 1[tiab]

    Args:
        query (str): The query to search for
        max_results (int): The maximum number of results to return, default is 50
        out_dir (str): Directory to download open-access PDFs into; None skips downloading
    """

    # Initialize the PubMedRetriever object
    retriever = PubMedRetriever(
        top_k_results=max_results,
    )

    # Search for articles related to the query
    related_articles = retriever.invoke(query)

    # Extract the relevant information from the search results
    data_list = []
    downloads = []
    for article in related_articles:
        # ipdb.set_trace()
        url = "https://pubmed.ncbi.nlm.nih.gov/" + article.metadata.get('uid', '')
        content = "No abstract available."
        if article.page_content:
            content = Summarize_Agent(article.page_content, mini_handler)
        title = article.metadata.get('Title', 'No Title')
        if type(title) == dict:
            title = title['#text']

        pdf_line = ''
        if out_dir:
            pmid = article.metadata.get('uid', '')
            pdf_path, msg = download_paper(out_dir, pmid or title, pmid=pmid)
            downloads.append((title, pdf_path, msg))
            pdf_line = '\n ' + 'PDF: ' + (pdf_path or f'Not downloaded ({msg})')

        data_list.append('Title: ' + title + '\n '+ "Url: "+ url + pdf_line + '\n ' + 'Summary: ' + content)

    if out_dir:
        report_downloads("PubMed", query, downloads, out_dir)

    return "\n".join(data_list) + '\n' + "Data source: PubMed"



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run search_PubMed on one or more queries.")
    parser.add_argument("--query", nargs="+", required=True, help="One or more queries, each quoted.")
    parser.add_argument("--max_results", type=int, default=3, help="Max papers per query (capped at 15).")
    parser.add_argument("--out_dir", default=None, help="Download open-access PDFs here. Default: no downloads.")
    args = parser.parse_args()
    for q in args.query:
        print(f"===== {q}")
        print(search_PubMed(query=q, max_results=args.max_results, out_dir=args.out_dir))
