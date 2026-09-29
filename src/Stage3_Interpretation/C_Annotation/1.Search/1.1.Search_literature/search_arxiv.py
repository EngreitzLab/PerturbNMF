from langchain_community.retrievers import ArxivRetriever
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # llm_summary_agent / pdf_download
from llm_summary_agent import Summarize_Agent
from pdf_download import download_paper, report_downloads

# ArXiv API
def search_Arxiv(
    query: str = "", 
    max_results: int = 3,
    mini_handler = None,
    out_dir: str = None
) -> list[dict]:
    """Search ArXiv for papers related to the query

    When searching, you should consider:

    Args:
        query (str): The query to search for
        max_results (int): The maximum number of results to return, default is 3
        out_dir (str): Directory to download PDFs into; None skips downloading
    """

    # Initialize the ArxivRetriever object
    retriever = ArxivRetriever(
        top_k_results=max_results,
        load_max_docs=max_results,
        load_all_available_meta=True
    )

    # Search for papers related to the query
    related_papers = retriever.invoke(query)

    # Extract the relevant information from the search results
    data_list = []
    downloads = []
    for paper in related_papers:

        url = paper.metadata.get('Entry ID', 'No URL')
        
        content = "No abstract available."
        if paper.page_content:
            content = Summarize_Agent(paper.page_content, mini_handler)
        
        title = paper.metadata.get('Title', paper.metadata.get('title', 'No Title'))
        authors = paper.metadata.get('Authors', paper.metadata.get('authors', 'Unknown Authors'))
        
        if isinstance(authors, list):
            authors = ', '.join(authors)
            
        pdf_line = ''
        if out_dir:
            # http://arxiv.org/abs/2101.00001v1 -> https://arxiv.org/pdf/2101.00001v1
            arxiv_id = url.rstrip('/').rsplit('/abs/', 1)[-1] if '/abs/' in url else None
            pdf_urls = [f"https://arxiv.org/pdf/{arxiv_id}"] if arxiv_id else []
            pdf_path, msg = download_paper(out_dir, f"arXiv_{arxiv_id}" if arxiv_id else title,
                                           doi=paper.metadata.get('DOI'), pdf_urls=pdf_urls)
            downloads.append((title, pdf_path, msg))
            pdf_line = '\n' + 'PDF: ' + (pdf_path or f'Not downloaded ({msg})')

        data_list.append('Title: ' + title + '\n' + 'Url: ' + url + pdf_line + '\n' + 'Summary: ' + content)

    if out_dir:
        report_downloads("ArXiv", query, downloads, out_dir)

    return "\n".join(data_list) + '\n' + "Data source: ArXiv"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run search_Arxiv on one or more queries.")
    parser.add_argument("--query", nargs="+", required=True, help="One or more queries, each quoted.")
    parser.add_argument("--max_results", type=int, default=3, help="Max papers per query (capped at 15).")
    parser.add_argument("--out_dir", default=None, help="Download open-access PDFs here. Default: no downloads.")
    args = parser.parse_args()
    for q in args.query:
        print(f"===== {q}")
        print(search_Arxiv(query=q, max_results=args.max_results, out_dir=args.out_dir))
