import os
import requests

from tools.page_fetch import fetch_page_content_and_summarize


# @tool
def GoogleSearchTool(
    args,
    query: str,
    mini_handler,
    read_content: bool = True,
    return_num: int = 5,
    screenshot: bool = False,
    start: int = 1  
):
    """
    Use API to get search results from Google. Optionally, fetch content from the links and summarize.

    Args:
        query (str): The search query.
        mini_handler: The LLM handler for summarization.
        read_content (bool): Whether to read the content of each link.
        return_num (int): Number of results to return (1–10).
        screenshot (bool): Whether to take a screenshot.
        start (int): 1-based index of the first result to return (e.g., 11 for items 11–20).

    Returns:
        str: The search results or an error message.
    """

    # Basic validation
    if not isinstance(query, str) or not query.strip():
        return "Error: query must be a non-empty string."
    if not getattr(args, "google_api", None):
        return "Error: missing args.google_api (API key)."
    if not getattr(args, "search_engine_id", None):
        return "Error: missing args.search_engine_id (CSE cx)."
    try:
        return_num = int(return_num)
        start = int(start)
    except Exception:
        return "Error: return_num and start must be integers."

    if not (1 <= return_num <= 10):
        return "Error: return_num must be between 1 and 10."
    if start < 1 or start > 91:  # The official limit is usually around 100, but we conservatively use 91 to ensure start+num-1 <= 100
        return "Error: start must be between 1 and 91."

    try:
        api_key = args.google_api
        search_engine_id = args.search_engine_id

        url = 'https://www.googleapis.com/customsearch/v1'
        params = {
            'key': api_key,
            'cx': search_engine_id,
            'q': query.strip(),
            'num': return_num,
            'start': start
        }


        response = requests.get(url, params=params, timeout=20)

        if response.status_code != 200:
            try:
                err = response.json()
            except Exception:
                err = response.text
            return f"Error: {response.status_code}\n{err}"

        results_json = response.json()
        items = results_json.get('items', [])
        if not items:
            return "No results found."

        search_results = []
        for i, item in enumerate(items):
            title = item.get('title')
            link = item.get('link')
            snippet = item.get('snippet', 'No snippet available.')
            # print(f"Result {i}: {title}\nSnippet: {snippet}\nURL: {link}\n")
            # ipdb.set_trace()
            if read_content and link:
                try:
                    page_content = fetch_page_content_and_summarize(args, link, mini_handler, False)
                    search_results.append(page_content)
                except Exception as e:
                    search_results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")
                    
                # print(f"Fetched and summarized content from: {link}\n")
            else:
                search_results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")

        return "\n\n".join(search_results)

    except requests.RequestException as re:
        return f"Network error during Google search: {re}"
    except Exception as e:
        return f"Error during Google search: {e}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Google Search Tool")
    parser.add_argument('--google_api', type=str, default=os.getenv("GOOGLE_API_KEY"), help='Google API key')
    parser.add_argument('--search_engine_id', type=str, default=os.getenv("GOOGLE_CSE_ID"), help='Google CSE id (cx)')
    args = parser.parse_args()
    result = GoogleSearchTool(args, "Genetic disease", mini_handler=None, read_content=False, return_num=5, screenshot=False)
    print(result)
