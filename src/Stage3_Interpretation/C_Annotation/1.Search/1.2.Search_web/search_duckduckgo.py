from duckduckgo_search import DDGS

from tools.page_fetch import fetch_page_content_and_summarize


# @tool
def DuckDuckGoSearchTool(args, query: str, read_content: bool = True, return_num: int = 5, mini_handler=None):
    """
    Use the DDGS (DuckDuckGo Search) to get search results. Optionally, fetch content from the links and summarize.

    Args:
        query (str): The search query.
        read_content (bool): Whether to read the content of each link.
        return_num (int): The number of search results to return.
        mini_handler: The LLM handler for summarization.

    Returns:
        str: The search results or an error message.
    """

    try:
        # Perform a DuckDuckGo search using DDGS
        ddgs = DDGS()
        results = ddgs.text(query, max_results = return_num)  # Modify max_results if needed

        # If no results found
        if not results:
            return "No results found on DuckDuckGo."


        search_results = []

        for result in results:
            title = result.get('title')
            snippet = result.get('snippet', 'No snippet available.')
            link = result.get('link')


            # If read_content is True, fetch and parse the content of the linked page
            if read_content:
                page_content = fetch_page_content_and_summarize(args, link, mini_handler, False)
                search_results.append(page_content)
            else:
                search_results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")

        return "\n\n".join(search_results)

    except Exception as e:
        return f"Error during DuckDuckGo search: {e}"


if __name__ == "__main__":
    result = DuckDuckGoSearchTool(None, "Genetic disease", read_content=False, return_num=5)
    print(result)
