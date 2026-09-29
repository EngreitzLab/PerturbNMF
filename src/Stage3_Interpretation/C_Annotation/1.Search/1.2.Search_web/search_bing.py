import os
import time
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from tools.page_fetch import fetch_page_content_and_summarize
os.environ['DISPLAY'] = ':99'

ua = UserAgent()


def BingSearchTool(args, query: str, mini_handler, read_content: bool = True, return_num: int = 5, screenshot: bool = False):
    """
    Use Selenium to get search results from Bing. Optionally, fetch content from the links and summarize.

    Args:
        args: Namespace: The argument namespace containing configurations.
        query (str): The search query.
        mini_handler: The LLM handler for summarization.
        read_content (bool): Whether to read the content of each link.
        return_num (int): The number of search results to return.
        screenshot (bool): Whether to take a screenshot.

    Returns:
        str: Search results or an error message.
    """
    driver = None
    results = []


    url = f"https://www.bing.com/search?q={query}"

    user_agent = ua.random

    options = Options()
    if not getattr(args, 'visualize', False):
        options.add_argument("--headless")
    options.add_argument(f"user-agent={user_agent}")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-application-cache")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1550,1000")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--incognito")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--blink-settings=imagesEnabled=false")  

    service = Service(args.chrome_driver)
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(url)

    time.sleep(5)

    driver.set_window_size(1280, 800)

    if screenshot:
        driver.save_screenshot("bing.png")

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    search_results = soup.find_all("li", class_="b_algo")
    success_count = 0

    for result in search_results:
        title_elem = result.find("h2")
        if not title_elem:
            continue

        title = title_elem.get_text().strip()

        link_elem = result.find("a")
        if not link_elem or not link_elem.has_attr("href"):
            continue

        link = link_elem["href"]

        if not (link.startswith('http://') or link.startswith('https://')):
            continue

        if read_content:
            try:
                page_content = fetch_page_content_and_summarize(args, link, mini_handler, False)
                results.append(f"Title: {title}\nURL: {link}\n\n Content:{page_content}")
            except:
                continue
        else:
            snippet_elem = result.find("p")
            snippet = snippet_elem.get_text().strip() if snippet_elem else "No snippet available."
            results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")

        success_count += 1
        if success_count >= return_num:
            break


    if results:
        return "\n\n-----------------------\n\n".join(results)
    else:
        return "No results found on Bing."


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bing Search Tool")
    parser.add_argument('--chrome_driver', type=str, default="/usr/local/bin/chromedriver", help='Path to ChromeDriver')
    parser.add_argument('--visualize', action='store_true', help='Visualize the search results')
    args = parser.parse_args()
    result = BingSearchTool(args, "Genetic disease", mini_handler=None, read_content=False, return_num=5, screenshot=False)
    print(result)
