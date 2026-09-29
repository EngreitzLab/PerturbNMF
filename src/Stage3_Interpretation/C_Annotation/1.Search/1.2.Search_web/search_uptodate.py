from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup

import ipdb

ua = UserAgent()


# @tool
def UptodateSearchTool(args, query: str):
    """
    Use the Uptodate Website to search for Essamble results.

    Args:
        query (str): The search query.

    Returns:
        str: The search results or an error message.
    """
    try:
        url = f"https://www.uptodate.com/login"

        headers = {'User-Agent': ua.random}

        options = Options()
        if not args.visualize:
            options.add_argument("--headless")  # Run in headless mode (no GUI)
        # options.add_argument("--no-sandbox")  # This is helpful in certain environments (e.g., Docker)
        # options.add_argument("--disable-dev-shm-usage")  # Helps in environments with limited shared memory
        options.add_argument(f'user-agent={headers["User-Agent"]}')

        service = Service(args.chrome_driver)  # Update with the path to your chromedriver
        driver = webdriver.Chrome(service=service, options=options)

        driver.get(url)
        time.sleep(1)
        
        # ipdb.set_trace()
        # click //*[@id="onetrust-accept-btn-handler"]
        try:
            element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="onetrust-accept-btn-handler"]'))
            )
            element.click()
        except Exception as e:
            print(f"Error: {str(e)}")
        
        # input the username into //*[@id="username"]
        driver.find_element(By.XPATH, '//*[@id="userName"]').send_keys(args.uptodate_user)
        
        # click continue //*[@id="loginForm"]/fieldset/div[2]/span
        driver.find_element(By.XPATH, '//*[@id="loginForm"]/fieldset/div[2]/span').click()
        
        # input the password into //*[@id="password"]
        driver.find_element(By.XPATH, '//*[@id="password"]').send_keys(args.uptodate_pwd)
        
        # find the login button
        login_button = driver.find_element(By.ID, 'btnLoginSubmit')
        
        # maximum the window
        driver.maximize_window()

        # move to the login button and click
        actions = ActionChains(driver)
        actions.move_to_element(login_button).click().perform()

        # input the query into //*[@id="tbSearch"]
        time.sleep(1)
        driver.find_element(By.ID, 'tbSearch').send_keys(query)
        
        # click //*[@id="search-combobox"]/div[1]/div/form/span[3]
        time.sleep(1)
        driver.find_element(By.XPATH, '//*[@id="search-combobox"]/div[1]/div/form/span[3]').click()
        
        # wait for the search results to be fully rendered
        # WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "search-result")))
        
        # click first result
        try:
            element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'search-result'))
            )
            element.click()
        except Exception as e:
            print(f"Error: {str(e)}")
        
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.searchResultLink.search-result-primary-link[role='heading']"))
        )
        element.click()
        
        time.sleep(1)
        
        # get all paragraphs
        all_paragraph = []
        # WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "topic-title")))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        paragraphs = soup.find_all('p')
        for paragraph in paragraphs:
            all_paragraph.append(paragraph.text)
        
        all_paragraph = '\n'.join(all_paragraph)

        
        return all_paragraph
    except Exception as e:
        return str(e)
    finally:
        driver.quit()


if __name__ == '__main__':
    print(UptodateSearchTool('COVID-19'))
