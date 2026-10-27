"""
Test Selenium screenshot capture of Streamlit app
"""
import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions

def test_driver():
    # Try Chrome first
    try:
        print("Testing Chrome headless driver...")
        chrome_opts = ChromeOptions()
        chrome_opts.add_argument("--headless=new")
        chrome_opts.add_argument("--window-size=1920,1080")
        chrome_opts.add_argument("--disable-gpu")
        chrome_opts.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=chrome_opts)
        driver.get("http://localhost:8501")
        time.sleep(3)
        print("Successfully opened http://localhost:8501 with Chrome!")
        print("Page Title:", driver.title)
        driver.save_screenshot("documentation/figures/screenshots/test_chrome.png")
        print("Saved test_chrome.png")
        driver.quit()
        return "chrome"
    except Exception as e_chrome:
        print(f"Chrome failed: {e_chrome}")
        # Try Edge
        try:
            print("Testing Edge headless driver...")
            edge_opts = EdgeOptions()
            edge_opts.add_argument("--headless=new")
            edge_opts.add_argument("--window-size=1920,1080")
            edge_opts.add_argument("--disable-gpu")
            driver = webdriver.Edge(options=edge_opts)
            driver.get("http://localhost:8501")
            time.sleep(3)
            print("Successfully opened http://localhost:8501 with Edge!")
            print("Page Title:", driver.title)
            driver.save_screenshot("documentation/figures/screenshots/test_edge.png")
            print("Saved test_edge.png")
            driver.quit()
            return "edge"
        except Exception as e_edge:
            print(f"Edge failed: {e_edge}")
            return None

if __name__ == "__main__":
    res = test_driver()
    print("Result:", res)
