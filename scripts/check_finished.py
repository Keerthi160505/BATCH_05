"""
Take screenshot of localhost:8501 right now
"""
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1600,1050")
opts.add_argument("--disable-gpu")
opts.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=opts)
driver.get("http://localhost:8501")
time.sleep(5)

driver.save_screenshot("documentation/figures/screenshots/check_finished.png")
print("Saved check_finished.png!")
driver.quit()
