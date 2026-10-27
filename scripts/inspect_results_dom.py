"""
Inspect elements on the results page
"""
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1600,1050")
opts.add_argument("--disable-gpu")
opts.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=opts)
driver.get("http://localhost:8501")
time.sleep(4)

# Search for any element containing 'Leaderboard' or 'Champion' or 'Sustainability'
keywords = ['Champion', 'Leaderboard', 'Sustainability', 'Pareto', 'Export']
for kw in keywords:
    els = driver.find_elements(By.XPATH, f"//*[contains(text(), '{kw}')]")
    print(f"Keyword '{kw}': {len(els)} elements found")
    for e in els[:3]:
        print(f"  Tag: {e.tag_name}, class: {repr(e.get_attribute('class'))[:50]}, data-testid: {repr(e.get_attribute('data-testid'))}")

driver.save_screenshot("documentation/figures/screenshots/inspect_results.png")
print("Saved inspect_results.png!")
driver.quit()
