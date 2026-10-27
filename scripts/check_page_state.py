"""
Check current state of page and take screenshot
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
time.sleep(3)

print("Page Title:", driver.title)
driver.save_screenshot("documentation/figures/screenshots/current_state.png")

# Check all buttons and tabs
tabs = driver.find_elements(By.CSS_SELECTOR, "[role='tab'], [data-baseweb='tab'], button")
print(f"Found {len(tabs)} button/tab elements:")
for t in tabs[:25]:
    txt = t.text.replace('\n', ' ')
    if txt:
        print(" - Tag:", t.tag_name, "Role:", t.get_attribute("role"), "Text:", repr(txt))

driver.quit()
