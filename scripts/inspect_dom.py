"""
Dump Streamlit DOM structure
"""
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless=new")
options.add_argument("--window-size=1600,1050")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=options)
driver.get("http://localhost:8501")
time.sleep(4)

# Print all elements with data-testid
elements = driver.find_elements(By.CSS_SELECTOR, "[data-testid]")
print(f"Total elements with data-testid: {len(elements)}")
for el in elements[:40]:
    tid = el.get_attribute("data-testid")
    txt = el.text[:40].replace('\n', ' ') if el.text else ""
    print(f"Tag: {el.tag_name}, data-testid: {tid}, text: {txt}")

driver.quit()
