"""
Test interacting with Streamlit selectbox and buttons
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

sbox = driver.find_element(By.CSS_SELECTOR, "[data-testid='stSelectbox']")
print("Found stSelectbox:", sbox.get_attribute("outerHTML")[:200])

# Find clickable child
target = sbox.find_element(By.CSS_SELECTOR, "div[role='combobox'], div[data-baseweb='select'], input")
print("Target tag:", target.tag_name, "role:", target.get_attribute("role"))
target.click()
time.sleep(1)

# Find options
opts = driver.find_elements(By.CSS_SELECTOR, "li[role='option'], div[role='option'], [data-testid='stSelectboxVirtualDropdown'] li")
print(f"Found {len(opts)} options:")
for o in opts:
    print(" -", repr(o.text))

driver.quit()
