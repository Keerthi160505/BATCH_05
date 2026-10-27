"""
Test XPath search in Streamlit
"""
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = Options()
options.add_argument("--headless=new")
options.add_argument("--window-size=1600,1050")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=options)
driver.get("http://localhost:8501")

wait = WebDriverWait(driver, 15)
label_el = wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Select Benchmark Dataset')]")))
print("Found label:", label_el.text)

# Find parent selectbox container
parent = label_el.find_element(By.XPATH, "./ancestor::div[contains(@data-testid, 'stSelectbox') or contains(@class, 'stSelectbox')]")
print("Found parent container:", parent.tag_name)

# Find clickable element inside parent
clickable = parent.find_element(By.XPATH, ".//div[@role='combobox'] | .//input | .//div[@data-baseweb='select']")
print("Found clickable:", clickable.tag_name, "class:", clickable.get_attribute("class"))

clickable.click()
time.sleep(1)

# List open options
options_els = driver.find_elements(By.XPATH, "//li[@role='option'] | //div[@role='option']")
print(f"Found {len(options_els)} options:")
for opt in options_els:
    print(" -", opt.text)

driver.quit()
