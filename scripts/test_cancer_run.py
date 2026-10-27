"""
Test evaluating Breast Cancer Diagnostic on live Streamlit dashboard
"""
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1600,1050")
opts.add_argument("--disable-gpu")
opts.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=opts)
driver.get("http://localhost:8501")

wait = WebDriverWait(driver, 25)
print("Waiting for dataset selectbox...")
label_ds = wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Select Benchmark Dataset')]")))

# Open selectbox
parent_sbox = label_ds.find_element(By.XPATH, "./ancestor::div[contains(@data-testid, 'stSelectbox') or contains(@class, 'stSelectbox')]")
clickable = parent_sbox.find_element(By.XPATH, ".//div[@role='combobox'] | .//input | .//div[@data-baseweb='select']")
clickable.click()
time.sleep(1.0)

# Click Breast Cancer
opt_cancer = wait.until(EC.element_to_be_clickable((By.XPATH, "//li[contains(., 'Breast Cancer')] | //div[@role='option' and contains(., 'Breast Cancer')]")))
opt_cancer.click()
print("Selected Breast Cancer Diagnostic!")
time.sleep(2.0)

# Wait for sidebar buttons
wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "section[data-testid='stSidebar'] button")) > 15)
sidebar_buttons = driver.find_elements(By.CSS_SELECTOR, "section[data-testid='stSidebar'] button")

run_btn = None
for b in sidebar_buttons:
    if b.get_attribute("data-testid") == "stBaseButton-primary" or "Run" in b.text:
        run_btn = b
        break

print("Clicking Run button...")
driver.execute_script("arguments[0].click();", run_btn)

print("Waiting for results...")
for i in range(15): # 30s max
    time.sleep(2)
    kpis = driver.find_elements(By.XPATH, "//*[contains(text(), 'Green AI Champion')]")
    if len(kpis) > 0:
        print(f"SUCCESS AT {(i+1)*2}s! Found Green AI Champion!")
        driver.save_screenshot("documentation/figures/screenshots/cancer_results_success.png")
        print("Saved cancer_results_success.png!")
        break

driver.quit()
