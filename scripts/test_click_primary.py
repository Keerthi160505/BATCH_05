"""
Test clicking primary button and observing results
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

btn = driver.find_element(By.CSS_SELECTOR, "[data-testid='stBaseButton-primary']")
print("Found primary button:", btn.get_attribute("data-testid"))
driver.execute_script("arguments[0].click();", btn)
print("Clicked button via JS!")

# Wait up to 30 seconds for evaluation to finish
for i in range(15):
    time.sleep(2)
    # Check if tabs are present
    tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
    print(f"Elapsed { (i+1)*2 }s: Found {len(tabs)} tabs")
    if len(tabs) >= 3:
        print("Success! Tabs appeared!")
        driver.save_screenshot("documentation/figures/screenshots/test_results_live.png")
        print("Saved test_results_live.png!")
        break

driver.quit()
