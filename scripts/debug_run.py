"""
Diagnostic script to check what is displayed after clicking Run
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

# Find run button
btn = driver.find_element(By.XPATH, "//button[contains(., 'Run Green AI Evaluation')]")
print("Found button:", btn.text)

# Click using JS to be 100% sure event fires
driver.execute_script("arguments[0].scrollIntoView(true);", btn)
time.sleep(1)
driver.execute_script("arguments[0].click();", btn)
print("Clicked button via JS!")

for i in range(1, 8):
    time.sleep(2)
    print(f"Elapsed {i*2}s...")
    driver.save_screenshot(f"documentation/figures/screenshots/run_debug_{i*2}s.png")
    
    # Check for errors or results
    errors = driver.find_elements(By.CSS_SELECTOR, "[data-testid='stAlert'], [data-testid='stNotification']")
    for err in errors:
        print("Alert text:", err.text)
        
    kpi = driver.find_elements(By.XPATH, "//*[contains(text(), 'Green AI Champion')]")
    if kpi:
        print("Found Green AI Champion! Evaluation succeeded!")
        break

driver.quit()
