"""
Wait for sidebar button hydration and click
"""
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1600,1050")
opts.add_argument("--disable-gpu")
opts.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=opts)
driver.get("http://localhost:8501")

wait = WebDriverWait(driver, 20)
# Wait until sidebar buttons are hydrated
wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "section[data-testid='stSidebar'] button")) > 15)

sidebar_buttons = driver.find_elements(By.CSS_SELECTOR, "section[data-testid='stSidebar'] button")
print(f"Hydrated! Found {len(sidebar_buttons)} sidebar buttons.")

# Find the primary button
run_btn = None
for b in sidebar_buttons:
    if b.get_attribute("data-testid") == "stBaseButton-primary" or "Run" in b.text:
        run_btn = b
        break

if run_btn:
    print("Found run button! Triggering evaluation click...")
    driver.execute_script("arguments[0].click();", run_btn)
    
    # Wait for results tabs to appear
    print("Waiting for results tabs...")
    wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")) >= 4)
    time.sleep(4)
    print("Evaluation completed! Tabs rendered!")
    driver.save_screenshot("documentation/figures/screenshots/test_results_hydrated.png")
    print("Saved test_results_hydrated.png successfully!")
else:
    print("Could not find primary run button.")

driver.quit()
