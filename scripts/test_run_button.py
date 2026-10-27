"""
Click Run button and wait for completion
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

sidebar = driver.find_element(By.XPATH, "//section[@data-testid='stSidebar']")
buttons = sidebar.find_elements(By.TAG_NAME, "button")
run_btn = buttons[-2]
print(f"Target button: text={repr(run_btn.text[:20])}, testid={run_btn.get_attribute('data-testid')}")

driver.execute_script("arguments[0].click();", run_btn)
print("Clicked Run button! Waiting for evaluation...")

for i in range(25):
    time.sleep(2)
    # Check for tabs
    tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
    print(f"Elapsed {(i+1)*2}s: {len(tabs)} tabs found")
    if len(tabs) >= 3:
        print("SUCCESS! Results rendered!")
        driver.save_screenshot("documentation/figures/screenshots/test_results_success.png")
        break

driver.quit()
