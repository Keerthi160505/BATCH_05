"""
Test clicking run button and waiting 40 seconds for completion in same session
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
run_btn = None
for b in buttons:
    if b.get_attribute("data-testid") == "stBaseButton-primary":
        run_btn = b
        break

print("Found run button! Clicking...")
driver.execute_script("arguments[0].click();", run_btn)
print("Clicked! Now waiting 40 seconds for model training and profiling to complete...")

for sec in range(1, 21):
    time.sleep(2)
    tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
    print(f"Elapsed {sec*2}s: {len(tabs)} tabs found")
    if len(tabs) >= 3:
        print(f"EVALUATION COMPLETE AT {sec*2}s! TABS RENDERED!")
        driver.save_screenshot("documentation/figures/screenshots/results_success_verified.png")
        break

driver.quit()
