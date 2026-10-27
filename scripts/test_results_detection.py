"""
Test detecting results via kpi-card and button[role='tab']
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
    if b.get_attribute("data-testid") == "stBaseButton-primary" or "Run" in b.text:
        run_btn = b
        break

print("Clicking Run button...")
driver.execute_script("arguments[0].click();", run_btn)

for i in range(35): # 70 seconds max
    time.sleep(2)
    kpis = driver.find_elements(By.XPATH, "//*[contains(text(), 'Green AI Champion')]")
    tabs = driver.find_elements(By.XPATH, "//button[@role='tab']")
    print(f"Elapsed {(i+1)*2}s: KPIs={len(kpis)}, Tabs={len(tabs)}")
    if len(kpis) > 0 or len(tabs) > 0:
        print("FOUND RESULTS! Details:")
        print(f"  KPI elements: {len(kpis)}")
        print(f"  Tab elements: {len(tabs)}")
        for t in tabs:
            print("   Tab:", repr(t.text))
        driver.save_screenshot("documentation/figures/screenshots/detected_results.png")
        print("Saved detected_results.png!")
        break

driver.quit()
