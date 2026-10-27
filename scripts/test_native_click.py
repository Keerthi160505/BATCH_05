"""
Test native click on run button with 60s timeout
"""
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1600,1050")
opts.add_argument("--disable-gpu")
opts.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=opts)
driver.get("http://localhost:8501")

wait = WebDriverWait(driver, 20)
wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "section[data-testid='stSidebar'] button")) > 15)

sidebar_buttons = driver.find_elements(By.CSS_SELECTOR, "section[data-testid='stSidebar'] button")
run_btn = None
for b in sidebar_buttons:
    if b.get_attribute("data-testid") == "stBaseButton-primary":
        run_btn = b
        break

print("Found button, scrolling to view...")
driver.execute_script("arguments[0].scrollIntoView({behavior: 'instant', block: 'center'});", run_btn)
time.sleep(1)

print("Clicking natively with ActionChains...")
ActionChains(driver).move_to_element(run_btn).click().perform()

print("Waiting up to 60s for tabs...")
for i in range(30):
    time.sleep(2)
    tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
    print(f"Elapsed {(i+1)*2}s: {len(tabs)} tabs found")
    if len(tabs) >= 3:
        print("SUCCESS! Results rendered!")
        driver.save_screenshot("documentation/figures/screenshots/test_native_success.png")
        break

driver.quit()
