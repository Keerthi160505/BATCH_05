"""
Master Screenshot Capture Script for Dynamic Green AI Framework
Automates Chrome Headless to capture 7 publication-ready screenshots from live Streamlit app:
1. dashboard_home.png
2. dashboard_configuration.png
3. dashboard_model_selection.png
4. dashboard_model_results.png
5. dashboard_sustainability_results.png
6. dashboard_pareto_frontier.png
7. dashboard_detailed_results.png
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "documentation", "figures", "screenshots"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1600,1050")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=opts)
    driver.set_window_size(1600, 1050)
    return driver

def capture(driver, name):
    path = os.path.join(OUTPUT_DIR, name)
    driver.save_screenshot(path)
    sz = os.path.getsize(path)
    print(f"Captured {name}: {sz:,} bytes")
    return path

def run():
    driver = get_driver()
    try:
        wait = WebDriverWait(driver, 20)
        print("Navigating to http://localhost:8501...")
        driver.get("http://localhost:8501")
        
        # Wait for Streamlit to initialize
        wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Select Benchmark Dataset')]")))
        time.sleep(3.5) # allow styling and font loads to settle
        
        # 1. Capture Dashboard Home
        print("1. Capturing dashboard_home.png...")
        capture(driver, "dashboard_home.png")
        
        # 2. Select 'Upload Custom CSV' to capture dataset configuration & upload dropzone
        print("2. Capturing dashboard_configuration.png...")
        label_ds = driver.find_element(By.XPATH, "//*[contains(text(), 'Select Benchmark Dataset')]")
        parent_sbox = label_ds.find_element(By.XPATH, "./ancestor::div[contains(@data-testid, 'stSelectbox') or contains(@class, 'stSelectbox')]")
        clickable_sbox = parent_sbox.find_element(By.XPATH, ".//div[@role='combobox'] | .//input | .//div[@data-baseweb='select']")
        clickable_sbox.click()
        time.sleep(1.0)
        
        # Click 'Upload Custom CSV' option
        opt_upload = wait.until(EC.element_to_be_clickable((By.XPATH, "//li[contains(., 'Upload Custom CSV')] | //div[@role='option' and contains(., 'Upload Custom CSV')]")))
        opt_upload.click()
        time.sleep(2.5) # allow file uploader dropzone to render
        capture(driver, "dashboard_configuration.png")
        
        # 3. Select 'Adult Census Income (UCI)' for model selection & benchmarking
        print("3. Switching back to canonical Adult Census benchmark...")
        label_ds = driver.find_element(By.XPATH, "//*[contains(text(), 'Select Benchmark Dataset')]")
        parent_sbox = label_ds.find_element(By.XPATH, "./ancestor::div[contains(@data-testid, 'stSelectbox') or contains(@class, 'stSelectbox')]")
        clickable_sbox = parent_sbox.find_element(By.XPATH, ".//div[@role='combobox'] | .//input | .//div[@data-baseweb='select']")
        clickable_sbox.click()
        time.sleep(1.0)
        
        opt_adult = wait.until(EC.element_to_be_clickable((By.XPATH, "//li[contains(., 'Adult Census Income')] | //div[@role='option' and contains(., 'Adult Census Income')]")))
        opt_adult.click()
        time.sleep(2.0)
        
        # Capture Model Selection & Sampling Controls in sidebar
        print("4. Capturing dashboard_model_selection.png...")
        sidebar = driver.find_element(By.XPATH, "//section[@data-testid='stSidebar']")
        driver.execute_script("arguments[0].scrollTop = 320;", sidebar)
        time.sleep(1.5)
        capture(driver, "dashboard_model_selection.png")
        
        # 4. Trigger Real Evaluation
        print("5. Clicking 'Run Green AI Evaluation' button...")
        driver.execute_script("arguments[0].scrollTop = 800;", sidebar)
        time.sleep(1.0)
        
        run_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Run Green AI Evaluation')]")))
        run_btn.click()
        print("Evaluation initiated! Waiting for profiler and models to finish...")
        
        # Wait for results tabs to appear
        wait_eval = WebDriverWait(driver, 60)
        tab_overview = wait_eval.until(EC.presence_of_element_located((By.XPATH, "//button[@role='tab' and contains(., 'Overview')] | //div[@data-baseweb='tab' and contains(., 'Overview')]")))
        time.sleep(4.0) # allow Plotly graphs and KPI animations to settle
        print("Evaluation finished successfully!")
        
        # Scroll main window to top
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1.5)
        
        # 5. Capture Model Performance & Leaderboard Results (Tab 1)
        print("6. Capturing dashboard_model_results.png (Tab 1)...")
        capture(driver, "dashboard_model_results.png")
        
        # 6. Capture Sustainability Telemetry (Tab 2)
        print("7. Capturing dashboard_sustainability_results.png (Tab 2)...")
        tab_sust = driver.find_element(By.XPATH, "//button[@role='tab' and contains(., 'Sustainability')] | //div[@data-baseweb='tab' and contains(., 'Sustainability')]")
        tab_sust.click()
        time.sleep(3.0) # allow Plotly charts to render
        capture(driver, "dashboard_sustainability_results.png")
        
        # 7. Capture Pareto Frontier Analysis (Tab 3)
        print("8. Capturing dashboard_pareto_frontier.png (Tab 3)...")
        tab_pareto = driver.find_element(By.XPATH, "//button[@role='tab' and contains(., 'Pareto')] | //div[@data-baseweb='tab' and contains(., 'Pareto')]")
        tab_pareto.click()
        time.sleep(3.0) # allow Pareto curve & scatter to render
        capture(driver, "dashboard_pareto_frontier.png")
        
        # 8. Capture Detailed Results & Export Controls (Tab 4)
        print("9. Capturing dashboard_detailed_results.png (Tab 4)...")
        tab_export = driver.find_element(By.XPATH, "//button[@role='tab' and contains(., 'Detailed Results')] | //div[@data-baseweb='tab' and contains(., 'Detailed Results')] | //button[@role='tab' and contains(., 'Export')]")
        tab_export.click()
        time.sleep(2.0)
        
        # Open the IEEE LaTeX Expander
        try:
            expander = driver.find_element(By.XPATH, "//*[contains(text(), 'View Formatted IEEE LaTeX Code')]")
            expander.click()
            time.sleep(1.5)
        except Exception as e_exp:
            print("Expander notice:", e_exp)
            
        capture(driver, "dashboard_detailed_results.png")
        print("ALL 7 DASHBOARD SCREENSHOTS CAPTURED SUCCESSFULLY!")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    run()
