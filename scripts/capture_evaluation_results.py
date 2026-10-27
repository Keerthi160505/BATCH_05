"""
Capture Genuine Post-Evaluation Results Screenshots
Triggers real benchmark evaluation on running Streamlit app,
waits up to 120 seconds for model training and profiler execution,
and captures the 4 evaluation result screens:
- dashboard_model_results.png
- dashboard_sustainability_results.png
- dashboard_pareto_frontier.png
- dashboard_detailed_results.png
"""

import os
import sys
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "documentation", "figures", "screenshots"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1600,1050")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=opts)
    driver.set_window_size(1600, 1050)
    return driver

def capture(driver, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    driver.save_screenshot(path)
    sz = os.path.getsize(path)
    print(f"Successfully captured: {filename} ({sz:,} bytes)", flush=True)
    return path

def run_evaluation_capture():
    driver = setup_driver()
    try:
        wait = WebDriverWait(driver, 120)
        print("Navigating to http://localhost:8501...", flush=True)
        driver.get("http://localhost:8501")
        
        # Explicitly wait for sidebar buttons hydration
        print("Waiting for sidebar to hydrate...", flush=True)
        wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "section[data-testid='stSidebar'] button")) > 15)
        
        sidebar = driver.find_element(By.XPATH, "//section[@data-testid='stSidebar']")
        buttons = sidebar.find_elements(By.TAG_NAME, "button")
        run_btn = None
        for b in buttons:
            if b.get_attribute("data-testid") == "stBaseButton-primary" or "Run" in b.text:
                run_btn = b
                break
                
        if not run_btn:
            raise RuntimeError("Could not locate primary evaluation button in sidebar.")
            
        print("Found Run button! Triggering real evaluation on Adult Census...", flush=True)
        driver.execute_script("arguments[0].click();", run_btn)
        print("Evaluation initiated! Waiting up to 120s for all 6 models to complete...", flush=True)
        
        # Poll for tabs to render
        tabs = []
        for i in range(60): # 60 * 2 = 120 seconds max
            time.sleep(2)
            tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
            print(f"Elapsed {(i+1)*2}s: {len(tabs)} tabs found", flush=True)
            if len(tabs) >= 4:
                print("EVALUATION COMPLETED! Tabs are visible!", flush=True)
                break
                
        if len(tabs) < 4:
            raise TimeoutError("Evaluation did not finish within 120 seconds.")
            
        # Allow Plotly charts and KPI cards to fully render
        time.sleep(4.5)
        
        # 1. SCREENSHOT 4: MODEL PERFORMANCE & LEADERBOARD (TAB 1)
        print("Capturing dashboard_model_results.png (Tab 1)...", flush=True)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1.5)
        capture(driver, "dashboard_model_results.png")
        
        # 2. SCREENSHOT 5: COMPUTATIONAL SUSTAINABILITY RESULTS (TAB 2)
        print("Capturing dashboard_sustainability_results.png (Tab 2)...", flush=True)
        tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
        for t in tabs:
            if "Sustainability" in t.text:
                t.click()
                break
        time.sleep(3.5) # allow Plotly stacked bar charts to render
        capture(driver, "dashboard_sustainability_results.png")
        
        # 3. SCREENSHOT 6: PARETO FRONTIER ANALYSIS (TAB 3)
        print("Capturing dashboard_pareto_frontier.png (Tab 3)...", flush=True)
        tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
        for t in tabs:
            if "Pareto" in t.text:
                t.click()
                break
        time.sleep(3.5) # allow Plotly scatter and frontier to render
        capture(driver, "dashboard_pareto_frontier.png")
        
        # 4. SCREENSHOT 7: DETAILED RESULTS & EXPORT CONTROLS (TAB 4)
        print("Capturing dashboard_detailed_results.png (Tab 4)...", flush=True)
        tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
        for t in tabs:
            if "Detailed Results" in t.text or "Export" in t.text:
                t.click()
                break
        time.sleep(2.0)
        
        # Expand LaTeX expander
        try:
            expander = driver.find_element(By.XPATH, "//*[contains(text(), 'View Formatted IEEE LaTeX Code')]")
            expander.click()
            time.sleep(1.5)
        except Exception as e:
            print(f"Expander click notice: {e}", flush=True)
            
        capture(driver, "dashboard_detailed_results.png")
        print("ALL POST-EVALUATION SCREENS CAPTURED SUCCESSFULLY!", flush=True)

    finally:
        driver.quit()

if __name__ == "__main__":
    run_evaluation_capture()
