"""
Automated Dashboard Screenshot Capture Script using Selenium & Headless Chrome
Captures 7 genuine, publication-quality screenshots of the running Streamlit app:
1. dashboard_home.png (Initial state: Hero card, 3 pillars, status badge 'Framework Ready', educational panel)
2. dashboard_configuration.png (Dataset configuration, benchmark selector, custom CSV upload dropzone, scaler)
3. dashboard_model_controls.png (Sampling controls, 6 model checkboxes, alpha slider, sub-weights, run button)
4. dashboard_model_results.png (Post-evaluation: Header KPI cards, overview leaderboard, GAI ranking chart)
5. dashboard_sustainability_results.png (Tab 2: Energy breakdown, carbon emissions, CPU vs RAM, latency)
6. dashboard_pareto_frontier.png (Tab 3: Interactive Pareto scatter plot with frontier curve and star markers)
7. dashboard_detailed_results.png (Tab 4: Full telemetry table, IEEE LaTeX export button, CSV download button)
"""

import os
import sys
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "documentation", "figures", "screenshots"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_chrome_driver():
    options = ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1050")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1600, 1050)
    return driver

def wait_for_streamlit_loaded(driver, timeout=20):
    """Wait until Streamlit main container is present and not loading."""
    wait = WebDriverWait(driver, timeout)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stApp']")))
    time.sleep(2.5) # Allow dynamic React and CSS render to settle

def capture_screenshot(driver, filename, crop_sidebar=False):
    """Saves a high-resolution screenshot to output directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    driver.save_screenshot(path)
    print(f"Captured: {path} ({os.path.getsize(path):,} bytes)")
    return path

def run_screenshot_pipeline():
    driver = setup_chrome_driver()
    try:
        print("Navigating to live Streamlit dashboard (http://localhost:8501)...")
        driver.get("http://localhost:8501")
        wait_for_streamlit_loaded(driver)
        
        # -------------------------------------------------------------
        # 1. SCREENSHOT 1: DASHBOARD HOME (INITIAL STATE)
        # -------------------------------------------------------------
        print("1. Capturing Dashboard Home Screen...")
        time.sleep(2)
        capture_screenshot(driver, "dashboard_home.png")

        # -------------------------------------------------------------
        # 2. SCREENSHOT 2: DATASET CONFIGURATION & UPLOAD CUSTOM CSV
        # -------------------------------------------------------------
        print("2. Configuring dataset upload...")
        # Open dataset selectbox in sidebar and choose 'Upload Custom CSV'
        sidebar = driver.find_element(By.CSS_SELECTOR, "[data-testid='stSidebar']")
        selectbox = sidebar.find_element(By.CSS_SELECTOR, "[data-baseweb='select']")
        selectbox.click()
        time.sleep(1)
        
        # Click 'Upload Custom CSV' option
        options = driver.find_elements(By.CSS_SELECTOR, "[role='option']")
        for opt in options:
            if "Upload Custom CSV" in opt.text:
                opt.click()
                break
        time.sleep(2)
        
        # Focus on sidebar configuration
        driver.set_window_size(1500, 950)
        capture_screenshot(driver, "dashboard_configuration.png")

        # Re-select 'Adult Census Income (UCI)' or 'Breast Cancer Diagnostic' for evaluation
        print("Re-selecting canonical benchmark dataset...")
        selectbox = sidebar.find_element(By.CSS_SELECTOR, "[data-baseweb='select']")
        selectbox.click()
        time.sleep(1)
        options = driver.find_elements(By.CSS_SELECTOR, "[role='option']")
        for opt in options:
            if "Adult Census Income" in opt.text:
                opt.click()
                break
        time.sleep(2)

        # -------------------------------------------------------------
        # 3. SCREENSHOT 3: MODEL SELECTION & SAMPLING CONTROLS
        # -------------------------------------------------------------
        print("3. Capturing Model Selection & Sampling Controls...")
        # Scroll sidebar slightly down to show model checkboxes and alpha slider
        driver.execute_script("arguments[0].scrollTop = 250;", sidebar)
        time.sleep(1.5)
        capture_screenshot(driver, "dashboard_model_selection.png")

        # -------------------------------------------------------------
        # 4. RUN REAL BENCHMARK EVALUATION
        # -------------------------------------------------------------
        print("Executing benchmark evaluation on canonical Adult Census...")
        # Scroll to run button in sidebar
        driver.execute_script("arguments[0].scrollTop = 700;", sidebar)
        time.sleep(1)
        
        # Find Run button
        run_btn = None
        buttons = sidebar.find_elements(By.TAG_NAME, "button")
        for b in buttons:
            if "Run Green AI Evaluation" in b.text:
                run_btn = b
                break
        
        if run_btn:
            print("Clicking '🚀 Run Green AI Evaluation'...")
            run_btn.click()
            
            # Wait for evaluation spinner to finish (takes ~4-8 seconds)
            print("Waiting for evaluation and profiler execution...")
            WebDriverWait(driver, 60).until(
                lambda d: len(d.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")) >= 4
            )
            time.sleep(3.5) # Allow Plotly graphs and KPI cards to complete animation
            print("Evaluation completed successfully!")

        # -------------------------------------------------------------
        # 4. SCREENSHOT 4: MODEL PERFORMANCE & LEADERBOARD RESULTS
        # -------------------------------------------------------------
        print("4. Capturing Model Performance Results (Tab 1: Overview & Leaderboard)...")
        # Scroll main window to show header KPIs and Leaderboard
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        capture_screenshot(driver, "dashboard_model_results.png")

        # -------------------------------------------------------------
        # 5. SCREENSHOT 5: COMPUTATIONAL SUSTAINABILITY RESULTS (TAB 2)
        # -------------------------------------------------------------
        print("5. Capturing Computational Sustainability Results (Tab 2)...")
        tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
        for t in tabs:
            if "Sustainability Telemetry" in t.text or "Sustainability" in t.text:
                t.click()
                break
        time.sleep(2.5) # Allow Plotly graphs to render
        capture_screenshot(driver, "dashboard_sustainability_results.png")

        # -------------------------------------------------------------
        # 6. SCREENSHOT 6: PARETO FRONTIER ANALYSIS (TAB 3)
        # -------------------------------------------------------------
        print("6. Capturing Pareto Frontier Analysis (Tab 3)...")
        tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
        for t in tabs:
            if "Pareto Frontier" in t.text or "Pareto" in t.text:
                t.click()
                break
        time.sleep(2.5)
        capture_screenshot(driver, "dashboard_pareto_frontier.png")

        # -------------------------------------------------------------
        # 7. SCREENSHOT 7: DETAILED RESULTS TABLE & EXPORT CONTROLS (TAB 4)
        # -------------------------------------------------------------
        print("7. Capturing Detailed Results & Export Controls (Tab 4)...")
        tabs = driver.find_elements(By.CSS_SELECTOR, "[data-baseweb='tab']")
        for t in tabs:
            if "Detailed Results" in t.text or "Export" in t.text:
                t.click()
                break
        time.sleep(2)
        
        # Click on the IEEE LaTeX expander to open it
        expanders = driver.find_elements(By.CSS_SELECTOR, "[data-testid='stExpander']")
        for exp in expanders:
            if "IEEE LaTeX" in exp.text or "LaTeX" in exp.text:
                exp.click()
                break
        time.sleep(1.5)
        capture_screenshot(driver, "dashboard_detailed_results.png")

        print("All 7 genuine dashboard screenshots captured successfully!")

    finally:
        driver.quit()

if __name__ == "__main__":
    run_screenshot_pipeline()
