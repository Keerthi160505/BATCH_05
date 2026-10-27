"""
Capture genuine evaluation results screenshots from the live Streamlit dashboard.
Captures:
- dashboard_model_results.png (Tab 1: Overview & Leaderboard)
- dashboard_sustainability_results.png (Tab 2: Sustainability Telemetry)
- dashboard_pareto_frontier.png (Tab 3: Pareto Frontier Analysis)
- dashboard_detailed_results.png (Tab 4: Detailed Results & Export)
"""
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1600,1100")
opts.add_argument("--disable-gpu")
opts.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=opts)
out_dir = os.path.abspath("documentation/figures/screenshots")
os.makedirs(out_dir, exist_ok=True)

try:
    print("Navigating to http://localhost:8501 ...", flush=True)
    driver.get("http://localhost:8501")
    wait = WebDriverWait(driver, 30)

    # Wait for selectbox
    print("Waiting for dataset selectbox...", flush=True)
    label_ds = wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Select Benchmark Dataset')]")))
    parent_sbox = label_ds.find_element(By.XPATH, "./ancestor::div[contains(@data-testid, 'stSelectbox') or contains(@class, 'stSelectbox')]")
    clickable = parent_sbox.find_element(By.XPATH, ".//div[@role='combobox'] | .//input | .//div[@data-baseweb='select']")
    clickable.click()
    time.sleep(1.0)

    # Click Breast Cancer Diagnostic option
    opt_cancer = wait.until(EC.element_to_be_clickable((By.XPATH, "//li[contains(., 'Breast Cancer')] | //div[@role='option' and contains(., 'Breast Cancer')]")))
    opt_cancer.click()
    print("Selected Breast Cancer Diagnostic benchmark!", flush=True)
    time.sleep(2.0)

    # Wait for sidebar hydration
    wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "section[data-testid='stSidebar'] button")) > 15)
    sidebar_buttons = driver.find_elements(By.CSS_SELECTOR, "section[data-testid='stSidebar'] button")
    
    run_btn = None
    for b in sidebar_buttons:
        if b.get_attribute("data-testid") == "stBaseButton-primary" or "Run" in b.text:
            run_btn = b
            break
            
    if not run_btn:
        raise RuntimeError("Primary run button not found!")

    print("Found 'Run Green AI Evaluation' button. Clicking...", flush=True)
    driver.execute_script("arguments[0].click();", run_btn)

    print("Benchmarking all 6 ML models (Logistic Regression, Decision Tree, LightGBM, Random Forest, XGBoost, CatBoost)...", flush=True)
    print("Waiting for evaluation completion and KPI cards to render...", flush=True)
    
    # Wait up to 100s for completion
    kpi_found = False
    for sec in range(1, 51):
        time.sleep(2)
        kpis = driver.find_elements(By.XPATH, "//*[contains(text(), 'Green AI Champion')]")
        if len(kpis) > 0:
            print(f"Evaluation finished successfully at {sec*2} seconds!", flush=True)
            kpi_found = True
            break
        if sec % 5 == 0:
            print(f"Elapsed {sec*2}s: still executing models...", flush=True)

    if not kpi_found:
        raise TimeoutError("Evaluation did not complete within 100 seconds!")

    # Let Plotly charts render completely
    time.sleep(3.0)

    # 1. Capture Model Results (Tab 1)
    results_path = os.path.join(out_dir, "dashboard_model_results.png")
    driver.save_screenshot(results_path)
    print(f"Captured: {results_path}", flush=True)

    # Locate the tab buttons
    tab_buttons = driver.find_elements(By.XPATH, "//button[@role='tab']")
    print(f"Found {len(tab_buttons)} tab buttons:", [t.text for t in tab_buttons], flush=True)

    # 2. Click Tab 2: Sustainability Telemetry
    print("Switching to Sustainability Telemetry tab...", flush=True)
    tab_sustain = None
    for t in tab_buttons:
        if "Sustainability" in t.text:
            tab_sustain = t
            break
    if not tab_sustain:
        tab_sustain = driver.find_element(By.XPATH, "//*[contains(text(), 'Sustainability Telemetry') or contains(text(), 'Sustainability')]")
    driver.execute_script("arguments[0].click();", tab_sustain)
    time.sleep(3.0)
    sustain_path = os.path.join(out_dir, "dashboard_sustainability_results.png")
    driver.save_screenshot(sustain_path)
    print(f"Captured: {sustain_path}", flush=True)

    # 3. Click Tab 3: Pareto Frontier Analysis
    print("Switching to Pareto Frontier tab...", flush=True)
    tab_buttons = driver.find_elements(By.XPATH, "//button[@role='tab']")
    tab_pareto = None
    for t in tab_buttons:
        if "Pareto" in t.text:
            tab_pareto = t
            break
    if not tab_pareto:
        tab_pareto = driver.find_element(By.XPATH, "//*[contains(text(), 'Pareto Frontier') or contains(text(), 'Pareto')]")
    driver.execute_script("arguments[0].click();", tab_pareto)
    time.sleep(3.0)
    pareto_path = os.path.join(out_dir, "dashboard_pareto_frontier.png")
    driver.save_screenshot(pareto_path)
    print(f"Captured: {pareto_path}", flush=True)

    # 4. Click Tab 4: Detailed Results & Export
    print("Switching to Detailed Results & Export tab...", flush=True)
    tab_buttons = driver.find_elements(By.XPATH, "//button[@role='tab']")
    tab_details = None
    for t in tab_buttons:
        if "Detailed" in t.text or "Export" in t.text:
            tab_details = t
            break
    if not tab_details:
        tab_details = driver.find_element(By.XPATH, "//*[contains(text(), 'Detailed Results')]")
    driver.execute_script("arguments[0].click();", tab_details)
    time.sleep(2.0)
    
    # Try to expand IEEE LaTeX table if expander exists
    expanders = driver.find_elements(By.XPATH, "//div[@data-testid='stExpander']//summary | //summary[contains(., 'LaTeX')]")
    for exp in expanders:
        if "LaTeX" in exp.text or "IEEE" in exp.text:
            try:
                driver.execute_script("arguments[0].click();", exp)
                time.sleep(1.0)
            except Exception:
                pass
                
    detailed_path = os.path.join(out_dir, "dashboard_detailed_results.png")
    driver.save_screenshot(detailed_path)
    print(f"Captured: {detailed_path}", flush=True)

    print("ALL 4 POST-EVALUATION DASHBOARD SCREENSHOTS CAPTURED SUCCESSFULLY!", flush=True)

finally:
    driver.quit()
