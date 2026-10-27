"""
Find all buttons in sidebar
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
print(f"Total buttons in sidebar: {len(buttons)}")
for i, b in enumerate(buttons):
    print(f"Button {i}: text={repr(b.text)}, data-testid={repr(b.get_attribute('data-testid'))}, class={repr(b.get_attribute('class'))}")

driver.quit()
