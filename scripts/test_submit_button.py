"""
Standalone test: verifies the #send-betslip button is clickable
after ZoomEngage overlay removal. Does NOT submit the form.

Run with: uv run python scripts/test_submit_button.py
"""
import sys
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from toto_ai.config import settings
from toto_ai.scraper.winner_scraper import WINNER_BASE_URL, WINNER16_PATH

options = Options()
options.add_argument("--window-size=1280,800")
options.add_argument("--lang=he-IL")
# Non-headless so user can observe

driver = webdriver.Chrome(options=options)

try:
    # --- Login ---
    print("Navigating to winner.co.il...")
    driver.get(WINNER_BASE_URL)

    # Wait for ZoomEngage popup (appears on page load, before clicking anything)
    print("Waiting 10s for ZoomEngage popup to appear...")
    time.sleep(10)
    close_btn = driver.find_elements(By.CSS_SELECTOR, "#ZA_CAMP_SLIDE_CLOSE_BUTTON")
    if close_btn:
        print("ZoomEngage close button found — clicking it...")
        driver.execute_script("arguments[0].click()", close_btn[0])
        time.sleep(1)
    else:
        print("ZoomEngage close button not found (popup did not appear)")

    # Now click the login button (ZoomEngage is gone so click should register)
    login_btn = None
    for selector in ["[class*='login']", "[class*='Login']"]:
        els = driver.find_elements(By.CSS_SELECTOR, selector)
        if els:
            login_btn = els[0]
            break
    if not login_btn:
        for btn in driver.find_elements(By.CSS_SELECTOR, "button, a, span"):
            if btn.text.strip() in ("התחברות", "כניסה", "התחבר"):
                login_btn = btn
                break

    if login_btn:
        print(f"Login button found: {login_btn.text.strip()!r} — clicking...")
        login_btn.click()
    else:
        print("FAIL: Could not find login button")
        input("Press Enter to close browser...")
        sys.exit(1)

    # Wait for password field (login modal animation)
    try:
        password_input = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#password, input[type='password']"))
        )
    except Exception:
        print("FAIL: Login modal did not appear (password field not found)")
        input("Press Enter to close browser...")
        sys.exit(1)

    try:
        username_input = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#userName, input[name='userName']"))
        )
    except Exception:
        print("FAIL: Could not find username/email field")
        input("Press Enter to close browser...")
        sys.exit(1)

    username_input.clear()
    username_input.send_keys(settings.WINNER_USERNAME)
    password_input.clear()
    password_input.send_keys(settings.WINNER_PASSWORD)

    submit = None
    for sel in ["button[type='submit']", "input[type='submit']"]:
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        if els:
            submit = els[0]
            break
    if submit:
        submit.click()
    else:
        from selenium.webdriver.common.keys import Keys
        password_input.send_keys(Keys.RETURN)

    time.sleep(5)
    print("Login attempted.")

    # --- Navigate to form ---
    print("Navigating to Winner 16 form...")
    driver.get(WINNER_BASE_URL + WINNER16_PATH)
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "[id^='bet-button-']"))
    )
    print("Form loaded.")

    # --- Check for overlays before removal ---
    overlays_before = driver.find_elements(
        By.CSS_SELECTOR, '[id*="ZA_CAMP"], [class*="za_"], [id*="zoomengage"]'
    )
    print(f"ZoomEngage elements found before removal: {len(overlays_before)}")
    for el in overlays_before:
        print(f"  - id={el.get_attribute('id')!r} class={el.get_attribute('class')!r}")

    # --- Dismiss ZoomEngage popup via its close button if present ---
    close_btn = driver.find_elements(By.CSS_SELECTOR, "#ZA_CAMP_SLIDE_CLOSE_BUTTON")
    if close_btn:
        print("ZoomEngage close button found — clicking it...")
        driver.execute_script("arguments[0].click()", close_btn[0])
        time.sleep(1)
    else:
        print("ZoomEngage close button not found (popup may not have appeared yet)")

    overlays_after = driver.find_elements(
        By.CSS_SELECTOR, '[id*="ZA_CAMP"], [class*="za_"], [id*="zoomengage"]'
    )
    print(f"ZoomEngage elements remaining after dismiss: {len(overlays_after)}")

    # --- Find and inspect submit button ---
    btns = driver.find_elements(By.CSS_SELECTOR, "#send-betslip")
    if not btns:
        print("FAIL: #send-betslip button not found")
        input("Press Enter to close browser...")
        sys.exit(1)

    submit_btn = btns[0]
    is_displayed = submit_btn.is_displayed()
    is_enabled = submit_btn.is_enabled()
    print(f"#send-betslip: displayed={is_displayed}, enabled={is_enabled}")

    # Scroll to and highlight it (red outline = easy to spot)
    driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center'}); "
        "arguments[0].style.outline = '3px solid red';",
        submit_btn,
    )
    time.sleep(2)

    # Check what element is actually on top at the button's center point
    rect = driver.execute_script(
        "var r = arguments[0].getBoundingClientRect(); return r;", submit_btn
    )
    cx = (rect["left"] + rect["right"]) / 2
    cy = (rect["top"] + rect["bottom"]) / 2
    top_element = driver.execute_script(f"return document.elementFromPoint({cx}, {cy});")
    top_id = top_element.get_attribute("id") if top_element else None
    top_tag = top_element.tag_name if top_element else None

    if top_id == "send-betslip" or top_element == submit_btn:
        print(f"PASS: Button is on top at ({cx:.0f}, {cy:.0f}) — nothing blocking it")
    else:
        print(
            f"FAIL: Element on top at ({cx:.0f}, {cy:.0f}) is "
            f"<{top_tag} id={top_id!r}> — still blocked"
        )

    input("Press Enter to close browser...")

finally:
    driver.quit()
