"""Isolated test for Winner 16 form submission flow.

Fills only 14/16 matches to prevent actual submission.
Run: uv run python test_submit.py
"""

import time

from dotenv import load_dotenv

load_dotenv()

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from toto_ai.scraper.winner_scraper import WINNER_BASE_URL, WINNER16_SUBMIT_PATH
from toto_ai.submitter.winner_submitter import WinnerSubmitter

MATCHES_TO_FILL = 14  # fill only 14/16 to prevent real submission


def main():
    # Force non-headless so we can watch
    from toto_ai.config import settings

    settings.HEADLESS = False

    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,800")
    options.add_argument("--lang=he-IL")

    print("Launching Chrome (visible)...")
    driver = webdriver.Chrome(options=options)
    submitter = WinnerSubmitter()

    try:
        # Step 1-4: Login (includes navigation, refresh, ZoomEngage dismiss)
        print("\n=== STEP 1: Login ===")
        submitter._login(driver)
        print("Login successful!")

        # Step 5: Navigate to form
        print("\n=== STEP 2: Navigate to form ===")
        submitter._navigate_to_form(driver)
        print("Form loaded!")

        # Step 6: Fill 14 out of 16 matches (all "1" predictions, column 0)
        print(f"\n=== STEP 3: Fill {MATCHES_TO_FILL}/16 matches ===")
        for match_idx in range(MATCHES_TO_FILL):
            btn_id = f"bet-button-0-{match_idx}-1"
            btn = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.ID, btn_id))
            )
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'})", btn)
            driver.execute_script("arguments[0].click()", btn)
            print(f"  Filled match {match_idx + 1}: '1' (button: {btn_id})")
            time.sleep(0.15)
        print(f"Filled {MATCHES_TO_FILL} matches. Matches 15 & 16 left empty.")

        # Step 7: Try to click submit (should fail — form incomplete)
        print("\n=== STEP 4: Click submit button ===")
        try:
            submitter._submit_form(driver)
            print("Submit button clicked (form should reject due to incomplete predictions)")
        except Exception as e:
            print(f"Submit failed as expected: {e}")

        # Pause for inspection
        print("\n=== DONE ===")
        print("Browser will stay open for 60 seconds for inspection...")
        time.sleep(60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print("Browser will stay open for 60 seconds for inspection...")
        time.sleep(60)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
