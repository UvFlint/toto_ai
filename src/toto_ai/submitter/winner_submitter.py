from __future__ import annotations

import time
from typing import Literal

from toto_ai.analyzer.models import FullReport
from toto_ai.config import settings
from toto_ai.scraper.winner_scraper import WINNER_BASE_URL, WINNER16_SUBMIT_PATH
from toto_ai.submitter.models import SubmissionColumn, SubmissionResult
from toto_ai.console import console


def deduplicate_columns(report: FullReport) -> list[SubmissionColumn]:
    """Group model columns by their prediction tuple, merging duplicates."""
    seen: dict[tuple[str, ...], SubmissionColumn] = {}

    for col in report.columns:
        sorted_preds = sorted(col.predictions, key=lambda p: p.match_number)
        key = tuple(p.prediction for p in sorted_preds)
        preds_list: list[Literal["1", "X", "2"]] = list(key)

        if key in seen:
            seen[key].source_models.append(col.model_name)
        else:
            seen[key] = SubmissionColumn(
                column_index=len(seen) + 1,
                source_models=[col.model_name],
                predictions=preds_list,
            )

    return list(seen.values())


class WinnerSubmitter:
    """Submit predictions to winner.co.il using Selenium."""

    def submit_predictions(self, columns: list[SubmissionColumn]) -> list[SubmissionResult]:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()
        if settings.HEADLESS:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1280,800")
        options.add_argument("--lang=he-IL")

        console.print("[dim]Launching Chrome for submission...[/dim]")
        driver = webdriver.Chrome(options=options)
        results: list[SubmissionResult] = []

        try:
            self._login(driver)
            results = self._fill_all_and_submit(driver, columns)
        except Exception as e:
            console.print(f"[bold red]Submission error: {e}[/bold red]")
            submitted_indices = {r.column_index for r in results}
            for col in columns:
                if col.column_index not in submitted_indices:
                    results.append(
                        SubmissionResult(
                            column_index=col.column_index,
                            success=False,
                            message=str(e),
                        )
                    )
            if not settings.HEADLESS:
                input("Browser paused — press Enter to close...")
        finally:
            driver.quit()

        return results

    def _dismiss_zoom_engage(self, driver: object, wait_secs: int = 3) -> None:
        """Dismiss ZoomEngage popup if present, then remove all its DOM elements."""
        from selenium.webdriver.common.by import By

        time.sleep(wait_secs)
        try:
            close_btn = driver.find_elements(By.CSS_SELECTOR, "#ZA_CAMP_SLIDE_CLOSE_BUTTON")  # type: ignore[attr-defined]
            if close_btn:
                console.print("[dim]Dismissing ZoomEngage popup...[/dim]")
                driver.execute_script("arguments[0].click()", close_btn[0])  # type: ignore[attr-defined]
                time.sleep(2)
        except Exception:
            pass
        # Belt-and-suspenders: nuke all ZoomEngage DOM elements
        try:
            driver.execute_script(  # type: ignore[attr-defined]
                'document.querySelectorAll(\'[id*="ZA_CAMP"], [class*="za_"], [id*="zoomengage"]\')'
                ".forEach(function(el) { el.remove(); });"
            )
        except Exception:
            pass

    def _login(self, driver: object) -> None:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        console.print("[dim]Logging in to winner.co.il...[/dim]")
        driver.get(WINNER_BASE_URL + WINNER16_SUBMIT_PATH)  # type: ignore[attr-defined]

        # Wait for and dismiss ZoomEngage popup (appears on page load)
        self._dismiss_zoom_engage(driver, wait_secs=10)

        # Page renders broken on direct navigation — refresh fixes it
        driver.refresh()  # type: ignore[attr-defined]
        time.sleep(3)
        self._dismiss_zoom_engage(driver, wait_secs=3)

        # Click login button (ZoomEngage is gone so click registers)
        login_btn = None
        for selector in [
            "[class*='login']",
            "[class*='Login']",
            "button[class*='enter']",
            "[data-test*='login']",
            "a[href*='login']",
            ".header-login",
            "#login-btn",
        ]:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)  # type: ignore[attr-defined]
                if elements:
                    login_btn = elements[0]
                    break
            except Exception:
                continue

        if not login_btn:
            try:
                for btn in driver.find_elements(By.CSS_SELECTOR, "button, a, span"):  # type: ignore[attr-defined]
                    if btn.text.strip() in ("התחברות", "כניסה", "התחבר"):
                        login_btn = btn
                        break
            except Exception:
                pass

        if login_btn:
            login_btn.click()

        # Wait for login modal fields (username first, then password)
        username_input = WebDriverWait(driver, 10).until(  # type: ignore[arg-type]
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#userName, input[name='userName']"))
        )
        password_input = WebDriverWait(driver, 5).until(  # type: ignore[arg-type]
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#password, input[type='password']"))
        )

        username_input.clear()
        username_input.send_keys(settings.WINNER_USERNAME)
        password_input.clear()
        password_input.send_keys(settings.WINNER_PASSWORD)

        # Find submit button
        submit_btn = None
        for selector in [
            "button[type='submit']",
            "input[type='submit']",
            "button[class*='login']",
            "button[class*='submit']",
        ]:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)  # type: ignore[attr-defined]
                if elements:
                    submit_btn = elements[0]
                    break
            except Exception:
                continue

        if not submit_btn:
            try:
                for btn in driver.find_elements(By.CSS_SELECTOR, "button"):  # type: ignore[attr-defined]
                    if btn.text.strip() in ("התחברות", "כניסה", "התחבר", "שלח"):
                        submit_btn = btn
                        break
            except Exception:
                pass

        # Dismiss popup again before submitting the login modal
        self._dismiss_zoom_engage(driver, wait_secs=2)

        if submit_btn:
            driver.execute_script("arguments[0].click()", submit_btn)  # type: ignore[attr-defined]
        else:
            from selenium.webdriver.common.keys import Keys

            password_input.send_keys(Keys.RETURN)

        # Give the SPA time to process the auth response before polling
        time.sleep(2)

        # Verify login succeeded by waiting for the deposit/logged-in indicator
        def _login_succeeded(drv: object) -> bool:
            # Strategy 1: "הפקד" (deposit) text appears — clearest logged-in signal
            for el in drv.find_elements(By.CSS_SELECTOR, "button, a, span"):  # type: ignore[attr-defined]
                try:
                    if "הפקד" in (el.text or ""):
                        return True
                except Exception:
                    continue
            # Strategy 2: login/register button is still visible → not yet logged in
            for el in drv.find_elements(By.CSS_SELECTOR, "button, a, span"):  # type: ignore[attr-defined]
                try:
                    txt = (el.text or "").strip()
                    if "כניסה/הרשמה" in txt or txt == "כניסה":
                        return False
                except Exception:
                    continue
            # Strategy 3: yellow header button (original fallback)
            return bool(drv.find_elements(By.CSS_SELECTOR, "button.header-yellow-button"))  # type: ignore[attr-defined]

        try:
            WebDriverWait(driver, 20).until(_login_succeeded)  # type: ignore[arg-type]
            console.print("[green]Login verified — deposit button found[/green]")
        except Exception:
            raise RuntimeError(
                f"Login failed — deposit button not found. "
                f"URL: {driver.current_url}, Title: {driver.title}"  # type: ignore[attr-defined]
            )

    def _navigate_to_form(self, driver: object) -> None:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        console.print("[dim]Navigating to Winner 16 form...[/dim]")
        driver.get(WINNER_BASE_URL + WINNER16_SUBMIT_PATH)  # type: ignore[attr-defined]

        WebDriverWait(driver, 15).until(  # type: ignore[arg-type]
            EC.presence_of_element_located((By.CSS_SELECTOR, "[id^='bet-button-']"))
        )
        console.print("[dim]Form loaded — bet buttons detected[/dim]")
        self._dismiss_zoom_engage(driver, wait_secs=3)

    def _fill_column(self, driver: object, column_index: int, predictions: list[str]) -> None:
        """Fill a single column using deterministic button IDs.

        Button ID pattern: bet-button-{col}-{match}-{1|x|2}
        col: 0-based column index
        match: 0-based match index (0-15)
        prediction: '1', 'x', or '2' (lowercase)
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        console.print(f"[dim]Filling column {column_index} predictions...[/dim]")

        for match_idx, pred in enumerate(predictions):
            btn_id = f"bet-button-{column_index}-{match_idx}-{pred.lower()}"
            btn = WebDriverWait(driver, 5).until(  # type: ignore[arg-type]
                EC.presence_of_element_located((By.ID, btn_id))
            )
            driver.execute_script(  # type: ignore[attr-defined]
                "arguments[0].scrollIntoView({block: 'center'})", btn
            )
            driver.execute_script("arguments[0].click()", btn)  # type: ignore[attr-defined]
            time.sleep(0.15)

        console.print("[green]All 16 predictions filled[/green]")

    def _submit_form(self, driver: object) -> None:
        from selenium.webdriver.common.by import By

        console.print("[dim]Submitting form...[/dim]")

        submit_selectors = [
            "#send-betslip",
            "button[id='send-betslip']",
            "button[class*='submit']",
            "button[class*='send']",
            "[class*='submit-btn']",
            "input[type='submit']",
        ]

        submit_btn = None
        for selector in submit_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)  # type: ignore[attr-defined]
                if elements:
                    submit_btn = elements[0]
                    break
            except Exception:
                continue

        if not submit_btn:
            try:
                all_buttons = driver.find_elements(By.CSS_SELECTOR, "button, a, span, input")  # type: ignore[attr-defined]
                for btn in all_buttons:
                    text = btn.text.strip()
                    if text in ("שלח טופס", "שלח", "אישור", "שליחה"):
                        submit_btn = btn
                        break
            except Exception:
                pass

        if not submit_btn:
            raise RuntimeError(
                "Could not find submit button. Try running with HEADLESS=false to inspect the page."
            )

        self._dismiss_zoom_engage(driver, wait_secs=2)

        driver.execute_script("arguments[0].click()", submit_btn)  # type: ignore[attr-defined]
        time.sleep(5)
        console.print("[green]Form submitted[/green]")

    def _fill_all_and_submit(
        self, driver: object, columns: list[SubmissionColumn]
    ) -> list[SubmissionResult]:
        try:
            self._navigate_to_form(driver)
            for col in columns:
                models_str = ", ".join(col.source_models)
                console.print(f"\n[bold]Filling column {col.column_index} ({models_str})...[/bold]")
                self._fill_column(driver, col.column_index - 1, col.predictions)
            self._submit_form(driver)
            confirmation = self._handle_success_popup(driver)
            return [
                SubmissionResult(
                    column_index=col.column_index,
                    success=True,
                    message=f"Submitted successfully (confirmation: {confirmation})",
                )
                for col in columns
            ]
        except Exception as e:
            console.print(f"[bold red]Submission failed: {e}[/bold red]")
            return [
                SubmissionResult(column_index=col.column_index, success=False, message=str(e))
                for col in columns
            ]

    def _handle_success_popup(self, driver: object) -> str:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        WebDriverWait(driver, 15).until(  # type: ignore[arg-type]
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.modal-body"))
        )

        confirmation = ""
        try:
            info_ps = driver.find_elements(By.CSS_SELECTOR, "div.info p")  # type: ignore[attr-defined]
            if len(info_ps) >= 2:
                confirmation = info_ps[1].text.strip()
        except Exception:
            pass

        console.print(f"[green]Form submitted successfully! Confirmation: {confirmation}[/green]")

        try:
            buttons = driver.find_elements(By.CSS_SELECTOR, "div.form-buttons button")  # type: ignore[attr-defined]
            for btn in buttons:
                if "משחק חדש" in btn.text:
                    btn.click()
                    break
        except Exception:
            pass

        time.sleep(2)
        return confirmation
