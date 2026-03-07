"""
Standalone script to test the Winner API login and form submission flow.

Usage:
    uv run python tests/test_winner_api.py

This discovers the API flow and validates that login works.
Delete this file once the API flow is confirmed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from uuid import uuid4

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

WINNER_BASE_URL = "https://www.winner.co.il"
LOGIN_ENDPOINT = "/api/v2/publicapi/Login"

DEVICE_ID = uuid4().hex

COMMON_HEADERS = {
    "appversion": "2.5.1",
    "deviceid": DEVICE_ID,
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "useragentdata": json.dumps(
        {
            "brands": [
                {"brand": "Google Chrome", "version": "131"},
                {"brand": "Chromium", "version": "131"},
                {"brand": "Not_A Brand", "version": "24"},
            ],
            "mobile": False,
            "platform": "Windows",
        }
    ),
    "content-type": "application/json",
    "accept": "application/json, text/plain, */*",
    "origin": WINNER_BASE_URL,
    "referer": f"{WINNER_BASE_URL}/",
}


async def test_login() -> dict | None:
    """Test login to winner.co.il API."""
    username = os.environ.get("WINNER_USERNAME", "")
    password = os.environ.get("WINNER_PASSWORD", "")

    if not username or not password:
        print("ERROR: Set WINNER_USERNAME and WINNER_PASSWORD in .env")
        return None

    # The hash appears to be MD5 of the password
    password_hash = hashlib.md5(password.encode()).hexdigest()

    payload = {
        "userName": username,
        "password": password,
        "hash": password_hash,
    }

    print(f"Attempting login as: {username}")
    print(f"Device ID: {DEVICE_ID}")
    print(f"Password hash (MD5): {password_hash}")
    print(f"Endpoint: {WINNER_BASE_URL}{LOGIN_ENDPOINT}")
    print()

    async with httpx.AsyncClient(
        base_url=WINNER_BASE_URL,
        headers=COMMON_HEADERS,
        follow_redirects=True,
        timeout=30.0,
    ) as client:
        # First, visit the main page to get Incapsula cookies
        print("Step 1: Fetching main page for cookies...")
        try:
            main_resp = await client.get("/")
            print(f"  Status: {main_resp.status_code}")
            print(f"  Cookies: {dict(client.cookies)}")
        except Exception as e:
            print(f"  Warning: Main page fetch failed: {e}")

        print()

        # Step 2: Attempt login
        print("Step 2: Sending login request...")
        try:
            resp = await client.post(LOGIN_ENDPOINT, json=payload)
            print(f"  Status: {resp.status_code}")
            print(f"  Headers: {dict(resp.headers)}")

            try:
                data = resp.json()
                # Mask sensitive fields
                safe_data = {k: v for k, v in data.items()} if isinstance(data, dict) else data
                if isinstance(safe_data, dict):
                    for key in ("password", "token", "sessionId"):
                        if key in safe_data:
                            safe_data[key] = "***MASKED***"
                print(f"  Response: {json.dumps(safe_data, indent=2, ensure_ascii=False)}")
                return data if isinstance(data, dict) else None
            except Exception:
                print(f"  Response text: {resp.text[:500]}")
                return None
        except httpx.HTTPError as e:
            print(f"  HTTP Error: {e}")
            return None


async def test_form_submission(login_data: dict) -> None:
    """Test form submission endpoint (placeholder — endpoint TBD)."""
    print("\n" + "=" * 60)
    print("Form submission endpoint not yet captured.")
    print("To capture it:")
    print("  1. Open Chrome DevTools (F12) → Network tab")
    print("  2. Log in to winner.co.il")
    print("  3. Navigate to Winner 16 form")
    print("  4. Fill and submit a column")
    print("  5. Copy the request as cURL and paste it here")
    print("=" * 60)


async def main() -> None:
    print("=" * 60)
    print("Winner API Discovery Test")
    print("=" * 60)
    print()

    login_data = await test_login()

    if login_data:
        print("\n[OK] Login request completed")
        await test_form_submission(login_data)
    else:
        print("\n[FAIL] Login failed or returned no data")
        print()
        print("If you got a 403 or empty response, the API likely requires")
        print("a validation_token from Incapsula bot protection.")
        print("In that case, we'll fall back to the Selenium approach.")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())




