FROM python:3.11-slim

# Install system dependencies + Google Chrome
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget gnupg curl ca-certificates fonts-noto \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub \
        | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] \
        http://dl.google.com/linux/chrome/deb/ stable main" \
        > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y --no-install-recommends google-chrome-stable \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency manifest + source (hatchling needs src/ to build the package)
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/

# Install all dependencies (no dev extras); uv creates .venv inside /app
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
# Force headless Chrome inside the container
ENV HEADLESS=true

# Selenium Manager (bundled with Selenium 4.6+) will auto-download the matching
# ChromeDriver on first run — no manual chromedriver install needed.
CMD ["uv", "run", "toto", "--premium", "--schedule", "--send-auto"]
