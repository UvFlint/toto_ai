from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

from toto_ai.analyzer.models import FullReport


def send_report_email(
    report: FullReport,
    report_path: Path,
    *,
    sender: str,
    password: str,
    recipient: str,
) -> None:
    """Send the markdown report to recipient via Gmail SMTP."""
    form_label = report.form_number or "Unknown"
    subject = f"Winner 16 AI Report — Form {form_label}"

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    body = report_path.read_text(encoding="utf-8")
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # Attach the markdown file
    attachment = MIMEBase("application", "octet-stream")
    attachment.set_payload(report_path.read_bytes())
    encoders.encode_base64(attachment)
    attachment.add_header(
        "Content-Disposition",
        f'attachment; filename="{report_path.name}"',
    )
    msg.attach(attachment)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
