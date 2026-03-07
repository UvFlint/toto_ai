from __future__ import annotations

from datetime import datetime
from pathlib import Path

from toto_ai.analyzer.models import FullReport


def write_report_to_file(report: FullReport) -> Path:
    """Write the analysis report to a markdown file in the reports/ folder."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    form_slug = (
        report.form_number.replace("#", "").replace(" ", "_") if report.form_number else "unknown"
    )
    filepath = reports_dir / f"report_{form_slug}_{timestamp}.md"

    lines: list[str] = []

    # Header
    lines.append("# Winner 16 AI Analysis Report")
    lines.append(f"**Form:** {report.form_number or 'Unknown'}  ")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Predictions table
    lines.append("## Predictions")
    lines.append("")
    ext_headers = [report.external_column.model_name] if report.external_column else []
    headers = (
        ["#", "Home", "Away"]
        + ext_headers
        + [col.model_name for col in report.columns]
        + ["Consensus", "Union"]
    )
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    consensus_map = {c.match_number: c for c in report.consensus}

    for match_num in range(1, 17):
        preds_by_model: dict[str, object] = {}
        for col in report.columns:
            for p in col.predictions:
                if p.match_number == match_num:
                    preds_by_model[col.model_name] = p
                    break

        first_pred = next(iter(preds_by_model.values()), None)
        home = first_pred.home_team if first_pred else ""  # type: ignore[union-attr]
        away = first_pred.away_team if first_pred else ""  # type: ignore[union-attr]

        row = [str(match_num), home, away]

        # External column cell
        if report.external_column:
            ext_pred = next(
                (p for p in report.external_column.predictions if p.match_number == match_num), None
            )
            row.append(f"{ext_pred.prediction} ({ext_pred.confidence:.0%})" if ext_pred else "-")

        for col in report.columns:
            p = preds_by_model.get(col.model_name)
            row.append(f"{p.prediction} ({p.confidence:.0%})" if p else "-")  # type: ignore[union-attr]

        con = consensus_map.get(match_num)
        row.append(f"{con.prediction} [{con.agreement_count}/3]" if con and con.prediction else "?")

        # Union (AI columns only)
        seen: set[str] = set()
        for col in report.columns:
            if col.column_type != "ai":
                continue
            p = preds_by_model.get(col.model_name)
            if p:
                seen.add(p.prediction)  # type: ignore[union-attr]
        union_str = "/".join(p for p in ["1", "X", "2"] if p in seen)
        row.append(union_str)

        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Banker picks
    lines.append("## Banker Picks (all models agree, confidence ≥70%)")
    lines.append(
        ", ".join(f"#{n}" for n in report.banker_picks) if report.banker_picks else "_None_"
    )
    lines.append("")

    # Upset alerts
    lines.append("## Upset Alerts (models disagree)")
    lines.append(
        ", ".join(f"#{n}" for n in report.upset_alerts) if report.upset_alerts else "_None_"
    )
    lines.append("")

    # Detailed reasoning per model
    lines.append("## Detailed Reasoning")
    all_detail_cols = ([report.external_column] if report.external_column else []) + list(
        report.columns
    )
    for col in all_detail_cols:
        lines.append(f"\n### {col.model_name}")
        for p in sorted(col.predictions, key=lambda x: x.match_number):
            lines.append(
                f"**#{p.match_number}** {p.home_team} vs {p.away_team}:"
                f" **{p.prediction}** ({p.confidence:.0%})"
            )
            lines.append(f"> {p.reasoning}")
            for factor in p.key_factors:
                lines.append(f"- {factor}")
        lines.append("")

    # Form fill guide
    lines.append("## Form Fill Guide")
    if report.external_column:
        preds_sorted = sorted(report.external_column.predictions, key=lambda x: x.match_number)
        guide = " | ".join(f"{p.match_number}:{p.prediction}" for p in preds_sorted)
        lines.append(f"**{report.external_column.model_name}:** {guide}")
    for col in report.columns:
        preds_sorted = sorted(col.predictions, key=lambda x: x.match_number)
        guide = " | ".join(f"{p.match_number}:{p.prediction}" for p in preds_sorted)
        lines.append(f"**{col.model_name}:** {guide}")

    consensus_guide = " | ".join(
        f"{c.match_number}:{c.prediction}" if c.prediction else f"{c.match_number}:?"
        for c in sorted(report.consensus, key=lambda x: x.match_number)
    )
    lines.append(f"**Consensus:** {consensus_guide}")

    union_parts: list[str] = []
    total_columns = 1
    for match_num in range(1, 17):
        seen_preds: set[str] = set()
        for col in report.columns:
            for p in col.predictions:
                if p.match_number == match_num:
                    seen_preds.add(p.prediction)
                    break
        u = "/".join(p for p in ["1", "X", "2"] if p in seen_preds)
        union_parts.append(f"{match_num}:{u}" if u else f"{match_num}:?")
        total_columns *= len(u.split("/")) if u else 1
    cost_nis = total_columns * 3
    union_guide = " | ".join(union_parts)
    lines.append(f"**Union:** {union_guide}")
    lines.append(
        f"_Covers **{total_columns}** column{'s' if total_columns != 1 else ''} — **{cost_nis} ₪**_"
    )

    # API usage & cost
    lines.append("")
    lines.append("## API Usage & Cost")
    has_cost = any(col.usage.cost_usd > 0 for col in report.columns)
    headers = ["Model", "Input tokens", "Output tokens"]
    if has_cost:
        headers.append("Cost (USD)")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    total_in = total_out = 0
    total_cost = 0.0
    for col in report.columns:
        u = col.usage
        total_in += u.input_tokens
        total_out += u.output_tokens
        total_cost += u.cost_usd
        row = [col.model_name, f"{u.input_tokens:,}", f"{u.output_tokens:,}"]
        if has_cost:
            row.append(f"${u.cost_usd:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    # Research costs (news, stats)
    for rc in report.research_costs:
        u = rc.usage
        total_in += u.input_tokens
        total_out += u.output_tokens
        total_cost += u.cost_usd
        row = [f"*{rc.label}*", f"{u.input_tokens:,}", f"{u.output_tokens:,}"]
        if has_cost:
            row.append(f"${u.cost_usd:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    total_row = ["**Total**", f"**{total_in:,}**", f"**{total_out:,}**"]
    if has_cost:
        total_row.append(f"**${total_cost:.4f}**")
    lines.append("| " + " | ".join(total_row) + " |")

    filepath.write_text("\n".join(lines), encoding="utf-8")

    # Save structured JSON alongside the markdown for tooling (e.g. submit script)
    json_path = filepath.with_suffix(".json")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    return filepath
