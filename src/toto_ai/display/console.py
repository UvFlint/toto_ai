from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from toto_ai.analyzer.models import FullReport
from toto_ai.console import console


def _confidence_color(confidence: float) -> str:
    if confidence >= 0.75:
        return "green"
    if confidence >= 0.5:
        return "yellow"
    return "red"


def _prediction_display(prediction: str, confidence: float) -> Text:
    color = _confidence_color(confidence)
    return Text(f"{prediction} ({confidence:.0%})", style=color)


def _union_for_match(report: FullReport, match_number: int) -> tuple[str, str]:
    """Return (union_str, color) for a match across all model columns."""
    seen: set[str] = set()
    for col in report.columns:
        if col.column_type != "ai":
            continue
        for pred in col.predictions:
            if pred.match_number == match_number:
                seen.add(pred.prediction)
                break
    union_str = "/".join(p for p in ["1", "X", "2"] if p in seen)
    color = "green" if len(seen) == 1 else ("yellow" if len(seen) == 2 else "red")
    return union_str, color


def display_report(report: FullReport) -> None:
    """Display the full analysis report in the console."""
    if not report.columns:
        console.print("[red]No predictions available.[/red]")
        return

    # Header
    console.print()
    console.rule("[bold blue]Winner 16 AI Analysis Report[/bold blue]")
    console.print()

    # Summary table
    table = Table(
        title="Match Predictions",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
        expand=True,
    )

    table.add_column("#", justify="center", width=3)
    table.add_column("Home", justify="right", min_width=15)
    table.add_column("Away", justify="left", min_width=15)

    if report.external_column:
        table.add_column(report.external_column.model_name, justify="center", width=14, style="dim")

    for col in report.columns:
        table.add_column(col.model_name, justify="center", width=14)

    table.add_column("Consensus", justify="center", width=10, style="bold")
    table.add_column("Union", justify="center", width=9)

    # Get match count from first available column
    first_col = report.columns[0] if report.columns else report.external_column
    match_count = len(first_col.predictions) if first_col else 0

    for i in range(1, match_count + 1):
        row: list[str | Text] = [str(i)]

        # Get home/away names from first available column's prediction
        home = away = "?"
        source_col = report.columns[0] if report.columns else report.external_column
        if source_col:
            for pred in source_col.predictions:
                if pred.match_number == i:
                    home = pred.home_team
                    away = pred.away_team
                    break

        row.append(home)
        row.append(away)

        # External column (reference only — shown dimmed, excluded from Union/Consensus)
        if report.external_column:
            ext_found = False
            for pred in report.external_column.predictions:
                if pred.match_number == i:
                    row.append(_prediction_display(pred.prediction, pred.confidence))
                    ext_found = True
                    break
            if not ext_found:
                row.append(Text("-", style="dim"))

        # Each model's prediction
        for col in report.columns:
            pred_found = False
            for pred in col.predictions:
                if pred.match_number == i:
                    row.append(_prediction_display(pred.prediction, pred.confidence))
                    pred_found = True
                    break
            if not pred_found:
                row.append(Text("-", style="dim"))

        # Consensus
        total_models = len(report.columns)
        for cm in report.consensus:
            if cm.match_number == i:
                if cm.prediction:
                    style = "bold green" if cm.agreement_count == total_models else "yellow"
                    row.append(
                        Text(f"{cm.prediction} [{cm.agreement_count}/{total_models}]", style=style)
                    )
                else:
                    row.append(Text("???", style="red"))
                break
        else:
            row.append(Text("-", style="dim"))

        # Union
        union_str, union_color = _union_for_match(report, i)
        row.append(Text(union_str, style=f"bold {union_color}"))

        table.add_row(*row)

    console.print(table)
    console.print()

    # Banker picks
    if report.banker_picks:
        bankers = ", ".join(f"#{n}" for n in report.banker_picks)
        console.print(
            Panel(
                f"[bold green]Banker Picks (high confidence, all models agree): {bankers}[/bold green]",
                title="Bankers",
                border_style="green",
            )
        )

    # Upset alerts
    if report.upset_alerts:
        upsets = ", ".join(f"#{n}" for n in report.upset_alerts)
        console.print(
            Panel(
                f"[bold red]Upset Alerts (models disagree): {upsets}[/bold red]",
                title="Caution",
                border_style="red",
            )
        )

    # Detailed reasoning for each model
    console.print()
    console.rule("[bold]Detailed Reasoning[/bold]")
    console.print()

    all_detail_cols = ([report.external_column] if report.external_column else []) + list(
        report.columns
    )
    for col in all_detail_cols:
        label = f"[bold cyan]{col.model_name}[/bold cyan]"
        if col is report.external_column:
            label = f"[dim cyan]{col.model_name} (reference)[/dim cyan]"
        console.print(label)
        for pred in sorted(col.predictions, key=lambda p: p.match_number):
            conf_color = _confidence_color(pred.confidence)
            console.print(
                f"  [{conf_color}]#{pred.match_number}[/{conf_color}] "
                f"{pred.home_team} vs {pred.away_team}: "
                f"[bold {conf_color}]{pred.prediction}[/bold {conf_color}] "
                f"({pred.confidence:.0%}) - {pred.reasoning}"
            )
            if pred.key_factors:
                factors = ", ".join(pred.key_factors)
                console.print(f"    [dim]Factors: {factors}[/dim]")
        console.print()

    # Column summary for filling the form
    console.rule("[bold]Form Fill Guide[/bold]")
    console.print()
    if report.external_column:
        preds = sorted(report.external_column.predictions, key=lambda p: p.match_number)
        column_str = " | ".join(f"{p.match_number}:{p.prediction}" for p in preds)
        console.print(f"[dim]{report.external_column.model_name}:[/dim] {column_str}")
    for col in report.columns:
        preds = sorted(col.predictions, key=lambda p: p.match_number)
        column_str = " | ".join(f"{p.match_number}:{p.prediction}" for p in preds)
        console.print(f"[cyan]{col.model_name}:[/cyan] {column_str}")
    console.print()

    if report.consensus:
        consensus_preds = sorted(
            [c for c in report.consensus if c.prediction],
            key=lambda c: c.match_number,
        )
        if consensus_preds:
            consensus_str = " | ".join(f"{c.match_number}:{c.prediction}" for c in consensus_preds)
            console.print(f"[bold green]Consensus:[/bold green] {consensus_str}")
            undecided = [c for c in report.consensus if not c.prediction]
            if undecided:
                undecided_str = ", ".join(f"#{c.match_number}" for c in undecided)
                console.print(f"[yellow]Undecided (pick your own): {undecided_str}[/yellow]")

    # Union guide (based on AI columns only)
    match_count = len(report.columns[0].predictions) if report.columns else 0
    union_parts: list[str] = []
    total_columns = 1
    for i in range(1, match_count + 1):
        u, _ = _union_for_match(report, i)
        union_parts.append(f"{i}:{u}")
        total_columns *= len(u.split("/"))
    cost_nis = total_columns * 3
    union_guide = " | ".join(union_parts)
    console.print(f"[bold magenta]Union:[/bold magenta] {union_guide}")
    console.print(
        f"[dim]Covers [bold]{total_columns}[/bold] column{'s' if total_columns != 1 else ''} "
        f"— [bold magenta]{cost_nis} ₪[/bold magenta][/dim]"
    )
    console.print()

    # Cost summary
    _display_cost_summary(report)


def display_test_report(report: FullReport, matches: list) -> None:
    """Display backtesting results comparing AI predictions to actual match results."""
    results_map: dict[int, str] = {}
    match_info: dict[int, tuple[str, str]] = {}
    for m in matches:
        if m.result:
            results_map[m.match_number] = m.result
            match_info[m.match_number] = (m.home_team, m.away_team)

    if not results_map:
        console.print("[red]No actual results available for comparison.[/red]")
        return

    console.print()
    console.rule("[bold blue]Backtesting Results[/bold blue]")
    console.print()

    # Build scoring table
    table = Table(
        title="Predictions vs Actual Results",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
        expand=True,
    )

    table.add_column("#", justify="center", width=3)
    table.add_column("Home", justify="right", min_width=12)
    table.add_column("Away", justify="left", min_width=12)
    table.add_column("Actual", justify="center", width=7, style="bold")

    if report.external_column:
        table.add_column(report.external_column.model_name, justify="center", width=12)

    for col in report.columns:
        table.add_column(col.model_name, justify="center", width=12)

    table.add_column("Consensus", justify="center", width=10)
    table.add_column("Union", justify="center", width=10)

    # Track scores
    model_scores: dict[str, int] = {}
    consensus_correct = 0
    consensus_total = 0
    union_correct = 0

    if report.external_column:
        model_scores[report.external_column.model_name] = 0
    for col in report.columns:
        model_scores[col.model_name] = 0

    match_count = len(results_map)

    for match_num in sorted(results_map.keys()):
        actual = results_map[match_num]
        home, away = match_info[match_num]
        row: list[str | Text] = [str(match_num), home, away, actual]

        # External column
        if report.external_column:
            pred = _find_prediction(report.external_column, match_num)
            if pred:
                correct = pred == actual
                if correct:
                    model_scores[report.external_column.model_name] += 1
                style = "bold green" if correct else "red"
                mark = "V" if correct else "X"
                row.append(Text(f"{pred} {mark}", style=style))
            else:
                row.append(Text("-", style="dim"))

        # AI model columns
        for col in report.columns:
            pred = _find_prediction(col, match_num)
            if pred:
                correct = pred == actual
                if correct:
                    model_scores[col.model_name] += 1
                style = "bold green" if correct else "red"
                mark = "V" if correct else "X"
                row.append(Text(f"{pred} {mark}", style=style))
            else:
                row.append(Text("-", style="dim"))

        # Consensus
        for cm in report.consensus:
            if cm.match_number == match_num and cm.prediction:
                correct = cm.prediction == actual
                consensus_total += 1
                if correct:
                    consensus_correct += 1
                style = "bold green" if correct else "red"
                mark = "V" if correct else "X"
                row.append(Text(f"{cm.prediction} {mark}", style=style))
                break
        else:
            row.append(Text("-", style="dim"))

        # Union
        union_str, _ = _union_for_match(report, match_num)
        union_preds = union_str.split("/") if union_str else []
        union_hit = actual in union_preds
        if union_hit:
            union_correct += 1
        if union_str:
            style = "bold green" if union_hit else "red"
            mark = "V" if union_hit else "X"
            row.append(Text(f"{union_str} {mark}", style=style))
        else:
            row.append(Text("-", style="dim"))

        table.add_row(*row)

    console.print(table)
    console.print()

    # Summary panel
    summary_table = Table(
        title="Accuracy Summary",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
    )
    summary_table.add_column("Model", min_width=14)
    summary_table.add_column("Correct", justify="center", width=8)
    summary_table.add_column("Total", justify="center", width=6)
    summary_table.add_column("Accuracy", justify="center", width=10)

    best_score = max(model_scores.values()) if model_scores else 0

    for name, correct in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
        pct = correct / match_count * 100 if match_count else 0
        style = "bold green" if correct == best_score else ""
        summary_table.add_row(
            Text(name, style=style),
            Text(str(correct), style=style),
            str(match_count),
            Text(f"{pct:.0f}%", style=style),
        )

    if consensus_total:
        pct = consensus_correct / consensus_total * 100
        summary_table.add_section()
        summary_table.add_row(
            Text("Consensus", style="bold"),
            Text(str(consensus_correct), style="bold"),
            str(consensus_total),
            Text(f"{pct:.0f}%", style="bold"),
        )

    if match_count:
        pct = union_correct / match_count * 100
        summary_table.add_section()
        summary_table.add_row(
            Text("Union", style="bold"),
            Text(str(union_correct), style="bold"),
            str(match_count),
            Text(f"{pct:.0f}%", style="bold"),
        )

    console.print(summary_table)
    console.print()


def _find_prediction(col, match_number: int) -> str | None:
    """Find a model's prediction for a given match number."""
    for pred in col.predictions:
        if pred.match_number == match_number:
            return pred.prediction
    return None


def _display_cost_summary(report: FullReport) -> None:
    """Display a cost summary table for all models."""
    cost_table = Table(
        title="API Usage & Cost",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
    )
    cost_table.add_column("Model", min_width=14)
    cost_table.add_column("Input tokens", justify="right")
    cost_table.add_column("Output tokens", justify="right")
    cost_table.add_column("Cost (USD)", justify="right")

    all_costs = [(col.model_name, col.usage) for col in report.columns]
    has_cost = any(u.cost_usd > 0 for _, u in all_costs)

    total_in = total_out = 0
    total_cost = 0.0
    for label, u in all_costs:
        total_in += u.input_tokens
        total_out += u.output_tokens
        total_cost += u.cost_usd
        cost_str = f"${u.cost_usd:.4f}" if has_cost else "-"
        cost_table.add_row(
            label,
            f"{u.input_tokens:,}",
            f"{u.output_tokens:,}",
            cost_str,
        )

    # Research costs (news, stats)
    if report.research_costs:
        cost_table.add_section()
        for rc in report.research_costs:
            u = rc.usage
            total_in += u.input_tokens
            total_out += u.output_tokens
            total_cost += u.cost_usd
            cost_str = f"${u.cost_usd:.4f}" if has_cost else "-"
            cost_table.add_row(
                f"[dim]{rc.label}[/dim]",
                f"{u.input_tokens:,}",
                f"{u.output_tokens:,}",
                cost_str,
            )

    total_cost_str = f"${total_cost:.4f}" if has_cost else "-"
    cost_table.add_section()
    cost_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_in:,}[/bold]",
        f"[bold]{total_out:,}[/bold]",
        f"[bold]{total_cost_str}[/bold]",
    )

    console.print(cost_table)
    console.print()
