"""
Report Generator — HTML reports with equity curves, drawdown charts,
monthly returns tables, and comprehensive metrics.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from backtester.engine import BacktestResult
from backtester.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def generate_report(
    result: BacktestResult,
    output_dir: Optional[str] = None,
) -> str:
    """
    Generate an HTML backtest report.

    Args:
        result: BacktestResult from engine.run()
        output_dir: Directory to save report (default: backtester/reports/)

    Returns:
        Path to the generated HTML file
    """
    output_dir = output_dir or REPORTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    m = result.metrics or PerformanceMetrics()
    filename = f"backtest_{result.run_id}.html"
    filepath = os.path.join(output_dir, filename)

    # Prepare chart data
    equity_labels = []
    equity_values = []
    dd_values = []
    peak = result.starting_equity

    for ts, eq in result.equity_curve:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        equity_labels.append(dt.strftime("%Y-%m-%d %H:%M"))
        equity_values.append(round(eq, 4))
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        dd_values.append(round(-dd, 2))

    # Monthly returns data
    monthly_labels = sorted(m.monthly_returns.keys())
    monthly_values = [round(m.monthly_returns[k], 2) for k in monthly_labels]
    monthly_colors = [
        "'rgba(34,197,94,0.7)'" if v >= 0 else "'rgba(239,68,68,0.7)'"
        for v in monthly_values
    ]

    # Best/worst trades
    best_trades = sorted(result.trades, key=lambda t: t.get("pnl_usd", 0), reverse=True)[:10]
    worst_trades = sorted(result.trades, key=lambda t: t.get("pnl_usd", 0))[:10]

    html = _build_html(
        result=result,
        m=m,
        equity_labels=equity_labels,
        equity_values=equity_values,
        dd_values=dd_values,
        monthly_labels=monthly_labels,
        monthly_values=monthly_values,
        monthly_colors=monthly_colors,
        best_trades=best_trades,
        worst_trades=worst_trades,
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Report saved to {filepath}")
    return filepath


def _build_html(
    result: BacktestResult,
    m: PerformanceMetrics,
    equity_labels: list,
    equity_values: list,
    dd_values: list,
    monthly_labels: list,
    monthly_values: list,
    monthly_colors: list,
    best_trades: list,
    worst_trades: list,
) -> str:
    """Build the full HTML report string."""
    start_dt = datetime.fromtimestamp(result.start_time, tz=timezone.utc).strftime("%Y-%m-%d")
    end_dt = datetime.fromtimestamp(result.end_time, tz=timezone.utc).strftime("%Y-%m-%d")

    # Build breakdown tables
    symbol_rows = _breakdown_rows(m.by_symbol)
    direction_rows = _breakdown_rows(m.by_direction)
    regime_rows = _breakdown_rows(m.by_regime)
    session_rows = _breakdown_rows(m.by_session)
    exit_rows = _breakdown_rows(m.by_exit_reason)

    # Build trades table
    best_trade_rows = _trade_rows(best_trades)
    worst_trade_rows = _trade_rows(worst_trades)

    # Downsample chart data if too many points
    max_points = 500
    if len(equity_labels) > max_points:
        step = len(equity_labels) // max_points
        equity_labels = equity_labels[::step]
        equity_values = equity_values[::step]
        dd_values = dd_values[::step]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Report — {result.run_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 20px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; color: #f8fafc; }}
  h2 {{ font-size: 1.1rem; margin: 24px 0 12px; color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 6px; }}
  .subtitle {{ color: #64748b; font-size: 0.85rem; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 14px; }}
  .card .label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
  .card .value {{ font-size: 1.3rem; font-weight: 700; margin-top: 4px; }}
  .positive {{ color: #22c55e; }}
  .negative {{ color: #ef4444; }}
  .neutral {{ color: #f8fafc; }}
  .chart-container {{ background: #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
  canvas {{ max-height: 300px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ text-align: left; padding: 8px 10px; background: #1e293b; color: #94a3b8;
       font-weight: 600; border-bottom: 2px solid #334155; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #1e293b; }}
  tr:hover {{ background: #1e293b; }}
  .table-wrap {{ background: #1e293b; border-radius: 8px; padding: 4px; margin-bottom: 16px; overflow-x: auto; }}
  .overfit {{ color: #f59e0b; font-weight: 700; }}
  .footer {{ text-align: center; color: #475569; font-size: 0.75rem; margin-top: 30px; padding: 12px; }}
</style>
</head>
<body>
<div class="container">

<h1>Backtest Report</h1>
<p class="subtitle">Run ID: {result.run_id} &nbsp;|&nbsp; {start_dt} to {end_dt} &nbsp;|&nbsp;
   Symbols: {', '.join(result.symbols)}</p>

<!-- Key Metrics -->
<h2>Performance Summary</h2>
<div class="grid">
  <div class="card">
    <div class="label">Total Trades</div>
    <div class="value neutral">{m.total_trades}</div>
  </div>
  <div class="card">
    <div class="label">Win Rate</div>
    <div class="value {'positive' if m.win_rate >= 0.5 else 'negative'}">{m.win_rate:.1%}</div>
  </div>
  <div class="card">
    <div class="label">Total P&L</div>
    <div class="value {'positive' if m.total_pnl >= 0 else 'negative'}">${m.total_pnl:.2f}</div>
  </div>
  <div class="card">
    <div class="label">Total Return</div>
    <div class="value {'positive' if m.total_return_pct >= 0 else 'negative'}">{m.total_return_pct:.1f}%</div>
  </div>
  <div class="card">
    <div class="label">Profit Factor</div>
    <div class="value {'positive' if m.profit_factor >= 1.5 else 'negative' if m.profit_factor < 1.0 else 'neutral'}">{m.profit_factor:.2f}</div>
  </div>
  <div class="card">
    <div class="label">Sharpe Ratio</div>
    <div class="value {'positive' if m.sharpe_ratio >= 1.0 else 'negative' if m.sharpe_ratio < 0 else 'neutral'}">{m.sharpe_ratio:.2f}</div>
  </div>
  <div class="card">
    <div class="label">Sortino Ratio</div>
    <div class="value {'positive' if m.sortino_ratio >= 1.0 else 'negative' if m.sortino_ratio < 0 else 'neutral'}">{m.sortino_ratio:.2f}</div>
  </div>
  <div class="card">
    <div class="label">Calmar Ratio</div>
    <div class="value neutral">{m.calmar_ratio:.2f}</div>
  </div>
  <div class="card">
    <div class="label">Max Drawdown</div>
    <div class="value negative">{m.max_drawdown_pct:.1f}%</div>
  </div>
  <div class="card">
    <div class="label">Recovery Factor</div>
    <div class="value neutral">{m.recovery_factor:.2f}</div>
  </div>
  <div class="card">
    <div class="label">Avg Win / Loss</div>
    <div class="value neutral">${m.avg_win:.3f} / ${m.avg_loss:.3f}</div>
  </div>
  <div class="card">
    <div class="label">Avg R:R</div>
    <div class="value {'positive' if m.avg_rr >= 1.5 else 'neutral'}">{m.avg_rr:.2f}</div>
  </div>
  <div class="card">
    <div class="label">Trades/Day</div>
    <div class="value neutral">{m.trades_per_day:.1f}</div>
  </div>
  <div class="card">
    <div class="label">Win/Loss Streak</div>
    <div class="value neutral">{m.longest_win_streak}W / {m.longest_loss_streak}L</div>
  </div>
  <div class="card">
    <div class="label">Total Fees</div>
    <div class="value negative">${m.total_fees:.2f} ({m.fee_drag_pct:.1f}%)</div>
  </div>
  <div class="card">
    <div class="label">Expectancy</div>
    <div class="value {'positive' if m.expectancy > 0 else 'negative'}">${m.expectancy:.4f}</div>
  </div>
</div>

<!-- Equity Curve -->
<h2>Equity Curve</h2>
<div class="chart-container">
  <canvas id="equityChart"></canvas>
</div>

<!-- Drawdown -->
<h2>Drawdown</h2>
<div class="chart-container">
  <canvas id="ddChart"></canvas>
</div>

<!-- Monthly Returns -->
<h2>Monthly Returns</h2>
<div class="chart-container">
  <canvas id="monthlyChart"></canvas>
</div>

<!-- Breakdown Tables -->
<h2>Win Rate by Symbol</h2>
<div class="table-wrap">
<table>
  <tr><th>Symbol</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>P&L</th><th>Avg P&L</th><th>PF</th></tr>
  {symbol_rows}
</table>
</div>

<h2>Win Rate by Direction</h2>
<div class="table-wrap">
<table>
  <tr><th>Direction</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>P&L</th><th>Avg P&L</th><th>PF</th></tr>
  {direction_rows}
</table>
</div>

<h2>Win Rate by Regime</h2>
<div class="table-wrap">
<table>
  <tr><th>Regime</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>P&L</th><th>Avg P&L</th><th>PF</th></tr>
  {regime_rows}
</table>
</div>

<h2>Win Rate by Session</h2>
<div class="table-wrap">
<table>
  <tr><th>Session</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>P&L</th><th>Avg P&L</th><th>PF</th></tr>
  {session_rows}
</table>
</div>

<h2>Exit Reasons</h2>
<div class="table-wrap">
<table>
  <tr><th>Reason</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>P&L</th><th>Avg P&L</th><th>PF</th></tr>
  {exit_rows}
</table>
</div>

<!-- Best/Worst Trades -->
<h2>Best Trades</h2>
<div class="table-wrap">
<table>
  <tr><th>ID</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Duration</th><th>Reason</th></tr>
  {best_trade_rows}
</table>
</div>

<h2>Worst Trades</h2>
<div class="table-wrap">
<table>
  <tr><th>ID</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Duration</th><th>Reason</th></tr>
  {worst_trade_rows}
</table>
</div>

<div class="footer">
  Generated by Backtester &mdash; {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
</div>

</div><!-- .container -->

<script>
const eqLabels = {json.dumps(equity_labels)};
const eqValues = {json.dumps(equity_values)};
const ddValues = {json.dumps(dd_values)};
const moLabels = {json.dumps(monthly_labels)};
const moValues = {json.dumps(monthly_values)};
const moColors = [{','.join(monthly_colors)}];

// Equity chart
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: eqLabels,
    datasets: [{{
      label: 'Equity (USD)',
      data: eqValues,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.1)',
      fill: true,
      tension: 0.2,
      pointRadius: 0,
      borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ display: true, ticks: {{ maxTicksLimit: 10, color: '#64748b' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// Drawdown chart
new Chart(document.getElementById('ddChart'), {{
  type: 'line',
  data: {{
    labels: eqLabels,
    datasets: [{{
      label: 'Drawdown %',
      data: ddValues,
      borderColor: '#ef4444',
      backgroundColor: 'rgba(239,68,68,0.1)',
      fill: true,
      tension: 0.2,
      pointRadius: 0,
      borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ display: true, ticks: {{ maxTicksLimit: 10, color: '#64748b' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});

// Monthly returns
new Chart(document.getElementById('monthlyChart'), {{
  type: 'bar',
  data: {{
    labels: moLabels,
    datasets: [{{
      label: 'Return %',
      data: moValues,
      backgroundColor: moColors,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#64748b' }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""


def _breakdown_rows(breakdown: Dict[str, dict]) -> str:
    """Generate HTML table rows from a breakdown dict."""
    rows = []
    for key, vals in sorted(breakdown.items()):
        pnl = vals["total_pnl"]
        pnl_class = "positive" if pnl > 0 else "negative" if pnl < 0 else "neutral"
        rows.append(
            f"<tr>"
            f"<td>{key}</td>"
            f"<td>{vals['trades']}</td>"
            f"<td>{vals['wins']}</td>"
            f"<td>{vals['losses']}</td>"
            f"<td>{vals['win_rate']:.1%}</td>"
            f"<td class='{pnl_class}'>${pnl:.3f}</td>"
            f"<td>${vals['avg_pnl']:.4f}</td>"
            f"<td>{vals['profit_factor']:.2f}</td>"
            f"</tr>"
        )
    return "\n  ".join(rows)


def _trade_rows(trades: list) -> str:
    """Generate HTML table rows for trade listings."""
    rows = []
    for t in trades:
        pnl = t.get("pnl_usd", 0)
        pnl_class = "positive" if pnl > 0 else "negative"
        duration = t.get("duration_s", 0) or 0
        if duration >= 3600:
            dur_str = f"{duration / 3600:.1f}h"
        elif duration >= 60:
            dur_str = f"{duration / 60:.0f}m"
        else:
            dur_str = f"{duration:.0f}s"

        rows.append(
            f"<tr>"
            f"<td>{t.get('trade_id', '')}</td>"
            f"<td>{t.get('symbol', '')}</td>"
            f"<td>{t.get('side', '')}</td>"
            f"<td>${t.get('entry_price', 0):.2f}</td>"
            f"<td>${t.get('exit_price', 0) or 0:.2f}</td>"
            f"<td class='{pnl_class}'>${pnl:.4f}</td>"
            f"<td>{dur_str}</td>"
            f"<td>{t.get('exit_reason', '')}</td>"
            f"</tr>"
        )
    return "\n  ".join(rows)
