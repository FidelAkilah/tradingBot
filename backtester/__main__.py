"""
CLI entry point for the backtesting framework.

Usage:
  python -m backtester sync-data --days 90
  python -m backtester run --start 2025-01-01 --end 2025-03-20
  python -m backtester optimize --param confidence_threshold --range 0.4,0.8,0.05
  python -m backtester report --run-id <id>
  python -m backtester status
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

# Add parent directory to path
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from config import CONFIG
from backtester.data_manager import DataManager
from backtester.engine import BacktestEngine, BacktestResult
from backtester.metrics import PerformanceMetrics
from backtester.optimizer import WalkForwardOptimizer, parse_param_range, OPTIMIZABLE_PARAMS
from backtester.report import generate_report

logger = logging.getLogger("backtester")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Commands ────────────────────────────────────────────────────

def cmd_sync_data(args):
    """Download historical candle data from Binance."""
    dm = DataManager()
    symbols = args.symbols.split(",") if args.symbols else CONFIG.trading.manual_pairs
    timeframes = args.timeframes.split(",") if args.timeframes else ["15m", "1h", "4h", "1d"]

    print(f"Syncing data for {len(symbols)} symbols, {len(timeframes)} timeframes, {args.days} days")
    print(f"  Symbols: {symbols}")
    print(f"  Timeframes: {timeframes}")

    asyncio.run(dm.sync_data(
        symbols=symbols,
        timeframes=timeframes,
        days=args.days,
        rate_limit_delay=args.rate_limit,
    ))

    # Show status after sync
    _print_sync_status(dm)
    dm.close()


def cmd_run(args):
    """Run a backtest."""
    dm = DataManager()
    symbols = args.symbols.split(",") if args.symbols else CONFIG.trading.manual_pairs

    print(f"Running backtest: {args.start} to {args.end}")
    print(f"  Symbols: {symbols}")

    engine = BacktestEngine(data_manager=dm)
    result = engine.run(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        primary_tf=args.timeframe,
    )

    _print_results(result)

    # Auto-generate report
    if not args.no_report:
        report_path = generate_report(result)
        print(f"\nReport saved: {report_path}")

    # Save result for later report generation
    _save_result(result)
    dm.close()


def cmd_optimize(args):
    """Run walk-forward parameter optimization."""
    dm = DataManager()
    symbols = args.symbols.split(",") if args.symbols else CONFIG.trading.manual_pairs
    param_values = parse_param_range(args.range)

    print(f"Optimizing: {args.param}")
    print(f"  Values: {param_values}")
    print(f"  Period: {args.start} to {args.end}")

    optimizer = WalkForwardOptimizer(data_manager=dm)
    result = optimizer.optimize(
        param_name=args.param,
        param_range=param_values,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
    )

    _print_optimization_result(result)
    dm.close()


def cmd_report(args):
    """Generate report from a saved run."""
    result = _load_result(args.run_id)
    if result is None:
        print(f"Error: No saved result found for run_id '{args.run_id}'")
        sys.exit(1)

    report_path = generate_report(result)
    print(f"Report saved: {report_path}")


def cmd_status(args):
    """Show data sync status."""
    dm = DataManager()
    _print_sync_status(dm)
    dm.close()


def cmd_list_params(args):
    """List optimizable parameters."""
    print("\nOptimizable Parameters:")
    print("-" * 70)
    for name, info in sorted(OPTIMIZABLE_PARAMS.items()):
        print(f"  {name:30s}  {info['path']:35s}")
        print(f"    {info['description']}")
    print()


# ── Output helpers ──────────────────────────────────────────────

def _print_results(result: BacktestResult):
    """Print backtest results to console."""
    m = result.metrics
    if m is None:
        print("No results (no trades executed)")
        return

    print(f"\n{'=' * 60}")
    print(f"  BACKTEST RESULTS — {result.run_id}")
    print(f"{'=' * 60}")
    print(f"  Period:          {datetime.fromtimestamp(result.start_time, tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(result.end_time, tz=timezone.utc).strftime('%Y-%m-%d')}")
    print(f"  Symbols:         {', '.join(result.symbols)}")
    print(f"  Starting Equity: ${result.starting_equity:.2f}")
    print(f"  Ending Equity:   ${m.ending_equity:.2f}")
    print()
    print(f"  Total Trades:    {m.total_trades}")
    print(f"  Win Rate:        {m.win_rate:.1%} ({m.winning_trades}W / {m.losing_trades}L)")
    print(f"  Total P&L:       ${m.total_pnl:+.2f} ({m.total_return_pct:+.1f}%)")
    print(f"  Profit Factor:   {m.profit_factor:.2f}")
    print(f"  Avg Win:         ${m.avg_win:.4f}")
    print(f"  Avg Loss:        ${m.avg_loss:.4f}")
    print(f"  Avg R:R:         {m.avg_rr:.2f}")
    print(f"  Expectancy:      ${m.expectancy:.4f}")
    print()
    print(f"  Sharpe Ratio:    {m.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:   {m.sortino_ratio:.2f}")
    print(f"  Calmar Ratio:    {m.calmar_ratio:.2f}")
    print(f"  Max Drawdown:    {m.max_drawdown_pct:.1f}% (${m.max_drawdown_usd:.2f})")
    print(f"  Recovery Factor: {m.recovery_factor:.2f}")
    print()
    print(f"  Trades/Day:      {m.trades_per_day:.1f}")
    print(f"  Avg Hold:        {m.avg_hold_duration_s / 3600:.1f}h")
    print(f"  Win Streak:      {m.longest_win_streak}")
    print(f"  Loss Streak:     {m.longest_loss_streak}")
    print(f"  Total Fees:      ${m.total_fees:.2f} ({m.fee_drag_pct:.1f}% drag)")
    print(f"{'=' * 60}\n")

    if m.by_symbol:
        print("  By Symbol:")
        for sym, data in sorted(m.by_symbol.items()):
            pnl = data["total_pnl"]
            print(f"    {sym:12s}  {data['trades']:3d} trades  "
                  f"WR={data['win_rate']:.0%}  PnL=${pnl:+.3f}  PF={data['profit_factor']:.2f}")


def _print_optimization_result(result):
    """Print optimization results."""
    print(f"\n{'=' * 60}")
    print(f"  OPTIMIZATION RESULTS — {result.param_name}")
    print(f"{'=' * 60}")
    print(f"  IS period:  {result.in_sample_period}")
    print(f"  OOS period: {result.out_of_sample_period}")
    print()

    print(f"  {'Value':>10s}  {'IS Score':>10s}  {'OOS Score':>10s}  "
          f"{'Degrad%':>8s}  {'IS WR':>6s}  {'OOS WR':>6s}  {'IS PF':>6s}  {'OOS PF':>6s}  Overfit?")
    print("  " + "-" * 90)

    for run in result.runs:
        val = list(run.params.values())[0]
        is_m = run.in_sample
        oos_m = run.out_of_sample
        flag = " OVERFIT" if run.is_overfit else ""

        print(
            f"  {val:>10}  {run.is_score:>10.2f}  {run.oos_score:>10.2f}  "
            f"{run.degradation_pct:>7.1f}%  "
            f"{is_m.win_rate if is_m else 0:>5.0%}  "
            f"{oos_m.win_rate if oos_m else 0:>5.0%}  "
            f"{is_m.profit_factor if is_m else 0:>5.2f}  "
            f"{oos_m.profit_factor if oos_m else 0:>5.2f}  "
            f"{flag}"
        )

    print()
    if result.best_params:
        print(f"  Best: {result.best_params}")
        print(f"  IS Score: {result.best_is_score:.2f}, OOS Score: {result.best_oos_score:.2f}")
    if result.overfit_warning:
        print(f"  WARNING: Potential overfitting detected!")
    print()


def _print_sync_status(dm: DataManager):
    """Print data sync status."""
    status = dm.get_sync_status()
    if not status:
        print("\nNo data synced yet. Run: python -m backtester sync-data --days 90")
        return

    print(f"\n{'Symbol':12s}  {'TF':5s}  {'Candles':>8s}  {'First':12s}  {'Last':12s}  {'Synced':20s}")
    print("-" * 75)
    for s in status:
        first = (datetime.fromtimestamp(s.first_timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                 if s.first_timestamp else "—")
        last = (datetime.fromtimestamp(s.last_timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                if s.last_timestamp else "—")
        synced = (datetime.fromtimestamp(s.last_sync, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                  if s.last_sync else "—")
        print(f"{s.symbol:12s}  {s.timeframe:5s}  {s.total_candles:>8d}  {first:12s}  {last:12s}  {synced:20s}")
    print()


# ── Result persistence ──────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def _save_result(result: BacktestResult):
    """Save BacktestResult metadata for later report generation."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, f"result_{result.run_id}.json")

    data = {
        "run_id": result.run_id,
        "start_time": result.start_time,
        "end_time": result.end_time,
        "param_overrides": result.param_overrides,
        "symbols": result.symbols,
        "starting_equity": result.starting_equity,
        "trades": result.trades,
        "equity_curve": result.equity_curve,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, default=str)

    logger.debug(f"Result saved to {filepath}")


def _load_result(run_id: str) -> BacktestResult:
    """Load a saved BacktestResult."""
    filepath = os.path.join(RESULTS_DIR, f"result_{run_id}.json")
    if not os.path.exists(filepath):
        # Try partial match
        for fname in os.listdir(RESULTS_DIR):
            if run_id in fname and fname.endswith(".json"):
                filepath = os.path.join(RESULTS_DIR, fname)
                break
        else:
            return None

    with open(filepath) as f:
        data = json.load(f)

    from backtester.metrics import compute_metrics
    metrics = compute_metrics(
        data["trades"],
        [tuple(p) for p in data["equity_curve"]],
        data["starting_equity"],
    )

    return BacktestResult(
        run_id=data["run_id"],
        start_time=data["start_time"],
        end_time=data["end_time"],
        config_snapshot={},
        param_overrides=data.get("param_overrides", {}),
        trades=data["trades"],
        equity_curve=[tuple(p) for p in data["equity_curve"]],
        metrics=metrics,
        symbols=data["symbols"],
        starting_equity=data["starting_equity"],
    )


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="backtester",
        description="Backtesting framework for the Binance USDT-M Futures Swing Trading Bot",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # sync-data
    sp = subparsers.add_parser("sync-data", help="Download historical candle data from Binance")
    sp.add_argument("--days", type=int, default=90, help="Number of days to download (default: 90)")
    sp.add_argument("--symbols", type=str, default=None,
                    help="Comma-separated symbols (default: config pairs)")
    sp.add_argument("--timeframes", type=str, default=None,
                    help="Comma-separated timeframes (default: 15m,1h,4h,1d)")
    sp.add_argument("--rate-limit", type=float, default=0.35,
                    help="Delay between API calls in seconds (default: 0.35)")
    sp.add_argument("-v", "--verbose", action="store_true")

    # run
    sp = subparsers.add_parser("run", help="Run a backtest")
    sp.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    sp.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    sp.add_argument("--symbols", type=str, default=None,
                    help="Comma-separated symbols (default: config pairs)")
    sp.add_argument("--timeframe", type=str, default="1h",
                    help="Primary timeframe for stepping (default: 1h)")
    sp.add_argument("--no-report", action="store_true",
                    help="Skip auto-generating HTML report")
    sp.add_argument("-v", "--verbose", action="store_true")

    # optimize
    sp = subparsers.add_parser("optimize", help="Walk-forward parameter optimization")
    sp.add_argument("--param", type=str, required=True,
                    help="Parameter to optimize (use 'list-params' to see options)")
    sp.add_argument("--range", type=str, required=True,
                    help="Value range: 'start,end,step' or 'val1,val2,val3'")
    sp.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    sp.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    sp.add_argument("--symbols", type=str, default=None,
                    help="Comma-separated symbols (default: config pairs)")
    sp.add_argument("-v", "--verbose", action="store_true")

    # report
    sp = subparsers.add_parser("report", help="Generate HTML report from saved run")
    sp.add_argument("--run-id", type=str, required=True, help="Run ID to generate report for")
    sp.add_argument("-v", "--verbose", action="store_true")

    # status
    sp = subparsers.add_parser("status", help="Show data sync status")
    sp.add_argument("-v", "--verbose", action="store_true")

    # list-params
    sp = subparsers.add_parser("list-params", help="List optimizable parameters")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(getattr(args, "verbose", False))

    commands = {
        "sync-data": cmd_sync_data,
        "run": cmd_run,
        "optimize": cmd_optimize,
        "report": cmd_report,
        "status": cmd_status,
        "list-params": cmd_list_params,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
