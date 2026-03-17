#!/usr/bin/env python3
"""
Shadow Trade Performance Analyzer

Reads shadow_trades.jsonl and shadow_trades_signals.jsonl to produce
a comprehensive post-mortem of your bot's performance.

Includes:
- Net P&L after Binance fees (maker/taker)
- Per-symbol breakdown
- Win/loss analysis by exit reason
- VPIN regime analysis (did high VPIN trades do worse?)
- Hold time distribution
- Signal quality metrics
- Hourly activity heatmap
- Compounding simulation

Usage:
    python analyze_performance.py
    python analyze_performance.py --fee 0.075   # BNB discount
    python analyze_performance.py --capital 1000000 --currency IDR
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional


def load_trades(path: str = "shadow_trades.jsonl") -> List[dict]:
    """Load closed trades from JSONL log."""
    trades = []
    if not os.path.exists(path):
        print(f"  No trade log found at: {path}")
        return trades

    with open(path) as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if record.get("event") == "CLOSE":
                    trades.append(record["trade"])
            except (json.JSONDecodeError, KeyError):
                continue

    return trades


def load_signals(path: str = "shadow_trades_signals.jsonl") -> List[dict]:
    """Load signal log."""
    signals = []
    if not os.path.exists(path):
        return signals

    with open(path) as f:
        for line in f:
            try:
                signals.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    return signals


def analyze(trades: List[dict], signals: List[dict], fee_pct: float = 0.1,
            capital: float = 10000.0, currency: str = "USD"):
    """Run the full analysis."""

    if not trades:
        print("\n  No closed trades to analyze.")
        print("  Run the bot in shadow mode first, then re-run this script.\n")
        return

    # ───────────────────────────────────────
    # 1. BASIC STATS
    # ───────────────────────────────────────

    n = len(trades)
    wins = [t for t in trades if (t.get("pnl_usd") or 0) > 0]
    losses = [t for t in trades if (t.get("pnl_usd") or 0) <= 0]
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n * 100 if n > 0 else 0

    gross_pnl = sum(t.get("pnl_usd", 0) or 0 for t in trades)

    # Fee calculation: each trade = 1 buy + 1 sell = 2 * fee_pct% * trade_value
    total_volume = sum(t.get("usd_value", 0) or 0 for t in trades) * 2  # round trip
    total_fees = total_volume * (fee_pct / 100.0)
    net_pnl = gross_pnl - total_fees

    avg_win = sum(t.get("pnl_usd", 0) or 0 for t in wins) / n_wins if n_wins else 0
    avg_loss = sum(t.get("pnl_usd", 0) or 0 for t in losses) / n_losses if n_losses else 0

    max_win = max((t.get("pnl_usd", 0) or 0 for t in trades), default=0)
    max_loss = min((t.get("pnl_usd", 0) or 0 for t in trades), default=0)

    gross_profit = sum(t.get("pnl_usd", 0) or 0 for t in wins)
    gross_loss_val = abs(sum(t.get("pnl_usd", 0) or 0 for t in losses))
    profit_factor = gross_profit / gross_loss_val if gross_loss_val > 0 else float("inf")

    durations = [t.get("duration_s", 0) or 0 for t in trades if (t.get("duration_s") or 0) > 0]
    avg_duration = sum(durations) / len(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    min_duration = min(durations) if durations else 0

    # Expectancy = (win_rate * avg_win) - (loss_rate * |avg_loss|)
    expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * abs(avg_loss))
    expectancy_after_fees = expectancy - (fee_pct / 100 * 2 * (capital * 0.02))  # approx

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║          SHADOW TRADE PERFORMANCE ANALYSIS                ║
╠═══════════════════════════════════════════════════════════╣

  Total Closed Trades:    {n}
  Win / Loss:             {n_wins}W / {n_losses}L
  Win Rate:               {win_rate:.1f}%

  ── P&L ──────────────────────────────────────
  Gross P&L (no fees):    ${gross_pnl:+,.2f}
  Total Trading Volume:   ${total_volume:,.2f}
  Binance Fees ({fee_pct}%):    -${total_fees:,.2f}
  ═══════════════════════════════════════════
  NET P&L (after fees):   ${net_pnl:+,.2f}
  ═══════════════════════════════════════════

  Avg Win:                ${avg_win:+,.2f}
  Avg Loss:               ${avg_loss:,.2f}
  Max Win:                ${max_win:+,.2f}
  Max Loss:               ${max_loss:,.2f}
  Profit Factor:          {profit_factor:.2f}
  Expectancy/Trade:       ${expectancy:+,.4f}

  ── Hold Time ────────────────────────────────
  Average:                {avg_duration:.1f}s ({avg_duration/60:.1f}min)
  Shortest:               {min_duration:.1f}s
  Longest:                {max_duration:.1f}s ({max_duration/60:.1f}min)""")

    # ───────────────────────────────────────
    # 2. PER-SYMBOL BREAKDOWN
    # ───────────────────────────────────────

    by_symbol: Dict[str, list] = defaultdict(list)
    for t in trades:
        by_symbol[t["symbol"]].append(t)

    print(f"""
  ── Per Symbol ───────────────────────────────""")
    print(f"  {'Symbol':<14} {'Trades':>6} {'WR':>6} {'Gross':>10} {'Fees':>8} {'Net':>10} {'Avg Dur':>8}")
    print(f"  {'─'*14} {'─'*6} {'─'*6} {'─'*10} {'─'*8} {'─'*10} {'─'*8}")

    for sym, sym_trades in sorted(by_symbol.items()):
        sn = len(sym_trades)
        sw = sum(1 for t in sym_trades if (t.get("pnl_usd") or 0) > 0)
        swr = sw / sn * 100 if sn > 0 else 0
        sg = sum(t.get("pnl_usd", 0) or 0 for t in sym_trades)
        svol = sum(t.get("usd_value", 0) or 0 for t in sym_trades) * 2
        sf = svol * (fee_pct / 100)
        snet = sg - sf
        sdurs = [t.get("duration_s", 0) or 0 for t in sym_trades if (t.get("duration_s") or 0) > 0]
        savg = sum(sdurs) / len(sdurs) if sdurs else 0
        print(f"  {sym:<14} {sn:>6} {swr:>5.1f}% ${sg:>+8.2f} ${sf:>6.2f} ${snet:>+8.2f} {savg:>6.0f}s")

    # ───────────────────────────────────────
    # 3. EXIT REASON ANALYSIS
    # ───────────────────────────────────────

    by_reason: Dict[str, list] = defaultdict(list)
    for t in trades:
        by_reason[t.get("exit_reason", "unknown")].append(t)

    print(f"""
  ── By Exit Reason ───────────────────────────""")
    print(f"  {'Reason':<16} {'Count':>6} {'Avg PnL':>10} {'WR':>6}")
    print(f"  {'─'*16} {'─'*6} {'─'*10} {'─'*6}")

    for reason, rtrades in sorted(by_reason.items()):
        rc = len(rtrades)
        ravg = sum(t.get("pnl_usd", 0) or 0 for t in rtrades) / rc if rc else 0
        rw = sum(1 for t in rtrades if (t.get("pnl_usd") or 0) > 0)
        rwr = rw / rc * 100 if rc > 0 else 0
        print(f"  {reason:<16} {rc:>6} ${ravg:>+8.2f} {rwr:>5.1f}%")

    # ───────────────────────────────────────
    # 4. VPIN REGIME ANALYSIS
    # ───────────────────────────────────────

    print(f"""
  ── VPIN @ Entry Analysis ────────────────────
  (Did VPIN predict trade outcomes?)
""")

    low_vpin = [t for t in trades if (t.get("vpin_ema") or t.get("vpin") or 0) < 0.4]
    mid_vpin = [t for t in trades if 0.4 <= (t.get("vpin_ema") or t.get("vpin") or 0) < 0.7]
    high_vpin = [t for t in trades if (t.get("vpin_ema") or t.get("vpin") or 0) >= 0.7]

    for label, group in [("Low VPIN (<0.4)", low_vpin), ("Mid VPIN (0.4-0.7)", mid_vpin), ("High VPIN (>0.7)", high_vpin)]:
        if group:
            gn = len(group)
            gpnl = sum(t.get("pnl_usd", 0) or 0 for t in group)
            gwr = sum(1 for t in group if (t.get("pnl_usd") or 0) > 0) / gn * 100
            print(f"  {label:<22} {gn:>3} trades | PnL=${gpnl:>+7.2f} | WR={gwr:.0f}%")
        else:
            print(f"  {label:<22}   0 trades")

    # ───────────────────────────────────────
    # 5. COMPOSITE SCORE ANALYSIS
    # ───────────────────────────────────────

    print(f"""
  ── Composite Score @ Entry ──────────────────""")

    scores = [(t.get("composite_score", 0), t.get("pnl_usd", 0) or 0) for t in trades]
    strong = [(s, p) for s, p in scores if abs(s) >= 0.5]
    medium = [(s, p) for s, p in scores if 0.3 <= abs(s) < 0.5]
    weak = [(s, p) for s, p in scores if abs(s) < 0.3]

    for label, group in [("Strong (|score|>=0.5)", strong), ("Medium (0.3-0.5)", medium), ("Weak (<0.3)", weak)]:
        if group:
            gn = len(group)
            gpnl = sum(p for _, p in group)
            gwr = sum(1 for _, p in group if p > 0) / gn * 100
            print(f"  {label:<24} {gn:>3} trades | PnL=${gpnl:>+7.2f} | WR={gwr:.0f}%")
        else:
            print(f"  {label:<24}   0 trades")

    # ───────────────────────────────────────
    # 6. COMPOUNDING SIMULATION
    # ───────────────────────────────────────

    print(f"""
  ── Compounding Simulation ({currency}) ──────""")

    balance = capital
    peak = capital
    max_dd = 0
    equity_curve = [balance]

    for t in trades:
        pnl_pct = (t.get("pnl_pct") or 0) / 100.0
        # Subtract fees from each trade's return
        fee_drag = (fee_pct / 100.0) * 2  # round trip
        net_return = pnl_pct - fee_drag
        balance *= (1 + net_return)
        equity_curve.append(balance)
        peak = max(peak, balance)
        dd = (peak - balance) / peak * 100
        max_dd = max(max_dd, dd)

    total_return = (balance - capital) / capital * 100

    print(f"  Starting Capital:     {currency} {capital:,.0f}")
    print(f"  Final Balance:        {currency} {balance:,.0f}")
    print(f"  Total Return:         {total_return:+.2f}%")
    print(f"  Max Drawdown:         {max_dd:.2f}%")
    print(f"  Peak Balance:         {currency} {peak:,.0f}")

    # ───────────────────────────────────────
    # 7. SIGNAL LOG STATS (if available)
    # ───────────────────────────────────────

    if signals:
        total_signals = len(signals)
        buy_signals = sum(1 for s in signals if s.get("suggestion") == "BUY")
        sell_signals = sum(1 for s in signals if s.get("suggestion") == "SELL")
        hold_signals = total_signals - buy_signals - sell_signals
        blocked_by_vpin = sum(1 for s in signals if s.get("vpin_blocked_entry") is True)

        print(f"""
  ── Signal Quality ───────────────────────────
  Total OB Snapshots:     {total_signals:,}
  BUY Signals:            {buy_signals:,} ({buy_signals/total_signals*100:.2f}%)
  SELL Signals:           {sell_signals:,} ({sell_signals/total_signals*100:.2f}%)
  HOLD (no signal):       {hold_signals:,} ({hold_signals/total_signals*100:.2f}%)
  Blocked by VPIN:        {blocked_by_vpin:,} ({blocked_by_vpin/total_signals*100:.2f}%)
  Signal-to-Trade Ratio:  {total_signals/(n if n else 1):.0f}:1 (snapshots per trade)""")

        # Spread stats
        spreads = [s.get("spread_pct", 0) for s in signals if s.get("spread_pct")]
        if spreads:
            avg_spread = sum(spreads) / len(spreads)
            print(f"  Avg Spread:             {avg_spread:.4f}%")

        # VPIN distribution across all signals
        vpins = [s.get("vpin_ema", s.get("vpin", 0)) for s in signals]
        if vpins:
            vpin_above_block = sum(1 for v in vpins if v >= 0.8)
            print(f"  VPIN >= 0.8 (blocked):  {vpin_above_block/len(vpins)*100:.1f}% of all snapshots")

    # ───────────────────────────────────────
    # 8. KEY PROBLEMS DETECTED
    # ───────────────────────────────────────

    print(f"""
  ── Diagnosis ────────────────────────────────""")

    problems = []

    if avg_duration < 120:
        problems.append(
            f"  - Avg hold time is only {avg_duration:.0f}s. Trades are exiting too fast,\n"
            f"    mostly from wall_pulled. The bot is scalping at a timeframe\n"
            f"    where walls are unreliable anchors."
        )

    wall_pulled_pct = len(by_reason.get("wall_pulled", [])) / n * 100 if n else 0
    if wall_pulled_pct > 50:
        problems.append(
            f"  - {wall_pulled_pct:.0f}% of exits are 'wall_pulled'. Walls are being\n"
            f"    removed/eaten before price reaches TP. Consider using walls as\n"
            f"    entry confirmation rather than anchoring orders to them."
        )

    if total_fees > abs(gross_pnl):
        problems.append(
            f"  - Fees (${total_fees:.2f}) exceed gross PnL (${gross_pnl:.2f}).\n"
            f"    The strategy needs wider targets to overcome fee drag."
        )

    if high_vpin and sum(1 for t in high_vpin if (t.get("pnl_usd") or 0) < 0) > len(high_vpin) * 0.6:
        problems.append(
            f"  - High VPIN trades have poor outcomes. The VPIN gate isn't\n"
            f"    aggressive enough at blocking entries in toxic flow."
        )

    if signals:
        vpin_block_rate = blocked_by_vpin / total_signals * 100 if total_signals else 0
        if vpin_block_rate > 50:
            problems.append(
                f"  - VPIN blocks entries {vpin_block_rate:.0f}% of the time. The thresholds\n"
                f"    are too tight for this market. Consider raising vpin_block_entry_above."
            )

    stablecoin_trades = [t for t in trades if any(s in t["symbol"] for s in ["FDUSD", "USDC", "TUSD", "DAI", "BUSD"])]
    if stablecoin_trades:
        problems.append(
            f"  - {len(stablecoin_trades)} trades on stablecoins (FDUSD/USDC). These pairs\n"
            f"    have no directional opportunity. Exclude them from pair selection."
        )

    if problems:
        for p in problems:
            print(p)
    else:
        print("  No critical problems detected.")

    print(f"""
╚═══════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="Analyze shadow trading performance")
    parser.add_argument("--trades", default="shadow_trades.jsonl", help="Trade log file")
    parser.add_argument("--signals", default="shadow_trades_signals.jsonl", help="Signal log file")
    parser.add_argument("--fee", type=float, default=0.1, help="Binance fee %% per side (default: 0.1)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Starting capital for compounding sim")
    parser.add_argument("--currency", default="USD", help="Currency label (e.g., USD, IDR)")
    args = parser.parse_args()

    trades = load_trades(args.trades)
    signals = load_signals(args.signals)
    analyze(trades, signals, fee_pct=args.fee, capital=args.capital, currency=args.currency)


if __name__ == "__main__":
    main()
