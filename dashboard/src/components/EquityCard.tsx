'use client';

import { useStatus } from '@/lib/hooks';
import { TrendingUp, TrendingDown, DollarSign } from 'lucide-react';

export function EquityCard() {
  const { data } = useStatus();

  const equity = data?.equity_usd || data?.capital_usd || 0;
  const capitalIDR = data?.capital_idr || 1_000_000;
  const dailyPnl = data?.daily_pnl_usd || 0;
  const drawdown = data?.drawdown_pct || 0;
  const targetPct = data?.daily_target_pct || 10;
  const dailyPnlPct = equity > 0 ? (dailyPnl / equity) * 100 : 0;
  const progressPct = Math.min((dailyPnlPct / targetPct) * 100, 100);
  const isProfit = dailyPnl >= 0;

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs font-bold tracking-wider text-text-secondary uppercase">
          Account Equity
        </h2>
        <DollarSign className="w-4 h-4 text-accent-blue" />
      </div>

      {/* Main equity display */}
      <div className="mb-4">
        <div className="text-3xl font-bold text-text-primary">
          ${equity.toFixed(2)}
        </div>
        <div className="text-sm text-text-secondary mt-0.5">
          IDR {capitalIDR.toLocaleString()}
        </div>
      </div>

      {/* Daily P&L */}
      <div className="flex items-center gap-2 mb-4">
        {isProfit ? (
          <TrendingUp className="w-4 h-4 text-accent-green" />
        ) : (
          <TrendingDown className="w-4 h-4 text-accent-red" />
        )}
        <span className={`text-lg font-bold ${isProfit ? 'text-accent-green glow-green' : 'text-accent-red glow-red'}`}>
          {isProfit ? '+' : ''}{dailyPnl.toFixed(2)} USD
        </span>
        <span className={`text-sm ${isProfit ? 'text-accent-green' : 'text-accent-red'}`}>
          ({isProfit ? '+' : ''}{dailyPnlPct.toFixed(2)}%)
        </span>
      </div>

      {/* Daily target progress bar */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-xs text-text-secondary">
          <span>Daily Target: {targetPct}%</span>
          <span>{progressPct.toFixed(1)}%</span>
        </div>
        <div className="w-full h-2 bg-bg-secondary rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              progressPct >= 100 ? 'bg-accent-green' :
              progressPct >= 50 ? 'bg-accent-blue' :
              'bg-accent-yellow'
            }`}
            style={{ width: `${Math.max(progressPct, 0)}%` }}
          />
        </div>
      </div>

      {/* Drawdown */}
      {drawdown > 0 && (
        <div className="mt-3 flex items-center gap-1 text-xs text-accent-red">
          <span>Drawdown: {drawdown.toFixed(2)}%</span>
        </div>
      )}
    </div>
  );
}
