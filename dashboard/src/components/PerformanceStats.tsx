'use client';

import { usePerformance, useStatus } from '@/lib/hooks';
import { Trophy, Target, BarChart3, Clock } from 'lucide-react';

function StatBox({ label, value, sub, icon: Icon, color }: {
  label: string; value: string; sub?: string;
  icon: any; color: string;
}) {
  return (
    <div className="bg-bg-secondary/50 rounded-lg p-3 flex items-start gap-3">
      <div className={`p-2 rounded-lg ${color}`}>
        <Icon className="w-4 h-4" />
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-wider text-text-secondary font-bold">
          {label}
        </div>
        <div className="text-lg font-bold text-text-primary">{value}</div>
        {sub && <div className="text-[10px] text-text-secondary">{sub}</div>}
      </div>
    </div>
  );
}

export function PerformanceStats() {
  const { data: perf } = usePerformance();
  const { data: status } = useStatus();

  const totalTrades = perf?.total_trades || status?.total_trades || 0;
  const winRate = perf?.win_rate || (status?.win_rate) || 0;
  const profitFactor = perf?.profit_factor || 0;
  const avgDuration = perf?.avg_duration_s || 0;
  const totalPnl = perf?.total_pnl_usd || 0;
  const leverage = status?.leverage || 30;

  const formatDuration = (s: number) => {
    if (s < 60) return `${s.toFixed(0)}s`;
    if (s < 3600) return `${(s / 60).toFixed(0)}m`;
    return `${(s / 3600).toFixed(1)}h`;
  };

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5">
      <h2 className="text-xs font-bold tracking-wider text-text-secondary uppercase mb-4">
        Performance
      </h2>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatBox
          label="Win Rate"
          value={`${winRate.toFixed(1)}%`}
          sub={`${perf?.wins || 0}W / ${perf?.losses || 0}L`}
          icon={Trophy}
          color="bg-accent-green/10 text-accent-green"
        />
        <StatBox
          label="Total P&L"
          value={`${totalPnl >= 0 ? '+' : ''}$${totalPnl.toFixed(2)}`}
          sub={`${leverage}x leverage`}
          icon={Target}
          color={totalPnl >= 0 ? "bg-accent-green/10 text-accent-green" : "bg-accent-red/10 text-accent-red"}
        />
        <StatBox
          label="Profit Factor"
          value={profitFactor === Infinity ? "∞" : profitFactor.toFixed(2)}
          sub={`${totalTrades} trades`}
          icon={BarChart3}
          color="bg-accent-blue/10 text-accent-blue"
        />
        <StatBox
          label="Avg Hold Time"
          value={formatDuration(avgDuration)}
          sub="per trade"
          icon={Clock}
          color="bg-accent-purple/10 text-accent-purple"
        />
      </div>
    </div>
  );
}
