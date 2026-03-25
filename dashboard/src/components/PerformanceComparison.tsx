'use client';

import { useAdvancedPerformance, useDailyEquity } from '@/lib/hooks';
import { TrendingUp } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
  Area, ComposedChart,
} from 'recharts';

function StatCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="bg-bg-secondary/40 rounded-lg p-3 text-center">
      <div className="text-[9px] text-text-secondary uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-lg font-bold ${color || 'text-text-primary'}`}>{value}</div>
      {sub && <div className="text-[9px] text-text-secondary mt-0.5">{sub}</div>}
    </div>
  );
}

function CumulativePnlChart({ gross, net }: { gross: any[]; net: any[] }) {
  if (!gross.length) return null;

  const merged = gross.map((g, i) => ({
    time: new Date((g.timestamp || 0) * 1000).toLocaleDateString([], { month: 'short', day: 'numeric' }),
    gross: g.value,
    net: net[i]?.value ?? g.value,
    feeImpact: Math.abs(g.value - (net[i]?.value ?? g.value)),
  }));

  // Downsample if too many points
  const data = merged.length > 100
    ? merged.filter((_, i) => i % Math.ceil(merged.length / 100) === 0 || i === merged.length - 1)
    : merged;

  return (
    <div className="bg-bg-secondary/30 rounded-lg p-4">
      <div className="text-[10px] text-text-secondary uppercase tracking-wider mb-3">
        Cumulative P&L — Gross vs Net (Fee Impact)
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
          <XAxis
            dataKey="time"
            tick={{ fill: '#9ca3af', fontSize: 9 }}
            axisLine={false}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: '#9ca3af', fontSize: 9 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `$${v.toFixed(2)}`}
          />
          <Tooltip
            contentStyle={{
              background: '#1a1f2e',
              border: '1px solid #2a3042',
              borderRadius: 8,
              fontSize: 11,
            }}
            formatter={(v: number, name: string) => [
              `$${v.toFixed(4)}`,
              name === 'gross' ? 'Gross P&L' : name === 'net' ? 'Net P&L' : 'Fee Impact',
            ]}
          />
          <Legend
            wrapperStyle={{ fontSize: 10, paddingTop: 8 }}
            formatter={(val: string) =>
              val === 'gross' ? 'Gross P&L' : val === 'net' ? 'Net P&L' : 'Fee Impact'
            }
          />
          <Area
            type="monotone"
            dataKey="feeImpact"
            fill="#f59e0b"
            fillOpacity={0.1}
            stroke="none"
          />
          <Line
            type="monotone"
            dataKey="gross"
            stroke="#10b981"
            strokeWidth={2}
            dot={false}
            strokeOpacity={0.5}
            strokeDasharray="4 2"
          />
          <Line
            type="monotone"
            dataKey="net"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function WinRateGauge({ label, pct }: { label: string; pct: number }) {
  const color = pct >= 55 ? 'text-accent-green' : pct >= 45 ? 'text-accent-yellow' : 'text-accent-red';
  const barColor = pct >= 55 ? 'bg-accent-green' : pct >= 45 ? 'bg-accent-yellow' : 'bg-accent-red';
  return (
    <div className="bg-bg-secondary/40 rounded-lg p-3">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[9px] text-text-secondary uppercase tracking-wider">{label}</span>
        <span className={`text-sm font-bold ${color}`}>{pct.toFixed(1)}%</span>
      </div>
      <div className="w-full h-1.5 bg-bg-primary rounded-full overflow-hidden">
        <div
          className={`h-full ${barColor} rounded-full transition-all duration-500`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
    </div>
  );
}

export function PerformanceComparison() {
  const { data, isLoading } = useAdvancedPerformance();

  if (isLoading) {
    return (
      <div className="bg-bg-card rounded-xl border border-border-dim p-5">
        <div className="h-80 shimmer rounded-lg" />
      </div>
    );
  }

  const sharpe = data?.sharpe ?? 0;
  const sortino = data?.sortino ?? 0;
  const maxDD = data?.max_drawdown_pct ?? 0;
  const wr7 = data?.rolling_7d_wr ?? 0;
  const wr30 = data?.rolling_30d_wr ?? 0;
  const totalTrades = data?.total_trades ?? 0;
  const avgHold = data?.avg_hold_s ?? 0;

  const formatHold = (s: number) => {
    if (!s || s <= 0) return '-';
    if (s >= 3600) return `${(s / 3600).toFixed(1)}h`;
    if (s >= 60) return `${Math.round(s / 60)}m`;
    return `${Math.round(s)}s`;
  };

  const sharpeColor = sharpe > 1 ? 'text-accent-green' : sharpe > 0 ? 'text-accent-yellow' : 'text-accent-red';
  const sortinoColor = sortino > 1.5 ? 'text-accent-green' : sortino > 0 ? 'text-accent-yellow' : 'text-accent-red';
  const ddColor = maxDD < 5 ? 'text-accent-green' : maxDD < 15 ? 'text-accent-yellow' : 'text-accent-red';

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5 fade-in space-y-4">
      <div className="flex items-center gap-2">
        <TrendingUp className="w-4 h-4 text-accent-green" />
        <h2 className="text-xs font-bold tracking-wider text-text-secondary uppercase">
          Performance Comparison
        </h2>
        <span className="text-[10px] text-text-secondary">({totalTrades} trades)</span>
      </div>

      {/* Key metrics row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <StatCard label="Sharpe Ratio" value={sharpe.toFixed(2)} color={sharpeColor} />
        <StatCard label="Sortino Ratio" value={sortino.toFixed(2)} color={sortinoColor} />
        <StatCard label="Max Drawdown" value={`${maxDD.toFixed(1)}%`} color={ddColor} />
        <StatCard label="Total Trades" value={String(totalTrades)} />
        <StatCard label="Avg Hold" value={formatHold(avgHold)} />
      </div>

      {/* Rolling win rates */}
      <div className="grid grid-cols-2 gap-3">
        <WinRateGauge label="Rolling 7-Trade Win Rate" pct={wr7} />
        <WinRateGauge label="Rolling 30-Trade Win Rate" pct={wr30} />
      </div>

      {/* Cumulative P&L chart */}
      <CumulativePnlChart
        gross={data?.cumulative_gross || []}
        net={data?.cumulative_net || []}
      />
    </div>
  );
}
