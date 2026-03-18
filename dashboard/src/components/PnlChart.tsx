'use client';

import { usePnlChart } from '@/lib/hooks';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';

export function PnlChart() {
  const { data } = usePnlChart();

  const chartData = (data?.data || []).map((d: any) => ({
    ...d,
    time: new Date(d.timestamp * 1000).toLocaleString('en-US', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    }),
    pnl: d.cumulative_pnl,
  }));

  const hasData = chartData.length > 0;
  const latestPnl = hasData ? chartData[chartData.length - 1].pnl : 0;
  const isPositive = latestPnl >= 0;

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs font-bold tracking-wider text-text-secondary uppercase">
          Cumulative P&L
        </h2>
        {hasData && (
          <span className={`text-sm font-bold ${isPositive ? 'text-accent-green' : 'text-accent-red'}`}>
            {isPositive ? '+' : ''}${latestPnl.toFixed(2)}
          </span>
        )}
      </div>

      <div className="h-64">
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={isPositive ? '#10b981' : '#ef4444'} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={isPositive ? '#10b981' : '#ef4444'} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3042" />
              <XAxis
                dataKey="time"
                tick={{ fill: '#6b7280', fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: '#2a3042' }}
              />
              <YAxis
                tick={{ fill: '#6b7280', fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: '#2a3042' }}
                tickFormatter={(v) => `$${v}`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1a1f2e',
                  border: '1px solid #2a3042',
                  borderRadius: '8px',
                  color: '#e5e7eb',
                  fontSize: '12px',
                }}
                formatter={(value: number) => [`$${value.toFixed(2)}`, 'P&L']}
              />
              <ReferenceLine y={0} stroke="#4b5563" strokeDasharray="3 3" />
              <Area
                type="monotone"
                dataKey="pnl"
                stroke={isPositive ? '#10b981' : '#ef4444'}
                strokeWidth={2}
                fill="url(#pnlGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center text-text-secondary text-sm">
            No trade data yet. Waiting for first closed trade...
          </div>
        )}
      </div>
    </div>
  );
}
