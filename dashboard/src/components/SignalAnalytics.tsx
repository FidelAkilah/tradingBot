'use client';

import { useWinrateAnalytics } from '@/lib/hooks';
import { BarChart3 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

function BreakdownTable({
  title,
  data,
  keyField,
}: {
  title: string;
  data: any[];
  keyField: string;
}) {
  if (!data || data.length === 0) return null;
  return (
    <div>
      <h3 className="text-[10px] text-text-secondary uppercase tracking-wider mb-2">{title}</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-text-secondary border-b border-border-dim">
              <th className="text-left py-1.5 px-2">{keyField}</th>
              <th className="text-right py-1.5 px-2">Trades</th>
              <th className="text-right py-1.5 px-2">WR</th>
              <th className="text-right py-1.5 px-2">P&L</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row: any, i: number) => {
              const key = row[keyField] || row.bucket || row.symbol || row.side || row.regime || row.session || row.exit_reason || row.dow || 'unknown';
              const trades = row.trades || 0;
              const wins = row.wins || 0;
              const wr = trades > 0 ? (wins / trades) * 100 : 0;
              const pnl = row.total_pnl || 0;
              return (
                <tr key={i} className="border-b border-border-dim/30 hover:bg-bg-secondary/30">
                  <td className="py-1.5 px-2 font-medium text-text-primary">{key}</td>
                  <td className="py-1.5 px-2 text-right text-text-secondary">{trades}</td>
                  <td className={`py-1.5 px-2 text-right font-medium ${wr >= 50 ? 'text-accent-green' : 'text-accent-red'}`}>
                    {wr.toFixed(0)}%
                  </td>
                  <td className={`py-1.5 px-2 text-right font-medium ${pnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                    ${pnl.toFixed(3)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function WinRateBarChart({ data, keyField }: { data: any[]; keyField: string }) {
  if (!data || data.length === 0) return null;

  const chartData = data.map((d: any) => {
    const key = d[keyField] || d.bucket || d.symbol || d.side || 'unknown';
    const trades = d.trades || 0;
    const wins = d.wins || 0;
    return {
      name: key.replace('/USDT', ''),
      winRate: trades > 0 ? (wins / trades) * 100 : 0,
      trades,
    };
  });

  return (
    <ResponsiveContainer width="100%" height={140}>
      <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
        <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} axisLine={false} tickLine={false} domain={[0, 100]} />
        <Tooltip
          contentStyle={{ background: '#1a1f2e', border: '1px solid #2a3042', borderRadius: 8, fontSize: 11 }}
          formatter={(v: number) => [`${v.toFixed(0)}%`, 'Win Rate']}
        />
        <Bar dataKey="winRate" radius={[4, 4, 0, 0]}>
          {chartData.map((entry, i) => (
            <Cell key={i} fill={entry.winRate >= 50 ? '#10b981' : '#ef4444'} fillOpacity={0.8} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function SignalAnalytics() {
  const { data, isLoading } = useWinrateAnalytics();

  if (isLoading) {
    return (
      <div className="bg-bg-card rounded-xl border border-border-dim p-5">
        <div className="h-96 shimmer rounded-lg" />
      </div>
    );
  }

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5 fade-in space-y-6">
      <div className="flex items-center gap-2">
        <BarChart3 className="w-4 h-4 text-accent-blue" />
        <h2 className="text-xs font-bold tracking-wider text-text-secondary uppercase">
          Signal Quality Analytics
        </h2>
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-bg-secondary/30 rounded-lg p-3">
          <div className="text-[10px] text-text-secondary uppercase mb-2">Win Rate by Confidence</div>
          <WinRateBarChart data={data?.by_confidence || []} keyField="bucket" />
        </div>
        <div className="bg-bg-secondary/30 rounded-lg p-3">
          <div className="text-[10px] text-text-secondary uppercase mb-2">Win Rate by Symbol</div>
          <WinRateBarChart data={data?.by_symbol || []} keyField="symbol" />
        </div>
      </div>

      {/* Breakdown tables */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <BreakdownTable title="By Confidence" data={data?.by_confidence || []} keyField="bucket" />
        <BreakdownTable title="By Symbol" data={data?.by_symbol || []} keyField="symbol" />
        <BreakdownTable title="By Direction" data={data?.by_direction || []} keyField="side" />
        <BreakdownTable title="By Regime" data={data?.by_regime || []} keyField="regime" />
        <BreakdownTable title="By Session" data={data?.by_session || []} keyField="session" />
        <BreakdownTable title="By Day of Week" data={data?.by_day_of_week || []} keyField="dow" />
        <BreakdownTable title="By Exit Reason" data={data?.by_exit_reason || []} keyField="exit_reason" />
      </div>
    </div>
  );
}
