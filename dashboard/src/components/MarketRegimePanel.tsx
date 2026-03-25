'use client';

import { useRegime } from '@/lib/hooks';
import { Activity, TrendingUp, TrendingDown, Minus, Lock, Unlock } from 'lucide-react';

function RegimeBadge({ regime }: { regime: string }) {
  const cfg: Record<string, { label: string; color: string; bg: string }> = {
    strong_trend: { label: 'STRONG TREND', color: 'text-accent-green', bg: 'bg-accent-green/10' },
    trend: { label: 'TREND', color: 'text-accent-green', bg: 'bg-accent-green/10' },
    weak: { label: 'WEAK', color: 'text-accent-yellow', bg: 'bg-accent-yellow/10' },
    ranging: { label: 'RANGING', color: 'text-accent-red', bg: 'bg-accent-red/10' },
    squeezing: { label: 'SQUEEZE', color: 'text-accent-purple', bg: 'bg-accent-purple/10' },
    unknown: { label: 'UNKNOWN', color: 'text-text-secondary', bg: 'bg-bg-secondary' },
  };
  const c = cfg[regime] || cfg.unknown;
  return (
    <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${c.color} ${c.bg}`}>
      {c.label}
    </span>
  );
}

function TrendIcon({ trend }: { trend: string }) {
  if (trend.includes('bullish'))
    return <TrendingUp className="w-3.5 h-3.5 text-accent-green" />;
  if (trend.includes('bearish'))
    return <TrendingDown className="w-3.5 h-3.5 text-accent-red" />;
  return <Minus className="w-3.5 h-3.5 text-text-secondary" />;
}

function ADXBar({ value }: { value: number }) {
  const pct = Math.min((value / 60) * 100, 100);
  const color =
    value >= 25 ? 'bg-accent-green' :
    value >= 20 ? 'bg-accent-yellow' :
    'bg-accent-red';
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-bg-secondary rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] text-text-secondary w-6 text-right">{value.toFixed(0)}</span>
    </div>
  );
}

export function MarketRegimePanel() {
  const { data, isLoading } = useRegime();
  const pairs = data?.pairs || {};

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5 fade-in">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-4 h-4 text-accent-blue" />
        <h2 className="text-xs font-bold tracking-wider text-text-secondary uppercase">
          Market Regime
        </h2>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3].map(i => <div key={i} className="h-12 bg-bg-secondary/50 rounded-lg shimmer" />)}
        </div>
      ) : Object.keys(pairs).length === 0 ? (
        <div className="text-sm text-text-secondary text-center py-6">
          Waiting for market data...
        </div>
      ) : (
        <div className="space-y-2">
          {Object.entries(pairs).map(([symbol, p]: [string, any]) => (
            <div
              key={symbol}
              className={`flex items-center gap-3 p-3 rounded-lg border transition-colors ${
                p.eligible
                  ? 'bg-bg-secondary/30 border-accent-green/20'
                  : 'bg-bg-secondary/20 border-border-dim/50 opacity-70'
              }`}
            >
              {/* Symbol + Trend */}
              <div className="w-24 flex items-center gap-1.5">
                <TrendIcon trend={p.trend} />
                <span className="text-sm font-bold text-text-primary">
                  {symbol.replace('/USDT', '')}
                </span>
              </div>

              {/* Regime badge */}
              <RegimeBadge regime={p.regime} />

              {/* ADX */}
              <div className="flex-1">
                <div className="text-[10px] text-text-secondary mb-0.5">ADX</div>
                <ADXBar value={p.adx || 0} />
              </div>

              {/* RSI */}
              <div className="w-16 text-center">
                <div className="text-[10px] text-text-secondary">RSI</div>
                <div className={`text-xs font-medium ${
                  p.rsi_1h > 70 ? 'text-accent-red' :
                  p.rsi_1h < 30 ? 'text-accent-green' :
                  'text-text-primary'
                }`}>
                  {(p.rsi_1h || 50).toFixed(0)}
                </div>
              </div>

              {/* Session */}
              <div className="w-14 text-center">
                <div className="text-[10px] text-text-secondary">Session</div>
                <div className="text-[10px] text-text-primary capitalize">{p.session}</div>
              </div>

              {/* Eligibility */}
              <div className="w-8 flex justify-center" title={p.block_reason || 'Eligible'}>
                {p.eligible ? (
                  <Unlock className="w-3.5 h-3.5 text-accent-green" />
                ) : (
                  <Lock className="w-3.5 h-3.5 text-accent-red" />
                )}
              </div>

              {/* Squeeze indicator */}
              {(p.squeeze_active || p.squeeze_releasing) && (
                <div className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                  p.squeeze_releasing ? 'bg-accent-purple/20 text-accent-purple' : 'bg-accent-yellow/20 text-accent-yellow'
                }`}>
                  {p.squeeze_releasing ? 'RELEASE' : 'SQZ'}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
