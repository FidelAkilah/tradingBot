'use client';

import { useSignals } from '@/lib/hooks';
import { Radio, TrendingUp, TrendingDown, Minus, Filter } from 'lucide-react';

function SignalCard({ signal }: { signal: any }) {
  const suggestion = signal.suggestion;
  const confidence = signal.swing_confidence || 0;
  const adxBlocked = signal.adx_blocked;
  const regimeBlocked = signal.regime_blocked;
  const sessionBlocked = signal.session_blocked;

  // Determine status
  let status: 'taken' | 'filtered' | 'low_conf' = 'taken';
  let statusLabel = 'SIGNAL';
  let statusColor = 'border-accent-green/30 bg-accent-green/5';
  let dotColor = 'bg-accent-green';

  if (!suggestion) {
    status = 'low_conf';
    statusLabel = 'NO SIGNAL';
    statusColor = 'border-border-dim/30 bg-bg-secondary/20';
    dotColor = 'bg-text-secondary';
  } else if (adxBlocked || regimeBlocked || sessionBlocked) {
    status = 'filtered';
    statusLabel = adxBlocked ? 'ADX BLOCK' : regimeBlocked ? 'REGIME BLOCK' : 'SESSION BLOCK';
    statusColor = 'border-accent-yellow/30 bg-accent-yellow/5';
    dotColor = 'bg-accent-yellow';
  } else if (confidence < 0.55) {
    status = 'low_conf';
    statusLabel = 'LOW CONF';
    statusColor = 'border-accent-red/30 bg-accent-red/5';
    dotColor = 'bg-accent-red';
  }

  const time = signal.timestamp
    ? new Date(signal.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '';

  return (
    <div className={`flex items-center gap-3 p-2.5 rounded-lg border ${statusColor} signal-enter`}>
      <div className={`w-2 h-2 rounded-full ${dotColor} flex-shrink-0`} />

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-text-primary">
            {(signal.symbol || '').replace('/USDT', '')}
          </span>
          {suggestion && (
            <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
              suggestion === 'BUY' ? 'bg-accent-green/20 text-accent-green' : 'bg-accent-red/20 text-accent-red'
            }`}>
              {suggestion}
            </span>
          )}
          <span className="text-[10px] text-text-secondary capitalize">
            {signal.swing_trend || 'neutral'}
          </span>
        </div>
        <div className="flex items-center gap-3 mt-0.5 text-[10px] text-text-secondary">
          <span>Conf: {(confidence * 100).toFixed(0)}%</span>
          <span>ADX: {(signal.adx || 0).toFixed(0)}</span>
          <span>${(signal.mid_price || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
          {signal.regime && <span className="capitalize">{signal.regime}</span>}
        </div>
      </div>

      <div className="flex flex-col items-end flex-shrink-0">
        <span className={`text-[9px] font-bold uppercase px-1.5 py-0.5 rounded ${
          status === 'taken' ? 'text-accent-green bg-accent-green/10' :
          status === 'filtered' ? 'text-accent-yellow bg-accent-yellow/10' :
          'text-text-secondary bg-bg-secondary'
        }`}>
          {statusLabel}
        </span>
        <span className="text-[9px] text-text-secondary mt-0.5">{time}</span>
      </div>
    </div>
  );
}

export function LiveSignalMonitor() {
  const { data, isLoading } = useSignals(30);
  const signals = data?.signals || [];

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5 fade-in">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Radio className="w-4 h-4 text-accent-green" />
          <h2 className="text-xs font-bold tracking-wider text-text-secondary uppercase">
            Live Signal Monitor
          </h2>
          <div className="w-2 h-2 rounded-full bg-accent-green pulse-dot" />
        </div>
        <span className="text-[10px] text-text-secondary">{signals.length} signals</span>
      </div>

      {/* Legend */}
      <div className="flex gap-3 mb-3 text-[10px]">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-accent-green" />Trade taken</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-accent-yellow" />Filtered</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-accent-red" />Low confidence</span>
      </div>

      <div className="space-y-1.5 max-h-[400px] overflow-y-auto pr-1">
        {isLoading ? (
          <div className="space-y-2">
            {[1, 2, 3, 4, 5].map(i => <div key={i} className="h-14 bg-bg-secondary/30 rounded-lg shimmer" />)}
          </div>
        ) : signals.length === 0 ? (
          <div className="text-sm text-text-secondary text-center py-8">
            Waiting for signals...
          </div>
        ) : (
          signals.map((s: any, i: number) => <SignalCard key={i} signal={s} />)
        )}
      </div>
    </div>
  );
}
