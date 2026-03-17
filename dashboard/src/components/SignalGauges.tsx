'use client';

import { usePrices, useSignals } from '@/lib/hooks';

function Gauge({ label, value, max, color, unit }: {
  label: string; value: number; max: number; color: string; unit?: string;
}) {
  const pct = Math.min((value / max) * 100, 100);
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle
          cx="50" cy="50" r={radius}
          fill="none" stroke="#1f2937" strokeWidth="8"
        />
        <circle
          cx="50" cy="50" r={radius}
          fill="none" stroke={color} strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 50 50)"
          style={{ transition: 'stroke-dashoffset 0.5s ease' }}
        />
        <text x="50" y="46" textAnchor="middle" fill="#e5e7eb" fontSize="14" fontWeight="bold">
          {value.toFixed(value < 10 ? 2 : 0)}
        </text>
        {unit && (
          <text x="50" y="60" textAnchor="middle" fill="#9ca3af" fontSize="9">
            {unit}
          </text>
        )}
      </svg>
      <span className="text-[10px] text-text-secondary mt-1 uppercase tracking-wider font-bold">
        {label}
      </span>
    </div>
  );
}

function TrendBadge({ trend, confidence }: { trend: string; confidence: number }) {
  const color = trend === 'bullish' ? 'text-accent-green bg-accent-green/10' :
                trend === 'bearish' ? 'text-accent-red bg-accent-red/10' :
                'text-text-secondary bg-bg-secondary';
  const arrow = trend === 'bullish' ? '↑' : trend === 'bearish' ? '↓' : '→';

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-bold ${color}`}>
      {arrow} {trend} ({(confidence * 100).toFixed(0)}%)
    </span>
  );
}

export function SignalGauges() {
  const { data: priceData } = usePrices();
  const { data: signalData } = useSignals(20);

  const prices = priceData?.prices || {};
  const pairs = priceData?.pairs || [];
  const recentSignals = signalData?.signals || [];

  // Get latest signal per pair
  const latestByPair: Record<string, any> = {};
  for (const sig of recentSignals) {
    if (!latestByPair[sig.symbol]) {
      latestByPair[sig.symbol] = sig;
    }
  }

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5">
      <h2 className="text-xs font-bold tracking-wider text-text-secondary uppercase mb-4">
        Live Signals
      </h2>

      {pairs.length > 0 ? (
        <div className="space-y-4">
          {pairs.map((pair: string) => {
            const signal = latestByPair[pair];
            const priceInfo = prices[pair];
            const vpin = signal?.vpin || 0;
            const score = signal?.composite_score || 0;
            const rsi = priceInfo?.rsi_1h || signal?.rsi_1h || 50;
            const trend = priceInfo?.trend || signal?.swing_trend || 'neutral';
            const confidence = priceInfo?.confidence || signal?.swing_confidence || 0;

            return (
              <div key={pair} className="bg-bg-secondary/30 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-bold text-text-primary">
                    {pair.replace('/USDT', '')}
                  </span>
                  <TrendBadge trend={trend} confidence={confidence} />
                </div>

                <div className="flex items-center justify-around">
                  <Gauge label="VPIN" value={vpin} max={1} color={
                    vpin > 0.85 ? '#ef4444' : vpin > 0.6 ? '#f59e0b' : '#10b981'
                  } />
                  <Gauge label="Score" value={Math.abs(score)} max={1} color={
                    score > 0.3 ? '#10b981' : score < -0.3 ? '#ef4444' : '#6b7280'
                  } />
                  <Gauge label="RSI" value={rsi} max={100} color={
                    rsi > 70 ? '#ef4444' : rsi < 30 ? '#10b981' : '#3b82f6'
                  } />
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="h-40 flex items-center justify-center text-text-secondary text-sm">
          Waiting for signal data...
        </div>
      )}
    </div>
  );
}
