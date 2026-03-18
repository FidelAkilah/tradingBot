'use client';

import { usePositions, useStatus } from '@/lib/hooks';
import { ArrowUpRight, ArrowDownRight } from 'lucide-react';

export function PositionsPanel() {
  const { data } = usePositions();
  const { data: status } = useStatus();

  const positions = data?.positions || [];
  const leverage = status?.leverage || 30;

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs font-bold tracking-wider text-text-secondary uppercase">
          Open Positions
        </h2>
        <span className="text-xs text-text-secondary">
          {positions.length} active
        </span>
      </div>

      {positions.length > 0 ? (
        <div className="space-y-3">
          {positions.map((pos: any, i: number) => {
            const isBuy = pos.side === 'BUY';
            const pnl = pos.pnl_usd || 0;
            const isProfit = pnl >= 0;
            const pnlPct = pos.entry_price ? ((pos.current_price || pos.entry_price) / pos.entry_price - 1) * 100 * (isBuy ? 1 : -1) : 0;
            const holdTime = pos.entry_time ? (Date.now() / 1000 - pos.entry_time) : 0;
            const holdMin = Math.floor(holdTime / 60);
            const holdHr = Math.floor(holdMin / 60);

            return (
              <div
                key={i}
                className={`rounded-lg border p-3 ${
                  isBuy ? 'border-accent-green/20 bg-accent-green/5' : 'border-accent-red/20 bg-accent-red/5'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {isBuy ? (
                      <ArrowUpRight className="w-4 h-4 text-accent-green" />
                    ) : (
                      <ArrowDownRight className="w-4 h-4 text-accent-red" />
                    )}
                    <span className="text-sm font-bold text-text-primary">
                      {pos.symbol?.replace('/USDT', '')}
                    </span>
                    <span className={`text-xs font-bold ${isBuy ? 'text-accent-green' : 'text-accent-red'}`}>
                      {pos.side} {leverage}x
                    </span>
                  </div>
                  <span className={`text-sm font-bold ${isProfit ? 'text-accent-green' : 'text-accent-red'}`}>
                    {isProfit ? '+' : ''}{pnl.toFixed(2)} USD
                    <span className="text-[10px] ml-1 opacity-70">
                      ({pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%)
                    </span>
                  </span>
                </div>

                <div className="grid grid-cols-4 gap-2 text-xs text-text-secondary">
                  <div>
                    <span className="text-text-secondary/50">Entry:</span>{' '}
                    <span className="text-text-primary">${pos.entry_price?.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-text-secondary/50">Mark:</span>{' '}
                    <span className="text-accent-blue">${(pos.current_price || pos.entry_price)?.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-text-secondary/50">TP:</span>{' '}
                    <span className="text-accent-green">${pos.target_price?.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-text-secondary/50">SL:</span>{' '}
                    <span className="text-accent-red">${pos.stop_price?.toFixed(2)}</span>
                  </div>
                </div>

                <div className="flex items-center justify-between mt-2 text-[10px] text-text-secondary/60">
                  <span>Margin: ${pos.usd_value?.toFixed(2) || '—'}</span>
                  <span>Trend: {pos.swing_trend || 'N/A'}</span>
                  <span>Hold: {holdHr > 0 ? `${holdHr}h ${holdMin % 60}m` : `${holdMin}m`}</span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="h-40 flex items-center justify-center text-text-secondary text-sm">
          No open positions
        </div>
      )}
    </div>
  );
}
