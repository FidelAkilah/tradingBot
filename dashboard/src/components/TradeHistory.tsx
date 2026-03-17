'use client';

import { useTrades, useStatus } from '@/lib/hooks';

export function TradeHistory() {
  const { data } = useTrades(50);
  const { data: status } = useStatus();

  const trades = data?.trades || [];
  const leverage = status?.leverage || 30;

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs font-bold tracking-wider text-text-secondary uppercase">
          Trade History
        </h2>
        <span className="text-xs text-text-secondary">
          Last {trades.length} trades
        </span>
      </div>

      {trades.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-text-secondary/60 border-b border-border-dim">
                <th className="text-left py-2 px-2 font-bold">Time</th>
                <th className="text-left py-2 px-2 font-bold">Pair</th>
                <th className="text-left py-2 px-2 font-bold">Side</th>
                <th className="text-right py-2 px-2 font-bold">Entry</th>
                <th className="text-right py-2 px-2 font-bold">Exit</th>
                <th className="text-right py-2 px-2 font-bold">P&L (w/ {leverage}x)</th>
                <th className="text-left py-2 px-2 font-bold">Reason</th>
                <th className="text-left py-2 px-2 font-bold">Trend</th>
                <th className="text-right py-2 px-2 font-bold">Duration</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((trade: any, i: number) => {
                const isBuy = trade.side === 'BUY';
                const pnl = trade.pnl_usd || 0;
                const pnlLev = pnl * leverage;
                const isProfit = pnl > 0;
                const duration = trade.duration_s || 0;
                const durStr = duration > 3600
                  ? `${(duration / 3600).toFixed(1)}h`
                  : duration > 60
                    ? `${(duration / 60).toFixed(0)}m`
                    : `${duration.toFixed(0)}s`;

                const time = trade.entry_time
                  ? new Date(trade.entry_time * 1000).toLocaleString('en-US', {
                      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
                    })
                  : 'N/A';

                const reason = trade.exit_reason || (trade.is_open ? 'OPEN' : '—');
                const reasonColor = reason === 'take_profit' ? 'text-accent-green' :
                                    reason === 'stop_loss' ? 'text-accent-red' :
                                    reason === 'OPEN' ? 'text-accent-blue' :
                                    'text-accent-yellow';

                return (
                  <tr
                    key={i}
                    className="border-b border-border-dim/30 hover:bg-bg-secondary/30 transition-colors"
                  >
                    <td className="py-2 px-2 text-text-secondary">{time}</td>
                    <td className="py-2 px-2 font-bold text-text-primary">
                      {trade.symbol?.replace('/USDT', '')}
                    </td>
                    <td className="py-2 px-2">
                      <span className={`font-bold ${isBuy ? 'text-accent-green' : 'text-accent-red'}`}>
                        {trade.side}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-right text-text-primary">
                      ${trade.entry_price?.toFixed(2)}
                    </td>
                    <td className="py-2 px-2 text-right text-text-primary">
                      {trade.exit_price ? `$${trade.exit_price.toFixed(2)}` : '—'}
                    </td>
                    <td className={`py-2 px-2 text-right font-bold ${
                      trade.is_open ? 'text-accent-blue' :
                      isProfit ? 'text-accent-green' : 'text-accent-red'
                    }`}>
                      {trade.is_open ? 'OPEN' : `${isProfit ? '+' : ''}$${pnlLev.toFixed(4)}`}
                    </td>
                    <td className={`py-2 px-2 ${reasonColor}`}>
                      {reason}
                    </td>
                    <td className="py-2 px-2 text-text-secondary">
                      {trade.swing_trend || '—'}
                    </td>
                    <td className="py-2 px-2 text-right text-text-secondary">
                      {trade.is_open ? '—' : durStr}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="h-32 flex items-center justify-center text-text-secondary text-sm">
          No trades yet. Bot is analyzing markets...
        </div>
      )}
    </div>
  );
}
