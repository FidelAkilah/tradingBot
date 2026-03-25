'use client';

import { useState } from 'react';
import { useTrades } from '@/lib/hooks';
import { fetchAPI, getAPIUrl } from '@/lib/api';
import { BookOpen, ChevronDown, ChevronUp, Download, Search, X } from 'lucide-react';

function formatDuration(s: number): string {
  if (!s || s <= 0) return '-';
  if (s >= 3600) return `${(s / 3600).toFixed(1)}h`;
  if (s >= 60) return `${Math.round(s / 60)}m`;
  return `${Math.round(s)}s`;
}

function TradeCard({ trade, expanded, onToggle }: { trade: any; expanded: boolean; onToggle: () => void }) {
  const [notes, setNotes] = useState(trade.notes || '');
  const [saving, setSaving] = useState(false);
  const pnl = trade.pnl_usd || 0;
  const isWin = pnl > 0;
  const time = trade.entry_time ? new Date(trade.entry_time * 1000).toLocaleString() : '';
  const exitTime = trade.exit_time ? new Date(trade.exit_time * 1000).toLocaleString() : '';

  const saveNotes = async () => {
    setSaving(true);
    try {
      await fetch(getAPIUrl(`/api/trades/${trade.trade_id}/notes`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes }),
      });
    } catch {}
    setSaving(false);
  };

  return (
    <div className={`rounded-lg border transition-colors ${
      isWin ? 'border-accent-green/20 bg-accent-green/5' : 'border-accent-red/20 bg-accent-red/5'
    }`}>
      {/* Header row */}
      <button onClick={onToggle} className="w-full flex items-center gap-3 p-3 text-left">
        <div className={`w-1.5 h-10 rounded-full ${isWin ? 'bg-accent-green' : 'bg-accent-red'}`} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold text-text-primary">
              {(trade.symbol || '').replace('/USDT', '')}
            </span>
            <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
              trade.side === 'BUY' ? 'bg-accent-green/20 text-accent-green' : 'bg-accent-red/20 text-accent-red'
            }`}>
              {trade.side} {trade.leverage || 10}x
            </span>
            <span className="text-[10px] text-text-secondary">{time}</span>
          </div>
          <div className="flex items-center gap-3 mt-0.5 text-[10px] text-text-secondary">
            <span>Entry: ${(trade.entry_price || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
            <span>Exit: ${(trade.exit_price || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
            <span>{formatDuration(trade.duration_s)}</span>
            <span className="capitalize">{trade.exit_reason}</span>
          </div>
        </div>
        <div className="text-right flex-shrink-0">
          <div className={`text-sm font-bold ${isWin ? 'text-accent-green' : 'text-accent-red'}`}>
            {isWin ? '+' : ''}${pnl.toFixed(4)}
          </div>
          <div className="text-[10px] text-text-secondary">
            {(trade.pnl_pct || 0).toFixed(2)}%
          </div>
        </div>
        {expanded ? <ChevronUp className="w-4 h-4 text-text-secondary" /> : <ChevronDown className="w-4 h-4 text-text-secondary" />}
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-3 pb-3 space-y-3 fade-in border-t border-border-dim/30 pt-3 ml-4">
          {/* Indicator grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
            {[
              { label: 'Confidence', value: `${((trade.swing_confidence || 0) * 100).toFixed(0)}%` },
              { label: 'ADX', value: (trade.adx || 0).toFixed(1) },
              { label: 'RSI 1h', value: (trade.rsi_1h || 50).toFixed(0) },
              { label: 'RSI 4h', value: (trade.rsi_4h || 50).toFixed(0) },
              { label: 'Regime', value: trade.regime || '-' },
              { label: 'Trend', value: trade.swing_trend || '-' },
              { label: 'Post-fee R:R', value: (trade.post_fee_rr || 0).toFixed(2) },
              { label: 'ATR TP%', value: `${(trade.atr_tp_pct || 0).toFixed(2)}%` },
              { label: 'ATR SL%', value: `${(trade.atr_sl_pct || 0).toFixed(2)}%` },
              { label: 'Kelly%', value: `${((trade.kelly_pct || 0) * 100).toFixed(1)}%` },
              { label: 'OBV', value: trade.obv_trend || '-' },
              { label: 'Funding', value: `${((trade.funding_rate || 0) * 100).toFixed(3)}%` },
              { label: 'OI Change', value: `${(trade.oi_change_pct || 0).toFixed(1)}%` },
              { label: 'OI Conviction', value: trade.oi_conviction || '-' },
              { label: 'Volume Press.', value: trade.volume_pressure || '-' },
              { label: 'VPIN', value: (trade.vpin || 0).toFixed(3) },
              { label: 'Fee Cost', value: `${(trade.fee_cost_pct || 0).toFixed(3)}%` },
              { label: 'DD Mult', value: (trade.drawdown_mult || 1).toFixed(2) },
            ].map((item, i) => (
              <div key={i} className="bg-bg-secondary/40 rounded p-1.5">
                <div className="text-[9px] text-text-secondary uppercase">{item.label}</div>
                <div className="text-[11px] text-text-primary font-medium capitalize">{item.value}</div>
              </div>
            ))}
          </div>

          {/* Partial TP timeline */}
          {(trade.tp1_hit || trade.tp2_hit || trade.tp3_hit) && (
            <div>
              <div className="text-[10px] text-text-secondary uppercase mb-1.5">Partial TP Timeline</div>
              <div className="flex gap-2">
                {[
                  { label: 'TP1', hit: trade.tp1_hit, price: trade.tp1_price, pnl: trade.tp1_pnl },
                  { label: 'TP2', hit: trade.tp2_hit, price: trade.tp2_price, pnl: trade.tp2_pnl },
                  { label: 'TP3', hit: trade.tp3_hit, price: trade.tp3_price, pnl: trade.tp3_pnl },
                ].map((tp, i) => (
                  <div key={i} className={`flex-1 rounded p-2 text-center ${
                    tp.hit ? 'bg-accent-green/10 border border-accent-green/30' : 'bg-bg-secondary/30 border border-border-dim/30'
                  }`}>
                    <div className="text-[10px] font-bold">{tp.label}</div>
                    <div className="text-[10px]">
                      {tp.hit ? `$${(tp.price || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}` : '-'}
                    </div>
                    {tp.pnl != null && tp.hit && (
                      <div className={`text-[10px] font-medium ${(tp.pnl || 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                        {tp.pnl >= 0 ? '+' : ''}${(tp.pnl || 0).toFixed(4)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* AI Advisor */}
          {trade.advisor_recommendation && (
            <div>
              <div className="text-[10px] text-text-secondary uppercase mb-1.5">AI Advisor</div>
              <div className={`rounded p-2 border ${
                trade.advisor_recommendation === 'PROCEED' ? 'bg-accent-green/5 border-accent-green/20' :
                trade.advisor_recommendation === 'SKIP' ? 'bg-accent-red/5 border-accent-red/20' :
                'bg-accent-gold/5 border-accent-gold/20'
              }`}>
                <div className="flex items-center gap-2 mb-1">
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                    trade.advisor_recommendation === 'PROCEED' ? 'bg-accent-green/20 text-accent-green' :
                    trade.advisor_recommendation === 'SKIP' ? 'bg-accent-red/20 text-accent-red' :
                    'bg-accent-gold/20 text-accent-gold'
                  }`}>
                    {trade.advisor_recommendation}
                  </span>
                  {trade.advisor_confidence_adj != null && trade.advisor_confidence_adj !== 0 && (
                    <span className={`text-[10px] font-medium ${
                      trade.advisor_confidence_adj > 0 ? 'text-accent-green' : 'text-accent-red'
                    }`}>
                      {trade.advisor_confidence_adj > 0 ? '+' : ''}{(trade.advisor_confidence_adj * 100).toFixed(0)}% conf
                    </span>
                  )}
                </div>
                {trade.advisor_reasoning && (
                  <p className="text-[10px] text-text-secondary leading-relaxed">{trade.advisor_reasoning}</p>
                )}
              </div>
            </div>
          )}

          {/* Notes */}
          <div>
            <div className="text-[10px] text-text-secondary uppercase mb-1">Notes</div>
            <div className="flex gap-2">
              <input
                type="text"
                value={notes}
                onChange={e => setNotes(e.target.value)}
                placeholder="Add trade notes..."
                className="flex-1 bg-bg-secondary/50 border border-border-dim rounded px-2 py-1 text-xs text-text-primary placeholder:text-text-secondary/50 outline-none focus:border-accent-blue"
              />
              <button
                onClick={saveNotes}
                disabled={saving}
                className="px-2 py-1 bg-accent-blue/20 text-accent-blue text-[10px] font-bold rounded hover:bg-accent-blue/30 disabled:opacity-50"
              >
                {saving ? '...' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function TradeJournal() {
  const { data, isLoading } = useTrades(100);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [filter, setFilter] = useState('');
  const [sideFilter, setSideFilter] = useState<string>('all');
  const [outcomeFilter, setOutcomeFilter] = useState<string>('all');

  const trades = (data?.trades || [])
    .filter((t: any) => !t.is_open)
    .filter((t: any) => {
      if (filter && !t.symbol?.toLowerCase().includes(filter.toLowerCase())) return false;
      if (sideFilter !== 'all' && t.side !== sideFilter) return false;
      if (outcomeFilter === 'win' && (t.pnl_usd || 0) <= 0) return false;
      if (outcomeFilter === 'loss' && (t.pnl_usd || 0) > 0) return false;
      return true;
    });

  const exportCSV = () => {
    if (!trades.length) return;
    const headers = ['trade_id', 'symbol', 'side', 'entry_price', 'exit_price', 'pnl_usd', 'pnl_pct', 'duration_s', 'exit_reason', 'swing_confidence', 'adx', 'regime'];
    const rows = trades.map((t: any) => headers.map(h => t[h] ?? '').join(','));
    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trades_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5 fade-in">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-accent-purple" />
          <h2 className="text-xs font-bold tracking-wider text-text-secondary uppercase">
            Trade Journal
          </h2>
          <span className="text-[10px] text-text-secondary">({trades.length} trades)</span>
        </div>
        <button onClick={exportCSV} className="flex items-center gap-1 text-[10px] text-accent-blue hover:text-accent-blue/80">
          <Download className="w-3 h-3" /> Export CSV
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-2 mb-4">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-text-secondary" />
          <input
            type="text"
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="Filter symbol..."
            className="pl-6 pr-2 py-1 bg-bg-secondary/50 border border-border-dim rounded text-[11px] text-text-primary placeholder:text-text-secondary/50 outline-none focus:border-accent-blue w-32"
          />
        </div>
        {['all', 'BUY', 'SELL'].map(s => (
          <button
            key={s}
            onClick={() => setSideFilter(s)}
            className={`px-2 py-1 rounded text-[10px] font-bold uppercase ${
              sideFilter === s ? 'bg-accent-blue/20 text-accent-blue' : 'bg-bg-secondary/30 text-text-secondary hover:text-text-primary'
            }`}
          >
            {s === 'all' ? 'All Sides' : s}
          </button>
        ))}
        {['all', 'win', 'loss'].map(o => (
          <button
            key={o}
            onClick={() => setOutcomeFilter(o)}
            className={`px-2 py-1 rounded text-[10px] font-bold uppercase ${
              outcomeFilter === o ? 'bg-accent-blue/20 text-accent-blue' : 'bg-bg-secondary/30 text-text-secondary hover:text-text-primary'
            }`}
          >
            {o === 'all' ? 'All' : o === 'win' ? 'Winners' : 'Losers'}
          </button>
        ))}
      </div>

      {/* Trade list */}
      <div className="space-y-2 max-h-[600px] overflow-y-auto pr-1">
        {isLoading ? (
          <div className="space-y-2">
            {[1, 2, 3].map(i => <div key={i} className="h-16 bg-bg-secondary/30 rounded-lg shimmer" />)}
          </div>
        ) : trades.length === 0 ? (
          <div className="text-sm text-text-secondary text-center py-8">No trades match filters</div>
        ) : (
          trades.map((t: any) => (
            <TradeCard
              key={t.trade_id || t.id}
              trade={t}
              expanded={expandedId === (t.trade_id || t.id)}
              onToggle={() => setExpandedId(expandedId === (t.trade_id || t.id) ? null : (t.trade_id || t.id))}
            />
          ))
        )}
      </div>
    </div>
  );
}
