'use client';

import { useRiskAnalytics, useCorrelation, useExposure } from '@/lib/hooks';
import { ShieldAlert, AlertTriangle, Gauge } from 'lucide-react';

function DrawdownGauge({ pct, max }: { pct: number; max: number }) {
  const ratio = Math.min(pct / max, 1);
  const color =
    pct >= 15 ? '#ef4444' :
    pct >= 10 ? '#f59e0b' :
    pct >= 5 ? '#3b82f6' :
    '#10b981';

  return (
    <div className="bg-bg-secondary/50 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] text-text-secondary uppercase tracking-wider">Current Drawdown</span>
        <span className="text-[10px] text-text-secondary">Max: {max}%</span>
      </div>
      <div className="flex items-end gap-3">
        <span className="text-2xl font-black" style={{ color }}>{pct.toFixed(1)}%</span>
        <div className="flex-1">
          <div className="w-full h-3 bg-bg-primary rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${ratio * 100}%`, backgroundColor: color }}
            />
          </div>
          <div className="flex justify-between mt-1 text-[9px] text-text-secondary">
            <span>0%</span>
            <span>5%</span>
            <span>10%</span>
            <span>15%</span>
            <span>25%</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function DailyPnlBar({ pnl, target, lossLimit }: { pnl: number; target: number; lossLimit: number }) {
  const maxRange = Math.max(target, lossLimit, Math.abs(pnl)) * 1.2;
  const center = 50;
  const pnlPct = (pnl / maxRange) * 50;
  const targetMark = (target / maxRange) * 50;
  const lossMark = (lossLimit / maxRange) * 50;

  return (
    <div className="bg-bg-secondary/50 rounded-lg p-4">
      <div className="text-[10px] text-text-secondary uppercase tracking-wider mb-2">Daily P&L Progress</div>
      <div className="relative w-full h-6 bg-bg-primary rounded-full overflow-hidden">
        {/* Loss zone */}
        <div className="absolute left-0 h-full bg-accent-red/10" style={{ width: `${center}%` }} />
        {/* Profit zone */}
        <div className="absolute right-0 h-full bg-accent-green/10" style={{ width: `${center}%` }} />

        {/* P&L bar */}
        {pnl >= 0 ? (
          <div
            className="absolute top-0 h-full bg-accent-green/40 rounded-r-full"
            style={{ left: `${center}%`, width: `${Math.min(pnlPct, 50)}%` }}
          />
        ) : (
          <div
            className="absolute top-0 h-full bg-accent-red/40 rounded-l-full"
            style={{ right: `${center}%`, width: `${Math.min(Math.abs(pnlPct), 50)}%` }}
          />
        )}

        {/* Target line */}
        <div
          className="absolute top-0 h-full w-0.5 bg-accent-yellow"
          style={{ left: `${center + targetMark}%` }}
          title={`Target: $${target.toFixed(2)}`}
        />

        {/* Loss limit line */}
        <div
          className="absolute top-0 h-full w-0.5 bg-accent-red"
          style={{ left: `${center - lossMark}%` }}
          title={`Loss limit: -$${lossLimit.toFixed(2)}`}
        />

        {/* Center line */}
        <div className="absolute top-0 h-full w-px bg-text-secondary/30" style={{ left: '50%' }} />
      </div>

      <div className="flex justify-between mt-1 text-[9px]">
        <span className="text-accent-red">-${lossLimit.toFixed(2)}</span>
        <span className={`font-bold ${pnl >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
          ${pnl.toFixed(4)}
        </span>
        <span className="text-accent-yellow">+${target.toFixed(2)}</span>
      </div>
    </div>
  );
}

function PositionDetail({ position }: { position: any }) {
  return (
    <div className="flex items-center gap-3 p-2 bg-bg-secondary/30 rounded-lg">
      <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
        position.side === 'BUY' ? 'bg-accent-green/20 text-accent-green' : 'bg-accent-red/20 text-accent-red'
      }`}>
        {position.side}
      </span>
      <span className="text-xs font-medium text-text-primary">
        {(position.symbol || '').replace('/USDT', '')}
      </span>
      <span className="text-[10px] text-text-secondary">{position.leverage}x</span>
      <div className="flex-1" />
      <div className="text-right text-[10px]">
        <div className="text-text-secondary">Kelly: {((position.kelly_pct || 0) * 100).toFixed(0)}%</div>
        <div className="text-text-secondary">DD mult: {(position.drawdown_mult || 1).toFixed(2)}</div>
      </div>
    </div>
  );
}

function CorrelationMatrix({ data }: { data: any }) {
  const matrix = data?.matrix || {};
  const pairs = data?.pairs || [];
  const correlatedPositions = data?.correlated_positions || [];

  if (pairs.length === 0) return null;

  const shortPairs = pairs.map((p: string) => p.replace('/USDT', ''));

  return (
    <div className="bg-bg-secondary/50 rounded-lg p-4">
      <div className="text-[10px] text-text-secondary uppercase tracking-wider mb-2">
        Position Correlation
      </div>
      {correlatedPositions.length > 0 ? (
        <div className="space-y-1">
          {correlatedPositions.map((cp: any, i: number) => (
            <div key={i} className={`flex items-center justify-between p-2 rounded text-xs ${
              cp.high_corr && cp.same_direction
                ? 'bg-accent-red/10 border border-accent-red/20'
                : 'bg-bg-secondary/30'
            }`}>
              <span className="text-text-primary">
                {cp.pair[0]?.replace('/USDT', '')} ↔ {cp.pair[1]?.replace('/USDT', '')}
              </span>
              <span className={cp.high_corr ? 'text-accent-red font-bold' : 'text-text-secondary'}>
                {cp.correlation?.toFixed(2)}
              </span>
              <span className={`text-[10px] ${cp.same_direction ? 'text-accent-yellow' : 'text-accent-green'}`}>
                {cp.same_direction ? 'Same Dir' : 'Hedge'}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-[10px] text-text-secondary text-center py-3">
          No correlated open positions
        </div>
      )}
    </div>
  );
}

export function RiskDashboard() {
  const { data: risk } = useRiskAnalytics();
  const { data: correlation } = useCorrelation();
  const { data: exposure } = useExposure();

  const drawdown = risk?.drawdown_pct || 0;
  const maxDD = risk?.max_drawdown_pct || 25;
  const dailyPnl = risk?.daily_pnl || 0;
  const mode = risk?.mode || 'normal';
  const positions = risk?.positions || [];

  const targetAmount = (risk?.daily_target_pct || 2) * (risk?.pct_achieved || 0) / 100;
  const lossLimit = risk?.daily_loss_limit || 0.5;

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5 fade-in space-y-4">
      <div className="flex items-center gap-2">
        <ShieldAlert className="w-4 h-4 text-accent-yellow" />
        <h2 className="text-xs font-bold tracking-wider text-text-secondary uppercase">
          Risk Dashboard
        </h2>
        {risk?.is_halted && (
          <span className="text-[10px] font-bold text-accent-red bg-accent-red/10 px-2 py-0.5 rounded">
            HALTED: {risk.halt_reason}
          </span>
        )}
      </div>

      {/* Drawdown gauge */}
      <DrawdownGauge pct={drawdown} max={maxDD} />

      {/* Daily P&L bar */}
      <DailyPnlBar pnl={dailyPnl} target={lossLimit * 2} lossLimit={lossLimit} />

      {/* Sizer state summary */}
      {risk?.sizer_state && (
        <div className="bg-bg-secondary/50 rounded-lg p-4">
          <div className="text-[10px] text-text-secondary uppercase tracking-wider mb-2">Position Sizing</div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-lg font-bold text-text-primary">
                {((risk.sizer_state.kelly_fraction || 0) * 100).toFixed(0)}%
              </div>
              <div className="text-[9px] text-text-secondary">Kelly</div>
            </div>
            <div>
              <div className="text-lg font-bold text-text-primary">
                {risk.sizer_state.consecutive_losses || 0}
              </div>
              <div className="text-[9px] text-text-secondary">Consec. Losses</div>
            </div>
            <div>
              <div className="text-lg font-bold text-text-primary capitalize">
                {mode}
              </div>
              <div className="text-[9px] text-text-secondary">Mode</div>
            </div>
          </div>
        </div>
      )}

      {/* Open positions with leverage */}
      {positions.length > 0 && (
        <div>
          <div className="text-[10px] text-text-secondary uppercase tracking-wider mb-2">
            Open Position Sizing
          </div>
          <div className="space-y-1">
            {positions.map((p: any, i: number) => <PositionDetail key={i} position={p} />)}
          </div>
        </div>
      )}

      {/* Correlation heat map */}
      <CorrelationMatrix data={correlation} />

      {/* Exposure */}
      {exposure && (
        <div className="bg-bg-secondary/50 rounded-lg p-4">
          <div className="text-[10px] text-text-secondary uppercase tracking-wider mb-2">
            Portfolio Exposure
          </div>
          <div className="grid grid-cols-2 gap-3 text-center text-xs">
            <div>
              <div className={`text-lg font-bold ${exposure.breach_long ? 'text-accent-red' : 'text-accent-green'}`}>
                {(exposure.net_long || 0).toFixed(1)}x
              </div>
              <div className="text-[9px] text-text-secondary">Net Long (max {exposure.max_long}x)</div>
            </div>
            <div>
              <div className={`text-lg font-bold ${exposure.breach_short ? 'text-accent-red' : 'text-accent-green'}`}>
                {(exposure.net_short || 0).toFixed(1)}x
              </div>
              <div className="text-[9px] text-text-secondary">Net Short (max {exposure.max_short}x)</div>
            </div>
          </div>
          <div className="text-center mt-2 text-[10px] text-text-secondary capitalize">
            Bias: {exposure.directional_bias || 'neutral'}
          </div>
        </div>
      )}
    </div>
  );
}
