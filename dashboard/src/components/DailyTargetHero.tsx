'use client';

import { useDailyTarget, useDailyTargetHistory, useDailyTargetProjection, useStatus } from '@/lib/hooks';
import { Target, TrendingUp, Clock, Shield, Zap, Pause, AlertTriangle } from 'lucide-react';

const IDR_RATE = 16_300;

function CircularGauge({ pct, size = 160 }: { pct: number; size?: number }) {
  const radius = (size - 16) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.min(Math.max(pct, 0), 150);
  const offset = circumference - (clamped / 100) * circumference;

  const color =
    pct >= 100 ? '#fbbf24' :
    pct >= 70 ? '#10b981' :
    pct >= 30 ? '#f59e0b' :
    '#ef4444';

  const isGold = pct >= 100;

  return (
    <div className={`relative inline-flex items-center justify-center ${isGold ? 'ring-glow-gold rounded-full' : ''}`}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke="#1e293b" strokeWidth="8"
        />
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke={color} strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-3xl font-black ${isGold ? 'text-yellow-400 glow-gold' : ''}`}>
          {pct.toFixed(0)}%
        </span>
        <span className="text-[10px] text-text-secondary uppercase tracking-wider">
          {pct >= 100 ? 'BONUS' : 'of target'}
        </span>
      </div>
    </div>
  );
}

function ModeBadge({ mode }: { mode: string }) {
  const config: Record<string, { icon: typeof Zap; color: string; bg: string }> = {
    normal: { icon: Zap, color: 'text-accent-blue', bg: 'bg-accent-blue/10' },
    aggressive: { icon: TrendingUp, color: 'text-accent-yellow', bg: 'bg-accent-yellow/10' },
    protecting: { icon: Shield, color: 'text-accent-green', bg: 'bg-accent-green/10' },
    halted: { icon: Pause, color: 'text-accent-red', bg: 'bg-accent-red/10' },
  };
  const c = config[mode] || config.normal;
  const Icon = c.icon;

  return (
    <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full ${c.bg} ${c.color} text-xs font-bold uppercase tracking-wider`}>
      <Icon className="w-3.5 h-3.5" />
      {mode}
    </div>
  );
}

function CalendarHeatmap({ history }: { history: any[] }) {
  if (!history || history.length === 0) return null;

  return (
    <div>
      <div className="text-[10px] text-text-secondary uppercase tracking-wider mb-2">Last 30 Days</div>
      <div className="flex flex-wrap gap-1">
        {history.slice(-30).map((day: any, i: number) => {
          const bg = day.target_hit
            ? day.exceeded ? 'bg-yellow-500' : 'bg-accent-green'
            : day.actual_pct > 0 ? 'bg-accent-yellow/60' : 'bg-accent-red/60';
          return (
            <div
              key={i}
              className={`w-3.5 h-3.5 rounded-sm ${bg} cursor-pointer`}
              title={`${day.date}: ${(day.actual_pct || 0).toFixed(2)}% (target: ${day.target_pct}%)`}
            />
          );
        })}
      </div>
    </div>
  );
}

function TimeRemainingBar() {
  const now = new Date();
  const utcH = now.getUTCHours();
  const utcM = now.getUTCMinutes();
  const elapsed = utcH * 60 + utcM;
  const total = 24 * 60;
  const pct = (elapsed / total) * 100;
  const hoursLeft = ((total - elapsed) / 60).toFixed(1);

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] text-text-secondary">
        <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> Trading Day</span>
        <span>{hoursLeft}h remaining</span>
      </div>
      <div className="w-full h-1.5 bg-bg-secondary rounded-full overflow-hidden">
        <div
          className="h-full bg-accent-blue/60 rounded-full transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function DailyTargetHero() {
  const { data: target } = useDailyTarget();
  const { data: status } = useStatus();
  const { data: historyData } = useDailyTargetHistory(30);
  const { data: projData } = useDailyTargetProjection();

  const progress = target?.progress;
  const pctAchieved = progress?.pct_achieved || 0;
  const mode = progress?.mode || 'normal';

  const equity = status?.shadow_equity || status?.equity_usd || status?.capital_usd || 0;
  const dayOpen = progress?.day_open_equity || equity;
  const targetPct = progress?.daily_target_pct || 2.0;
  const targetEquity = dayOpen * (1 + targetPct / 100);
  const remaining = Math.max(targetEquity - equity, 0);

  const history = historyData?.history || [];
  const projections = projData?.projections || {};
  const streak = projData?.streak_days || 0;

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5 fade-in">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Target className="w-5 h-5 text-accent-yellow" />
          <h2 className="text-sm font-bold tracking-wider text-text-primary uppercase">
            Daily Compound Target
          </h2>
        </div>
        <ModeBadge mode={mode} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Gauge */}
        <div className="lg:col-span-3 flex flex-col items-center justify-center gap-3">
          <CircularGauge pct={pctAchieved} />
          {streak > 0 && (
            <div className="text-xs text-accent-green font-medium">
              {streak} day streak
            </div>
          )}
        </div>

        {/* Center: Balance info */}
        <div className="lg:col-span-5 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-bg-secondary/50 rounded-lg p-3">
              <div className="text-[10px] text-text-secondary uppercase tracking-wider">Day-Open</div>
              <div className="text-sm font-bold text-text-primary">${dayOpen.toFixed(2)}</div>
              <div className="text-[10px] text-text-secondary">Rp {(dayOpen * IDR_RATE).toLocaleString()}</div>
            </div>
            <div className="bg-bg-secondary/50 rounded-lg p-3">
              <div className="text-[10px] text-text-secondary uppercase tracking-wider">Current</div>
              <div className={`text-sm font-bold ${equity >= dayOpen ? 'text-accent-green' : 'text-accent-red'}`}>
                ${equity.toFixed(2)}
              </div>
              <div className="text-[10px] text-text-secondary">Rp {(equity * IDR_RATE).toLocaleString()}</div>
            </div>
            <div className="bg-bg-secondary/50 rounded-lg p-3">
              <div className="text-[10px] text-text-secondary uppercase tracking-wider">Target</div>
              <div className="text-sm font-bold text-accent-yellow">${targetEquity.toFixed(2)}</div>
              <div className="text-[10px] text-text-secondary">(+{targetPct}%)</div>
            </div>
            <div className="bg-bg-secondary/50 rounded-lg p-3">
              <div className="text-[10px] text-text-secondary uppercase tracking-wider">Remaining</div>
              <div className={`text-sm font-bold ${remaining <= 0 ? 'text-accent-green' : 'text-accent-yellow'}`}>
                {remaining <= 0 ? 'TARGET HIT' : `$${remaining.toFixed(2)}`}
              </div>
              <div className="text-[10px] text-text-secondary">
                {remaining > 0 ? `Rp ${(remaining * IDR_RATE).toLocaleString()}` : 'Bonus territory'}
              </div>
            </div>
          </div>

          <TimeRemainingBar />
        </div>

        {/* Right: Calendar + Projections */}
        <div className="lg:col-span-4 space-y-4">
          <CalendarHeatmap history={history} />

          {Object.keys(projections).length > 0 && (
            <div>
              <div className="text-[10px] text-text-secondary uppercase tracking-wider mb-2">
                Compound Projection
              </div>
              <div className="space-y-1.5">
                {Object.entries(projections).map(([key, proj]: [string, any]) => (
                  <div key={key} className="flex items-center justify-between text-xs">
                    <span className="text-text-secondary">{proj.days}d</span>
                    <span className="text-text-primary font-medium">
                      ${proj.at_actual_rate?.toFixed(2)}
                    </span>
                    <span className="text-text-secondary">
                      (Rp {((proj.at_actual_rate || 0) * IDR_RATE).toLocaleString(undefined, { maximumFractionDigits: 0 })})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
