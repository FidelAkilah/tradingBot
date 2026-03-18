'use client';

import { useStatus } from '@/lib/hooks';
import { Activity, Wifi, WifiOff, Zap } from 'lucide-react';

export function StatusBar() {
  const { data, error } = useStatus();

  // If we have data, the API is reachable — we're connected
  const apiReachable = !!data;
  const isRunning = data?.status === 'running';
  const isShadow = data?.is_shadow;

  return (
    <header className="bg-bg-secondary border-b border-border-dim px-4 py-2 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <Zap className="w-5 h-5 text-accent-yellow" />
        <span className="text-sm font-bold tracking-wider text-text-primary">
          SWING BOT
        </span>
        {isShadow && (
          <span className="px-2 py-0.5 text-[10px] bg-accent-purple/20 text-accent-purple rounded-full font-bold tracking-wider">
            SHADOW
          </span>
        )}
        {!isShadow && isRunning && (
          <span className="px-2 py-0.5 text-[10px] bg-accent-green/20 text-accent-green rounded-full font-bold tracking-wider">
            LIVE
          </span>
        )}
      </div>

      <div className="flex items-center gap-4 text-xs text-text-secondary">
        {data?.pairs?.map((pair: string) => (
          <span key={pair} className="text-text-secondary/70">
            {pair.replace('/USDT', '')}
          </span>
        ))}

        <span className="text-text-secondary/50">|</span>

        <span>{data?.leverage || 30}x</span>
        <span>{data?.margin_type || 'CROSSED'}</span>

        <span className="text-text-secondary/50">|</span>

        <div className="flex items-center gap-1.5">
          {apiReachable && isRunning ? (
            <>
              <div className="w-2 h-2 bg-accent-green rounded-full pulse-dot" />
              <Wifi className="w-3.5 h-3.5 text-accent-green" />
              <span className="text-accent-green">Connected</span>
            </>
          ) : apiReachable ? (
            <>
              <Activity className="w-3.5 h-3.5 text-accent-yellow" />
              <span className="text-accent-yellow">{data?.status || 'Starting...'}</span>
            </>
          ) : error ? (
            <>
              <WifiOff className="w-3.5 h-3.5 text-accent-red" />
              <span className="text-accent-red">Disconnected</span>
            </>
          ) : (
            <>
              <Activity className="w-3.5 h-3.5 text-accent-yellow" />
              <span className="text-accent-yellow">Connecting...</span>
            </>
          )}
        </div>

        {data?.uptime_human && (
          <span className="text-text-secondary/50">
            uptime: {data.uptime_human}
          </span>
        )}
      </div>
    </header>
  );
}
