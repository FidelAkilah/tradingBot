'use client';

import { StatusBar } from '@/components/StatusBar';
import { EquityCard } from '@/components/EquityCard';
import { PnlChart } from '@/components/PnlChart';
import { PositionsPanel } from '@/components/PositionsPanel';
import { TradeHistory } from '@/components/TradeHistory';
import { SignalGauges } from '@/components/SignalGauges';
import { PerformanceStats } from '@/components/PerformanceStats';

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-bg-primary">
      {/* Top status bar */}
      <StatusBar />

      <main className="max-w-[1600px] mx-auto px-4 py-4 space-y-4">
        {/* Row 1: Equity + Performance Stats */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <EquityCard />
          <div className="lg:col-span-2">
            <PerformanceStats />
          </div>
        </div>

        {/* Row 2: P&L Chart (full width) */}
        <PnlChart />

        {/* Row 3: Signal Gauges + Open Positions */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <SignalGauges />
          <PositionsPanel />
        </div>

        {/* Row 4: Trade History */}
        <TradeHistory />
      </main>
    </div>
  );
}
