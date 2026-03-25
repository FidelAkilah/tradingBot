'use client';

import { useState } from 'react';
import { StatusBar } from '@/components/StatusBar';
import { EquityCard } from '@/components/EquityCard';
import { PnlChart } from '@/components/PnlChart';
import { PositionsPanel } from '@/components/PositionsPanel';
import { TradeHistory } from '@/components/TradeHistory';
import { SignalGauges } from '@/components/SignalGauges';
import { PerformanceStats } from '@/components/PerformanceStats';
import { DailyTargetHero } from '@/components/DailyTargetHero';
import { MarketRegimePanel } from '@/components/MarketRegimePanel';
import { SignalAnalytics } from '@/components/SignalAnalytics';
import { TradeJournal } from '@/components/TradeJournal';
import { LiveSignalMonitor } from '@/components/LiveSignalMonitor';
import { RiskDashboard } from '@/components/RiskDashboard';
import { PerformanceComparison } from '@/components/PerformanceComparison';
import { AILearning } from '@/components/AILearning';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'signals', label: 'Signals' },
  { id: 'trades', label: 'Trades' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'risk', label: 'Risk' },
  { id: 'ai', label: 'AI Learning' },
] as const;

type TabId = typeof TABS[number]['id'];

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabId>('overview');

  return (
    <div className="min-h-screen bg-bg-primary">
      <StatusBar />

      <main className="max-w-[1600px] mx-auto px-4 py-4">
        {/* Tab navigation */}
        <div className="flex items-center gap-1 mb-5 border-b border-border-dim overflow-x-auto">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`relative px-4 py-2.5 text-xs font-bold uppercase tracking-wider whitespace-nowrap transition-colors ${
                activeTab === tab.id
                  ? 'text-accent-blue'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              {tab.label}
              {activeTab === tab.id && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-accent-blue tab-active-bar" />
              )}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="space-y-4">
          {activeTab === 'overview' && <OverviewTab />}
          {activeTab === 'signals' && <SignalsTab />}
          {activeTab === 'trades' && <TradesTab />}
          {activeTab === 'analytics' && <AnalyticsTab />}
          {activeTab === 'risk' && <RiskTab />}
          {activeTab === 'ai' && <AITab />}
        </div>
      </main>
    </div>
  );
}

function OverviewTab() {
  return (
    <div className="space-y-4 fade-in">
      {/* Hero: Daily Target + Equity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <DailyTargetHero />
        </div>
        <EquityCard />
      </div>

      {/* P&L Chart */}
      <PnlChart />

      {/* Signal Gauges + Open Positions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <SignalGauges />
        <PositionsPanel />
      </div>

      {/* Performance Stats + Market Regime */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PerformanceStats />
        <MarketRegimePanel />
      </div>

      {/* Recent Trades */}
      <TradeHistory />
    </div>
  );
}

function SignalsTab() {
  return (
    <div className="space-y-4 fade-in">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <LiveSignalMonitor />
        <MarketRegimePanel />
      </div>
      <SignalAnalytics />
    </div>
  );
}

function TradesTab() {
  return (
    <div className="space-y-4 fade-in">
      <TradeJournal />
    </div>
  );
}

function AnalyticsTab() {
  return (
    <div className="space-y-4 fade-in">
      <PerformanceComparison />
      <SignalAnalytics />
    </div>
  );
}

function RiskTab() {
  return (
    <div className="space-y-4 fade-in">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <RiskDashboard />
        <div className="space-y-4">
          <MarketRegimePanel />
          <PositionsPanel />
        </div>
      </div>
    </div>
  );
}

function AITab() {
  return (
    <div className="space-y-4 fade-in">
      <AILearning />
    </div>
  );
}
