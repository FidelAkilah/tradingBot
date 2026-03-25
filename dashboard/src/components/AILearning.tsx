'use client';

import { useLearning, useAdvisorStats, useAdvisorKBPerformance } from '@/lib/hooks';
import { Brain, Lightbulb, BookOpen, Zap, CheckCircle, XCircle, AlertTriangle, TrendingUp } from 'lucide-react';

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-bg-secondary/30 rounded-lg border border-border-dim/30 p-3 text-center">
      <div className="text-lg font-bold text-text-primary">{value}</div>
      <div className="text-[10px] text-text-secondary uppercase mt-0.5">{label}</div>
      {sub && <div className="text-[9px] text-text-secondary mt-0.5">{sub}</div>}
    </div>
  );
}

function KBEntryRow({ entry }: { entry: any }) {
  const statusColor = entry.status === 'boosted' ? 'text-accent-green' :
    entry.status === 'flagged' ? 'text-accent-red' : 'text-text-secondary';
  const StatusIcon = entry.status === 'boosted' ? TrendingUp :
    entry.status === 'flagged' ? AlertTriangle : CheckCircle;

  const content = typeof entry.content === 'object' ? entry.content : {};
  const summary = Object.values(content).filter(Boolean).slice(0, 2).join(' — ');

  return (
    <div className="flex items-center gap-3 p-2 bg-bg-secondary/20 rounded-lg border border-border-dim/20">
      <StatusIcon className={`w-3.5 h-3.5 flex-shrink-0 ${statusColor}`} />
      <div className="flex-1 min-w-0">
        <div className="text-[11px] text-text-primary truncate">{summary || entry.category}</div>
        <div className="text-[9px] text-text-secondary flex gap-2 mt-0.5">
          <span className="capitalize">{entry.category}</span>
          <span>{entry.source_title}</span>
        </div>
      </div>
      <div className="flex-shrink-0 text-right">
        <div className="text-[10px] font-medium text-text-primary">{entry.times_applied}x</div>
        <div className={`text-[9px] font-medium ${
          (entry.success_rate || 0) >= 0.65 ? 'text-accent-green' :
          (entry.success_rate || 0) < 0.30 ? 'text-accent-red' : 'text-text-secondary'
        }`}>
          {((entry.success_rate || 0) * 100).toFixed(0)}% win
        </div>
      </div>
    </div>
  );
}

export function AILearning() {
  const { data: learning, isLoading: loadingLearning } = useLearning();
  const { data: advisorStats, isLoading: loadingAdvisor } = useAdvisorStats();
  const { data: kbPerf, isLoading: loadingKB } = useAdvisorKBPerformance();

  const stats = advisorStats || {};
  const entries = kbPerf?.entries || [];
  const kbEntries = learning?.total_entries || 0;
  const recentIngestions = learning?.recent_ingestions || [];
  const isLoading = loadingLearning || loadingAdvisor || loadingKB;

  const hasData = stats.total_consultations > 0 || kbEntries > 0;

  return (
    <div className="bg-bg-card rounded-xl border border-border-dim p-5 fade-in space-y-5">
      <div className="flex items-center gap-2">
        <Brain className="w-4 h-4 text-accent-purple" />
        <h2 className="text-xs font-bold tracking-wider text-text-secondary uppercase">
          AI Learning & Advisor
        </h2>
        {hasData && (
          <span className="text-[9px] bg-accent-green/10 text-accent-green font-bold px-1.5 py-0.5 rounded uppercase">
            Active
          </span>
        )}
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3].map(i => <div key={i} className="h-16 bg-bg-secondary/30 rounded-lg shimmer" />)}
        </div>
      ) : !hasData ? (
        <div className="text-center py-10 space-y-4">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-accent-purple/10">
            <Brain className="w-7 h-7 text-accent-purple/60" />
          </div>
          <div>
            <div className="text-sm font-medium text-text-primary">AI Learning Not Yet Active</div>
            <div className="text-[11px] text-text-secondary mt-1.5 max-w-md mx-auto leading-relaxed">
              Ingest trading knowledge to enable the AI advisor. Run:
              <code className="block mt-1 bg-bg-secondary/50 px-2 py-1 rounded text-[10px] font-mono">
                python -m ai_learning ingest-youtube &lt;URL&gt;
              </code>
            </div>
          </div>
        </div>
      ) : (
        <>
          {/* Advisor Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard
              label="KB Entries"
              value={kbEntries}
            />
            <StatCard
              label="Consultations"
              value={stats.total_consultations || 0}
              sub={`${stats.calls_today || 0}/${stats.daily_budget || 100} today`}
            />
            <StatCard
              label="Agreement Rate"
              value={`${((stats.agreement_rate || 0) * 100).toFixed(0)}%`}
              sub={`${stats.agreements || 0} agree, ${stats.overrides || 0} override`}
            />
            <StatCard
              label="Cache Hits"
              value={stats.cache_size || 0}
              sub="active cache entries"
            />
          </div>

          {/* Recent Consultations */}
          {stats.recent_consultations?.length > 0 && (
            <div>
              <div className="text-[10px] text-text-secondary uppercase mb-2 font-bold">Recent Consultations</div>
              <div className="space-y-1.5 max-h-48 overflow-y-auto">
                {stats.recent_consultations.map((rec: any, i: number) => (
                  <div key={i} className="flex items-center gap-2 p-2 bg-bg-secondary/20 rounded text-[10px]">
                    <span className={`font-bold px-1 py-0.5 rounded ${
                      rec.recommendation === 'PROCEED' ? 'bg-accent-green/20 text-accent-green' :
                      rec.recommendation === 'SKIP' ? 'bg-accent-red/20 text-accent-red' :
                      'bg-accent-gold/20 text-accent-gold'
                    }`}>
                      {rec.recommendation}
                    </span>
                    <span className="text-text-primary font-medium">{rec.symbol}</span>
                    {rec.confidence_adjustment !== 0 && (
                      <span className={rec.confidence_adjustment > 0 ? 'text-accent-green' : 'text-accent-red'}>
                        {rec.confidence_adjustment > 0 ? '+' : ''}{(rec.confidence_adjustment * 100).toFixed(0)}%
                      </span>
                    )}
                    <span className="text-text-secondary truncate flex-1">{rec.reasoning}</span>
                    {rec.trade_outcome && (
                      <span className={`font-bold ${rec.trade_outcome === 'win' ? 'text-accent-green' : 'text-accent-red'}`}>
                        {rec.trade_outcome === 'win' ? 'W' : 'L'}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* KB Performance */}
          {entries.length > 0 && (
            <div>
              <div className="text-[10px] text-text-secondary uppercase mb-2 font-bold">Knowledge Base Performance</div>
              <div className="space-y-1.5 max-h-64 overflow-y-auto">
                {entries.map((entry: any, i: number) => (
                  <KBEntryRow key={entry.id || i} entry={entry} />
                ))}
              </div>
            </div>
          )}

          {/* Recent Ingestions */}
          {recentIngestions.length > 0 && (
            <div>
              <div className="text-[10px] text-text-secondary uppercase mb-2 font-bold">Recent Ingestions</div>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {recentIngestions.map((ing: any, i: number) => (
                  <div key={i} className="flex items-center gap-2 p-1.5 text-[10px]">
                    {ing.status === 'success' ? (
                      <CheckCircle className="w-3 h-3 text-accent-green flex-shrink-0" />
                    ) : (
                      <XCircle className="w-3 h-3 text-accent-red flex-shrink-0" />
                    )}
                    <span className="capitalize text-text-secondary">{ing.source_type}</span>
                    <span className="text-text-primary truncate flex-1">{ing.url || ing.source_url}</span>
                    <span className="text-text-secondary">{ing.entries_added || 0} entries</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
