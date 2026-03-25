import useSWR from 'swr';
import { getAPIUrl } from './api';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export function useStatus() {
  return useSWR(getAPIUrl('/api/status'), fetcher, { refreshInterval: 3000 });
}

export function useTrades(limit = 50) {
  return useSWR(getAPIUrl(`/api/trades?limit=${limit}`), fetcher, { refreshInterval: 5000 });
}

export function usePositions() {
  return useSWR(getAPIUrl('/api/positions'), fetcher, { refreshInterval: 3000 });
}

export function usePerformance() {
  return useSWR(getAPIUrl('/api/performance'), fetcher, { refreshInterval: 10000 });
}

export function usePnlChart() {
  return useSWR(getAPIUrl('/api/pnl-chart'), fetcher, { refreshInterval: 10000 });
}

export function useSignals(limit = 100) {
  return useSWR(getAPIUrl(`/api/signals?limit=${limit}`), fetcher, { refreshInterval: 5000 });
}

export function usePrices() {
  return useSWR(getAPIUrl('/api/prices'), fetcher, { refreshInterval: 5000 });
}

// ── New hooks for dashboard overhaul ──────────────────────────

export function useDailyTarget() {
  return useSWR(getAPIUrl('/api/daily-target'), fetcher, { refreshInterval: 3000 });
}

export function useDailyTargetHistory(days = 30) {
  return useSWR(getAPIUrl(`/api/daily-target/history?days=${days}`), fetcher, { refreshInterval: 60000 });
}

export function useDailyTargetProjection() {
  return useSWR(getAPIUrl('/api/daily-target/projection'), fetcher, { refreshInterval: 30000 });
}

export function useRegime() {
  return useSWR(getAPIUrl('/api/regime'), fetcher, { refreshInterval: 5000 });
}

export function useWinrateAnalytics() {
  return useSWR(getAPIUrl('/api/analytics/winrate'), fetcher, { refreshInterval: 15000 });
}

export function useRiskAnalytics() {
  return useSWR(getAPIUrl('/api/analytics/risk'), fetcher, { refreshInterval: 5000 });
}

export function useAdvancedPerformance() {
  return useSWR(getAPIUrl('/api/analytics/performance'), fetcher, { refreshInterval: 15000 });
}

export function useCorrelation() {
  return useSWR(getAPIUrl('/api/correlation'), fetcher, { refreshInterval: 10000 });
}

export function useExposure() {
  return useSWR(getAPIUrl('/api/exposure'), fetcher, { refreshInterval: 5000 });
}

export function useDailyEquity(limit = 90) {
  return useSWR(getAPIUrl(`/api/daily-equity?limit=${limit}`), fetcher, { refreshInterval: 30000 });
}

export function useLearning() {
  return useSWR(getAPIUrl('/api/learning/recent'), fetcher, { refreshInterval: 60000 });
}

export function useAdvisorStats() {
  return useSWR(getAPIUrl('/api/advisor/stats'), fetcher, { refreshInterval: 30000 });
}

export function useAdvisorKBPerformance() {
  return useSWR(getAPIUrl('/api/advisor/kb-performance'), fetcher, { refreshInterval: 60000 });
}
