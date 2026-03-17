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
