/**
 * ORACLE - useSimulation: finance scenario state, market data, persistence.
 * Fixes: applied-request tracking (dirty detection), deterministic seed lock,
 * backend health polling, run counter for provenance.
 */
import { useState, useCallback, useEffect, useRef } from 'react';
import { FinanceParams, FinanceResponse, SavedScenario, PortfolioParams, MarketQuote } from '../types';
import { runFinanceSimulation, runPortfolioSimulation, fetchQuote, fetchHistory, calibrateMarket, fetchHealth } from '../api/simulationApi';

const STORAGE_KEY = 'oracle_saved_scenarios';

export const DEFAULT_FINANCE_PARAMS: FinanceParams = {
  symbol: 'SPY',
  s0: 585,
  mu_ann: 0.09,
  sigma_ann: 0.17,
  t_days: 90,
  trading_days: 252,
  num_paths: 8000,
  risk_free: 0.04,
  strike: 600,
  option_type: 'call',
};

export type AppliedRequest =
  | { kind: 'asset'; params: FinanceParams; seed: number | null }
  | { kind: 'portfolio'; req: PortfolioParams; seed: number | null }
  | null;

/** Stable numeric seed from a string (djb2). Same inputs -> same seed. */
export function hashToSeed(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = (Math.imul(h, 33) ^ s.charCodeAt(i)) >>> 0;
  return h % 1_000_000_000;
}

const randomSeed = () => Math.floor(Math.random() * 1_000_000_000);

export interface UseSimulationReturn {
  loading: boolean;
  error: string | null;
  data: FinanceResponse | null;
  params: FinanceParams;
  setParams: (p: FinanceParams) => void;
  quote: MarketQuote | null;
  history: number[] | null;
  historySource: string;
  marketLoading: boolean;
  savedScenarios: SavedScenario[];
  applied: AppliedRequest;
  runCount: number;
  seedLocked: boolean;
  setSeedLocked: (v: boolean) => void;
  lastSeed: number | null;
  backendOnline: boolean | null;
  backendSystem: string;
  simulate: (p?: FinanceParams, seedOverride?: number | null) => Promise<void>;
  simulatePortfolio: (p: PortfolioParams) => Promise<void>;
  reroll: () => Promise<void>;
  loadQuote: (symbol: string) => Promise<void>;
  loadHistory: (symbol: string, range?: string) => Promise<void>;
  autoCalibrate: (symbol: string) => Promise<void>;
  saveScenario: (name?: string) => void;
  clearSavedScenarios: () => void;
  removeScenario: (id: string) => void;
}

export function useSimulation(): UseSimulationReturn {
  const [loading, setLoading] = useState(false);
  const [marketLoading, setMarketLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<FinanceResponse | null>(null);
  const [params, setParams] = useState<FinanceParams>(DEFAULT_FINANCE_PARAMS);
  const [quote, setQuote] = useState<MarketQuote | null>(null);
  const [history, setHistory] = useState<number[] | null>(null);
  const [historySource, setHistorySource] = useState('');
  const [applied, setApplied] = useState<AppliedRequest>(null);
  const [runCount, setRunCount] = useState(0);
  const [seedLocked, setSeedLocked] = useState(true);
  const [lastSeed, setLastSeed] = useState<number | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [backendSystem, setBackendSystem] = useState('');
  const lastPortfolio = useRef<PortfolioParams | null>(null);
  const [savedScenarios, setSavedScenarios] = useState<SavedScenario[]>(() => {
    try {
      const s = localStorage.getItem(STORAGE_KEY);
      return s ? (JSON.parse(s) as SavedScenario[]) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(savedScenarios));
    } catch { /* ignore */ }
  }, [savedScenarios]);

  // Backend health polling (item 6)
  useEffect(() => {
    let alive = true;
    const check = async () => {
      const h = await fetchHealth();
      if (!alive) return;
      setBackendOnline(h.online);
      setBackendSystem(h.system ?? '');
    };
    void check();
    const id = setInterval(check, 15000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const simulate = useCallback(async (p?: FinanceParams, seedOverride?: number | null) => {
    const eff = p ?? params;
    const seed = seedOverride ?? (seedLocked
      ? hashToSeed(JSON.stringify([
        eff.symbol, eff.s0, eff.mu_ann, eff.sigma_ann, eff.t_days,
        eff.trading_days, eff.num_paths, eff.risk_free, eff.strike, eff.option_type,
      ]))
      : randomSeed());
    setLoading(true);
    setError(null);
    try {
      const res = await runFinanceSimulation(eff, seed);
      setData(res);
      setParams(eff);
      setApplied({ kind: 'asset', params: { ...eff }, seed: res.metadata.seed ?? seed });
      setLastSeed(res.metadata.seed ?? seed);
      setRunCount((c) => c + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unexpected error');
    } finally {
      setLoading(false);
    }
  }, [params, seedLocked]);

  const simulatePortfolio = useCallback(async (p: PortfolioParams) => {
    setLoading(true);
    setError(null);
    try {
      const seed = seedLocked
        ? hashToSeed(JSON.stringify([p.assets, p.corr_level, p.t_days, p.num_paths, p.risk_free]))
        : randomSeed();
      const res = await runPortfolioSimulation(p, seed);
      setData(res);
      lastPortfolio.current = p;
      setApplied({ kind: 'portfolio', req: JSON.parse(JSON.stringify(p)) as PortfolioParams, seed });
      setLastSeed(seed);
      setRunCount((c) => c + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unexpected error');
    } finally {
      setLoading(false);
    }
  }, [seedLocked]);

  const reroll = useCallback(async () => {
    if (applied?.kind === 'portfolio' && lastPortfolio.current) {
      // fresh random draw of the same portfolio inputs
      setLoading(true);
      setError(null);
      try {
        const seed = randomSeed();
        const res = await runPortfolioSimulation(lastPortfolio.current, seed);
        setData(res);
        setApplied({ kind: 'portfolio', req: lastPortfolio.current, seed });
        setLastSeed(seed);
        setRunCount((c) => c + 1);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Unexpected error');
      } finally {
        setLoading(false);
      }
      return;
    }
    // same asset inputs, new random draw
    const eff = applied?.kind === 'asset' ? applied.params : params;
    await simulate(eff, randomSeed());
  }, [applied, params, simulate]);

  const loadQuote = useCallback(async (symbol: string) => {
    setMarketLoading(true);
    try {
      const q = await fetchQuote(symbol);
      setQuote(q);
      setParams((prev) => ({ ...prev, symbol: q.symbol.toUpperCase(), s0: q.price }));
    } catch (e) {
      setError(e instanceof Error ? `Market: ${e.message}` : 'Market error');
    } finally {
      setMarketLoading(false);
    }
  }, []);

  const loadHistory = useCallback(async (symbol: string, range = '1Y') => {
    try {
      const h = await fetchHistory(symbol, range);
      setHistory(h.closes);
      setHistorySource(h.source);
    } catch { /* keep silent, chart hides */ }
  }, []);

  const autoCalibrate = useCallback(async (symbol: string) => {
    setMarketLoading(true);
    try {
      const c = await calibrateMarket(symbol);
      setParams((prev) => ({
        ...prev, symbol: symbol.toUpperCase(), s0: c.s0, mu_ann: c.mu_ann, sigma_ann: c.sigma_ann,
        trading_days: symbol.includes('BTC') || symbol.includes('ETH') ? 365 : 252,
      }));
      await loadHistory(symbol);
    } catch (e) {
      setError(e instanceof Error ? `Calibration: ${e.message}` : 'Calibration error');
    } finally {
      setMarketLoading(false);
    }
  }, [loadHistory]);

  const saveScenario = useCallback((name?: string) => {
    if (!data) return;
    const sc: SavedScenario = {
      id: `scen_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      name: name || `${data.metadata.symbol ?? 'PORT'} · ${data.metadata.t_days}d · VaR $${data.stats.var95_loss}`,
      params: { ...params },
      stats: data.stats,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    setSavedScenarios((prev) => [sc, ...prev].slice(0, 30));
  }, [data, params]);

  const clearSavedScenarios = useCallback(() => {
    setSavedScenarios([]);
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
  }, []);

  const removeScenario = useCallback((id: string) => {
    setSavedScenarios((prev) => prev.filter((s) => s.id !== id));
  }, []);

  return {
    loading, error, data, params, setParams, quote, history, historySource,
    marketLoading, savedScenarios, applied, runCount, seedLocked, setSeedLocked,
    lastSeed, backendOnline, backendSystem,
    simulate, simulatePortfolio, reroll,
    loadQuote, loadHistory, autoCalibrate, saveScenario, clearSavedScenarios, removeScenario,
  };
}
