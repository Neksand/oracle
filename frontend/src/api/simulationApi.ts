/**
 * ORACLE - API layer: finance simulation, portfolio, market data,
 * persistent database layer, auth, security demo, research benchmarks.
 */

import axios from 'axios';
import { FinanceParams, FinanceResponse, PortfolioParams, MarketQuote, MarketPreset } from '../types';

// Relative by default so the same build works locally and through the
// demo tunnel (Vite proxies /api -> backend). Override with VITE_API_BASE_URL.
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '/api';

/** Bearer token kept in memory + localStorage (demo token store). */
let authToken: string | null = localStorage.getItem('oracle_token');
export function setAuthToken(token: string | null): void {
  authToken = token;
  if (token) localStorage.setItem('oracle_token', token);
  else localStorage.removeItem('oracle_token');
}
export function getAuthToken(): string | null {
  return authToken;
}
function authHeaders(): Record<string, string> {
  return authToken ? { Authorization: `Bearer ${authToken}` } : {};
}

export async function runFinanceSimulation(params: FinanceParams, seed?: number | null): Promise<FinanceResponse> {
  try {
    const payload = {
      symbol: params.symbol,
      s0: params.s0,
      mu_ann: params.mu_ann,
      sigma_ann: params.sigma_ann,
      t_days: params.t_days,
      trading_days: params.trading_days,
      num_paths: params.num_paths,
      risk_free: params.risk_free,
      strike: params.strike ?? null,
      option_type: params.option_type,
      seed: seed ?? null,
    };
    const response = await axios.post<FinanceResponse>(
      `${API_BASE_URL}/finance/simulate`, payload,
      { headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, timeout: 60000 },
    );
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const detail = error.response?.data?.detail || error.message || 'Unknown network error';
      throw new Error(`Scenario computation failed: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);
    }
    throw error;
  }
}

export async function runPortfolioSimulation(params: PortfolioParams, seed?: number | null): Promise<FinanceResponse> {
  try {
    const payload = {
      assets: params.assets.map((a) => ({
        symbol: a.symbol, s0: a.s0, mu_ann: a.mu_ann, sigma_ann: a.sigma_ann, weight: a.weight,
      })),
      corr_level: params.corr_level,
      t_days: params.t_days,
      trading_days: params.trading_days,
      num_paths: params.num_paths,
      risk_free: params.risk_free,
      seed: seed ?? null,
    };
    const response = await axios.post<FinanceResponse>(
      `${API_BASE_URL}/finance/portfolio`, payload,
      { headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, timeout: 60000 },
    );
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const detail = error.response?.data?.detail || error.message || 'Unknown network error';
      throw new Error(`Portfolio computation failed: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);
    }
    throw error;
  }
}

export async function fetchHealth(): Promise<{ online: boolean; system?: string }> {
  try {
    const base = API_BASE_URL.replace(/\/api$/, '');
    const r = await axios.get<{ status: string; system: string }>(`${base}/health`, { timeout: 60000 });
    return { online: r.data?.status === 'online', system: r.data?.system };
  } catch {
    return { online: false };
  }
}

export async function fetchQuote(symbol: string): Promise<MarketQuote> {
  const r = await axios.get<MarketQuote>(`${API_BASE_URL}/market/quote`, { params: { symbol }, timeout: 60000 });
  return r.data;
}

export async function fetchHistory(symbol: string, range = '1Y'): Promise<{ closes: number[]; source: string; symbol: string }> {
  const r = await axios.get(`${API_BASE_URL}/market/history`, { params: { symbol, range }, timeout: 60000 });
  return r.data;
}

export async function calibrateMarket(symbol: string): Promise<{ s0: number; mu_ann: number; sigma_ann: number; source: string; n_obs: number }> {
  const r = await axios.post(`${API_BASE_URL}/market/calibrate`, { symbol }, { timeout: 60000 });
  return r.data;
}

export async function fetchPresets(): Promise<MarketPreset[]> {
  const r = await axios.get<MarketPreset[]>(`${API_BASE_URL}/market/presets`, { timeout: 60000 });
  return r.data;
}

/** Export scenario trajectories + histogram as CSV (price space). */
export function exportToCSV(data: FinanceResponse, filename = 'oracle_scenario.csv'): void {
  const { sample_paths, time_axis_days, stats } = data;
  const header = 'path_idx,' + time_axis_days.map((t) => `day_${t}`).join(',') + '\n';
  const rows = sample_paths.map((p, i) => `${i},${p.map((v) => v.toFixed(2)).join(',')}`).join('\n');
  const footer = `\n# expected_ST=${stats.expected_ST}, p5=${stats.p5_ST}, p95=${stats.p95_ST}, VaR95_loss=${stats.var95_loss}, CVaR95=${stats.cvar95_loss}\n`;
  const blob = new Blob([header + rows + footer], { type: 'text/csv;charset=utf-8;' });
  triggerDownload(blob, filename.endsWith('.csv') ? filename : `${filename}.csv`);
}

export function exportToJSON(data: FinanceResponse, filename = 'oracle_scenario.json'): void {
  const slim = { ...data, positions: `[${data.positions.length} floats omitted]`, colors: `[${data.colors.length} floats omitted]` };
  const blob = new Blob([JSON.stringify(slim, null, 2)], { type: 'application/json;charset=utf-8;' });
  triggerDownload(blob, filename.endsWith('.json') ? filename : `${filename}.json`);
}

export function buildReportText(data: FinanceResponse): string {
  const s = data.stats;
  const m = data.metadata;
  const sym = m.symbol ?? (m.symbols ?? []).join('+') ?? 'PORT';
  return [
    `ORACLE — Scenario report: ${sym}`,
    `Horizon: ${m.t_days} days (${m.t_years ?? ''}y), paths: ${m.total_points}`,
    `Current level S0: ${m.s0 ?? m.s0_port}`,
    `Expected terminal value: ${s.expected_ST} | Median: ${s.median_ST}`,
    `90% corridor (P5–P95): [${s.p5_ST}, ${s.p95_ST}]`,
    `VaR95 (loss): ${s.var95_loss} | CVaR95: ${s.cvar95_loss} | VaR99: ${s.var99_loss}`,
    `P(below S0): ${(s.prob_decline * 100).toFixed(1)}% | P(loss >10%): ${(s.prob_loss_gt10 * 100).toFixed(1)}%`,
    s.call_price_bs != null ? `Call BS: ${s.call_price_bs} (MC: ${s.call_price_mc}) | Put BS: ${s.put_price_bs} (MC: ${s.put_price_mc})` : '',
    (m as { diversification_benefit?: number }).diversification_benefit != null
      ? `Diversification benefit: ${(m as { diversification_benefit?: number }).diversification_benefit}` : '',
  ].filter(Boolean).join('\n');
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Database layer (Assignment 3)
// ---------------------------------------------------------------------------
export interface ScenarioLookupRow {
  id: number; name: string; description: string; horizon_days: number;
  num_paths: number; owner: string; created_at: string | null;
}
export interface RiskJoinRow {
  scenario_id: number; scenario: string; n_assets: number; n_runs: number;
  best_expected_st: number; avg_var95_loss: number; worst_cvar95: number;
}

export async function dbLookupScenarios(username = 'alice'):
  Promise<{ query: string; count: number; scenarios: ScenarioLookupRow[] }> {
  const r = await axios.get(`${API_BASE_URL}/db/scenarios/lookup`,
    { params: { username }, headers: authHeaders(), timeout: 60000 });
  return r.data;
}

export async function dbRiskJoin(): Promise<{ query: string; rows: RiskJoinRow[] }> {
  const r = await axios.get(`${API_BASE_URL}/db/scenarios/risk-join`,
    { headers: authHeaders(), timeout: 60000 });
  return r.data;
}

export interface SavedScenario {
  id: number; name: string; horizon_days: number;
  assets: { symbol: string; weight: number }[];
}

export async function dbMyScenarios(): Promise<SavedScenario[]> {
  const r = await axios.get(`${API_BASE_URL}/db/scenarios`,
    { headers: authHeaders(), timeout: 60000 });
  return r.data;
}

export interface AuthResult { access_token: string; role: string; username: string }

export async function login(username: string, password: string): Promise<AuthResult> {
  const form = new URLSearchParams();
  form.append('username', username);
  form.append('password', password);
  const r = await axios.post(`${API_BASE_URL}/auth/login`, form,
    { headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, timeout: 60000 });
  setAuthToken(r.data.access_token);
  localStorage.setItem('oracle_user', r.data.username ?? username);
  return r.data;
}

export async function register(username: string, email: string, password: string): Promise<AuthResult> {
  const r = await axios.post(`${API_BASE_URL}/auth/register`,
    { username, email, password }, { timeout: 60000 });
  setAuthToken(r.data.access_token);
  localStorage.setItem('oracle_user', r.data.username ?? username);
  return r.data;
}

export async function me(): Promise<{ username: string; email: string; role: string }> {
  const r = await axios.get(`${API_BASE_URL}/auth/me`, { headers: authHeaders(), timeout: 60000 });
  return r.data;
}

export function logout(): void {
  setAuthToken(null);
  localStorage.removeItem('oracle_user');
}

export function currentUser(): string | null {
  return localStorage.getItem('oracle_user');
}

// ---------------------------------------------------------------------------
// API keys (programmatic access)
// ---------------------------------------------------------------------------
export interface ApiKeyInfo {
  id: number; name: string; prefix: string;
  created_at: string; last_used_at: string | null; revoked: boolean;
}

export async function listApiKeys(): Promise<ApiKeyInfo[]> {
  const r = await axios.get(`${API_BASE_URL}/keys`, { headers: authHeaders(), timeout: 60000 });
  return r.data;
}

export async function createApiKey(name: string): Promise<{ id: number; name: string; prefix: string; api_key: string; note: string }> {
  const r = await axios.post(`${API_BASE_URL}/keys`, { name }, { headers: authHeaders(), timeout: 60000 });
  return r.data;
}

export async function revokeApiKey(id: number): Promise<void> {
  await axios.delete(`${API_BASE_URL}/keys/${id}`, { headers: authHeaders(), timeout: 60000 });
}

export function apiErrorMessage(e: unknown): string {
  if (axios.isAxiosError(e)) {
    const detail = (e.response?.data as { detail?: unknown } | undefined)?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail) && detail.length && typeof (detail[0] as { msg?: string }).msg === 'string') {
      return (detail[0] as { msg: string }).msg;
    }
    if (e.response?.status === 429) return 'Too many attempts. Please wait and try again.';
    if (e.code === 'ECONNABORTED' || !e.response) {
      return 'Backend is waking up (free tier, up to ~40s). Wait a moment and press Sign in again.';
    }
  }
  return e instanceof Error ? e.message : String(e);
}

// ---------------------------------------------------------------------------
// Security showcase (SQLi demo + audit log)
// ---------------------------------------------------------------------------
export interface InjectionResult {
  provided_payload: string;
  unsafe: { version: string; executed_sql: string; rows_leaked: number; scenario_names: string[] };
  safe: { version: string; executed_sql: string; rows_leaked: number; scenario_names: string[] };
  verdict: string;
  lesson: string[];
}

export async function runInjectionDemo(userInput: string): Promise<InjectionResult> {
  const r = await axios.post(`${API_BASE_URL}/security/sql_injection_demo`,
    { user_input: userInput }, { headers: authHeaders(), timeout: 60000 });
  return r.data;
}

export interface AuditEntry {
  id: number; event_type: string; actor: string; detail: string; blocked: boolean; created_at: string | null;
}

export async function fetchAuditLog(limit = 15): Promise<AuditEntry[]> {
  const r = await axios.get(`${API_BASE_URL}/security/audit_log`,
    { params: { limit }, headers: authHeaders(), timeout: 60000 });
  return r.data;
}

// ---------------------------------------------------------------------------
// Research benchmarks
// ---------------------------------------------------------------------------
export interface BenchmarkResponse {
  data: { symbol: string; source: string; n_obs: number; first_date: string | null; last_close: number };
  calibration: { mu_ann: number; sigma_ann: number; loglik: number };
  diagnostics: {
    ks_test: Record<string, unknown>;
    qq_points: { p: number; theoretical_normal: number; theoretical_student_t: number; empirical: number }[];
    kupiec_var95: Record<string, unknown>;
  };
  terminal_distribution_comparison: Record<string, number | string>[];
  vol_forecast_rmse: { winner?: string | null; rows: { model: string; rmse: number }[]; note?: string; lstm_available?: boolean };
  execution_time_ms: number;
  methodology: string[];
}

export async function runBenchmark(symbol = 'SPY', useLstm = true): Promise<BenchmarkResponse> {
  const r = await axios.post(`${API_BASE_URL}/research/benchmark`,
    { symbol, lookback_days: 750, use_lstm: useLstm }, { headers: authHeaders(), timeout: 180000 });
  return r.data;
}

/** Download the .sql dump produced server-side (appendix for the report). */
export async function exportSqlDump(): Promise<void> {
  const r = await axios.post(`${API_BASE_URL}/db/export`, null,
    { headers: authHeaders(), timeout: 60000 });
  const blob = new Blob([r.data.sql], { type: 'application/sql;charset=utf-8;' });
  triggerDownload(blob, r.data.filename || 'oracle_dump.sql');
}
