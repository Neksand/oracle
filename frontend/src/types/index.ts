/**
 * ORACLE - Core Type Definitions
 * Economic Scenario Engine: GBM single-asset, portfolio, options, market data.
 * Legacy NEXUS types kept for backward compatibility.
 */

// --- Legacy ---
export interface SimulationParams {
  mu: number;
  sigma: number;
  num_paths: number;
  steps: number;
}

export interface SimulationStatistics {
  mean: number[];
  std: number[];
  percentile_5: number[];
  percentile_95: number[];
  var_95: number;
  var_99: number;
}

// --- Finance domain ---
export interface FinanceParams {
  symbol: string;
  s0: number;
  mu_ann: number; // 0.08 = 8%/year
  sigma_ann: number;
  t_days: number;
  trading_days: number;
  num_paths: number;
  risk_free: number;
  strike?: number;
  option_type: 'call' | 'put';
}

export interface PortfolioAsset {
  symbol: string;
  s0: number;
  mu_ann: number;
  sigma_ann: number;
  weight: number; // 0..1
}

export interface PortfolioParams {
  assets: PortfolioAsset[];
  corr_level: number;
  t_days: number;
  trading_days: number;
  num_paths: number;
  risk_free: number;
}

export interface MoneyStatistics {
  expected_ST: number;
  median_ST: number;
  p5_ST: number;
  p25_ST: number;
  p75_ST: number;
  p95_ST: number;
  mean_logret: number;
  std_logret: number;
  var95_loss: number;
  cvar95_loss: number;
  var99_loss: number;
  prob_decline: number;
  prob_loss_gt10: number;
  prob_gain_gt10: number;
  sharpe_like: number;
  skew: number;
  kurt: number;
  call_price_mc?: number | null;
  put_price_mc?: number | null;
  call_price_bs?: number | null;
  put_price_bs?: number | null;
}

export interface HistogramData {
  centers: number[];
  counts: number[];
}

export interface FinanceResponse {
  positions: number[];
  colors: number[];
  sample_paths: number[][];
  time_axis_days: number[];
  histogram: HistogramData;
  stats: MoneyStatistics;
  statistics: SimulationStatistics;
  metadata: FinanceMetadata;
  per_asset_expected?: number[];
  diversification_benefit?: number;
}

export interface RequestEcho {
  kind: 'asset' | 'portfolio';
  symbol?: string;
  symbols?: string[];
  s0?: number;
  mu_ann?: number;
  sigma_ann?: number;
  t_days?: number;
  strike?: number | null;
  corr_level?: number;
}

export interface FinanceMetadata {
  execution_time: number;
  total_points: number;
  paths?: number;
  symbol?: string;
  symbols?: string[];
  s0?: number;
  s0_port?: number;
  mu_ann?: number;
  sigma_ann?: number;
  t_days?: number;
  t_years?: number;
  trading_days?: number;
  strike?: number | null;
  risk_free?: number;
  tail_particles_count?: number;
  tail_percentage?: number;
  weights?: number[];
  run_id?: string;
  run_at?: string;
  seed?: number | null;
  request_echo?: RequestEcho | null;
  [key: string]: string | number | boolean | string[] | number[] | undefined | null | RequestEcho;
}

export interface MarketQuote {
  symbol: string;
  price: number;
  currency: string;
  change_24h_pct?: number | null;
  source: string;
  as_of: string;
}

export interface MarketPreset {
  symbol: string;
  price: number;
  mu: number;
  sigma: number;
  currency: string;
}

export interface SavedScenario {
  id: string;
  name?: string;
  params: FinanceParams;
  stats: MoneyStatistics;
  timestamp: string;
}

export type ViewMode = 'dist3d' | 'fan2d' | 'hist';
