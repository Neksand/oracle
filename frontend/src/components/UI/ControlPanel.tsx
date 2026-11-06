import React, { useEffect, useState } from 'react';
import { FinanceParams, PortfolioParams, MarketPreset } from '../../types';
import { fetchPresets } from '../../api/simulationApi';
import { Play, RotateCw, Search, Wand2, Layers, LineChart as LineIcon, Sliders, HelpCircle, Lock, LockOpen, Dices } from 'lucide-react';
import type { AppliedRequest } from '../../hooks/useSimulation';

export interface ControlPanelProps {
  params: FinanceParams;
  applied: AppliedRequest;
  loading: boolean;
  marketLoading: boolean;
  seedLocked: boolean;
  lastSeed: number | null;
  onToggleSeedLock: (v: boolean) => void;
  onReroll: () => void;
  onSimulate: (p: FinanceParams) => void;
  onSimulatePortfolio: (p: PortfolioParams) => void;
  onLivePrice: (symbol: string) => void;
  onAutoCalibrate: (symbol: string) => void;
}

const HORIZONS = [
  { days: 30, label: '1M' },
  { days: 90, label: '3M' },
  { days: 252, label: '1Y' },
  { days: 365, label: '1Y+' },
];

const VOL_PRESETS = [
  { label: 'Calm · 12%', value: 0.12, hint: 'Bonds, defensive' },
  { label: 'Market · 18%', value: 0.18, hint: 'S&P 500-like' },
  { label: 'Active · 35%', value: 0.35, hint: 'Growth stocks' },
  { label: 'Crypto · 65%', value: 0.65, hint: 'BTC / ETH-like' },
];

export const ControlPanel: React.FC<ControlPanelProps> = ({
  params, applied, loading, marketLoading, seedLocked, lastSeed,
  onToggleSeedLock, onReroll, onSimulate, onSimulatePortfolio, onLivePrice, onAutoCalibrate,
}) => {
  const [tab, setTab] = useState<'asset' | 'portfolio'>('asset');
  const [form, setForm] = useState<FinanceParams>(params);
  const [presets, setPresets] = useState<MarketPreset[]>([]);
  const [showHelp, setShowHelp] = useState(false);
  const [pAssets, setPAssets] = useState([
    { symbol: 'SPY', s0: 585, mu_ann: 0.09, sigma_ann: 0.17, weight: 0.6 },
    { symbol: 'BTC-USD', s0: 97500, mu_ann: 0.25, sigma_ann: 0.62, weight: 0.4 },
  ]);
  const [corr, setCorr] = useState(0.3);
  const [pDays, setPDays] = useState(90);

  useEffect(() => { setForm(params); }, [params.symbol, params.s0, params.mu_ann, params.sigma_ann]);
  useEffect(() => { fetchPresets().then(setPresets).catch(() => undefined); }, []);

  const applyPreset = (pr: MarketPreset) => {
    const isCrypto = pr.symbol.includes('BTC') || pr.symbol.includes('ETH');
    const next: FinanceParams = {
      ...form, symbol: pr.symbol, s0: pr.price, mu_ann: pr.mu, sigma_ann: pr.sigma,
      trading_days: isCrypto ? 365 : 252,
    };
    setForm(next);
    onSimulate(next);
  };

  const weightTotal = pAssets.reduce((s, a) => s + a.weight, 0);

  // --- Dirty detection: do the charts reflect the current form? ---
  const ASSET_KEYS = ['symbol', 's0', 'mu_ann', 'sigma_ann', 't_days', 'trading_days', 'risk_free', 'strike', 'option_type'] as const;
  const assetDirty = applied == null
    ? false
    : applied.kind !== 'asset'
      ? true
      : ASSET_KEYS.some((k) => (form[k] ?? null) !== (applied.params[k] ?? null));

  const normAssets = (list: typeof pAssets) => {
    const tot = list.reduce((s, a) => s + a.weight, 0) || 1;
    return list.map((a) => ({
      symbol: a.symbol, s0: +a.s0.toFixed(2), mu_ann: +a.mu_ann.toFixed(4),
      sigma_ann: +a.sigma_ann.toFixed(4), weight: +(a.weight / tot).toFixed(4),
    }));
  };
  const portfolioDirty = applied == null
    ? false
    : applied.kind !== 'portfolio'
      ? true
      : JSON.stringify({ a: normAssets(pAssets), c: +corr.toFixed(2), t: pDays, rf: form.risk_free }) !==
        JSON.stringify({
          a: normAssets(applied.req.assets.map((a) => ({ ...a }))),
          c: +applied.req.corr_level.toFixed(2), t: applied.req.t_days, rf: applied.req.risk_free,
        });

  return (
    <div className="bg-[#121215] border border-zinc-800 rounded-xl p-5 shadow-xl space-y-4 text-zinc-100 font-sans">
      <div className="border-b border-zinc-800/80 pb-3 flex items-start justify-between">
        <div>
          <h1 className="text-base font-semibold tracking-tight text-white">ORACLE</h1>
          <p className="text-xs text-zinc-400 mt-0.5">Economic Scenario Engine · capital distribution outlook</p>
        </div>
        <button type="button" onClick={() => setShowHelp(!showHelp)} title="How it works"
          className="text-zinc-400 hover:text-white p-1.5 rounded-md hover:bg-zinc-800/60">
          <HelpCircle className="w-4 h-4" />
        </button>
      </div>

      {showHelp && (
        <div className="bg-sky-950/30 border border-sky-900/40 rounded-lg p-3 text-[11px] text-sky-100/90 leading-relaxed space-y-1.5">
          <p><strong>1. Pick an asset</strong> — click a ticker or type your own, then hit <em>Live</em> for the latest price.</p>
          <p><strong>2. Tune expectations</strong> — expected return (μ) shifts the outlook up/down; volatility (σ) widens it.</p>
          <p><strong>3. Run scenarios</strong> — 8,000 Monte-Carlo paths show where capital may land, including stress tails.</p>
          <p><strong>Seed locked?</strong> — with the lock on, identical inputs always give an identical chart, so any change you see is a real input effect. Use <em>Re-roll</em> for a fresh random draw.</p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-1 p-1 bg-zinc-900/90 rounded-lg border border-zinc-800/80 text-xs">
        {(['asset', 'portfolio'] as const).map((t) => (
          <button key={t} type="button" onClick={() => setTab(t)}
            className={`py-1.5 rounded-md font-medium flex items-center justify-center gap-1.5 ${tab === t ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-zinc-200'}`}>
            {t === 'asset' ? <LineIcon className="w-3.5 h-3.5" /> : <Layers className="w-3.5 h-3.5" />}
            {t === 'asset' ? 'Single asset' : 'Portfolio'}
          </button>
        ))}
      </div>

      {tab === 'asset' && (
        <form onSubmit={(e) => { e.preventDefault(); onSimulate(form); }} className="space-y-3.5 text-xs">
          <div className="bg-zinc-900/60 p-3 rounded-lg border border-zinc-800/70 space-y-2">
            <label className="text-zinc-300 font-medium block">Instrument</label>
            <div className="flex flex-wrap gap-2">
              <input value={form.symbol} onChange={(e) => setForm({ ...form, symbol: e.target.value.toUpperCase() })}
                className="flex-1 min-w-[6rem] bg-zinc-950 border border-zinc-700 rounded px-2 py-1.5 font-mono text-zinc-100 outline-none focus:border-sky-500 uppercase" placeholder="SPY" />
              <button type="button" onClick={() => onLivePrice(form.symbol)} disabled={marketLoading}
                className="shrink-0 whitespace-nowrap px-2.5 py-1.5 rounded bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-200 flex items-center gap-1" title="Fetch the latest live price">
                <Search className="w-3.5 h-3.5" />{marketLoading ? '…' : 'Live'}
              </button>
              <button type="button" onClick={() => onAutoCalibrate(form.symbol)} disabled={marketLoading}
                className="shrink-0 whitespace-nowrap px-2.5 py-1.5 rounded bg-sky-950/60 hover:bg-sky-900/60 border border-sky-800/60 text-sky-200 flex items-center gap-1" title="Auto-fit expected return & volatility from market history">
                <Wand2 className="w-3.5 h-3.5" />Auto-fit
              </button>
            </div>
            {presets.length > 0 && (
              <div>
                <div className="text-[10px] text-zinc-500 mb-1">Quick start — click to load & run:</div>
                <div className="flex flex-wrap gap-1.5">
                  {presets.map((p) => (
                    <button key={p.symbol} type="button" onClick={() => applyPreset(p)} disabled={loading}
                      className="px-2 py-1 rounded bg-zinc-950/60 border border-zinc-800 text-zinc-300 hover:text-white hover:border-sky-600 font-mono text-[11px]" title={`μ ${(p.mu * 100).toFixed(0)}% · σ ${(p.sigma * 100).toFixed(0)}%`}>
                      {p.symbol}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div className="bg-zinc-900/60 p-3 rounded-lg border border-zinc-800/70">
              <label className="text-zinc-300 font-medium block mb-1" title="Current market price — the starting point of every path">Current price S₀, $</label>
              <input type="number" min="0.01" step="0.01" value={form.s0}
                onChange={(e) => setForm({ ...form, s0: Math.max(0.01, parseFloat(e.target.value) || 0.01) })}
                className="w-full bg-zinc-950 border border-zinc-700 rounded px-2 py-1 text-right font-mono text-zinc-100 outline-none focus:border-sky-500" />
            </div>
            <div className="bg-zinc-900/60 p-3 rounded-lg border border-zinc-800/70">
              <label className="text-zinc-300 font-medium block mb-1" title="Risk-free rate used for Sharpe ratio and option pricing">Risk-free rate, %</label>
              <input type="number" min="0" max="25" step="0.25" value={+(form.risk_free * 100).toFixed(2)}
                onChange={(e) => setForm({ ...form, risk_free: (parseFloat(e.target.value) || 0) / 100 })}
                className="w-full bg-zinc-950 border border-zinc-700 rounded px-2 py-1 text-right font-mono text-zinc-100 outline-none focus:border-sky-500" />
            </div>
          </div>

          <div className="bg-zinc-900/60 p-3 rounded-lg border border-zinc-800/70 space-y-1.5">
            <div className="flex justify-between items-center">
              <label className="text-zinc-300 font-medium" title="Average yearly drift. Positive lifts the whole distribution upward.">Expected return μ, % / year</label>
              <span className={`px-2 py-0.5 rounded font-mono ${form.mu_ann >= 0 ? 'bg-emerald-950/50 text-emerald-300' : 'bg-rose-950/50 text-rose-300'}`}>
                {(form.mu_ann * 100).toFixed(1)}%
              </span>
            </div>
            <input type="range" min="-0.3" max="0.6" step="0.01" value={form.mu_ann}
              onChange={(e) => setForm({ ...form, mu_ann: parseFloat(e.target.value) })} className="w-full accent-sky-400" />
            <div className="flex justify-between text-[10px] text-zinc-500 font-mono"><span>-30%</span><span>0%</span><span>+60%</span></div>
          </div>

          <div className="bg-zinc-900/60 p-3 rounded-lg border border-zinc-800/70 space-y-1.5">
            <div className="flex justify-between items-center">
              <label className="text-zinc-300 font-medium" title="Yearly volatility. Higher values = wider, more uncertain outcomes.">Volatility σ, % / year</label>
              <span className="bg-zinc-800 px-2 py-0.5 rounded font-mono">{(form.sigma_ann * 100).toFixed(0)}%</span>
            </div>
            <input type="range" min="0.05" max="1.2" step="0.01" value={form.sigma_ann}
              onChange={(e) => setForm({ ...form, sigma_ann: parseFloat(e.target.value) })} className="w-full accent-indigo-400" />
            <div className="grid grid-cols-2 gap-1 pt-1">
              {VOL_PRESETS.map((v) => (
                <button key={v.label} type="button" onClick={() => setForm({ ...form, sigma_ann: v.value })}
                  title={v.hint}
                  className={`px-2 py-1 rounded border text-[11px] ${Math.abs(form.sigma_ann - v.value) < 0.02 ? 'bg-zinc-700 border-zinc-500 text-white' : 'bg-zinc-950/50 border-zinc-800 text-zinc-400 hover:text-zinc-200'}`}>
                  {v.label}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-zinc-900/60 p-3 rounded-lg border border-zinc-800/70 space-y-2">
            <div className="flex justify-between items-center"><span className="text-zinc-300 font-medium">Horizon</span>
              <span className="font-mono text-zinc-200">{form.t_days} days · {(form.t_days / 365).toFixed(2)}y</span></div>
            <div className="grid grid-cols-4 gap-1.5 font-mono">
              {HORIZONS.map((h) => (
                <button key={h.days} type="button" onClick={() => setForm({ ...form, t_days: h.days })}
                  className={`py-1.5 rounded border ${form.t_days === h.days ? 'bg-zinc-800 border-zinc-500 text-white' : 'bg-zinc-950/40 border-zinc-800 text-zinc-400'}`}>{h.label}</button>
              ))}
            </div>
            <label className="flex items-center gap-2 text-[11px] text-zinc-400">
              <input type="checkbox" checked={form.trading_days === 365}
                onChange={(e) => setForm({ ...form, trading_days: e.target.checked ? 365 : 252 })} />
              Crypto calendar (365 days/year instead of 252)
            </label>
          </div>

          <div className="bg-zinc-900/60 p-3 rounded-lg border border-zinc-800/70 space-y-2">
            <div className="flex items-center gap-1.5 text-zinc-300 font-medium"><Sliders className="w-3.5 h-3.5 text-zinc-400" />Option valuation <span className="text-zinc-500 font-normal">(optional)</span></div>
            <div className="grid grid-cols-2 gap-2">
              <input type="number" min="0" step="any" value={form.strike ?? ''} placeholder={`Strike K (≈$${form.s0})`}
                onChange={(e) => setForm({ ...form, strike: e.target.value ? parseFloat(e.target.value) : undefined })}
                className="bg-zinc-950 border border-zinc-700 rounded px-2 py-1.5 font-mono text-zinc-100 outline-none" />
              <div className="grid grid-cols-2 gap-1">
                {(['call', 'put'] as const).map((o) => (
                  <button key={o} type="button" onClick={() => setForm({ ...form, option_type: o })}
                    className={`rounded border py-1.5 font-mono uppercase ${form.option_type === o ? 'bg-zinc-700 border-zinc-500 text-white' : 'text-zinc-400 border-zinc-800'}`}>{o}</button>
                ))}
              </div>
            </div>
            <p className="text-[10px] text-zinc-500">Leave empty to skip. Strike and risk-free only reprice options — the price fan and histogram stay the same.</p>
          </div>

          {assetDirty && (
            <div className="flex items-center gap-2 text-[11px] text-amber-300 bg-amber-950/30 border border-amber-800/40 rounded-lg px-3 py-2">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse shrink-0" />
              Inputs changed — charts still show the previous run. Press Run to refresh.
            </div>
          )}
          <button type="submit" disabled={loading}
            className={`w-full py-2.5 rounded-lg font-medium text-sm bg-zinc-100 hover:bg-white disabled:opacity-60 text-zinc-950 flex items-center justify-center gap-2 ${assetDirty && !loading ? 'ring-2 ring-amber-400/80' : ''}`}>
            {loading ? <><RotateCw className="w-4 h-4 animate-spin" />Running scenarios…</> : <><Play className="w-4 h-4 fill-current" />Run scenarios</>}
          </button>
          <div className="flex items-center gap-2">
            <button type="button" onClick={() => onToggleSeedLock(!seedLocked)} title={seedLocked ? 'Locked: identical inputs always give an identical chart — differences you see are real input effects' : 'Unlocked: every run draws fresh randomness'}
              className={`flex-1 py-1.5 rounded-lg border text-[11px] flex items-center justify-center gap-1.5 ${seedLocked ? 'bg-sky-950/50 border-sky-800/60 text-sky-200' : 'bg-zinc-900 border-zinc-800 text-zinc-400'}`}>
              {seedLocked ? <Lock className="w-3.5 h-3.5" /> : <LockOpen className="w-3.5 h-3.5" />}
              {seedLocked ? 'Seed locked' : 'Seed free'}
            </button>
            <button type="button" onClick={onReroll} disabled={loading} title="Same inputs, new random draw"
              className="flex-1 py-1.5 rounded-lg border bg-zinc-900 border-zinc-800 text-zinc-300 text-[11px] flex items-center justify-center gap-1.5 hover:border-zinc-600 disabled:opacity-60">
              <Dices className="w-3.5 h-3.5" />Re-roll
            </button>
            {lastSeed != null && <span className="font-mono text-[10px] text-zinc-500" title="Random seed of the last run">#{lastSeed}</span>}
          </div>
        </form>
      )}

      {tab === 'portfolio' && (
        <form onSubmit={(e) => {
          e.preventDefault();
          const tot = weightTotal || 1;
          onSimulatePortfolio({
            assets: pAssets.map((a) => ({ ...a, weight: a.weight / tot })),
            corr_level: corr, t_days: pDays, trading_days: 252, num_paths: form.num_paths, risk_free: form.risk_free,
          });
        }} className="space-y-3 text-xs">
          <p className="text-[11px] text-zinc-500">Mix up to 4 assets. Weights are normalized automatically{weightTotal > 0 && <> (now {Math.round(weightTotal * 100)}%)</>}.</p>
          {pAssets.map((a, i) => (
            <div key={i} className="bg-zinc-900/60 p-3 rounded-lg border border-zinc-800/70 space-y-2">
              <div className="grid grid-cols-2 gap-2">
                <input value={a.symbol} onChange={(e) => { const c = [...pAssets]; c[i] = { ...c[i], symbol: e.target.value.toUpperCase() }; setPAssets(c); }}
                  className="bg-zinc-950 border border-zinc-700 rounded px-2 py-1 font-mono uppercase" placeholder="Ticker" aria-label="Ticker" />
                <input type="number" value={a.s0} min="0.01" step="1" onChange={(e) => { const c = [...pAssets]; c[i] = { ...c[i], s0: parseFloat(e.target.value) || 1 }; setPAssets(c); }}
                  className="bg-zinc-950 border border-zinc-700 rounded px-2 py-1 font-mono text-right" title="Current price S0" aria-label="Price" />
              </div>
              <div className="grid grid-cols-3 gap-2 font-mono">
                <label className="space-y-1"><span className="text-zinc-500 text-[10px]">μ %</span>
                  <input type="number" value={+(a.mu_ann * 100).toFixed(1)} step="1" onChange={(e) => { const c = [...pAssets]; c[i] = { ...c[i], mu_ann: (parseFloat(e.target.value) || 0) / 100 }; setPAssets(c); }}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded px-1 py-1 text-right" /></label>
                <label className="space-y-1"><span className="text-zinc-500 text-[10px]">σ %</span>
                  <input type="number" value={+(a.sigma_ann * 100).toFixed(0)} step="1" onChange={(e) => { const c = [...pAssets]; c[i] = { ...c[i], sigma_ann: (parseFloat(e.target.value) || 0) / 100 }; setPAssets(c); }}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded px-1 py-1 text-right" /></label>
                <label className="space-y-1"><span className="text-zinc-500 text-[10px]">weight %</span>
                  <input type="number" value={+(a.weight * 100).toFixed(0)} step="5" onChange={(e) => { const c = [...pAssets]; c[i] = { ...c[i], weight: (parseFloat(e.target.value) || 0) / 100 }; setPAssets(c); }}
                    className="w-full bg-zinc-950 border border-zinc-700 rounded px-1 py-1 text-right" /></label>
              </div>
              {pAssets.length > 2 && (
                <button type="button" onClick={() => setPAssets(pAssets.filter((_, j) => j !== i))} className="text-rose-400 text-[11px] hover:text-rose-300">Remove</button>
              )}
            </div>
          ))}
          {pAssets.length < 4 && (
            <button type="button" onClick={() => setPAssets([...pAssets, { symbol: 'QQQ', s0: 515, mu_ann: 0.11, sigma_ann: 0.22, weight: 0.2 }])}
              className="w-full py-1.5 rounded border border-dashed border-zinc-700 text-zinc-300 hover:border-zinc-500">+ Add asset</button>
          )}
          <div className="bg-zinc-900/60 p-3 rounded-lg border border-zinc-800/70 space-y-1.5">
            <div className="flex justify-between" title="Average co-movement of assets. Lower = better diversification."><span className="text-zinc-300">Correlation ρ</span><span className="font-mono">{corr.toFixed(2)}</span></div>
            <input type="range" min="-0.5" max="0.9" step="0.05" value={corr} onChange={(e) => setCorr(parseFloat(e.target.value))} className="w-full accent-emerald-400" />
            <div className="flex justify-between"><span className="text-zinc-300">Horizon</span><span className="font-mono">{pDays} days</span></div>
            <div className="grid grid-cols-4 gap-1 font-mono">
              {HORIZONS.map((h) => (
                <button key={h.days} type="button" onClick={() => setPDays(h.days)} className={`py-1 rounded border ${pDays === h.days ? 'bg-zinc-700 border-zinc-500 text-white' : 'text-zinc-400 border-zinc-800'}`}>{h.label}</button>
              ))}
            </div>
          </div>
          {portfolioDirty && (
            <div className="flex items-center gap-2 text-[11px] text-amber-300 bg-amber-950/30 border border-amber-800/40 rounded-lg px-3 py-2">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse shrink-0" />
              Inputs changed — charts still show the previous run. Press Run to refresh.
            </div>
          )}
          <button type="submit" disabled={loading} className={`w-full py-2.5 rounded-lg bg-zinc-100 text-zinc-950 font-medium disabled:opacity-60 flex items-center justify-center gap-2 ${portfolioDirty && !loading ? 'ring-2 ring-amber-400/80' : ''}`}>
            {loading ? <><RotateCw className="w-4 h-4 animate-spin" />Running…</> : <><Play className="w-4 h-4 fill-current" />Run portfolio</>}
          </button>
        </form>
      )}

      <div className="border-t border-zinc-800/80 pt-3 space-y-1.5 text-[11px] text-zinc-300">
        <span className="font-medium text-zinc-400 block">Economic regimes (3D colors)</span>
        <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-[#00f5ff]" />Baseline — 60% most likely outcomes</div>
        <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-[#a855f7]" />Stress — 25% moderate deviations</div>
        <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-[#ff1744]" />Crisis — 15% tail events (VaR zone)</div>
      </div>
    </div>
  );
};
