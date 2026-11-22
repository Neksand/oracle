import React, { useState } from 'react';
import { AlertCircle, Cpu, Loader2 } from 'lucide-react';
import { runMicroVol, type MicroVolResponse } from '../../api/simulationApi';

const ASSETS = ['BTC-USD', 'ETH-USD', 'SPY'];
const LOOKBACK_DAYS = 750;

/** Paper v35 benchmark, Table III: these are not live-demo numbers. */
const PAPER = {
  params: 100,
  ratio: '788×',
  margin: '0.02 QLIKE',
  sevenYearCI: 'excludes zero',
} as const;

function decimal(value: number | null | undefined, digits = 4): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function parameterCount(params: MicroVolResponse['params']): number | null {
  if (typeof params === 'number') return Number.isFinite(params) ? params : null;
  return typeof params?.count === 'number' ? params.count : null;
}

/** Extract held-out days whether the API sends a string or a {test} object. */
function testDays(split: MicroVolResponse['split']): number | null {
  if (split && typeof split === 'object') {
    if (typeof split.test === 'number') return split.test;
    if (typeof split.test_days === 'number') return split.test_days;
  }
  if (typeof split === 'string') {
    const match = split.match(/test[^0-9]{0,16}(\d[\d,]*)/i);
    if (match) return Number(match[1].replace(/,/g, ''));
  }
  return null;
}

function splitLabel(split: MicroVolResponse['split']): string {
  if (typeof split === 'string') return split;
  return split?.scheme ?? '';
}

const Metric: React.FC<{ label: string; value: string; hint?: string }> = ({ label, value, hint }) => (
  <div className="bg-[#0c0d12] p-3">
    <p className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</p>
    <p className="mt-1 font-mono text-sm tabular-nums text-zinc-100">{value}</p>
    {hint && <p className="mt-0.5 text-[10px] text-zinc-600">{hint}</p>}
  </div>
);

/**
 * MicroVol: the compact, CPU-only volatility model. All figures shown here are
 * echoed from POST /api/research/microvol — nothing is mocked or estimated.
 */
export const MicroVolPanel: React.FC = () => {
  const [symbol, setSymbol] = useState('SPY');
  const [busy, setBusy] = useState(false);
  const [data, setData] = useState<MicroVolResponse | null>(null);
  const [err, setErr] = useState('');

  const run = async () => {
    const asset = symbol.trim().toUpperCase();
    if (!asset) {
      setErr('Enter an asset symbol, for example SPY.');
      return;
    }
    if (asset === symbol) setSymbol(asset);
    setBusy(true);
    setErr('');
    try {
      setData(await runMicroVol(asset, LOOKBACK_DAYS));
    } catch (e) {
      setData(null);
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const params = data ? parameterCount(data.params) : null;
  const heldOutDays = data ? testDays(data.split) : null;
  const vol = data ? (data.forecast?.next_day_vol ?? data.forecast?.vol_daily) : undefined;
  const variance = data ? (data.forecast?.next_day_var ?? data.forecast?.variance_daily) : undefined;
  const var95 = data?.forecast?.var95;
  const annualizedVol = data?.forecast?.vol_annualized;
  const improvement = data ? (data.improvement ?? data.qlike_har - data.qlike_model) : null;
  const split = data ? splitLabel(data.split) : '';

  return (
    <div className="mx-auto max-w-3xl space-y-4 pb-6">
      {/* Input / action */}
      <section className="rounded-lg border border-zinc-800 bg-[#0c0d12] p-4 sm:p-5">
        <div className="mb-4 flex items-center gap-2">
          <Cpu className="h-4 w-4 text-amber-500" aria-hidden="true" />
          <h2 className="font-serif text-base font-semibold tracking-tight text-zinc-100">MicroVol</h2>
          <span className="ml-auto font-mono text-[11px] text-zinc-500">{LOOKBACK_DAYS}-day lookback</span>
        </div>

        <form
          className="flex flex-col gap-3 sm:flex-row sm:items-end"
          onSubmit={(e) => { e.preventDefault(); void run(); }}
        >
          <div className="w-full sm:w-48">
            <label htmlFor="microvol-symbol" className="mb-1.5 block text-[11px] uppercase tracking-wider text-zinc-400">
              Asset
            </label>
            <input
              id="microvol-symbol"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="SPY"
              autoComplete="off"
              spellCheck={false}
              aria-describedby="microvol-asset-hint"
              className="w-full rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 font-mono text-sm text-zinc-100 outline-none focus-visible:border-amber-500 focus-visible:ring-1 focus-visible:ring-amber-500/50"
            />
          </div>

          <div className="flex items-center gap-1.5" role="group" aria-label="Preset assets">
            {ASSETS.map((asset) => (
              <button
                key={asset}
                type="button"
                onClick={() => setSymbol(asset)}
                aria-pressed={symbol === asset}
                className={`rounded-md border px-2.5 py-1.5 font-mono text-[11px] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-amber-500 ${
                  symbol === asset
                    ? 'border-amber-500/60 text-amber-400'
                    : 'border-zinc-700 text-zinc-400 hover:border-zinc-500 hover:text-zinc-200'
                }`}
              >
                {asset}
              </button>
            ))}
          </div>

          <button
            type="submit"
            disabled={busy}
            className="inline-flex items-center justify-center gap-2 rounded-md bg-amber-500 px-3.5 py-2 text-xs font-semibold text-zinc-950 hover:bg-amber-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-500/60 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0c0d12] disabled:opacity-50 sm:ml-auto"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : null}
            {busy ? 'Training…' : 'Train 100-param model'}
          </button>
        </form>

        <p id="microvol-asset-hint" className="mt-2 text-[11px] text-zinc-500">
          Fits a compact variance model on CPU from {LOOKBACK_DAYS} daily observations and scores it out of sample against HAR.
        </p>
      </section>

      {err && (
        <div role="alert" className="flex items-start gap-2 rounded-lg border border-rose-900/70 bg-[#0c0d12] px-3 py-2.5 text-xs text-rose-300">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-rose-400" aria-hidden="true" />
          <span>{err}</span>
        </div>
      )}

      <div aria-live="polite">
        {busy && (
          <p className="flex items-center gap-2 font-mono text-[11px] text-zinc-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
            Training on {symbol || 'asset'} — CPU only, this can take a moment.
          </p>
        )}

        {!busy && !data && !err && (
          <section className="rounded-lg border border-zinc-800 bg-[#0c0d12] p-4">
            <p className="text-xs text-zinc-400">No run yet.</p>
            <p className="mt-1 text-[11px] text-zinc-500">
              Pick an asset and train the model to see the out-of-sample QLIKE comparison and the next-session forecast.
            </p>
          </section>
        )}

        {data && !busy && (
          <div className="space-y-4">
            {/* Provenance */}
            <p className="font-mono text-[11px] leading-relaxed text-zinc-500">
              <span className="text-zinc-300">{data.model}</span>
              {' · '}{data.symbol}{' · '}{data.source}{' · '}{data.n_obs.toLocaleString('en-US')} obs
              {split ? ` · ${split}` : ''}{data.cached ? ' · cached' : ''}
            </p>

            {/* Demo caveat: live target differs from the paper benchmark */}
            <div className="rounded-lg border border-amber-900/40 bg-amber-950/10 px-3 py-2.5 text-[11px] leading-relaxed text-amber-200/80">
              <span className="font-semibold text-amber-300">Live demo target.</span>{' '}
              This run uses close-only daily variance; absolute QLIKE and the gap are not
              comparable to the paper's HAR-X benchmark. The paper result is separate:
              non-inferior within a {PAPER.margin} margin at {PAPER.ratio} compression.
            </div>

            {/* Paper benchmark, clearly separated from the live run */}
            <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-zinc-800 bg-zinc-800 sm:grid-cols-4">
              <Metric label="Paper params" value={String(PAPER.params)} hint="Table III" />
              <Metric label="Paper compression" value={PAPER.ratio} hint="vs 78,785" />
              <Metric label="Non-inferiority" value={PAPER.margin} hint="95%" />
              <Metric label="7-year CI" value={PAPER.sevenYearCI} hint="bootstrap" />
            </div>

            {/* Model metrics */}
            <section className="overflow-hidden rounded-lg border border-zinc-800">
              <div className="grid grid-cols-2 gap-px bg-zinc-800 sm:grid-cols-3">
                <Metric label="Parameters" value={params != null ? params.toLocaleString('en-US') : '—'} hint="fitted" />
                <Metric label="Train time" value={`${decimal(data.train_seconds, 2)} s`} hint="CPU" />
                <Metric label="Test days" value={heldOutDays != null ? heldOutDays.toLocaleString('en-US') : '—'} hint="held out" />
                <Metric label="QLIKE (demo)" value={decimal(data.qlike_model, 4)} hint="close-only" />
                <Metric label="HAR (demo)" value={decimal(data.qlike_har, 4)} hint="baseline" />
                <Metric label="Demo gap" value={decimal(improvement, 4)} hint="HAR − model" />
              </div>
            </section>

            {/* Forecast */}
            <section className="rounded-lg border border-zinc-800 bg-[#0c0d12] p-4 sm:p-5">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h3 className="text-xs font-semibold text-zinc-200">Next-session forecast</h3>
                <span className="font-mono text-[10px] text-zinc-500">{data.symbol}</span>
              </div>
              <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-3">
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-zinc-500">Next-day vol</p>
                  <p className="mt-1 font-mono text-2xl font-semibold tabular-nums text-zinc-100">
                    {decimal(vol, 4)}
                  </p>
                </div>
                <div className="sm:border-l sm:border-zinc-800 sm:pl-4">
                  <p className="text-[10px] uppercase tracking-wider text-zinc-500">Next-day variance</p>
                  <p className="mt-1 font-mono text-lg tabular-nums text-zinc-200">
                    {decimal(variance, 6)}
                  </p>
                </div>
                <div className="sm:border-l sm:border-zinc-800 sm:pl-4">
                  {var95 != null ? (
                    <>
                      <p className="text-[10px] uppercase tracking-wider text-zinc-500">VaR 95%</p>
                      <p className="mt-1 font-mono text-lg tabular-nums text-zinc-200">{decimal(var95, 4)}</p>
                    </>
                  ) : (
                    <>
                      <p className="text-[10px] uppercase tracking-wider text-zinc-500">Annualized vol</p>
                      <p className="mt-1 font-mono text-lg tabular-nums text-zinc-200">{decimal(annualizedVol, 4)}</p>
                    </>
                  )}
                </div>
              </div>
            </section>

            <p className="text-[11px] leading-relaxed text-zinc-400">
              <span className="text-amber-400">Why this matters:</span>{' '}
              {params != null ? `${params}-parameter` : 'Compact'} model trained in {decimal(data.train_seconds, 2)} s
              on CPU only — no GPU — so the same forecast runs anywhere the API does.
            </p>

            {data.note && <p className="text-[11px] leading-relaxed text-zinc-500">{data.note}</p>}
          </div>
        )}
      </div>
    </div>
  );
};

export default MicroVolPanel;
