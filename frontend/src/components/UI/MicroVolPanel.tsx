import React, { useState } from 'react';
import { AlertCircle, Cpu, Loader2 } from 'lucide-react';
import { runMicroVol, type MicroVolResponse } from '../../api/simulationApi';

const ASSETS = ['BTC-USD', 'ETH-USD', 'SPY'];
const LOOKBACK_DAYS = 750;

/**
 * Paper v38 benchmark (Table III, flat H=4). Static constants sourced from the
 * paper -- never computed from a live run.
 */
const PAPER = {
  params: 116,
  baselineParams: '78,785',
  ratio: '679×',
  margin: '0.02 QLIKE',
  ci: 'includes 0',
} as const;

/**
 * Table III: asset-mean QLIKE gap against RCM for a flat H=4 calendar residual.
 * Gap = QLIKE(model) - QLIKE(RCM); negative favours the model. BTC, ETH and
 * SOL are pooled (asset mean); SPY is a single-asset point estimate.
 */
const PAPER_PROTOCOL = [
  { window: '5-year', gap: -0.0068, ci: '[-0.0299, +0.0117]' },
  { window: '7-year', gap: -0.0111, ci: '[-0.0370, +0.0084]' },
  { window: 'Disjoint', gap: -0.0431, ci: '[-0.1243, +0.0196]' },
  { window: 'SPY', gap: 0.0366, ci: null },
] as const;

function decimal(value: number | null | undefined, digits = 4): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function signed(value: number | null | undefined, digits = 4): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  const fixed = value.toFixed(digits);
  return value > 0 ? `+${fixed}` : fixed;
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

/** Static Table III reproduction. Clearly separated from the live close-only run. */
const PaperProtocolCard: React.FC = () => (
  <section className="rounded-lg border border-zinc-800 bg-[#0c0d12] p-4 sm:p-5">
    <div className="flex flex-wrap items-baseline justify-between gap-2">
      <h2 className="font-serif text-sm font-semibold tracking-tight text-zinc-100">
        Paper protocol
      </h2>
      <span className="font-mono text-[10px] uppercase tracking-wider text-amber-400">
        paper benchmark, not a live run
      </span>
    </div>
    <p className="mt-1.5 text-[11px] leading-relaxed text-zinc-500">
      Table III, flat H=4 calendar residual, {PAPER.params} parameters ({PAPER.ratio} fewer than
      the {PAPER.baselineParams}-parameter RCM). BTC, ETH and SOL are asset-mean gaps; SPY is a
      single-asset point estimate. Gap = QLIKE(model) − QLIKE(RCM); negative favours the model.
    </p>

    <div className="mt-3 overflow-x-auto">
      <table className="w-full text-left font-mono text-[11px] tabular-nums">
        <thead>
          <tr className="text-[10px] uppercase tracking-wider text-zinc-500">
            <th className="py-1.5 pr-3 font-medium">Window</th>
            <th className="py-1.5 pr-3 font-medium">Gap vs RCM</th>
            <th className="py-1.5 font-medium">Date-clustered 95% CI</th>
          </tr>
        </thead>
        <tbody className="text-zinc-200">
          {PAPER_PROTOCOL.map((row) => (
            <tr key={row.window} className="border-t border-zinc-800/80">
              <td className="py-1.5 pr-3 text-zinc-400">{row.window}</td>
              <td className={`py-1.5 pr-3 ${row.gap < 0 ? 'text-emerald-400' : 'text-zinc-200'}`}>
                {signed(row.gap)}
              </td>
              <td className="py-1.5 text-zinc-400">
                {row.ci ?? 'single asset, no CI reported'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>

    <p className="mt-2.5 text-[10px] leading-relaxed text-zinc-600">
      Every crypto interval includes zero: this is parity at {PAPER.ratio} compression, not
      dominance. TSR-100 (100 parameters, 788×) was explored after the ladder and is
      <span className="text-amber-400/90"> exploratory</span>, not confirmatory. SPY is the one
      window where RCM remains better. {PAPER.margin} is the paper's non-inferiority margin.
    </p>
  </section>
);

/**
 * MicroVol live demo: a compact CPU-trained variance model. Every live figure
 * is echoed from POST /api/research/microvol. The paper protocol card above is
 * static and must never be mixed with these numbers.
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
  const annualization = data?.annualization;
  const gap = data ? (data.qlike_gap ?? data.qlike_model - data.qlike_har) : null;
  const split = data ? splitLabel(data.split) : '';
  const fallback = Boolean(data && (data.fallback || data.torch_available === false));

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
            {busy ? 'Training…' : 'Train model'}
          </button>
        </form>

        <p id="microvol-asset-hint" className="mt-2 text-[11px] text-zinc-500">
          Live demo: trains on the server CPU in seconds from {LOOKBACK_DAYS} daily closes and
          scores the next-day variance out of sample against HAR. It does not reproduce the paper
          protocol below.
        </p>
      </section>

      <PaperProtocolCard />

      {err && (
        <div role="alert" className="flex items-start gap-2 rounded-lg border border-rose-900/70 bg-[#0c0d12] px-3 py-2.5 text-xs text-rose-300">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-rose-400" aria-hidden="true" />
          <span>{err}</span>
        </div>
      )}

      <div aria-live="polite" className="space-y-4">
        {busy && (
          <p className="flex items-center gap-2 font-mono text-[11px] text-zinc-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
            Training on {symbol || 'asset'} — server CPU only, this can take a few seconds.
          </p>
        )}

        {!busy && !data && !err && (
          <section className="rounded-lg border border-zinc-800 bg-[#0c0d12] p-4">
            <p className="text-xs text-zinc-400">No live run yet.</p>
            <p className="mt-1 text-[11px] text-zinc-500">
              Pick an asset and train the model to see its out-of-sample QLIKE against HAR on a
              close-only demo split. The paper protocol card above is separate and static.
            </p>
          </section>
        )}

        {data && !busy && (
          <div className="space-y-4">
            {/* Live-run provenance */}
            <p className="font-mono text-[11px] leading-relaxed text-zinc-500">
              Live demo · <span className="text-zinc-300">{data.model}</span>
              {' · '}{data.symbol}{' · '}{data.source}{' · '}{data.n_obs.toLocaleString('en-US')} obs
              {split ? ` · ${split}` : ''}{data.cached ? ' · cached' : ''}
            </p>

            {fallback && (
              <div
                role="status"
                className="rounded-lg border border-rose-900/60 bg-rose-950/10 px-3 py-2.5 text-[11px] leading-relaxed text-rose-200/90"
              >
                <span className="font-semibold text-rose-300">Fallback run — no sequence model trained.</span>{' '}
                torch {data.torch_available === false ? 'is unavailable or the GRU failed to train' : 'fallback'}
                {data.torch_error ? ` (${data.torch_error})` : ''}. These figures are the{' '}
                19-parameter m3 calendar-HAR backbone only; the 100-parameter MicroVol claim does
                not apply to this run.
              </div>
            )}

            {/* Demo target caveat */}
            <div className="rounded-lg border border-amber-900/40 bg-amber-950/10 px-3 py-2.5 text-[11px] leading-relaxed text-amber-200/80">
              <span className="font-semibold text-amber-300">Live demo, close-only.</span>{' '}
              This run uses daily close-to-close variance on a 70/10/20 demo split —{' '}
              <span className="text-amber-300">demo split ≠ paper split</span>. Absolute QLIKE and
              the demo gap are not comparable to the paper's full HAR-X protocol (Table III above).
            </div>

            {/* Live metrics */}
            <section className="overflow-hidden rounded-lg border border-zinc-800">
              <div className="grid grid-cols-2 gap-px bg-zinc-800 sm:grid-cols-3">
                <Metric
                  label="Parameters"
                  value={params != null ? params.toLocaleString('en-US') : '—'}
                  hint={fallback ? 'HAR only, 0 sequence params' : 'fitted'}
                />
                <Metric label="Train time" value={`${decimal(data.train_seconds, 2)} s`} hint="server CPU" />
                <Metric
                  label="Test days"
                  value={heldOutDays != null ? heldOutDays.toLocaleString('en-US') : '—'}
                  hint="demo split only"
                />
                <Metric label="QLIKE (demo)" value={decimal(data.qlike_model, 4)} hint="close-only" />
                <Metric label="HAR (demo)" value={decimal(data.qlike_har, 4)} hint="baseline" />
                <Metric
                  label="Demo gap"
                  value={signed(gap, 4)}
                  hint="model − HAR; negative = model better"
                />
              </div>
            </section>

            {/* Forecast */}
            <section className="rounded-lg border border-zinc-800 bg-[#0c0d12] p-4 sm:p-5">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h3 className="text-xs font-semibold text-zinc-200">Next-session forecast</h3>
                <span className="font-mono text-[10px] text-zinc-500">
                  {data.symbol}
                  {annualization ? ` · annualised ×√${annualization}` : ''}
                </span>
              </div>
              <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-3">
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-zinc-500">Next-day vol</p>
                  <p className="mt-1 font-mono text-2xl font-semibold tabular-nums text-zinc-100">
                    {decimal(vol, 4)}
                  </p>
                  <p className="mt-0.5 text-[10px] text-zinc-600">daily, not annualised</p>
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
                      <p className="mt-0.5 text-[10px] text-zinc-600">daily return quantile</p>
                    </>
                  ) : (
                    <>
                      <p className="text-[10px] uppercase tracking-wider text-zinc-500">Annualized vol</p>
                      <p className="mt-1 font-mono text-lg tabular-nums text-zinc-200">{decimal(annualizedVol, 4)}</p>
                      <p className="mt-0.5 text-[10px] text-zinc-600">
                        {annualization === 365
                          ? '365 calendar days (24/7 crypto)'
                          : annualization === 252
                            ? '252 trading days (equity)'
                            : 'annualisation factor not reported'}
                      </p>
                    </>
                  )}
                </div>
              </div>
            </section>

            <p className="text-[11px] leading-relaxed text-zinc-400">
              <span className="text-amber-400">Why this matters:</span>{' '}
              {fallback
                ? 'a 19-parameter linear calendar-HAR still prices the next day on a server CPU, but no sequence model was trained in this run.'
                : `${params ?? 'Compact'}-parameter model trained in ${decimal(data.train_seconds, 2)} s on a server CPU — no GPU — so the same forecast runs anywhere the API does.`}
              {gap != null && gap > 0 && !fallback
                ? ' On this demo split the model did not beat HAR; the paper claim is non-inferiority within a 0.02 QLIKE margin on the full HAR-X target, not this close-only demo.'
                : ''}
            </p>

            {data.note && <p className="text-[11px] leading-relaxed text-zinc-500">{data.note}</p>}
          </div>
        )}
      </div>
    </div>
  );
};

export default MicroVolPanel;
