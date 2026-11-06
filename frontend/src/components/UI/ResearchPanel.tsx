import React, { useState } from 'react';
import { FlaskConical, Loader2, TrendingUp, Sigma, GitCompareArrows, Award } from 'lucide-react';
import { runBenchmark, BenchmarkResponse } from '../../api/simulationApi';

/**
 * Research panel: scientific model benchmark
 * (GBM vs Student-t vs Merton jump diffusion vs LSTM volatility forecaster).
 */
export const ResearchPanel: React.FC = () => {
  const [symbol, setSymbol] = useState('SPY');
  const useLstm = true;
  const [busy, setBusy] = useState(false);
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [err, setErr] = useState('');
  const ranOnce = React.useRef(false);

  const run = async () => {
    setBusy(true);
    setErr('');
    setData(null);
    try {
      setData(await runBenchmark(symbol));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // run once on first tab open (fast mode: no LSTM in the initial hit, refresh for the full one)
  React.useEffect(() => {
    if (!ranOnce.current) {
      ranOnce.current = true;
      const t = setTimeout(() => {
        runBenchmark(symbol, useLstm).then(setData).catch((e) => setErr(e instanceof Error ? e.message : String(e)));
      }, 300);
      return () => clearTimeout(t);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="max-w-4xl mx-auto space-y-4">
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4">
        <div className="flex items-center gap-2 mb-3">
          <FlaskConical className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-white">Scientific Model Benchmark — is plain GBM enough?</h2>
        </div>
        <p className="text-xs text-zinc-400 mb-3">
          Loads <strong>{symbol}</strong> history from the internet (Yahoo Finance), calibrates four models and
          validates them statistically: MLE fit, Kolmogorov–Smirnov test, QQ-tail comparison, Kupiec VaR backtest,
          and RMSE of the volatility forecast (constant-GBM vs EWMA vs LSTM).
        </p>
        <div className="flex gap-2 items-center">
          <input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            className="bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-xs font-mono w-32" placeholder="SPY" />
          <button onClick={() => void run()} disabled={busy}
            className="flex items-center gap-1.5 bg-cyan-700 hover:bg-cyan-600 disabled:opacity-50 text-white text-xs font-medium px-4 py-2 rounded-lg">
            {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <TrendingUp className="w-4 h-4" />}
            {busy ? 'Training LSTM + sampling 60k paths…' : 'Run benchmark'}
          </button>
          <span className="text-[10px] text-zinc-500 font-mono">try: SPY · NVDA · BTC-USD</span>
        </div>
        {err && <p className="text-xs text-rose-400 mt-2">{err}</p>}
      </div>

      {data && (
        <>
          {/* data provenance */}
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4 text-xs">
            <div className="flex items-center gap-2 mb-1">
              <Sigma className="w-3.5 h-3.5 text-zinc-400" />
              <p className="text-zinc-200 font-semibold">Data provenance</p>
            </div>
            <p className="text-zinc-400 font-mono text-[11px]">
              {data.data.source} · {data.data.n_obs} daily observations · last close ${data.data.last_close.toLocaleString('en-US')}
              {' '}· MLE: μ={data.calibration.mu_ann.toFixed(3)}/y, σ={data.calibration.sigma_ann.toFixed(3)}/y
            </p>
          </div>

          {/* terminal distribution comparison */}
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4">
            <div className="flex items-center gap-2 mb-3">
              <GitCompareArrows className="w-3.5 h-3.5 text-cyan-400" />
              <p className="text-xs text-zinc-200 font-semibold">Terminal distribution over a 1-year horizon (3 stochastic models)</p>
            </div>
            <div className="overflow-x-auto rounded-lg border border-zinc-800">
              <table className="w-full text-[11px]">
                <thead className="bg-zinc-950 text-zinc-400 font-mono">
                  <tr>{['model', 'E[ST]', 'median', 'P5', 'P95', 'VaR95', 'CVaR95', 'skew', 'ex.kurt', 'P(decline)'].map((h) => (
                    <th key={h} className="text-left px-3 py-2 border-b border-zinc-800">{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {data.terminal_distribution_comparison.map((r) => (
                    <tr key={String(r.model)} className="odd:bg-zinc-900/40">
                      {['model', 'expected_ST', 'median_ST', 'p5_ST', 'p95_ST', 'VaR95_return', 'CVaR95_return', 'skew', 'excess_kurt', 'prob_decline'].map((k) => (
                        <td key={k} className={`px-3 py-1.5 ${k === 'CVaR95_return' ? 'text-rose-300' : k === 'p5_ST' ? 'text-amber-300' : ''} ${k === 'model' ? 'text-zinc-100' : 'font-mono'}`}>
                          {k === 'expected_ST' || k === 'median_ST' || k === 'p5_ST' || k === 'p95_ST'
                            ? Number(r[k]).toLocaleString('en-US')
                            : String(r[k])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* KS + Kupiec */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4 text-[11px]">
              <p className="text-zinc-200 font-semibold text-xs mb-2">Kolmogorov–Smirnov: are returns Normal?</p>
              <pre className="text-[10px] bg-black/60 rounded-lg p-3 overflow-x-auto text-zinc-300 whitespace-pre-wrap">
                {JSON.stringify(data.diagnostics.ks_test, null, 2)}
              </pre>
            </div>
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4 text-[11px]">
              <p className="text-zinc-200 font-semibold text-xs mb-2">Kupiec VaR95 backtest (unconditional coverage)</p>
              <pre className="text-[10px] bg-black/60 rounded-lg p-3 overflow-x-auto text-zinc-300 whitespace-pre-wrap">
                {JSON.stringify(data.diagnostics.kupiec_var95, null, 2)}
              </pre>
            </div>
          </div>

          {/* vol forecast RMSE */}
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4">
            <div className="flex items-center gap-2 mb-2">
              <Award className="w-3.5 h-3.5 text-amber-400" />
              <p className="text-xs text-zinc-200 font-semibold">
                Volatility forecast RMSE (annualized) — winner: <span className="text-amber-300">{data.vol_forecast_rmse.winner ?? '—'}</span>
              </p>
            </div>
            {data.vol_forecast_rmse.rows.length > 0 ? (
              <div className="space-y-1.5">
                {data.vol_forecast_rmse.rows
                  .slice()
                  .sort((a, b) => a.rmse - b.rmse)
                  .map((r) => (
                    <div key={r.model} className="flex items-center gap-2 text-[11px]">
                      <span className="w-48 shrink-0 text-zinc-300">{r.model}</span>
                      <div className="flex-1 h-2.5 bg-zinc-950 rounded overflow-hidden border border-zinc-800">
                        <div className="h-full bg-cyan-600/70"
                          style={{ width: `${Math.max(6, 100 * r.rmse / (data.vol_forecast_rmse.rows[0]?.rmse || 1))}%` }} />
                      </div>
                      <span className="font-mono text-zinc-400 w-16 text-right">{r.rmse.toFixed(3)}</span>
                    </div>
                  ))}
              </div>
            ) : (
              <p className="text-[11px] text-zinc-500 font-mono">{data.vol_forecast_rmse.note ?? 'Forecast benchmark pending.'}</p>
            )}
          </div>

          <p className="text-[10px] text-zinc-500 font-mono">compute time: {data.execution_time_ms} ms</p>
        </>
      )}
    </div>
  );
};

export default ResearchPanel;
