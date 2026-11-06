import React from 'react';
import { MoneyStatistics, FinanceMetadata } from '../../types';
import { Clock, Scale, TrendingUp, AlertTriangle, Landmark } from 'lucide-react';

export interface StatisticsPanelProps {
  stats: MoneyStatistics | null;
  metadata: FinanceMetadata | null;
}

const fmt$ = (v: number) =>
  (v ?? 0).toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 });
const fmtPct = (v: number) => `${((v ?? 0) * 100).toFixed(1)}%`;

export const StatisticsPanel: React.FC<StatisticsPanelProps> = ({ stats, metadata }) => {
  if (!stats) {
    return (
      <div className="bg-[#121215] border border-zinc-800 rounded-xl p-5 text-center text-zinc-500 text-xs font-sans">
        <Scale className="w-6 h-6 mx-auto mb-2 text-zinc-600" />
        No scenario results yet
        <p className="text-[11px] text-zinc-600 mt-0.5">
          Pick a ticker above and press <strong>Run scenarios</strong>.
        </p>
      </div>
    );
  }
  const exec = metadata?.execution_time;
  return (
    <div className="bg-[#121215] border border-zinc-800 rounded-xl p-5 space-y-4 text-zinc-200 text-xs">
      <div className="border-b border-zinc-800/80 pb-2.5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Landmark className="w-4 h-4 text-zinc-400" />
          <h2 className="font-semibold text-white text-sm">Capital outlook & risk</h2>
        </div>
        {exec != null && (
          <span className="flex items-center gap-1 text-[11px] text-zinc-400 bg-zinc-900 px-2 py-0.5 rounded border border-zinc-800 font-mono" title="Compute time and number of simulated paths">
            <Clock className="w-3 h-3" />{Number(exec).toFixed(0)} ms · {(metadata?.total_points ?? 0).toLocaleString()} paths
          </span>
        )}
      </div>

      <div className="bg-zinc-900/70 border border-zinc-800/80 rounded-lg p-3 space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-medium text-zinc-300 flex items-center gap-1.5"><TrendingUp className="w-3.5 h-3.5 text-sky-400" />Expected outcome</span>
          <span className="text-[10px] bg-zinc-800 px-2 py-0.5 rounded font-mono">T = {metadata?.t_days}d</span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-zinc-950/70 p-2.5 rounded border border-zinc-800/60" title="Average terminal value across all paths">
            <span className="text-[10px] text-zinc-500 block">Mean terminal value</span>
            <div className="text-base font-semibold text-white font-mono">{fmt$(stats.expected_ST)}</div>
            <span className="text-[10px] text-zinc-400 font-mono">median {fmt$(stats.median_ST)}</span>
          </div>
          <div className="bg-zinc-950/70 p-2.5 rounded border border-zinc-800/60" title="Middle 90% of outcomes — your likely range">
            <span className="text-[10px] text-zinc-500 block">Likely range 90% (P5–P95)</span>
            <div className="text-xs font-semibold font-mono mt-1">[{fmt$(stats.p5_ST)}, {fmt$(stats.p95_ST)}]</div>
            <span className="text-[10px] text-zinc-500">quartiles: {fmt$(stats.p25_ST)} / {fmt$(stats.p75_ST)}</span>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-2 font-mono text-center">
          <div className="bg-zinc-950/60 p-1.5 rounded border border-zinc-800/60" title="Share of paths ending below today's price"><div className="text-[10px] text-zinc-500 font-sans">Ends lower</div><div className="text-amber-300">{fmtPct(stats.prob_decline)}</div></div>
          <div className="bg-zinc-950/60 p-1.5 rounded border border-zinc-800/60" title="Share of paths losing more than 10%"><div className="text-[10px] text-zinc-500 font-sans">Down &gt;10%</div><div className="text-rose-300">{fmtPct(stats.prob_loss_gt10)}</div></div>
          <div className="bg-zinc-950/60 p-1.5 rounded border border-zinc-800/60" title="Share of paths gaining more than 10%"><div className="text-[10px] text-zinc-500 font-sans">Up &gt;10%</div><div className="text-emerald-300">{fmtPct(stats.prob_gain_gt10)}</div></div>
        </div>
      </div>

      <div className="bg-zinc-900/40 border border-rose-900/30 rounded-lg p-3 space-y-2">
        <div className="flex items-center gap-1.5 text-rose-300 font-medium">
          <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />Tail risk (VaR / CVaR)
        </div>
        <div className="grid grid-cols-3 gap-2 font-mono text-center">
          <div className="p-2 rounded bg-zinc-950/60 border border-rose-900/40" title="Loss exceeded only 5% of the time"><div className="text-[10px] text-zinc-400 font-sans">VaR 95%</div><div className="text-sm text-rose-200 font-semibold">{fmt$(stats.var95_loss)}</div><div className="text-[10px] text-zinc-500 font-sans">5% worst</div></div>
          <div className="p-2 rounded bg-zinc-950/60 border border-rose-900/40" title="Average loss once you are in the worst 5%"><div className="text-[10px] text-zinc-400 font-sans">CVaR 95%</div><div className="text-sm text-rose-200 font-semibold">{fmt$(stats.cvar95_loss)}</div><div className="text-[10px] text-zinc-500 font-sans">avg tail</div></div>
          <div className="p-2 rounded bg-zinc-950/60 border border-rose-900/40" title="Loss exceeded only 1% of the time"><div className="text-[10px] text-zinc-400 font-sans">VaR 99%</div><div className="text-sm text-rose-200 font-semibold">{fmt$(stats.var99_loss)}</div><div className="text-[10px] text-zinc-500 font-sans">1% extreme</div></div>
        </div>
        <p className="text-[11px] text-zinc-400">Red particles in 3D are outcomes beyond the 85th percentile distance — the crisis zone.</p>
      </div>

      <div className="grid grid-cols-2 gap-2 font-mono text-center">
        <div className="bg-zinc-900/60 p-2 rounded border border-zinc-800/70" title="Excess return per unit of volatility"><div className="text-[10px] text-zinc-500 font-sans">Sharpe-like (μ−r)/σ</div><div className="text-zinc-100">{stats.sharpe_like.toFixed(2)}</div></div>
        <div className="bg-zinc-900/60 p-2 rounded border border-zinc-800/70" title="Asymmetry and fat tails of the distribution"><div className="text-[10px] text-zinc-500 font-sans">Skew / kurtosis</div><div className="text-zinc-100">{stats.skew.toFixed(2)} / {stats.kurt.toFixed(2)}</div></div>
      </div>

      {stats.call_price_bs != null && (
        <div className="bg-zinc-900/60 p-3 rounded-lg border border-zinc-800/70 font-mono">
          <div className="text-[11px] text-zinc-400 font-sans mb-1">Option value (K={metadata?.strike}, r={((metadata?.risk_free ?? 0) * 100).toFixed(1)}%)</div>
          <div className="grid grid-cols-2 gap-2 text-center">
            <div><div className="text-[10px] text-zinc-500 font-sans">CALL BS / MC</div><div className="text-emerald-300">${stats.call_price_bs?.toFixed(2)} / ${stats.call_price_mc?.toFixed(2)}</div></div>
            <div><div className="text-[10px] text-zinc-500 font-sans">PUT BS / MC</div><div className="text-sky-300">${stats.put_price_bs?.toFixed(2)} / ${stats.put_price_mc?.toFixed(2)}</div></div>
          </div>
        </div>
      )}

      {(metadata?.diversification_benefit as number | undefined) != null && (metadata?.symbols?.length ?? 0) > 0 && (
        <div className="bg-emerald-950/30 p-2.5 rounded border border-emerald-900/40 font-mono text-center" title="How much less tail risk the mix has vs. holding assets separately">
          <span className="text-[11px] text-emerald-300">Diversification saves {fmt$((metadata?.diversification_benefit as number) ?? 0)} off VaR95</span>
        </div>
      )}
    </div>
  );
};
