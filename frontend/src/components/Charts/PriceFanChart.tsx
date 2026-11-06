import React, { useMemo } from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts';

export interface PriceFanChartProps {
  samplePaths: number[][];
  timeAxis: number[];
  s0: number;
  p5?: number;
  p95?: number;
  history?: number[] | null;
}

/** 2D fan: subsampled trajectories + median + P10-P90 band + optional history tail. */
export const PriceFanChart: React.FC<PriceFanChartProps> = ({ samplePaths, timeAxis, s0, p5, p95, history }) => {
  const data = useMemo(() => {
    const n = timeAxis.length;
    if (!n) return [];
    const cols: number[][] = [];
    for (let j = 0; j < n; j++) {
      const col = samplePaths.map((p) => p[j]).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
      cols.push(col);
    }
    const q = (arr: number[], k: number) => (arr.length ? arr[Math.min(arr.length - 1, Math.floor(arr.length * k))] : NaN);
    return timeAxis.map((t, j) => {
      const col = cols[j];
      const row: Record<string, number> = { day: Math.round(t), p10: q(col, 0.1), p50: q(col, 0.5), p90: q(col, 0.9) };
      for (let k = 0; k < Math.min(12, samplePaths.length); k += 4) row[`t${k}`] = samplePaths[k][j];
      return row;
    });
  }, [samplePaths, timeAxis]);

  const histTail = useMemo(() => {
    if (!history || history.length < 2) return [];
    const tail = history.slice(-60);
    return tail.map((v, i) => ({ day: -tail.length + i, hist: v }));
  }, [history]);

  // Recharts needs ONE dataset on the chart; per-<Line data> silently renders
  // nothing for the fan. Merge history rows (negative days) with fan rows.
  const chartData = useMemo(
    () => [...histTail, ...data] as Array<Record<string, number>>,
    [histTail, data],
  );

  if (!data.length) return <div className="text-zinc-500 text-xs p-4">No trajectories to display yet — run a scenario first.</div>;

  return (
    <div className="w-full h-72 bg-[#0c0d12] border border-zinc-800 rounded-xl p-3">
      <div className="text-[11px] text-zinc-400 mb-1 font-sans">
        Price fan · P10–P90 band + median · S₀ = ${s0.toLocaleString('en-US')}
        <span className="text-zinc-600"> — most paths should stay inside the band</span>
      </div>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
          <CartesianGrid stroke="#1f1f23" strokeDasharray="3 3" />
          <XAxis dataKey="day" tick={{ fill: '#71717a', fontSize: 10 }} tickLine={false} type="number" domain={['dataMin', 'dataMax']} />
          <YAxis tick={{ fill: '#71717a', fontSize: 10 }} tickLine={false} width={72} domain={['auto', 'auto']} tickFormatter={(v: number) => `$${Math.round(v).toLocaleString()}`} />
          <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', fontSize: 11 }} labelFormatter={(v) => `Day ${v}`} formatter={(v) => [`$${Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}`, '']} />
          <ReferenceLine y={s0} stroke="#e4e4e7" strokeDasharray="4 4" label={{ value: 'Today', fill: '#e4e4e7', fontSize: 10 }} />
          {p5 != null && <ReferenceLine y={p5} stroke="#f43f5e" strokeDasharray="2 4" label={{ value: 'P5 floor', fill: '#f43f5e', fontSize: 10 }} />}
          {p95 != null && <ReferenceLine y={p95} stroke="#22c55e" strokeDasharray="2 4" label={{ value: 'P95 cap', fill: '#22c55e', fontSize: 10 }} />}
          {histTail.length > 0 && <Line dataKey="hist" stroke="#a1a1aa" strokeWidth={1.5} isAnimationActive={false} dot={false} name="Recent history" connectNulls />}
          <Line dataKey="p90" stroke="#71717a" strokeWidth={1.2} isAnimationActive={false} dot={false} name="P90" connectNulls />
          <Line dataKey="p50" stroke="#38bdf8" strokeWidth={2.4} isAnimationActive={false} dot={false} name="Median path" connectNulls />
          <Line dataKey="p10" stroke="#71717a" strokeWidth={1.2} isAnimationActive={false} dot={false} name="P10" connectNulls />
          {[0, 4, 8].map((k) => (
            <Line key={k} dataKey={`t${k}`} stroke="#818cf8" strokeWidth={0.8} isAnimationActive={false} dot={false} opacity={0.6} name={`Sample ${k}`} connectNulls />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
