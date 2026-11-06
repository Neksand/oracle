import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, Cell } from 'recharts';

export interface DistributionHistProps {
  centers: number[];
  counts: number[];
  expected: number;
  p5: number;
  p95: number;
}

export const DistributionHist: React.FC<DistributionHistProps> = ({ centers, counts, expected, p5, p95 }) => {
  const data = centers.map((c, i) => ({ price: Math.round(c), n: counts[i] ?? 0, stress: c <= p5 || c >= p95 }));
  return (
    <div className="w-full h-64 bg-[#0c0d12] border border-zinc-800 rounded-xl p-3">
      <div className="text-[11px] text-zinc-400 mb-1">
        Terminal value distribution S<sub>T</sub> · red bars are tails beyond P5 / P95
      </div>
      <ResponsiveContainer width="100%" height="88%">
        <BarChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
          <CartesianGrid stroke="#1f1f23" strokeDasharray="3 3" />
          <XAxis dataKey="price" tick={{ fill: '#71717a', fontSize: 9 }} tickLine={false} minTickGap={24} tickFormatter={(v: number) => `$${v.toLocaleString()}`} />
          <YAxis tick={{ fill: '#71717a', fontSize: 10 }} tickLine={false} width={44} />
          <Tooltip contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', fontSize: 11 }} formatter={(v, _n, p) => [`${v} paths`, `$${Number(p?.payload?.price ?? 0).toLocaleString()}`]} />
          <ReferenceLine x={Math.round(expected)} stroke="#38bdf8" label={{ value: 'Mean', fill: '#38bdf8', fontSize: 10 }} />
          <Bar dataKey="n" name="Paths">
            {data.map((d, i) => (
              <Cell key={i} fill={d.stress ? '#f43f5e' : '#22d3ee'} opacity={d.stress ? 0.85 : 0.7} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
