import React from 'react';
import { FinanceResponse } from '../../types';
import { exportToCSV, exportToJSON, buildReportText } from '../../api/simulationApi';
import { Download, FileSpreadsheet, FileCode, Check, Bookmark, ClipboardCopy } from 'lucide-react';
import { useState } from 'react';

export interface ExportPanelProps {
  data: FinanceResponse | null;
  onSaveScenario?: () => void;
  savedScenariosCount?: number;
}

export const ExportPanel: React.FC<ExportPanelProps> = ({ data, onSaveScenario, savedScenariosCount = 0 }) => {
  const [msg, setMsg] = useState<string | null>(null);
  if (!data) return null;
  const flash = (t: string) => { setMsg(t); setTimeout(() => setMsg(null), 2500); };
  const sym = data.metadata.symbol ?? (data.metadata.symbols ?? []).join('+') ?? 'PORT';

  return (
    <div className="bg-[#121215] border border-zinc-800 rounded-xl p-5 space-y-3.5 text-zinc-200 text-xs">
      <div className="border-b border-zinc-800/80 pb-2.5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Download className="w-4 h-4 text-zinc-400" />
          <h3 className="font-semibold text-white text-sm">Export & history</h3>
        </div>
        {msg && <span className="flex items-center gap-1 text-[11px] text-emerald-400"><Check className="w-3.5 h-3.5" />{msg}</span>}
      </div>
      <div className="grid grid-cols-3 gap-2 text-center font-mono">
        <div className="bg-zinc-900/60 p-2 rounded-lg border border-zinc-800/70"><span className="text-[10px] text-zinc-500 block font-sans">Paths</span><span>{(data.metadata.total_points ?? 0).toLocaleString()}</span></div>
        <div className="bg-zinc-900/60 p-2 rounded-lg border border-zinc-800/70"><span className="text-[10px] text-zinc-500 block font-sans">Horizon</span><span>{data.metadata.t_days}d</span></div>
        <div className="bg-zinc-900/60 p-2 rounded-lg border border-zinc-800/70"><span className="text-[10px] text-zinc-500 block font-sans">Mean S<sub>T</sub></span><span>${data.stats.expected_ST.toLocaleString('en-US', { maximumFractionDigits: 0 })}</span></div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <button onClick={() => { exportToCSV(data, `oracle_${sym}_${Date.now()}.csv`); flash('CSV downloaded'); }}
          className="py-2 px-3 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 flex items-center justify-center gap-2" title="Download sampled price paths as a spreadsheet">
          <FileSpreadsheet className="w-3.5 h-3.5 text-zinc-400" />Paths CSV</button>
        <button onClick={() => { exportToJSON(data, `oracle_${sym}_${Date.now()}.json`); flash('JSON downloaded'); }}
          className="py-2 px-3 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 flex items-center justify-center gap-2" title="Download full metrics payload">
          <FileCode className="w-3.5 h-3.5 text-zinc-400" />Report JSON</button>
      </div>
      <button onClick={() => { void navigator.clipboard.writeText(buildReportText(data)); flash('Report copied'); }}
        className="w-full py-2 px-3 rounded-lg bg-zinc-950/60 hover:bg-zinc-900 border border-zinc-800 flex items-center justify-center gap-2" title="Copy a plain-text summary to clipboard">
        <ClipboardCopy className="w-3.5 h-3.5 text-zinc-400" />Copy text summary</button>
      {onSaveScenario && (
        <button onClick={onSaveScenario}
          className="w-full py-2 px-3 rounded-lg bg-zinc-950/60 hover:bg-zinc-900 border border-zinc-800 flex items-center justify-center gap-2" title="Bookmark this scenario below">
          <Bookmark className="w-3.5 h-3.5 text-zinc-400" />Bookmark scenario ({savedScenariosCount})</button>
      )}
    </div>
  );
};
