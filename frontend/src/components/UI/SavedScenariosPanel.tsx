import React from 'react';
import { SavedScenario, FinanceParams } from '../../types';
import { Bookmark, Trash2, RotateCcw } from 'lucide-react';

export interface SavedScenariosPanelProps {
  scenarios: SavedScenario[];
  onRestore: (params: FinanceParams) => void;
  onClear: () => void;
  onRemove: (id: string) => void;
}

export const SavedScenariosPanel: React.FC<SavedScenariosPanelProps> = ({ scenarios, onRestore, onClear, onRemove }) => {
  if (scenarios.length === 0) return null;
  return (
    <div className="bg-[#121215] border border-zinc-800 rounded-xl p-5 space-y-3 text-zinc-200 text-xs">
      <div className="border-b border-zinc-800/80 pb-2.5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bookmark className="w-4 h-4 text-zinc-400" />
          <h3 className="font-semibold text-white text-sm">Bookmarked scenarios</h3>
        </div>
        <button onClick={onClear} className="text-zinc-500 hover:text-rose-300 flex items-center gap-1 text-[11px]" title="Remove all bookmarks">
          <Trash2 className="w-3.5 h-3.5" />Clear
        </button>
      </div>
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {scenarios.map((s) => (
          <div key={s.id} className="bg-zinc-900/60 border border-zinc-800/70 rounded-lg p-2.5 flex items-center gap-2">
            <div className="flex-1 min-w-0">
              <div className="text-zinc-200 font-medium truncate">{s.name}</div>
              <div className="text-[10px] text-zinc-500 font-mono truncate">
                E ${s.stats.expected_ST.toLocaleString('en-US', { maximumFractionDigits: 0 })} · VaR ${s.stats.var95_loss.toLocaleString('en-US', { maximumFractionDigits: 0 })} · {s.timestamp}
              </div>
            </div>
            <button onClick={() => onRestore(s.params)} title="Re-run this scenario"
              className="p-1.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200">
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
            <button onClick={() => onRemove(s.id)} title="Delete bookmark"
              className="p-1.5 rounded hover:bg-zinc-800 text-zinc-500 hover:text-rose-300">
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};
