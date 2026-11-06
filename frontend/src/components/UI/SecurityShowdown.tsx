import React, { useState } from 'react';
import { ShieldAlert, ShieldCheck, PlayCircle, Database, Lock } from 'lucide-react';
import { runInjectionDemo, InjectionResult, exportSqlDump } from '../../api/simulationApi';

/**
 * Security Showdown: live unsafe-vs-parameterized SQL comparison
 * (Assignment 3, Cybersecurity task 25%).
 */
export const SecurityShowdown: React.FC = () => {
  const [payload, setPayload] = useState("' OR 1=1 --");
  const [result, setResult] = useState<InjectionResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [exportMsg, setExportMsg] = useState('');
  const ranOnce = React.useRef(false);

  const run = async () => {
    setBusy(true);
    try {
      setResult(await runInjectionDemo(payload));
    } catch (e) {
      setResult(null);
      setExportMsg(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  // Auto-run once with the classic payload so the screen is never empty (demo/screenshots)
  React.useEffect(() => {
    if (!ranOnce.current) {
      ranOnce.current = true;
      const t = setTimeout(() => void run(), 300);
      return () => clearTimeout(t);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const doExport = async () => {
    setExportMsg('Generating dump…');
    try {
      await exportSqlDump();
      setExportMsg('oracle_dump.sql downloaded — appendix for the report.');
    } catch (e) {
      setExportMsg(`Export failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-4">
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4">
        <div className="flex items-center gap-2 mb-3">
          <Database className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-white">SQL Injection Showdown — same data, two query styles</h2>
        </div>
        <p className="text-xs text-zinc-400 mb-3">
          The backend answers the <strong>same</strong> user input twice: once with a vulnerable
          f-string SQL query, once with a bound-parameter (prepared) statement. Try the classic
          <code className="mx-1 text-rose-300">' OR 1=1 --</code> to impersonate every user, or a username like <code className="text-emerald-300">alice</code>.
        </p>
        <div className="flex gap-2">
          <input
            value={payload}
            onChange={(e) => setPayload(e.target.value)}
            className="flex-1 bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-xs font-mono text-zinc-100 focus:outline-none focus:border-cyan-600"
            placeholder="enter SQL to be executed against both versions"
          />
          <button onClick={() => void run()} disabled={busy}
            className="flex items-center gap-1.5 bg-cyan-700 hover:bg-cyan-600 disabled:opacity-50 text-white text-xs font-medium px-4 rounded-lg">
            <PlayCircle className="w-4 h-4" />{busy ? 'Running…' : 'Run both queries'}
          </button>
        </div>
      </div>

      {result && (
        <div className="grid md:grid-cols-2 gap-4">
          <div className={`rounded-xl border p-4 ${result.unsafe.rows_leaked > result.safe.rows_leaked ? 'border-rose-800 bg-rose-950/30' : 'border-zinc-800 bg-zinc-900/70'}`}>
            <div className="flex items-center gap-2 mb-2">
              <ShieldAlert className="w-4 h-4 text-rose-400" />
              <h3 className="text-xs font-semibold text-rose-300">{result.unsafe.version}</h3>
            </div>
            <pre className="text-[10px] bg-black/60 rounded-lg p-3 overflow-x-auto text-rose-200 whitespace-pre-wrap">{result.unsafe.executed_sql}</pre>
            <p className="text-xs text-zinc-300 mt-2">
              Rows available to the attacker: <strong className="font-mono">{result.unsafe.rows_leaked}</strong>
              {result.unsafe.scenario_names.length > 0 && (
                <span className="block mt-1 font-mono text-[10px] text-rose-300">
                  leaked: {result.unsafe.scenario_names.join(' · ')}
                </span>
              )}
            </p>
          </div>
          <div className="rounded-xl border border-emerald-900/70 bg-emerald-950/20 p-4">
            <div className="flex items-center gap-2 mb-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <h3 className="text-xs font-semibold text-emerald-300">{result.safe.version}</h3>
            </div>
            <pre className="text-[10px] bg-black/60 rounded-lg p-3 overflow-x-auto text-emerald-200 whitespace-pre-wrap">{result.safe.executed_sql}</pre>
            <p className="text-xs text-zinc-300 mt-2">
              Rows visible: <strong className="font-mono">{result.safe.rows_leaked}</strong>
              {result.safe.scenario_names.length > 0 && (
                <span className="block mt-1 font-mono text-[10px] text-emerald-300">
                  returned: {result.safe.scenario_names.join(' · ')}
                </span>
              )}
            </p>
          </div>
        </div>
      )}

      {result && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4">
          <div className="flex items-center gap-2 mb-1">
            <Lock className="w-3.5 h-3.5 text-zinc-400" />
            <p className="text-xs font-semibold text-zinc-200">Verdict</p>
          </div>
          <p className={`text-xs mb-2 font-mono ${result.verdict.startsWith('BREACH') ? 'text-rose-400' : 'text-emerald-400'}`}>{result.verdict}</p>
          <ul className="text-[11px] text-zinc-400 list-disc list-inside space-y-1">
            {result.lesson.map((l) => <li key={l}>{l}</li>)}
          </ul>
        </div>
      )}

      <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4 flex items-center justify-between">
        <p className="text-xs text-zinc-400">
          Assignment artefact: export the working database as an SQL script for the report appendix.
        </p>
        <button onClick={() => void doExport()}
          className="flex items-center gap-1.5 bg-zinc-700 hover:bg-zinc-600 text-white text-xs px-3 py-2 rounded-lg">
          <Database className="w-4 h-4" />Download oracle_dump.sql
        </button>
      </div>
      {exportMsg && <p className="text-[11px] text-zinc-500 font-mono">{exportMsg}</p>}
    </div>
  );
};

export default SecurityShowdown;
