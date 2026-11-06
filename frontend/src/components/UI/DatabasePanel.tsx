import React, { useEffect, useState } from 'react';
import { Database, KeyRound, LogIn, LogOut, Loader2, ListOrdered } from 'lucide-react';
import { dbLookupScenarios, dbRiskJoin, login, dbMyScenarios, setAuthToken,
  ScenarioLookupRow, RiskJoinRow, SavedScenario,
} from '../../api/simulationApi';

/**
 * Database panel: the three real-user queries from Assignment 3
 * rendered as tables (screenshot source for the report).
 */
export const DatabasePanel: React.FC = () => {
  const [lookup, setLookup] = useState<ScenarioLookupRow[] | null>(null);
  const [riskRows, setRiskRows] = useState<RiskJoinRow[] | null>(null);
  const [username, setUsername] = useState('alice');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');

  const [user, setUser] = useState<string | null>(localStorage.getItem('oracle_user'));
  const [pw, setPw] = useState('');
  const [authMsg, setAuthMsg] = useState('');
  const [mySc, setMySc] = useState<SavedScenario[] | null>(null);

  const runQueries = async (u: string) => {
    setBusy(true);
    setErr('');
    try {
      const [l, j] = await Promise.all([dbLookupScenarios(u), dbRiskJoin()]);
      setLookup(l.scenarios);
      setRiskRows(j.rows);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    void runQueries(username);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const doLogin = async () => {
    setAuthMsg('Authenticating…');
    try {
      const res = await login(username, pw);
      localStorage.setItem('oracle_user', username);
      setUser(username);
      setAuthMsg(`JWT issued (role: ${res.role})`);
      setMySc(await dbMyScenarios());
    } catch (e) {
      setAuthMsg(`Login failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const doLogout = () => {
    setAuthToken(null);
    localStorage.removeItem('oracle_user');
    setUser(null);
    setAuthMsg('Signed out, token removed.');
    setMySc(null);
  };

  return (
    <div className="max-w-5xl mx-auto space-y-4">
      {/* Auth */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4">
        <div className="flex items-center gap-2 mb-3">
          <KeyRound className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-white">Authentication (JWT + bcrypt)</h2>
          <span className="ml-auto text-xs text-zinc-400">
            {user ? <>signed in: <strong className="text-emerald-400">{user}</strong></> : 'anonymous'}
          </span>
          {user ? (
            <button onClick={doLogout} className="flex items-center gap-1 text-xs bg-zinc-700 hover:bg-zinc-600 px-3 py-1.5 rounded-lg text-white">
              <LogOut className="w-3.5 h-3.5" />Sign out
            </button>
          ) : null}
        </div>
        <div className="flex gap-2 items-center">
          <input value={username} onChange={(e) => setUsername(e.target.value)}
            className="bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-xs font-mono w-40" placeholder="username" />
          <input value={pw} onChange={(e) => setPw(e.target.value)} type="password" disabled={!!user}
            className="bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-2 text-xs font-mono w-40" placeholder="password" />
          <button onClick={() => void doLogin()} disabled={!!user || !pw}
            className="flex items-center gap-1.5 bg-cyan-700 hover:bg-cyan-600 disabled:opacity-40 text-white text-xs px-4 py-2 rounded-lg">
            <LogIn className="w-4 h-4" />Sign in
          </button>
          <span className="text-[11px] text-zinc-500 font-mono">
            demo: alice / Alice#Oracle26 · bob / Bob#Oracle26 · admin / Admin#Oracle26
          </span>
        </div>
        {authMsg && <p className="text-[11px] mt-2 font-mono text-zinc-400">{authMsg}</p>}
        {mySc && (
          <div className="mt-3">
            <p className="text-[11px] text-zinc-300 mb-1 font-semibold">My scenarios (SELECT by JWT identity):</p>
            <div className="flex flex-wrap gap-2">
              {mySc.map((s) => (
                <span key={s.id} className="text-[10px] bg-zinc-800 border border-zinc-700 rounded px-2 py-1 font-mono">
                  #{s.id} {s.name} · {s.horizon_days}d · {s.assets.map((a) => a.symbol).join('+')}
                </span>
              ))}
              {mySc.length === 0 && <span className="text-[11px] text-zinc-500">No scenarios yet.</span>}
            </div>
          </div>
        )}
      </div>

      {/* Query runner */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-4">
        <div className="flex items-center gap-2 mb-3">
          <ListOrdered className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-white">Three real-user queries (Task 3)</h2>
          <button onClick={() => void runQueries(username)} disabled={busy}
            className="ml-auto flex items-center gap-1.5 bg-zinc-700 hover:bg-zinc-600 text-white text-xs px-3 py-1.5 rounded-lg">
            {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Database className="w-3.5 h-3.5" />}
            Re-run
          </button>
        </div>
        {err && <p className="text-xs text-rose-400">{err}</p>}

        {lookup && (
          <div className="mb-4">
            <p className="text-[11px] text-zinc-400 mb-2 font-mono">
              Q1 · SELECT s.* FROM scenarios s JOIN users u ... WHERE u.username = &apos;{username}&apos; -- {lookup.length} rows
            </p>
            <div className="overflow-x-auto rounded-lg border border-zinc-800">
              <table className="w-full text-[11px]">
                <thead className="bg-zinc-950 text-zinc-400 font-mono">
                  <tr>{['id', 'name', 'owner', 'horizon d', 'paths', 'description'].map((h) => (
                    <th key={h} className="text-left px-3 py-2 border-b border-zinc-800">{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {lookup.map((s) => (
                    <tr key={s.id} className="odd:bg-zinc-900/40">
                      <td className="px-3 py-1.5 font-mono">{s.id}</td>
                      <td className="px-3 py-1.5 text-zinc-100">{s.name}</td>
                      <td className="px-3 py-1.5 font-mono text-cyan-300">{s.owner}</td>
                      <td className="px-3 py-1.5 font-mono">{s.horizon_days}</td>
                      <td className="px-3 py-1.5 font-mono">{s.num_paths}</td>
                      <td className="px-3 py-1.5 text-zinc-400">{s.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {riskRows && (
          <div>
            <p className="text-[11px] text-zinc-400 mb-2 font-mono">
              Q2 · SELECT sc.id, sc.name, COUNT(DISTINCT assets), AVG(runs.var95_loss) ... 3-table JOIN + GROUP BY
            </p>
            <div className="overflow-x-auto rounded-lg border border-zinc-800">
              <table className="w-full text-[11px]">
                <thead className="bg-zinc-950 text-zinc-400 font-mono">
                  <tr>{['scenario', 'assets', 'runs', 'best E[ST]', 'avg VaR95 loss', 'worst CVaR95'].map((h) => (
                    <th key={h} className="text-left px-3 py-2 border-b border-zinc-800">{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {riskRows.map((r) => (
                    <tr key={r.scenario_id} className="odd:bg-zinc-900/40">
                      <td className="px-3 py-1.5 text-zinc-100">{r.scenario}</td>
                      <td className="px-3 py-1.5 font-mono">{r.n_assets}</td>
                      <td className="px-3 py-1.5 font-mono">{r.n_runs}</td>
                      <td className="px-3 py-1.5 font-mono text-emerald-300">${r.best_expected_st.toLocaleString('en-US')}</td>
                      <td className="px-3 py-1.5 font-mono text-amber-300">${r.avg_var95_loss.toLocaleString('en-US')}</td>
                      <td className="px-3 py-1.5 font-mono text-rose-300">${r.worst_cvar95.toLocaleString('en-US')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-[10px] text-zinc-500 mt-2">
              Q3 (INSERT) is executed by the &laquo;persist&raquo; button after JWT sign-in — single ACID transaction with FK cascades; every simulation run also INSERTs a row automatically.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DatabasePanel;
