import React, { useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { ControlPanel } from './components/UI/ControlPanel';
import { StatisticsPanel } from './components/UI/StatisticsPanel';
import { ExportPanel } from './components/UI/ExportPanel';
import { SavedScenariosPanel } from './components/UI/SavedScenariosPanel';
import { ResearchPanel } from './components/UI/ResearchPanel';
import { ApiPanel } from './components/UI/ApiPanel';
import { AuthGate } from './components/Auth/AuthGate';
import { SceneSetup } from './components/Scene3D/SceneSetup';
import { ProbabilityCloud } from './components/Scene3D/ProbabilityCloud';
import { PriceFanChart } from './components/Charts/PriceFanChart';
import { DistributionHist } from './components/Charts/DistributionHist';
import { useSimulation } from './hooks/useSimulation';
import { logout } from './api/simulationApi';
import { ViewMode } from './types';
import { Compass, AlertCircle, Loader2, Box, LineChart as LineIcon, BarChart3, Braces, FlaskConical, Activity, LogOut, UserRound } from 'lucide-react';

type AppTab = 'simulator' | 'research' | 'api';

const APP_TABS: { id: AppTab; label: string; icon: typeof Box; hint: string }[] = [
  { id: 'simulator', label: 'Simulator', icon: Activity, hint: 'GBM scenario engine' },
  { id: 'research', label: 'Models', icon: FlaskConical, hint: 'Volatility model comparison' },
  { id: 'api', label: 'API', icon: Braces, hint: 'Programmatic access with API keys' },
];

export const App: React.FC = () => {
  const [user, setUser] = useState<string | null>(() => localStorage.getItem('oracle_user'));
  const sim = useSimulation();
  const { loading, error, data, quote, history, marketLoading } = sim;
  const [view, setView] = useState<ViewMode>('fan2d');
  const [tab, setTab] = useState<AppTab>(
    () => ((['simulator', 'research', 'api'] as const)
        .includes((new URLSearchParams(window.location.search).get('tab') ?? '') as AppTab)
      ? ((new URLSearchParams(window.location.search).get('tab') ?? 'simulator') as AppTab)
      : 'simulator'),
  );

  const echo = data?.metadata.request_echo;
  const runTime = data?.metadata.run_at ? new Date(data.metadata.run_at).toLocaleTimeString('en-GB') : '';
  const runKey = data?.metadata.run_id ?? `run-${sim.runCount}`;

  useEffect(() => {
    void sim.simulate();
    void sim.loadHistory('SPY', '1Y');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sym = data?.metadata.symbol ?? (data?.metadata.symbols ?? []).join(' + ') ?? '—';
  const s0 = data?.metadata.s0 ?? data?.metadata.s0_port ?? sim.params.s0;
  const tDays = data?.metadata.t_days ?? sim.params.t_days;

  if (!user) {
    return <AuthGate onAuthenticated={(u) => { setUser(u); }} />;
  }

  const signOut = () => {
    logout();
    setUser(null);
  };

  return (
    <div className="flex flex-col md:flex-row h-screen w-screen bg-[#09090b] text-zinc-100 overflow-hidden font-sans select-none">
      {tab === 'simulator' && (
      <aside className="w-full md:w-96 h-auto md:h-full max-h-[46vh] md:max-h-none flex flex-col p-3 md:p-4 space-y-3 md:space-y-4 overflow-y-auto bg-[#0c0d12] border-b md:border-b-0 md:border-r border-zinc-800/80 shrink-0 z-20 shadow-2xl">
        <ControlPanel
          params={sim.params} applied={sim.applied} loading={loading} marketLoading={marketLoading}
          seedLocked={sim.seedLocked} lastSeed={sim.lastSeed}
          onToggleSeedLock={sim.setSeedLocked}
          onReroll={() => void sim.reroll()}
          onSimulate={(p) => void sim.simulate(p)}
          onSimulatePortfolio={(p) => void sim.simulatePortfolio(p)}
          onLivePrice={(s) => void sim.loadQuote(s)}
          onAutoCalibrate={(s) => void sim.autoCalibrate(s)}
        />
        <StatisticsPanel stats={data?.stats ?? null} metadata={data?.metadata ?? null} />
        <ExportPanel data={data} onSaveScenario={() => sim.saveScenario()} savedScenariosCount={sim.savedScenarios.length} />
        <SavedScenariosPanel
          scenarios={sim.savedScenarios}
          onRestore={(p) => void sim.simulate(p)}
          onClear={sim.clearSavedScenarios}
          onRemove={sim.removeScenario}
        />
      </aside>
      )}

      <main className="flex-1 min-h-0 relative overflow-y-auto md:overflow-hidden flex flex-col">
        {/* App-level tabs (Assignment 3 screens) */}
        <div className="absolute top-4 left-6 z-10 flex items-center gap-1 bg-zinc-900/90 backdrop-blur-md rounded-lg border border-zinc-800 p-1" role="tablist" aria-label="App sections">
          {APP_TABS.map((t) => (
            <button key={t.id} role="tab" aria-selected={tab === t.id} title={t.hint}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs ${tab === t.id ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-zinc-200'}`}>
              <t.icon className="w-4 h-4" />{t.label}
            </button>
          ))}
        </div>

        <div className="absolute top-4 right-6 z-20 flex items-center gap-1 bg-zinc-900/90 backdrop-blur-md rounded-lg border border-zinc-800 p-1">
          <span className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-zinc-300">
            <UserRound className="w-3.5 h-3.5" />{user}
          </span>
          <button onClick={signOut} title="Sign out"
            className="flex items-center gap-1 px-2 py-1.5 rounded-md text-xs text-zinc-400 hover:text-white hover:bg-zinc-800">
            <LogOut className="w-3.5 h-3.5" /><span className="hidden sm:inline">Sign out</span>
          </button>
        </div>

        {tab !== 'simulator' && (
          <div className="flex-1 overflow-y-auto pt-20 px-6 pb-8">
            {tab === 'research' && <ResearchPanel />}
            {tab === 'api' && <ApiPanel />}
          </div>
        )}

        {tab === 'simulator' && (
        <div className="absolute top-16 right-6 z-10 flex items-center gap-2.5 text-xs">
          <div className="flex items-center gap-1 bg-zinc-900/90 backdrop-blur-md rounded-lg border border-zinc-800 p-1" role="tablist" aria-label="View switcher">
            {([
              { id: 'fan2d', label: 'Paths', icon: LineIcon, hint: 'Possible price trajectories over time' },
              { id: 'hist', label: 'Outcomes', icon: BarChart3, hint: 'Distribution of terminal values' },
              { id: 'dist3d', label: '3D regimes', icon: Box, hint: 'Economic regimes in 3D' },
            ] as { id: ViewMode; label: string; icon: typeof Box; hint: string }[]).map((v) => (
              <button key={v.id} role="tab" aria-selected={view === v.id} title={v.hint} onClick={() => setView(v.id)}
                className={`flex items-center gap-1 px-2.5 py-1.5 rounded-md ${view === v.id ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-zinc-200'}`}>
                <v.icon className="w-3.5 h-3.5" />{v.label}
              </button>
            ))}
          </div>
          <div className="hidden md:flex items-center gap-1.5 bg-zinc-900/90 px-3 py-1.5 rounded-lg border border-zinc-800 text-zinc-300 font-mono" title={sim.backendOnline ? `Backend connected (${sim.backendSystem})` : 'Backend not reachable — run uvicorn on :8000'}>
            <span className={`w-2 h-2 rounded-full ${sim.backendOnline == null ? 'bg-zinc-500 animate-pulse' : sim.backendOnline ? 'bg-emerald-400' : 'bg-rose-500'}`} />
            {quote ? `${quote.symbol} $${quote.price.toLocaleString('en-US')} · ${quote.source}` : `${sym} · S₀ $${Number(s0).toLocaleString('en-US')}`}
          </div>
          {data && (
            <div className="flex items-center gap-1.5 bg-zinc-900/90 px-3 py-1.5 rounded-lg border border-zinc-800 font-mono" title="Particles in the 3D regime cloud">
              <Compass className="w-3.5 h-3.5 text-zinc-400" />
              {(data.positions.length / 3).toLocaleString()} pts · {tDays}d
            </div>
          )}
        </div>
        )}

        {error && (
          <div className="absolute top-16 left-6 right-6 z-30 flex items-center gap-3 p-4 bg-zinc-900/95 border border-rose-900/60 rounded-xl text-rose-200 text-xs">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <div className="flex-1">
              <p className="font-medium text-white">Could not run the calculation</p>
              <p className="text-zinc-400">{error} — check that the backend is running on :8000.</p>
            </div>
          </div>
        )}
        {tab === 'simulator' && loading && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm pointer-events-none">
            <Loader2 className="w-8 h-8 text-zinc-300 animate-spin" />
            <p className="mt-4 text-zinc-200 text-xs tracking-wider uppercase">Simulating 8,000 market paths…</p>
          </div>
        )}

        {tab === 'simulator' && (
        <>
        {!data && !loading && !error && (
          <div className="absolute inset-0 z-10 flex items-center justify-center p-8 pointer-events-none">
            <div className="max-w-md bg-zinc-900/90 border border-zinc-800 p-6 rounded-2xl">
              <h2 className="text-sm font-semibold text-white">Welcome to ORACLE</h2>
              <ol className="text-xs text-zinc-400 mt-2 space-y-1.5 list-decimal list-inside">
                <li>Click a ticker (try <strong>SPY</strong> or <strong>BTC-USD</strong>) in the left panel.</li>
                <li>Press <strong>Auto-fit</strong> to calibrate from real market history.</li>
                <li>Hit <strong>Run scenarios</strong> — read the fan, the outcomes, and the tail risk.</li>
              </ol>
            </div>
          </div>
        )}

        <div className="flex-1 pt-16 px-6 pb-6 overflow-y-auto">
          {data && view !== 'dist3d' && (
            <div className="max-w-5xl mx-auto space-y-4">
              <div className="text-xs text-zinc-400 bg-zinc-900/60 border border-zinc-800/70 rounded-lg px-3 py-2" title={echo ? `Built from ${echo.kind === 'asset' ? `${echo.symbol} · S₀ $${echo.s0} · μ ${((echo.mu_ann ?? 0) * 100).toFixed(1)}% · σ ${((echo.sigma_ann ?? 0) * 100).toFixed(0)}%` : `${(echo.symbols ?? []).join(' + ')}`} · ${echo.t_days}d · seed ${data.metadata.seed ?? '—'} · run ${data.metadata.run_id ?? '—'} at ${runTime}` : 'Run provenance'}>
                <strong className="text-zinc-200">{sym}</strong>
                {' '}· {tDays}-day outlook · mean <strong className="text-zinc-100 font-mono">${data.stats.expected_ST.toLocaleString('en-US', { maximumFractionDigits: 0 })}</strong>
                {' '}· likely range <span className="font-mono">[${data.stats.p5_ST.toLocaleString('en-US', { maximumFractionDigits: 0 })}, ${data.stats.p95_ST.toLocaleString('en-US', { maximumFractionDigits: 0 })}]</span>
                {' '}· tail loss (VaR95) <span className="font-mono text-rose-300">${data.stats.var95_loss.toLocaleString('en-US', { maximumFractionDigits: 0 })}</span>
                <span className="text-zinc-500 font-mono"> · run #{sim.runCount}{runTime ? ` · ${runTime}` : ''}{sim.seedLocked ? ' · seed locked' : ''}</span>
              </div>
              {view === 'fan2d' && (
                <PriceFanChart samplePaths={data.sample_paths} timeAxis={data.time_axis_days}
                  s0={s0 ?? 100} p5={data.stats.p5_ST} p95={data.stats.p95_ST} history={history} />
              )}
              {view === 'hist' && (
                <DistributionHist centers={data.histogram.centers} counts={data.histogram.counts}
                  expected={data.stats.expected_ST} p5={data.stats.p5_ST} p95={data.stats.p95_ST} />
              )}
              <div className="h-64 rounded-xl overflow-hidden border border-zinc-800 relative">
                <Canvas camera={{ position: [15, 15, 15], fov: 60 }} gl={{ antialias: true }}>
                  <color attach="background" args={['#09090b']} />
                  <SceneSetup />
                  <ProbabilityCloud positions={data.positions} colors={data.colors} runKey={runKey} />
                </Canvas>
                <button onClick={() => setView('dist3d')}
                  className="absolute bottom-3 right-3 text-[11px] bg-zinc-900/90 border border-zinc-700 rounded px-2 py-1 text-zinc-300 hover:text-white" title="Open full 3D regime view">
                  Expand 3D view
                </button>
              </div>
            </div>
          )}
          {(!data || view === 'dist3d') && (
            <Canvas camera={{ position: [15, 15, 15], fov: 60 }} gl={{ antialias: true }} className="w-full h-full">
              <color attach="background" args={['#09090b']} />
              <SceneSetup />
              {data && <ProbabilityCloud positions={data.positions} colors={data.colors} runKey={runKey} />}
            </Canvas>
          )}
        </div>

        {view === 'dist3d' && (
          <div className="absolute bottom-4 left-6 z-10 text-[11px] text-zinc-400 bg-zinc-900/90 border border-zinc-800 rounded-lg px-3 py-2">
            Y-axis = capital level (log-return) · X/Z = market noise · Drag to rotate · Scroll to zoom
          </div>
        )}
        </>
        )}
      </main>
    </div>
  );
};

export default App;
