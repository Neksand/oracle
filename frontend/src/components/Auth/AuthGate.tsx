import React, { useState } from 'react';
import { Activity, Loader2, LockKeyhole, Mail, UserRound, ArrowRight, ShieldCheck } from 'lucide-react';
import { apiErrorMessage, login, register } from '../../api/simulationApi';

type Mode = 'login' | 'register';

interface Props {
  onAuthenticated: (username: string) => void;
}

/**
 * Product-grade auth gate: the app is not usable until the visitor signs in.
 * Login is throttled server-side (3 failed attempts -> 5 minute lock).
 */
export const AuthGate: React.FC<Props> = ({ onAuthenticated }) => {
  const [mode, setMode] = useState<Mode>('login');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError('');
    try {
      const res = mode === 'login'
        ? await login(username.trim(), password)
        : await register(username.trim(), email.trim(), password);
      onAuthenticated(res.username);
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const switchMode = (next: Mode) => {
    setMode(next);
    setError('');
    setPassword('');
  };

  return (
    <div className="min-h-screen w-screen bg-[#09090b] text-zinc-100 flex items-center justify-center p-4 font-sans">
      <div className="w-full max-w-md">
        <div className="flex items-center gap-3 mb-8 justify-center">
          <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Activity className="w-6 h-6 text-black" />
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">ORACLE</h1>
            <p className="text-xs text-zinc-500">Economic Scenario Engine</p>
          </div>
        </div>

        <div className="rounded-2xl border border-zinc-800 bg-[#0c0d12] p-6 shadow-2xl">
          <div className="flex gap-1 mb-6 p-1 rounded-xl bg-zinc-900/80 border border-zinc-800">
            {(['login', 'register'] as Mode[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => switchMode(m)}
                className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                  mode === m ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                {m === 'login' ? 'Sign in' : 'Create account'}
              </button>
            ))}
          </div>

          <form onSubmit={submit} className="space-y-3">
            <label className="block">
              <span className="text-xs text-zinc-400 mb-1.5 flex items-center gap-1.5">
                <UserRound className="w-3.5 h-3.5" /> Username
              </span>
              <input
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                required
                minLength={3}
                maxLength={32}
                placeholder="your username"
                className="w-full rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-2.5 text-sm outline-none focus:border-cyan-500/70 focus:ring-2 focus:ring-cyan-500/10"
              />
            </label>

            {mode === 'register' && (
              <label className="block">
                <span className="text-xs text-zinc-400 mb-1.5 flex items-center gap-1.5">
                  <Mail className="w-3.5 h-3.5" /> Email
                </span>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                  required
                  placeholder="you@example.com"
                  className="w-full rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-2.5 text-sm outline-none focus:border-cyan-500/70 focus:ring-2 focus:ring-cyan-500/10"
                />
              </label>
            )}

            <label className="block">
              <span className="text-xs text-zinc-400 mb-1.5 flex items-center gap-1.5">
                <LockKeyhole className="w-3.5 h-3.5" /> Password
              </span>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                required
                minLength={8}
                placeholder={mode === 'register' ? 'min 8 chars, letter + digit' : 'your password'}
                className="w-full rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-2.5 text-sm outline-none focus:border-cyan-500/70 focus:ring-2 focus:ring-cyan-500/10"
              />
            </label>

            {error && (
              <div className="rounded-lg border border-red-900/60 bg-red-950/30 px-3 py-2 text-xs text-red-300">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={busy}
              className="w-full mt-2 flex items-center justify-center gap-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-60 text-black font-semibold text-sm py-2.5 transition-colors"
            >
              {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <ArrowRight className="w-4 h-4" />}
              {busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}
            </button>
          </form>

          <div className="mt-5 flex items-start gap-2 text-[11px] leading-relaxed text-zinc-500">
            <ShieldCheck className="w-3.5 h-3.5 mt-0.5 text-emerald-500/80 shrink-0" />
            <span>
              Passwords are stored as bcrypt hashes. After 3 failed attempts the account is locked for 5 minutes.
              Your session stays on this device only.
            </span>
          </div>
        </div>

        <p className="text-center text-[11px] text-zinc-600 mt-6">
          ORACLE · GBM scenarios, VaR/CVaR, option pricing, volatility research
        </p>
      </div>
    </div>
  );
};
