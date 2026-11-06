import React, { useEffect, useState } from 'react';
import { Braces, Copy, KeyRound, Loader2, Plus, Trash2, Check } from 'lucide-react';
import {
  apiErrorMessage, createApiKey, listApiKeys, revokeApiKey, type ApiKeyInfo,
} from '../../api/simulationApi';

/**
 * API access panel: lets a signed-in user mint/revoke API keys and copy a
 * ready-to-run curl example. The key is shown once, at creation.
 */
export const ApiPanel: React.FC = () => {
  const [keys, setKeys] = useState<ApiKeyInfo[]>([]);
  const [name, setName] = useState('integration');
  const [fresh, setFresh] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const [copied, setCopied] = useState(false);

  const refresh = async () => {
    try {
      setKeys(await listApiKeys());
    } catch (e) {
      setErr(apiErrorMessage(e));
    }
  };

  useEffect(() => { void refresh(); }, []);

  const create = async () => {
    setBusy(true); setErr(''); setCopied(false);
    try {
      const created = await createApiKey(name.trim() || 'integration');
      setFresh(created.api_key);
      await refresh();
    } catch (e) {
      setErr(apiErrorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const revoke = async (id: number) => {
    try {
      await revokeApiKey(id);
      await refresh();
    } catch (e) {
      setErr(apiErrorMessage(e));
    }
  };

  const copy = async (value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setErr('Clipboard blocked by the browser — select the key manually.');
    }
  };

  const curlExample = `curl -X POST ${window.location.origin}/api/v1/simulate \\
  -H "X-API-Key: ${fresh ?? '<your-key>'}" \\
  -H "Content-Type: application/json" \\
  -d '{"symbol":"SPY","s0":585,"mu_ann":0.09,"sigma_ann":0.17,"t_days":30,"num_paths":1000}'`;

  return (
    <div className="max-w-4xl mx-auto space-y-4">
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 p-5">
        <div className="flex items-center gap-2 mb-1">
          <Braces className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-white">Programmatic access</h2>
        </div>
        <p className="text-xs text-zinc-400 mb-4">
          Create a key to call ORACLE from scripts or another service. Keys are stored hashed and shown once.
        </p>

        <div className="flex gap-2">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="key name (e.g. notebook)"
            className="flex-1 rounded-lg bg-zinc-950 border border-zinc-800 px-3 py-2 text-sm outline-none focus:border-cyan-500/70"
          />
          <button
            onClick={create}
            disabled={busy}
            className="flex items-center gap-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-60 text-black text-sm font-semibold px-3 py-2"
          >
            {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
            Create key
          </button>
        </div>

        {err && (
          <div className="mt-3 rounded-lg border border-red-900/60 bg-red-950/30 px-3 py-2 text-xs text-red-300">
            {err}
          </div>
        )}

        {fresh && (
          <div className="mt-4 rounded-lg border border-emerald-900/60 bg-emerald-950/20 p-3">
            <p className="text-[11px] text-emerald-300 mb-2">Copy this key now — it will not be shown again.</p>
            <div className="flex items-center gap-2">
              <code className="flex-1 text-xs text-emerald-200 bg-black/40 rounded px-2 py-1.5 overflow-x-auto">
                {fresh}
              </code>
              <button onClick={() => void copy(fresh)} className="text-emerald-300 hover:text-emerald-200 p-1.5">
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              </button>
            </div>
          </div>
        )}

        <div className="mt-5">
          <h3 className="text-xs font-semibold text-zinc-300 mb-2 flex items-center gap-2">
            <KeyRound className="w-3.5 h-3.5" /> Your keys
          </h3>
          {keys.length === 0 ? (
            <p className="text-xs text-zinc-500">No keys yet.</p>
          ) : (
            <div className="space-y-1.5">
              {keys.map((k) => (
                <div key={k.id} className="flex items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-950/60 px-3 py-2">
                  <code className="text-xs text-zinc-300">{k.prefix}…</code>
                  <span className="text-xs text-zinc-500">{k.name}</span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded ${k.revoked ? 'bg-red-950 text-red-300' : 'bg-emerald-950 text-emerald-300'}`}>
                    {k.revoked ? 'revoked' : 'active'}
                  </span>
                  <span className="ml-auto text-[10px] text-zinc-600">
                    {k.last_used_at ? `last used ${new Date(k.last_used_at).toLocaleDateString()}` : 'never used'}
                  </span>
                  {!k.revoked && (
                    <button onClick={() => void revoke(k.id)} className="text-zinc-500 hover:text-red-400 p-1">
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="mt-5 rounded-lg border border-zinc-800 bg-black/40 p-3">
          <p className="text-[11px] text-zinc-400 mb-2">Example request</p>
          <pre className="text-[11px] leading-relaxed text-zinc-300 overflow-x-auto">{curlExample}</pre>
        </div>
      </div>
    </div>
  );
};
