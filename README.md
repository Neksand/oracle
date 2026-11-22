# ORACLE — Economic Scenario Engine

Frontend on GitHub Pages, backend (FastAPI) on Render.

- **App:** https://neksand.github.io/oracle/
- **API:** https://oracle-api-rvuq.onrender.com/docs

## Demo login

`admin` / `Admin#Oracle26` (role: admin) or `alice` / `Alice#Oracle26` (role: user).
Both are seeded automatically on first boot.

## MicroVol Studio — 100-parameter CPU volatility model

The `MicroVol` tab trains a 100-parameter volatility model directly in the browser session:
19-parameter calendar-HAR backbone + a 2-unit GRU residual, CPU-only, no CUDA.

- Backend: `POST /api/research/microvol` with `{"symbol": "SPY", "lookback_days": 750}`.
- Frontend: `frontend/src/components/UI/MicroVolPanel.tsx`.
- Model: `backend/app/research/microvol.py`.
- Paper benchmark (Table III): 100 parameters, 788x fewer than the 78,785-parameter RCM,
  non-inferior within a 0.02 QLIKE margin, seven-year bootstrap interval excludes zero.
- Live demo uses a close-only daily-variance target; its absolute QLIKE is not comparable
  to the paper's full HAR-X benchmark. The panel labels this explicitly.

![MicroVol Studio](docs/microvol-studio.png)

## Local run

```bash
# backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# frontend
cd frontend && npm install && npm run dev   # http://localhost:5173
```

Vite proxies `/api` and `/health` to `127.0.0.1:8000`, so no env vars are needed locally.

## Deploy

1. **Frontend** — push to `main`; `.github/workflows/pages.yml` builds `frontend/` and publishes it to Pages.
2. **Backend** — Render Blueprint ([render.yaml](render.yaml)):
   https://render.com/deploy?repo=https://github.com/Neksand/oracle
   The free plan sleeps after 15 min idle; the first request after that takes ~30 s.

The Render service is `oracle-api-rvuq` (the plain `oracle-api` subdomain was already taken).
Its URL is set as the repository variable `VITE_API_BASE_URL`
(Settings → Secrets and variables → Actions → Variables); the Pages workflow bakes it into
the build, so changing the backend URL only requires updating that variable and re-running
the workflow.

## Notes

- `CORS_ORIGINS` env var on the backend controls allowed browser origins (defaults include
  `https://neksand.github.io`).
- Render's filesystem is ephemeral: the SQLite demo DB lives in `/tmp` and is re-seeded on
  every deploy/restart. Data you create in the UI will not survive a restart.
