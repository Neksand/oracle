# ORACLE — Economic Scenario Engine

Frontend on GitHub Pages, backend (FastAPI) on Render.

- **App:** https://neksand.github.io/oracle/
- **API:** https://oracle-api.onrender.com/docs

## Demo login

`alice` / `Alice#Oracle26` (seeded automatically on first boot).

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

If Render assigns a URL different from `https://oracle-api.onrender.com`, set the repository
variable `VITE_API_BASE_URL` (Settings → Secrets and variables → Actions → Variables, value
`https://<your-service>.onrender.com/api`) and re-run the Pages workflow.

## Notes

- `CORS_ORIGINS` env var on the backend controls allowed browser origins (defaults include
  `https://neksand.github.io`).
- Render's filesystem is ephemeral: the SQLite demo DB lives in `/tmp` and is re-seeded on
  every deploy/restart. Data you create in the UI will not survive a restart.
