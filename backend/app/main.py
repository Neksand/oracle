"""ORACLE - FastAPI Entrypoint.

Economic Scenario Engine: GBM Monte Carlo, portfolio risk,
option valuation, live market calibration.

Assignment 3 additions:
    - real data layer (SQLite + SQLAlchemy): scenarios, runs, market data
    - authentication (JWT + bcrypt) and security-showdown SQLi demo
    - research benchmarking (GBM vs Student-t vs Merton vs LSTM on internet data)
"""

import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from .routers.simulation import router as simulation_router
    from .routers.finance import router as finance_router
    from .routers.db import router as db_router
    from .routers.auth import router as auth_router
    from .routers.api_keys import router as api_keys_router, public as public_api_router
    from .routers.research import router as research_router
    from .db.database import init_db
    from .db.seed import seed as seed_db
except (ImportError, ValueError):
    from routers.simulation import router as simulation_router
    from routers.finance import router as finance_router
    from routers.db import router as db_router
    from routers.auth import router as auth_router
    from routers.api_keys import router as api_keys_router, public as public_api_router
    from routers.research import router as research_router
    from db.database import init_db
    from db.seed import seed as seed_db

app = FastAPI(
    title="ORACLE API",
    description="Economic Scenario Engine: GBM Monte Carlo, portfolio tail risk (VaR/CVaR), "
                "option valuation, live market calibration + persistent data layer "
                "(scenarios/runs/market_data) + auth + research benchmarks (LSTM).",
    version="3.1.0",
)

# CORS: explicit allow-list instead of "*" (assignment: unauthorized access)
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:4173,http://127.0.0.1:5173,"
        "https://neksand.github.io",
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lightweight in-memory rate limiter (per client IP): 120 req / 60 s
# ---------------------------------------------------------------------------
RATE_LIMIT_REQUESTS = 120
RATE_LIMIT_WINDOW_S = 60.0
_hits: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=RATE_LIMIT_REQUESTS))


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip = (request.client.host if request.client else "local") or "local"
    now = time.perf_counter()
    q = _hits[ip]
    while q and now - q[0] > RATE_LIMIT_WINDOW_S:
        q.popleft()
    if len(q) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            {"detail": "Rate limit exceeded (120 req/min). Slow down."},
            status_code=429,
            headers={"Retry-After": str(int(RATE_LIMIT_WINDOW_S))},
        )
    q.append(now)
    return await call_next(request)


# Mount routers under /api (main namespace) and root (legacy compatibility)
for r in (simulation_router, finance_router, db_router, auth_router, research_router, api_keys_router):
    app.include_router(r, prefix="/api")
    app.include_router(r)

# API-key protected integration surface (X-API-Key header)
app.include_router(public_api_router, prefix="/api")


@app.on_event("startup")
def startup() -> None:
    """Create tables and seed demo data on first launch."""
    init_db()
    try:
        seed_db()
    except Exception as exc:  # already-seeded DB (unique constraint) is fine
        print(f"[ORACLE] seed skipped: {type(exc).__name__}: {exc}")


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "online", "system": "ORACLE_CORE_V3.1"}
