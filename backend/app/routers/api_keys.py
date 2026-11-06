"""ORACLE - API keys + programmatic (machine-to-machine) API.

The web app uses JWT. Integrations use a long-lived API key sent as
`X-API-Key`, so the same product can be consumed remotely without a browser.

Keys are stored as SHA-256 hashes; the plaintext value is returned exactly
once, at creation time.
"""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.orm_models import ApiKey, User
from app.models import FinanceSimulateRequest
from app.finance_engine import simulate_single_asset
from app.security import (
    generate_api_key,
    get_current_user,
    hash_api_key,
    require_api_key,
)

router = APIRouter(tags=["api-keys"])


class KeyCreate(BaseModel):
    name: str = Field(default="default", min_length=1, max_length=64)


class KeyCreated(BaseModel):
    id: int
    name: str
    prefix: str
    api_key: str
    note: str = "Store this key now — it is shown only once."


class KeyInfo(BaseModel):
    id: int
    name: str
    prefix: str
    created_at: dt.datetime
    last_used_at: dt.datetime | None
    revoked: bool


@router.post("/keys", response_model=KeyCreated, status_code=201,
             summary="Create an API key (JWT required)")
async def create_key(req: KeyCreate, db: Session = Depends(get_db),
                     user: User = Depends(get_current_user)):
    raw = generate_api_key()
    row = ApiKey(
        user_id=user.id,
        name=req.name,
        prefix=raw[:12],
        key_hash=hash_api_key(raw),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return KeyCreated(id=row.id, name=row.name, prefix=row.prefix, api_key=raw)


@router.get("/keys", response_model=list[KeyInfo], summary="List your API keys (masked)")
async def list_keys(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    rows = db.query(ApiKey).filter(ApiKey.user_id == user.id).order_by(ApiKey.id.desc()).all()
    return [KeyInfo(id=r.id, name=r.name, prefix=r.prefix, created_at=r.created_at,
                    last_used_at=r.last_used_at, revoked=r.revoked) for r in rows]


@router.delete("/keys/{key_id}", status_code=204, summary="Revoke an API key")
async def revoke_key(key_id: int, db: Session = Depends(get_db),
                     user: User = Depends(get_current_user)):
    row = db.query(ApiKey).filter(ApiKey.id == key_id, ApiKey.user_id == user.id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Key not found")
    row.revoked = True
    db.commit()
    return None


# ---------------------------------------------------------------------------
# Public (key-protected) API — for remote integrations
# ---------------------------------------------------------------------------
public = APIRouter(tags=["public-api"])


@public.get("/v1/me", summary="Check key validity and see the owning account")
async def api_me(user: User = Depends(require_api_key)):
    return {"username": user.username, "role": user.role, "auth": "api_key"}


@public.get("/v1/health", summary="Service health for integrations")
async def api_health(user: User = Depends(require_api_key)):
    return {"status": "online", "system": "ORACLE_CORE_V3.1", "auth": "api_key"}


@public.post("/v1/simulate", summary="Run a single-asset Monte Carlo scenario")
async def api_simulate(req: FinanceSimulateRequest, user: User = Depends(require_api_key)):
    pos, col, paths, taxis, hist, stats, legacy, meta = simulate_single_asset(
        s0=req.s0, mu_ann=req.mu_ann, sigma_ann=req.sigma_ann,
        t_days=req.t_days, trading_days=req.trading_days,
        num_paths=req.num_paths, num_sample_paths=req.num_sample_paths,
        bins=req.bins, risk_free=req.risk_free, strike=req.strike,
        option_type=req.option_type, seed=req.seed, symbol=req.symbol,
    )
    return {
        "symbol": req.symbol,
        "paths": paths,
        "time_axis_days": taxis,
        "histogram": hist,
        "stats": stats,
        "statistics": legacy,
        "metadata": meta,
        "requested_by": user.username,
    }
