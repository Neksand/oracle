"""ORACLE - Database API router: scenarios, risk joins, SQL export, security demo."""
from __future__ import annotations

import os
import subprocess  # noqa: S404 only used with argv list, no shell
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import queries
from app.db.database import DB_PATH, get_db
from app.db.orm_models import AuditLog, Scenario
from app.security import get_current_user
from app.db.orm_models import User

router = APIRouter(tags=["database"])


class AssetIn(BaseModel):
    symbol: str = Field(default="CUSTOM", max_length=24)
    s0: float = Field(default=100.0, gt=0, le=10_000_000)
    mu_ann: float = Field(default=0.08, ge=-1.0, le=3.0)
    sigma_ann: float = Field(default=0.25, ge=0.01, le=5.0)
    weight: float = Field(default=1.0, ge=0.0, le=10.0)


class ScenarioIn(BaseModel):
    name: str = Field(default="New scenario", max_length=128)
    description: str = Field(default="", max_length=2000)
    horizon_days: int = Field(default=90, ge=5, le=1260)
    trading_days: int = Field(default=252)
    num_paths: int = Field(default=8000, ge=500, le=50000)
    risk_free: float = Field(default=0.04, ge=-0.05, le=0.25)
    assets: List[AssetIn] = Field(default_factory=list)


class ScenarioLookupResponse(BaseModel):
    query: str
    username: str
    count: int
    scenarios: List[dict[str, Any]]


def _audit(db: Session, event: str, actor: str, detail: str, blocked: bool = False) -> None:
    db.add(AuditLog(event_type=event, actor=actor, detail=detail[:500], blocked=blocked))
    db.commit()


# ---------------------------------------------------------------------------
# Q1: lookup scenarios of a user (public demo endpoint, login-protected in prod)
# ---------------------------------------------------------------------------
@router.get("/db/scenarios/lookup", response_model=ScenarioLookupResponse,
            summary="Q1: SELECT scenarios by owner (lookup)")
async def db_lookup(username: str = "alice", db: Session = Depends(get_db)):
    return queries.lookup_user_scenarios(db, username)


# ---------------------------------------------------------------------------
# Q2: 3-table JOIN with GROUP BY aggregates
# ---------------------------------------------------------------------------
@router.get("/db/scenarios/risk-join",
            summary="Q2: JOIN scenario-assets-runs with GROUP BY aggregates")
async def db_risk_join(db: Session = Depends(get_db)):
    return queries.join_scenario_risk(db)


# ---------------------------------------------------------------------------
# Q3: INSERT new scenario (auth required; user_id taken from JWT, not from body)
# ---------------------------------------------------------------------------
@router.post("/db/scenarios",
             summary="Q3: INSERT scenario + assets in one transaction (JWT required)")
async def db_insert(payload: ScenarioIn,
                    db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required to save scenarios")
    created = queries.insert_scenario(db, user.id, payload.model_dump())
    _audit(db, "scenario_created", user.username, f"Scenario #{created['scenario_id']} '{created['name']}'")
    return created


@router.get("/db/scenarios", summary="GET my scenarios (JWT required)")
async def my_scenarios(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    rows = db.query(Scenario).filter(Scenario.user_id == user.id).order_by(Scenario.created_at.desc()).all()
    return [
        {"id": s.id, "name": s.name, "horizon_days": s.horizon_days,
         "assets": [{"symbol": a.symbol, "weight": a.weight} for a in s.assets]}
        for s in rows
    ]


# ---------------------------------------------------------------------------
# Export the whole DB as .sql script (appendix for the report)
# ---------------------------------------------------------------------------
@router.post("/db/export", summary="Export full database as .sql dump (text)")
async def db_export(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if not DB_PATH or not os.path.exists(DB_PATH):
        raise HTTPException(status_code=404, detail="Database file not found; run: python -m app.db.seed")
    proc = subprocess.run(
        ["sqlite3", DB_PATH, ".dump"],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0 or not proc.stdout:
        # fallback: pure-python dump via sqlalchemy iterate + triggers omitted
        content = _python_dump()
    else:
        content = proc.stdout
    _audit(db, "sql_export", user.username if user else "anonymous", "Exported oracle_dump.sql")
    return {"filename": "oracle_dump.sql", "bytes": len(content), "sql": content}


def _python_dump() -> str:
    """Minimal SQL dump when the sqlite3 CLI is unavailable."""
    import sqlite3

    lines = []
    conn = sqlite3.connect(DB_PATH)
    for line in conn.iterdump():
        lines.append(line.split(";")[0] + ";" if not line.endswith(";") else line)
    conn.close()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SECURITY SHOWDOWN: unsafe vs parameterized query, same user input
# ---------------------------------------------------------------------------
class InjectionProbe(BaseModel):
    user_input: str = Field(default="' OR 1=1 --", max_length=120)


@router.post("/security/sql_injection_demo",
             summary="Live demo: f-string SQL vs parameterized query")
async def sql_injection_demo(probe: InjectionProbe, db: Session = Depends(get_db)):
    unsafe = queries.unsafe_lookup_scenarios(db, probe.user_input)
    safe = queries.safe_lookup_scenarios(db, probe.user_input)
    attacked = unsafe["rows_leaked"] != safe["rows_leaked"]
    db.add(AuditLog(
        event_type="sql_injection_probe",
        actor="anonymous",
        detail=f"payload={probe.user_input!r} -> unsafe_rows={unsafe['rows_leaked']} safe_rows={safe['rows_leaked']}",
        blocked=True,
    ))
    db.commit()
    return {
        "provided_payload": probe.user_input,
        "unsafe": unsafe,
        "safe": safe,
        "verdict": ("BREACH: unsafe query leaked rows owned by other users"
                    if attacked else
                    "No leak observed for this payload (try the classic \"' OR 1=1 --\")"),
        "lesson": [
            "Unsafe: f-string concatenation lets the user close the quote and append OR 1=1 --.",
            "Safe: driver sends SQL template + user bytes separately (bound parameters).",
            "User input is DATA, never CODE. Same lesson as CSRF/XSS: separation of data and control plane.",
        ],
    }


@router.get("/security/audit_log", summary="Recent security events (JWT required)")
async def audit_log(limit: int = 15, db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)):
    rows = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(min(limit, 100)).all()
    return [
        {"id": r.id, "event_type": r.event_type, "actor": r.actor,
         "detail": r.detail, "blocked": r.blocked,
         "created_at": r.created_at.isoformat() if r.created_at else None}
        for r in rows
    ]
