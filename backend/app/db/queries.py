"""ORACLE - The three real-user queries (Assignment 3, Task 3).

    Q1 LOOKUP : SELECT scenarios of one user ordered by creation time
    Q2 JOIN   : scenario -> assets -> aggregated run risk metrics
    Q3 INSERT : persist a new scenario together with its assets (transaction)

Also contains the SQL-injection teaching pair:

    unsafe_lookup_scenarios()   -- f-string SQL  (VULNERABLE, demo only)
    safe_lookup_scenarios()     -- parameterized query (the correct one)
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import desc, func, select, text
from sqlalchemy.orm import Session

from app.db.orm_models import Scenario, ScenarioAsset, SimulationRun, User


# ---------------------------------------------------------------------------
# Q1: simple lookup - find scenarios of a user
# ---------------------------------------------------------------------------
def lookup_user_scenarios(db: Session, username: str) -> dict[str, Any]:
    """SELECT s.*, u.username
       FROM scenarios s JOIN users u ON u.id = s.user_id
       WHERE u.username = :username
       ORDER BY s.created_at DESC;"""
    stmt = (
        select(Scenario, User.username)
        .join(User, User.id == Scenario.user_id)
        .where(User.username == username)
        .order_by(desc(Scenario.created_at))
    )
    rows = db.execute(stmt).all()
    return {
        "query": "lookup_user_scenarios (SELECT + JOIN owner + ORDER BY)",
        "username": username,
        "count": len(rows),
        "scenarios": [
            {
                "id": sc.id,
                "name": sc.name,
                "description": sc.description,
                "horizon_days": sc.horizon_days,
                "num_paths": sc.num_paths,
                "owner": owner,
                "created_at": sc.created_at.isoformat() if sc.created_at else None,
            }
            for sc, owner in rows
        ],
    }


# ---------------------------------------------------------------------------
# Q2: JOIN - per-scenario asset composition + aggregated risk of past runs
# ---------------------------------------------------------------------------
def join_scenario_risk(db: Session) -> dict[str, Any]:
    """SELECT sc.id, sc.name, COUNT(DISTINCT a.id) AS n_assets,
              COUNT(DISTINCT r.id) AS n_runs,
              MAX(r.expected_st)   AS best_expected_st,
              AVG(r.var95_loss)    AS avg_var95_loss,
              MAX(r.cvar95_loss)   AS worst_cvar95
       FROM scenarios sc
       LEFT JOIN scenario_assets r2 ...
       LEFT JOIN simulation_runs r ...
       GROUP BY sc.id;

    Implemented with three correlated aggregates under one GROUP BY.
    """
    stmt = (
        select(
            Scenario.id,
            Scenario.name,
            func.count(func.distinct(ScenarioAsset.id)).label("n_assets"),
            func.count(func.distinct(SimulationRun.id)).label("n_runs"),
            func.max(SimulationRun.expected_st).label("best_expected_st"),
            func.avg(SimulationRun.var95_loss).label("avg_var95_loss"),
            func.max(SimulationRun.cvar95_loss).label("worst_cvar95"),
        )
        .join(ScenarioAsset, ScenarioAsset.scenario_id == Scenario.id)
        .outerjoin(SimulationRun, SimulationRun.scenario_id == Scenario.id)
        .group_by(Scenario.id, Scenario.name)
        .order_by(func.count(func.distinct(SimulationRun.id)).desc())
    )
    rows = db.execute(stmt).all()
    return {
        "query": "join_scenario_risk (3-table JOIN + GROUP BY aggregates)",
        "rows": [
            {
                "scenario_id": r.id,
                "scenario": r.name,
                "n_assets": int(r.n_assets),
                "n_runs": int(r.n_runs),
                "best_expected_st": round(float(r.best_expected_st or 0), 2),
                "avg_var95_loss": round(float(r.avg_var95_loss or 0), 2),
                "worst_cvar95": round(float(r.worst_cvar95 or 0), 2),
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Q3: INSERT - create scenario + assets atomically (one transaction)
# ---------------------------------------------------------------------------
def insert_scenario(db: Session, user_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    sc = Scenario(
        user_id=user_id,
        name=str(payload.get("name", "Unnamed"))[:128],
        description=str(payload.get("description", ""))[:2000],
        horizon_days=int(payload["horizon_days"]),
        trading_days=int(payload.get("trading_days", 252)),
        num_paths=int(payload.get("num_paths", 8000)),
        risk_free=float(payload.get("risk_free", 0.04)),
    )
    for a in payload.get("assets", []):
        sc.assets.append(
            ScenarioAsset(
                symbol=str(a.get("symbol", "CUSTOM"))[:24],
                s0=float(a.get("s0", 100.0)),
                mu_ann=float(a.get("mu_ann", 0.08)),
                sigma_ann=float(a.get("sigma_ann", 0.25)),
                weight=float(a.get("weight", 1.0)),
            )
        )
    db.add(sc)
    db.commit()
    db.refresh(sc)
    return {
        "query": "insert_scenario (INSERT + FK cascade, single transaction)",
        "scenario_id": sc.id,
        "name": sc.name,
        "assets": [{"symbol": a.symbol, "weight": a.weight} for a in sc.assets],
    }


# ---------------------------------------------------------------------------
# CYBERSECURITY DEMO: unsafe vs safe equivalent query
# ---------------------------------------------------------------------------
def unsafe_lookup_scenarios(db: Session, user_input: str) -> dict[str, Any]:
    """*** INTENTIONALLY VULNERABLE - teaching example only ***

    Bad SQL built by f-string concatenation:
        "... WHERE u.username = '<input>'"
    An attacker sends  ' OR 1=1 --  and reads ALL users' scenarios.
    """
    sql = f"SELECT COUNT(*) FROM scenarios s JOIN users u ON u.id = s.user_id WHERE u.username = '{user_input}'"
    # raw connection: no parameter binding whatsoever
    count = db.execute(text(sql)).scalar_one()
    rows = db.execute(
        text(f"SELECT s.name FROM scenarios s JOIN users u ON u.id = s.user_id WHERE u.username = '{user_input}'")
    ).all()
    return {
        "version": "unsafe (f-string concatenation)",
        "executed_sql": sql,
        "rows_leaked": int(count),
        "scenario_names": [r[0] for r in rows],
    }


def safe_lookup_scenarios(db: Session, user_input: str) -> dict[str, Any]:
    """Parameterized twin of unsafe_lookup_scenarios.

    The driver transmits the SQL template with placeholders (:username) and
    the user bytes separately; user_input can never alter the parsed plan.
    """
    sql = ("SELECT s.name FROM scenarios s JOIN users u ON u.id = s.user_id "
           "WHERE u.username = :username")
    rows = db.execute(text(sql), {"username": user_input}).all()
    return {
        "version": "safe (parameterized / prepared statement)",
        "executed_sql": sql,
        "rows_leaked": len(rows),
        "scenario_names": [r[0] for r in rows],
    }
