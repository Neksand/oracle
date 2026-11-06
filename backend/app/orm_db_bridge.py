"""ORACLE - Bridge between finance runs and the persistent data layer.

Every executed Monte-Carlo run is INSERTed into simulation_runs
(single-row transaction, best-effort: analytics never fails because of storage).
"""
from __future__ import annotations

from typing import Any

from app.db.database import SessionLocal
from app.db.orm_models import AuditLog, SimulationRun


def persist_simulation_run(meta: dict[str, Any], stats: dict[str, Any]) -> None:
    """INSERT one SimulationRun row; mirrors the request echo in metadata."""
    try:
        s0 = meta.get("s0", meta.get("s0_port", 100.0))
        symbols = [meta["symbol"]] if meta.get("symbol") else meta.get("symbols", [])
        run = SimulationRun(
            scenario_id=None,
            run_uid=meta.get("run_id") or uuid4hex()[:12],
            kind="portfolio" if len(symbols) > 1 else "asset",
            symbols_json=symbols,
            s0=float(s0),
            mu_ann=float(meta.get("mu_ann", meta.get("mu_port", 0.08)) or 0.0),
            sigma_ann=float(meta.get("sigma_ann", meta.get("sig_port", 0.2)) or 0.0),
            t_days=int(meta.get("t_days", 90)),
            num_paths=int(meta.get("num_paths", meta.get("total_points", 8000))),
            seed=meta.get("seed"),
            expected_st=float(stats.get("expected_ST", 0.0)),
            var95_loss=float(stats.get("var95_loss", 0.0)),
            cvar95_loss=float(stats.get("cvar95_loss", 0.0)),
            var99_loss=float(stats.get("var99_loss", 0.0)),
            prob_decline=float(stats.get("prob_decline", 0.0)),
            execution_ms=float(meta.get("execution_time", 0.0)),
        )
        with SessionLocal() as session:
            session.add(run)
            session.add(AuditLog(
                event_type="simulation_run", actor="anonymous",
                detail=f"{run.kind} {symbols} {run.t_days}d seed={run.seed}",
            ))
            session.commit()
    except Exception as exc:  # storage is never on the trading path
        print(f"[ORACLE] persist_simulation_run skipped: {exc}")


def uuid4hex() -> str:
    import uuid as _uuid

    return _uuid.uuid4().hex
