"""ORACLE - Finance & Market API router."""
import datetime
import uuid
from typing import List
import numpy as np
from fastapi import APIRouter, HTTPException, Query, status

try:
    from ..orm_db_bridge import persist_simulation_run
except ImportError:
    from app.orm_db_bridge import persist_simulation_run

try:
    from ..models import (
        FinanceSimulateRequest, FinanceSimulateResponse, MoneyStatistics,
        SimulationStatistics, HistogramPayload, PortfolioSimulateRequest,
        PortfolioSimulateResponse, QuoteResponse, HistoryResponse,
        CalibrateRequest, CalibrateResponse,
    )
    from ..finance_engine import simulate_single_asset, simulate_portfolio, _colorize
    from ..finance_engine import _money_stats
    from ..market import fetch_quote, fetch_history, calibrate
except (ImportError, ValueError):
    from app.models import (
        FinanceSimulateRequest, FinanceSimulateResponse, MoneyStatistics,
        SimulationStatistics, HistogramPayload, PortfolioSimulateRequest,
        PortfolioSimulateResponse, QuoteResponse, HistoryResponse,
        CalibrateRequest, CalibrateResponse,
    )
    from app.finance_engine import simulate_single_asset, simulate_portfolio, _colorize
    from app.finance_engine import _money_stats
    from app.market import fetch_quote, fetch_history, calibrate

router = APIRouter(tags=["finance"])


@router.post("/finance/simulate", response_model=FinanceSimulateResponse,
             summary="GBM economic scenario (single asset)")
async def finance_simulate(req: FinanceSimulateRequest) -> FinanceSimulateResponse:
    try:
        pos, col, paths, taxis, hist, stats, legacy, meta = simulate_single_asset(
            s0=req.s0, mu_ann=req.mu_ann, sigma_ann=req.sigma_ann,
            t_days=req.t_days, trading_days=req.trading_days,
            num_paths=req.num_paths, num_sample_paths=req.num_sample_paths,
            bins=req.bins, risk_free=req.risk_free, strike=req.strike,
            option_type=req.option_type, seed=req.seed, symbol=req.symbol,
        )
        meta["run_id"] = uuid.uuid4().hex[:12]
        meta["run_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta["seed"] = req.seed
        meta["request_echo"] = {
            "kind": "asset", "symbol": req.symbol, "s0": req.s0,
            "mu_ann": req.mu_ann, "sigma_ann": req.sigma_ann,
            "t_days": req.t_days, "strike": req.strike,
        }
        persist_simulation_run(meta, stats)
        return FinanceSimulateResponse(
            positions=pos, colors=col, sample_paths=paths, time_axis_days=taxis,
            histogram=HistogramPayload(**hist), stats=MoneyStatistics(**stats),
            statistics=SimulationStatistics(**legacy), metadata=meta,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Finance computation failure: {exc}") from exc


@router.post("/finance/portfolio", response_model=PortfolioSimulateResponse,
             summary="Correlated multi-asset portfolio scenario")
async def portfolio_simulate(req: PortfolioSimulateRequest) -> PortfolioSimulateResponse:
    try:
        if len(req.assets) < 2:
            raise HTTPException(status_code=400, detail="Provide at least 2 assets.")
        if len(req.assets) > 6:
            raise HTTPException(status_code=400, detail="Max 6 assets.")
        s0 = [a.s0 for a in req.assets]
        mu = [a.mu_ann for a in req.assets]
        sg = [a.sigma_ann for a in req.assets]
        w = [a.weight for a in req.assets]
        n = len(req.assets)
        if req.corr is not None:
            corr = req.corr
        else:
            corr = [[1.0 if i == j else req.corr_level for j in range(n)] for i in range(n)]
        port_st, per_exp, extra = simulate_portfolio(
            s0, mu, sg, w, corr, req.t_days, req.trading_days,
            req.num_paths, req.num_sample_paths, req.bins, req.risk_free, req.seed)
        s0_port = extra["s0_port"]
        t_years = extra["t_years"]
        mu_port = float(np.average(np.array(mu), weights=np.array(w) / (sum(w) or 1)))
        sig_port = float(np.std((port_st / s0_port - 1.0)) / np.sqrt(t_years)) if t_years > 0 else 0.2
        stats = _money_stats(port_st, s0_port, mu_port, max(sig_port, 0.05), t_years, req.risk_free, None)
        counts, edges = np.histogram(port_st, bins=req.bins)
        centers = ((edges[:-1] + edges[1:]) / 2).tolist()
        # 3D cloud from portfolio log-returns
        lr = np.log(port_st / s0_port)
        rng = np.random.default_rng(req.seed)
        endp = np.stack([rng.normal(size=len(lr)) * 3.0, (lr * 9.0).astype(float),
                         rng.normal(size=len(lr)) * 3.0], axis=1).astype(np.float32)
        dist = np.linalg.norm(endp, axis=1).astype(np.float32)
        colors, _, _, v95, v99, tail = _colorize(dist)
        legacy = {"mean": [0.0, round(float(np.mean(lr)), 4), 0.0],
                  "std": [1.0, round(float(np.std(lr)), 4), 1.0],
                  "percentile_5": [0.0, round(float(np.percentile(lr, 5)), 4), 0.0],
                  "percentile_95": [0.0, round(float(np.percentile(lr, 95)), 4), 0.0],
                  "var_95": round(float(v95) / 9.0, 4), "var_99": round(float(v99) / 9.0, 4)}
        import time
        cols = min(100, req.t_days)
        stride = max(1, req.t_days // cols)
        sp = np.array(extra["sample_port_paths"])[:, ::stride].tolist()
        taxis = [float(i * stride + 1) for i in range(len(sp[0]))] if sp else []
        meta = {"execution_time": 0.0, "total_points": req.num_paths, "paths": req.num_paths,
                "symbols": [a.symbol for a in req.assets], "weights": w,
                "s0_port": round(s0_port, 2), "t_days": req.t_days, "t_years": round(t_years, 4),
                "tail_particles_count": tail, "tail_percentage": round(tail / len(lr) * 100, 1),
                "run_id": uuid.uuid4().hex[:12],
                "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "seed": req.seed,
                "request_echo": {"kind": "portfolio",
                                 "symbols": [a.symbol for a in req.assets],
                                 "t_days": req.t_days, "corr_level": req.corr_level}}
        persist_simulation_run({**meta, "s0_port": s0_port, "mu_port": mu_port,
                                "sig_port": sig_port}, stats)
        return PortfolioSimulateResponse(
            positions=endp.ravel().tolist(), colors=colors.ravel().tolist(),
            sample_paths=sp, time_axis_days=taxis,
            histogram=HistogramPayload(centers=[round(float(c), 4) for c in centers],
                                       counts=[int(c) for c in counts]),
            stats=MoneyStatistics(**stats), statistics=SimulationStatistics(**legacy),
            metadata=meta, per_asset_expected=[round(float(x), 2) for x in per_exp],
            diversification_benefit=extra["benefit"])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Portfolio computation failure: {exc}") from exc


@router.get("/market/quote", response_model=QuoteResponse)
async def market_quote(symbol: str = Query(default="SPY")) -> QuoteResponse:
    try:
        return QuoteResponse(**fetch_quote(symbol))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Quote failure: {exc}") from exc


@router.get("/market/history", response_model=HistoryResponse)
async def market_history(symbol: str = Query(default="SPY"),
                         range: str = Query(default="1Y")) -> HistoryResponse:
    try:
        return HistoryResponse(**fetch_history(symbol, range))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"History failure: {exc}") from exc


@router.post("/market/calibrate", response_model=CalibrateResponse)
async def market_calibrate(req: CalibrateRequest) -> CalibrateResponse:
    try:
        return CalibrateResponse(**calibrate(req.symbol, req.lookback_days))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Calibration failure: {exc}") from exc


@router.get("/market/presets")
async def market_presets() -> List[dict]:
    from ..market import FALLBACK_PRESETS
    try:
        presets = FALLBACK_PRESETS
    except (ImportError, NameError):
        from app.market import FALLBACK_PRESETS as presets
    return [{"symbol": k, **v} for k, v in presets.items()]
