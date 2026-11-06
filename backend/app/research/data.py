"""ORACLE - Research data loading: internet historical prices with layered fallbacks.

Sources (in order):
    1. yfinance        (Yahoo Finance API over the internet)
    2. Stooq CSV       via app.market.fetch_history (already cached in-memory)
    3. local DB        market_data table filled by seed / previous fetches
    4. synthetic GBM   seeded fallback so the pipeline still runs offline
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.orm_models import MarketData


@dataclass
class PriceSeries:
    symbol: str
    closes: np.ndarray
    source: str          # yahoo|stooq|database|synthetic
    dates: list[str]

    @property
    def log_returns(self) -> np.ndarray:
        return np.diff(np.log(self.closes))


def _try_yfinance(symbol: str, days: int) -> np.ndarray | None:
    try:
        import yfinance as yf

        ticker_map: dict[str, str] = {
            "GOLD": "GLD", "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD",
            "EURUSD": "EURUSD=X", "SPY": "SPY", "QQQ": "QQQ",
            "AAPL": "AAPL", "NVDA": "NVDA",
        }
        tk = ticker_map.get(symbol.upper(), symbol)
        period_years = max(1, int(np.ceil(days / 250)))
        df = yf.download(tk, period=f"{min(period_years, 5)}y", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 30:
            return None
        closes = df["Close"].dropna()
        if hasattr(closes, "to_frame"):
            vals = closes.to_numpy().ravel()
        else:
            vals = np.asarray(closes).ravel()
        return np.asarray(vals[-days:], dtype=float)
    except Exception:
        return None


def _from_db(symbol: str, days: int) -> tuple[np.ndarray, list[str]] | None:
    try:
        with SessionLocal() as session:
            rows = session.scalars(
                select(MarketData)
                .where(MarketData.symbol == symbol.upper())
                .order_by(MarketData.trade_date.desc())
                .limit(days)
            ).all()
        if len(rows) < 30:
            return None
        rows.reverse()
        return (
            np.array([r.close for r in rows], dtype=float),
            [r.trade_date.isoformat() for r in rows],
        )
    except Exception:
        return None


def load_history(symbol: str, days: int = 750) -> PriceSeries:
    """Layered accessor; each layer logged in the response for the report."""
    sym = symbol.upper().strip()

    y = _try_yfinance(sym, days)
    if y is not None and len(y) >= 60:
        return PriceSeries(sym, y, "yahoo (internet)", dates=[])

    from app.market import fetch_history

    h = fetch_history(sym, "2Y")
    if h and h.get("source") in ("stooq", "binance") and len(h["closes"]) >= 60:
        return PriceSeries(sym, np.array(h["closes"], dtype=float),
                           f"{h['source']} (internet)", dates=h.get("dates") or [])

    db_hit = _from_db(sym, days)
    if db_hit is not None:
        return PriceSeries(sym, db_hit[0], "local database (market_data)", dates=db_hit[1])

    # seeded synthetic GBM so research still works offline
    from app.market import FALLBACK_PRESETS

    p = FALLBACK_PRESETS.get(sym, FALLBACK_PRESETS["SPY"])
    rng = np.random.default_rng(7)
    rets = rng.normal(p["mu"] / 252.0, p["sigma"] / np.sqrt(252.0), days)
    closes = p["price"] * np.exp(np.cumsum(rets))
    return PriceSeries(sym, closes, "synthetic GBM (offline fallback)",
                       dates=[(dt.date.today() - dt.timedelta(days=i)).isoformat() for i in range(days)][::-1])
