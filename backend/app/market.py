"""ORACLE market data: live quotes + history + calibration with fallbacks."""
from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import numpy as np

CACHE_TTL_S = 60
_cache: Dict[str, Tuple[float, dict]] = {}

FALLBACK_PRESETS = {
    "SPY":  {"price": 585.0, "mu": 0.09, "sigma": 0.17, "currency": "USD"},
    "QQQ":  {"price": 515.0, "mu": 0.11, "sigma": 0.22, "currency": "USD"},
    "AAPL": {"price": 232.0, "mu": 0.10, "sigma": 0.27, "currency": "USD"},
    "NVDA": {"price": 138.0, "mu": 0.18, "sigma": 0.48, "currency": "USD"},
    "BTC-USD": {"price": 97500.0, "mu": 0.25, "sigma": 0.62, "currency": "USD"},
    "ETH-USD": {"price": 3650.0, "mu": 0.20, "sigma": 0.70, "currency": "USD"},
    "GOLD": {"price": 2680.0, "mu": 0.06, "sigma": 0.15, "currency": "USD"},
    "EURUSD": {"price": 1.085, "mu": 0.005, "sigma": 0.075, "currency": "USD"},
}

BINANCE_MAP = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}


def _get_cached(key: str):
    item = _cache.get(key)
    if item and time.time() - item[0] < CACHE_TTL_S:
        return item[1]
    return None


def _set_cached(key: str, val: dict):
    _cache[key] = (time.time(), val)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_quote(symbol: str) -> dict:
    sym = symbol.upper().strip()
    ck = f"q:{sym}"
    hit = _get_cached(ck)
    if hit:
        return hit
    # 1) Binance for crypto (no key, CORS handled backend-side)
    if sym in BINANCE_MAP:
        try:
            import urllib.request, json
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={BINANCE_MAP[sym]}"
            with urllib.request.urlopen(url, timeout=8) as r:
                j = json.loads(r.read().decode())
            out = {"symbol": sym, "price": float(j["lastPrice"]), "currency": "USD",
                   "change_24h_pct": round(float(j.get("priceChangePercent", 0.0)), 2),
                   "source": "binance", "as_of": _now_iso()}
            _set_cached(ck, out)
            return out
        except Exception:
            pass
    # 2) Stooq free CSV (stocks/forex, no key)
    try:
        import urllib.request
        stooq = sym.replace("-", ".").lower()
        if sym in ("GOLD",): stooq = "xauusd"
        if sym in ("EURUSD",): stooq = "eurusd"
        url = f"https://stooq.com/q/l/?s={stooq}.us&f=sd2t2ohlcv&h&e=csv"
        with urllib.request.urlopen(url, timeout=8) as r:
            txt = r.read().decode().strip().splitlines()
        if len(txt) >= 2:
            parts = txt[1].split(",")
            price = float(parts[6]) if parts[6] not in ("N/D", "") else 0.0
            prev = float(parts[2]) if parts[2] not in ("N/D", "") else price
            if price > 0:
                chg = round((price - prev) / prev * 100, 2) if prev else 0.0
                out = {"symbol": sym, "price": price, "currency": "USD",
                       "change_24h_pct": chg, "source": "stooq", "as_of": _now_iso()}
                _set_cached(ck, out)
                return out
    except Exception:
        pass
    # 3) Fallback presets
    p = FALLBACK_PRESETS.get(sym, FALLBACK_PRESETS["SPY"])
    out = {"symbol": sym, "price": p["price"], "currency": p["currency"],
           "change_24h_pct": None, "source": "fallback", "as_of": _now_iso()}
    _set_cached(ck, out)
    return out


def fetch_history(symbol: str, rng: str = "1Y") -> dict:
    sym = symbol.upper().strip()
    ck = f"h:{sym}:{rng}"
    hit = _get_cached(ck)
    if hit:
        return hit
    days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 260, "2Y": 520}.get(rng.upper(), 260)
    # Binance klines for crypto
    if sym in BINANCE_MAP:
        try:
            import urllib.request, json
            url = f"https://api.binance.com/api/v3/klines?symbol={BINANCE_MAP[sym]}&interval=1d&limit={min(days, 500)}"
            with urllib.request.urlopen(url, timeout=10) as r:
                j = json.loads(r.read().decode())
            closes = [float(k[4]) for k in j]
            out = {"symbol": sym, "closes": closes, "dates": [], "source": "binance"}
            _set_cached(ck, out)
            return out
        except Exception:
            pass
    # Stooq daily CSV
    try:
        import urllib.request
        stooq = sym.replace("-", ".").lower()
        if sym == "GOLD": stooq = "xauusd"
        if sym == "EURUSD": stooq = "eurusd"
        url = f"https://stooq.com/q/d/l/?s={stooq}.us&d1={days}&e=csv" if False else \
              f"https://stooq.com/q/d/l/?s={stooq}.us&i=d"
        with urllib.request.urlopen(url, timeout=10) as r:
            lines = r.read().decode().strip().splitlines()
        closes = []
        for ln in lines[-days:]:
            p = ln.split(",")
            if len(p) >= 5:
                try:
                    closes.append(float(p[4]))
                except ValueError:
                    continue
        if len(closes) > 20:
            out = {"symbol": sym, "closes": closes, "dates": [], "source": "stooq"}
            _set_cached(ck, out)
            return out
    except Exception:
        pass
    # Synthetic fallback GBM history (seeded) — keeps UI functional offline
    p = FALLBACK_PRESETS.get(sym, FALLBACK_PRESETS["SPY"])
    rng_np = np.random.default_rng(abs(hash(sym)) % (2 ** 31))
    rets = rng_np.normal(p["mu"] / 252, p["sigma"] / np.sqrt(252), days)
    closes = (p["price"] * np.exp(np.cumsum(rets[::-1] - rets[-1]))).tolist()
    out = {"symbol": sym, "closes": [round(float(c), 2) for c in closes], "dates": [], "source": "fallback"}
    _set_cached(ck, out)
    return out


def calibrate(symbol: str, lookback_days: int = 252) -> dict:
    h = fetch_history(symbol, "2Y")
    s0 = fetch_quote(symbol)["price"]
    if h["source"] == "fallback" or len(h["closes"]) < 30:
        p = FALLBACK_PRESETS.get(symbol.upper(), FALLBACK_PRESETS["SPY"])
        return {"symbol": symbol.upper(), "s0": s0, "mu_ann": p["mu"],
                "sigma_ann": p["sigma"], "n_obs": len(h["closes"]), "source": h["source"]}
    arr = np.array(h["closes"][-lookback_days:], dtype=float)
    lr = np.diff(np.log(arr))
    sigma_ann = float(np.std(lr, ddof=1) * np.sqrt(252))
    mu_ann = float(np.mean(lr) * 252 + 0.5 * sigma_ann ** 2)
    mu_ann = max(-0.5, min(1.0, mu_ann))
    sigma_ann = max(0.05, min(1.5, sigma_ann))
    return {"symbol": symbol.upper(), "s0": s0, "mu_ann": round(mu_ann, 4),
            "sigma_ann": round(sigma_ann, 4), "n_obs": len(lr), "source": h["source"]}
