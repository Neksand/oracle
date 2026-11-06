"""ORACLE - Seed script: 8-10+ realistic rows per table (Assignment 3, Task 3).

Run:  python -m app.db.seed
Result: backend/oracle.db populated with demo users, scenarios, assets,
simulation runs, market history and audit events.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
from sqlalchemy import select

from app.db.database import DB_PATH, SessionLocal, init_db
from app.db.orm_models import (
    ApiKey,
    AuditLog,
    MarketData,
    Scenario,
    ScenarioAsset,
    SimulationRun,
    User,
)


def hash_password(raw: str) -> str:
    import bcrypt

    return bcrypt.hashpw(raw.encode(), bcrypt.gensalt(rounds=10)).decode()


def _gbm_history(s0: float, mu: float, sigma: float, days: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu / 252.0, sigma / math.sqrt(252.0), days)
    path = s0 * np.exp(np.cumsum(rets[::-1] - rets[-1]))  # ends exactly at s0
    return [float(np.round(x, 4)) for x in path]


SYMBOL_PARAMS = {
    "SPY": {"s0": 585.0, "mu": 0.09, "sigma": 0.17, "seed": 11},
    "QQQ": {"s0": 515.0, "mu": 0.11, "sigma": 0.22, "seed": 12},
    "AAPL": {"s0": 232.0, "mu": 0.10, "sigma": 0.27, "seed": 13},
    "NVDA": {"s0": 138.0, "mu": 0.18, "sigma": 0.48, "seed": 14},
    "BTC-USD": {"s0": 97500.0, "mu": 0.25, "sigma": 0.62, "seed": 15},
    "GOLD": {"s0": 2680.0, "mu": 0.06, "sigma": 0.15, "seed": 16},
}


def seed():
    init_db()
    session = SessionLocal()
    try:
        # ------------------------------------------------------------------ users
        admin = User(
            username="admin",
            email="admin@oracle.app",
            password_hash=hash_password("Admin#Oracle26"),
            role="admin",
        )
        alice = User(
            username="alice",
            email="alice@example.com",
            password_hash=hash_password("Alice#Oracle26"),
            role="user",
        )
        bob = User(username="bob", email="bob@example.com",
                   password_hash=hash_password("Bob#Oracle26"), role="user")
        # five more demo accounts so every table satisfies the assignment's
        # "at least 8-10 realistic rows" requirement (users: 8 in total)
        carol = User(username="carol", email="carol@example.com",
                     password_hash=hash_password("Carol#Oracle26"), role="user")
        dave = User(username="dave", email="dave@example.com",
                    password_hash=hash_password("Dave#Oracle26"), role="user")
        erin = User(username="erin", email="erin@example.com",
                    password_hash=hash_password("Erin#Oracle26"), role="user")
        frank = User(username="frank", email="frank@example.com",
                     password_hash=hash_password("Frank#Oracle26"), role="user")
        grace = User(username="grace", email="grace@example.com",
                     password_hash=hash_password("Grace#Oracle26"), role="user")
        session.add_all([admin, alice, bob, carol, dave, erin, frank, grace])
        session.flush()

        # Demo API key for the teacher's remote-API check (plaintext is documented
        # in deliverables/ict_access/CREDENTIALS.md; only the hash lives here).
        import hashlib

        demo_key = "oracle_demo_2026_access"
        if not session.query(ApiKey).filter(ApiKey.prefix == demo_key[:12]).first():
            session.add(ApiKey(
                user_id=admin.id,
                name="teacher-demo",
                prefix=demo_key[:12],
                key_hash=hashlib.sha256(demo_key.encode()).hexdigest(),
            ))
            session.flush()

        # -------------------------------------------------------------- scenarios
        def mk_scenario(user: User, name: str, desc: str, horizon: int,
                        assets: list[tuple[str, float]]) -> Scenario:
            sc = Scenario(
                user_id=user.id, name=name, description=desc, horizon_days=horizon,
                trading_days=252, num_paths=8000, risk_free=0.04,
            )
            sc.assets = [
                ScenarioAsset(
                    symbol=sym,
                    s0=SYMBOL_PARAMS[sym]["s0"],
                    mu_ann=SYMBOL_PARAMS[sym]["mu"],
                    sigma_ann=SYMBOL_PARAMS[sym]["sigma"],
                    weight=w,
                )
                for sym, w in assets
            ]
            return sc

        scen_home_ret = mk_scenario(
            alice, "Retirement Core", "Low-vol pension-style allocation",
            252, [("SPY", 0.5), ("GOLD", 0.3), ("BTC-USD", 0.2)],
        )
        scen_tech = mk_scenario(
            bob, "Aggressive Tech", "High-beta equity bet with crypto kicker",
            90, [("NVDA", 0.4), ("QQQ", 0.3), ("BTC-USD", 0.3)],
        )
        scen_even = mk_scenario(
            bob, "Balanced Duo", "Equities vs gold barbell",
            180, [("SPY", 0.6), ("GOLD", 0.4)],
        )
        scen_single = mk_scenario(
            alice, "BTC Cycle Watch", "Crypto-only scenario (365d calendar)",
            365, [("BTC-USD", 1.0)],
        )
        session.add_all([scen_home_ret, scen_tech, scen_even, scen_single])
        session.flush()

        # ------------------------------------------------ additional demo portfolios
        scen_income = mk_scenario(
            carol, "Dividend Ladder", "Income sleeve: broad equity + gold ballast",
            365, [("SPY", 0.6), ("GOLD", 0.4)],
        )
        session.add(scen_income)
        session.flush()  # id 5 -> keeps the Q3 demonstration row at id 6

        scen_smoke = mk_scenario(
            alice, "Smoke Test Portfolio", "Q3 insert check",
            120, [("SPY", 0.7), ("GOLD", 0.3)],
        )
        session.add(scen_smoke)
        session.flush()  # id 6, referenced by the report's Q3 section

        scen_crypto = mk_scenario(
            dave, "Crypto Basket", "High-volatility crypto core with an equity hedge",
            90, [("BTC-USD", 0.7), ("SPY", 0.3)],
        )
        scen_momentum = mk_scenario(
            erin, "Tech Momentum", "Semis + megacap growth, 6-month horizon",
            180, [("QQQ", 0.5), ("NVDA", 0.5)],
        )
        scen_defensive = mk_scenario(
            frank, "Defensive Mix", "Gold-heavy hedge against an equity drawdown",
            252, [("GOLD", 0.5), ("SPY", 0.5)],
        )
        scen_growth = mk_scenario(
            grace, "Balanced Growth", "Two-ETF core portfolio, one-year horizon",
            252, [("SPY", 0.5), ("QQQ", 0.5)],
        )
        session.add_all([scen_crypto, scen_momentum, scen_defensive, scen_growth])
        session.flush()

        # --------------------------------------------------------- simulation_runs
        rng = np.random.default_rng(42)
        run_defs = [
            (scen_home_ret, "asset", ["SPY"], 585.0, 0.09, 0.17, 252),
            (scen_tech, "portfolio", ["NVDA", "QQQ", "BTC-USD"], 214.9, 0.18, 0.48, 90),
            (scen_even, "portfolio", ["SPY", "GOLD"], 100.0, 0.075, 0.16, 180),
            (scen_single, "asset", ["BTC-USD"], 97500.0, 0.25, 0.62, 365),
        ]
        for sc, kind, syms, s0, mu, sg, td in run_defs:
            for j in range(3):  # 12 runs total
                st = s0 * math.exp((mu - 0.5 * sg**2) * td / 252.0 + sg * math.sqrt(td / 252.0) * rng.normal())
                var95 = abs(float(rng.normal(0.11, 0.02)) * s0)
                session.add(
                    SimulationRun(
                        scenario_id=sc.id, run_uid=f"run{rng.integers(1e11, 1e12):011d}",
                        kind=kind, symbols_json=syms, s0=round(s0, 2), mu_ann=mu, sigma_ann=sg,
                        t_days=td, num_paths=8000, seed=int(rng.integers(1, 99999)),
                        expected_st=round(float(st), 2),
                        var95_loss=round(var95, 2),
                        cvar95_loss=round(var95 * 1.25, 2),
                        var99_loss=round(var95 * 1.45, 2),
                        prob_decline=round(float(rng.uniform(0.2, 0.45)), 4),
                        execution_ms=round(float(rng.uniform(60, 220)), 1),
                    )
                )

        extra_defs = [
            (scen_income, ["SPY", "GOLD"], 1653.0, 0.078, 0.14, 365),
            (scen_crypto, ["BTC-USD", "SPY"], 68425.5, 0.202, 0.45, 90),
            (scen_momentum, ["QQQ", "NVDA"], 326.5, 0.145, 0.30, 180),
            (scen_defensive, ["GOLD", "SPY"], 1632.5, 0.075, 0.14, 252),
            (scen_growth, ["SPY", "QQQ"], 550.0, 0.10, 0.18, 252),
        ]
        for sc, syms, s0, mu, sg, td in extra_defs:
            for _ in range(2):  # 10 runs for the additional portfolios
                st = s0 * math.exp((mu - 0.5 * sg**2) * td / 252.0 + sg * math.sqrt(td / 252.0) * rng.normal())
                var95 = abs(float(rng.normal(0.11, 0.02)) * s0)
                session.add(
                    SimulationRun(
                        scenario_id=sc.id, run_uid=f"run{rng.integers(1e11, 1e12):011d}",
                        kind="portfolio", symbols_json=syms, s0=round(s0, 2),
                        mu_ann=mu, sigma_ann=sg, t_days=td, num_paths=8000,
                        seed=int(rng.integers(1, 99999)),
                        expected_st=round(float(st), 2),
                        var95_loss=round(var95, 2),
                        cvar95_loss=round(var95 * 1.25, 2),
                        var99_loss=round(var95 * 1.45, 2),
                        prob_decline=round(float(rng.uniform(0.2, 0.45)), 4),
                        execution_ms=round(float(rng.uniform(60, 220)), 1),
                    )
                )

        # ------------------------------------------------------------ market_data
        today = dt.date.today()
        trade_dates = []
        d = today
        while len(trade_dates) < 120:
            if d.weekday() < 5:  # trading days only
                trade_dates.append(d)
            d -= dt.timedelta(days=1)
        trade_dates.reverse()
        for sym, p in SYMBOL_PARAMS.items():
            hist = _gbm_history(p["s0"], p["mu"], p["sigma"], len(trade_dates), p["seed"])
            for date, close in zip(trade_dates, hist):
                session.add(MarketData(
                    symbol=sym, trade_date=date, close=close, source="seed",
                ))

        # -------------------------------------------------------------- audit_log
        events = [
            ("login_success", "alice", "OAuth2 password flow, JWT issued", False),
            ("login_failed", "anonymous", "Wrong password for user 'bob' from 127.0.0.1", True),
            ("sql_injection_probe", "anonymous", "Input \"' OR 1=1 --\" caught by parameterized query", True),
            ("login_success", "bob", "OAuth2 password flow, JWT issued", False),
            ("sql_export", "alice", "Exported oracle_dump.sql via /api/db/export", False),
            ("rate_limited", "anonymous", "50 /finance/simulate requests in 10s -> 429", True),
            ("simulation_run", "bob", "POST /api/finance/simulate NVDA 90d seed=1337", False),
            ("unauthorized_access", "anonymous", "GET /api/scenarios without Bearer token -> 401", True),
        ]
        for ev, actor, detail, blocked in events:
            session.add(AuditLog(event_type=ev, actor=actor, detail=detail, blocked=blocked))

        session.commit()

        # quick sanity counts (only rows we explicitly created)
        q = lambda m: len(session.scalars(select(m)).all())  # noqa: E731
        print("Seeded:",
              f"users={q(User)}", f"scenarios={q(Scenario)}",
              f"assets={q(ScenarioAsset)}", f"runs={q(SimulationRun)}",
              f"market_data={q(MarketData)}", f"audit={q(AuditLog)}",
              f"-> {DB_PATH}")
    finally:
        session.close()


if __name__ == "__main__":
    seed()
