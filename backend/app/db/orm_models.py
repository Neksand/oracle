"""ORACLE - ORM entities (Assignment 3, Task 2: Database Structure).

Six entities:

    users            - account owner (bcrypt hash, role), due to cybersecurity task
    scenarios        - a named parameterconfiguration (what the user saves)
    scenario_assets  - assets inside a scenario (1:N composite)
    simulation_runs  - every executed MC run + headline risk metrics (audit trail)
    market_data      - daily historical prices (one time-series row per symbol/date)
    audit_log        - security-relevant events (login, export, sql-demo attempts)

Relationships:
    users 1---N scenarios 1---N scenario_assets
    users 1---N scenarios 1---N simulation_runs
    scenarios N---1 ... market_data linked symbolically (by symbol column)
"""
from __future__ import annotations

import datetime as dt
from typing import List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)  # bcrypt, never plain
    role: Mapped[str] = mapped_column(String(16), default="user", nullable=False)  # user|admin
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    scenarios: Mapped[List["Scenario"]] = relationship(back_populates="owner")


class Scenario(Base):
    """A saved portfolio/scenario configuration (replaces localStorage)."""
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)          # t_days
    trading_days: Mapped[int] = mapped_column(Integer, default=252, nullable=False)
    num_paths: Mapped[int] = mapped_column(Integer, default=8000, nullable=False)
    risk_free: Mapped[float] = mapped_column(Float, default=0.04, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False, index=True
    )

    owner: Mapped[User] = relationship(back_populates="scenarios")
    assets: Mapped[List["ScenarioAsset"]] = relationship(
        back_populates="scenario", cascade="all, delete-orphan", order_by="ScenarioAsset.id"
    )
    runs: Mapped[List["SimulationRun"]] = relationship(back_populates="scenario")


class ScenarioAsset(Base):
    """Asset line inside a scenario (join table scenario <-> asset params)."""
    __tablename__ = "scenario_assets"
    __table_args__ = (UniqueConstraint("scenario_id", "symbol", name="uq_scenario_symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scenario_id: Mapped[int] = mapped_column(
        ForeignKey("scenarios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    s0: Mapped[float] = mapped_column(Float, nullable=False)          # current price
    mu_ann: Mapped[float] = mapped_column(Float, nullable=False)      # expected (drift) annual
    sigma_ann: Mapped[float] = mapped_column(Float, nullable=False)   # volatility
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    scenario: Mapped[Scenario] = relationship(back_populates="assets")


class SimulationRun(Base):
    """Result metadata of one Monte-Carlo execution (INSERT after /simulate)."""
    __tablename__ = "simulation_runs"
    __table_args__ = (Index("ix_runs_scenario_time", "scenario_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scenario_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("scenarios.id", ondelete="SET NULL"), nullable=True, index=True
    )
    run_uid: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    kind: Mapped[str] = mapped_column(String(16), default="asset")  # asset|portfolio
    symbols_json: Mapped[dict] = mapped_column(JSON, default=list)
    s0: Mapped[float] = mapped_column(Float, nullable=False)
    mu_ann: Mapped[float] = mapped_column(Float, nullable=False)
    sigma_ann: Mapped[float] = mapped_column(Float, nullable=False)
    t_days: Mapped[int] = mapped_column(Integer, nullable=False)
    num_paths: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    expected_st: Mapped[float] = mapped_column(Float, nullable=False)
    var95_loss: Mapped[float] = mapped_column(Float, nullable=False)
    cvar95_loss: Mapped[float] = mapped_column(Float, nullable=False)
    var99_loss: Mapped[float] = mapped_column(Float, nullable=False)
    prob_decline: Mapped[float] = mapped_column(Float, nullable=False)
    execution_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False, index=True
    )
    # optional FK to audit entry
    audit_event: Mapped[Optional[str]] = mapped_column(String(32), default="simulation_run")

    scenario: Mapped[Optional[Scenario]] = relationship(back_populates="runs")


class MarketData(Base):
    """Daily close-price history used for calibration (time-series slice)."""
    __tablename__ = "market_data"
    __table_args__ = (UniqueConstraint("symbol", "trade_date", name="uq_symbol_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    trade_date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(16), default="stooq", nullable=False)  # stooq|binance|seed
    fetched_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False
    )


class AuditLog(Base):
    """Security event trail (login attempts, exports, injection demo probes)."""
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(64), default="anonymous", nullable=False)
    detail: Mapped[str] = mapped_column(Text, default="", nullable=False)
    blocked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False, index=True
    )


class ApiKey(Base):
    """Hashed API key for programmatic access (never stored in plaintext)."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), default="default", nullable=False)
    prefix: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False
    )
    last_used_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    owner: Mapped[User] = relationship()


class UsageEvent(Base):
    """Smart-feature data: what users actually do in the product.

    Captured by the analytics layer (scenario runs, API calls, sign-ins).
    Feeds the "product usage" section of the report and the CJM evidence.
    """

    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)  # run|api_call|login|register
    symbol: Mapped[Optional[str]] = mapped_column(String(24), nullable=True, index=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False, index=True
    )

    user: Mapped[Optional[User]] = relationship()
