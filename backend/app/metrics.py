"""
Prometheus metrics for ORACLE backend.
Exposes /metrics endpoint for scraping.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Callable, Any

from fastapi import Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# HTTP metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Business metrics
simulation_requests_total = Counter(
    "simulation_requests_total",
    "Total Monte-Carlo simulation requests",
    ["status", "asset_count"],
)

simulation_duration_seconds = Histogram(
    "simulation_duration_seconds",
    "Simulation execution time",
    ["asset_count"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

benchmark_runs_total = Counter(
    "benchmark_runs_total",
    "Total volatility benchmark runs",
    ["model", "status"],
)

benchmark_rmse = Gauge(
    "benchmark_rmse_latest",
    "Latest RMSE from volatility benchmark",
    ["model", "horizon"],
)

sql_injection_blocked_total = Counter(
    "sql_injection_blocked_total",
    "Total SQL injection attempts blocked",
    ["payload_type"],
)

active_users_total = Gauge(
    "active_users_total",
    "Number of active users (logged in within last hour)",
)

database_size_bytes = Gauge(
    "database_size_bytes",
    "Size of the database file in bytes",
)

database_connections_active = Gauge(
    "database_connections_active",
    "Number of active database connections",
)

rate_limit_exceeded_total = Counter(
    "rate_limit_exceeded_total",
    "Total rate limit exceeded events",
    ["client_ip"],
)

# System info
app_info = Info(
    "oracle_app_info",
    "ORACLE application information",
)

# ---------------------------------------------------------------------------
# Middleware for automatic HTTP metrics
# ---------------------------------------------------------------------------


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically collect HTTP metrics."""

    def __init__(self, app, exclude_paths: list[str] | None = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/metrics", "/health", "/favicon.ico"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        method = request.method
        # Normalize endpoint path (replace path params with placeholders)
        endpoint = self._normalize_path(request.url.path)

        start_time = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            status_code = 500
            raise
        finally:
            duration = time.perf_counter() - start_time
            http_requests_total.labels(
                method=method, endpoint=endpoint, status=status_code
            ).inc()
            http_request_duration_seconds.labels(
                method=method, endpoint=endpoint
            ).observe(duration)

    def _normalize_path(self, path: str) -> str:
        """Normalize path parameters to placeholders."""
        # Common patterns: /api/db/scenarios/123 -> /api/db/scenarios/{id}
        import re
        path = re.sub(r"/\d+", "/{id}", path)
        path = re.sub(r"/[a-f0-9-]{36}", "/{uuid}", path)
        return path


# ---------------------------------------------------------------------------
# Helper functions for business metrics
# ---------------------------------------------------------------------------


def record_simulation(status: str, asset_count: int, duration: float) -> None:
    """Record a simulation request."""
    simulation_requests_total.labels(status=status, asset_count=str(asset_count)).inc()
    simulation_duration_seconds.labels(asset_count=str(asset_count)).observe(duration)


def record_benchmark(model: str, status: str, rmse: float | None = None, horizon: str = "5d") -> None:
    """Record a benchmark run."""
    benchmark_runs_total.labels(model=model, status=status).inc()
    if rmse is not None:
        benchmark_rmse.labels(model=model, horizon=horizon).set(rmse)


def record_sql_injection(payload_type: str) -> None:
    """Record a blocked SQL injection attempt."""
    sql_injection_blocked_total.labels(payload_type=payload_type).inc()


def set_active_users(count: int) -> None:
    """Set active users count."""
    active_users_total.set(count)


def set_database_size(size_bytes: int) -> None:
    """Set database size."""
    database_size_bytes.set(size_bytes)


def set_database_connections(count: int) -> None:
    """Set active database connections."""
    database_connections_active.set(count)


def record_rate_limit(client_ip: str) -> None:
    """Record rate limit exceeded."""
    rate_limit_exceeded_total.labels(client_ip=client_ip).inc()


def set_app_info(version: str, git_sha: str, build_date: str) -> None:
    """Set application info."""
    app_info.info({
        "version": version,
        "git_sha": git_sha,
        "build_date": build_date,
    })


# ---------------------------------------------------------------------------
# Decorator for timing functions
# ---------------------------------------------------------------------------


def timed(metric: Histogram, **labels):
    """Decorator to time a function with a histogram."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                metric.labels(**labels).observe(time.perf_counter() - start)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                metric.labels(**labels).observe(time.perf_counter() - start)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ---------------------------------------------------------------------------
# FastAPI endpoint for metrics
# ---------------------------------------------------------------------------


async def metrics_endpoint() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)