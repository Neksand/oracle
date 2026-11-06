"""Project NEXUS - Vectorized Math Engine.

High-performance 3D Brownian Motion Monte Carlo simulation engine
with explicit Cyan, Purple, and Red spatial probability encoding.
"""

import time
from typing import Any, Dict, List, Tuple
import numpy as np


def generate_3d_stochastic_cloud(
    mu: float,
    sigma: float,
    num_paths: int,
    steps: int,
    t_max: float = 1.0,
) -> Tuple[List[float], List[float], Dict[str, Any], Dict[str, Any]]:
    """Generate 3D stochastic particle cloud using vectorized Brownian Motion.

    Formulation:
        dX = mu * dt + sigma * sqrt(dt) * N(0, 1) independently for X, Y, Z.
        Visual scale factor (k_scale = 4.5) ensures spatial movements and
        dispersion are clearly perceptible in the 3D viewport.

    Color Zones:
        - Cyan [0.0, 0.95, 1.0] (0% - 60%): High probability core
        - Purple [0.65, 0.12, 0.95] (60% - 85%): Standard deviation envelope
        - Red [1.0, 0.05, 0.12] (85% - 100%): VaR tail-risk outliers

    Args:
        mu: Drift parameter (directional trend rate).
        sigma: Volatility / noise parameter (dispersion intensity).
        num_paths: Number of Monte Carlo trajectories (100 to 50,000).
        steps: Discretization intervals per path (10 to 200).
        t_max: Time horizon (default 1.0).

    Returns:
        Tuple of:
            - positions: Interleaved coordinates [x1, y1, z1, x2, y2, z2, ...]
            - colors: Interleaved RGB values [r1, g1, b1, r2, g2, b2, ...]
            - statistics: Distribution metrics (mean, std, percentiles, VaR 95%, VaR 99%)
            - metadata: Execution telemetry, particle counts, and configuration
    """
    start_time = time.perf_counter()

    safe_paths = max(10, int(num_paths))
    safe_steps = max(2, int(steps))

    # Time step parameters
    dt = np.float32(t_max / safe_steps)
    sqrt_dt = np.sqrt(dt, dtype=np.float32)

    # Visual amplification factor for 3D coordinate space
    k_scale = 9.0

    # 3D drift: strong vertical elevation on Y (growth/decline), lateral dispersion on X and Z
    drift = np.array(
        [mu * 0.5 * dt * k_scale, mu * 2.0 * dt * k_scale, mu * 0.5 * dt * k_scale],
        dtype=np.float32,
    )
    diffusion = np.float32(sigma * sqrt_dt * k_scale)

    # Pure vectorized generator: shape (num_paths, steps, 3)
    rng = np.random.default_rng()
    increments = rng.normal(
        loc=drift,
        scale=diffusion,
        size=(safe_paths, safe_steps, 3),
    ).astype(np.float32)

    # Cumulative path integration
    paths = np.cumsum(increments, axis=1, dtype=np.float32)

    # Terminal state positions representing the Monte Carlo distribution at horizon
    endpoints = paths[:, -1, :]  # Shape: (num_paths, 3)

    # Euclidean distance of each particle from origin
    distances = np.linalg.norm(endpoints, axis=1).astype(np.float32)

    # Statistical percentiles for color zone encoding
    p60 = float(np.percentile(distances, 60))
    p85 = float(np.percentile(distances, 85))
    var_95 = float(np.percentile(distances, 95))
    var_99 = float(np.percentile(distances, 99))

    # Explicit 3-Color Mapping:
    # 1. Cyan [0.0, 0.95, 1.0] - Core 60%
    # 2. Purple [0.65, 0.12, 0.95] - Mid-range 25%
    # 3. Bright Red [1.0, 0.05, 0.12] - Tail Risk 15% (VaR extremes)
    colors = np.zeros((safe_paths, 3), dtype=np.float32)

    mask_cyan = distances <= p60
    mask_purple = (distances > p60) & (distances <= p85)
    mask_red = distances > p85

    colors[mask_cyan] = [0.0, 0.95, 1.0]      # Cyan
    colors[mask_purple] = [0.65, 0.12, 0.95]  # Purple
    colors[mask_red] = [1.0, 0.05, 0.12]     # Red

    # Metrics normalized to standard mathematical units
    mean_vec = (np.mean(endpoints, axis=0) / k_scale).astype(float).tolist()
    std_vec = (np.std(endpoints, axis=0) / k_scale).astype(float).tolist()
    pct_5_vec = (np.percentile(endpoints, 5, axis=0) / k_scale).astype(float).tolist()
    pct_95_vec = (np.percentile(endpoints, 95, axis=0) / k_scale).astype(float).tolist()

    statistics: Dict[str, Any] = {
        "mean": [round(v, 4) for v in mean_vec],
        "std": [round(v, 4) for v in std_vec],
        "percentile_5": [round(v, 4) for v in pct_5_vec],
        "percentile_95": [round(v, 4) for v in pct_95_vec],
        "var_95": round(float(var_95 / k_scale), 4),
        "var_99": round(float(var_99 / k_scale), 4),
    }

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    tail_count = int(np.sum(mask_red))

    metadata: Dict[str, Any] = {
        "execution_time": round(elapsed_ms, 2),
        "total_points": safe_paths,
        "paths": safe_paths,
        "steps": safe_steps,
        "mu": mu,
        "sigma": sigma,
        "tail_particles_count": tail_count,
        "tail_percentage": round(float(tail_count / safe_paths * 100), 1),
    }

    # Interleaved flat lists for WebGL Float32Array consumption
    positions = endpoints.ravel().tolist()
    flat_colors = colors.ravel().tolist()

    return positions, flat_colors, statistics, metadata
