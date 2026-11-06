"""Project NEXUS - Simulation API Router.

Provides POST endpoint /simulate to trigger vectorized Monte Carlo simulations
and return distribution geometries and tail-risk statistics.
"""

from fastapi import APIRouter, HTTPException, status
try:
    from ..models import SimulationRequest, SimulationResponse, SimulationStatistics
    from ..math_engine import generate_3d_stochastic_cloud
except (ImportError, ValueError):
    try:
        from app.models import SimulationRequest, SimulationResponse, SimulationStatistics
        from app.math_engine import generate_3d_stochastic_cloud
    except ImportError:
        from models import SimulationRequest, SimulationResponse, SimulationStatistics
        from math_engine import generate_3d_stochastic_cloud

router = APIRouter(tags=["simulation"])


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute 3D Stochastic Particle Cloud",
    description="Vectorized calculation of 3D Geometric Brownian Motion with distance-based color mapping and VaR metrics.",
)
async def simulate(request: SimulationRequest) -> SimulationResponse:
    """Execute stochastic 3D Brownian motion simulation and return particle vertices with tail-risk statistics."""
    try:
        positions, colors, stats, metadata = generate_3d_stochastic_cloud(
            mu=request.mu,
            sigma=request.sigma,
            num_paths=request.num_paths,
            steps=request.steps,
        )

        return SimulationResponse(
            positions=positions,
            colors=colors,
            statistics=SimulationStatistics(**stats),
            metadata=metadata,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stochastic computation failure: {str(exc)}",
        ) from exc
