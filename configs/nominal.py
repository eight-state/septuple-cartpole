"""Single source of truth for the n=7 nominal trajectory and its grid.

This file pins which n=7 nominal the repo loads and the grid it was solved
on. Every entry point (``reproduce_n7.py``, the validation scripts) imports
``NOMINAL`` from here. Nothing else hard-codes the filename or the grid.

The shipped n=7 nominal is the 1 ms DENSIFICATION of a 4 ms collocation
solve: the 4 ms plan (``nom_n7_4ms.npz``, RK4 transcription defect 4.1e-10,
peak feedforward 23.3 N, terminal 0.0115 deg) integrated node-by-node through
the simulator's exact ZOH stepping (4x RK4 substeps of 0.25 ms per 1 ms
tick), so the dense reference is sim-consistent within each 4 ms segment
with node-boundary seams <= 8.4e-5. Closed-loop validation runs the REAL
saturated plant at the 1 ms control rate with exact-ZOH discrete TVLQR
(closed-loop monodromy rho = 0.197).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent
REPO = CONFIGS_DIR.parent
RESULTS = REPO / "results"


@dataclass(frozen=True)
class NominalSpec:
    """A nominal file plus the grid it was solved on."""

    file: str                 # filename inside results/
    grid_dt_s: float          # node spacing of the saved nominal
    n_nodes: int              # number of control intervals (len(u))
    horizon_s: float          # trajectory duration
    is_native_1ms: bool       # True when grid == the 1 ms control tick
    label: str                # short human label for the rigor status

    @property
    def path(self) -> Path:
        return RESULTS / self.file

    @property
    def grid_ms(self) -> float:
        return self.grid_dt_s * 1e3


# Shipped n=7 dense nominal: 1 ms grid (8000 control intervals, 8.0 s,
# 16-state), densified from the 4 ms collocation solve. Peak feedforward
# 23.3 N, terminal 0.0115 deg, max 4 ms-node seam 8.34e-5.
NOMINAL = NominalSpec(
    file="nom_n7_dense1ms.npz",
    grid_dt_s=0.001,
    n_nodes=8000,
    horizon_s=8.0,
    is_native_1ms=True,
    label="densified 4 ms collocation nominal (1 ms grid)",
)

# The coarse 4 ms parent solve (warm-start source; used by the per-IC replan
# in the composite gate).
NOMINAL_4MS = NominalSpec(
    file="nom_n7_4ms.npz",
    grid_dt_s=0.004,
    n_nodes=2000,
    horizon_s=8.0,
    is_native_1ms=False,
    label="4 ms collocation parent solve",
)
