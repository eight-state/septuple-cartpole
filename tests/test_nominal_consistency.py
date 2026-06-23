"""Committed checks on the shipped n=7 dense nominal's rigor facts.

The n=7 dense nominal (``results/nom_n7_dense1ms.npz``) is the 1 ms
DENSIFICATION of a 4 ms collocation solve: within each 4 ms segment it IS the
simulator's own ZOH integration of the held node force (defect ~1e-13), and
at 4 ms node boundaries it carries the parent solve's transcription seam.
The committed claims tested here:

  1. intra-segment ticks reproduce exactly under the simulator's ZOH step
     (defect < 1e-10),
  2. node-boundary seams < 2e-4 state units (measured max 8.34e-5),
  3. the 4 ms parent solve's own RK4-4ms transcription defect < 1e-8
     (measured 4.1e-10),
  4. peak feedforward force < 30 N (measured 23.3 N) — 5x margin to the
     150 N bound.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec

REPO = Path(__file__).resolve().parent.parent
DENSE = REPO / "results" / "nom_n7_dense1ms.npz"
PARENT = REPO / "results" / "nom_n7_4ms.npz"
STRIDE = 4  # 1 ms ticks per 4 ms parent node


def _zoh_step(model, x, u, control_dt, rk4_max_step):
    n_sub = max(1, int(np.ceil(control_dt / rk4_max_step)))
    dt_sub = control_dt / n_sub
    xx = np.asarray(x, dtype=float).reshape(-1).copy()
    for _ in range(n_sub):
        xx = model.rk4_step(xx, u, dt_sub)
    return xx


def _model():
    spec = CartPoleSpec().with_n_links(7)
    return NLinkCartPole(spec), spec


def test_dense_nominal_intra_segment_exact() -> None:
    """Ticks NOT at 4 ms boundaries reproduce under the sim's ZOH step."""
    assert DENSE.exists(), f"missing nominal: {DENSE}"
    d = np.load(DENSE)
    X = np.asarray(d["x"], float)
    U = np.asarray(d["u"], float).reshape(-1)
    model, spec = _model()
    n = model.n
    assert (X.shape[1] // 2) - 1 == 7
    max_defect = 0.0
    for k in range(len(U)):
        if k % STRIDE == 0:
            continue  # first tick of a segment steps across the parent seam
        xn = _zoh_step(model, X[k], float(U[k]), spec.control_dt_s,
                       spec.rk4_max_step_s)
        max_defect = max(max_defect, float(np.max(np.abs(xn - X[k + 1]))))
    assert max_defect < 1e-10, (
        f"intra-segment densification not exact: {max_defect:.3e}")


def test_dense_nominal_seams_bounded() -> None:
    """4 ms node-boundary seams stay below the committed 2e-4 bound."""
    d = np.load(DENSE)
    X = np.asarray(d["x"], float)
    U = np.asarray(d["u"], float).reshape(-1)
    model, spec = _model()
    worst = 0.0
    for k in range(len(U)):
        if k % STRIDE != 0:
            continue  # seam-crossing ticks: first tick of each segment
        xn = _zoh_step(model, X[k], float(U[k]), spec.control_dt_s,
                       spec.rk4_max_step_s)
        worst = max(worst, float(np.max(np.abs(xn - X[k + 1]))))
    assert worst < 2e-4, f"node seam too large: {worst:.3e}"


def test_parent_solve_transcription_defect() -> None:
    """The 4 ms parent satisfies its own RK4-4ms transcription to < 1e-8."""
    d = np.load(PARENT)
    X = np.asarray(d["x"], float)
    U = np.asarray(d["u"], float).reshape(-1)
    model, _ = _model()
    h = float(d["horizon"]) / len(U)
    worst = 0.0
    for k in range(0, len(U), 7):  # sampled (every 7th node) for speed
        x = X[k]
        u = float(U[k])
        k1 = model.f(x, u); k2 = model.f(x + 0.5 * h * np.asarray(k1).reshape(-1), u)
        k3 = model.f(x + 0.5 * h * np.asarray(k2).reshape(-1), u)
        k4 = model.f(x + h * np.asarray(k3).reshape(-1), u)
        xn = x + (h / 6.0) * (np.asarray(k1).reshape(-1)
                              + 2 * np.asarray(k2).reshape(-1)
                              + 2 * np.asarray(k3).reshape(-1)
                              + np.asarray(k4).reshape(-1))
        worst = max(worst, float(np.max(np.abs(xn - X[k + 1]))))
    assert worst < 1e-8, f"parent transcription defect: {worst:.3e}"


def test_peak_feedforward_force_margin() -> None:
    d = np.load(DENSE)
    U = np.asarray(d["u"], float)
    assert float(np.abs(U).max()) < 30.0
