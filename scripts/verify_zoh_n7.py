"""Verify the 1-step ZOH defect of a polished nominal at the 1ms control rate.

Mirrors the n=5 verification: load the nominal (states + forces), and for EACH
control tick integrate the held nominal force through the simulator's exact ZOH
step (n_sub = ceil(control_dt / rk4_max_step) = 4 RK4 substeps of 0.25ms each),
then compare the integrated next-state to the stored nominal next-state.

A 1ms-zoh-consistent nominal has max 1-step defect ~1e-13 (machine precision).

Usage: verify_zoh_n7.py <nom_path> <n>
"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.dynamics import NLinkCartPole


def main():
    nom_path = sys.argv[1]
    n = int(sys.argv[2])
    d = np.load(nom_path, allow_pickle=True)
    X = d["states"] if "states" in d else d["x"]
    u = d["forces"] if "forces" in d else d["u"]
    horizon = float(d["T"]) if "T" in d else float(d["horizon"])

    m = NLinkCartPole(CartPoleSpec().with_n_links(n))
    control_dt = m.spec.control_dt_s
    rk4_max = m.spec.rk4_max_step_s
    n_sub = max(1, int(np.ceil(control_dt / rk4_max)))
    dt_sub = control_dt / n_sub

    Nu = len(u)
    Nx = X.shape[0]
    print(f"nom={nom_path} n={n} horizon={horizon}s nodes_x={Nx} nodes_u={Nu} "
          f"control_dt={control_dt*1e3:.3f}ms n_sub={n_sub} dt_sub={dt_sub*1e3:.4f}ms",
          flush=True)
    if Nx != Nu + 1:
        print(f"WARNING: expected Nx=Nu+1, got Nx={Nx} Nu={Nu}")
    dt_grid = horizon / (Nx - 1)
    print(f"node spacing = {dt_grid*1e3:.4f}ms (this must be ~1.000ms for 1ms rigor)")

    defects = np.empty(Nu)
    for k in range(Nu):
        x = X[k].astype(float).copy()
        uk = float(u[k])
        for _ in range(n_sub):
            x = m.rk4_step(x, uk, dt_sub)
        defects[k] = np.max(np.abs(x - X[k + 1]))
    print(f"1-step ZOH defect: MAX={defects.max():.3e} MEAN={defects.mean():.3e} "
          f"ticks>1e-6={int((defects>1e-6).sum())} ticks>1e-9={int((defects>1e-9).sum())}",
          flush=True)
    # terminal angle from upright
    ang = X[-1, 1:1 + n]
    term = float(np.rad2deg(np.abs(((ang + np.pi) % (2 * np.pi)) - np.pi)).max())
    print(f"terminal max angle-from-upright={term:.4f}deg peakF={np.max(np.abs(u)):.2f}N")


if __name__ == "__main__":
    main()
