"""Build the trackable n7 swing-up nominal that never existed (H6 gap).

Recipe proven on the catch (steer_catch3): solve collocation on a 4 ms grid
(RK4 step error ~2e-7 vs real sim), warm-started from nom_n7_gluck_cont
(real swing-up, peakF 27.5 N, terminal 0.25 deg, but defects 1.35e-1), then
densify to the exact 1 ms sim grid. Terminal ball tight enough to hand off
to the steering/static catch.

Output: results/nom_n7_4ms.npz (coarse) + nom_n7_dense1ms.npz (densified).
"""
import os, sys, time
from pathlib import Path
import numpy as np

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.collocation import solve_trajopt

n = 7
spec = CartPoleSpec(n_links=n, cart_mass_kg=1.0, link_masses_kg=[0.10] * n,
                    link_lengths_m=[0.50] * n,
                    damping_links_n_m_s_rad=[0.0] * n, force_bound_n=150.0)
m = NLinkCartPole(spec)

d = np.load("results/nom_n7_gluck_cont.npz")
X1 = d["x"]          # (8001, 16) on 1 ms grid
U1 = d["u"]          # (8000,)
T = float(d["horizon"])  # 8.0

H_NODE = 0.004
N = int(round(T / H_NODE))          # 2000
stride = int(round(H_NODE / 0.001))  # 4
Xw = X1[::stride]                    # (2001, 16) warm start states
# warm-start controls: average over each 4 ms block
Uw = U1[: N * stride].reshape(N, stride).mean(axis=1)
assert len(Xw) == N + 1, (len(Xw), N)

x0 = m.x_equilibrium("down")
print(f"[NOM-4MS] n={n} T={T}s N={N} warm-start from gluck_cont", flush=True)
t0 = time.time()
res = solve_trajopt(
    m, x0, horizon_s=T, n_nodes=N,
    terminal_tol_rad=2e-4,       # 0.011 deg ball; catch handles the rest
    force_bound=100.0,           # headroom vs 150 (gluck peak was 27.5)
    w_u=1e-4,
    x_init_guess=Xw, u_init_guess=Uw,
    zoh_consistent=False, max_iter=3000, print_level=5)
print(f"[NOM-4MS] solver={res.solver_status} defect={res.max_defect:.3e} "
      f"peakF={np.abs(res.u).max():.1f}N {time.time()-t0:.0f}s", flush=True)
ang = res.x[-1, 1:1 + n]
term = np.rad2deg(np.max(np.abs(((ang + np.pi) % (2 * np.pi)) - np.pi)))
print(f"[NOM-4MS] terminal angle dev {term:.4f} deg, "
      f"rates max {np.abs(res.x[-1, 2+n:]).max():.5f}", flush=True)
np.savez("results/nom_n7_4ms.npz", x=res.x, u=res.u, horizon=T,
         n=n, force=150.0, n_nodes=N)

if res.success:
    # densify to 1 ms grid with exact sim stepping
    control_dt = spec.control_dt_s
    n_sub = max(1, int(np.ceil(control_dt / spec.rk4_max_step_s)))
    dt_sub = control_dt / n_sub
    Xd = [res.x[0]]
    Ud = []
    seams = []
    for k in range(N):
        x = res.x[k].astype(float).copy()
        for _ in range(stride):
            for _ in range(n_sub):
                x = m.rk4_step(x, float(res.u[k]), dt_sub)
            Xd.append(x.copy())
            Ud.append(float(res.u[k]))
        seams.append(float(np.max(np.abs(x - res.x[k + 1]))))
    Xd = np.array(Xd); Ud = np.array(Ud)
    print(f"[NOM-4MS] densified: {len(Xd)} ticks, max seam "
          f"{max(seams):.2e}", flush=True)
    np.savez("results/nom_n7_dense1ms.npz", x=Xd, u=Ud, horizon=T,
             n=n, force=150.0, n_nodes=len(Ud))
    print("saved results/nom_n7_dense1ms.npz", flush=True)
else:
    print("NLP DID NOT CONVERGE - no densify", flush=True)
