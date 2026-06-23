"""Unperturbed closed-loop n=7 swing-up + balance, take 2: DISCRETE TVLQR."""
import os, sys, time
from pathlib import Path
import numpy as np

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.lqr import StaticLQRPolicy, static_lqr
from cartpole_race.rollout import static_hold_rollout
from _dtvlqr import DiscreteTVLQR

n = 7
spec = CartPoleSpec(n_links=n, cart_mass_kg=1.0, link_masses_kg=[0.10] * n,
                    link_lengths_m=[0.50] * n,
                    damping_links_n_m_s_rad=[0.0] * n, force_bound_n=150.0)
m = NLinkCartPole(spec)
d = np.load("results/nom_n7_dense1ms.npz")
X = d["x"]; U = d["u"]; T = float(d["horizon"])
dt = spec.control_dt_s

t0 = time.time()
tv = DiscreteTVLQR(m, X, U, dt, progress=True)
print(f"[DTVLQR] built in {time.time()-t0:.0f}s  rho={tv.monodromy():.4g} "
      f"maxK={np.abs(tv.K).max():.3e}", flush=True)

K, P = static_lqr(m)
static_pol = StaticLQRPolicy(m, K)
static_pol.P = P
x0 = m.x_equilibrium("down")


def pol(x, t):
    return float(np.clip(tv.policy(x, t), -150, 150))


t1, x1, u1 = m.rollout_zoh(x0, pol, T, dt, spec.rk4_max_step_s)
xup = m.x_equilibrium("up")
dev = np.rad2deg(np.max(np.abs(x1[:, 1:1 + n] - X[: len(x1), 1:1 + n])))
xh = x1[-1]
hdev = np.rad2deg(np.max(np.abs(((xh[1:1 + n] - xup[1:1 + n] + np.pi)
                                 % (2 * np.pi)) - np.pi)))
hrate = np.max(np.abs(xh[2 + n:]))
print(f"[SWINGUP] maxdev={dev:.5f}deg handoff_dev={hdev:.5f}deg "
      f"handoff_rate={hrate:.6f}rad/s peakF={np.max(np.abs(u1)):.1f}N "
      f"cart=[{x1[:,0].min():.2f},{x1[:,0].max():.2f}]m", flush=True)

succ, info = static_hold_rollout(m, xh, static_pol, hold_time_s=5.0)
print(f"[HOLD] success={succ} "
      f"{ {k: (round(v,5) if isinstance(v,float) else v) for k,v in info.items() if not hasattr(v,'shape') and k!='final_state'} }")
if succ:
    print("\n*** n=7 SWING-UP + BALANCE: UNPERTURBED CLOSED-LOOP PASS ***")
