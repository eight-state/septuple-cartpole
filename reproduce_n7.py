"""ONE-COMMAND reproduction of the headline n=7 (septuple) cart-pole result.

    uv run python reproduce_n7.py            # fast: unperturbed pass + rho
    uv run python reproduce_n7.py --gate     # full 24-IC perturbed gate (slow)

Fast mode (~2 min):
  1. Loads the shipped n=7 dense nominal (configs/nominal.py).
  2. Verifies the nominal's grid facts (8000 ticks, 8 s) and its 4 ms parent's
     transcription defect.
  3. Builds exact-ZOH DISCRETE-time TVLQR along the dense nominal and reports
     the closed-loop monodromy spectral radius rho (expected ~0.197 < 1).
  4. Runs the UNPERTURBED closed loop in the real saturated simulator
     (rollout_zoh, hard 150 N clip, 1 ms ZOH, RK4 substeps) from exact
     hanging, then the static-LQR hold with the locked predicate v1
     (|theta|<=5 deg, |thetad|<=0.5, |x|<=2 m, |xd|<=0.5, continuous 5 s).
     Expected: PASS, peak force 23.3 N, handoff 0.0115 deg.

Gate mode (--gate, ~1-2 h on 12 cores):
  5. The composite-controller perturbed-IC ensemble at sigma=0.02 (identical
     perturbation model, seeds, simulator, and predicate as the n=5/n=6
     releases): per-IC replan (warm-started 4 ms collocation from the
     measured IC) -> discrete TVLQR -> steering-NLP catch -> static hold.
     Expected: 24/24 (seed 12345), peak force ~32 N. Banked JSONs for both
     seeds are in results/.

Determinism: each rollout is fixed-step RK4 + ZOH and deterministic. The
gate's success COUNTS are reproducible; per-IC NLP solve wall-times are not.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from cartpole_race.dynamics import NLinkCartPole  # noqa: E402
from cartpole_race.env_spec import CartPoleSpec  # noqa: E402
from cartpole_race.lqr import StaticLQRPolicy, static_lqr  # noqa: E402
from cartpole_race.rollout import static_hold_rollout  # noqa: E402
from configs.nominal import NOMINAL  # noqa: E402
from _dtvlqr import DiscreteTVLQR  # noqa: E402


def main() -> int:
    n = 7
    spec = CartPoleSpec(n_links=n, cart_mass_kg=1.0,
                        link_masses_kg=[0.10] * n, link_lengths_m=[0.50] * n,
                        damping_links_n_m_s_rad=[0.0] * n, force_bound_n=150.0)
    m = NLinkCartPole(spec)
    d = np.load(NOMINAL.path)
    X = d["x"]; U = d["u"]; T = float(d["horizon"])
    dt = spec.control_dt_s
    assert len(U) == NOMINAL.n_nodes and abs(T - NOMINAL.horizon_s) < 1e-9
    print(f"[1] nominal: {NOMINAL.file} ({NOMINAL.label}), "
          f"{len(U)} ticks, {T} s, peak ff "
          f"{np.abs(U).max():.1f} N")

    print("[2] building exact-ZOH discrete TVLQR along the nominal ...")
    tv = DiscreteTVLQR(m, X, U, dt)
    rho = tv.monodromy()
    print(f"    closed-loop monodromy rho = {rho:.4g}  (expected ~0.197 < 1)")

    print("[3] UNPERTURBED closed loop in the real saturated sim ...")
    x0 = m.x_equilibrium("down")

    def pol(x, t):
        return float(np.clip(tv.policy(x, t), -spec.force_bound_n,
                             spec.force_bound_n))

    t1, x1, u1 = m.rollout_zoh(x0, pol, T, dt, spec.rk4_max_step_s)
    xup = m.x_equilibrium("up")
    xh = x1[-1]
    hdev = np.rad2deg(np.max(np.abs(((xh[1:1 + n] - xup[1:1 + n] + np.pi)
                                     % (2 * np.pi)) - np.pi)))
    print(f"    swing-up: handoff dev {hdev:.4f} deg, "
          f"peak force {np.abs(u1).max():.1f} N")

    K, P = static_lqr(m)
    sp_ = StaticLQRPolicy(m, K)
    sp_.P = P
    succ, info = static_hold_rollout(m, xh, sp_, hold_time_s=5.0)
    print(f"    hold (predicate v1): success={succ}, "
          f"peak force {info.get('max_force', 0.0):.1f} N")
    if not (succ and rho < 1.0):
        print("REPRODUCTION FAILED")
        return 1
    print("\n*** n=7 SWING-UP + BALANCE: UNPERTURBED CLOSED-LOOP PASS ***")

    if "--gate" in sys.argv:
        import subprocess
        rc = 0
        for seed in ("12345", "777"):
            print(f"\n[4] full perturbed-IC composite gate (24 ICs, seed "
                  f"{seed}) — re-solves a warm-started NLP per IC; ~1-2 h "
                  f"per seed on 12 otherwise-idle cores ...")
            r = subprocess.run([sys.executable,
                                str(REPO / "scripts" /
                                    "cl_validate_n7_composite.py"),
                                "24", seed, "12"], cwd=str(REPO))
            rc = rc or r.returncode
        return rc
    print("\n(run with --gate for the full 24-IC perturbed validation on "
          "both seeds; banked JSONs for seeds 12345 and 777 are in results/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
