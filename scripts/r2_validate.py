"""R2 closed-loop validation: whole-trajectory TVLQR along a saved nominal.

Given a saved nominal nom_n{n}.npz (or nom_n{n}_coarse.npz):
  1. Build whole-trajectory TVLQR linearized ALONG the nominal, terminal
     Lambda = P_static (the CARE solution at upright).
  2. Compute the closed-loop discrete monodromy along the nominal (product of
     per-tick closed-loop transition matrices) -> spectral radius.
  3. Run >= N_IC perturbed initial conditions through rollout_zoh closed-loop
     IN PARALLEL (20 workers) -> success fraction + Wilson 95% CI.

Success of a perturbed run = continuous in-success-set hold for the final 5 s
(reuses funnels.in_success_set via rollout.static_hold-style tail check), with
force/track respected over the whole rollout.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

OUT = Path("results")


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a binomial success fraction."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def build_tvlqr_along_nominal(model, t_nom, x_nom, u_nom, qf_scale_static=True):
    """TVLQR linearized along the nominal, terminal Lambda = P_static.

    Per Glueck/Kugi: terminal cost = the infinite-horizon CARE solution at
    upright so the time-varying gains converge to the steady-state LQR at t=T.
    """
    from cartpole_race.lqr import static_lqr
    from cartpole_race.tvlqr import TVLQR
    _, P = static_lqr(model)
    Qf = P  # Lambda = P_static (exact, not scaled)
    tv = TVLQR(model, t_nom, x_nom, u_nom, Qf=Qf, n_eval=400)
    return tv, P


def closed_loop_monodromy(model, tv, t_nom, x_nom, u_nom, control_dt):
    """Discrete closed-loop monodromy along the nominal at the CONTROL rate.

    The real closed loop runs at ``control_dt`` (1 ms), so the monodromy must be
    formed on that grid, not the coarse nominal spacing. Each control step is
    discretized exactly via the block-matrix exponential of the continuous
    (A, B) linearization at the nominal point (zero-order hold). With
        M = [[A*dt, B*dt], [0, 0]]   (size nx+1),  E = expm(M),
    the exact ZOH lift is Phi = E[:nx, :nx], Gamma = E[:nx, nx:]. This is exact
    even when A is singular at upright (the cart-position integrator makes A
    singular, so the A^{-1}(Phi-I)B form does not apply here). Closed loop
    Acl = Phi - Gamma K(t); monodromy is the product over every control tick
    spanning [t0, tf]. Return spectral radius.

    Nominal (x, u) at each tick come from the TVLQR's own interpolation of the
    nominal so the linearization point matches what the controller sees.
    """
    import scipy.linalg as _sla
    nx = model.nx
    t0 = float(t_nom[0])
    tf = float(t_nom[-1])
    n_ticks = int(round((tf - t0) / control_dt))
    M = np.eye(nx)
    for k in range(n_ticks):
        tk = t0 + k * control_dt
        xk, uk = tv._nom_at(tk)
        A, B = model.linearize(xk, uk)
        # Exact ZOH lift via the block-matrix exponential, valid even when A is
        # singular at upright: E = expm([[A*dt, B*dt], [0, 0]]) gives
        # Phi = E[:nx, :nx], Gamma = E[:nx, nx:].
        Mblk = np.zeros((nx + 1, nx + 1))
        Mblk[:nx, :nx] = A * control_dt
        Mblk[:nx, nx:] = B.reshape(nx, 1) * control_dt
        E = _sla.expm(Mblk)
        Phi = E[:nx, :nx]
        Gamma = E[:nx, nx:]
        K = tv.K_at(tk)  # (1, nx)
        Acl = Phi - Gamma @ K
        M = Acl @ M
    eig = np.linalg.eigvals(M)
    return float(np.max(np.abs(eig))), M


# ---- parallel perturbed-IC worker -----------------------------------------
def _run_one_ic(args):
    """Closed-loop TVLQR->static rollout from a perturbed IC. Picklable."""
    import numpy as _np
    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec
    from cartpole_race.lqr import StaticLQRPolicy, static_lqr
    from cartpole_race.tvlqr import TVLQR
    from cartpole_race.rollout import simulate_handoff

    (n, t_nom, x_nom, u_nom, force_bound, dx, hold_s) = args
    spec = CartPoleSpec().with_n_links(n)
    spec = spec.model_copy(update={"force_bound_n": force_bound})
    model = NLinkCartPole(spec)
    K, P = static_lqr(model)
    t_nom = _np.array(t_nom)
    x_nom = _np.array(x_nom)
    u_nom = _np.array(u_nom)
    # TVLQR expects matched-length (t, x, u). u has N entries vs N+1 knots;
    # pad u with its last value so interpolation over the knot grid is valid.
    if len(u_nom) == len(t_nom) - 1:
        u_nom = _np.append(u_nom, u_nom[-1])
    tv = TVLQR(model, t_nom, x_nom, u_nom, Qf=P, n_eval=400)
    static_pol = StaticLQRPolicy(model, K)
    static_pol.P = P
    catch_horizon = float(t_nom[-1])

    # Perturbed IC about the nominal start (hanging-ish); rho_static large so
    # the handoff machine is governed by the funnel test in_success_set.
    x0 = x_nom[0] + _np.array(dx)
    # rho_static: a generous level so the V-gate doesn't block the switch; the
    # actual success test is the continuous 5 s in-success-set hold.
    rho_static = 1e9
    res = simulate_handoff(
        model, x0, tv, P, rho_static, catch_horizon,
        hold_time_s=hold_s, static_lqr_policy=static_pol,
    )
    # MAX_ABS_X over the WHOLE rollout (load-bearing: documents cart excursion
    # during swing-up vs the +/-2 m hold predicate and the +/-10 m rail bound).
    max_abs_x = (
        float(_np.max(_np.abs(res.x_log[:, 0]))) if len(res.x_log) else 0.0
    )
    return {
        "success": bool(res.success),
        "max_force": float(res.max_force),
        "max_force_demanded": float(res.max_force_demanded),
        "saturated": bool(res.saturated),
        "max_abs_x": max_abs_x,
        "hold": float(res.hold_time_achieved),
        "reason": res.failure_reason,
    }


def perturbed_ic_study(n, t_nom, x_nom, u_nom, force_bound, n_ic=24,
                       pos_sigma=0.02, ang_sigma=0.02, vel_sigma=0.02,
                       hold_s=5.0, max_workers=20, seed=12345):
    """Fan out n_ic perturbed-IC closed-loop rollouts in parallel."""
    rng = np.random.default_rng(seed)
    nx = 2 * (n + 1)
    jobs = []
    for _ in range(n_ic):
        dx = np.zeros(nx)
        dx[0] = rng.normal(0, pos_sigma)              # cart pos
        dx[1:1 + n] = rng.normal(0, ang_sigma, n)     # angles
        dx[1 + n] = rng.normal(0, vel_sigma)          # cart vel
        dx[2 + n:] = rng.normal(0, vel_sigma, n)      # ang vels
        jobs.append((n, t_nom.tolist(), x_nom.tolist(), u_nom.tolist(),
                     force_bound, dx.tolist(), hold_s))

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_one_ic, j) for j in jobs]
        for fut in as_completed(futs):
            results.append(fut.result())
    wall = time.time() - t0
    k = sum(1 for r in results if r["success"])
    p, lo, hi = wilson_ci(k, len(results))
    # Number of ICs whose raw demanded force hit the saturation bound on any
    # tick (the actual count behind the "rides the bound" claim).
    n_saturated = sum(1 for r in results if r.get("saturated"))
    return {
        "n_ic": len(results), "n_success": k, "frac": p,
        "wilson_lo": lo, "wilson_hi": hi, "wall": wall,
        "force_bound": force_bound,
        "max_force_over_runs": max((r["max_force"] for r in results),
                                   default=0.0),
        "max_force_demanded_over_runs": max(
            (r.get("max_force_demanded", 0.0) for r in results), default=0.0),
        "n_saturated_ics": n_saturated,
        "max_abs_x_over_runs": max((r["max_abs_x"] for r in results),
                                   default=0.0),
        "seed": seed,
        "pos_sigma": pos_sigma, "ang_sigma": ang_sigma, "vel_sigma": vel_sigma,
    }


def validate(nom_path, n, forces=(150.0,), n_ic=24):
    """Full validation at each force in ``forces`` for a saved nominal."""
    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec

    d = np.load(nom_path)
    x_nom = d["x"]
    u_nom = d["u"]
    horizon = float(d["horizon"])
    spec = CartPoleSpec().with_n_links(n)
    model = NLinkCartPole(spec)
    control_dt = spec.control_dt_s
    N = len(u_nom)
    t_nom = np.linspace(0.0, N * control_dt, N + 1)

    # t_nom matches the x knot grid (N+1 points).
    t_nom = np.linspace(0.0, horizon, len(x_nom))
    # u has N entries; pad to N+1 for TVLQR's matched-length interpolation.
    u_pad = np.append(u_nom, u_nom[-1]) if len(u_nom) == len(t_nom) - 1 \
        else u_nom

    tv, P = build_tvlqr_along_nominal(model, t_nom, x_nom, u_pad)
    # monodromy uses per-tick spacing of the nominal grid; iterate over N ticks.
    dt_grid = horizon / (len(x_nom) - 1)
    rho, _ = closed_loop_monodromy(model, tv, t_nom, x_nom, u_nom, dt_grid)

    report = {"n": n, "horizon": horizon, "nodes": len(x_nom) - 1,
              "monodromy_rho": rho, "studies": {}}
    for fb in forces:
        st = perturbed_ic_study(n, t_nom, x_nom, u_nom, fb, n_ic=n_ic)
        report["studies"][str(int(fb))] = st
        print(f"[validate n={n} F={fb:.0f}] success {st['n_success']}/"
              f"{st['n_ic']} = {st['frac']:.2f} "
              f"CI[{st['wilson_lo']:.2f},{st['wilson_hi']:.2f}] "
              f"rho={rho:.4f} {st['wall']:.1f}s", flush=True)
    return report


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    # Prefer the 1ms zoh-consistent nominal (trackable); fall back to coarse.
    for cand in (f"nom_n{n}_zoh.npz", f"nom_n{n}.npz", f"nom_n{n}_coarse.npz"):
        path = OUT / cand
        if path.exists():
            break
    forces = (150.0,)
    if len(sys.argv) > 3:
        forces = tuple(float(x) for x in sys.argv[3].split(","))
    rep = validate(path, n, forces=forces)
    (OUT / f"validate_n{n}.json").write_text(json.dumps(rep, indent=2))
    print("saved", OUT / f"validate_n{n}.json", flush=True)
