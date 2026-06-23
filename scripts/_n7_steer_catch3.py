"""Steering catch v3: coarse NLP plan + DENSIFICATION to the exact 1 ms sim
grid + TVLQR on the dense grid.

Why v1/v2 failed: TVLQR linearly interpolates a 10 ms reference; with
||K||~4e5 the interpolation error of states between nodes (1e-4..1e-3 rad)
is treated as tracking error -> phantom corrections of 100s of N -> divergence.

Densify: for each plan node k, integrate the REAL sim stepping (ZOH 1 ms,
4x RK4 substeps) from the PLAN state x_k under constant u_k, recording every
tick. Per-tick defect ~ transcription error / 10; node-boundary jump ~1e-7.
TVLQR then sees a reference with no interpolation artifacts (policy queries
land exactly on grid ticks).
"""
import os, sys, time, json
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

N_LINKS = 7
T_STEER = 2.0
N_NODES = 500
U_PLAN = 100.0
TERM_TOL = 2e-5


def densify(m, res):
    """Integrate the real sim stepping segment-wise from plan nodes."""
    spec = m.spec
    control_dt = spec.control_dt_s
    n_sub = max(1, int(np.ceil(control_dt / spec.rk4_max_step_s)))
    dt_sub = control_dt / n_sub
    N = len(res.u)
    ticks_per_node = int(round((res.t[1] - res.t[0]) / control_dt))
    Xd = [res.x[0]]
    Ud = []
    for k in range(N):
        x = res.x[k].astype(float).copy()
        for _ in range(ticks_per_node):
            for _ in range(n_sub):
                x = m.rk4_step(x, float(res.u[k]), dt_sub)
            Xd.append(x.copy())
            Ud.append(float(res.u[k]))
    Xd = np.array(Xd)
    Ud = np.array(Ud)
    td = np.arange(len(Xd)) * control_dt
    # seam jumps at node boundaries (integration vs plan node)
    seams = [float(np.max(np.abs(Xd[(k + 1) * ticks_per_node] - res.x[k + 1])))
             for k in range(N)]
    return td, Xd, Ud, float(np.max(seams))


def one_ic(args):
    (n, deg, dx_ang, t_steer, u_plan, n_nodes) = args
    import numpy as _np
    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec
    from cartpole_race.collocation import solve_trajopt
    from cartpole_race.lqr import StaticLQRPolicy, static_lqr
    from cartpole_race.tvlqr import TVLQR
    from cartpole_race.rollout import simulate_handoff

    spec = CartPoleSpec(n_links=n, cart_mass_kg=1.0,
                        link_masses_kg=[0.10] * n, link_lengths_m=[0.50] * n,
                        damping_links_n_m_s_rad=[0.0] * n, force_bound_n=150.0)
    m = NLinkCartPole(spec)
    xup = m.x_equilibrium("up")
    x0 = xup.copy()
    x0[1:1 + n] += dx_ang

    t0 = time.time()
    res = solve_trajopt(m, x0, horizon_s=t_steer, n_nodes=n_nodes,
                        terminal_tol_rad=TERM_TOL, force_bound=u_plan,
                        w_u=1e-4, zoh_consistent=False, max_iter=900,
                        print_level=0)
    solve_s = time.time() - t0
    if not res.success:
        return {"deg": deg, "success": False, "stage": "nlp",
                "solver": res.solver_status, "solve_s": solve_s}

    td, Xd, Ud, seam = densify(m, res)
    K, P = static_lqr(m)
    u_pad = _np.append(Ud, Ud[-1])
    tv = TVLQR(m, td, Xd, u_pad, Qf=P, n_eval=400)
    static_pol = StaticLQRPolicy(m, K)
    static_pol.P = P
    hr = simulate_handoff(m, x0, tv, P, 1e9, float(td[-1]),
                          hold_time_s=5.0, static_lqr_policy=static_pol)
    return {"deg": deg, "success": bool(hr.success), "stage": "cl",
            "max_force": float(hr.max_force), "seam": seam,
            "plan_peakF": float(_np.max(_np.abs(res.u))),
            "solve_s": solve_s}


def main():
    degs = [float(s) for s in (sys.argv[1] if len(sys.argv) > 1
                               else "0.05,0.1,0.2,0.5,1.0,2.0").split(",")]
    ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    u_plan = float(sys.argv[4]) if len(sys.argv) > 4 else U_PLAN
    n = N_LINKS
    print(f"[STEER-CATCH v3 densified] n={n} T={T_STEER}s nodes={N_NODES} "
          f"u_plan={u_plan}N ntr={ntr}", flush=True)
    all_rows = {}
    for deg in degs:
        rng = np.random.default_rng(1)
        jobs = []
        for _ in range(ntr):
            dx = rng.normal(0, np.deg2rad(deg), n)
            jobs.append((n, deg, dx, T_STEER, u_plan, N_NODES))
        results = []
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one_ic, j) for j in jobs]
            for f in as_completed(futs):
                r = f.result()
                results.append(r)
        ok = sum(1 for r in results if r["success"])
        nlp_fail = sum(1 for r in results if r.get("stage") == "nlp")
        pf = max((r.get("max_force", 0.0) for r in results), default=0.0)
        seam = max((r.get("seam", 0.0) for r in results), default=0.0)
        print(f"  perturb {deg:>5}deg -> hold {ok}/{ntr} "
              f"(nlp_fails={nlp_fail}) peakF~{pf:.0f}N maxseam={seam:.1e} "
              f"({time.time()-t0:.0f}s)", flush=True)
        all_rows[deg] = results
        if ok == 0:
            break
    out = Path("results/steer_catch_n7_v3.json")
    out.write_text(json.dumps(all_rows, indent=1, default=str))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
