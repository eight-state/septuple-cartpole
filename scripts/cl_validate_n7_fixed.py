"""APPLES-TO-APPLES n=7 gate: the SAME controller architecture as the n=5/n=6
releases — a single FIXED nominal + exact-ZOH discrete TVLQR tracking ->
static-LQR hold, with NO per-IC replanning and NO steering catch.

This is the honest like-for-like number against n5 (88/88) and n6 (48/48):
it shows how far a fixed-nominal feedback law alone carries n=7 at sigma=0.02,
before the heavier composite controller (cl_validate_n7_composite.py) adds
per-IC replanning to reach 24/24. Same perturbation model, simulator, seeds,
and predicate v1 as every other gate here.

Usage: cl_validate_n7_fixed.py [n_ic] [seed] [workers]
Writes: results/clvalidate_n7_fixed_seed<seed>.json
"""
import os, sys, time, json
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
NOM_DENSE = str(REPO / "results" / "nom_n7_dense1ms.npz")
HOLD_S = 5.0


def one_ic(args):
    (dx, tag) = args
    import numpy as _np
    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec
    from cartpole_race.lqr import static_lqr, wrap_state_error
    from cartpole_race.funnels import in_success_set
    from _dtvlqr import DiscreteTVLQR

    n = 7
    spec = CartPoleSpec(n_links=n, cart_mass_kg=1.0,
                        link_masses_kg=[0.10] * n, link_lengths_m=[0.50] * n,
                        damping_links_n_m_s_rad=[0.0] * n, force_bound_n=150.0)
    m = NLinkCartPole(spec)
    fb = spec.force_bound_n
    dt = spec.control_dt_s
    track = spec.track_half_length_m
    d = _np.load(NOM_DENSE)
    Xn = d["x"]; Un = d["u"]; Tn = float(d["horizon"])
    K, P = static_lqr(m)
    Krow = _np.asarray(K).reshape(-1)
    xup = m.x_equilibrium("up")
    tv = DiscreteTVLQR(m, Xn, Un, dt)
    x0 = Xn[0] + _np.asarray(dx)

    maxF = 0.0

    def track_pol(x, t):
        nonlocal maxF
        u = tv.policy(x, t)
        maxF = max(maxF, abs(u))
        return float(_np.clip(u, -fb, fb))

    t1, x1, u1 = m.rollout_zoh(x0, track_pol, Tn, dt, spec.rk4_max_step_s)
    xh = x1[-1]
    if _np.any(_np.isnan(xh)):
        return {"tag": tag, "success": False, "fail_stage": "track_nan"}

    def hold_pol(x, t):
        nonlocal maxF
        u = -float(Krow @ wrap_state_error(x, xup, n))
        maxF = max(maxF, abs(u))
        return float(_np.clip(u, -fb, fb))

    t3, x3, u3 = m.rollout_zoh(xh, hold_pol, HOLD_S + 1.0, dt,
                               spec.rk4_max_step_s)
    in_set = [in_success_set(m, xx) for xx in x3]
    run = 0
    for v_ in in_set:
        run = run + 1 if v_ else 0
    track_ok = bool(max(float(_np.max(_np.abs(x1[:, 0]))),
                        float(_np.max(_np.abs(x3[:, 0])))) <= track)
    return {"tag": tag,
            "success": bool(max(0, run - 1) * dt >= HOLD_S - 1e-9 and track_ok),
            "max_force_demanded": round(maxF, 3),
            "track_ok": track_ok}


def main():
    n_ic = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    n = 7
    nx = 2 * (n + 1)
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n_ic):
        dx = np.zeros(nx)
        dx[0] = rng.normal(0, 0.02)
        dx[1:1 + n] = rng.normal(0, 0.02, n)
        dx[1 + n] = rng.normal(0, 0.02)
        dx[2 + n:] = rng.normal(0, 0.02, n)
        jobs.append((dx, i))
    print(f"[CL-FIXED n=7] n5/n6-equivalent (fixed nominal + discrete TVLQR, "
          f"NO replan) n_ic={n_ic} seed={seed} sigma=0.02", flush=True)
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one_ic, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print("  ", json.dumps(r, default=str), flush=True)
    k = sum(1 for r in results if r.get("success"))
    print(f"[CL-FIXED n=7] {k}/{n_ic} success (fixed-nominal feedback only) "
          f"({time.time()-t0:.0f}s)", flush=True)
    import hashlib, subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                capture_output=True, text=True,
                                timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                     cwd=REPO, capture_output=True, text=True,
                     timeout=10).stdout.strip())
    except Exception:
        commit = None; dirty = None
    nom_sha = hashlib.sha256(Path(NOM_DENSE).read_bytes()).hexdigest()
    out = REPO / "results" / f"clvalidate_n7_fixed_seed{seed}.json"
    out.write_text(json.dumps(
        {"controller": "fixed nominal + discrete TVLQR, no replan "
                       "(n5/n6-equivalent architecture)",
         "n_success": k, "n_ic": n_ic, "seed": seed,
         "commit_sha": commit, "git_dirty": dirty, "nominal_sha256": nom_sha,
         "results": sorted(results, key=lambda r: r["tag"])},
        indent=1, default=str))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
