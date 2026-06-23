"""n=7 closed-loop gate: ONE program, ONE uniform policy, self-describing output.

Gate physics identical to the n=5/n=6 releases: REAL saturated sim
(rollout_zoh, hard 150 N clip, 1 ms ZOH, RK4 substeps), sigma=0.02 perturbed
ICs at hanging (cart pos/vel + every link angle/rate), locked predicate v1
(|theta|<=5 deg, |thetad|<=0.5, |x|<=2 m, |xd|<=0.5, continuous 5 s hold;
force/track over the whole rollout).

THE COMPOSITE CONTROLLER (one policy, applied uniformly to every IC):
  stage A  replan-at-t0: re-solve the 8 s swing-up NLP from the measured
           perturbed state (warm-started from the unperturbed 4 ms nominal),
           densify, track with exact-ZOH discrete TVLQR.
  stage B  pre-roll fallback, triggered ONLY by stage A's NLP failing its
           iteration budget AT t=0 — a signal causally available before any
           motion. (If stage A's NLP succeeds but its tracking later
           diverges, the IC is a FAILURE: a real controller cannot rewind
           time and try a different plan from t=0.) Stage B tracks the
           FIXED nominal for the benign first 2 s, then replans the
           remaining 6 s from the measured mid-state. The stage used is
           recorded per IC in the output.
  then     steering-NLP catch from the measured handoff state -> static LQR
           hold.

FORCE PREDICATE NOTE: the applied force is clipped to the bound by the
simulator (hard np.clip), so "force within bound over the whole rollout"
holds BY CONSTRUCTION and gates nothing — same as the n=5/n=6 releases. It
is therefore not part of the success conjunction. The honest saturation
disclosure is the separate pre-clip demand log: per-IC max_force_demanded
and the run-level n_saturated_ics.

ONE solver tier for every NLP in every stage (recorded in the output JSON):
  mu_strategy=adaptive, acceptable_tol=1e-4, acceptable_iter=8,
  max_iter=1500. The budget is ITERATION-ONLY by design: a wall/CPU-time cap
  would make stage selection (and worst-case pass/fail) depend on machine
  speed and load, breaking the determinism of the success counts. With the
  pinned single-threaded environment (OMP/BLAS=1) the solver's iteration
  path, the stage taken, and every rollout are machine-independent; only
  wall time varies. The honest judge is the saturated-sim hold predicate
  downstream, which a sloppy plan cannot fake.

Per-IC output includes max CLIPPED force, max DEMANDED (pre-clip) force, and
whether the demand ever hit the bound (the saturation-disclosure fields the
n=5/n=6 releases carry).

Usage: cl_validate_n7_composite.py [n_ic] [seed] [workers]
Writes: results/clvalidate_n7_composite_seed<seed>.json
"""
import os, sys, time, json
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

SOLVER_TIER = {
    "CARTPOLE_MU_STRATEGY": "adaptive",
    "CARTPOLE_ACCEPTABLE_TOL": "1e-4",
    "CARTPOLE_ACCEPTABLE_ITER": "8",
}
for k, v in SOLVER_TIER.items():
    os.environ[k] = v
# Iteration-only budget — deliberately NO max_cpu_time (see module docstring).
os.environ.pop("CARTPOLE_MAX_CPU_S", None)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

NOM_DENSE = str(REPO / "results" / "nom_n7_dense1ms.npz")
NOM_4MS = str(REPO / "results" / "nom_n7_4ms.npz")
T_STEER = 2.0
N_STEER_NODES = 500
U_PLAN = 100.0
TERM_TOL = 2e-5
HOLD_S = 5.0
T_PREROLL = 2.0
MAX_ITER = 1500


class _ForceLogged:
    """Wrap a raw-policy callable; clip at fb; record demanded force."""

    def __init__(self, raw, fb):
        self.raw = raw
        self.fb = fb
        self.max_demand = 0.0
        self.saturated = False

    def __call__(self, x, t):
        u = float(self.raw(x, t))
        a = abs(u)
        if a > self.max_demand:
            self.max_demand = a
        if a > self.fb:
            self.saturated = True
        return float(np.clip(u, -self.fb, self.fb))


def one_ic(args):
    (dx, tag) = args
    import numpy as _np
    from cartpole_race.dynamics import NLinkCartPole
    from cartpole_race.env_spec import CartPoleSpec
    from cartpole_race.collocation import solve_trajopt
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
    n_sub = max(1, int(_np.ceil(dt / spec.rk4_max_step_s)))
    dt_sub = dt / n_sub
    stride = 4

    dd = _np.load(NOM_DENSE)
    Xn_f = dd["x"]; Un_f = dd["u"]; Tn = float(dd["horizon"])
    dc = _np.load(NOM_4MS)
    Xc = dc["x"]; Uc = dc["u"]
    Ncoarse = len(Uc)
    K, P = static_lqr(m)
    Krow = _np.asarray(K).reshape(-1)
    xup = m.x_equilibrium("up")
    x0 = Xn_f[0] + _np.asarray(dx)
    rec = {"tag": tag}
    demands = []
    logs_xu = []

    def densify(Xp, Up):
        Xd = [Xp[0]]; Ud = []
        for k in range(len(Up)):
            xx = Xp[k].astype(float).copy()
            for _ in range(stride):
                for _ in range(n_sub):
                    xx = m.rk4_step(xx, float(Up[k]), dt_sub)
                Xd.append(xx.copy()); Ud.append(float(Up[k]))
        return _np.array(Xd), _np.array(Ud)

    # ---- stage A: replan-at-t0 ----
    t0 = time.time()
    rp = solve_trajopt(m, x0, horizon_s=Tn, n_nodes=Ncoarse,
                       terminal_tol_rad=2e-4, force_bound=U_PLAN, w_u=1e-4,
                       x_init_guess=Xc, u_init_guess=Uc,
                       zoh_consistent=False, max_iter=MAX_ITER, print_level=0)
    rec["stageA_solve_s"] = round(time.time() - t0, 1)
    if rp.success:
        # Stage A committed: causal — once the plan exists and the rollout
        # starts, tracking divergence is a FAILURE, never a rewind.
        rec["stage"] = "A_replan_t0"
        Xd, Ud = densify(rp.x, rp.u)
        tv1 = DiscreteTVLQR(m, Xd, Ud, dt)
        pol1 = _ForceLogged(tv1.policy, fb)
        t1, x1, u1 = m.rollout_zoh(x0, pol1, Tn, dt, spec.rk4_max_step_s)
        xh = x1[-1]
        if _np.any(_np.isnan(xh)) or float(_np.max(
                _np.abs(xh[1:1 + n] - Xd[-1, 1:1 + n]))) > 0.5:
            rec.update(success=False, fail_stage="A_track_diverged")
            return rec
        demands.append(pol1)
        logs_xu.append((x1, u1))
    else:
        # ---- stage B: pre-roll fallback (causal trigger: the t=0 NLP
        # failed its iteration budget before any motion) ----
        rec["stage"] = "B_preroll"
        tvf = DiscreteTVLQR(m, Xn_f, Un_f, dt)
        polf = _ForceLogged(tvf.policy, fb)
        tA, xA, uA = m.rollout_zoh(x0, polf, T_PREROLL, dt,
                                   spec.rk4_max_step_s)
        demands.append(polf)
        logs_xu.append((xA, uA))
        x_mid = xA[-1]
        T_rem = Tn - T_PREROLL
        N_rem = int(round(T_rem / (stride * dt)))
        kc = int(round(T_PREROLL / (stride * dt)))
        t0 = time.time()
        rp2 = solve_trajopt(m, x_mid, horizon_s=T_rem, n_nodes=N_rem,
                            terminal_tol_rad=2e-4, force_bound=U_PLAN,
                            w_u=1e-4, x_init_guess=Xc[kc:kc + N_rem + 1],
                            u_init_guess=Uc[kc:kc + N_rem],
                            zoh_consistent=False, max_iter=MAX_ITER,
                            print_level=0)
        rec["stageB_solve_s"] = round(time.time() - t0, 1)
        if not rp2.success:
            rec.update(success=False, fail_stage="B_replan_nlp")
            return rec
        Xd, Ud = densify(rp2.x, rp2.u)
        tv2 = DiscreteTVLQR(m, Xd, Ud, dt)
        pol2 = _ForceLogged(tv2.policy, fb)
        t1, x1, u1 = m.rollout_zoh(x_mid, pol2, T_rem, dt,
                                   spec.rk4_max_step_s)
        demands.append(pol2)
        logs_xu.append((x1, u1))
        xh = x1[-1]
        if _np.any(_np.isnan(xh)):
            rec.update(success=False, fail_stage="B_track")
            return rec

    hdev = float(_np.max(_np.abs(((xh[1:1 + n] - xup[1:1 + n] + _np.pi)
                                  % (2 * _np.pi)) - _np.pi)))
    rec["handoff_dev_rad"] = round(hdev, 8)

    # ---- steering catch ----
    t0 = time.time()
    st = solve_trajopt(m, xh, horizon_s=T_STEER, n_nodes=N_STEER_NODES,
                       terminal_tol_rad=TERM_TOL, force_bound=U_PLAN,
                       w_u=1e-4, zoh_consistent=False, max_iter=MAX_ITER,
                       print_level=0)
    rec["steer_solve_s"] = round(time.time() - t0, 1)
    if not st.success:
        rec.update(success=False, fail_stage="steer_nlp")
        return rec
    Xs, Us = densify(st.x, st.u)
    tv3 = DiscreteTVLQR(m, Xs, Us, dt)
    pol3 = _ForceLogged(tv3.policy, fb)
    t2, x2, u2 = m.rollout_zoh(xh, pol3, T_STEER, dt, spec.rk4_max_step_s)
    demands.append(pol3)
    logs_xu.append((x2, u2))

    # ---- static hold ----
    def _static_raw(x, t):
        return -float(Krow @ wrap_state_error(x, xup, n))

    pol4 = _ForceLogged(_static_raw, fb)
    t3, x3, u3 = m.rollout_zoh(x2[-1], pol4, HOLD_S + 1.0, dt,
                               spec.rk4_max_step_s)
    demands.append(pol4)
    logs_xu.append((x3, u3))
    in_set = [in_success_set(m, xx) for xx in x3]
    run = 0
    for v_ in in_set:
        run = run + 1 if v_ else 0
    peakF = float(max(_np.max(_np.abs(u)) for _, u in logs_xu))
    track_ok = bool(max(float(_np.max(_np.abs(x[:, 0]))) for x, _ in logs_xu)
                    <= track)
    # NOTE: no force conjunct — applied force respects the bound BY
    # CONSTRUCTION (simulator np.clip), so such a check is vacuous (it can
    # never fail) and would misleadingly imply a live gate. Saturation
    # honesty lives in max_force_demanded / saturated below.
    rec.update(
        success=bool(max(0, run - 1) * dt >= HOLD_S - 1e-9 and track_ok),
        peakF=round(peakF, 3),
        max_force_demanded=round(max(d.max_demand for d in demands), 3),
        saturated=bool(any(d.saturated for d in demands)),
        track_ok=track_ok)
    if not rec["success"]:
        rec.setdefault("fail_stage", "hold")
    return rec


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
    print(f"[CL-COMPOSITE n=7] uniform-policy gate  n_ic={n_ic} seed={seed} "
          f"sigma=0.02  solver_tier={SOLVER_TIER}", flush=True)
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one_ic, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print("  ", json.dumps(r, default=str), flush=True)
    k = sum(1 for r in results if r.get("success"))
    n_sat = sum(1 for r in results if r.get("saturated"))
    mfd = max((r.get("max_force_demanded", 0.0) for r in results),
              default=0.0)
    print(f"[CL-COMPOSITE n=7] {k}/{n_ic} success  "
          f"max_force_demanded={mfd:.1f}N n_saturated_ics={n_sat} "
          f"({time.time()-t0:.0f}s)", flush=True)
    # provenance + statistics (parity with the n=6 release JSONs).
    # commit_sha is only meaningful with git_dirty=False: a dirty tree means
    # the recorded commit cannot regenerate this artifact.
    import hashlib, subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                capture_output=True, text=True,
                                timeout=10).stdout.strip() or None
        porcelain = subprocess.run(["git", "status", "--porcelain"],
                                   cwd=REPO, capture_output=True, text=True,
                                   timeout=10).stdout
        dirty = bool(porcelain.strip())
    except Exception:
        commit = None
        dirty = None
    nom_sha = hashlib.sha256(Path(NOM_DENSE).read_bytes()).hexdigest()
    z = 1.959963984540054
    ph = k / n_ic
    den = 1 + z * z / n_ic
    ctr = (ph + z * z / (2 * n_ic)) / den
    hw = z * ((ph * (1 - ph) / n_ic + z * z / (4 * n_ic * n_ic)) ** 0.5) / den
    wilson = [max(0.0, ctr - hw), min(1.0, ctr + hw)]
    # subset runs must not clobber the banked full-gate artifact
    suffix = f"_n{n_ic}" if n_ic != 24 else ""
    out = REPO / "results" / f"clvalidate_n7_composite_seed{seed}{suffix}.json"
    out.write_text(json.dumps(
        {"n_success": k, "n_ic": n_ic, "seed": seed,
         "wilson_95": wilson, "commit_sha": commit, "git_dirty": dirty,
         "nominal_sha256": nom_sha,
         "solver_tier": SOLVER_TIER, "max_iter": MAX_ITER,
         "max_force_demanded_over_runs": mfd, "n_saturated_ics": n_sat,
         "results": sorted(results, key=lambda r: r["tag"])},
        indent=1, default=str))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
