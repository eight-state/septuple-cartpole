# METHOD: what is new at n=7

This is a delta-doc. The shared method spine (plant model, Glück/Kugi exact
inversion for the original nominal family, collocation, TVLQR concept,
validation harness) is documented in the n=5 and n=6 releases. This document
covers only what n=7 required, including the refutation of the internal
impossibility verdict that preceded it.

## 1. The impossibility verdict and its keystone error

After n=6, an internal adversarial program attacked n=7 and failed nine ways
(aggressive/gentle TVLQR tracking, NMPC, cc-iLQG, collocation replays,
staggered inversion, CLF-QP, force sweeps to 5000 N, plant co-design, PPO).
Two reverse-adversary passes distilled the failures to one claim:

> the 150 N-saturated upright catch basin at n=7 is ~0.01° / 0.01 rad/s,
> measured with the static LQR, which is locally optimal, hence
> controller-independent; every swing-up must hand off into it; therefore
> single-cart n=7 is impossible and needs a second actuator.

The keystone error: **local gain-optimality does not bound the achievable
basin under input saturation.** With seven unstable poles sharing one bounded
input, every linear law needs high gain (the LQR gain norm floors at ~4.1e5
as the control penalty grows), and clipping a high-gain multi-mode command
misdirects the single input, the classic windup phenomenon the
nested-saturation / low-and-high-gain literature (Hu & Lin) addresses. The
saturated static-LQR basin is ~0.01°; the plant's recoverable set is not.

The controller-independent object is the **null controllable region**. For
each real unstable mode lambda_i with biorthogonal left eigenvector w_i,
recoverability under ANY |u| <= u_max requires

    |w_i^T x| <= u_max |w_i^T B| / lambda_i ,

a hard per-mode necessary condition (`scripts/_ncr_hard_bound.py`; the ratio
is normalization-invariant). At the n=7 upright (unstable spectrum 2.37 ...
29.26) this allows pure-angle perturbations of **10 to 70 degrees** across
random directions; gate-sigma draws pass with worst modal ratio 0.26. The
joint-NCR support function shrinks the per-mode box by ~2x, still degrees.
Nothing physical collapses between n=6 and n=7.

A steering controller realizes this region: a 2 s constrained trajectory NLP
(4 ms RK4 nodes, |u| <= 100 N) from upright-plus-perturbation to a 2e-5 rad
terminal ball, tracked closed-loop, holds **0.1°+ perturbations in the real
saturated simulator** (10x the "controller-independent" basin) under the
same locked 5 s hold predicate (`scripts/_n7_steer_catch3.py`).

## 2. Artifact #1: no trackable n=7 nominal existed

Every historical n=7 closed-loop gate failure ran on 250-node nominals whose
closed loop had monodromy rho ~ 1e248 to 1e298 (untrackable), while the n=6 pass
used a 2800-node consistent nominal (rho = 0.028). The "0/24" results said
nothing about n=7; they measured broken inputs.

The missing artifact: re-solve the full 8 s swing-up as a 2000-node, 4 ms RK4
collocation NLP (IPOPT/MUMPS), warm-started from the legacy continuous
nominal. Converged with transcription defect 4.1e-10, **peak force 23.3 N**
of 150, terminal 0.0115°. The 4 ms grid matters twice: (lambda*h)^5 RK4 error
is ~1e-8 per step at lambda ~ 29 (10 ms nodes give ~2e-3 and fail), and 4 ms
is an integer multiple of the 1 ms control tick (2.5 ms is not, and breaks
densification).

**Why a COARSER grid than n=5/n=6 succeeded where the native-1 ms solve
failed: the design principle.** The 1 ms NLP (8000 nodes) is not a stricter
problem, only a 4x larger one, and with unstable dynamics its conditioning
worsens with node count; it never converged across months of attempts and
helped manufacture the impossibility verdict. The accuracy the fine grid
buys (defect 1e-13 vs 1e-10) is thousands of times below the deviation
scale the 1 ms feedback corrects anyway (~1e-5 rad). The reference needs
exactly two properties: a defect small enough not to lie to the controller,
and closed-loop trackability (rho < 1), NOT open-loop perfection
(open-loop replays of chaotic chains always diverge; that observation
misled the original program). Put precision only where nothing downstream
can recover it: feedback recovers plan imperfection for free; nothing
recovers an optimizer that never converges.

**Densification:** each 4 ms node's constant force is integrated through the
simulator's exact ZOH stepping (4x RK4 0.25 ms substeps per 1 ms tick),
recording every tick. The dense reference is bit-exactly sim-consistent
within segments, with node-boundary seams <= 8.34e-5 (the parent's
transcription error). Committed as tests.

## 3. Artifact #2: the continuous-Riccati TVLQR is unstable at n=7

The runtime's TVLQR integrates the continuous Riccati ODE and stores gains on
an n_eval=400 grid (20 ms effective resolution), interpolating both gains and
the reference. Along the n=7 nominal this closed loop has **monodromy
rho = 47.85**, unstable before any saturation argument applies. Two
mechanisms: gain mistiming against the lambda ~ 29 fast modes, and linear
interpolation of the reference turning reference-grid error into phantom
tracking error multiplied by ||K|| ~ 1e6.

Fix (`scripts/_dtvlqr.py`): discrete-time TVLQR. Exact ZOH discretization of
the linearization at EVERY 1 ms tick via the block matrix exponential,
backward discrete Riccati recursion, per-tick gains, zero interpolation.
Along the same nominal: **rho = 0.197**. Build time ~3 s for 8000 ticks.

With the consistent nominal + discrete TVLQR, the unperturbed closed loop
tracks with max deviation 0.00078°, hands off at 0.0115°, and the static-LQR
hold passes: the first n=7 closed-loop swing-up + balance. The "mid-swing
force spike" of the impossibility analyses (204 N ... 1.8e7 N) never
appears: those figures were gain x error on untrackable nominals; with
deviations ~1e-5 rad the demand stays in the tens of newtons.

## 4. The perturbed gate: causal per-IC replanning at t=0

Tracking the FIXED nominal from sigma=0.02 ICs passes 18/24: the six largest
draws saturate at t ~ 2.65 s, where the gain schedule ramps (2e3 -> 7.7e5)
against not-yet-contracted IC error. The fix is replan-then-track (causal, compute-unconstrained, not real-time):
**re-solve the swing-up NLP from the measured perturbed state at t=0**,
warm-started from the unperturbed nominal (banked solve times: 594 s to
27,291 s ≈ 7.6 h; the "56 to 178 s" in earlier revisions was a stale figure
from the superseded CPU-capped build), then track the replanned trajectory.
Causal (measured state only, no rewinds) but NOT real-time; no real-time
claim is made anywhere in this repo. The terminal handoff (~2e-4 rad) goes through the
steering catch into the static hold.

Result: **24/24 (seed 12345), 24/24 (seed 777)**, demanded force <= 33.9 N, track
within +/-10 m (cart excursion < ~5 m), locked 5 s hold, identical
perturbation model / simulator / predicate as n=5/n=6.

**Gate evidence construction (post-review).** The original development runs
mixed two programs: a main gate run for seed 777 that crashed
(BrokenProcessPool) after 23/23-attempted passes, plus a separate pre-roll
rescue script for the unattempted IC, under looser solver settings. An
adversarial review (2026-06-11) flagged that aggregation. The shipped gate
(`cl_validate_n7_composite.py`) now implements ONE uniform policy: stage A =
replan-at-t0; stage B = pre-roll fallback (track the fixed nominal through
the benign first 2 s, where deviation contracts ~5× near hanging, then replan
the remainder), triggered ONLY by stage A's t=0 NLP missing its iteration
budget, a signal causally available before any motion. If a committed
plan's tracking later diverges, the IC is scored as a FAILURE: a controller
cannot rewind time. (An earlier revision had a divergence-triggered
fallback, which a third external review correctly flagged as non-causal; no
banked result ever exercised that branch, all 48 banked ICs converged via
stage A, and the earlier CPU-time-capped build had routed ICs through stage
B purely by load-dependent budget misses. The per-IC stage field in the
JSONs is the ground truth.)

**Determinism (corrected after a second external review).** The first
v1.0.1 build budgeted solves by CPU time (1800 s), which an independent
re-execution proved makes stage selection load-dependent (the same IC took
stage B under contention and stage A on an idle machine, both passing, with
different trajectories). All budgets are now iteration-count-only
(max_iter=1500, no time caps): on a given platform with the pinned
single-threaded numerics the solver path, stage selection, and success
counts are load-independent. Cross-platform bitwise identity of
IPOPT/MUMPS float paths is NOT established by version pinning (different
platform binaries can pivot differently), so the precise claim is
load-independence per platform, not universal machine-independence.
Incidentally the round-2 re-execution showed the same IC passes via BOTH
stages, so the per-IC outcome is robust to which plan the policy lands on.
One solver tier for every NLP (adaptive mu, acceptable_tol 1e-4,
acceptable_iter 8, max_iter 1500, no time caps): relaxed tolerances are
safe because the saturated-sim hold predicate is the judge, not the solver
status. The per-IC stage and solver settings are recorded in the output
JSON together with commit_sha and a git_dirty flag (a dirty tree means the
recorded commit cannot regenerate the artifact), and both seed JSONs in
`results/` are single-run artifacts regenerated from a clean checkout.
Two measurement notes: (a) handoff_dev_rad ≈ 2.0e-4 across all 48 ICs is
the replan NLP's terminal-ball constraint being active, so it measures the
plan's terminal box, not tracking quality (tracking error is ~1e-5 rad);
(b) the force-in-bound property holds by construction (simulator np.clip)
and is deliberately NOT a success conjunct, so the live saturation disclosure
is max_force_demanded / n_saturated_ics. Controllers may carry compute;
everything is causal.

## 5. Cross-n statement

n=6 passed with the simple recipe because its continuous TVLQR happened to
contract (rho = 0.028) and its saturated static-LQR basin (~0.05°) exceeded
its contracted handoff scatter. At n=7 both margins crossed zero, and both
were restored by standard tools (discrete-time gains; basin realization via
steering; per-IC replanning). Through n=7, cross-n difficulty grows through
**controller numerics** (transcription fidelity, gain discretization, basin
realization), not through a hard single-input authority wall. Peak force
used (NOMINAL, i.e. unperturbed feedforward peaks): 21.6 N (n=6, of 60)
and 23.3 N (n=7, of 150); perturbed-validation peaks are 38.6 N applied
(n=6) and 33.9 N demanded (n=7).

## 6. API note vs the n=5/n=6 runtimes

This repo's `dynamics.rollout_zoh` returns the 3-tuple `(t, x, u)`; the
n=5/n=6 releases return a 4-tuple that additionally carries the raw
(pre-clip) demanded force, which fed their `max_abs_force_demanded`
disclosure. Here the same disclosure is produced at the gate layer instead: a
force-logging policy wrapper records the pre-clip demand of every stage, and
the gate JSONs report `max_force_demanded` per IC plus
`max_force_demanded_over_runs` / `n_saturated_ics` per run. Same
information, different plumbing, noted to avoid confusion when diffing the
runtimes.

## 7. Internal provenance

The internal research notes (claims ledger with verified/failed/audited rows,
two prior reverse-adversary passes, and the pass-3 refutation) live in the
private research archive (`cartpole-research/research-notes/`,
`n7-claims-ledger.md`, `n7-reverse-adversary-{1,2,3}.md`). The headline
artifacts and numbers in this repo are regenerated from the committed code
and the shipped nominals.
