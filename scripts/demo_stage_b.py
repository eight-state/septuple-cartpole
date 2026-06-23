"""DEMONSTRATION (not a validation artifact): exercise the stage-B fallback
end-to-end under the FINAL shipped policy code.

Under the final iteration-only budget all 48 banked validation ICs converge
via stage A, so stage B has zero executions in the banked evidence (review
round 5, finding N7-2). This script forces the fallback path honestly: it
wraps solve_trajopt so the FIRST solve of the IC (stage A's t=0 replan) gets
an artificially tiny iteration budget (max_iter=10, guaranteed budget-miss),
while every subsequent solve (stage B's remainder replan, the steering
catch) runs the real shipped budget. Everything else — policy logic, sim,
predicate — is the shipped gate code, unmodified.

Output: results/stageB_demonstration_seed12345_tag0.json, with an explicit
"demonstration" field. It shows the fallback works; it proves nothing about
validation statistics and is excluded from all counts.

    uv run python scripts/demo_stage_b.py
"""
import io, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import cl_validate_n7_composite as G
from cartpole_race import collocation as C

_real = C.solve_trajopt
_calls = {"k": 0}


def _wrapped(*a, **kw):
    _calls["k"] += 1
    if _calls["k"] == 1:               # stage A's t=0 replan only
        kw["max_iter"] = 10            # guaranteed budget miss
    return _real(*a, **kw)


# the gate module imported solve_trajopt inside one_ic from cartpole_race
C.solve_trajopt = _wrapped
# also patch the symbol one_ic resolves (it imports inside the function)
import cartpole_race.collocation
cartpole_race.collocation.solve_trajopt = _wrapped

rng = np.random.default_rng(12345)
nx = 16
dx = np.zeros(nx)
dx[0] = rng.normal(0, 0.02)
dx[1:8] = rng.normal(0, 0.02, 7)
dx[8] = rng.normal(0, 0.02)
dx[9:] = rng.normal(0, 0.02, 7)

rec = G.one_ic((dx, 0))
rec["demonstration"] = ("stage-A budget artificially reduced to max_iter=10 "
                        "to force the stage-B fallback; NOT a validation "
                        "artifact; excluded from all counts")
print(json.dumps(rec, indent=1, default=str))
out = REPO / "results" / "stageB_demonstration_seed12345_tag0.json"
io.open(out, "w", encoding="utf-8").write(
    json.dumps(rec, indent=1, default=str))
print("saved", out)
assert rec.get("stage") == "B_preroll", "fallback did not engage"
assert rec.get("success"), "stage-B demonstration did not pass"
print("STAGE-B DEMONSTRATION: PASS")
