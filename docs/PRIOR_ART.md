# Prior art: multi-link inverted pendulums (n=7)

This is the accounting behind the claim that this repo is, to our knowledge,
the **first public n=7 cart-pole swing-up-and-balance artifact by any
method**, and certainly the first open-source, code-reproducible one. The
full prior-art tables for n=5 and n=6 (Glück 2013 hardware triple, Lam &
Davison 2006 base-torque stabilization to n=7, the unverified quintuple
video, Kotelovych 2024 Isaac Sim n=5 balance, yacine's 2026-06-09 public RL
n=6 post) live in the sibling releases:

- n=5: https://github.com/eight-state/quintuple-cartpole
- n=6: https://github.com/eight-state/sextuple-cartpole (docs/PRIOR_ART.md)

What changes at n=7:

| Work | System | Links | Task | Why distinct from this repo |
|---|---|---|---|---|
| Oh et al. 2025 (RL) | Cart | n=4 | Swing-up + balance | The standing **published** frontier by any method. RL; three links short. |
| Lam & Davison 2006 | Bottom-pivot torque chain (**not a cart**) | up to n=7 | **Balance only** | Different plant (base torque), different task (local stabilization, never swing-up). |
| yacine (@yacineMTB), 2026-06-09 | Cart (MuJoCo, pufferlib PPO) | n=6 | Swing-up + balance (RL) | Public-first at n=6 (conceded in the n=6 repo). No n=7 claim, no released code artifact seen. |
| Our n=5 / n=6 releases | Cart | 5, 6 | Swing-up + balance | The immediate predecessors; n=7 was subsequently judged "~99% impossible" by our own internal adversarial program before this repo refuted that. |
| **This repo (septuple-cartpole)** | Cart (single 150 N force) | **n=7** | **Swing-up + balance** | To our knowledge no public n=7 cart-pole swing-up claim exists by ANY method as of 2026-06-11 (searched: peer-reviewed literature, preprints, public posts; the published field stops at n=4). Reproducible from a clean clone in one command; both validation seeds regenerated in-repo. |

## Honest scope

Same boundary as the siblings: simulation only (1 kHz saturated ODE sim, not
hardware), full-state feedback, exact model, deterministic; robustness is
empirical (script-verified counts under a documented perturbation
distribution and committed predicate v1), not a theorem. The "first" claim is
"first public artifact we could find," dated 2026-06-10 (first closed-loop
pass) / 2026-06-11 (review-hardened v1.0.1), it is falsifiable by
counter-example and we will concede priority exactly as the n=6 repo did if
a prior public n=7 claim surfaces.
