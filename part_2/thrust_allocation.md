# Thrust allocation, Part 2

Code: [thrust_allocation.py](thrust_allocation.py), settings: `AllocationConfig` in
[config.py](config.py). Everything that differs from the Part 1 allocator
(`part_1/thrust_allocation.py`) is marked `# Part 2` in the code.

## 1. Method (kept from Part 1)

The allocation follows the structure of Johansen, Fuglseth, Tøndel and Fossen (2008),
*Optimal constrained control allocation in marine surface vessels with rudders*:
extended thrust variables, a non-convex attainable thrust region split into convex
polyhedral pieces, and one convex QP per combination of pieces.

* **Extended thrust**: `u = [f_T, X_1, Y_1, X_2, Y_2]`. The tunnel thrust `f_T` acts along
  its fixed 90° axis, and each azimuth is described by its two body-frame force components.
  This gives a constant, linear `tau = B u` (rank 3). Every QP works on normalized
  quantities: `u_n = u / u_max` and `tau_n = tau / tau_scale`.
* **Thrust limit**: `|(X_k, Y_k)| <= u_max` is approximated by an inscribed 16-gon, which
  loses at most 1.9 % of the thrust in some directions. The tunnel has the box
  `|f_T| <= u_max`.
* **Forbidden zones**: each azimuth must not push within ±20° of its outboard direction,
  because its jet would then hit the other azimuth. This makes the set non-convex. It is
  split into two half-disks per azimuth, so 2 × 2 = 4 convex QPs are solved every step. The
  cheapest one is used, with a 10 % hysteresis against switching between combinations.
* **Online instead of offline**: the paper computes the explicit solution of each convex
  piece offline with multi-parametric QP. Here, each QP is solved online by Clarabel through
  cvxpy. The problems are DPP, so cvxpy compiles them only once. If no QP succeeds, a
  weighted pseudo-inverse is used (Section 6).

## 2. Steady-state QP (Part 2)

For each combination of half-disks:

```
min   u_n' H u_n + (u_n - u_prev)' M (u_n - u_prev) + s' Q s + q_lambda (1 - lambda)
s.t.  B_n u_n + s = lambda * tau_n,     0 <= lambda <= 1
      16-gon per azimuth, half-plane of the chosen half-disk, |f_T| <= 1
```

The new variable `lambda` scales the demanded wrench:

* **Feasible `tau_c`**: `lambda = 1` and `s ≈ 0`, so `B u = tau_c`. In 60 random feasible
  demands the error was below 0.3 N and 3 Nm.
* **Infeasible `tau_c`**: `lambda < 1`, so `B u = lambda * tau_c`. The achieved wrench has
  the same direction as the demand and a smaller magnitude. This is the rule asked for in
  Section 8.2 of the assignment.

  * **What it keeps**: the ratios X : Y : N. A position correction keeps its direction, and
    no yaw moment appears that was not asked for.
  * **What it gives up**: all three DOFs are reduced by the same factor, even a DOF that
    could have been delivered on its own. When the surge demand is far too large, yaw is
    therefore reduced just as much.

**Heading priority (option, `yaw_priority = True`).** `lambda = [lambda_F, lambda_N]` has
separate factors: `B u + s = [lambda_F X, lambda_F Y, lambda_N N]`. The moment factor is
weighted 10× higher (`q_lambda_yaw = 1000`), so the moment is kept first, and the force
(X, Y) still keeps its direction. Examples:

* [992, -427, 99.8] → [155, -67, 99.8], with `lambda = [0.16, 1.0]`.
* [200, 100, 600] → [71, 35, 600]. Uniform scaling gives [102, 51, 305].

In Simulation 2 (Section 5) this option only helped together with `rate_aware`. On its own,
it made the run worse, so it is off by default.

Measured with one call of the allocator through ideal actuators:

| Demand [X, Y, N] (kN, kNm) | Part 1 (slack only) | Part 2 (`lambda`) | lambda |
|---|---|---|---|
| [0, 400, 0] | [-11, 184, -1468] | [0, 94.7, 0] | 0.237 |
| [200, 100, 0] | [154, 71, -118] | [134, 67, 0] | 0.669 |
| [0, 0, 5000] | [0, -101, 2380] | [0, 0, 1270] | 0.254 |

With `lambda`, the direction error is below 0.003° in every case.

**Weights.**

* `q_lambda` must be larger than the thrust cost that a smaller wrench saves. That cost is
  `dJ/dlambda ≈ 2 (H + M) |u_n|² < 10` here, so `q_lambda = 100`.
* `Q` must be much larger than `q_lambda`. Otherwise the solver trades `lambda` against
  slack, and the slack, which can point in any direction, grows to about
  `q_lambda / (2 Q |tau_n|)`. For that reason `Q` was raised from 1e4 (Part 1) to 1e6.
  This leaves a slack of about 1e-4 of the scale, roughly 10 N and 0.2 kNm.
* `H` and `M` are the same as in Part 1.

The Part 1 rate cone (`eps_rate`) is removed from this QP. Rate limits are handled by the
actuator model or by the optional second stage (Section 5).

## 3. From extended forces to thruster commands

* **Tunnel**: `u = f_T`, `alpha = 90°`.
* **Azimuth with `|f| >= 500 N`**: the force can be produced by `(|f|, atan2(Y, X))` or by
  `(-|f|, atan2(Y, X) + pi)` (Part 2). The allocator takes the option that needs less
  rotation from the **actual** angle `alpha_now`. It only switches to the other option if
  that saves more than 10° (`reverse_hysteresis`).

  For example, slowing down from forward thrust now gives reverse thrust at 0°, instead of
  a 180° slew that takes 15 s at 12 °/s. The supplied actuator model treats reverse thrust
  like forward thrust. A real azimuth is weaker in reverse; `allow_reverse_thrust = False`
  switches back to the Part 1 behaviour.
* **Azimuth with `|f| < 500 N`**: the angle is held and the force is projected onto it, as
  in Part 1.

## 4. Allocation and actuator dynamics: the residual

The allocator computes a steady-state command, `B(alpha_cmd) u_cmd = lambda tau_c - s`. The
supplied actuator model then ramps the thrust (4 or 10 kN/s) and slews the azimuths (12 °/s).
The vessel receives `tau_thr = B(alpha) u` from the actual states. The residual of the
assignment splits into two parts:

```
r_tau = tau_c - tau_thr
      = [ tau_c - tau_alloc ]  +  [ tau_alloc - tau_thr ]
        = (1 - lambda) tau_c + s    actuator transient
        steady state: u_max         (thrust ramp, azimuth slew)
```

Here `tau_alloc = B(alpha_cmd) u_cmd`.

* **Steady-state part**: comes from the allocator. It is nonzero only while the demand
  exceeds the thrust limits (`lambda < 1`). This is the persistent part, and it is what
  winds up the integrator.
* **Transient part**: comes from the actuator model. It is nonzero while the thrusters are
  still ramping or turning, and it disappears once they reach the command.

The controller's back-calculation anti-windup (`apply_external_aw`) uses `tau_thr`, so it
sees both parts.

The allocator logs every call. `allocator.history_arrays()` returns `t`, `lam` ([force,
moment]), `slack`
[N, N, Nm], `tau_alloc` [N, N, Nm], the steady-state command `u_ss` and `alpha_ss`, `combo`
and `fallback`. `run()` in `run_case_part_2.py` attaches them as `logs.allocation`:

```python
logs = rc.sim2()
h = logs.allocation
r_ss = logs.tau_d[:, [0, 1, 5]] - h["tau_alloc"]        # u_max part
r_tr = h["tau_alloc"] - logs.tau_thr[:, [0, 1, 5]]      # ramp / slew part
plt.plot(h["t"], h["lam"][:, 0])                        # where u_max is active
```

## 5. Optional extension: rate-aware second stage (`rate_aware = True`)

The assignment calls this an optional extension, so it is off by default. When it is on, a
second QP runs after the steady-state QPs:

* **Predicted angle**: each azimuth is fixed at the angle it reaches in the next step when
  it turns towards the steady-state angle at full speed:
  `alpha_next = alpha_now + clip(wrap(alpha_ss - alpha_now), ±alpha_dot_max dt)`.
  Its force then lies on a line.
* **Thrust bounds**: each signed thrust can only change by `u_dot_max dt`. While an azimuth
  turns through its forbidden sector, it gets no thrust into that sector.
* **Same QP**: the cost has the same `H`, `M` and `Q`, but `u_prev` is replaced by the
  steady-state solution `u_ss`. The target is `lambda tau_c` from the first stage, so the
  direction rule still applies. The slack now carries the error that the rates cause.

The result can be reached exactly in one step, so `tau_thr[k+1] = tau_alloc[k]`. While an
azimuth turns, the QP uses the tunnel and the other azimuth to make up for it, and it does
not push in the wrong direction. Once the thrusters have arrived, the second stage returns
the steady-state solution. It looks only one step ahead and costs one more QP, about
+2 ms per step.

With `rate_aware`, the commands go to the actuators one step at a time, so the supplied
plot "set-point vs actual" shows almost no difference. To see the active rate limits, plot
the steady-state target `logs.allocation["u_ss"]` / `["alpha_ss"]` against the actual state.

### Simulation 2 comparison

Setup: four-corner test, 1500 s, `dt = 0.1 s`, constrained thrusters, no environment, raw
feedback. Controller: Part 2 `DPController()`. Reference: the Part 1 reference model
(`REF_CFG_XY` / `REF_CFG_PSI`), because `part_2/reference.py` was still the pass-through
template. Errors are measured against the reference; "transient" is the RMS of
`tau_alloc - tau_thr`.

| Metric | Part 1 alloc. | steady (default) | steady + yaw prio | rate-aware | rate + yaw prio |
|---|---|---|---|---|---|
| RMS error N / E [m] | 8.2 / 7.8 | 5.6 / 2.2 | 8.4 / 5.4 | 5.4 / 2.4 | 5.5 / 2.6 |
| RMS / max heading error [°] | 41 / 107 | 7.3 / 51 | 8.2 / 42 | 4.9 / 27 | 3.3 / 18 |
| legs settled (of 5) | 0 | 4 | 2 | 4 | 4 |
| RMS residual X / Y / N [kN, kNm] | 340 / 277 / 353 | 227 / 63 / 51 | 392 / 111 / 74 | 227 / 52 / 37 | 225 / 72 / 30 |
| transient part [kN, kNm] | – | 55 / 12 / 53 | 105 / 23 / 74 | 0.07 / 0.12 / 0.8 | ≈ 0 |
| azimuth travel [°] | 14970 / 14837 | 2270 / 2854 | 3945 / 3050 | 1647 / 1779 | 1884 / 1848 |
| mean utilisation | 82 % | 18 % | 49 % | 15 % | 16 % |
| wall time | 118 s | 120 s | 129 s | 147 s | 164 s |

* **Part 1 allocator under Part 2 limits**: it never settles. The likely causes are the 180°
  slew (15 s) that every deceleration needs, and the distorted infeasible wrenches, which
  turn the vessel. The two causes were not tested separately.
* **Rate-aware stage**: the transient part of the residual disappears. The heading error
  drops (RMS 7.3° → 4.9°, max 51° → 27°) and the azimuths turn about 30 % less.
* **Heading priority on its own**: it changes the azimuth targets abruptly. The slews then
  push in the wrong direction, and legs 2–4 do not settle.
* **Leg 1 (pure surge)**: it fails to settle in every variant. The cause is the demand, not
  the allocation: with the Part 1 reference, the controller asks for up to 780–940 kN of
  surge, while about 157 kN is available. `lambda < 1` for 86 % of leg 1, and the vessel
  overshoots by 16 m. With a slower reference (`wn = 0.04`, 0.5 m/s), the demand peaks at
  65 kN, `lambda` stays 1, and leg 1 settles to 0.02 m. The Part 2 reference model has to be
  matched to the thrust limits; this is Section 7 of the assignment.
* **Update — the controller gains were the main cause.** `part_2/controller.py` imported
  `LQR_Gains` from `part_1.config`, so the runs above used the Part 1 gains (surge
  Kp = 299 kN/m). With the import fixed to `part_2.config` (Kp = 75 kN/m), the default
  allocator and the same Part 1 reference give:
  * all 5 legs settled, leg 1 to 0.000 m;
  * peak demand 144 kN / 169 kN / 330 kNm, instead of 993 / 434 / 1079;
  * `lambda < 1` 2.8 % of the time, instead of 24 %;
  * max error against the reference 5.3 m / 6.4 m / 15.7°, instead of 26 m / 16 m / 52°.

  With the slower reference as well, `lambda` stays 1 and the max error is 1.3 m / 1.3 m /
  12.3°. The table above was not rerun with the fixed gains.
* **Solver statuses**: in 600 s of rate-aware operation, 24 004 steady-state QPs were all
  optimal. Of 6001 second-stage QPs, 4 returned `optimal_inaccurate`, which is accepted.
  There were no fallbacks.

## 6. Fallback

If no QP succeeds, the weighted pseudo-inverse `u = H^-1 B' (B H^-1 B')^-1 tau_n` is used,
without forbidden zones. Part 2 change: when a thruster would exceed its limit, the whole
vector is scaled by one factor `lambda`, instead of clipping each thruster on its own. This
keeps the direction, as in Section 2. `allocator.n_fallback` counts how often it was used.
It was 0 in all test runs.

## 7. Reproducibility and run time

* **Wall-clock limits**: in Part 1, `qp_time_limit` (0.02 s) and `step_time_budget`
  (0.05 s) depended on the computer's speed and load. Two runs of the same case could
  therefore take different fallback decisions. They are now `None` by default. Only
  `max_iter = 50` limits the solver, and that limit is deterministic.
* **Run time**: about 7 ms per step (9 ms with `rate_aware`) at `dt = 0.1 s`. Simulation 2
  takes about 2 minutes, and the Simulation 5 sweep (36 × 600 s) about 26 minutes.

## 8. Settings (`AllocationConfig`, part_2/config.py)

| Field | Value | Part 1 | Meaning |
|---|---|---|---|
| `H` | diag(1) | same | thrust use |
| `M` | diag(0.1) | same | change from the previous solution |
| `Q` | diag(1e6) | 1e4 | wrench error (slack) |
| `q_lambda` | 100 | new | weight on `1 - lambda` |
| `yaw_priority` | False | new | separate, higher-weighted scale for the moment |
| `q_lambda_yaw` | 1000 | new | weight on `1 - lambda_N` (only with `yaw_priority`) |
| `forbidden_half_width` | 20° | same | half-width of the forbidden sector |
| `n_poly` | 16 | same | sides of the thrust polygon |
| `hysteresis` | 0.10 | same | switching of sector combination |
| `u_min_dir` | 500 N | same | hold the azimuth angle below this thrust |
| `allow_reverse_thrust` | True | new | allow `(-u, alpha + pi)` |
| `reverse_hysteresis` | 10° | new | rotation that must be saved to change sign |
| `rate_aware` | False | replaces `eps_rate` | second stage, Section 5 |
| `max_iter` | 50 | same | Clarabel iterations per QP |
| `qp_time_limit` | None | 0.02 s | wall-clock limit per QP |
| `step_time_budget` | None | 0.05 s | wall-clock limit per step |
| `use_fallback_only` | False | same | pseudo-inverse only, for comparison |

The checks build `ThrustAllocator(thrusters)`, so the final values must be these defaults.

## 9. Changes from Part 1

| Where | Change |
|---|---|
| `_build_problems` | adds `lambda = [lambda_F, lambda_N]` (`B u + s = lambda tau`, `0 <= lambda <= 1`, cost `q_lambda (1 - lambda)`, both factors equal unless `yaw_priority`); removes the rate cone |
| `_to_thruster_commands`, `_choose_sign` | chooses the sign of each azimuth command from the actual angle, with hysteresis |
| `_build_rate_problem`, `_set_rate_limits`, `_forbidden_bounds`, `_solve_rate_problem` | optional rate-aware second stage; replaces `_set_rate_cone` |
| `_fallback` | scales the whole solution instead of clipping each thruster |
| `_solve_subproblems`, `__init__` | wall-clock limits only if set; the solutions also return `lambda` and `s` |
| `allocate` | runs the second stage when enabled, clips to `u_max` against solver tolerances, logs |
| `history_arrays`, `_log`, `_wrench` | logging of `lambda`, slack and allocated wrench |
| `config.py` | Part 2 `AllocationConfig` (new fields; `Q`; no time limits) |
| `run_case_part_2.py`, `run()` | attaches `logs.allocation` |

`part_1/` is unchanged.

## 10. Verification

* `python check.py --part 2`: the allocation check passes. That covers the four isolated
  wrenches, which must be reproduced through ideal actuators, `|u_cmd| <= u_max` at every
  call, and the infeasible 400 kN sway demand. It also passes with `rate_aware = True`.
* `pytest tests`: 21 passed and 5 skipped. The 5 skipped tests are the Part 2 checks whose
  blocks are still templates: current, wind, utilisation, reference model and NPO.
* DPP holds for all QPs, also with `yaw_priority` and `rate_aware`, so cvxpy compiles each
  problem only once.
* Feasible demands are met exactly. Infeasible ones keep their direction (Section 2).

## 11. Limitations

* With uniform scaling (the default), a far too large demand in one DOF reduces the others
  by the same factor. Heading priority exists as an option, but it needs `rate_aware` to
  help (Section 5).
* The 16-gon and the half-disks are linear approximations. They give up up to 1.9 % of the
  thrust, and the forbidden sectors are fixed in the body frame.
* The rate-aware stage is greedy: it looks one step ahead and always turns the azimuths
  towards the current steady-state angle.
* Reverse thrust is assumed as strong as forward thrust, as in the supplied actuator model.
* Thruster–hull interaction and losses from current are not modelled.
