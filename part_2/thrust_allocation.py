"""
Thrust allocation following

    T. A. Johansen, T. P. Fuglseth, P. Tøndel, T. I. Fossen (2008),
    "Optimal constrained control allocation in marine surface vessels
    with rudders", Control Engineering Practice.

The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

The azimuths each have a forbidden zone, which makes the attainable thrust
region non-convex. It is split into convex pieces and one QP is solved per
combination of pieces; the cheapest solution is used.(MIQP)

In difference to the paper the QPs are solved online and not offline, so
each solve has an iteration limit, and a scaled pseudo-inverse (without
forbidden zones) is used if no QP succeeds.

Part 2 (changes to the Part 1 allocator are marked "Part 2" in the code,
see part_2/thrust_allocation.md):
- an infeasible demand is scaled down, B u = lambda tau_c with
  0 <= lambda <= 1, so the achieved wrench keeps the requested direction
- an azimuth is commanded (u, alpha) or (-u, alpha + pi), whichever needs
  less rotation from its actual angle
- optional second stage (cfg.rate_aware): a QP over the thrust and azimuth
  states the thrusters can reach within one step
- no wall-clock limits by default, so every run is reproducible
- the fallback scales the whole solution instead of clipping each thruster
"""
import time
from itertools import product

import cvxpy as cp
import numpy as np

from models.thruster_dynamics import ThrusterConfig
from part_2.config import AllocationConfig
from simulation.utils import wrap_angle_pi as wrap


class ThrustAllocator:

    def __init__(self, thrusters: list[ThrusterConfig],
                 cfg: AllocationConfig | None = None):
        self.thrusters = thrusters
        self.cfg = cfg if cfg is not None else AllocationConfig()
        self.n = len(thrusters)
        self.is_az = [th.kind == "azimuth" for th in thrusters]

        # Part 2: same options for every QP, time limit only if set
        self._solver_opts = {"max_iter": self.cfg.max_iter}
        if self.cfg.qp_time_limit is not None:
            self._solver_opts["time_limit"] = self.cfg.qp_time_limit

        self._build_B()
        self._build_regions()
        self._build_problems()
        if self.cfg.rate_aware:
            self._build_rate_problem()
        self.reset()

    def reset(self) -> None:
        self._u_prev: np.ndarray | None = None
        self._alpha_prev: np.ndarray | None = None
        self._combo: int | None = None
        self.used_fallback = False
        self.n_fallback = 0
        # Part 2: +1 forward, -1 reverse thrust, per thruster
        self._sign_prev = np.ones(self.n)
        # Part 2: logged every call, see history_arrays()
        self.last_lambda = np.ones(2)
        self.history = {key: [] for key in ("t", "lam", "slack", "tau_alloc", "u_ss",
                                            "alpha_ss", "combo", "fallback")}

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: np.ndarray | None = None,
        alpha_now: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tau_c = np.asarray(tau_d, float).reshape(-1)[[0, 1, 5]]
        if self._u_prev is None:
            self._init_previous(u_now, alpha_now)

        self.p_tau.value = tau_c / self.tau_scale
        self.p_u0.value = self._u_prev

        if self.cfg.use_fallback_only:
            costs, sols = np.full(len(self.problems), np.inf), []
        else:
            costs, sols = self._solve_subproblems()
        self.used_fallback = not np.isfinite(costs).any()
        if self.used_fallback:
            self.n_fallback += 1
            u_ext, lam, slack = self._fallback(self.p_tau.value)
        else:
            self._combo = self._select_combo(costs)
            u_ext, lam, slack = sols[self._combo]
        self._u_prev = u_ext
        self.last_lambda = lam

        # Part 2: the sign of an azimuth command depends on its actual angle
        alpha_ref = self._alpha_prev if alpha_now is None else np.asarray(alpha_now, float)
        u_cmd, alpha_cmd = self._to_thruster_commands(u_ext, alpha_ref)
        self._alpha_prev = alpha_cmd.copy()
        u_ss, alpha_ss = u_cmd.copy(), alpha_cmd.copy()

        # Part 2: optional second stage, a command the thrusters reach in one step
        if self.cfg.rate_aware and u_now is not None and alpha_now is not None:
            rate_sol = self._solve_rate_problem(dt, lam, u_ext, alpha_ss,
                                                np.asarray(u_now, float),
                                                np.asarray(alpha_now, float))
            if rate_sol is not None:
                u_cmd, alpha_cmd, slack = rate_sol

        # Part 2: solver tolerances must never give more than u_max
        u_max = np.array([th.u_max for th in self.thrusters])
        u_cmd = np.clip(u_cmd, -u_max, u_max)
        self._log(t, lam, slack, u_cmd, alpha_cmd, u_ss, alpha_ss)
        return u_cmd, alpha_cmd

    def history_arrays(self) -> dict[str, np.ndarray]:
        """Part 2: logged allocation signals, one row per allocate() call.

        lam              [force, moment] scale of the demand, < 1 when tau_c
                         is not attainable (equal without yaw_priority)
        slack            [N, N, Nm] QP wrench error on top of lam * tau_c
        tau_alloc        [N, N, Nm] wrench of the command, B(alpha_cmd) u_cmd
        u_ss, alpha_ss   steady-state command, differs from the returned
                         command only with rate_aware
        combo, fallback  sector combination used (-1 = fallback)
        """
        return {key: np.asarray(val) for key, val in self.history.items()}

    def _build_B(self) -> None:
        """Extended thrust u = [f_bow, X1, Y1, X2, Y2] with tau = B u ."""
        self.idx = []  # slice of u belonging to each thruster
        cols, scale = [], []
        for th, az in zip(self.thrusters, self.is_az):
            start = len(scale)
            if az:
                cols += [[1.0, 0.0, -th.y], [0.0, 1.0, th.x]]
                scale += [th.u_max, th.u_max]
            else:
                a = th.alpha0
                cols += [[np.cos(a), np.sin(a), th.x * np.sin(a) - th.y * np.cos(a)]]
                scale += [th.u_max]
            self.idx.append(slice(start, len(scale)))
        self.B = np.array(cols).T
        self.nu = self.B.shape[1]

        # Normalize weights
        self.u_scale = np.array(scale)
        B_scaled = self.B * self.u_scale
        self.tau_scale = np.linalg.norm(B_scaled, axis=1)
        self.B_norm = B_scaled / self.tau_scale[:, None]

    def _build_regions(self) -> None:
        # Each azimuth region is a list of half-plane normals n (n . [X, Y] >= 0),
        # None means the full disk
        self.regions = []
        for th, az in zip(self.thrusters, self.is_az):
            self.regions.append(self._azimuth_pieces(th) if az else [None])
        self.combos = list(product(*[range(len(r)) for r in self.regions]))

    def _azimuth_pieces(self, th: ThrusterConfig) -> list:
        """Split disk minus forbidden sector into two half-disks."""
        w = self.cfg.forbidden_half_width
        if w <= 0.0:
            return [None]
        # Thrust pointing outboard blows the jet across onto the other azimuth
        c = np.sign(th.y) * np.pi / 2
        # Half-disk A covers angles [c+w, c+w+pi], B covers [c-w-pi, c-w]
        nA, nB = c + w + np.pi / 2, c - w - np.pi / 2
        return [np.array([np.cos(nA), np.sin(nA)]), np.array([np.cos(nB), np.sin(nB)])]

    def _build_problems(self) -> None:
        self.p_tau = cp.Parameter(3)
        self.p_u0 = cp.Parameter(self.nu)

        H, M, Q = self.cfg.H, self.cfg.M, self.cfg.Q

        # Weighted pseudo-inverse for the fallback
        H_inv = np.linalg.inv(H + 1e-6 * np.eye(self.nu))
        self.B_pinv = H_inv @ self.B_norm.T @ np.linalg.inv(self.B_norm @ H_inv @ self.B_norm.T)

        # Weights are diagonal. sum_squares instead of quad_form keeps the
        # problem DPP, so cvxpy only compiles it once.
        # Part 2: kept as attributes, the rate-aware QP uses them too
        self.sqrt_H, self.sqrt_M, self.sqrt_Q = np.sqrt(H), np.sqrt(M), np.sqrt(Q)

        # Polygon inside the unit circle, corners on |u_k| = 1
        angles = 2 * np.pi * np.arange(self.cfg.n_poly) / self.cfg.n_poly
        poly_normals = np.column_stack([np.cos(angles), np.sin(angles)])
        poly_radius = np.cos(np.pi / self.cfg.n_poly)

        self.problems = []
        for combo in self.combos:
            u = cp.Variable(self.nu)
            s = cp.Variable(3)
            # Part 2: lam = [force, moment] scales the demand and is < 1 only if
            # tau_c is not attainable, so s stays ~0 and B u keeps the direction
            # of tau_c. Without yaw_priority both factors are the same
            lam = cp.Variable(2)
            scale = cp.hstack([lam[0], lam[0], lam[1]])
            cons = [self.B_norm @ u + s == cp.multiply(scale, self.p_tau),
                    lam >= 0.0, lam <= 1.0]
            if self.cfg.yaw_priority:
                cost_lam = (self.cfg.q_lambda * (1.0 - lam[0])
                            + self.cfg.q_lambda_yaw * (1.0 - lam[1]))
            else:
                cons += [lam[1] == lam[0]]
                cost_lam = self.cfg.q_lambda * (1.0 - lam[0])
            for k, piece in enumerate(combo):
                u_k = u[self.idx[k]]
                if not self.is_az[k]:
                    cons += [cp.abs(u_k) <= 1.0]
                    continue
                cons += [poly_normals @ u_k <= poly_radius]
                normal = self.regions[k][piece]
                if normal is not None:
                    cons += [normal @ u_k >= 0.0]
                # Part 2: the Part 1 rate cone is removed here, rates are left
                # to the actuator model or to the rate-aware second stage
            cost = (cp.sum_squares(self.sqrt_H @ u)
                    + cp.sum_squares(self.sqrt_M @ (u - self.p_u0))
                    + cp.sum_squares(self.sqrt_Q @ s)
                    + cost_lam)
            self.problems.append((cp.Problem(cp.Minimize(cost), cons), u, lam, s))

    def _build_rate_problem(self) -> None:
        """Part 2: same QP, constrained to what the thrusters reach in one step.

        Each azimuth angle is fixed at the angle it reaches next step, so its
        force lies on a line through the origin, and each signed thrust can
        only change by u_rate * dt. The target lam * tau_c comes from the
        steady-state QPs, the cost pulls towards their solution u_ss.
        """
        self.p_target = cp.Parameter(3)
        self.p_uss = cp.Parameter(self.nu)
        self.p_lo = cp.Parameter(self.n)
        self.p_hi = cp.Parameter(self.n)
        self.p_axis = {k: cp.Parameter(2) for k in range(self.n) if self.is_az[k]}
        self.p_perp = {k: cp.Parameter(2) for k in range(self.n) if self.is_az[k]}

        u = cp.Variable(self.nu)
        s = cp.Variable(3)
        cons = [self.B_norm @ u + s == self.p_target]
        for k in range(self.n):
            u_k = u[self.idx[k]]
            # Signed thrust along the direction of the next step
            t_k = self.p_axis[k] @ u_k if self.is_az[k] else u_k[0]
            cons += [t_k >= self.p_lo[k], t_k <= self.p_hi[k]]
            if self.is_az[k]:
                cons += [self.p_perp[k] @ u_k == 0.0]
        cost = (cp.sum_squares(self.sqrt_H @ u)
                + cp.sum_squares(self.sqrt_M @ (u - self.p_uss))
                + cp.sum_squares(self.sqrt_Q @ s))
        self.rate_problem = (cp.Problem(cp.Minimize(cost), cons), u, s)

    def _init_previous(self, u_now: np.ndarray | None,
                       alpha_now: np.ndarray | None) -> None:
        # First step: start from the actual thruster state
        if u_now is not None and alpha_now is not None:
            self._u_prev = self._to_extended(np.asarray(u_now, float),
                                             np.asarray(alpha_now, float))
            self._alpha_prev = np.asarray(alpha_now, float).copy()
            # Part 2: keep the sign the thrusters already run with
            self._sign_prev = np.where(np.asarray(u_now, float) < 0.0, -1.0, 1.0)
        else:
            self._u_prev = np.zeros(self.nu)
            self._alpha_prev = np.array([th.alpha0 for th in self.thrusters])

    def _set_rate_limits(self, dt: float, alpha_target: np.ndarray,
                         u_now: np.ndarray, alpha_now: np.ndarray) -> np.ndarray:
        """Part 2: replaces _set_rate_cone. One-step reachable set from the
        actual state, returns the angles reached next step."""
        alpha_next = np.zeros(self.n)
        lo, hi = np.zeros(self.n), np.zeros(self.n)
        for k, th in enumerate(self.thrusters):
            du = th.u_rate * dt
            lo_k, hi_k = max(-th.u_max, u_now[k] - du), min(th.u_max, u_now[k] + du)
            if self.is_az[k]:
                # Turn towards the steady-state angle at full rotation speed
                step = th.rot_speed * dt
                a = wrap(alpha_now[k] + np.clip(wrap(alpha_target[k] - alpha_now[k]), -step, step))
                lo_k, hi_k = self._forbidden_bounds(th, a, lo_k, hi_k)
                self.p_axis[k].value = np.array([np.cos(a), np.sin(a)])
                self.p_perp[k].value = np.array([-np.sin(a), np.cos(a)])
            else:
                a = th.alpha0
            alpha_next[k] = a
            lo[k], hi[k] = lo_k / th.u_max, hi_k / th.u_max
        self.p_lo.value, self.p_hi.value = lo, hi
        return alpha_next

    def _forbidden_bounds(self, th: ThrusterConfig, a: float,
                          lo: float, hi: float) -> tuple[float, float]:
        """Part 2: no thrust into the forbidden sector while an azimuth turns
        through it (as fast as the thrust rate allows)."""
        w = self.cfg.forbidden_half_width
        if w <= 0.0:
            return lo, hi
        c = np.sign(th.y) * np.pi / 2
        if abs(wrap(a - c)) < w:  # positive thrust points into the sector
            hi = max(min(hi, 0.0), lo)
        elif abs(wrap(a + np.pi - c)) < w:  # negative thrust points into it
            lo = min(max(lo, 0.0), hi)
        return lo, hi

    def _solve_subproblems(self) -> tuple[np.ndarray, list]:
        # Previous combination first, so it is solved even if time runs out
        costs = np.full(len(self.problems), np.inf)
        sols = [None] * len(self.problems)
        order = list(range(len(self.problems)))
        if self._combo is not None:
            order.remove(self._combo)
            order.insert(0, self._combo)
        t_start = time.perf_counter()
        budget = self.cfg.step_time_budget
        for n_tried, i in enumerate(order):
            # Part 2: wall-clock budget only if set (None keeps runs reproducible)
            if budget is not None and n_tried > 0 and time.perf_counter() - t_start > budget:
                break
            prob, u, lam, s = self.problems[i]
            try:
                prob.solve(solver=cp.CLARABEL, **self._solver_opts)
            except cp.SolverError:
                continue
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u.value is not None:
                costs[i] = prob.value
                # Part 2: lam and the slack are returned too, for logging
                sols[i] = (u.value.copy(), np.clip(lam.value, 0.0, 1.0), s.value.copy())
        return costs, sols

    def _solve_rate_problem(self, dt: float, lam: np.ndarray, u_ss: np.ndarray,
                            alpha_target: np.ndarray, u_now: np.ndarray,
                            alpha_now: np.ndarray):
        """Part 2: thrusts for the angles the azimuths reach next step."""
        alpha_next = self._set_rate_limits(dt, alpha_target, u_now, alpha_now)
        self.p_target.value = lam[[0, 0, 1]] * self.p_tau.value
        self.p_uss.value = u_ss
        prob, u, s = self.rate_problem
        try:
            prob.solve(solver=cp.CLARABEL, **self._solver_opts)
        except cp.SolverError:
            return None
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or u.value is None:
            return None
        u_cmd = np.zeros(self.n)
        for k, th in enumerate(self.thrusters):
            u_k = u.value[self.idx[k]]
            t_k = self.p_axis[k].value @ u_k if self.is_az[k] else u_k[0]
            u_cmd[k] = t_k * th.u_max
        return u_cmd, alpha_next, s.value.copy()

    def _select_combo(self, costs: np.ndarray) -> int:
        # Hysteresis against chattering between combinations with similar cost
        best = int(np.argmin(costs))
        prev = self._combo
        if (prev is not None and np.isfinite(costs[prev])
                and costs[best] > (1.0 - self.cfg.hysteresis) * costs[prev]):
            return prev
        return best

    def _fallback(self, tau_n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """pseudo inverse (no forbidden zones included)."""
        u = self.B_pinv @ tau_n
        # Part 2: one factor for all thrusters instead of clipping each one,
        # so B u keeps the direction of tau_c
        ratio = 0.0
        for k in range(self.n):
            limit = np.cos(np.pi / self.cfg.n_poly) if self.is_az[k] else 1.0
            ratio = max(ratio, np.linalg.norm(u[self.idx[k]]) / limit)
        lam = 1.0 / max(1.0, ratio)
        return lam * u, np.full(2, lam), np.zeros(3)

    def _to_extended(self, u: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        u_ext = np.zeros(self.nu)
        for k, az in enumerate(self.is_az):
            if az:
                u_ext[self.idx[k]] = u[k] * np.array([np.cos(alpha[k]), np.sin(alpha[k])])
            else:
                u_ext[self.idx[k]] = u[k]
        return u_ext / self.u_scale

    def _to_thruster_commands(self, u_ext: np.ndarray,
                              alpha_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        forces = u_ext * self.u_scale
        u_cmd = np.zeros(self.n)
        alpha_cmd = np.zeros(self.n)
        for k, th in enumerate(self.thrusters):
            f = forces[self.idx[k]]
            if not self.is_az[k]:
                u_cmd[k], alpha_cmd[k] = f[0], th.alpha0
            elif np.hypot(*f) >= self.cfg.u_min_dir:
                # Part 2: (u, a) and (-u, a + pi) give the same force, take
                # the one with less rotation from the actual angle
                a = np.arctan2(f[1], f[0])
                sign = self._choose_sign(k, a, alpha_ref[k])
                u_cmd[k] = sign * np.hypot(*f)
                alpha_cmd[k] = a if sign > 0 else wrap(a + np.pi)
            else:
                # Angle is undefined near zero thrust, so keep the old one
                a = self._alpha_prev[k]
                u_cmd[k], alpha_cmd[k] = f[0] * np.cos(a) + f[1] * np.sin(a), a
        return u_cmd, alpha_cmd

    def _choose_sign(self, k: int, a: float, alpha_ref: float) -> float:
        """Part 2: +1 for (u, a), -1 for (-u, a + pi). The sign only changes if
        the other one saves more than reverse_hysteresis of rotation."""
        if not self.cfg.allow_reverse_thrust:
            return 1.0
        rot_fwd = abs(wrap(a - alpha_ref))
        rot_rev = np.pi - rot_fwd
        margin = self.cfg.reverse_hysteresis
        if self._sign_prev[k] > 0:
            sign = -1.0 if rot_rev < rot_fwd - margin else 1.0
        else:
            sign = 1.0 if rot_fwd < rot_rev - margin else -1.0
        self._sign_prev[k] = sign
        return sign

    def _wrench(self, u: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Part 2: [X, Y, N] of signed thrusts u at angles alpha."""
        x = np.array([th.x for th in self.thrusters])
        y = np.array([th.y for th in self.thrusters])
        c, s = np.cos(alpha), np.sin(alpha)
        return np.array([u @ c, u @ s, u @ (x * s - y * c)])

    def _log(self, t: float, lam: np.ndarray, slack: np.ndarray, u_cmd: np.ndarray,
             alpha_cmd: np.ndarray, u_ss: np.ndarray, alpha_ss: np.ndarray) -> None:
        """Part 2: one row of history per call."""
        h = self.history
        h["t"].append(t)
        h["lam"].append(lam)
        h["slack"].append(slack * self.tau_scale)
        h["tau_alloc"].append(self._wrench(u_cmd, alpha_cmd))
        h["u_ss"].append(u_ss)
        h["alpha_ss"].append(alpha_ss)
        h["combo"].append(-1 if self.used_fallback else self._combo)
        h["fallback"].append(self.used_fallback)
