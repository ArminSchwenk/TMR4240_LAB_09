"""
Thrust allocation following

    T. A. Johansen, T. P. Fuglseth, P. Tøndel, T. I. Fossen (2008),
    "Optimal constrained control allocation in marine surface vessels
    with rudders", Control Engineering Practice.

The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

The azimuths each have a forbidden zone, which makes the attainable thrust
region non-convex. It is split into convex pieces and one QP is solved per
combination of pieces; the cheapest solution is used.

In difference to the paper the QPs are solved online and not offline, so
each solve has an iteration and time limit, and a saturated pseudo-inverse
is used if no QP succeeds.
"""
import time
from itertools import product

import cvxpy as cp
import numpy as np

from models.thruster_dynamics import ThrusterConfig
from part_1.config import AllocationConfig


class ThrustAllocator:
    """Disjunctive QP thrust allocator (Johansen et al. 2008)."""

    def __init__(self, thrusters: list[ThrusterConfig],
                 cfg: AllocationConfig | None = None):
        self.thrusters = thrusters
        self.cfg = cfg if cfg is not None else AllocationConfig()
        self.n = len(thrusters)
        self.is_az = [th.kind == "azimuth" for th in thrusters]

        self._build_B()
        self._build_regions()
        self._build_problems()
        self.reset()

    def reset(self) -> None:
        self._u_prev: np.ndarray | None = None
        self._alpha_prev: np.ndarray | None = None
        self._combo: int | None = None
        self.used_fallback = False
        self.n_fallback = 0

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
        self._set_rate_cone(self._u_prev)

        costs, sols = self._solve_subproblems()
        self.used_fallback = not np.isfinite(costs).any()
        if self.used_fallback:
            self.n_fallback += 1
            u_ext = self._fallback(self.p_tau.value)
        else:
            self._combo = self._select_combo(costs)
            u_ext = sols[self._combo]
        self._u_prev = u_ext

        u_cmd, alpha_cmd = self._to_thruster_commands(u_ext)
        self._alpha_prev = alpha_cmd.copy()
        return u_cmd, alpha_cmd

    def _build_B(self) -> None:
        """Extended thrust u = [f_bow, X1, Y1, X2, Y2] with tau = B u (eq. 5)."""
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

        # Work in u / u_max and tau / tau_scale so the weights are comparable
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
        # Rate cone per azimuth: |normal . u_k| <= axis . u_k, with axis = eps * d
        self.cone_axis = {k: cp.Parameter(2) for k in range(self.n) if self.is_az[k]}
        self.cone_normal = {k: cp.Parameter(2) for k in range(self.n) if self.is_az[k]}

        H, M, Q = self.cfg.H, self.cfg.M, self.cfg.Q

        # Weighted pseudo-inverse for the fallback
        H_inv = np.linalg.inv(H + 1e-6 * np.eye(self.nu))
        self.B_pinv = H_inv @ self.B_norm.T @ np.linalg.inv(self.B_norm @ H_inv @ self.B_norm.T)

        # Weights are diagonal. sum_squares instead of quad_form keeps the
        # problem DPP, so cvxpy only compiles it once
        sqrt_H, sqrt_M, sqrt_Q = np.sqrt(H), np.sqrt(M), np.sqrt(Q)

        # Polygon inside the unit circle, corners on |u_k| = 1
        angles = 2 * np.pi * np.arange(self.cfg.n_poly) / self.cfg.n_poly
        poly_normals = np.column_stack([np.cos(angles), np.sin(angles)])
        poly_radius = np.cos(np.pi / self.cfg.n_poly)

        self.problems = []
        for combo in self.combos:
            u = cp.Variable(self.nu)
            s = cp.Variable(3)
            cons = [self.B_norm @ u + s == self.p_tau]
            for k, piece in enumerate(combo):
                u_k = u[self.idx[k]]
                if not self.is_az[k]:
                    cons += [cp.abs(u_k) <= 1.0]
                    continue
                cons += [poly_normals @ u_k <= poly_radius]
                normal = self.regions[k][piece]
                if normal is not None:
                    cons += [normal @ u_k >= 0.0]
                cons += [self.cone_normal[k] @ u_k <= self.cone_axis[k] @ u_k,
                         -self.cone_axis[k] @ u_k <= self.cone_normal[k] @ u_k]
            cost = (cp.sum_squares(sqrt_H @ u) + cp.sum_squares(sqrt_M @ (u - self.p_u0))
                    + cp.sum_squares(sqrt_Q @ s))
            self.problems.append((cp.Problem(cp.Minimize(cost), cons), u))

    def _init_previous(self, u_now: np.ndarray | None,
                       alpha_now: np.ndarray | None) -> None:
        # First step: start from the actual thruster state
        if u_now is not None and alpha_now is not None:
            self._u_prev = self._to_extended(np.asarray(u_now, float),
                                             np.asarray(alpha_now, float))
            self._alpha_prev = np.asarray(alpha_now, float).copy()
        else:
            self._u_prev = np.zeros(self.nu)
            self._alpha_prev = np.array([th.alpha0 for th in self.thrusters])

    def _set_rate_cone(self, u0: np.ndarray) -> None:
        eps = self.cfg.eps_rate
        for k in self.cone_axis:
            u0_k = u0[self.idx[k]]
            thrust = np.linalg.norm(u0_k) * self.u_scale[self.idx[k]][0]
            if eps is None or thrust < self.cfg.u_min_dir:
                # All zeros turns the constraint off
                self.cone_axis[k].value = np.zeros(2)
                self.cone_normal[k].value = np.zeros(2)
            else:
                d = u0_k / np.linalg.norm(u0_k)
                self.cone_axis[k].value = eps * d
                self.cone_normal[k].value = np.array([-d[1], d[0]])

    def _solve_subproblems(self) -> tuple[np.ndarray, list]:
        # Previous combination first, so it is solved even if time runs out
        costs = np.full(len(self.problems), np.inf)
        sols = [None] * len(self.problems)
        order = list(range(len(self.problems)))
        if self._combo is not None:
            order.remove(self._combo)
            order.insert(0, self._combo)
        t_start = time.perf_counter()
        for n_tried, i in enumerate(order):
            if n_tried > 0 and time.perf_counter() - t_start > self.cfg.step_time_budget:
                break
            prob, u = self.problems[i]
            try:
                prob.solve(solver=cp.CLARABEL, max_iter=self.cfg.max_iter,
                           time_limit=self.cfg.qp_time_limit)
            except cp.SolverError:
                continue
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u.value is not None:
                costs[i], sols[i] = prob.value, u.value.copy()
        return costs, sols

    def _select_combo(self, costs: np.ndarray) -> int:
        # Hysteresis against chattering between combinations with similar cost
        best = int(np.argmin(costs))
        prev = self._combo
        if (prev is not None and np.isfinite(costs[prev])
                and costs[best] > (1.0 - self.cfg.hysteresis) * costs[prev]):
            return prev
        return best

    def _fallback(self, tau_n: np.ndarray) -> np.ndarray:
        """Pseudo-inverse, saturated and pushed out of the forbidden zones.
        Does not reach tau exactly, but never exceeds the thruster limits."""
        u = self.B_pinv @ tau_n
        for k in range(self.n):
            u_k = u[self.idx[k]]
            if not self.is_az[k]:
                u[self.idx[k]] = np.clip(u_k, -1.0, 1.0)
                continue
            u_k = u_k / max(1.0, np.linalg.norm(u_k) / np.cos(np.pi / self.cfg.n_poly))
            normals = [n for n in self.regions[k] if n is not None]
            if normals and all(n @ u_k < 0.0 for n in normals):
                # Project onto the closer of the two half-disks
                options = [u_k - (n @ u_k) * n for n in normals]
                u_k = min(options, key=lambda o: np.linalg.norm(o - u_k))
            u[self.idx[k]] = u_k
        return u

    def _to_extended(self, u: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        u_ext = np.zeros(self.nu)
        for k, az in enumerate(self.is_az):
            if az:
                u_ext[self.idx[k]] = u[k] * np.array([np.cos(alpha[k]), np.sin(alpha[k])])
            else:
                u_ext[self.idx[k]] = u[k]
        return u_ext / self.u_scale

    def _to_thruster_commands(self, u_ext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        forces = u_ext * self.u_scale
        u_cmd = np.zeros(self.n)
        alpha_cmd = np.zeros(self.n)
        for k, th in enumerate(self.thrusters):
            f = forces[self.idx[k]]
            if not self.is_az[k]:
                u_cmd[k], alpha_cmd[k] = f[0], th.alpha0
            elif np.hypot(*f) >= self.cfg.u_min_dir:
                u_cmd[k], alpha_cmd[k] = np.hypot(*f), np.arctan2(f[1], f[0])
            else:
                # Angle is undefined near zero thrust, so keep the old one
                a = self._alpha_prev[k]
                u_cmd[k], alpha_cmd[k] = f[0] * np.cos(a) + f[1] * np.sin(a), a
        return u_cmd, alpha_cmd
