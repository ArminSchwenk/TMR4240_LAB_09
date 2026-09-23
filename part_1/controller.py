"""
Controller template

Students should implement a controller that maps the vessel state and the
full reference to a body-frame wrench. The simulator calls, once per step:

    controller.compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref) -> tau_d

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; the remaining components are
zero on input and ignored on output.

Inputs (full loop state and full reference):
    t       : current simulation time [s]
    dt      : time step [s]
    eta     : (6,) vessel NED state [N, E, z, phi, theta, psi]
              (use N = eta[0], E = eta[1], psi = eta[5])
    nu      : (6,) vessel BODY velocities [u, v, w, p, q, r]
              (use u = nu[0], v = nu[1], r = nu[5])
    eta_ref : (6,) NED reference state
              (use N_d = eta_ref[0], E_d = eta_ref[1], psi_d = eta_ref[5])
    nu_ref  : (6,) NED-frame reference velocities
              (use Ndot_d = nu_ref[0], Edot_d = nu_ref[1], psidot_d = nu_ref[5])
    acc_ref : (6,) NED-frame reference accelerations, same layout as nu_ref
              (use for model-based / inertia feedforward)

Output:
    tau_d   : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz] (N, Nm)
              (fill in Fx = tau_d[0], Fy = tau_d[1], Mz = tau_d[5];
               leave the other components zero)

Optional hooks the simulator will use IF you define them (safe to omit):
    reset()                                  — called before each run
    apply_external_aw(tau_applied, psi, dt)  — anti-windup with the (6,)
                                               wrench actually applied after
                                               allocation and the actuator
                                               model (ideal in Part 1)
    last_pid_body  : {"P","I","D"} -> (6,) BODY components   (logged)
    int_ned (2,), int_psi (float)            — integrator states (logged)

Constructor contract — the automated checks (``python check.py``, ``pytest``,
``notebooks/part_1_demo.ipynb``) construct your controller as
``DPController()`` with NO arguments, so your final tuned gains must be the
constructor defaults. Tuning only inside ``run_case_part1.py`` will pass your
own runs but fail the checks.
"""
import numpy as np
import scipy as sp
from simulation.utils import Rz, wrap_angle_pi
from part_1.config import LQR_Gains

DOF3 = np.array([0, 1, 5])


class DPController:
    """
    Error-state LQR dynamic positioning controller with integral and
    back-calculation anti-windup, combined with reference feedforward.
    """

    def __init__(self, *args, **kwargs):
        self._init_model()
        self._init_scaling()
        self._init_controller()
        self.reset()

    def compute(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        nu_ref: np.ndarray | None = None,
        acc_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the commanded 6-DOF body wrench sent to the thrust allocation."""
        
        dot_eta_ref = nu_ref if nu_ref is not None else np.zeros(6)
        ddot_eta_ref = acc_ref if acc_ref is not None else np.zeros(6)

        R = Rz(eta[5])  # BODY -> NED rotation matrix

        e_eta_ned, e_eta_body, e_nu_body = self.compute_errors(
            eta, nu, eta_ref, dot_eta_ref, R
        )

        integral_body = self._update_integrator(
            e_eta_ned, R, dt
        )

        x = np.concatenate([
            e_eta_body,
            e_nu_body,
            integral_body,
        ])

        tau_feedback = -self.K @ x
        tau_ff_ref = self._compute_feedforward(
            eta_ref, dot_eta_ref, ddot_eta_ref
        )

        tau_d = np.zeros(6)
        tau_d[DOF3] = tau_feedback + tau_ff_ref

        self._last_tau_d3 = tau_d[DOF3].copy()

        return tau_d

    def reset(self) -> None:
        """Reset controller state before a new simulation run."""
        self.int_ned = np.zeros(2)
        self.int_psi = 0.0
        self._last_tau_d3 = None

    def apply_external_aw(
        self,
        tau_applied: np.ndarray,
        psi: float,
        dt: float,
    ) -> None:
        """Apply back-calculation anti-windup from the applied body wrench."""
        if self._last_tau_d3 is None or self.aw_gain == 0.0 or dt <= 0.0:
            return

        tau_applied3 = tau_applied[DOF3]
        wrench_error_body = tau_applied3 - self._last_tau_d3
        int_correction_body = self._aw_body_map @ wrench_error_body
        int_correction_ned = Rz(psi) @ int_correction_body

        scale = self.aw_gain * dt
        self.int_ned += scale * int_correction_ned[:2]
        self.int_psi += scale * int_correction_ned[2]

    def compute_errors(
        self,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        dot_eta_ref: np.ndarray,
        R: np.ndarray
    ):
        """Compute position and velocity errors in NED and body frames."""
        eta3 = np.asarray(eta)[DOF3]
        eta_ref3 = np.asarray(eta_ref)[DOF3]

        nu_body = np.asarray(nu)[DOF3]
        dot_eta_ref_ned = np.asarray(dot_eta_ref)[DOF3]

        # Position / heading error in NED
        e_eta_ned = eta3 - eta_ref3
        e_eta_ned[2] = wrap_angle_pi(e_eta_ned[2])

        # Transform position error to BODY
        e_eta_body = R.T @ e_eta_ned

        # Desired NED velocity -> BODY
        nu_ref_body = R.T @ dot_eta_ref_ned

        # Velocity error in BODY
        e_nu_body = nu_body - nu_ref_body

        return e_eta_ned, e_eta_body, e_nu_body
       
    def _update_integrator(
        self,
        e_eta_ned: np.ndarray,
        R: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Update the integral of the position error in NED and return it in BODY frame."""
        self.int_ned += e_eta_ned[:2] * dt
        self.int_psi += e_eta_ned[2] * dt

        integral_ned = np.array([
            self.int_ned[0],
            self.int_ned[1],
            self.int_psi,
        ])

        return R.T @ integral_ned

    def _compute_feedforward(
        self,
        eta_ref: np.ndarray,
        dot_eta_ref: np.ndarray,
        ddot_eta_ref: np.ndarray,
    ) -> np.ndarray:
        """Compute model-based feedforward from the reference trajectory."""
        nu_ref_body, dot_nu_ref_body = self._compute_reference_kinematics(
            eta_ref,
            dot_eta_ref,
            ddot_eta_ref,
        )

        C_ref = self._coriolis_matrix(nu_ref_body)

        return (
            self.M3 @ dot_nu_ref_body
            + C_ref @ nu_ref_body
            + self.D3 @ nu_ref_body
        )

    def _compute_reference_kinematics(
        self,
        eta_ref: np.ndarray,
        dot_eta_ref: np.ndarray,
        ddot_eta_ref: np.ndarray,
    ):
        """Compute reference velocity and acceleration in the body frame."""
        eta_ref3 = np.asarray(eta_ref)[DOF3]
        dot_eta_ref3 = np.asarray(dot_eta_ref)[DOF3]
        ddot_eta_ref3 = np.asarray(ddot_eta_ref)[DOF3]

        psi_ref = eta_ref3[2]

        # BODY -> NED for the reference orientation
        R_ref = Rz(psi_ref)

        # Desired BODY velocity:
        nu_ref_body = R_ref.T @ dot_eta_ref3

        # Desired yaw rate
        r_ref = nu_ref_body[2]

        # Skew symmetric matrix
        S_r = np.array([
            [0.0,   -r_ref, 0.0],
            [r_ref,  0.0,   0.0],
            [0.0,    0.0,   0.0],
        ])

        # Desired BODY velocity derivative:
        # dot(nu_d)^b = R_d^T ddot(eta_d)^n - S(r_d) nu_d^b
        dot_nu_ref_body = R_ref.T @ ddot_eta_ref3 - S_r @ nu_ref_body

        return nu_ref_body, dot_nu_ref_body

    def _coriolis_matrix(self, nu):
        u, v, r = nu

        m11 = self.M3[0, 0]
        m22 = self.M3[1, 1]
        m23 = 0.5*(self.M3[1, 2] + self.M3[2, 1])

        C = np.array([
            [0.0, 0.0, -(m22 * v + m23 * r)],
            [0.0, 0.0,   m11 * u],
            [m22 * v + m23 * r, -m11 * u, 0.0]
        ])
        return C

    def _init_model(self):
        self.M3 = np.array([
            [6.007e5, 0.0, 0.0],
            [0.0, 7.067e5, -4.733e5],
            [0.0, -5.712e5, 5.456e7],
        ])

        self.D3 = np.diag([1117.6, 2.229e4, 1.95e6])

        Z = np.zeros((3, 3))
        I = np.eye(3)

        M3_inv = np.linalg.solve(self.M3, I)
        M3_inv_D3 = np.linalg.solve(self.M3, self.D3)

        self.A_raw = np.block([
            [Z, I, Z],
            [Z, -M3_inv_D3, Z],
            [I, Z, Z]
        ])

        self.B_raw = np.vstack([
            Z,
            M3_inv,
            Z
        ])

    def _init_scaling(self):
        self.L = 30 # meters, approximate length of the vessel

        T_3 = np.diag([1.0, 1.0, self.L])

        self.T_x = sp.linalg.block_diag(T_3, T_3, T_3)
        self.T_u = np.diag([1.0, 1.0, 1.0 / self.L])

        self.A_s = self.T_x @ self.A_raw @ np.linalg.inv(self.T_x)
        self.B_s = self.T_x @ self.B_raw @ np.linalg.inv(self.T_u)

    def _init_controller(self):
        self.gains = LQR_Gains()

        P = sp.linalg.solve_continuous_are(
            self.A_s,
            self.B_s,
            self.gains.Q_s,
            self.gains.R_s,
        )

        K_s = np.linalg.solve(
            self.gains.R_s,
            self.B_s.T @ P,
        )

        self.K = np.linalg.solve(self.T_u, K_s @ self.T_x)
    
        K_I = self.K[:, 6:9]
        if np.linalg.matrix_rank(K_I) < 3:
            raise ValueError("Integral gain matrix is singular")

        self.aw_gain = 1.0

        self._aw_body_map = np.linalg.solve(
            -K_I,
            np.eye(3),
        )
