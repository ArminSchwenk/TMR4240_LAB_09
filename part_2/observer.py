"""Student observer templates and their common simulator interface.

The simulator calls, once per step (pose measurement only, no velocity):

    observer.step(t, dt, eta_measured, tau_est) -> ObserverEstimate(eta, nu, bias)

    eta_measured : (6,) measured NED pose [N, E, z, phi, theta, psi]
    tau_est      : (6,) desired controller wrench (before thruster dynamics)
                   of the PREVIOUS step (zero at the first step)
    eta, nu      : (6,) low-frequency NED pose and BODY velocity estimates
    bias         : (6,) slowly varying bias estimate (NED force) — may be zeros

A plain ``(eta, nu, bias)`` tuple is accepted as well.  Before every run the
engine calls ``reset(eta0)`` with the TRUE initial pose, so an observer can
start from the vessel's actual position.  Select an implementation with
``Part2SimConfig(observer_type=...)`` (Simulations 4-7 use the
``SELECTED_OBSERVER`` constant of ``run_case_part_2.py``); extra constructor
arguments go through ``Part2SimConfig(observer_kwargs=...)``.  Both classes
intentionally start as pass-through placeholders; students must implement and
tune them, keeping the tuned parameters as constructor defaults because the
checks call ``select_observer(kind)`` without arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from simulation.utils import Rz, wrap_angle_pi
from part_2.config import KalmanConfig

@dataclass
class ObserverEstimate:
    eta: np.ndarray
    nu: np.ndarray
    bias: np.ndarray


class Observer:
    """Interface shared by all Part 2 observers."""

    name = "observer" 

    def reset(self, eta0: np.ndarray | None = None):
        pass

    def step(
        self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray
    ) -> ObserverEstimate:
        raise NotImplementedError

    @staticmethod
    def _placeholder(eta_measured: np.ndarray) -> ObserverEstimate:
        return ObserverEstimate(
            eta=np.asarray(eta_measured, dtype=float).reshape(6).copy(),
            nu=np.zeros(6),
            bias=np.zeros(6),
        )


class NonlinearPassiveObserver(Observer):
    """Template for a nonlinear passive wave-filtering observer.

    Suggested states are wave-frequency motion, low-frequency pose, BODY
    velocity, and slowly-varying bias.  Implement the correction and model
    propagation in :meth:`step`.
    """

    name = "nonlinear_passive"

    def __init__(self, *args, **kwargs):
        pass

    def step(self, t, dt, eta_measured, tau_est):
        # TODO: implement the nonlinear passive observer.
        return self._placeholder(eta_measured)


class KalmanFilterObserver(Observer):
    """Template for a discrete KF/EKF observer.

    Students define the process/measurement models and tune Q, R and P0.
    The input ``tau_est`` is the desired controller wrench before thruster
    dynamics.
    """

    name = "kalman"
    ACTIVE_DOF = np.array([0, 1, 5])

    def __init__(self, *args, **kwargs):
        cfg = KalmanConfig()

        # _____Model parameters_____
        self.M = np.array([
            [6.007e5, 0.0, 0.0],
            [0.0, 7.067e5, -4.733e5],
            [0.0, -5.712e5, 5.456e7],
        ])
        self.Minv = np.linalg.solve(self.M, np.eye(3))

        self.D = np.diag([1117.6, 2.229e4, 1.95e6])

        self.Tb = cfg.Tb

        # _____Wave model_____
        self.omega_wave = 2.0 * np.pi / cfg.peak_wave_period
        self.zeta_wave = cfg.wave_damping_ratio

        A_wave_1DOF = np.array([
            [0.0, 1.0],
            [-self.omega_wave**2, -2.0 * self.zeta_wave * self.omega_wave],
        ])
        self.Aw = np.kron(np.eye(3), A_wave_1DOF)

        # _____Covariance______
        self.Q = cfg.Q

        self.R = np.diag([
            0.20**2,
            0.20**2,
            np.deg2rad(0.20)**2,
        ])

        # _____Input matrix______
        self.B = np.zeros((15, 3))
        self.B[9:12, :] = self.Minv

        # _____Measurement matrix______
        # y = eta_LF + Cw @ xi
        Cw = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ])
        self.C = np.zeros((3, 15))
        self.C[:, :6] = Cw
        self.C[:, 6:9] = np.eye(3)

        # ____Noise input matrix____
        Ew = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        self.E = np.zeros((15, 6))
        self.E[:6, :3] = Ew
        self.E[12:15, 3:6] = np.eye(3)

        # ____Initial state and covariance____
        self.x_hat = np.zeros(15)
        self.P0 = cfg.P0
        self.P = self.P0.copy()

    def reset(self, eta0=None):
        self.x_hat = np.zeros(15)

        if eta0 is not None:
            self.x_hat[6:9] = eta0[self.ACTIVE_DOF]
            self.x_hat[8] = wrap_angle_pi(self.x_hat[8])

        self.P = self.P0.copy()
        self._first_step = True

    def step(self, t, dt, eta_measured, tau_est):

        y = eta_measured[self.ACTIVE_DOF]
        tau = tau_est[self.ACTIVE_DOF]

        A = self._continuous_jacobian(self.x_hat)
        Ad = np.eye(15) + A * dt
        Cd = self.C
        Ed = self.E * dt

        # ________________Prediction___________________
        if self._first_step:
            self._first_step = False
            x_pred = self.x_hat
            P_pred = self.P
        else:    
            x_pred = self.x_hat + dt * self._state_derivative(self.x_hat, tau)
            x_pred[8] = wrap_angle_pi(x_pred[8])  # wrap heading to (-pi, pi]
            P_pred = Ad @ self.P @ Ad.T + Ed @ self.Q @ Ed.T

        # ________________Correction___________________
        K = P_pred @ Cd.T @ np.linalg.inv(Cd @ P_pred @ Cd.T + self.R)

        innovation = y - Cd @ x_pred
        innovation[2] = wrap_angle_pi(innovation[2])
        self.x_hat = x_pred + K @ innovation
        self.x_hat[8] = wrap_angle_pi(self.x_hat[8])

        self.P = (np.eye(15) - K @ Cd) @ P_pred @ (np.eye(15) - K @ Cd).T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        
        return self._make_estimate()

    def _state_derivative(self, x, tau):
        xi = x[:6]
        eta = x[6:9]
        nu = x[9:12]
        bias = x[12:15]

        J = Rz(eta[2])

        dx = np.zeros(15)

        # Wave-frequency oscillator states
        dx[:6] = self.Aw @ xi

        # LF pose kinematics
        dx[6:9] = J @ nu

        # BODY velocity dynamics
        dx[9:12] = self.Minv @ (-self.D @ nu + J.T @ bias + tau)

        # Slowly varying bias
        dx[12:15] = -np.linalg.solve(self.Tb, bias)
 
        return dx
    
    def _continuous_jacobian(self, x):
        
        eta = x[6:9]
        nu = x[9:12]
        bias = x[12:15]

        psi = eta[2]
        u, v = nu[:2]
        
        J = Rz(psi)

        J_psi = np.array([
            [-np.sin(psi), -np.cos(psi), 0.0],
            [ np.cos(psi), -np.sin(psi), 0.0],
            [    0.0,           0.0,     0.0],
        ])

        jacobian = np.zeros((15, 15))
        # notation: F_i_j = ∂ dot_i / ∂j

        # ∂ dot_ξ / ∂ξ
        jacobian[:6, :6] = self.Aw
        
        # ∂ dot_η / ∂η
        F_eta_eta = np.array([
            [0, 0, -u * np.sin(psi) - v * np.cos(psi)],
            [0, 0,  u * np.cos(psi) - v * np.sin(psi)],
            [0, 0,                  0                ],  
        ])
        jacobian[6:9, 6:9] = F_eta_eta

        # ∂ dot_η / ∂ν
        jacobian[6:9, 9:12] = J

        # ∂ dot_ν / ∂ν
        jacobian[9:12, 9:12] = -self.Minv @ self.D

        # ∂ dot_ν / ∂η
        F_nu_eta = np.zeros((3, 3))
        F_nu_eta[:, 2] = self.Minv @ J_psi.T @ bias
        jacobian[9:12, 6:9] = F_nu_eta

        # ∂ dot_ν / ∂b
        jacobian[9:12, 12:15] = self.Minv @ J.T

        # ∂ dot_b / ∂b
        jacobian[12:15, 12:15] = -np.linalg.solve(self.Tb, np.eye(3))

        return jacobian

    def _make_estimate(self):
        """Convert internal 15D state to the required 6D outputs."""

        eta = np.zeros(6)
        nu = np.zeros(6)
        bias = np.zeros(6)

        eta[self.ACTIVE_DOF] = self.x_hat[6:9]
        nu[self.ACTIVE_DOF] = self.x_hat[9:12]
        bias[self.ACTIVE_DOF] = self.x_hat[12:15]

        eta[5] = wrap_angle_pi(eta[5])

        return ObserverEstimate(
            eta=eta,
            nu=nu,
            bias=bias,
        )

def select_observer(kind: str, **kwargs) -> Observer:
    """Construct one observer without changing simulator code."""
    choices = {
        "nonlinear_passive": NonlinearPassiveObserver,
        "kalman": KalmanFilterObserver,
    }
    try:
        return choices[kind](**kwargs)
    except KeyError as exc:
        raise ValueError(f"observer_type must be one of {tuple(choices)}") from exc


