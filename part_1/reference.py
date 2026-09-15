# """
# Reference template

# Students should filter or shape commanded setpoints before they are sent to
# the controller. The simulator calls, once per step:

#     ref.step(t, dt, eta_cmd) -> (eta_ref, nu_ref, acc_ref)

# All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
# yaw]. The 3-DOF model uses indices [0, 1, 5]; leave the rest zero.

# Inputs:
#     t       : current simulation time [s]
#     dt      : time step [s]
#     eta_cmd : (6,) commanded setpoint
#               (use N_cmd = eta_cmd[0], E_cmd = eta_cmd[1], psi_cmd = eta_cmd[5])

# Outputs (all NED-frame, (6,) each):
#     eta_ref : filtered reference
#               (fill in N_ref = [0], E_ref = [1], psi_ref = [5])
#     nu_ref  : reference velocities
#               (fill in Ndot_ref = [0], Edot_ref = [1], psidot_ref = [5])
#     acc_ref : reference accelerations
#               (fill in Nddot_ref = [0], Eddot_ref = [1], psiddot_ref = [5])

# The simulator forwards all three to the controller, so a smooth reference
# model here directly enables velocity/acceleration feedforward there.
# """
# from typing import Tuple
# import numpy as np

# # Per-axis tuning parameters live with the rest of the Part 1 configuration.
# from part_1.config import RefAxisConfig


# class ReferenceModel:
#     """
#     Template for student reference model.

#     The default implementation is pass-through, so eta_ref = eta_cmd and the
#     reference velocities/accelerations are zero.
#     """

#     def __init__(
#         self,
#         dt: float,
#         cfg_xy: RefAxisConfig | None = None,
#         cfg_psi: RefAxisConfig | None = None,
#     ):
#         self.dt = float(dt)
#         self.cfg_xy = cfg_xy if cfg_xy is not None else RefAxisConfig()
#         self.cfg_psi = cfg_psi if cfg_psi is not None else RefAxisConfig()
#         self.eta_ref = np.zeros(6)
#         self.nu_ref = np.zeros(6)
#         self.acc_ref = np.zeros(6)

#     def reset(self, eta0: np.ndarray) -> None:
#         """Initialize the reference at the vessel's current (6,) state."""
#         self.eta_ref = np.asarray(eta0, dtype=float).reshape(6).copy()
#         self.nu_ref = np.zeros(6)
#         self.acc_ref = np.zeros(6)

#     def step(
#         self, t: float, dt: float, eta_cmd: np.ndarray
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         # TODO: Replace this pass-through placeholder with your reference model.
#         self.eta_ref = np.asarray(eta_cmd, dtype=float).reshape(6).copy()
#         self.nu_ref = np.zeros(6)
#         self.acc_ref = np.zeros(6)
#         return self.eta_ref, self.nu_ref, self.acc_ref



from typing import Tuple
import numpy as np

from part_1.config import RefAxisConfig


class ReferenceModel:
    """
    Fast second-order reference model.

    The translational and yaw references are generated using

        x_ddot = wn^2 * (x_cmd - x_ref) - 2*zeta*wn*x_dot

    which gives a smooth, critically damped response for zeta = 1.

    The yaw error is always wrapped to the shortest angular distance.
    """

    def __init__(
        self,
        dt: float,
        cfg_xy: RefAxisConfig | None = None,
        cfg_psi: RefAxisConfig | None = None,
    ):
        self.dt = float(dt)
        self.cfg_xy = cfg_xy if cfg_xy is not None else RefAxisConfig()
        self.cfg_psi = cfg_psi if cfg_psi is not None else RefAxisConfig()

        self.eta_ref = np.zeros(6, dtype=float)
        self.nu_ref = np.zeros(6, dtype=float)
        self.acc_ref = np.zeros(6, dtype=float)

    def reset(self, eta0: np.ndarray) -> None:
        """Initialize the reference model at the current vessel state."""
        self.eta_ref = np.asarray(eta0, dtype=float).reshape(6).copy()

        # Keep the yaw reference in a numerically convenient range.
        self.eta_ref[5] = np.arctan2(
            np.sin(self.eta_ref[5]),
            np.cos(self.eta_ref[5]),
        )

        self.nu_ref.fill(0.0)
        self.acc_ref.fill(0.0)

    def step(
        self,
        t: float,
        dt: float,
        eta_cmd: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        cmd = np.asarray(eta_cmd, dtype=float).reshape(6)
        dt = float(dt)

        # Guard against invalid integration steps. Returning the current
        # state is preferable to injecting NaNs into the whole simulation.
        if not np.isfinite(dt) or dt <= 0.0:
            self.nu_ref.fill(0.0)
            self.acc_ref.fill(0.0)
            return self.eta_ref.copy(), self.nu_ref.copy(), self.acc_ref.copy()

        # ------------------------------------------------------------------
        # Position reference: N and E
        # ------------------------------------------------------------------
        wn_xy = max(float(self.cfg_xy.wn), 0.0)
        zeta_xy = max(float(self.cfg_xy.zeta), 0.0)

        error_xy = cmd[:2] - self.eta_ref[:2]

        # Second-order reference model:
        #
        #   x_ddot = wn^2 * error - 2*zeta*wn*x_dot
        #
        # This is computationally cheap and directly provides acceleration
        # feed-forward for the controller.
        acc_xy = (
            wn_xy * wn_xy * error_xy
            - 2.0 * zeta_xy * wn_xy * self.nu_ref[:2]
        )

        # Integrate velocity first. This gives a stable semi-implicit Euler
        # update and avoids unnecessary numerical overhead.
        next_vel_xy = self.nu_ref[:2] + acc_xy * dt

        # Optional velocity limit from RefAxisConfig.
        if self.cfg_xy.rate_limit is not None:
            vmax = abs(float(self.cfg_xy.rate_limit))
            next_vel_xy = np.clip(next_vel_xy, -vmax, vmax)

            # Recalculate the actual acceleration corresponding to the
            # velocity-limited reference.
            acc_xy = (next_vel_xy - self.nu_ref[:2]) / dt

        # Integrate position using the current velocity and acceleration.
        next_pos_xy = (
            self.eta_ref[:2]
            + self.nu_ref[:2] * dt
            + 0.5 * acc_xy * dt * dt
        )

        # If the numerical step crosses the target, clamp it and stop.
        # This prevents tiny residual oscillations around the setpoint.
        for i in range(2):
            if (
                (cmd[i] - self.eta_ref[i]) > 0.0
                and next_pos_xy[i] >= cmd[i]
            ) or (
                (cmd[i] - self.eta_ref[i]) < 0.0
                and next_pos_xy[i] <= cmd[i]
            ):
                next_pos_xy[i] = cmd[i]
                next_vel_xy[i] = 0.0
                acc_xy[i] = -self.nu_ref[i] / dt

        # ------------------------------------------------------------------
        # Yaw reference
        # ------------------------------------------------------------------
        wn_psi = max(float(self.cfg_psi.wn), 0.0)
        zeta_psi = max(float(self.cfg_psi.zeta), 0.0)

        # Shortest angular error. This is essential for commands such as
        # 3*pi/2 when the vessel is initially at zero.
        psi_error = np.arctan2(
            np.sin(cmd[5] - self.eta_ref[5]),
            np.cos(cmd[5] - self.eta_ref[5]),
        )

        acc_psi = (
            wn_psi * wn_psi * psi_error
            - 2.0 * zeta_psi * wn_psi * self.nu_ref[5]
        )

        next_vel_psi = self.nu_ref[5] + acc_psi * dt

        if self.cfg_psi.rate_limit is not None:
            vmax_psi = abs(float(self.cfg_psi.rate_limit))
            next_vel_psi = float(
                np.clip(next_vel_psi, -vmax_psi, vmax_psi)
            )
            acc_psi = (next_vel_psi - self.nu_ref[5]) / dt

        # Use the current yaw rate and acceleration to integrate the angle.
        next_psi = (
            self.eta_ref[5]
            + self.nu_ref[5] * dt
            + 0.5 * acc_psi * dt * dt
        )

        # Stop exactly at the target when the numerical integration reaches
        # or crosses it. The target itself is represented in wrapped form.
        psi_target = np.arctan2(
            np.sin(cmd[5]),
            np.cos(cmd[5]),
        )

        psi_remaining = np.arctan2(
            np.sin(psi_target - self.eta_ref[5]),
            np.cos(psi_target - self.eta_ref[5]),
        )

        psi_next_error = np.arctan2(
            np.sin(psi_target - next_psi),
            np.cos(psi_target - next_psi),
        )

        if abs(psi_next_error) > abs(psi_remaining):
            next_psi = psi_target
            next_vel_psi = 0.0
            acc_psi = -self.nu_ref[5] / dt

        # ------------------------------------------------------------------
        # Store state
        # ------------------------------------------------------------------
        self.eta_ref[:2] = next_pos_xy
        self.nu_ref[:2] = next_vel_xy
        self.acc_ref[:2] = acc_xy

        self.eta_ref[5] = next_psi
        self.nu_ref[5] = next_vel_psi
        self.acc_ref[5] = acc_psi

        # All unused 6-DOF components remain zero.
        self.eta_ref[2:5] = 0.0
        self.nu_ref[2:5] = 0.0
        self.acc_ref[2:5] = 0.0

        return (
            self.eta_ref.copy(),
            self.nu_ref.copy(),
            self.acc_ref.copy(),
        )