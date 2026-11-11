# part_2/config.py
# -----------------------------------------------------------------------------
# TMR4240 Marine Control Systems I
# Project – Design of Dynamic Positioning System
#
# Copyright (C) 2026: NTNU, Trondheim
# License: GPL-3.0-or-later
# -----------------------------------------------------------------------------
"""
Project Part 2 configuration — all tunable parameters in one place.

Like ``part_1/config.py``, this file belongs to you: put your own
configuration dataclasses here (observer parameters, wave-model settings,
controller gains, ...) so that every Part 2 simulation can be reconfigured
from this file and ``run_case_part_2.py`` without touching the engine.
"""
from dataclasses import dataclass, field
from math import pi
from typing import Any, Dict, Optional
import numpy as np

from models.thruster_dynamics import ThrusterConfig


# =============================================================================
# WAVE MODEL CONFIGURATION
# =============================================================================
@dataclass
class WaveConfig:
    """Wave model parameters for Part 2 simulations.
    
    Hs : Significant wave height [m]
    Tp : Peak wave period [s]
    beta_deg : Mean wave direction [deg] (convention "from", e.g. 45 = North-East)
    seed : Random seed for reproducibility
    """
    Hs: float = 1.75
    Tp: float = 10.0
    beta_deg: float = 45.0  # North-East by default for Sims 1, 3, 4, 6
    seed: int = 42

    @property
    def beta_rad(self) -> float:
        """Wave direction in radians."""
        return np.deg2rad(self.beta_deg)

    @property
    def omega_p(self) -> float:
        """Peak wave frequency [rad/s]."""
        return 2.0 * pi / self.Tp


# Prescribed sea state configurations
WAVE_DESIGN_SEA = WaveConfig(Hs=1.5, Tp=8.0, beta_deg=45.0, seed=123)   # Sims 1, 3, 4, 5
WAVE_HEAVY_SEA = WaveConfig(Hs=2.5, Tp=10.0, beta_deg=45.0, seed=123)   # Sim 6 (Robustness)


# =============================================================================
# CONTROLLER & OBSERVER CONFIGURATIONS
# =============================================================================
@dataclass
class LQR_Gains:
    """LQR gains for the DP controller"""
    Q: np.ndarray = field(default_factory=lambda: 1e5*np.diag([
        1/(1.0**2), 1/(1.0**2), 1/(np.deg2rad(4)**2),   # position / heading
        1/(1**2), 1/(1**2), 1/(np.deg2rad(5)**2),      # velocities
    ]))
    
    R: np.ndarray = field(default_factory=lambda: 1e5*np.diag([
        1/(60_000**2), 1/(60_000**2), 1/(60_000**2),   # Fx, Fy, Mz/L
    ]))

    Ti: float = 10.0  # integral time


@dataclass
class NPOConfig:
    """Nonlinear passive observer configuration (surge, sway, yaw).

    Tuned against Fossen (2000), Sec. 3.4: h_B(s) h_0(s) must have phase
    within +-90 deg (tests/passivity_test.py), which the rule
    1/Tb << K3/K4 < omega_0 < omega_c ensures.
    """
    # Wave model parameters
    peak_wave_period: float = 8.0  # seconds, gives omega_0 = 2*pi/Tp (design sea Tp)
    wave_damping_ratio: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.1]))  # zeta

    # Notch filter shape (used to build K1 and K2)
    # zeta_n = 2 widens the notch: the Gunnerus WF motion spreads over ~0.6-1.2 rad/s
    notch_damping_ratio: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 2.0]))  # zeta_n
    cutoff_ratio: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 2.0]))  # omega_c / omega_0

    # Velocity and bias injection gains.  K4 = diag(M) * k4
    k4: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.2]))  # [1/s^2]
    k3_factor: float = 0.02  # K3/K4 [rad/s], K3 = k3_factor * K4. Passivity: 1/Tb << K3/K4 < omega_0

    # Slowly varying bias time constants
    Tb: np.ndarray = field(default_factory=lambda: np.diag([300, 300, 300]))


@dataclass
class KalmanConfig:
    Q: np.ndarray = field(default_factory=lambda: np.diag(np.square([
        7, 7, 0.3,             # Noise entering WF pose dynamics
        5_000, 5_000, 8_000,    # Noise entering LF velocity dynamics
        600, 600, 1800           # Noise entering bias dynamics
    ])))

    # Slowly varying bias time constants
    Tb: np.ndarray = field(default_factory=lambda: np.diag([1_800, 1_800, 1_800])) 

    # Initial state covariance. Standard deviations squared on the diagonal.
    P0: np.ndarray = field(default_factory=lambda: np.diag(np.square([
        0.5, 0.5,                       # WF North
        0.5, 0.5,                       # WF East
        np.deg2rad(3.0),                # WF heading
        np.deg2rad(2.5),                # WF heading rate
        1.5, 1.5, np.deg2rad(4.0),      # LF pose
        0.25, 0.25, np.deg2rad(1.5),    # Body velocity
        1000, 1000, 3000                # Environmental bias
    ])))

    wave_period_surge: float = 8.0
    wave_damping_surge: float = 0.20

    wave_period_sway: float = 8.0
    wave_damping_sway: float = 0.20

    wave_period_yaw: float = 7.2
    wave_damping_yaw: float = 0.20


@dataclass
class RefAxisConfig:
    """Reference-model configuration for one axis (see part_2/reference.py).

    Tune these per simulation and justify the values in the report.

    Note: the automated checks build the default ``ReferenceModel(dt)``, which
    uses the field defaults below — so keep your final tuned values as the
    defaults here (overriding them only in ``run_case_part_2.py`` will not
    reach the checks).
    """
    wn: float = 0.08                     # natural frequency [rad/s] (placeholder)
    zeta: float = 1.0                   # damping ratio [-] (placeholder)
    rate_limit: Optional[float] = None  # max |x_dot| (m/s or rad/s); None = off


# Part 2: tuned values ported from part_1/config.py, used by the default
# thrust limits
REF_CFG_XY = RefAxisConfig(
    wn=0.08,
    zeta=1.0,
    rate_limit=1.0                      # Max velocity in m/s
)

REF_CFG_PSI = RefAxisConfig(
    wn=0.08,
    zeta=1.0,
    rate_limit=np.deg2rad(3)          # Max angular velocity
)


# =============================================================================
# THRUST ALLOCATION CONFIGURATION
# =============================================================================
@dataclass
class AllocationConfig:
    """
    The weights act on normalized extended thrust (divided by u_max), ordered
    [f_bow, X_az1, Y_az1, X_az2, Y_az2], and on the normalized wrench error.

    Same fields as the Part 1 AllocationConfig, changes are marked "Part 2".
    The checks build ThrustAllocator(thrusters), so keep the final values as
    the defaults here. See part_2/thrust_allocation.md.
    """
    H: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 1.0, 1.0, 1.0]))  # thrust use
    M: np.ndarray = field(default_factory=lambda: np.diag([0.1, 0.1, 0.1, 0.1, 0.1]))  # change from last step
    # Part 2: 1e4 -> 1e6, the slack has to stay small next to the lambda term
    Q: np.ndarray = field(default_factory=lambda: np.diag([1e6, 1e6, 1e6]))  # wrench error (slack)

    # Part 2: weight on (1 - lambda). lambda < 1 scales an infeasible demand
    # down along its own direction. Has to be larger than the thrust cost a
    # smaller wrench saves (< 10 with these H, M) and much smaller than Q
    q_lambda: float = 100.0
    # Part 2: separate scale for the yaw moment, weighted higher, so heading
    # is kept first and the force (X, Y) keeps its direction. False = one
    # scale for the whole wrench
    yaw_priority: bool = False
    q_lambda_yaw: float = 1000.0

    # Sector around the direction where an azimuth's jet hits the other one.
    # Set to 0 to deactivate
    forbidden_half_width: float = np.deg2rad(20.0)
    n_poly: int = 16  # polygon sides approximating |u| <= u_max
    hysteresis: float = 0.10  # only switch sector combination if cost drops by 10 %

    u_min_dir: float = 500.0  # [N] hold the azimuth angle below this thrust

    # Part 2: an azimuth gives the same force with (u, alpha) and (-u, alpha + pi).
    # Reverse thrust is used if it saves more than reverse_hysteresis of rotation
    allow_reverse_thrust: bool = True
    reverse_hysteresis: float = np.deg2rad(10.0)

    # Part 2: replaces the Part 1 eps_rate cone. Second QP so that the command
    # is reachable in one step (thrust rate, azimuth rotation speed).
    # Optional extension, False = steady-state allocation only
    rate_aware: bool = True

    # Solver limits, if no QP succeeds the scaled pseudo-inverse is used
    max_iter: int = 50  # solver iterations per QP
    # Part 2: wall-clock limits make the results depend on the computer, so
    # they are off (None) by default. Part 1 used 0.02 s and 0.05 s
    qp_time_limit: Optional[float] = None  # [s] per QP
    step_time_budget: Optional[float] = None  # [s] for all QPs in one step
    use_fallback_only: bool = False  # skip the QPs, only use the pseudo-inverse (for comparison)


# =============================================================================
# SIMULATION ENGINE CONFIGURATION
# =============================================================================
@dataclass
class Part2SimConfig:
    """Part 2-only clock and switches; it does not inherit Part 1 settings.

    ``use_sensor_noise`` switches measurement noise on for the optional
    sensor-noise part of the project.  The noise levels are fixed course
    parameters in ``models/sensors.py`` — the switch is the only sensor
    setting you select per simulation.  The mandatory simulations run with
    ideal measurements (``False``, the default).

    ``use_controller=False`` switches the DP system off: the controller is
    not called, the desired wrench is zero and the vessel drifts freely under
    the environmental loads (Simulation 1).  ``use_observer`` still works in
    that mode, so an observer can be watched on a drifting vessel.

    ``dt`` must be 0.1 or 0.01 s (the engine rejects other steps; the Part 1
    default of 0.05 s does not carry over).  ``observer_kwargs`` are passed to
    ``select_observer(observer_type, **observer_kwargs)`` when the engine
    builds the observer, so a differently tuned observer can be selected from
    the configuration (e.g. ``{"cfg": MyObserverConfig(...)}``).
    """
    dt: float = 0.1
    T: float = 1000.0
    method: str = "Euler"
    use_controller: bool = True
    use_reference: bool = True
    use_observer: bool = True
    # Which observer the engine builds when none is injected.  This is only a
    # plumbing default so the field is never empty, NOT a recommendation:
    # every preset that needs an observer names the type explicitly, and which
    # one you adopt is your Part 2, Simulation 4 decision.
    observer_type: str = "nonlinear_passive"
    use_sensor_noise: bool = False
    thruster_dynamics: bool = True
    bypass_actuators: bool = False
    observer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Wave configuration integration
    wave_config: WaveConfig = field(default_factory=lambda: WAVE_DESIGN_SEA)


def default_thrusters_part_2() -> list[ThrusterConfig]:
    """Same Gunnerus thruster fit as Project Part 1.

    Part 2 extends Part 1: the vessel and its three thrusters are unchanged.
    What is new in Part 2 is that the thruster *constraints* (max thrust,
    ramp rate ``u_rate`` and azimuth rotation speed) are enforced by the
    thruster dynamics and must be respected by your thrust allocation.
    The azimuth rotation limit is 2 rpm = 0.20944 rad/s.  Angles are in the
    BODY frame, where the fixed tunnel thruster points at ``+pi/2``.
    """
    rotation_speed = 2.0 * 2.0 * pi / 60.0
    return [
        ThrusterConfig("Tunnel_Bow", "tunnel", x=12.0, y=0.0,
                       u_max=32_000.0, u_rate=4_000.0,
                       rot_speed=0.0, alpha0=pi / 2.0),
        ThrusterConfig("Azimuth_1", "azimuth", x=-13.0, y=3.0,
                       u_max=80_000.0, u_rate=10_000.0,
                       rot_speed=rotation_speed, alpha0=0.0),
        ThrusterConfig("Azimuth_2", "azimuth", x=-13.0, y=-3.0,
                       u_max=80_000.0, u_rate=10_000.0,
                       rot_speed=rotation_speed, alpha0=0.0),
    ]
