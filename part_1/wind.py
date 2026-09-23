"""
Wind template

Students should compute generalized BODY-frame wind loads:
    tau_w6 = [Fx, Fy, Fz, Mx, My, Mz]

The simulator uses the 3-DOF subset [Fx, Fy, Mz] = tau_w6 indices [0, 1, 5]
and calls, once per step:

    wind.step(t, dt, eta, nu) -> (tau_w6, info)

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (gust spectra, time variation)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5])
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (RELATIVE wind: compute the loads from V_rw = V_wind - V_vessel,
            using the horizontal components nu[0], nu[1])

Outputs:
    tau_w6 : (6,) BODY loads
    info   : optional dict for logging, e.g.
             {"U": ambient speed, "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad)}
             Return {} (or None) if you do not need it.
             NOTE: "beta_ned" is always the direction the wind blows
             TOWARDS, even when the constructor semantics is "from" —
             convert before logging, do not log the raw constructor value.

Wind coefficient data
---------------------
The vessel wind coefficients C(alpha) = [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] are
provided in `data/wind_coeff.csv` (repository root), tabulated against the relative
wind angle alpha in degrees (0..360). Load them with:

    alpha_deg, C6 = load_wind_coefficients()

The wind loads are then computed as F_wind = U_rw^2 * C(alpha_rw), where U_rw
and alpha_rw are the relative wind speed and angle in the BODY frame.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

_WIND_COEFF_FILE = Path(__file__).resolve().parent.parent / "data" / "wind_coeff.csv"


def load_wind_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the vessel wind coefficient table.

    Returns
    -------
    alpha_deg : (M,) ndarray
        Relative wind angle grid [deg], from 0 to 360.
    C6 : (M, 6) ndarray
        Coefficients [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] at each angle.
    """
    table = np.loadtxt(_WIND_COEFF_FILE, delimiter=",", skiprows=1)
    return table[:, 0], table[:, 1:]

def ned_to_body_xy(v_ned_xy: np.ndarray, psi: float) -> np.ndarray:
    """
    Rotate a 2D vector from NED to BODY using yaw.

    v_body_xy = Rz(psi).T[:2,:2] @ v_ned_xy
    """
    v = np.asarray(v_ned_xy, dtype=float).reshape(2)
    c, s = float(np.cos(psi)), float(np.sin(psi))
    return np.array([ c*v[0] + s*v[1],
                     -s*v[0] + c*v[1]], dtype=float)

class Wind:
    """Template for student wind model.

    Constructor contract — the automated checks (``python check.py``,
    ``pytest``, ``notebooks/part_1_demo.ipynb``) construct your model with
    this signature, so keep it working:

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention —
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly-varying wind speed
        component [m/s] (required in Part 1; 0 disables it).
    tau_slow : time constant of the slow variation [s].
    seed : random seed for the slow component, so runs are reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, seed: int | None = None):
        
        self.mean_speed = float(mean_speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)
        self.seed = seed
       
        np.random.seed(self.seed)   

        #TODO: Review if this is the right way, Now random walk by Ornsten-Ulenbecks method is used
        #Values to calculate the solwly-varying wind speed
        self.theta_slow = 1/tau_slow
        self.U = np.array([mean_speed])
        self.C = load_wind_coefficients()[1]

        #Wind direction vector
        self.Vdir_ned = np.array([np.cos(self.beta), np.sin(self.beta)])
        if self.semantics == "from":
            self.Vdir_ned *= -1
        
    def get_windspeed(self, t, dt):
        #Finds wind speed using Örstein Uhlenbeck process around the mean wind speed
        #Since a specified seed is used the wind speed is technically predermined
        #This allows for a list U of wind speed
        
        slow_curr = self.U[-1]
        mean = slow_curr * np.exp(-self.theta_slow * dt) + self.mean_speed * (1 - np.exp(-self.theta_slow * dt))
        var = self.sigma_slow**2/(2*self.theta_slow) * (1- np.exp(-2*self.theta_slow*dt))
        sd = np.sqrt(var)

        slow_next = np.array([np.random.normal(loc = mean, scale = sd)])
        self.U =np.append(self.U,slow_next)

        return slow_next

    
    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        
        #Deriving Wind in body frame
        U = self.get_windspeed(t, dt)
        V_ned = U*self.Vdir_ned
        V_body = ned_to_body_xy(V_ned, eta[5]) -nu[:2]
        
        U_rs = np.linalg.norm(V_body)
        alpha_rs = np.arctan2(V_body[1],V_body[0])
        if alpha_rs < 0:
            alpha_rs += 2*np.pi
        
        
        #Wind coefficient index
        alpha_rs_deg = (360*alpha_rs/(2*np.pi)) % 360
        alpha_rs_indx = int(np.floor(alpha_rs_deg//10))
        d_alpha_rs_deg = alpha_rs_deg - alpha_rs_indx*10
        C_alpha = d_alpha_rs_deg/10*(self.C[alpha_rs_indx+1]-self.C[alpha_rs_indx]) + self.C[alpha_rs_indx]
        
        tau_w6 = U_rs**2*C_alpha
        info = {"U_ned": U[-1], "U_rs": U_rs , "beta_ned": self.beta, "alpha_body": alpha_rs}
        return tau_w6, info