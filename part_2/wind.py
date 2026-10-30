"""
Wind template (Project Part 2)

Part 2 extends the Part 1 wind model.  In addition to the mean and slowly
varying wind speed, the wind field must now contain

* a fluctuating **gust** component of the wind speed, generated from a wind
  spectrum of your choice (Harris, Davenport, NORSOK, ...), and
* a **slowly varying wind direction**, without a gust component, bounded to
  at most +/- 5 degrees around the mean direction.

You built such a wind generator in Assignment 2 -- reuse it.  The coefficient
lookup (relative wind -> loads) is unchanged from Part 1; only the ambient
wind field (speed and direction as functions of time) becomes richer.

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
             {"U": ambient speed (total), "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad),
              "U_mean": mean + slowly varying speed, "U_gust": gust speed}
             The Part 2 checks read "U_mean", "U_gust" and "beta_ned" to
             verify your wind field (gust present, direction within limit),
             so fill them in.
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

Nomenclature
----------------------
U: wind speed magnitude
V: wind speed direction vector
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

def wrap_angle(ang):
     return (ang + np.pi) % (2*np.pi) - np.pi

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
    """Template for the Part 2 wind model.

    Constructor contract -- the automated checks (``python check.py``,
    ``pytest``) and the Part 2 runner construct your model with this
    signature, so keep it working.  It is a superset of the Part 1 contract:
    your Part 1 model can be pasted in and extended.

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., tau_slow=...,
             gust=..., gust_params=..., sigma_dir=..., tau_dir=...,
             dir_limit=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : mean direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention --
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly varying wind speed
        component [m/s] (Part 1; 0 disables it).
    tau_slow : time constant of the slow speed variation [s].
    gust : switch the fluctuating (gust) speed component on.  The gust must
        be zero-mean and follow a wind spectrum; the spectrum and its
        parameters are your choice (see Assignment 2).  ``False`` gives the
        Part 1 wind field, which the checks use to isolate the gust.
    gust_params : dict with the parameters of *your* gust model (spectrum
        type, elevation, turbulence intensity, number of harmonics, ...).
        ``None`` means your own defaults.
    sigma_dir : standard deviation of the slowly varying wind direction
        [rad] (0 disables it).  The direction has NO gust component.
    tau_dir : time constant of the slow direction variation [s].
    dir_limit : hard bound on the direction excursion |beta(t) - beta| [rad];
        the project requires at most 5 degrees.
    seed : random seed for all stochastic components, so runs are
        reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, gust: bool = False,
                 gust_params: Dict | None = None, sigma_dir: float = 0.0,
                 tau_dir: float = 300.0, dir_limit: float = np.deg2rad(5.0),
                 seed: int | None = None):
        self.mean_speed = float(mean_speed)
        self.seed = seed
        self.semantics = semantics
        np.random.seed(self.seed)
        self.memory_length = 2
        self.C = load_wind_coefficients()[1]

        #Values for direction
        self.beta = wrap_angle(float(beta)) 
        if semantics.strip().lower() == 'from':
            self.beta += np.pi
            self.beta = wrap_angle(self.beta)

        self.dir_limit = float(dir_limit)
        self.sigma_dir = float(sigma_dir)
        self.tau_dir = float(tau_dir)
        self.theta_dir = 1/tau_dir

        self.Var_dir = np.zeros(self.memory_length)
        
        #Values to calculate the solwly-varying wind speed
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)
        self.theta_slow = 1/tau_slow

        self.Var_U = np.zeros(self.memory_length)

        #Values for wind gust
        # TODO: Decide default f_max, f_min and df
        self.gust = bool(gust)
        self.gust_params = dict(gust_params or {'L' : float(1800), 'kappa' : float(0.0026),'f_min' : float(0.1)  , 'f_max' : float(20), 'nf' : int(20)})

        self.f_gust = np.linspace(self.gust_params['f_min'], self.gust_params['f_max'], self.gust_params['nf'])
        #self.f_gust = np.arange(0,self.gust_params['f_max'],self.gust_params['df'])
        self.df_gust = (self.gust_params['f_max']- self.gust_params['f_min'])/self.gust_params['nf']
        self.Hf = np.zeros_like(self.f_gust)
        self.phi = np.array([0.02*np.pi*np.random.randint(0,100) for i in self.f_gust])
        for j,i in enumerate(self.f_gust):
            f_bar = (self.gust_params['L']*i)/ self.mean_speed
            Sf = (4*self.gust_params['kappa']*self.gust_params['L']*self.mean_speed)/(2+f_bar**2)**(5/6)
            Sf = np.sqrt(2*Sf*self.df_gust)
            self.Hf[j] = Sf
        
    def get_wind_gust(self, t):
        G = sum(self.Hf * np.cos(2*np.pi*self.f_gust*t + self.phi))
        return G

    def get_slow_var_windspeed(self, dt):
        U_curr = self.Var_U[-1]
        
        a = np.exp(-dt/self.tau_slow)
        zeta = np.random.normal(loc = 0, scale = 1)
        U_next = a*U_curr + self.sigma_slow*np.sqrt(1-a**2)* zeta 

        U_next += self.mean_speed
        if U_next < 0:  #Stopping negative wind
            self.Var_U = np.append(self.Var_U, 0)
        else:
            self.Var_U = np.append(self.Var_U, U_next - self.mean_speed)[-self.memory_length:]
        
        return U_next #- self.mean_speed

    def get_wind_direction(self, dt):
        dir_curr = self.Var_dir[-1]
        
        a = np.exp(-dt/self.tau_dir)
        zeta = np.random.normal(loc = 0, scale = 1)
        dir_next = a*dir_curr + self.sigma_dir*np.sqrt(1-a**2)* zeta 
        
        dir_next = min(dir_next, + self.dir_limit) #Check upper direction limit
        dir_next = max(dir_next, - self.dir_limit) #Check lower direction limit
        
        self.Var_dir = np.append(self.Var_dir,dir_next)[-self.memory_length:]
       
        return wrap_angle(dir_next + self.beta)


    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:       

        if self.gust:
            U_gust = self.get_wind_gust(t)
        else:
            U_gust = 0
        U_var = self.get_slow_var_windspeed(dt)
        U_abs = max(self.mean_speed + U_gust + U_var, 0)

        dir_curr = self.get_wind_direction(dt)    
        V_ned = np.array([np.cos(dir_curr), np.sin(dir_curr)])
        U_ned = U_abs * V_ned
        U_body = ned_to_body_xy(U_ned, eta[5]) -nu[:2]

        U_rw = np.linalg.norm(U_body)
        alpha_rw = np.arctan2(U_body[1],U_body[0])
        if alpha_rw < 0:
            alpha_rw += 2*np.pi

        alpha_rw_deg = (360*alpha_rw/(2*np.pi)) % 360
        alpha_rw_indx = int(np.floor(alpha_rw_deg//10))
        d_alpha_rw_deg = alpha_rw_deg - alpha_rw_indx*10
        C_alpha = d_alpha_rw_deg/10*(self.C[alpha_rw_indx+1]-self.C[alpha_rw_indx]) + self.C[alpha_rw_indx]

        if self.semantics.strip().lower() == 'from':
            dir4info = dir_curr + np.pi
            dir4info = wrap_angle(dir4info)

        tau_w6 = U_rw**2*C_alpha
        info = {"U": U_abs, "beta_ned": self.beta, "alpha_body": alpha_rw,
                "U_mean": self.mean_speed + U_var, "U_gust": U_gust}
        return tau_w6, info, U_abs, U_gust+self.mean_speed, U_var+self.mean_speed, dir4info, dir_curr
