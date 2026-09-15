"""
Thrust Allocation template

Students should implement an algorithm that maps the desired body-frame
wrench to individual thruster commands. The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state — use what your algorithm needs):
    t         : current simulation time [s]
    dt        : time step [s]              (rate-aware/dynamic allocation)
    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
                 = [Fx, Fy, Mz]; the other components are zero)
    u_now     : current actual thrusts [N]     (rate-aware allocation)
    alpha_now : current thruster angles [rad]  (minimize azimuth slewing)

Outputs:
    u_cmd     : signed thrust command for each thruster [N]
    alpha_cmd : thruster angle command for each thruster [rad]

Students may implement, for example:
    - pseudo-inverse allocation,
    - weighted least-squares allocation,
    - optimization-based allocation,
    - power-minimizing allocation.
"""
from typing import List, Optional, Tuple
import numpy as np
import cvxpy as cp
from models.thruster_dynamics import ThrusterConfig

class ThrustAllocator:
    """MIQP Thrust Allocator based on Johansen (2008)."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters
        self.n_thrusters = len(thrusters)
        
        # 1. Build the Configuration Matrix (B) dynamically
        B_list = []
        for t in self.thrusters:
            # Assuming ThrusterConfig has attributes: x, y, and a way to check if rotatable
            # Adjust these attribute names to match your simulator's exact ThrusterConfig class
            is_azimuth = getattr(t, 'is_steerable', True) 
            
            if is_azimuth:
                # B_r for rotatable azimuth thrusters
                B_i = np.array([
                    [1, 0],
                    [0, 1],
                    [-t.y, t.x]
                ])
                self.num_extended_vars += 2
            else:
                # B_f for fixed tunnel thrusters (assuming purely sway/y-axis force)
                B_i = np.array([
                    [0],
                    [1],
                    [t.x]
                ])
                self.num_extended_vars += 1
                
            B_list.append(B_i)
            
        self.B = np.hstack(B_list)
        
        # Tuning Matrices (Tweak these to prioritize power vs. tracking vs. slew rate)
        self.W = np.eye(self.B.shape[1]) * 1.0  # Power penalty (H in the paper)
        self.Q = np.eye(self.B.shape[1]) * 5.0  # Slew rate penalty (M in the paper)
        self.P = np.eye(3) * 1e4                # Slack penalty (Q in the paper)

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        # Target 3-DOF wrench [Fx, Fy, Mz]
        tau_target = tau_d[[0, 1, 5]]
        
        # Map current state to extended thrust vector (to penalize slew rate)
        u_ext_now = []
        if u_now is not None and alpha_now is not None:
            for i, t_conf in enumerate(self.thrusters):
                is_azimuth = getattr(t_conf, 'is_steerable', True)
                if is_azimuth:
                    u_ext_now.extend([u_now[i] * np.cos(alpha_now[i]), 
                                      u_now[i] * np.sin(alpha_now[i])])
                else:
                    u_ext_now.append(u_now[i])
        else:
            u_ext_now = np.zeros(self.B.shape[1])
        u_ext_now = np.array(u_ext_now)

        # 2. Define CVXPY Variables
        u_ext = cp.Variable(self.B.shape[1])
        slack = cp.Variable(3)
        
        # Binary variable for forbidden zone (The "Mixed-Integer" part)
        # 1 = Thrust in Sector A, 0 = Thrust in Sector B
        z = cp.Variable(boolean=True)

        # 3. Define the Cost Function
        # Minimizing energy (u_ext^2), slew rate (change in u_ext), and slack error
        cost = (cp.quad_form(u_ext, self.W) + 
                cp.quad_form(u_ext - u_ext_now, self.Q) + 
                cp.quad_form(slack, self.P))

        # 4. Constraints
        constraints = [
            self.B @ u_ext + slack == tau_target # Wrench mapping
        ]
        
        # --- Add Actuator Limits & Forbidden Zones Here ---
        # (Using a generic Big-M formulation as an example of decomposing convex sets)
        # Big_M = 1000
        # For a specific azimuth index 'idx':
        # constraints.append(u_ext[idx] <= Big_M * z) 
        # constraints.append(u_ext[idx] >= -Big_M * (1 - z))

        # 5. Solve the MIQP
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.ECOS_BB) # Use ECOS_BB, SCIP, or GUROBI

        # 6. Map extended thrust back to physical u_cmd and alpha_cmd
        u_cmd = np.zeros(self.n_thrusters)
        alpha_cmd = np.zeros(self.n_thrusters)
        
        ext_idx = 0
        for i, t_conf in enumerate(self.thrusters):
            is_azimuth = getattr(t_conf, 'is_steerable', True)
            if is_azimuth:
                ux = u_ext.value[ext_idx]
                uy = u_ext.value[ext_idx+1]
                u_cmd[i] = np.sqrt(ux**2 + uy**2)
                alpha_cmd[i] = np.arctan2(uy, ux)
                ext_idx += 2
            else:
                u_cmd[i] = u_ext.value[ext_idx]
                # Tunnel thruster angle is fixed (e.g., 90 deg sway)
                alpha_cmd[i] = np.pi / 2 
                ext_idx += 1

        return u_cmd, alpha_cmd