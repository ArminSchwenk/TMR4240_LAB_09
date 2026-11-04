"""Notch test for the NPO: does the observer remove wave-frequency motion?

Feed the observer a pure sinusoidal pose measurement in ONE degree of freedom,
y(t) = A sin(w t), with tau = 0 and heading held at 0 (so R = I), and measure
how much of it comes out in the LF estimate eta_hat.  Repeat over a range of w.

Expected result, per DOF:
  |eta_hat / y| ~ 1   at low w     (slow motion passes: the vessel really moves)
  |eta_hat / y| << 1  near w0      (wave motion is notched out)
  |xi2_hat / y| ~ 1   near w0      (the wave state absorbs it instead)

Simulated points (the real step() code) are plotted on top of the analytic
linear frequency response built from the observer's own matrices, so a
mismatch points to an implementation bug, not a tuning issue.

    python notch_test.py
"""
import numpy as np
import matplotlib.pyplot as plt

from part_2.observer import select_observer

DT = 0.1
A = 1.0                                   # amplitude: 1 m (N, E) or 1 rad (psi)
DOFS = {0: ("North", 0), 1: ("East", 1), 2: ("Yaw", 5)}   # 3-DOF index -> 6-DOF index
W_SIM = np.logspace(-2, np.log10(3.0), 18)                 # simulated frequencies [rad/s]
W_LIN = np.logspace(-3, 1, 600)


def simulate_gain(dof3, w):
    """Run step() on y = A sin(wt) in one DOF; return (|eta_hat|/A, |xi2_hat|/A)."""
    obs = select_observer("nonlinear_passive")
    obs.reset(np.zeros(6))
    period = 2 * np.pi / w
    t_settle = 600.0                      # let the transient (incl. slow bias mode) die out
    t_end = t_settle + max(10 * period, 300.0)
    n = int(t_end / DT)
    t = np.arange(n) * DT
    eta_hat = np.zeros(n)
    xi2_hat = np.zeros(n)
    i6 = DOFS[dof3][1]
    for k in range(n):
        y6 = np.zeros(6)
        y6[i6] = A * np.sin(w * t[k])
        est = obs.step(t[k], DT, y6, np.zeros(6))
        eta = est.eta if hasattr(est, "eta") else est[0]
        eta_hat[k] = eta[i6]
        xi2_hat[k] = obs.Cw[dof3] @ (obs.xi if hasattr(obs, "xi") else obs.x_hat[0:6])
    m = t >= t_settle

    def amp(sig):                         # least-squares fit of a sine at w
        X = np.column_stack([np.sin(w * t[m]), np.cos(w * t[m]), np.ones(m.sum())])
        c = np.linalg.lstsq(X, sig[m], rcond=None)[0]
        return np.hypot(c[0], c[1]) / A

    return amp(eta_hat), amp(xi2_hat)


def linear_response(obs, w):
    """Analytic y -> eta_hat and y -> Cw xi_hat (3x3 each) with R = I, tau = 0."""
    Z3, I3 = np.zeros((3, 3)), np.eye(3)
    # state x = [xi(6), eta(3), nu(3), b(3)];  y_tilde = y - eta - Cw xi
    Cyt = np.hstack([-obs.Cw, -I3, Z3, Z3])           # y_tilde = Cyt x + y
    F = np.zeros((15, 15))
    F[0:6, 0:6] = obs.Aw
    F[6:9, 9:12] = I3
    F[9:12, 9:12] = -obs.Minv @ obs.D
    F[9:12, 12:15] = obs.Minv
    F[12:15, 12:15] = -obs.Tb_inv
    G = np.vstack([obs.K1, obs.K2, obs.Minv @ obs.K4, obs.K3])   # injection of y_tilde
    Acl, Bcl = F + G @ Cyt, G
    C_eta = np.hstack([np.zeros((3, 6)), I3, Z3, Z3])
    C_xi = np.hstack([obs.Cw, Z3, Z3, Z3])
    H_eta, H_xi = [], []
    for wi in w:
        X = np.linalg.solve(1j * wi * np.eye(15) - Acl, Bcl)
        H_eta.append(C_eta @ X)
        H_xi.append(C_xi @ X)
    return np.array(H_eta), np.array(H_xi)


def main():
    obs = select_observer("nonlinear_passive")
    w0 = getattr(obs, "omega_0", None) or getattr(obs.cfg, "omega0", 2 * np.pi / 8.0)
    H_eta, H_xi = linear_response(obs, W_LIN)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for dof3, ax in enumerate(axes):
        name = DOFS[dof3][0]
        g_eta, g_xi = zip(*(simulate_gain(dof3, w) for w in W_SIM))
        ax.loglog(W_LIN, np.abs(H_eta[:, dof3, dof3]), "C0", label=r"$|\hat\eta / y|$ (linear)")
        ax.loglog(W_LIN, np.abs(H_xi[:, dof3, dof3]), "C1", label=r"$|\hat\eta_w / y|$ (linear)")
        ax.loglog(W_SIM, g_eta, "C0o", ms=4, label=r"$|\hat\eta / y|$ (step())")
        ax.loglog(W_SIM, g_xi, "C1s", ms=4, label=r"$|\hat\eta_w / y|$ (step())")
        ax.axvline(w0, color="grey", ls=":", lw=1)
        i0 = np.argmin(np.abs(W_LIN - w0))
        ax.set_title(f"{name}: LF gain at $\\omega_0$ = {np.abs(H_eta[i0, dof3, dof3]):.3f}")
        ax.set_xlabel(r"$\omega$ [rad/s]")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(1e-3, 3)
        print(f"{name:5s}  |eta_hat/y| at w0 = {np.abs(H_eta[i0, dof3, dof3]):.3f}   "
              f"min = {np.abs(H_eta[:, dof3, dof3]).min():.4f} "
              f"at w = {W_LIN[np.argmin(np.abs(H_eta[:, dof3, dof3]))]:.3f} rad/s")
    axes[0].set_ylabel("amplitude ratio [-]")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("NPO notch test: sinusoidal pose input, $\\tau = 0$, $\\psi = 0$")
    fig.tight_layout()
    fig.savefig("npo_notch_test.png", dpi=150)


if __name__ == "__main__":
    main()
