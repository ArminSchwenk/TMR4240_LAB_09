"""Passivity test for the NPO: is H(s) = H_B(s) H_0(s) strictly positive real?

Fossen's passivity proof for the nonlinear passive observer needs, per DOF,
the transfer function

    h_i(s) = h_B,i(s) h_0,i(s)

to be strictly positive real, which in practice means its phase stays within
+-90 deg at all frequencies.  Here

    h_0,i(s) : nu_tilde -> y_tilde of the wave/pose error subsystem
               (built from the observer's own Aw, K1, K2, Cw, so it tests the
               gains actually implemented, not a textbook formula)
    h_B,i(s) = (s + 1/T_i + K3_i/K4_i) / (s + 1/T_i)
               (bias + velocity injection, K4_i > 0 dropped: it only scales
               the magnitude and does not change the phase)

The usual sufficient tuning rule is 1/T_i << K3_i/K4_i < omega_0 < omega_c,
but the phase bound is the actual requirement, so it is checked directly
with control.frequency_response().

    python passivity_test.py
"""
import numpy as np
import control as ct
import matplotlib.pyplot as plt

from part_2.observer import select_observer

DOFS = ("North", "East", "Yaw")
W = np.logspace(-4, 2, 4000)             # [rad/s]
PHASE_LIMIT = 90.0                       # [deg]


def h0(obs, i):
    """nu_tilde -> y_tilde for DOF i, from the observer matrices (R = I).

    Error states [xi_tilde (2), eta_tilde (1)]:
        xi_tilde'  = Aw xi_tilde - K1 y_tilde
        eta_tilde' = nu_tilde    - K2 y_tilde
        y_tilde    = eta_tilde + Cw xi_tilde
    """
    s = slice(2 * i, 2 * i + 2)
    C = np.hstack([obs.Cw[[i], s], [[1.0]]])
    A = np.zeros((3, 3))
    A[0:2, 0:2] = obs.Aw[s, s]
    A -= np.vstack([obs.K1[s, [i]], [[obs.K2[i, i]]]]) @ C
    B = np.array([[0.0], [0.0], [1.0]])
    return ct.ss(A, B, C, 0.0)


def hB(obs, i):
    """Bias + velocity injection, normalised by K4_i."""
    inv_T = 1.0 / obs.Tb[i, i]
    ratio = obs.K3[i, i] / obs.K4[i, i]
    return ct.tf([1.0, inv_T + ratio], [1.0, inv_T])


def main():
    obs = select_observer("nonlinear_passive")

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey="row")
    all_ok = True
    for i, name in enumerate(DOFS):
        resp = ct.frequency_response(hB(obs, i) * h0(obs, i), W)
        H = np.asarray(resp.complex).squeeze()
        phase = np.degrees(np.unwrap(np.angle(H)))
        ok = np.all(np.abs(phase) < PHASE_LIMIT)
        all_ok &= ok

        k_min, k_max = np.argmin(phase), np.argmax(phase)
        ratio = obs.K3[i, i] / obs.K4[i, i]
        print(f"{name:5s}  K3/K4 = {ratio:.3f} rad/s, omega_c/omega_0 = "
              f"{obs.omega_c[i] / obs.omega_0:.2f}   phase in "
              f"[{phase[k_min]:.1f}, {phase[k_max]:.1f}] deg "
              f"(min at {W[k_min]:.2f} rad/s)   {'PASS' if ok else 'FAIL'}")

        ax_m, ax_p = axes[0, i], axes[1, i]
        ax_m.loglog(W, np.abs(H), "C0")
        ax_m.set_title(f"{name}: {'PASS' if ok else 'FAIL'}")
        ax_p.semilogx(W, phase, "C0")
        ax_p.axhspan(-PHASE_LIMIT, PHASE_LIMIT, color="C2", alpha=0.08)
        for lim in (-PHASE_LIMIT, PHASE_LIMIT):
            ax_p.axhline(lim, color="C3", ls="--", lw=1)
        ax_p.plot(W[k_min], phase[k_min], "C3o", ms=4)
        ax_p.annotate(f"{phase[k_min]:.1f}° at {W[k_min]:.2f} rad/s",
                      (W[k_min], phase[k_min]), textcoords="offset points",
                      xytext=(-90, -14), fontsize=8)
        for ax in (ax_m, ax_p):
            ax.axvline(obs.omega_0, color="grey", ls=":", lw=1)
            ax.grid(True, which="both", alpha=0.3)
        ax_p.set_xlabel(r"$\omega$ [rad/s]")
    axes[0, 0].set_ylabel(r"$|h_B h_0|$ [-]")
    axes[1, 0].set_ylabel("phase [deg]")
    axes[1, 0].set_ylim(-180, 180)
    axes[1, 0].set_yticks(np.arange(-180, 181, 45))
    fig.suptitle(r"NPO passivity test: $h_B(s)\,h_0(s)$ must have phase within $\pm 90°$")
    fig.tight_layout()
    fig.savefig("npo_passivity_test.png", dpi=150)
    print("ALL DOFs PASS" if all_ok else "PASSIVITY CONDITION VIOLATED")


if __name__ == "__main__":
    main()
