"""Run every NPO configuration we tried through the small tests.

    python literature/results/npo_tuning_history/run_tuning_history.py

For each config: passivity (phase of h_B h_0, as tests/passivity_test.py),
notch (y -> eta_hat gain, linearised as tests/notch_test.py) and the real
step() code at omega_0.  Writes results.md and notch_comparison.png here.
"""
import os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import control as ct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
os.chdir(ROOT)

from part_2.observer import NonlinearPassiveObserver   # noqa: E402
from part_2.config import NPOConfig                     # noqa: E402
import passivity_test as pt                             # noqa: E402
import notch_test as nt                                 # noqa: E402

W0 = 2 * np.pi / 8.0
# (id, source, zeta_n, cutoff ratio, k4, K3/K4, note)
CONFIGS = [
    ("V0", "git f489c3a (NPO_start)", 1.0, 1.2, 0.10, 0.17 * W0, "first draft; K3 = 0.17*omega_0*K4 hidden in observer.py"),
    ("V0b", "notes (prof. counter-example)", 1.0, 1.2, 0.10, 0.10, "rule 1/Tb << K3/K4 < w0 < wc holds, phase still fails"),
    ("V1", "notes ('before' table)", 1.0, 2.0, 0.09, 0.10, "cutoff raised to 2*omega_0"),
    ("V1a", "reconstructed step", 2.0, 2.0, 0.09, 0.10, "V1 + wider notch only"),
    ("V1b", "reconstructed step", 2.0, 2.0, 0.09, 0.02, "V1a + smaller bias corner"),
    ("V2", "git b9df7ab (FINAL)", 2.0, 2.0, 0.20, 0.02, "V1b + larger velocity gain k4 = 0.2"),
    ("V2+", "notes", 2.0, 2.0, 0.50, 0.02, "k4 = 0.5: LF peak too large"),
    ("E1", "run_02_K3", 2.0, 2.0, 0.20, 0.01, "noise experiment"),
    ("E2", "run_02_K3", 2.0, 2.0, 0.20, 0.005, "noise experiment"),
    ("E3", "run_02_cutoff", 2.0, 1.5, 0.20, 0.02, "noise experiment"),
    ("E4", "run_02_cutoff", 2.0, 1.2, 0.20, 0.02, "noise experiment"),
]


def make(zn, cr, k4, k3):
    return NonlinearPassiveObserver(NPOConfig(notch_damping_ratio=np.full(3, zn),
                                              cutoff_ratio=np.full(3, cr),
                                              k4=np.full(3, k4), k3_factor=k3))


def step_gain_at_w0(cfg_args):
    """Real step() code: sine at omega_0 in North, return |eta_hat|/|y|."""
    obs = make(*cfg_args)
    obs.reset(np.zeros(6))
    dt, w = 0.1, W0
    n = int((600.0 + 300.0) / dt)
    t = np.arange(n) * dt
    eh = np.zeros(n)
    for k in range(n):
        y = np.zeros(6); y[0] = np.sin(w * t[k])
        eh[k] = obs.step(t[k], dt, y, np.zeros(6)).eta[0]
    m = t >= 600.0
    X = np.column_stack([np.sin(w * t[m]), np.cos(w * t[m]), np.ones(m.sum())])
    c = np.linalg.lstsq(X, eh[m], rcond=None)[0]
    return np.hypot(c[0], c[1])


W = np.logspace(-4, 2, 20000)
WL = np.logspace(-3, 1, 1500)
rows, curves = [], {}
for cid, src, zn, cr, k4, k3, note in CONFIGS:
    obs = make(zn, cr, k4, k3)
    H = np.asarray(ct.frequency_response(pt.hB(obs, 0) * pt.h0(obs, 0), W).complex).squeeze()
    ph = np.degrees(np.unwrap(np.angle(H)))
    low = W < 10
    kmin = np.argmin(ph[low])
    rule = (1 / 300 < k3 / 3) and (k3 < W0) and (W0 < cr * W0)  # "<<" read as a factor >= 3
    He, _ = nt.linear_response(obs, WL)
    g = np.abs(He[:, 0, 0])
    curves[cid] = g
    rows.append(dict(id=cid, src=src, zn=zn, cr=cr, k4=k4, k3=k3, note=note,
                     ph_min=ph.min(), ph_low=ph[low][kmin], w_low=W[low][kmin],
                     passive=bool(np.all(np.abs(ph) < 90)), rule=rule,
                     leak=float(np.interp(W0, WL, g)), step_leak=step_gain_at_w0((zn, cr, k4, k3)),
                     peak=g.max(), w_peak=WL[np.argmax(g)], hf=float(np.interp(2.5, WL, g))))
    print(cid, "done", flush=True)

hdr = ("| ID | Source | ζn | ωc/ω0 | k4 | K3/K4 [rad/s] | Rule holds | Min phase below 10 rad/s | "
       "Passive (±90°) | Leak at ω0 (linear) | Leak at ω0 (`step()`) | LF peak | Leak at 2.5 rad/s | Note |")
lines = [hdr, "|" + "---|" * 14]
for r in rows:
    lines.append(f"| **{r['id']}** | {r['src']} | {r['zn']:g} | {r['cr']:g} | {r['k4']:g} | {r['k3']:.3g} | "
                 f"{'yes' if r['rule'] else 'no'} | {r['ph_low']:.1f}° @ {r['w_low']:.2f} | "
                 f"{'✅' if r['passive'] else '❌'} | {r['leak']:.3f} | {r['step_leak']:.3f} | "
                 f"{r['peak']:.2f} @ {r['w_peak']:.2f} | {r['hf']:.2f} | {r['note']} |")
(HERE / "results_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(8, 4.5))
for cid in ("V0", "V1", "V1a", "V2", "V2+"):
    ax.loglog(WL, curves[cid], label=cid + (" (final)" if cid == "V2" else ""), lw=2 if cid == "V2" else 1.2)
ax.axvline(W0, color="grey", ls=":", lw=1)
ax.set_xlabel(r"$\omega$ [rad/s]"); ax.set_ylabel(r"$|\hat\eta / y|$ [-]")
ax.set_title("NPO tuning history: gain from measured pose to LF estimate (North)")
ax.grid(True, which="both", alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(HERE / "notch_comparison.png", dpi=150)
