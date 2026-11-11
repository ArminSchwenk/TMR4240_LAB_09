"""Metrics and figures for the report: Simulations 3/4/6/7 and the sensor-noise
bonus, NPO focus (Kalman numbers are computed too, for the EKF author).

    python literature/results/run_03_report/report_metrics.py

Every run uses the same presets as run_case_part_2.py (environment(), SEED,
OBSERVER_OFFSET, four_corner_commands, WrenchSequence).  Writes
metrics.md, metrics.json and fig_*.png next to this file.
"""
import json, os, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
# Repository root: works from simulation/ and from literature/results/<run>/
ROOT = next(p for p in HERE.parents if (p / "run_case_part_2.py").exists())
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import run_case_part_2 as rc                                   # noqa: E402
import models.sensors as sens                                  # noqa: E402
from part_2.config import Part2SimConfig, default_thrusters_part_2  # noqa: E402
from part_2.utilization import thrust_utilization              # noqa: E402
from simulation.plotter import plot_dashboard                  # noqa: E402
from simulation.plotter_part_2 import (plot_observer, plot_observer_error,  # noqa: E402
                                       _moving_average)

DT, WIN = rc.DT, 30.0
DOF = [(0, "N", 1.0), (1, "E", 1.0), (5, "psi", np.degrees(1.0))]
THR = default_thrusters_part_2()
UMAX = np.array([th.u_max for th in THR])
M = {}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def save(fig, name, title=None):
    if title:
        fig.suptitle(title)
    fig.savefig(HERE / f"fig_{name}.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def window(L, t0, t1):
    return (L.t >= t0) & (L.t <= t1)


def band_rms(x, w_lo, w_hi):
    """RMS of the part of x inside [w_lo, w_hi] rad/s (FFT mask, detrended)."""
    x = np.asarray(x, float)
    x = x - _moving_average(x, DT, WIN)          # remove drift (periods > ~30 s)
    x = x[int(WIN / DT):-int(WIN / DT)]           # drop the edge-padded ends
    X = np.fft.rfft(x * np.hanning(len(x))) / np.sqrt(np.mean(np.hanning(len(x)) ** 2))
    w = 2 * np.pi * np.fft.rfftfreq(len(x), DT)
    keep = (w >= w_lo) & (w <= (w_hi if w_hi is not None else np.inf))
    return np.sqrt(np.mean(np.fft.irfft(X * keep, len(x)) ** 2))


def estimation_metrics(L, t0, t1):
    """Estimate vs. LF truth (30 s centred moving average of the true state)."""
    m = window(L, t0, t1)
    out = {}
    for j, name, s in DOF:
        true = np.unwrap(L.eta[:, j]) if j == 5 else L.eta[:, j]
        est = np.unwrap(L.eta_est[:, j]) if j == 5 else L.eta_est[:, j]
        meas = np.unwrap(L.eta_measured[:, j]) if j == 5 else L.eta_measured[:, j]
        lf = _moving_average(true, DT, WIN)
        e = est - lf
        out[f"lf_rms_{name}"] = s * np.sqrt(np.mean(e[m] ** 2))
        out[f"lf_max_{name}"] = s * np.max(np.abs(e[m]))
        # WF attenuation: RMS in the wave band (0.5-1.3 rad/s) of the estimate
        # divided by that of the measurement; smoothness: RMS of the estimate
        # above 0.5 rad/s (what the controller would see as fast jitter)
        out[f"wf_ratio_{name}"] = band_rms(est[m], 0.5, 1.3) / band_rms(meas[m], 0.5, 1.3)
        out[f"smooth_{name}"] = s * band_rms(est[m], 0.5, None)
        # lag: shift (0..20 s) that best aligns the estimate with the LF truth
        shifts = np.arange(0, int(20 / DT) + 1)
        idx = np.where(m)[0]
        idx = idx[idx >= shifts[-1]]
        cost = [np.mean((est[idx] - lf[idx - k]) ** 2) for k in shifts]
        out[f"lag_{name}"] = shifts[int(np.argmin(cost))] * DT
    for j, name, s in [(0, "u", 1.0), (1, "v", 1.0), (5, "r", np.degrees(1.0))]:
        lf = _moving_average(L.nu[:, j], DT, WIN)
        out[f"vel_rms_{name}"] = s * np.sqrt(np.mean((L.nu_est[m, j] - lf[m]) ** 2))
    return out


def station_metrics(L, t0, t1):
    """Closed-loop: tracking error vs. the reference, and actuator activity."""
    m = window(L, t0, t1)
    pos = np.hypot(L.eta[m, 0] - L.sp[m, 0], L.eta[m, 1] - L.sp[m, 1])
    psi = np.degrees(wrap(L.eta[m, 5] - L.sp[m, 5]))
    du = np.diff(L.u[m], axis=0) / DT
    dalpha = np.abs(np.diff(np.unwrap(L.alpha[m][:, 1:], axis=0), axis=0)).sum() / (t1 - t0) * 60
    res = (L.tau_d[m] - L.tau_thr[m])[:, [0, 1, 5]] * 1e-3
    # LF part of the tracking error (30 s moving average of true - reference)
    eN = _moving_average(L.eta[:, 0] - L.sp[:, 0], DT, WIN)[m]
    eE = _moving_average(L.eta[:, 1] - L.sp[:, 1], DT, WIN)[m]
    ep = np.degrees(_moving_average(wrap(L.eta[:, 5] - L.sp[:, 5]), DT, WIN)[m])
    return dict(pos_lf_rms=np.sqrt(np.mean(eN ** 2 + eE ** 2)), psi_lf_rms=np.sqrt(np.mean(ep ** 2)),
                psi_lf_max=np.abs(ep).max(),
                pos_rms=np.sqrt(np.mean(pos ** 2)), pos_max=pos.max(),
                psi_rms=np.sqrt(np.mean(psi ** 2)), psi_max=np.abs(psi).max(),
                util=thrust_utilization(L, THR)[m].mean(),
                thrust_rate=np.sqrt(np.mean((du * 1e-3) ** 2, axis=0)).sum(),
                az_deg_per_min=np.degrees(dalpha),
                res_Fxy=np.sqrt(np.mean(res[:, 0] ** 2 + res[:, 1] ** 2)),
                res_Mz=np.sqrt(np.mean(res[:, 2] ** 2)),
                sat_pct=100 * np.mean(np.any(np.abs(L.u[m]) >= 0.98 * UMAX, axis=1)))


def convergence(L, offset, frac=0.1):
    """First time the LF error falls below frac*initial offset, and the time
    after which it stays below it (pos: norm in m, heading in deg)."""
    out = {}
    lfN = _moving_average(L.eta[:, 0], DT, WIN)
    lfE = _moving_average(L.eta[:, 1], DT, WIN)
    lfp = _moving_average(np.unwrap(L.eta[:, 5]), DT, WIN)
    ep = np.hypot(L.eta_est[:, 0] - lfN, L.eta_est[:, 1] - lfE)
    eh = np.abs(np.degrees(np.unwrap(L.eta_est[:, 5]) - lfp))
    for name, e, e0 in (("pos", ep, np.hypot(offset[0], offset[1])),
                        ("psi", eh, np.degrees(abs(offset[5])))):
        thr = frac * e0
        valid = L.t <= L.t[-1] - WIN / 2
        below = (e < thr) & valid
        first = L.t[np.argmax(below)] if below.any() else np.nan
        above = np.where((e >= thr) & valid)[0]
        stay = L.t[above[-1] + 1] if len(above) else 0.0
        out[f"t_first_{name}"], out[f"t_stay_{name}"], out[f"thr_{name}"] = first, stay, thr
    return out


def run(cfg, cmd, env, controller=None, offset=None):
    sim = rc.make_sim(cfg, controller)
    sim.reset_state()
    if offset is not None:
        sim.observer.reset(offset)
    cur, wnd, wav = env
    return sim.run(cmd, current=cur, wind=wnd, waves=wav)


fixed = rc.WrenchSequence([(1.0, [2e3, 2e3, 2e3])])
t_all = time.time()

# ---- Simulation 4a/4b: fixed wrench, observer only ------------------------
for kind in ("nonlinear_passive", "kalman"):
    tag = "npo" if kind == "nonlinear_passive" else "kf"
    for waves in (True, False):
        cfg = Part2SimConfig(dt=DT, T=600.0, use_reference=False, use_observer=True, observer_type=kind)
        L = run(cfg, np.zeros(6), rc.environment(waves=waves), fixed)
        key = f"s4a_{tag}_{'waves' if waves else 'nowaves'}"
        M[key] = estimation_metrics(L, 60, 570)
        name = "NPO" if tag == "npo" else "EKF"
        if waves:
            save(plot_observer(L), key + "_obs", f"{name}, fixed wrench, with waves")
        save(plot_observer_error(L), key + "_err",
             f"{name} estimation errors, fixed wrench, {'with' if waves else 'without'} waves")
    cfg = Part2SimConfig(dt=DT, T=300.0, use_reference=False, use_observer=True, observer_type=kind)
    L = run(cfg, np.zeros(6), rc.environment(), fixed, offset=rc.OBSERVER_OFFSET)
    M[f"s4b_{tag}_offset"] = convergence(L, rc.OBSERVER_OFFSET)
    M[f"s4b_{tag}_offset"].update(estimation_metrics(L, 100, 270))
    save(plot_observer_error(L), f"s4b_{tag}_offset_err",
         f"{'NPO' if tag == 'npo' else 'EKF'} estimation errors, start offset [5 m, -5 m, 10 deg]")
    print("4a/4b", kind, "done", flush=True)

# ---- Simulation 4c: closed-loop station keeping, raw vs observer ----------
for label, use_obs, kind in (("raw", False, "nonlinear_passive"),
                             ("npo", True, "nonlinear_passive"),
                             ("kf", True, "kalman")):
    cfg = Part2SimConfig(dt=DT, T=600.0, use_observer=use_obs, observer_type=kind)
    L = run(cfg, np.zeros(6), rc.environment())
    M[f"s4c_{label}"] = station_metrics(L, 60, 600)
    if use_obs:
        M[f"s4c_{label}"].update(estimation_metrics(L, 60, 570))
    save(plot_dashboard(L) or plt.gcf(), f"s4c_{label}_dash")
print("4c done", flush=True)

# Only Simulation 4 (NPO, raw, EKF) is run now. Uncomment the blocks below to
# also get Simulation 6, 3/7 and the sensor-noise bonus.

# # ---- Simulation 6: heavy sea, 1000 s, NPO ---------------------------------
# cfg = Part2SimConfig(dt=DT, T=1000.0, use_observer=True, observer_type="nonlinear_passive")
# L = run(cfg, np.zeros(6), rc.environment(hs=rc.ROBUST_HS, tp=rc.ROBUST_TP))
# M["s6_win"] = station_metrics(L, 60, 600); M["s6_win"].update(estimation_metrics(L, 60, 570))
# M["s6_full"] = station_metrics(L, 60, 1000); M["s6_full"].update(estimation_metrics(L, 60, 970))
# save(plot_dashboard(L) or plt.gcf(), "s6_dash")
# save(plot_observer_error(L), "s6_err", "Simulation 6: NPO estimation errors, heavy sea")
# print("6 done", flush=True)
#
# # ---- Simulation 3 vs 7: four-corner, raw vs NPO ---------------------------
# for label, use_obs in (("s3_raw", False), ("s7_npo", True)):
#     cfg = Part2SimConfig(dt=DT, T=1500.0, use_observer=use_obs, observer_type="nonlinear_passive")
#     L = run(cfg, rc.four_corner_commands(cfg, 300.0), rc.environment())
#     M[label] = station_metrics(L, 0, 1500)
#     # per-leg settled error: last 100 s of each leg
#     legs = [(200 + 300 * i, 300 + 300 * i) for i in range(5)]
#     M[label]["settled_pos_rms"] = np.sqrt(np.mean([station_metrics(L, a, b)["pos_rms"] ** 2 for a, b in legs]))
#     M[label]["settled_psi_rms"] = np.sqrt(np.mean([station_metrics(L, a, b)["psi_rms"] ** 2 for a, b in legs]))
#     if use_obs:
#         M[label].update(estimation_metrics(L, 30, 1470))
#         save(plot_dashboard(L) or plt.gcf(), "s7_dash")
# print("3/7 done", flush=True)
#
# # ---- Bonus: sensor noise ---------------------------------------------------
# cfg = Part2SimConfig(dt=DT, T=600.0, use_reference=False, use_observer=True,
#                      observer_type="nonlinear_passive", use_sensor_noise=True)
# L = run(cfg, np.zeros(6), rc.environment(), fixed)
# M["bn_npo_fixed"] = estimation_metrics(L, 60, 570)
# save(plot_observer_error(L), "bn_npo_fixed_err", "NPO estimation errors, fixed wrench, waves + sensor noise")
# for label, use_obs in (("bn_raw", False), ("bn_npo", True)):
#     cfg = Part2SimConfig(dt=DT, T=600.0, use_observer=use_obs, observer_type="nonlinear_passive",
#                          use_sensor_noise=True)
#     L = run(cfg, np.zeros(6), rc.environment())
#     M[label] = station_metrics(L, 60, 600)
#     if use_obs:
#         M[label].update(estimation_metrics(L, 60, 570))
#     save(plot_dashboard(L) or plt.gcf(), f"{label}_dash")
# full = sens.SENSOR_ETA_STD.copy()
# for label, std in (("bn_npo_posnoise", full * [1, 1, 0, 0, 0, 0]),
#                    ("bn_npo_hdgnoise", full * [0, 0, 0, 0, 0, 1])):
#     sens.SENSOR_ETA_STD = std
#     cfg = Part2SimConfig(dt=DT, T=600.0, use_observer=True, observer_type="nonlinear_passive",
#                          use_sensor_noise=True)
#     L = run(cfg, np.zeros(6), rc.environment())
#     M[label] = station_metrics(L, 60, 600); M[label].update(estimation_metrics(L, 60, 570))
# sens.SENSOR_ETA_STD = full
# print("bonus done", flush=True)

# ---- write -----------------------------------------------------------------
M = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in M.items()}
(HERE / "metrics.json").write_text(json.dumps(M, indent=1))
lines = [f"# Report metrics (run {time.strftime('%Y-%m-%d %H:%M')}, {time.time() - t_all:.0f} s)\n"]
for k, v in M.items():
    lines.append(f"## {k}\n\n| metric | value |\n|---|---|")
    lines += [f"| {kk} | {vv:.4g} |" for kk, vv in v.items()]
    lines.append("")
(HERE / "metrics.md").write_text("\n".join(lines))
print("written")
