"""Print the Simulation 4 report tables (LaTeX) from metrics.json.

    python simulation/metrics_tables.py                        # Table V (NPO, EKF) + Table VI
    python simulation/metrics_tables.py path/to/metrics.json   # same, other run

metrics.json is written by report_metrics.py (default: next to this file).
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "metrics.json"
M = json.loads(path.read_text())


def triple(m, keys, fmt="{:.2f}"):
    """'N / E / yaw' cell, or '--' if the metric is missing."""
    if m is None or any(k not in m for k in keys):
        return "--"
    return " / ".join(fmt.format(m[k]) for k in keys)


def conv_time(t):
    """0 means the error was below the threshold from the first sample (the
    EKF's first correction pulls the estimate onto the measurement)."""
    return "$<$0.1" if t < 0.05 else f"{t:.0f}"


def estimation_table(tag, name):
    """Table V: estimation quality, open loop (4a) and offset start (4b)."""
    cols = [("With waves", M.get(f"s4a_{tag}_waves")),
            ("Without waves", M.get(f"s4a_{tag}_nowaves")),
            ("Wrong start", M.get(f"s4b_{tag}_offset"))]
    rows = [
        (r"LF error [m, m, $^\circ$]",
         lambda m, c: triple(m, ["lf_rms_N", "lf_rms_E", "lf_rms_psi"])),
        # leakage is meaningless without waves
        ("WF leakage [--]",
         lambda m, c: "--" if c == "Without waves"
         else triple(m, ["wf_ratio_N", "wf_ratio_E", "wf_ratio_psi"])),
        ("Lag [s]", lambda m, c: triple(m, ["lag_N", "lag_E", "lag_psi"], "{:.1f}")),
        (r"Velocity error [m/s, m/s, $^\circ$/s]",
         lambda m, c: triple(m, ["vel_rms_u", "vel_rms_v", "vel_rms_r"])),
        ("Convergence time [s]",
         lambda m, c: "--" if c != "Wrong start" or m is None
         else f"{conv_time(m['t_first_pos'])} (pos.) / {conv_time(m['t_first_psi'])} (head.)"),
    ]
    out = [r"\begin{table}[!t]", r"\centering",
           rf"\caption{{Simulation 4(a)/(b): {name} estimation quality, given as North / East / yaw.}}",
           rf"\label{{tab:sim4_{tag}}}",
           r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lccc}", r"\toprule",
           " & " + " & ".join(c for c, _ in cols) + r" \\", r"\midrule"]
    out += [label + " & " + " & ".join(f(m, c) for c, m in cols) + r" \\" for label, f in rows]
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(out)


def station_table():
    """Table VI: closed-loop station keeping (4c), raw vs NPO vs EKF."""
    cases = [(c, M[k]) for c, k in (("Raw measurements", "s4c_raw"), ("NPO", "s4c_npo"),
                                     ("EKF", "s4c_kf")) if k in M]
    rows = [
        ("Position error RMS / max [m]", lambda m: f"{m['pos_rms']:.2f} / {m['pos_max']:.2f}"),
        (r"Heading error RMS / max [$^\circ$]", lambda m: f"{m['psi_rms']:.1f} / {m['psi_max']:.1f}"),
        (r"Thrust use $U_T$ [\%]", lambda m: f"{m['util']:.1f}"),
        ("Thrust rate [kN/s]", lambda m: f"{m['thrust_rate']:.1f}"),
        (r"Azimuth rotation [$^\circ$/min]", lambda m: f"{m['az_deg_per_min']:.0f}"),
        (r"Saturation [\% of time]", lambda m: f"{m['sat_pct']:.1f}"),
    ]
    out = [r"\begin{table}[!t]", r"\centering",
           r"\caption{Simulation 4(c): station keeping at the origin, 60--600 s.}",
           r"\label{tab:sim4c}",
           r"\begin{tabular}{l" + "c" * len(cases) + "}", r"\toprule",
           " & " + " & ".join(c for c, _ in cases) + r" \\", r"\midrule"]
    out += [label + " & " + " & ".join(f(m) for _, m in cases) + r" \\" for label, f in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


print(estimation_table("npo", "NPO"), end="\n\n")
print(estimation_table("kf", "EKF"), end="\n\n")
print(station_table())
