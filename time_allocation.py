"""
Measure how long the Part 2 thrust allocation takes per call in a normal
closed-loop run.

The simulation is set up exactly as in ``run_case_part_2.py``; the only
difference is that every call of ``ThrustAllocator.allocate`` is timed with
``time.perf_counter``. The allocator code itself is not changed.

Usage (from the repository root):
    python time_allocation.py                      # Simulation 2, 1500 s
    python time_allocation.py --scenario sim3      # Simulation 3 (environment, raw feedback)
    python time_allocation.py --duration 300       # shorter run
    python time_allocation.py --stage1-only        # without the rate-aware second stage
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from part_2.config import AllocationConfig, Part2SimConfig, default_thrusters_part_2
from part_2.thrust_allocation import ThrustAllocator
from run_case_part_2 import DT, environment, four_corner_commands, make_sim


def build(scenario: str, duration: float):
    """Same settings as sim2() / sim3() in run_case_part_2.py."""
    if scenario == "sim2":
        cfg = Part2SimConfig(dt=DT, T=duration, use_reference=True,
                             use_observer=False, thruster_dynamics=True)
        env = environment(current=False, wind=False, waves=False)
    elif scenario == "sim3":
        cfg = Part2SimConfig(dt=DT, T=duration, use_observer=False)
        env = environment()
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return cfg, four_corner_commands(cfg, hold=300.0), env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=["sim2", "sim3"], default="sim2")
    ap.add_argument("--duration", type=float, default=1500.0, help="simulated time [s]")
    ap.add_argument("--stage1-only", action="store_true", help="rate_aware=False")
    args = ap.parse_args()

    cfg, eta_cmd, (current, wind, waves) = build(args.scenario, args.duration)
    sim = make_sim(cfg)
    if args.stage1_only:
        sim.allocator = ThrustAllocator(default_thrusters_part_2(), AllocationConfig(rate_aware=False))

    # Time every call of the allocator
    times = []
    allocate = sim.allocator.allocate

    def timed_allocate(*a, **k):
        t0 = time.perf_counter()
        out = allocate(*a, **k)
        times.append(time.perf_counter() - t0)
        return out

    sim.allocator.allocate = timed_allocate

    sim.reset_state()
    t_wall = time.perf_counter()
    sim.run(eta_cmd, current=current, wind=wind, waves=waves)
    t_wall = time.perf_counter() - t_wall

    ms = 1e3 * np.asarray(times)
    first, rest = ms[0], ms[1:]  # the first call also compiles the QPs
    budget = 1e3 * cfg.dt

    print(f"\nThrust allocation timing: {args.scenario}, {args.duration:.0f} s, "
          f"{'stage 1 only' if args.stage1_only else 'stages 1 + 2'}")
    print(f"  calls              {len(ms)}")
    print(f"  first call         {first:7.2f} ms  (includes compiling the QPs)")
    print(f"  mean               {rest.mean():7.2f} ms")
    print(f"  median             {np.median(rest):7.2f} ms")
    print(f"  95th percentile    {np.percentile(rest, 95):7.2f} ms")
    print(f"  99th percentile    {np.percentile(rest, 99):7.2f} ms")
    print(f"  max                {rest.max():7.2f} ms")
    for lim in (10.0, 20.0, budget):
        print(f"  calls > {lim:5.1f} ms   {np.mean(rest > lim) * 100:7.3f} %")
    print(f"  sampling time      {budget:7.1f} ms")
    print(f"  allocation share   {ms.sum() / 1e3 / t_wall * 100:5.1f} % of {t_wall:.1f} s wall-clock")


if __name__ == "__main__":
    main()
