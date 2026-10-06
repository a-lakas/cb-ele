"""Animation + metrics for a baseline method (shared by animate_*.py).

Writes, in --output-dir:
  <method>_metrics.csv     per-round metrics
  <method>_summary.csv     common metrics (docs/metrics.md), one row
  <method>_animation.gif   swarm + metric panels, same layout as CB-ELE
"""

import argparse
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

import cb_ele
import baselines
from cb_ele import pairwise_distances, components
from compare import trajectory_metrics

TITLES = {
    "pacnav": "PACNav (Ahmad et al. 2022)",
    "couzin": "Couzin et al. (2005) informed-individuals model",
}
INFORMED_COLOR = "#d62728"
UNINFORMED_COLOR = "#17becf"


def per_round_metrics(traj, config, arrival_radius):
    pos, vel, goal = traj["position"], traj["velocity"], traj["goal"]
    rounds, n, _ = pos.shape
    centroid = pos.mean(axis=1)
    speed = np.linalg.norm(vel, axis=2)
    rows = []
    for t in range(rounds):
        d = pairwise_distances(pos[t])
        np.fill_diagonal(d, np.inf)
        groups = components(d <= config.sensing_radius)
        moving = speed[t] > 1e-6
        u = vel[t, moving] / speed[t, moving, None]
        m = len(u)
        s = u.sum(axis=0)
        order = (s @ s - m) / (m * (m - 1)) if m > 1 else np.nan
        c_vel = centroid[t] - centroid[t - 1] if t else np.zeros(2)
        to_goal = goal - centroid[t]
        nv, ng = np.linalg.norm(c_vel), np.linalg.norm(to_goal)
        accuracy = (1.0 - np.arccos(np.clip(c_vel @ to_goal / (nv * ng), -1, 1)) / np.pi
                    if nv > 1e-9 and ng > 1e-9 else np.nan)
        rows.append({
            "round": t,
            "order": order,
            "heading_accuracy": accuracy,
            "mean_speed": float(speed[t].mean()),
            "normalized_mean_speed": float(speed[t].mean() / config.max_speed),
            "minimum_distance": float(d.min()),
            "component_count": len(groups),
            "largest_component_fraction": max(len(g) for g in groups) / n,
            "swarm_radius": float(np.linalg.norm(pos[t] - centroid[t], axis=1).max()),
            "centroid_distance": float(np.linalg.norm(to_goal)),
            "arrived_fraction": float((np.linalg.norm(pos[t] - goal, axis=1)
                                       <= arrival_radius).mean()),
        })
    return pd.DataFrame(rows)


def animate(method, traj, table, summary, config, args):
    pos, goal, informed = traj["position"], traj["goal"], traj["informed"]
    rounds = len(pos)
    frames = sorted(set(range(0, rounds, args.stride)) | {rounds - 1})
    colors = np.where(informed, INFORMED_COLOR, UNINFORMED_COLOR)

    fig = plt.figure(figsize=(22, 10), dpi=80, constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.9, 1.0])
    ax = fig.add_subplot(grid[0, :])
    left = fig.add_subplot(grid[1, 0])
    right = fig.add_subplot(grid[1, 1])

    ax.set(xlim=(-4, config.width + 4), ylim=(-2, config.height + 2),
           aspect="auto", xlabel="x (m)", ylabel="y (m)")
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.scatter(*goal, marker="X", s=210, color="limegreen", edgecolors="black")
    ax.add_patch(plt.Circle(goal, args.arrival_radius, fill=False, color="green",
                            linestyle="--", alpha=0.35))
    ax.text(goal[0] + 2.0, goal[1] + 1.0, "GOAL", color="green", weight="bold")
    v0 = traj["velocity"][0]
    agents = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=70, c=colors,
                        edgecolors="black", linewidths=0.6)
    arrows = ax.quiver(pos[0, :, 0], pos[0, :, 1], v0[:, 0], v0[:, 1],
                       color="black", alpha=0.55, scale=50, width=0.002)
    status = ax.text(0.012, 0.97, "", transform=ax.transAxes, va="top",
                     family="monospace", fontsize=7.6,
                     bbox=dict(facecolor="white", alpha=0.92, edgecolor="#888",
                               boxstyle="round"), zorder=20)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=c,
               markeredgecolor="black", label=l)
        for c, l in ((INFORMED_COLOR, "INFORMED (knows goal)"),
                     (UNINFORMED_COLOR, "UNINFORMED"))],
        loc="upper right", fontsize=6.7)
    ax.set_title(f"{TITLES[method]} — {config.width:.0f} m domain — "
                 f"informed fraction {args.informed:.2f}", weight="bold")

    x = table["round"].to_numpy()
    left_specs = [("arrived_fraction", "green", "Arrived fraction"),
                  ("largest_component_fraction", "blue", "Largest component"),
                  ("normalized_mean_speed", "black", "Normalized speed")]
    right_specs = [("order", "purple", "Order Ω"),
                   ("heading_accuracy", "orange", "Heading accuracy")]
    left_lines = [left.plot([], [], color=c, label=l)[0] for _, c, l in left_specs]
    right_lines = [right.plot([], [], color=c, label=l)[0] for _, c, l in right_specs]
    cursors = [a.axvline(0, color="black", linestyle="--", alpha=0.5)
               for a in (left, right)]
    left.set(xlim=(0, rounds - 1), ylim=(-0.05, 1.05), xlabel="Round",
             ylabel="Fraction", title="Arrival and connectivity")
    right.set(xlim=(0, rounds - 1), ylim=(-1.05, 1.05), xlabel="Round",
              ylabel="Normalized value", title="Coordination")
    for a in (left, right):
        a.grid(True, linestyle=":", alpha=0.3)
        a.legend(fontsize=6.7, loc="lower right")

    def update(k):
        t = frames[k]
        m = table.iloc[t]
        agents.set_offsets(pos[t])
        arrows.set_offsets(pos[t])
        v = traj["velocity"][t]
        arrows.set_UVC(v[:, 0], v[:, 1])
        status.set_text(
            f"t={t:4d}  informed={int(informed.sum())}/{len(informed)}\n"
            f"components={int(m.component_count):2d}  "
            f"largest={m.largest_component_fraction:.2f}\n"
            f"min d={m.minimum_distance:.2f} m  "
            f"radius={m.swarm_radius:.1f} m\n"
            f"order={m.order:+.3f}  accuracy={m.heading_accuracy:.3f}  "
            f"speed={m.mean_speed:.3f}\n"
            f"goal d={m.centroid_distance:.1f} m  "
            f"arrived={m.arrived_fraction:.2f}")
        end = t + 1
        for line, (col, _, _) in zip(left_lines, left_specs):
            line.set_data(x[:end], table[col].to_numpy()[:end])
        for line, (col, _, _) in zip(right_lines, right_specs):
            line.set_data(x[:end], table[col].to_numpy()[:end])
        for c in cursors:
            c.set_xdata([t, t])
        return (agents, arrows, status, *left_lines, *right_lines, *cursors)

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 * args.stride / args.fps)
    path = os.path.join(args.output_dir, f"{method}_animation.gif")
    t0 = time.perf_counter()
    anim.save(path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Animation written to {path} ({time.perf_counter() - t0:.1f} s)")
    return path


def main(method, argv=None):
    parser = argparse.ArgumentParser(
        description=f"{TITLES[method]}: metrics + GIF on the CB-ELE world.")
    parser.add_argument("--seed", type=int, default=cb_ele.CONFIG.seed)
    parser.add_argument("--rounds", type=int, default=2200)
    parser.add_argument("--informed", type=float, default=0.3,
                        help="fraction of agents that know the goal")
    parser.add_argument("--arrival-radius", type=float, default=60.0)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)

    config = replace(cb_ele.CONFIG, seed=args.seed, rounds=args.rounds)
    t0 = time.perf_counter()
    traj = baselines.run_baseline(config, baselines.BaselineConfig(
        method=method, informed_fraction=args.informed))
    print(f"[{method}] Simulation: {args.rounds} rounds, {config.n_agents} agents, "
          f"{time.perf_counter() - t0:.2f} s")

    os.makedirs(args.output_dir, exist_ok=True)
    table = per_round_metrics(traj, config, args.arrival_radius)
    metrics_path = os.path.join(args.output_dir, f"{method}_metrics.csv")
    table.to_csv(metrics_path, index=False, float_format="%.6f")
    summary = trajectory_metrics(traj, config, args.arrival_radius)
    for key in ("election_time", "unique_leader_fraction", "leader_changes"):
        summary.pop(key)                         # CB-ELE-only metrics
    summary = pd.Series({"method": method, "seed": args.seed,
                         "informed_fraction": args.informed, **summary})
    summary_path = os.path.join(args.output_dir, f"{method}_summary.csv")
    summary.to_frame().T.to_csv(summary_path, index=False, float_format="%.4f")
    print(f"\nCSV outputs:\n  {metrics_path}\n  {summary_path}")
    print("\nRun summary:")
    print(summary.to_string())

    if not args.no_animation:
        animate(method, traj, table, summary, config, args)
    print("\nDone.")
