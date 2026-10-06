"""Benchmark CB-ELE against baselines on common metrics, over many seeds.

    python compare.py --seeds 10 --rounds 3000 --output-dir results

Every method sees the same arena, start state (per seed), sensing radius,
hard-safety projection and goal.  Metrics are computed from per-round
trajectories only (positions, realised velocities, states), so they are
identical for every method; see `docs/metrics.md` for definitions and the
papers they come from.
"""

import argparse
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd

import cb_ele
import baselines
from cb_ele import pairwise_distances, components


def trajectory_metrics(traj, config):
    """Common metrics from a trajectory dict (position/velocity/state/goal)."""
    pos, vel, goal = traj["position"], traj["velocity"], traj["goal"]
    rounds, n, _ = pos.shape
    centroid = pos.mean(axis=1)
    centroid_goal = np.linalg.norm(centroid - goal, axis=1)
    agent_goal = np.linalg.norm(pos - goal, axis=2)

    speed = np.linalg.norm(vel, axis=2)
    unit_vel = vel / np.maximum(speed, 1e-9)[..., None]
    moving = speed > 1e-6
    order = np.array([
        np.linalg.norm(unit_vel[t, moving[t]].mean(axis=0))
        if moving[t].any() else np.nan for t in range(rounds)])

    min_dist = np.empty(rounds)
    n_components = np.empty(rounds, dtype=int)
    largest = np.empty(rounds)
    collisions = np.empty(rounds, dtype=int)
    for t in range(rounds):
        d = pairwise_distances(pos[t])
        np.fill_diagonal(d, np.inf)
        min_dist[t] = d.min()
        collisions[t] = int(np.triu(d < config.hard_safety_distance - 1e-6, 1).sum())
        groups = components(d <= config.sensing_radius)
        n_components[t] = len(groups)
        largest[t] = max(len(g) for g in groups) / n
    swarm_radius = np.linalg.norm(pos - centroid[:, None, :], axis=2).max(axis=1)

    arrived_any = agent_goal <= config.swarm_goal_radius
    reached = np.flatnonzero(centroid_goal <= config.swarm_goal_radius)
    t_goal = int(reached[0]) if len(reached) else np.nan
    t_90 = np.flatnonzero(arrived_any.mean(axis=1) >= 0.9)
    path_length = np.linalg.norm(vel, axis=2).sum(axis=0)
    progress = agent_goal[0] - agent_goal[-1]

    return {
        "success": bool(arrived_any[-1].mean() >= 0.9 and n_components[-1] == 1),
        "time_to_goal": t_goal,
        "time_90pct_arrived": int(t_90[0]) if len(t_90) else np.nan,
        "final_arrival_fraction": float(arrived_any[-1].mean()),
        "mean_order": float(np.nanmean(order[: t_goal if t_goal == t_goal else rounds])),
        "min_interagent_distance": float(min_dist.min()),
        "collision_rounds": int((collisions > 0).sum()),
        "fragmented_round_fraction": float((n_components > 1).mean()),
        "min_largest_component": float(largest.min()),
        "mean_swarm_radius": float(swarm_radius.mean()),
        "path_efficiency": float(np.mean(progress / np.maximum(path_length, 1e-9))),
    }


def methods(goal_knowledge):
    """(name, runner) pairs for one goal-knowledge setting."""
    informed = 1.0 if goal_knowledge == "all" else 0.1

    def cbele(cfg):
        if goal_knowledge != "all":
            cfg = replace(cfg, follower_goal_weight=0.0)
        return cb_ele.run_simulation(cfg, return_trajectory=True)[-1]

    def baseline(name):
        return lambda cfg: baselines.run_baseline(
            cfg, baselines.BaselineConfig(method=name, informed_fraction=informed))

    return [("cb-ele", cbele), ("couzin", baseline("couzin")),
            ("pacnav", baseline("pacnav"))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--goal-knowledge", choices=["all", "informed", "both"],
                        default="both",
                        help="all: every agent steers to the goal; informed: "
                             "only the CB-ELE leader / 10%% informed agents")
    parser.add_argument("--methods", default="cb-ele,couzin")  # pacnav: WIP
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)

    wanted = set(args.methods.split(","))
    settings = ["all", "informed"] if args.goal_knowledge == "both" else [args.goal_knowledge]
    rows = []
    for knowledge in settings:
        for name, run in methods(knowledge):
            if name not in wanted:
                continue
            for seed in range(args.seeds):
                cfg = replace(cb_ele.CONFIG, seed=seed, rounds=args.rounds,
                              verbose_timing=False, show_inline=False)
                t0 = time.perf_counter()
                row = trajectory_metrics(run(cfg), cfg)
                row.update(method=name, goal_knowledge=knowledge, seed=seed)
                rows.append(row)
                print(f"{knowledge:9s} {name:7s} seed {seed:2d}  "
                      f"goal@{row['time_to_goal']}  "
                      f"success={row['success']}  "
                      f"({time.perf_counter() - t0:.1f} s)", flush=True)

    table = pd.DataFrame(rows)
    front = ["goal_knowledge", "method", "seed"]
    table = table[front + [c for c in table.columns if c not in front]]
    os.makedirs(args.output_dir, exist_ok=True)
    per_seed = os.path.join(args.output_dir, "comparison_per_seed.csv")
    table.to_csv(per_seed, index=False, float_format="%.4f")

    summary = table.drop(columns="seed").groupby(
        ["goal_knowledge", "method"], sort=False).agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary_path = os.path.join(args.output_dir, "comparison_summary.csv")
    summary.to_csv(summary_path, float_format="%.4f")
    print(f"\nWritten {per_seed} and {summary_path}")
    means = table.drop(columns="seed").groupby(
        ["goal_knowledge", "method"], sort=False).mean()
    print(means.T.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
