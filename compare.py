"""Benchmark CB-ELE against baselines on common metrics, over many seeds.

    python compare.py --seeds 100 --rounds 3000 --informed 1.0,0.3

Every method sees the same arena, start state (per seed), sensing radius,
hard-safety projection and goal.  Metrics are computed from per-round
trajectories only (positions, realised velocities, states), so they are
identical for every method; definitions and sources are in
`docs/metrics.md`.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

import cb_ele
import baselines
from cb_ele import pairwise_distances, components, LEADER


METHODS = ("cb-ele", "couzin", "pacnav")


def trajectory_metrics(traj, config, arrival_radius, hold_rounds=30):
    """Common metrics (docs/metrics.md) from a trajectory dict."""
    pos, vel, goal = traj["position"], traj["velocity"], traj["goal"]
    state = traj["state"]
    rounds, n, _ = pos.shape
    centroid = pos.mean(axis=1)
    agent_goal = np.linalg.norm(pos - goal, axis=2)
    all_arrived = (agent_goal <= arrival_radius).all(axis=1)

    pair_min = np.empty(rounds)
    pair_mean = np.empty(rounds)
    collisions = np.empty(rounds, dtype=int)
    breaches = np.empty(rounds, dtype=int)
    n_components = np.empty(rounds, dtype=int)
    isolated = np.empty(rounds)
    iu = np.triu_indices(n, 1)
    for t in range(rounds):
        d = pairwise_distances(pos[t])
        np.fill_diagonal(d, np.inf)
        pair_min[t] = d.min()
        pair_mean[t] = d[iu].mean()
        collisions[t] = int((d[iu] < config.drone_span).sum())          # contact
        breaches[t] = int((d[iu] < config.hard_safety_distance - 0.01).sum())
        adjacency = d <= config.sensing_radius
        n_components[t] = len(components(adjacency))
        isolated[t] = float((~adjacency.any(axis=1)).mean())

    # Completion: first round all agents are within R_g (PACNav, Choi SR).
    done = np.flatnonzero(all_arrived)
    t_c = int(done[0]) if len(done) else None
    end = t_c + 1 if t_c is not None else rounds      # mission window
    success = t_c is not None and collisions[:end].sum() == 0

    # Order: mean pairwise velocity cosine (PACNav eq. 21).
    speed = np.linalg.norm(vel, axis=2)
    unit_vel = vel / np.maximum(speed, 1e-9)[..., None]
    moving = speed > 1e-6
    order = np.full(rounds, np.nan)
    for t in range(end):
        u = unit_vel[t, moving[t]]
        m = len(u)
        if m > 1:
            s = u.sum(axis=0)
            order[t] = (s @ s - m) / (m * (m - 1))

    # Group heading accuracy and cluster velocity ratio (Couzin; Horyna eq. 23).
    c_vel = np.diff(centroid, axis=0)[: end - 1]
    to_goal = (goal - centroid)[: end - 1]
    cos_err = np.sum(c_vel * to_goal, axis=1) / np.maximum(
        np.linalg.norm(c_vel, axis=1) * np.linalg.norm(to_goal, axis=1), 1e-12)
    accuracy = 1.0 - np.arccos(np.clip(cos_err, -1.0, 1.0)) / np.pi
    cvr = np.linalg.norm(c_vel, axis=1) / config.max_speed

    swarm_radius = np.linalg.norm(pos - centroid[:, None, :], axis=2).max(axis=1)
    progress = 1.0 - (np.linalg.norm(centroid[end - 1] - goal)
                      / np.linalg.norm(centroid[0] - goal))

    row = {
        "success": bool(success),
        "completion_time": t_c if t_c is not None else np.nan,
        "mission_progress": float(progress),
        "order": float(np.nanmean(order[:end])),
        "heading_accuracy": float(np.mean(accuracy)),
        "cluster_velocity_ratio": float(np.mean(cvr)),
        "min_distance_mean": float(pair_min[:end].mean()),
        "min_distance_overall": float(pair_min[:end].min()),
        "collisions": int(collisions[:end].sum()),
        "safety_breaches": int(breaches[:end].sum()),
        "swarm_radius": float(swarm_radius[:end].mean()),
        "mean_pair_distance": float(pair_mean[:end].mean()),
        "fragmented_fraction": float((n_components[:end] > 1).mean()),
        "isolated_fraction": float(isolated[:end].mean()),
        "fragmented_at_end": bool(n_components[end - 1] > 1),
    }

    # Election metrics: only meaningful for an elected leader (CB-ELE).
    leaders = (state == LEADER).sum(axis=1)
    unique = leaders == 1
    run = np.convolve(unique.astype(int), np.ones(hold_rounds, dtype=int), "valid")
    held = np.flatnonzero(run == hold_rounds)
    row["election_time"] = int(held[0]) if len(held) else np.nan
    row["unique_leader_fraction"] = float(unique[:end].mean())
    # After arrival (v0.7): does the swarm stay safe and settled at the goal?
    row["post_arrival_breaches"] = (int(breaches[end:].sum())
                                    if t_c is not None else np.nan)
    row["single_leader_at_end"] = bool(leaders[-1] == 1)
    row["leader_changes"] = int(np.count_nonzero(
        np.any(np.diff(state == LEADER, axis=0), axis=1)))
    return row


def run_one(task):
    method, informed, seed, rounds, arrival_radius, n_agents = task
    cfg = replace(cb_ele.CONFIG, seed=seed, rounds=rounds, n_agents=n_agents,
                  verbose_timing=False, show_inline=False)
    t0 = time.perf_counter()
    if method == "cb-ele":
        cfg = replace(cfg, follower_informed_fraction=informed)
        traj = cb_ele.run_simulation(cfg, return_trajectory=True)[-1]
    else:
        traj = baselines.run_baseline(cfg, baselines.BaselineConfig(
            method=method, informed_fraction=informed))
    row = trajectory_metrics(traj, cfg, arrival_radius)
    if method != "cb-ele":
        for key in ("election_time", "unique_leader_fraction", "leader_changes",
                    "single_leader_at_end"):
            row[key] = np.nan
    row.update(method=method, informed_fraction=informed, seed=seed,
               n_agents=n_agents, arrival_radius=arrival_radius, runtime_s=time.perf_counter() - t0)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--informed", default="1.0,0.3",
                        help="comma-separated informed fractions")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--arrival-radius", type=float, default=60.0,
                        help="R_g: all agents within it = completion")
    parser.add_argument("--sizes", default=str(cb_ele.CONFIG.n_agents),
                        help="comma-separated swarm sizes; R_g scales as "
                             "arrival_radius * sqrt(N / 30) so the swarm fits")
    parser.add_argument("--prefix", default="comparison",
                        help="output file prefix")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)

    tasks = [(m, float(p), s, args.rounds,
              args.arrival_radius * np.sqrt(int(n) / 30.0), int(n))
             for n in args.sizes.split(",")
             for p in args.informed.split(",")
             for m in args.methods.split(",")
             for s in range(args.seeds)]
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, row in enumerate(pool.map(run_one, tasks), 1):
            rows.append(row)
            if k % 5 == 0 or k == len(tasks):        # save progress as we go
                os.makedirs(args.output_dir, exist_ok=True)
                pd.DataFrame(rows).to_csv(os.path.join(
                    args.output_dir, f"{args.prefix}_partial.csv"), index=False)
            if k % 10 == 0 or k == len(tasks):
                print(f"[{k}/{len(tasks)}] {time.perf_counter() - t0:.0f} s",
                      flush=True)

    table = pd.DataFrame(rows)
    front = ["n_agents", "informed_fraction", "method", "seed"]
    table = table[front + [c for c in table.columns if c not in front]]
    os.makedirs(args.output_dir, exist_ok=True)
    per_seed = os.path.join(args.output_dir, f"{args.prefix}_per_seed.csv")
    table.to_csv(per_seed, index=False, float_format="%.4f")

    grouped = table.drop(columns="seed").groupby(
        ["n_agents", "informed_fraction", "method"], sort=False)
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary_path = os.path.join(args.output_dir, f"{args.prefix}_summary.csv")
    summary.to_csv(summary_path, float_format="%.4f")
    partial = os.path.join(args.output_dir, f"{args.prefix}_partial.csv")
    if os.path.exists(partial):
        os.remove(partial)
    print(f"\nWritten {per_seed} and {summary_path}\n")
    print(grouped.mean().T.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
