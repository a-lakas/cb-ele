import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import cb_ele as c, baselines as b
from compare import trajectory_metrics
def run(a):
    gain, inf, seed = a
    cfg = replace(c.CONFIG, seed=seed, rounds=3000)
    traj = b.run_baseline(cfg, b.BaselineConfig(method="petracek", informed_fraction=inf, petracek_goal_gain=gain))
    m = trajectory_metrics(traj, cfg, 60.0); m.update(gain=gain, informed=inf, seed=seed); return m
if __name__ == "__main__":
    tasks = [(g, f, s) for g in (0.5, 1.5, 4.0, 10.0) for f in (0.3, 1.0) for s in range(1000, 1005)]
    with ProcessPoolExecutor(4) as p:
        df = pd.DataFrame(list(p.map(run, tasks)))
    cols = ["success", "completion_time", "mission_progress", "order", "fragmented_fraction", "swarm_radius", "min_distance_mean", "safety_breaches", "collisions"]
    print(df.groupby(["informed", "gain"])[cols].mean().round(3).to_string())
