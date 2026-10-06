"""Leader-recognition hypothesis test: recognition off vs follow, followers
0% and 30% informed.  Usage: python recognition_test.py SEEDS OUT.csv"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import cb_ele as c
from compare import trajectory_metrics

def run(task):
    mode, inf, seed = task
    cfg = replace(c.CONFIG, seed=seed, rounds=3000, verbose_timing=False,
                  leader_recognition=mode, follower_informed_fraction=inf)
    _, _, t, e, tr, traj = c.run_simulation(cfg, return_trajectory=True)
    row = trajectory_metrics(traj, cfg, 60.0)
    row.update(mode=mode, informed=inf, seed=seed,
               precision=np.nanmean(t.recognition_precision) if mode != "off" else np.nan,
               recall=np.nanmean(t.recognition_recall) if mode != "off" else np.nan)
    return row

if __name__ == "__main__":
    seeds = int(sys.argv[1]); out = sys.argv[2]
    tasks = [(m, f, s) for f in (0.0, 0.3) for m in ("off", "follow") for s in range(seeds)]
    with ProcessPoolExecutor(4) as p:
        rows = list(p.map(run, tasks))
    df = pd.DataFrame(rows); df.to_csv(out, index=False)
    cols = ["success", "completion_time", "mission_progress", "order", "fragmented_fraction",
            "fragmented_at_end", "unique_leader_fraction", "election_time", "precision", "recall"]
    print(df.groupby(["informed", "mode"])[cols].mean().T.round(3).to_string())
