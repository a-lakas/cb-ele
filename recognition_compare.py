"""Recognition v0.5 (contact evidence) vs v0.6 (+ slowness evidence), 100
seeds, followers 0%/30% informed.  Usage: python recognition_compare.py SEEDS OUT.csv"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import cb_ele as c
from compare import trajectory_metrics
VARIANTS = {"v0.5": dict(recognition_slow_weight=0.0, recognition_threshold=2.0),
            "v0.6": dict(recognition_slow_weight=0.5, recognition_threshold=5.0)}
def run(a):
    name, inf, seed = a
    cfg = replace(c.CONFIG, seed=seed, rounds=3000, verbose_timing=False,
                  follower_informed_fraction=inf, **VARIANTS[name])
    _, _, t, e, tr, traj = c.run_simulation(cfg, return_trajectory=True)
    r = trajectory_metrics(traj, cfg, 60.0)
    r.update(variant=name, informed=inf, seed=seed,
             precision=np.nanmean(t.recognition_precision), recall=np.nanmean(t.recognition_recall))
    return r
if __name__ == "__main__":
    seeds = int(sys.argv[1])
    tasks = [(v, f, s) for f in (0.0, 0.3) for v in VARIANTS for s in range(seeds)]
    with ProcessPoolExecutor(4) as p:
        df = pd.DataFrame(list(p.map(run, tasks)))
    df.to_csv(sys.argv[2], index=False)
    cols = ["success", "completion_time", "order", "fragmented_fraction", "fragmented_at_end",
            "unique_leader_fraction", "election_time", "precision", "recall"]
    print(df.groupby(["informed", "variant"])[cols].mean().T.round(3).to_string())
    print(df.groupby(["informed", "variant"]).completion_time.median())
