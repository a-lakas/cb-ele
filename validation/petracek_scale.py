import sys, numpy as np
sys.path.insert(0, ".")
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import cb_ele as c, baselines as b
from cb_ele import pairwise_distances
def run(a):
    s, seed = a
    cfg = replace(c.CONFIG, seed=seed, rounds=400)
    traj = b.run_baseline(cfg, b.BaselineConfig(method="petracek", informed_fraction=1.0, petracek_goal_gain=4.0, petracek_length_scale=s))
    nn = []
    for t in range(200, 400, 20):
        d = pairwise_distances(traj["position"][t]); np.fill_diagonal(d, np.inf); nn.append(np.median(d.min(1)))
    return s, seed, float(np.mean(nn))
if __name__ == "__main__":
    tasks = [(s, sd) for s in (125.7, 400, 1000, 2500, 6000) for sd in (1000, 1001)]
    with ProcessPoolExecutor(4) as p:
        for r in p.map(run, tasks): print("s=%7.1f seed %d  median nearest-neighbour distance %.1f m" % r)
