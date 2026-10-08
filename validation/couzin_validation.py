"""Validate the Couzin et al. (2005) rule in baselines.py against the
paper's published findings, in the paper's own (dimensionless) setting:
unbounded plane, cohesive start, fixed preferred direction g.

Checks (as restated by Vicsek & Zafeiris 2012, sec. 5.4):
  V1  for fixed N, group accuracy increases (asymptotically) with the
      informed proportion p;
  V2  larger groups need a smaller proportion p for a given accuracy.

Parameters (Couzin 2005, from secondary sources; unverified against the
original): alpha = 1, rho = 6, speed s = 1, max turning rate 2 rad/s,
noise sd 0.01 rad, dt = 0.1 s, omega = 0.5.

Accuracy = 1 - |angle(group direction, g)| / pi, group direction = centroid
displacement over the last 20% of the run, averaged over replicates.

    python validation/couzin_validation.py OUT.csv
"""

import sys
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from baselines import _couzin_desired, _turn_towards
from cb_ele import pairwise_distances, components

ALPHA, RHO, SPEED, TURN, NOISE, DT, OMEGA = 1.0, 6.0, 1.0, 2.0, 0.01, 0.1, 0.5
STEPS = 1500
G = np.array([1.0, 0.0])                      # preferred direction


def run(task):
    n, p, rep = task
    rng = np.random.default_rng(10_000 * n + 1000 * int(round(100 * p)) + rep)
    # cohesive start: random positions in a disc of density ~1 per unit area
    r = np.sqrt(rng.uniform(0, 1, n)) * np.sqrt(n / np.pi)
    a = rng.uniform(-np.pi, np.pi, n)
    pos = np.column_stack((r * np.cos(a), r * np.sin(a)))
    heading = rng.uniform(-np.pi, np.pi, n)
    informed = np.zeros(n, dtype=bool)
    informed[rng.choice(n, int(round(p * n)), replace=False)] = True
    cfg = SimpleNamespace(sensing_radius=RHO)
    bcfg = SimpleNamespace(repulsion_radius=ALPHA, couzin_omega=OMEGA)
    far_goal = lambda: np.tile(pos.mean(axis=0) + 1e9 * G, (n, 1))  # unit(goal_i - x_i) = g
    track = []
    for _ in range(STEPS):
        d = pairwise_distances(pos)
        np.fill_diagonal(d, np.inf)
        nb = d <= RHO
        goal = far_goal()
        desired = np.array([_couzin_desired(i, pos, heading, d, nb, informed,
                                            goal, cfg, bcfg) for i in range(n)])
        ang = np.arctan2(desired[:, 1], desired[:, 0]) + rng.normal(0, NOISE, n)
        heading = _turn_towards(heading, ang, TURN * DT)
        pos = pos + SPEED * DT * np.column_stack((np.cos(heading), np.sin(heading)))
        track.append(pos.mean(axis=0))
    track = np.array(track)
    disp = track[-1] - track[int(0.8 * STEPS)]
    err = abs(np.arctan2(disp[1], disp[0]))
    d = pairwise_distances(pos); np.fill_diagonal(d, np.inf)
    fragmented = len(components(d <= RHO)) > 1
    return dict(n=n, p=p, rep=rep, accuracy=1 - err / np.pi, fragmented=fragmented)


if __name__ == "__main__":
    out = sys.argv[1]
    tasks = [(n, p, rep) for n in (10, 30, 100) for p in (0.0, 0.05, 0.1, 0.2, 0.4, 0.6)
             for rep in range(10)]
    with ProcessPoolExecutor(4) as ex:
        df = pd.DataFrame(list(ex.map(run, tasks)))
    df.to_csv(out, index=False)
    acc = df.groupby(["n", "p"]).accuracy.mean().unstack("n")
    frag = df.groupby(["n", "p"]).fragmented.mean().unstack("n")
    print("Accuracy (rows p, columns N):"); print(acc.round(3).to_string())
    print("\nFragmentation probability:"); print(frag.round(2).to_string())
