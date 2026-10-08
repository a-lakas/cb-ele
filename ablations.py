"""CB-ELE ablations (10 seeds each, 30 drones).

  crash        leader crash at round 300 (the crashed drone is removed), with
               and without leader-vanish detection; recovery = rounds until one
               leader holds for 30 rounds
  recognition  behavioral leader recognition off vs. follow, p in {0, 0.3}

    python ablations.py --seeds 10 --output-dir results
"""

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

import cb_ele
from compare import trajectory_metrics

warnings.filterwarnings("ignore", category=RuntimeWarning)
CRASH_ROUND = 300


def recovery_time(leader_count, crash, hold=30):
    one = leader_count == 1
    for r in range(crash + 1, len(one) - hold):
        if one[r:r + hold].all():
            return r - crash
    return np.nan


def run(task):
    experiment, variant, informed, seed, rounds = task
    cfg = replace(cb_ele.CONFIG, seed=seed, rounds=rounds, verbose_timing=False,
                  show_inline=False, follower_informed_fraction=informed)
    if experiment == "crash":
        cfg = replace(cfg, leader_failure_round=CRASH_ROUND,
                      leader_vanish_detection=(variant == "vanish detection"))
    else:
        cfg = replace(cfg, leader_recognition=variant)
    _, _, table, events, _, traj = cb_ele.run_simulation(cfg, return_trajectory=True)
    row = trajectory_metrics(traj, cfg, 60.0)
    row.update(experiment=experiment, variant=variant, informed_fraction=informed,
               seed=seed,
               recognition_precision=float(np.nanmean(table.recognition_precision)),
               recognition_recall=float(np.nanmean(table.recognition_recall)))
    if experiment == "crash":
        row["recovery_time"] = recovery_time(table.leader_count.to_numpy(), CRASH_ROUND)
        row["vanish_events"] = int((events.event == "leader_vanished").sum())
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)
    tasks = [("crash", v, 0.3, s, args.rounds)
             for v in ("vanish detection", "no vanish detection")
             for s in range(args.seeds)]
    tasks += [("recognition", v, p, s, args.rounds)
              for p in (0.0, 0.3) for v in ("follow", "off")
              for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        table = pd.DataFrame(list(pool.map(run, tasks)))
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "ablations_per_seed.csv")
    table.to_csv(path, index=False, float_format="%.4f")
    cols = ["success", "completion_time", "order", "fragmented_fraction",
            "fragmented_at_end", "unique_leader_fraction", "election_time",
            "recognition_precision", "recognition_recall"]
    extra = [c for c in ("recovery_time", "vanish_events") if c in table]
    print(table.groupby(["experiment", "informed_fraction", "variant"])[cols + extra]
          .mean().round(3).T.to_string())
    print(f"\nWritten {path}")


if __name__ == "__main__":
    main()
