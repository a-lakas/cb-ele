# CB-ELE — Collision-Based Leader Election for GPS/communication-denied UAV swarms

Current version **v0.5** (simulator CB-ELE v4.4: collision election,
mid-distance re-election experiment, follower catch-up boost).

| version | changes |
|---|---|
| v0.5 | behavioural leader recognition: followers infer and follow the leader from observed protocol behaviour ([details](docs/leader_recognition.md)) |
| v0.4 | fully local duel resolution (no central arbitration, no global leader count) |
| v0.3 | PACNav and Couzin baselines, common metrics from the paper review, 100-seed benchmark, per-baseline animation scripts |
| v0.2 | re-election candidates seeded on the goal side of the swarm (no leader trailing / swarm split); 1350 rounds |
| v0.1 | re-election no longer re-seeds the demoted leader; 1700 rounds; GIF figure height doubled |
| v0.0 | baseline import, headless server execution |

## Run locally / on a server

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python cb_ele.py --output-dir results
```

Outputs (in `--output-dir`):

| file | content |
|---|---|
| `cb_ele_metrics.csv` | per-round metrics (states, connectivity, safety, goal progress) |
| `cb_ele_animation.gif` | swarm animation + state / coordination plots |

Runs headless (Matplotlib `Agg` backend) outside Jupyter. Inside a notebook, the
animation is also shown inline.

Useful flags: `--rounds`, `--seed`, `--n-agents`, `--no-reelection`,
`--no-animation` (CSV only), `--animation-file x.mp4` (needs ffmpeg),
`--sweep N` (seed sweep → `cb_ele_sweep.csv`).

A full 1350-round run takes about 10 s of simulation + about 3 min of GIF rendering on one CPU.

## Self-hosted GitHub Actions runner

`.github/workflows/simulate.yml` runs the simulation on a `self-hosted` runner
(manual trigger: *Actions → CB-ELE simulation → Run workflow*) and uploads the CSV and
GIF as a build artifact.

## Benchmark against baselines

```bash
.venv/bin/python compare.py --seeds 100 --rounds 3000 --informed 1.0,0.3
```

This runs CB-ELE, PACNav (Ahmad et al. 2022) and Couzin et al. (2005) on identical starts
(`baselines.py`), and writes `comparison_per_seed.csv` and `comparison_summary.csv`.
Metric definitions are in [`docs/metrics.md`](docs/metrics.md), and the paper notes are in
[`docs/paper_review.md`](docs/paper_review.md).

## Related work

See [`docs/related_work.md`](docs/related_work.md) for the competitor literature and [`docs/literature_questions.md`](docs/literature_questions.md) for why leaders, GPS-free navigation and collision-based election prior art.

## Reference run

`results/` holds the v0.0 reference run (seed 42, 1350 rounds, 30 agents).
