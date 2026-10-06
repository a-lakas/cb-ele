# CB-ELE — Collision-Based Leader Election for GPS/communication-denied UAV swarms

Current version **v0.1** (simulator CB-ELE v4.4: collision election,
mid-distance re-election experiment, follower catch-up boost).

| version | changes |
|---|---|
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

A full 1700-round run takes about 12 s of simulation + about 3.5 min of GIF rendering on one CPU.

## Self-hosted GitHub Actions runner

`.github/workflows/simulate.yml` runs the simulation on a `self-hosted` runner
(manual trigger: *Actions → CB-ELE simulation → Run workflow*) and uploads the CSV and
GIF as a build artifact.

## Reference run

`results/` holds the v0.0 reference run (seed 42, 1700 rounds, 30 agents).
