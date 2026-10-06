# CB-ELE — Collision-Based Leader Election for GPS/communication-denied UAV swarms

Version **v0.0**: baseline import of the CB-ELE v4.4 simulator (collision election,
mid-distance re-election experiment, follower catch-up boost).

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

A full 3000-round run takes about 20 s of simulation + about 4.5 min of GIF rendering on one CPU.

## Self-hosted GitHub Actions runner

`.github/workflows/simulate.yml` runs the simulation on a `self-hosted` runner
(manual trigger: *Actions → CB-ELE simulation → Run workflow*) and uploads the CSV and
GIF as a build artifact.

## Reference run

`results/` holds the v0.0 reference run (seed 42, 3000 rounds, 30 agents).
