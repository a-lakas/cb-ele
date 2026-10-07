# Scalability: swarm size 10–80 (single seed)

**Command:**
`python compare.py --seeds 1 --rounds 3000 --sizes 10,20,30,50,80 --informed 0.3 --prefix scalability`

**Setup:**
- Seed 0, 30% of drones informed, 3000 rounds.
- Arrival radius R_g = 60 m × √(N/30), so the swarm can fit around the goal.
- Results are in `results/scalability_per_seed.csv`.
- **This is one seed per size: an illustration of trends, not a statistical result.**

| N | 10 | 20 | 30 | 50 | 80 |
|---|---|---|---|---|---|
| R_g (m) | 35 | 49 | 60 | 77 | 98 |
| **Success** CB-ELE / PACNav / Couzin | ✓ / ✓ / ✓ | ✓ / ✓ / ✗ | ✓ / ✓ / ✗ | ✓ / ✓ / ✗ | ✗ / ✓ / ✗ |
| **Completion time (rounds)** CB-ELE / PACNav / Couzin | 1571 / 1897 / 1489 | 1348 / 1844 / – | 1305 / 1824 / – | 1493 / 1975 / – | 2891* / 1994 / – |
| **Order** CB-ELE / PACNav / Couzin | 0.77 / 0.87 / 0.56 | 0.83 / 0.79 / 0.22 | 0.90 / 0.71 / 0.19 | 0.77 / 0.59 / 0.09 | 0.37 / 0.53 / 0.11 |
| **Rounds split** CB-ELE / PACNav / Couzin | 61% / 4% / 0% | 70% / 17% / 85% | 0% / 24% / 79% | 0% / 60% / 89% | 86% / 76% / 81% |
| **Swarm radius (m)** CB-ELE / PACNav / Couzin | 80 / 37 / 28 | 159 / 70 / 315 | 39 / 85 / 312 | 80 / 116 / 413 | 435 / 165 / 404 |
| **5 m breaches** CB-ELE / PACNav / Couzin | 0 / 0 / 0 | 0 / 0 / 11 | 0 / 0 / 13 | 7 / 18 / 9 | 796 / 0 / 0 |
| CB-ELE election time (rounds) | 80 | 106 | 86 | 268 | 55 |
| CB-ELE single-leader share of rounds | 34% | 41% | 91% | 84% | 64% |

\* All 80 drones arrived at round 2891, but one contact under 2 m earlier in the run counts
as a failure.

## Trends, with caveats

- **CB-ELE works best at 30–50 drones.** It is the fastest method at every size from 20
  to 50 and the best aligned at 20–50.
- **CB-ELE degrades at both ends.**
  - **10–20 drones:** the swarm splits for 60–70% of the run. With few drones, the chain of
    35 m sensing links breaks easily. It reconnects before the end, but has several leaders
    for most of the run (single-leader share 34–41%).
  - **80 drones:** cohesion and alignment collapse (order 0.37, swarm radius 435 m).
    Crowding at the goal, which sits next to the wall, causes 796 breaches of the 5 m limit,
    all in rounds 2363–2927.
- **PACNav is the most robust:** it succeeds at every size. Its completion time is flat
  (about 1850–2000 rounds), but its cohesion worsens steadily with N: split 4% of rounds
  at 10 drones, 76% at 80.
- **Couzin succeeds only at 10 drones.** With 30% informed it fails from 20 drones upward.

## Next steps

- Confirm with more seeds. 10 per size would take about 30 minutes.
- For large swarms:
  - Scale the start area with N. It is currently fixed at 20 × 90 m.
  - Make the leader's tether and gate depend on swarm size.
- For small swarms: possibly a sensing radius relative to the swarm's spacing.
