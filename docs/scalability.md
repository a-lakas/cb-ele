# Scalability: swarm size 10–80

## v0.8, 5 seeds per size (current)

**v0.8 fixes the 80-drone launch deadlock described in the v0.7 section below.**

1. **Only a hovering drone counts as a rival** (`rival_requires_hover`).
   - Duelling contenders hover; followers keep moving.
   - A touch by a moving drone no longer resets a candidate's quiet rounds.
   - A duel ends, and the candidate survives, if one round after contact no touching
     drone is hovering.
   - This is the fix that matters. In a dense crowd, followers brushing past kept every
     candidate frozen and its quiet count at zero, so no leader could emerge.
2. **The "leader lost" watchdog runs only after the follower has recognised a leader**
   (`watchdog_requires_leader`). This alone did not fix the deadlock: followers mistook
   duelling candidates for leaders, so the watchdog still fired.

Only CB-ELE was re-run. The baseline code is unchanged, so its rows are the same as in
v0.7. Per-seed results are in `results/scalability_per_seed.csv` (the v0.7 CB-ELE rows are
in `results/scalability_v07_per_seed.csv`); the chart is `results/scalability.png`.

| N | Success: CB-ELE / PACNav / Couzin | Median completion: CB-ELE / PACNav | Order: CB-ELE / PACNav | Rounds split: CB-ELE / PACNav | CB-ELE: one leader at end |
|---|---|---|---|---|---|
| 10 | **100%** / 100% / 100% | **1328** / 1834 | 0.86 / 0.87 | 26% / **5%** | 100% |
| 20 | **100%** / 100% / 0% | **1432** / 1888 | **0.87** / 0.77 | **2%** / 48% | 100% |
| 30 | **100%** / 80% / 0% | **1368** / 1973 | **0.83** / 0.69 | **23%** / 26% | 100% |
| 50 | **100%** / 60% / 0% | **1329** / 1905 | **0.83** / 0.54 | **5%** / 60% | 100% |
| 80 | **100%** / 80% / 0% | **1280** / 1839 | **0.78** / 0.56 | **13%** / 42% | 100% |

CB-ELE breached the 5 m limit 0 times at every size, during the mission and after arrival.

**CB-ELE, v0.7 → v0.8:**

| N | Success | Order | Rounds split | Single-leader share |
|---|---|---|---|---|
| 10 | 100% → 100% | 0.83 → 0.86 | 5% → **26%** (worse) | 85% → **68%** (worse) |
| 20 | 80% → **100%** | 0.84 → 0.87 | 17% → **2%** | 65% → **91%** |
| 30 | 80% → **100%** | 0.81 → 0.83 | 15% → **23%** (worse) | 91% → **74%** (worse) |
| 50 | 100% → 100% | 0.78 → 0.83 | 29% → **5%** | 64% → **74%** |
| 80 | 20% → **100%** | 0.16 → **0.78** | 7% → 13% | 21% → **65%** |

**Findings.**
- CB-ELE now succeeds in every run at every size (25 of 25). It is the fastest method at
  every size, flat at about 1300–1430 rounds, and the best aligned from 20 drones up.
- The trade-off is that because moving drones no longer hold candidates back, more
  candidates promote themselves. On some seeds this gives longer periods with several
  leaders and more splitting: 2 of 5 seeds at 10 and at 30 drones are split for about
  56–67% of rounds.
- With 5 seeds a success rate moves in steps of 20%. These are trends.
- The main 100-seed benchmark (30 drones, `results/comparison_*.csv`) was run before v0.7
  and v0.8, so it does not include these changes.

## v0.7, 5 seeds per size (superseded)

**Command:**
`python compare.py --seeds 5 --rounds 3000 --sizes 10,20,30,50,80 --informed 0.3 --prefix scalability`,
then `python scalability_plot.py results/scalability_per_seed.csv results/scalability.png`.

**Setup:**
- v0.7: the start area grows with N (60 m² per drone), and drones settle inside 100 m of the goal.
- 30% of drones informed; seeds 0–4; 3000 rounds.
- Arrival radius R_g = 60 m × √(N/30).
- Results are in `results/scalability_per_seed.csv`; the chart is `results/scalability.png`.

**5 seeds give trends, not significance:** a success rate moves in steps of 20%.

### Success and completion time

| N | Success: CB-ELE / PACNav / Couzin | Median completion (rounds): CB-ELE / PACNav / Couzin |
|---|---|---|
| 10 | 100% / 100% / 100% | **1587** / 1834 / 1655 |
| 20 | 80% / **100%** / 0% | **1377** / 1888 / – |
| 30 | 80% / 80% / 0% | **1333** / 1973 / – |
| 50 | **100%** / 60% / 0% | **1361** / 1905 / – |
| 80 | 20% / **80%** / 0% | 1331 (1 run) / 1839 / – |

### Coordination and cohesion

| N | Order: CB-ELE / PACNav / Couzin | Rounds split: CB-ELE / PACNav / Couzin |
|---|---|---|
| 10 | 0.83 / **0.87** / 0.52 | 5% / 5% / **3%** |
| 20 | **0.84** / 0.77 / 0.23 | **17%** / 48% / 84% |
| 30 | **0.81** / 0.69 / 0.19 | **15%** / 26% / 82% |
| 50 | **0.78** / 0.54 / 0.14 | **29%** / 60% / 85% |
| 80 | 0.16 / **0.56** / 0.13 | 7%\* / 42% / 83% |

### Safety and leadership

| N | 5 m breaches during the mission: CB-ELE / PACNav / Couzin | 5 m breaches after arrival: CB-ELE / PACNav | CB-ELE: one leader at the end |
|---|---|---|---|
| 10 | **0** / 0 / 1 | **0** / 0 | 100% |
| 20 | **0** / 0 / 14 | **0** / 2 | 80% |
| 30 | **0** / 6 / 34 | **0** / 22 | 100% |
| 50 | **0** / 21 / 15 | **0** / 2276 | 100% |
| 80 | **0** / 19 / 24 | **0** / 4203 | 20% |

\* At N = 80, CB-ELE's low split share and small radius come from a swarm that never
leaves the start (see below). They do not reflect good cohesion.

### Findings

- **10–50 drones:** CB-ELE is the fastest method at every size, the best aligned from 20
  to 50, and splits least from 20 to 50. It never breaches the 5 m limit, during the mission
  or after arrival. Settling at the goal (v0.7) works: 100% end with one leader at 10, 30
  and 50 drones (80% at 20).
- **PACNav** succeeds at every size, but more slowly. It crowds dangerously at the goal for
  large swarms: 2276 and 4203 breaches of the 5 m limit after arrival at 50 and 80 drones.
- **Couzin** succeeds only at 10 drones.
- **At 80 drones CB-ELE fails 4 runs out of 5 because the election never settles at
  launch.**
  - With 80 drones, about 60 pairs are in contact at any time.
  - Without a leader, followers' headings stay misaligned. Their "leader lost" check fires
    (392 times in 400 rounds for seed 1), and each time a follower becomes a challenger and
    then a candidate again.
  - The constant stream of candidates means no candidate gets 6 quiet rounds. Only 3
    leaders were promoted in 400 rounds, and each was knocked out again. The swarm stays
    at the start, with about 20 candidates and no leader.
  - Seed 0 escaped this, which is why the single-seed 80-drone GIF looked fine.
  - A likely fix is to run the "leader lost" check only for followers that have had a
    leader. This is not implemented.

## Earlier single-seed sweep (v0.6, before the v0.7 fixes)

Seed 0 only. The start area was 20 × 90 m at every size, and there was no settling at
the goal.

- CB-ELE was fastest at 20–50 drones.
- At 10–20 drones it was split for 61–70% of rounds.
- At 80 drones it collapsed (order 0.37, 796 breaches of the 5 m limit at the goal).

That sweep motivated the two v0.7 fixes.
