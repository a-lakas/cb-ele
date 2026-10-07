# Behavioural leader recognition (v0.5)

**Question.** With no communication, no IDs exchanged and no visual "I am leader" signal,
can followers tell which drone is the leader just by watching how it behaves under the
election protocol? If they can, does following it help?

**Answer.** Yes. Over 100 seeds, recognition from behaviour alone gets the protocol to a
single leader about 10× faster. Without goal-aware followers, it also raises success from
45% to 73%.

## What a drone observes

The protocol makes contenders (candidates and leaders) behave visibly differently from
followers. Every drone keeps a score `score[i, j]` for each neighbour j it can currently
sense (within 35 m). The score is updated once per round from the last motion step:

| Observed event | Who shows it | Score change |
|---|---|---|
| j was touched (another drone within 8 m) and **froze** (hovered) | contenders: candidates and leaders | +1 |
| j was touched and **backed away** | followers, passive drones | −2 |
| j and k hovered together, then **k left and j stayed** (j won a duel) | the duel winner | +3 (counted only if i also sees k) |

- **Decay:** each round the score is multiplied by 0.995, so it halves after about 140 rounds.
- **Limits:** the score is clipped to [0, 50].
- **Belief:** drone i's believed leader is the visible neighbour with the highest score, provided it is at least 2.

**Assumption:** drones can track individual neighbours over time, as UVDAR or visual
multi-target trackers do; PACNav assumes the same. Nothing is transmitted. Beliefs are
private and cannot be observed by other drones.

## How the belief is used

With `leader_recognition = "follow"`, the default:

- **Followers** add a leader term to their flocking rule:
  - a pull toward the believed leader, which grows once it is farther than the preferred
    15 m spacing;
  - alignment with the leader's heading.
- **Leaders** that believe another visible drone is a leader fly toward it. The resulting
  contact triggers a duel, which removes the duplicate. Without this, duplicate leaders only
  met at the goal (v0.4).

## Hypothesis test

`recognition_test.py`: 100 seeds, 3000 rounds, 30 drones; followers 0% or 30% informed.

| | uninformed, off | uninformed, **follow** | 30% informed, off | 30% informed, **follow** |
|---|---|---|---|---|
| Success | 45% | **73%** | 84% | **95%** |
| Completion time (median rounds) | 1757 | **1392** | 1472 | **1333** |
| Rounds with the swarm split | 48% | **20%** | 30% | **11%** |
| Split at the end | 48% | **22%** | 14% | **5%** |
| Exactly one leader (% of rounds) | 46% | **85%** | 40% | **81%** |
| Time until one leader holds for 30 rounds | 1111 | **96** | 796 | **113** |
| Recognition precision / recall | n/a | 0.75 / 0.50 | n/a | 0.78 / 0.64 |

Per-seed results are in `results/recognition_test.csv`.

**Reading the numbers:**
- Precision means: when a follower names a leader, it is right 75–78% of the time.
- Recall means: of the followers that can see the true leader, 50–64% identify it.
- The most common error is a recently demoted leader, which still "looks like" a leader
  until its score decays.

## Limits and open points

- **Uninformed followers still fail in 27% of runs.** Most failures end with the swarm split.
  Recall is the bottleneck: followers that never see the leader rely on alignment with
  their neighbours.
- **The parameters were set on two seeds and not tuned further.** Decay and threshold were
  chosen from a small sweep on seeds 42 and 0.
- **The weights are hand-set.** Learning them, or using a likelihood model of the
  protocol's states, could raise recall.

## v0.6: slowness evidence

**Which behaviours give the leader away?** `recognition_features.py` measures, per drone
and per round, behaviours that any neighbour could observe. It then scores how well each
one separates leaders from followers (AUC: 0.5 means no information, 1 means perfect).
Results are from 12 runs, followers 0% and 30% informed, saved in
`results/recognition_features.csv`.

| Observable behaviour (long-memory average) | AUC, leader vs follower |
|---|---|
| **Moves slower than its neighbours** (the leader waits for the swarm) | **0.95** |
| Freezes when touched | 0.95 |
| Never backs away when touched | 0.93 |
| Does not yield to drones within 12 m | 0.78 |
| Flies toward its neighbours | 0.77 |
| Heading persistence; position at the front of the swarm | weak (leaders are, if anything, behind) |

Slowness comes from the leader's own rules: it slows down when the swarm lags behind
(gate) or when its nearest neighbour is far (tether). Unlike freezing, it is visible
**every round**, not only when the leader is touched.

**What changed in v0.6.**
- Each round, every visible neighbour j earns
  `0.5 · (mean speed of the observer's visible neighbours − speed of j) / v_max`.
- The belief threshold rises from 2 to 5.

### Result

`recognition_compare.py`, 100 seeds, saved in `results/recognition_v05_vs_v06.csv`.

| | uninformed v0.5 | uninformed v0.6 | 30% informed v0.5 | 30% informed v0.6 |
|---|---|---|---|---|
| Recognition precision | 0.75 | **0.86** | 0.78 | **0.86** |
| Recognition recall | 0.50 | **0.88** | 0.64 | **0.89** |
| Success | 73% | 78% | 95% | 97% |
| Split at the end | 22% | 15% | 5% | 3% |
| Rounds with the swarm split | 20% | 16% | 11% | 12% |
| Median completion time (rounds) | 1392 | 1412 | 1333 | 1341 |
| Time until one leader holds (rounds) | 96 | 129 | 113 | 124 |

- **Recognition improves a lot:** recall rises by 0.25–0.38 and precision by about 0.1.
- **Mission outcomes do not change significantly.** These are paired seed-by-seed tests:

  | Followers | Success flips (fail→success / success→fail) | Success p | Split-time p (Wilcoxon) |
  |---|---|---|---|
  | uninformed | +24 / −19 | 0.54 | 0.44 |
  | 30% informed | +5 / −3 | 0.73 | 0.87 |

- **Conclusion:** followers now find the leader correctly about 9 times in 10, so
  recognition is no longer the bottleneck. The remaining failures, mostly a swarm that ends
  split when followers are uninformed, come from elsewhere: drones that are out of the
  leader's sensing range, and the leader's own pace.
- v0.6 stays the default because the recognition claim is much stronger. The cost is a
  slightly later single leader.
