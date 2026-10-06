# Common evaluation metrics

These metrics were selected after reviewing the papers in [`paper_review.md`](paper_review.md).
`compare.py` computes all of them the same way for every method. The inputs are the
per-round positions p_i(t), realised velocities v_i(t) and states. Metrics are evaluated
over the **mission window** [0, T_c], or over the whole run if the swarm never completes.

## Cross-paper usage

| Metric | PACNav | Petráček | Horyna | Schilling | Vásárhelyi¹ | Couzin¹ | Zuo² | Choi |
|---|---|---|---|---|---|---|---|---|
| Velocity order / alignment | ✓ Ω | | | | ✓ φ_corr | | | ✓ AL |
| Min inter-agent distance | ✓ | ✓ | | ✓ | (φ_coll) | | | ✓ MS |
| Mean inter-agent distance | ✓ | ✓ | ✓ d_n | ✓ | | | | |
| Swarm radius / max distance | ✓ | | | | | | | ✓ FR |
| Collisions | (free) | (free) | (free) | (free) | ✓ φ_coll | | | ✓ in SR |
| Success rate | (all) | | | | | | ✓ | ✓ SR |
| Completion time | ✓ | | | | | | | |
| Progress / group speed | | | ✓ CVR | | ✓ φ_vel | | | ✓ MP |
| Heading accuracy to goal | | | | | | ✓ | | |
| Fragmentation / disconnection | | | | | ✓ φ_disc | ✓ | | |
| Election time / leader metrics | | | | | | | ✓ | |

¹ Full text could not be retrieved; the metrics come from secondary sources. ² Search snippets only.

## Selected metrics

Notation:
- p_c = centroid
- g = goal
- d_ij = ‖p_i − p_j‖
- R_g = 60 m (arrival radius: a compact half-disc for 30 drones at 15 m spacing, since the goal is 5 m from the wall)
- R_s = 35 m (sensing radius)

| Column | Definition | Source |
|---|---|---|
| `success` | All agents within R_g of g at some t ≤ T, with no collision before then | PACNav completion; Choi SR |
| `completion_time` | T_c = first round with all agents within R_g | PACNav |
| `mission_progress` | 1 − ‖p_c(T_c) − g‖ / ‖p_c(0) − g‖ | Choi MP (on the centroid) |
| `order` | Ω = 1/(N(N−1)) Σ_{i≠j} v̂_i·v̂_j, averaged over time; moving agents only | PACNav eq. 21 |
| `heading_accuracy` | 1 − ∠(ṗ_c, g − p_c)/π, averaged over time | Couzin accuracy (normalisation ours) |
| `cluster_velocity_ratio` | ‖ṗ_c‖ / v_max, averaged over time | Horyna eq. 23 |
| `min_distance_mean`, `min_distance_overall` | min_{i≠j} d_ij: time average (Choi MS) and overall minimum | Choi, PACNav, Petráček, Schilling |
| `collisions` | Σ_t #{i<j : d_ij < 2 m}, i.e. physical contact of the 2 m drones | Choi; φ_coll |
| `safety_breaches` | Σ_t #{i<j : d_ij < 5 m − 1 cm}, i.e. the hard-safety envelope violated | ours |
| `swarm_radius` | max_i ‖p_i − p_c‖, averaged over time | Choi FR |
| `mean_pair_distance` | 2/(N(N−1)) Σ_{i<j} d_ij, averaged over time | PACNav, Petráček, Schilling, Horyna |
| `fragmented_fraction` | Fraction of rounds in which the R_s-graph has more than one component | Couzin fragmentation; Vásárhelyi φ_cluster |
| `isolated_fraction` | Time-averaged fraction of agents with no neighbour within R_s | Vásárhelyi φ_disc |
| `fragmented_at_end` | The R_s-graph is disconnected at T_c (or at the end of the run) | Couzin |
| `election_time` | First round from which exactly one leader is held for 30 rounds (CB-ELE only) | Zuo |
| `unique_leader_fraction` | Fraction of rounds with exactly one leader (CB-ELE only) | ours |
| `leader_changes` | Number of rounds in which the leader set changes (CB-ELE only) | ours |

Messages exchanged are **0 by construction** for CB-ELE, PACNav and Couzin.
This is the main contrast with radio-based elections such as Zuo 2022 or Raft.

## Comparison protocol

- **Trials:** 100 seeds per method and setting, as in Choi et al. For every method, a given seed produces identical start positions and headings.
- **Informed fraction p:**
  - p = 1.0: all agents know the goal.
  - p = 0.3: about PACNav's 1/3 case.
  - For CB-ELE, p applies to the followers; the elected leader always uses the goal.
- **Shared physics:** every method uses the same sensing radius (35 m), max speed (1.5 m/s) and hard-safety projection (5 m).
