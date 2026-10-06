# Paper review: competitors' setups and metrics

The review was done on 2026-10-06. Full texts were read for PACNav, Petráček 2020,
Horyna 2024, Schilling 2021 and Choi 2026 (arXiv), together with PACNav's
official code (github.com/ctu-mrs/pacnav).

Three papers could not be retrieved in full, because the publisher sites are blocked from
this environment:
- Vásárhelyi 2018 (Science Robotics)
- Couzin 2005 (Nature)
- Zuo 2022 (MDPI)

For these, the notes come from secondary sources, and anything marked *unverified* must be
checked against the original before it is cited. The PDFs are not stored in this repository.

## PACNav (Ahmad et al., Bioinspir. Biomim. 2022)

**Setup**
- **Simulation:** Gazebo, 50 × 50 m forest, goal 20 m away, 10 runs per case.
- **Cases:**
  - N=3 with 1 or 2 informed.
  - N=6 with 2 or 4 informed.
- **Real flight:** 4 UAVs with 1 informed.
- **Sensing:** UVDAR, line-of-sight only.
- **Speed:** v_max = 1 m/s (from the code).

**Metrics**
- Order Ω (eq. 21).
- Minimum, mean and maximum inter-UAV distance.
- Mission completion time, i.e. all UAVs within R_g of g. Means were 189–231 s, and every run succeeded.
- Target switches over time (Fig. 14).

**Algorithm**
- **Whom to follow:** each uninformed UAV picks j* = argmax γ_j + Σ_l σ_jl.
  - γ is path persistence (eq. 6); σ is path similarity (eq. 4).
  - Candidates are neighbours that are ≥ R_f away, have ≥ 3 path samples, and are not approaching the previous target.
- **Informed UAVs:** fly to g, slowing by max(V_m, 1 − d̄/(2R_f)) (eq. 15).
- **Uninformed UAVs:** damp their motion toward close neighbours (Alg. 3, α = 8).
- **Collision avoidance:** rotated repulsion (eqs 16–20).
- **Parameters:** R_f = 4 m, R^o = 2.5 m, K^n = 1.2, K^c = 1.0.
- **Paper inconsistency:** the stated forest density (ρ = 0.4) does not match eq. 22 with the Table 1 values, which give ρ ≈ 0.82.
- **Code vs paper:** the code selects targets incrementally rather than by a global argmax, and differs in the collision-force magnitude.

**Our reimplementation** (`baselines.py`)
- It runs in 2D without obstacles, so A* reduces to a short straight step toward the target.
- Lengths are scaled to our geometry:
  - R_f = 15 m, matching CB-ELE's preferred spacing.
  - R^o = 12 m, matching CB-ELE's avoidance radius.
  - K^c is scaled by (12/2.5)², so the balance between navigation and collision terms is preserved.
- The steering term saturates at v_max = 1.5 m/s before eq. 15's slow-down.

## Petráček et al. (Bioinspir. Biomim. 2020)

**Setup**
- Boids-like 3D model with no communication, using UVDAR (≤ 15 m).
- Simulations with 5, 6 and 9 UAVs; real flights with 3 and 4 UAVs.

**Metrics**
- Average and minimum inter-UAV distance over time.
- Minimum distance to obstacles: 2.2 m and 2.04 m in the real runs.
- Collision-free yes/no.
- Localisation error: RMSE 1.16 m and 1.70 m.
- Minimum distance as a function of localisation noise.

## Horyna et al. (IEEE RA-L 2024, arXiv:2404.18729)

**Setup**
- 6 real UAVs, more than 200 m at 5 m/s, using UVDAR and VIO.
- Desired spacing 13 m.

**Metrics**
- Cluster velocity ratio CVR = ‖ṗ_c‖/v_D (eq. 23): 0.87–0.93.
- Neighbour distance d_n ± σ_d: real 14.3–15.9 ± 4.6–5.5 m; simulation without communication 15.4–15.9 ± 3.6–4.0 m.
- Trajectory length.
- Velocity and position errors.

## Schilling, Schiano, Floreano (IEEE RA-L 2021)

**Setup**
- 3 real quadcopters with vision-only flocking (YOLOv3-tiny detector, about 6 m range).
- v_max = 0.5 m/s.

**Metrics**
- Inter-agent distance: minimum / mean was 1.37–1.82 / 2.32–2.42 m.
- Collision-free yes/no.
- Localisation error versus range.
- Detector AP 98.9%.

## Choi et al. (arXiv:2601.13657, 2026)

**Setup**
- 5 UAVs: 1 informed leader and 4 deep-RL followers using LiDAR (10 m).
- 100 trials in each of 5 environments.

**Metrics**
- SR: success rate.
- MP: mission progress.
- FR: max distance of any UAV from the centroid.
- MS: minimum separation.
- AL: alignment with the mean velocity.
- MDO: minimum distance to obstacles.

**PACNav as a baseline (Choi's results)**
- SR 100 → 31% from open space to dense forest.
- AL 0.87 → 0.73.

## Vásárhelyi et al. (Science Robotics 2018): secondary sources only

- The fitness is a product F_speed·F_coll·F_wall·F_corr·F_disc·F_cluster.
- It uses order parameters for:
  - velocity: φ_vel
  - collisions: φ_coll
  - leaving the arena: φ_wall
  - velocity correlation: φ_corr
  - disconnected agents: φ_disc
  - minimum cluster size: φ_cluster
- 30 real drones.
- *Unverified:* the exact definitions of φ_vel, φ_corr and φ_coll, and the collision radius.

## Couzin et al. (Nature 2005): secondary source (Vicsek & Zafeiris 2012, §5.4)

**Model**
- Repulsion zone: highest priority.
- Combined attraction and alignment zone.
- Informed individuals use d = (d̂ + ωg)/|d̂ + ωg|.

**Metrics**
- Group accuracy: normalised angular deviation from g.
- Elongation and fragmentation (*unverified* exact definitions).

**Our reimplementation**
- Repulsion radius 12 m and interaction radius 35 m.
- ω = 0.5 and speed 1.0 m/s.
- Turn rate limited to 40°/round.

## Zuo et al. (Electronics 2022): snippets only

- Raft-inspired voting election triggered by heartbeat loss.
- 30–100 followers; communication range 200–700 units; at most 50 steps per election.
- **Metric:** election success rate.
- Not implemented here: there is no full text, and the method needs radio.
