# Literature notes: why a leader, GPS-free goal navigation, collision-based election

Compiled 2026-10-06. References were checked on arXiv, publisher or repository pages, or
in independent search results. Markers:
- **(DOI unverified)**: the paper exists, but its DOI could not be resolved from this environment.
- **(partly verified)**: some metadata (for example the full author list) is unconfirmed.

## 1. Do swarms need a leader–follower model?

**Consensus view:**
- Leaders are **not needed for coherent motion**. Local rules alone produce alignment and
  cohesion, and leaderless designs avoid a single point of failure and scale better.
- Leaders become **necessary or very useful when information is unevenly distributed**,
  e.g. only some agents know the goal, route, map, or have GNSS.
- Leadership in animals is mostly **emergent, shared and context-dependent**.
- Recent UAV work therefore uses **hybrid designs**: a leaderless flocking layer with a few
  informed or elected leaders, plus re-election so the leader is not a single point of failure.

### Leaderless suffices for coherent motion

- Reynolds, "Flocks, herds and schools", SIGGRAPH 1987 (DOI unverified). Separation,
  alignment and cohesion with no leader.
- Vicsek et al., "Novel type of phase transition in a system of self-driven particles",
  *PRL* 75:1226, 1995, DOI 10.1103/PhysRevLett.75.1226.
- Jadbabaie, Lin & Morse, *IEEE TAC* 48(6):988–1001, 2003, DOI 10.1109/TAC.2003.812781.
  Proves leaderless heading consensus converges under connectivity, and that a
  leader–follower variant converges to the leader's heading.
- Şahin, LNCS 3342, 2005, DOI 10.1007/978-3-540-30552-1_2. Brambilla et al., *Swarm
  Intelligence* 7(1):1–41, 2013, DOI 10.1007/s11721-012-0075-2. Both credit robustness,
  scalability and flexibility to decentralised control.
- Vásárhelyi et al., *Science Robotics* 3(20):eaat3536, 2018, DOI 10.1126/scirobotics.aat3536.
  30 real drones flocking with no leader. Note that it uses GPS and radio.

### Leaders matter when information is uneven

- Couzin et al., *Nature* 433:513–516, 2005, DOI 10.1038/nature03236. A few informed
  individuals steer a naive group, and larger groups need a smaller informed fraction.
- Couzin et al., *Science* 334:1578–1580, 2011, DOI 10.1126/science.1210280.
  Uninformed members dampen an opinionated minority.
- Olfati-Saber, *IEEE TAC* 51(3):401–420, 2006. Without navigational feedback (a virtual
  leader), flocks tend to fragment.
- Rahmani, Ji, Mesbahi & Egerstedt, *SIAM J. Control Optim.* 48(1):162–186, 2009,
  DOI 10.1137/070688863. Where the leaders sit in the graph decides whether the swarm is
  controllable at all.
- Çelikkanat & Şahin, *Neural Comput. Appl.* 19(6):849–865, 2010,
  DOI 10.1007/s00521-010-0355-y. Informed robots steer a real flock.
- Gupta et al., "SwarmHawk", arXiv:2206.08874, 2022. Drones that lose GNSS switch to
  following assigned leaders visually: a sensor-rich-leader case.

### Animals: leadership is emergent and context-dependent

- Nagy et al., *Nature* 464:890–893, 2010, DOI 10.1038/nature08891. Pigeon flocks have a
  ranked, shared leadership hierarchy.
- Pettit et al., *Curr. Biol.* 25:3132–3137, 2015 (DOI unverified). The fastest birds become
  leaders, and leaders learn routes faster.
- Strandburg-Peshkin et al., *Science* 348:1358–1361, 2015, DOI 10.1126/science.aaa5099.
  Baboons follow initiators, not dominant individuals.

### Dynamic and fault-tolerant leadership in UAV swarms

- Zuo et al., *Electronics* 11(14):2143, 2022, DOI 10.3390/electronics11142143. Raft-like
  voting after the leader fails.
- Ganesan et al., "BOLD", *Sensors* 20(11):3134, 2020, DOI 10.3390/s20113134.
  Optimisation-based leader election for drones, with re-election.
- Tariverdi & Torresen, arXiv:2308.10097, 2023. Raft for formation control.

## 2. How does a leader fly to a goal without GPS?

| Approach | Drift / accuracy | Infrastructure or communication? | Key references |
|---|---|---|---|
| Visual- or LiDAR-inertial odometry (dead reckoning from launch) | ~0.5–2 % of distance; VINS-Mono 0.88 % over ~700 m, OKVIS 2.36 % on the same path | none | Qin, Li, Shen, *IEEE T-RO* 2018, arXiv:1708.03852; Campos et al., ORB-SLAM3, arXiv:2007.11898; Xu et al., FAST-LIO2, arXiv:2107.06829; Delmerico & Scaramuzza, ICRA 2018 |
| Map matching (camera to satellite image or orthophoto; TERCOM) | Bounded; <8 m over 0.85 km (Goforth & Lucey) | Prior map onboard | Couturier & Akhloufi, *RAS* 135:103666, 2021; Goforth & Lucey, ICRA 2019; Kinnari et al., arXiv:2103.14381; Gurgu et al., arXiv:2210.09727; Golden (TERCOM), SPIE 1980, DOI 10.1117/12.959127 |
| Bio-inspired path integration (polarized-light compass + optic flow) | AntBot ~0.67 % homing error; compass ~0.3–2° | none (needs a view of the sky) | Dupeyroux, Serres, Viollet, *Science Robotics* 4(27):eaau0307, 2019; Stone et al., *Curr. Biol.* 2017, DOI 10.1016/j.cub.2017.08.052; de Croon et al., *Nature* 610:485, 2022, DOI 10.1038/s41586-022-05182-2 |
| Signals of opportunity (LTE, TV, Starlink) and UWB anchors | Metre level | Third-party transmitters / deployed anchors; **breaks the infrastructure-free assumption** | Kassas et al., IEEE AES Magazine 2022; Mueller, Hamer, D'Andrea, ICRA 2015 |
| Terminal homing (visual target, snapshot, beacon, gas or light source) | Converges once the target is sensed | none, or a beacon at the goal | Zeil et al., JOSA A 2003; Warren et al., *RA-L* 2019, arXiv:1809.05757; Duisterhof et al., arXiv:2107.05490 |
| Swarm: only the leader navigates well, followers use relative sensing (UVDAR) | Swarm error ≈ leader error + decimetre-to-metre relative error | none | Walter et al., *RA-L* 2019, DOI 10.1109/LRA.2019.2901683; Horyna et al., arXiv:2404.18729; Pritzl et al., arXiv:2306.17544 |

**Realistic assumption for CB-ELE over ~1 km with no GPS and no communication:**
- **Goal representation:** the goal is a relative vector from the launch point.
- **Navigation error:** the leader's odometry error grows with distance, with
  σ ≈ 1 % of distance flown (sweep 0.5–3 %), i.e. **about 10 m at 1 km**.
- **Map-matching variant:** if the leader carries a prior map, the error stays bounded at about 5–10 m.
- **Terminal phase:** final convergence on the goal needs local sensing near the goal.
- **Leader ≠ swarm:** each drone's goal estimate drifts independently, so leaderless swarms
  receive conflicting directions. This is the main argument for a single leader in GPS-denied flight.

## 3. Prior art for collision- or encounter-based leader election

**No published work found that elects a leader by physical encounters in drones, or that
elects exactly one leader this way among ground robots.**

### Closest work, physical encounters

| Work | Mechanism | Closeness (1–5) |
|---|---|---|
| Schmickl et al., "Get in touch: cooperative decision making based on robot-to-robot collisions", *AAMAS journal* 18(1):133–155, 2009 (DOI unverified) | BEECLUST: bump into a robot, stop, wait (timer), continue. The only signal is the encounter. The outcome is a collective site choice. | **4** |
| Angluin et al., *Distributed Computing* 18(4):235–253, 2006, DOI 10.1007/s00446-005-0138-3 | Population protocols: pairwise encounters with the rule L+L→L+F | **4** for the rule, 1 for the embodiment |
| Doty & Soloveichik, arXiv:1502.04246 | Stable L+L→L+F election needs Ω(n) parallel time: a theoretical yardstick | 3 |
| Flocchini, Killick, Kranakis, Santoro, Yamashita, ISAAC 2019, DOI 10.4230/LIPIcs.ISAAC.2019.8 | Leader election when robots can talk only while co-located | 3–4 |
| Czyzowicz, Kranakis, Pacheco, ICALP 2013 / *Distributed Computing* 2014, DOI 10.1007/s00446-014-0234-3; Gąsieniec et al., arXiv:1504.07127; Friedetzky et al., MFCS 2012 | Bouncing robots on a ring: collisions as the only information, used for localization and symmetry breaking | 3 |
| Mayya, Pierpaoli, Nair, Egerstedt, *IEEE T-RO* 2019 (DOI unverified); RSS 2017 | Inter-robot collisions as the sole sensing modality, for localization | 3 |
| Mayya, Wilson, Egerstedt, *Swarm Intelligence* 13:115–143, 2019 | Task allocation driven by encounter rate | 3 |
| Mayya, Pierpaoli, Egerstedt, "Voluntary retreat", ICRA 2019, arXiv:1812.02193 | A proximity event makes one robot withdraw on its own, with no communication | 3–4 |
| Varughese et al., arXiv:1804.04202 | Leader election with pings and random timers (a 1-bit signal, not contact) | 3 |
| Gordon & Mehdiabadi 1999; Pratt 2005 | Ants: contact rate drives task switching and quorum decisions | 2 |

### Name-only: radio "collision detection"

In these works "collision" means simultaneous transmissions on a shared channel, not
physical contact. Cite them only to set them apart from CB-ELE.

- Willard, *SIAM J. Comput.* 1986, DOI 10.1137/0215032.
- Ghaffari & Haeupler, arXiv:1210.8439.
- Chang et al., arXiv:1609.08486.
- Beeping model: Dufoulon et al., DISC 2018.

### Positioning of CB-ELE

CB-ELE is an **embodied, aerial version of the population-protocol leader rule**, in the
BEECLUST and Mayya–Egerstedt line of work. These elements look new:
- candidates actively approach each other, which raises the encounter rate;
- a pairwise freeze followed by a random-timer race decides who withdraws;
- the local termination rule is "N contact-free rounds means I am leader";
- challenger-based re-election;
- application to GPS- and communication-denied UAVs.
