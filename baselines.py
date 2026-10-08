"""Baseline swarm-navigation methods for comparison with CB-ELE.

All baselines share CB-ELE's world: same arena, start distribution (same
RNG draw order, so a seed gives identical initial positions/headings),
sensing radius, hard-safety projection and goal.  Only the per-agent
steering rule differs.

Methods
-------
couzin  Couzin, Krause, Franks & Levin, Nature 433 (2005): zonal
        repulsion / orientation / attraction; a fraction `informed_fraction`
        of agents also steers towards the goal with weight `couzin_omega`.
pacnav  Ahmad et al., Bioinspiration & Biomimetics 17(6) (2022): see
        `_pacnav_step`.

Adaptations to the shared setting (not in the original papers) are marked
"ADAPT".
"""

from dataclasses import dataclass, replace
import numpy as np

import cb_ele
from cb_ele import Config, pairwise_distances, enforce_hard_safety, unit


@dataclass(frozen=True)
class BaselineConfig:
    method: str = "couzin"
    informed_fraction: float = 1.0           # share of agents that know the goal
    speed: float = 1.0                       # m/s, constant cruise speed
    repulsion_radius: float = 12.0           # = CB-ELE avoidance_radius
    turn_rate: float = np.deg2rad(40.0)      # max heading change per round
    noise: float = 0.02                      # rad, heading noise (std)
    couzin_omega: float = 0.5                # goal weight of informed agents
    # PACNav (paper Table 1 / official code), lengths scaled from the
    # paper's 1 m-radius UAVs to CB-ELE's 15 m preferred spacing.
    pacnav_max_speed: float = 1.5            # = CB-ELE max_speed (code: 1.0)
    pacnav_follow_radius: float = 15.0       # R_f   (paper 4.0 m)
    pacnav_collision_radius: float = 12.0    # R^o   (paper 2.5 m)
    pacnav_k_nav: float = 1.2                # K^n   (paper 1.2 s^-1)
    pacnav_k_col: float = 23.0               # K^c = 1.0 * (12/2.5)^2, keeps the
                                             # n/c ratio of eq. 11 under scaling
    pacnav_alpha: float = 8.0                # Alg. 3 exponent (code)
    pacnav_min_scale: float = 0.3            # V^m in eq. 15 (code)
    # Petracek et al. (2020), eqs. 4-8.  The paper gives no numeric values;
    # kappa gets a length scale so the pairwise equilibrium (kappa = 1)
    # lies at CB-ELE's 15 m preferred spacing with R_n = 35 m.
    petracek_rate: float = 1.0               # lambda [Hz]: one update per round
    petracek_length_scale: float = 125.7     # s: sqrt(s/d) - sqrt(s/R_n) = 1 at d = 15 m
    petracek_goal_gain: float = 1.5          # |v_n| [m/s] for informed agents
    pacnav_history: int = 6                  # K^p, path samples (>= 3)
    pacnav_lookahead: float = 1.5 / 1.2      # |a_n - p|: the A* waypoint is a
                                             # short step ahead; K^n * step =
                                             # max speed before eq. 15 scaling


def _informed_mask(n, fraction, rng):
    k = int(round(fraction * n))
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=k, replace=False)] = True
    return mask


def _turn_towards(heading, desired_angle, max_turn):
    delta = (desired_angle - heading + np.pi) % (2.0 * np.pi) - np.pi
    return heading + np.clip(delta, -max_turn, max_turn)


def _couzin_desired(i, position, heading, distance, neighbors, informed,
                    goal, cfg, bcfg):
    """Couzin et al. (2005) direction rule for agent i (unit vector).

    Eqs. 57-59 as restated by Vicsek & Zafeiris (2012, sec. 5.4):
    repulsion from neighbours within alpha has priority (eq. 57); otherwise
    d = sum_j (r_j - r_i)/|r_j - r_i| + sum_j v_j/|v_j| over neighbours
    j != i within rho (eq. 58, plain sums); informed agents use
    (d_hat + omega g) / |d_hat + omega g| (eq. 59).
    """
    too_close = np.flatnonzero(distance[i] < bcfg.repulsion_radius)
    if len(too_close):
        d = -np.sum([unit(position[j] - position[i]) for j in too_close], axis=0)
    else:
        ids = np.flatnonzero(neighbors[i])
        if len(ids):
            d_att = np.sum([unit(position[j] - position[i]) for j in ids], axis=0)
            d_ori = np.sum(np.column_stack((np.cos(heading[ids]),
                                            np.sin(heading[ids]))), axis=0)
            d = d_att + d_ori                       # v0.8 fix: eq. 58 sums
        else:
            d = np.array([np.cos(heading[i]), np.sin(heading[i])])
    d = unit(d)
    if informed[i]:
        d = unit(d + bcfg.couzin_omega * unit(goal - position[i]))
    return d


def run_baseline(config=cb_ele.CONFIG, bcfg=BaselineConfig()):
    """Run a baseline; returns a trajectory dict like CB-ELE's."""
    cfg = config
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_agents
    goal = np.array([cfg.width - 5.0, cfg.height / 2.0])
    # Same RNG draw order as cb_ele.run_simulation -> identical start state.
    position = cb_ele.initial_positions(rng, cfg)
    heading = rng.uniform(-np.pi, np.pi, n)
    informed = _informed_mask(n, bcfg.informed_fraction, rng)
    method_state = {}

    traj_position = np.zeros((cfg.rounds, n, 2))
    traj_velocity = np.zeros((cfg.rounds, n, 2))
    traj_state = np.where(informed, cb_ele.LEADER, cb_ele.FOLLOWER)
    traj_state = np.broadcast_to(traj_state, (cfg.rounds, n)).astype(np.int8)

    for r in range(cfg.rounds):
        distance = pairwise_distances(position)
        np.fill_diagonal(distance, np.inf)
        neighbors = distance <= cfg.sensing_radius

        if bcfg.method == "couzin":
            desired = np.array([
                _couzin_desired(i, position, heading, distance, neighbors,
                                informed, goal, cfg, bcfg)
                for i in range(n)])
            speed = np.full(n, bcfg.speed)
            max_turn = bcfg.turn_rate
        elif bcfg.method == "petracek":
            desired, speed = _petracek_step(position, distance, neighbors,
                                            informed, goal, cfg, bcfg,
                                            method_state)
            max_turn = np.pi                 # holonomic, velocity-controlled
        elif bcfg.method == "pacnav":
            desired, speed = _pacnav_step(position, heading, distance,
                                          neighbors, informed, goal, cfg,
                                          bcfg, method_state, r)
            max_turn = np.pi                 # holonomic, velocity-controlled
        else:
            raise ValueError(f"unknown method {bcfg.method!r}")

        # ADAPT: informed agents hold position once at the goal, like the
        # CB-ELE leader's anchor rule, so arrival can be measured.
        at_goal = informed & (np.linalg.norm(goal - position, axis=1)
                              <= cfg.goal_arrival_radius)
        speed = np.where(at_goal, 0.0, speed)

        desired_angle = np.arctan2(desired[:, 1], desired[:, 0])
        desired_angle += rng.normal(0.0, bcfg.noise, n)
        heading = _turn_towards(heading, desired_angle, max_turn)
        velocity = speed[:, None] * np.column_stack((np.cos(heading),
                                                     np.sin(heading)))
        previous = position
        position = enforce_hard_safety(position + velocity, cfg)
        traj_position[r] = position
        traj_velocity[r] = position - previous

    return {"position": traj_position, "velocity": traj_velocity,
            "state": traj_state, "goal": goal, "informed": informed}


def _petracek_step(position, distance, neighbors, informed, goal, cfg, bcfg,
                   method_state):
    """Petracek et al. (2020) Boids model for UAVs without communication.

    UNTESTED: not yet calibrated or validated; not in compare.py defaults.

    f_b = 1/|N| sum_j [x_ij + v_ij/lambda - kappa(x_ij, R_n) x_ij]   (eq. 4)
    kappa(x, r) = max(0, sqrt(s/|x|) - sqrt(s/r))   (eq. 5, length scale s)
    f = f_b + v_n/lambda, v_n = goal attraction for informed agents  (eqs. 3, 6)
    v = min(v_m, lambda |f|) f/|f|                                 (eqs. 7-8)
    Relative velocities v_ij use the neighbours' previous velocities.
    """
    n = len(position)
    lam, s_len = bcfg.petracek_rate, bcfg.petracek_length_scale
    r_n = cfg.sensing_radius
    prev_v = method_state.setdefault("prev_v", np.zeros((n, 2)))
    f = np.zeros((n, 2))
    for i in range(n):
        ids = np.flatnonzero(neighbors[i])
        if len(ids):
            x = position[ids] - position[i]
            d = np.maximum(distance[i, ids], 1e-6)
            kappa = np.maximum(0.0, np.sqrt(s_len / d) - np.sqrt(s_len / r_n))
            v_rel = prev_v[ids] - prev_v[i]
            f[i] = np.mean(x + v_rel / lam - kappa[:, None] * x, axis=0)
        if informed[i]:
            f[i] += bcfg.petracek_goal_gain * unit(goal - position[i]) / lam
    norm = np.linalg.norm(f, axis=1)
    speed = np.minimum(cfg.max_speed, lam * norm)
    desired = np.where(norm[:, None] > 1e-9, f / np.maximum(norm, 1e-9)[:, None],
                       np.zeros((n, 2)))
    method_state["prev_v"] = speed[:, None] * desired
    return desired, speed


def _cosines(a, b):
    """Row-wise cosine between vectors a and b (..., 2); 0 if either is ~0."""
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    ok = (na > 1e-9) & (nb > 1e-9)
    return np.where(ok, np.sum(a * b, axis=-1) / np.maximum(na * nb, 1e-18), 0.0)


def _pacnav_step(position, heading, distance, neighbors, informed, goal,
                 cfg, bcfg, method_state, round_number):
    """PACNav (Ahmad et al. 2022), 2D point-agent version.

    Paths H_ij are the last `pacnav_history` observed positions of j (every
    neighbour in sensing range is in line of sight: no occlusion, no
    obstacles, so A* reduces to the straight line to the target point).
    Uninformed agents follow j* = argmax_j gamma_j + sum_l sigma_jl
    (eq. 9) over candidates that are >= R_f away, have >= 3 samples and are
    not approaching the previous target (eq. 8); informed agents go to g.
    Control u = n + c (eqs 11, 15, Alg. 3, eqs 16-20).
    """
    n = len(position)
    rf, ro = bcfg.pacnav_follow_radius, bcfg.pacnav_collision_radius
    hist = method_state.setdefault("history", [])
    hist.insert(0, position.copy())                 # newest first
    del hist[bcfg.pacnav_history:]
    H = np.asarray(hist)                            # (L, n, 2)
    L = len(H)
    target_point = method_state.setdefault("target_point", position.copy())
    prev_u = method_state.setdefault("prev_u", np.zeros((n, 2)))

    if L >= 3:
        h = H[:-1] - H[1:]                          # h^m, m = 1 newest
        persistence = _cosines(h[1:], h[:-1]).mean(axis=0)            # eq. 6
        similarity = np.mean(_cosines(h[:, :, None, :], h[:, None, :, :]),
                             axis=0)                                  # eq. 4
    d = np.empty((n, 2))
    u = np.zeros((n, 2))
    for i in range(n):
        ids = np.flatnonzero(neighbors[i])
        if informed[i]:
            d[i] = goal
        else:
            d[i] = position[i]                      # q0: alone, no motion
            if L >= 3 and len(ids):
                prev = target_point[i]
                cand = [j for j in ids
                        if distance[i, j] >= rf
                        and np.linalg.norm(H[0, j] - prev)
                        >= np.linalg.norm(H[-1, j] - prev)]
                if cand:
                    cand = np.asarray(cand)
                    sub = similarity[np.ix_(cand, cand)]
                    score = persistence[cand] + sub.sum(axis=1) - np.diag(sub)
                    d[i] = H[0, cand[np.argmax(score)]]                # q1

        to_target = d[i] - position[i]
        step = min(float(np.linalg.norm(to_target)), bcfg.pacnav_lookahead)
        nav = bcfg.pacnav_k_nav * step * unit(to_target)              # eq. 12
        if informed[i]:
            scale = 1.0
            if len(ids):
                scale = max(bcfg.pacnav_min_scale,
                            1.0 - distance[i, ids].mean() / (2.0 * rf))  # eq. 15
            nav = scale * nav
        else:
            for j in ids:                                             # Alg. 3
                e = unit(position[j] - position[i])
                proj = float(nav @ e)
                if proj > 0.0:
                    damp = min(1.0, (distance[i, j] / rf) ** bcfg.pacnav_alpha)
                    nav = nav - proj * e + damp * proj * e

        col = np.zeros(2)
        for j in np.flatnonzero(distance[i] < ro):                    # eqs 16-20
            away = position[i] - position[j]
            dd = max(float(np.linalg.norm(away)), 1e-6)
            phi = np.pi / (2.0 * ro) * dd
            c, s_ = np.cos(phi), np.sin(phi)
            base = away / dd
            plus = np.array([c * base[0] - s_ * base[1], s_ * base[0] + c * base[1]])
            minus = np.array([c * base[0] + s_ * base[1], -s_ * base[0] + c * base[1]])
            pick = plus if plus @ prev_u[i] >= minus @ prev_u[i] else minus
            col += max(0.0, 1.0 / dd - 1.0 / ro) * pick
        u[i] = nav + bcfg.pacnav_k_col * col

    method_state["target_point"] = d
    method_state["prev_u"] = u
    norm = np.linalg.norm(u, axis=1)
    speed = np.minimum(norm, bcfg.pacnav_max_speed)
    desired = np.where(norm[:, None] > 1e-9, u / np.maximum(norm, 1e-9)[:, None],
                       np.column_stack((np.cos(heading), np.sin(heading))))
    return desired, speed
