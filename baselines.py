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
    """Couzin et al. (2005) direction rule for agent i (unit vector)."""
    too_close = np.flatnonzero(distance[i] < bcfg.repulsion_radius)
    if len(too_close):
        d = -np.sum([unit(position[j] - position[i]) for j in too_close], axis=0)
    else:
        ids = np.flatnonzero(neighbors[i])
        if len(ids):
            d_att = np.sum([unit(position[j] - position[i]) for j in ids], axis=0)
            d_ori = np.sum(np.column_stack((np.cos(heading[ids]),
                                            np.sin(heading[ids]))), axis=0)
            d_ori += np.array([np.cos(heading[i]), np.sin(heading[i])])
            d = unit(d_att) + unit(d_ori)
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
    position = np.column_stack((
        rng.uniform(5.0, 25.0, n),
        rng.uniform(5.0, cfg.height - 5.0, n),
    ))
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
        elif bcfg.method == "pacnav":
            desired, speed = _pacnav_step(position, heading, distance,
                                          neighbors, informed, goal, cfg,
                                          bcfg, method_state, r)
        else:
            raise ValueError(f"unknown method {bcfg.method!r}")

        # ADAPT: informed agents hold position once at the goal, like the
        # CB-ELE leader's anchor rule, so arrival can be measured.
        at_goal = informed & (np.linalg.norm(goal - position, axis=1)
                              <= cfg.goal_arrival_radius)
        speed = np.where(at_goal, 0.0, speed)

        desired_angle = np.arctan2(desired[:, 1], desired[:, 0])
        desired_angle += rng.normal(0.0, bcfg.noise, n)
        heading = _turn_towards(heading, desired_angle, bcfg.turn_rate)
        velocity = speed[:, None] * np.column_stack((np.cos(heading),
                                                     np.sin(heading)))
        previous = position
        position = enforce_hard_safety(position + velocity, cfg)
        traj_position[r] = position
        traj_velocity[r] = position - previous

    return {"position": traj_position, "velocity": traj_velocity,
            "state": traj_state, "goal": goal, "informed": informed}


def _pacnav_step(position, heading, distance, neighbors, informed, goal,
                 cfg, bcfg, method_state, round_number):
    raise NotImplementedError("PACNav is added once the paper is reviewed")
