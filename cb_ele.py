"""CB-ELE v4.4: collision election with mid-run re-election experiment.

Repository version: v0.6.

v0.6
----
- Leader recognition adds per-round slowness evidence (the leader's gate and
  tether make it slower than its neighbours); belief threshold 2 -> 5.
  Recognition precision/recall 0.75/0.50 -> 0.86/0.88 (uninformed
  followers, 100 seeds); mission outcomes unchanged within noise.

v0.5
----
- Behavioural leader recognition (`leader_recognition = "follow"`): each
  drone infers the leader from protocol behaviour it can observe (freezing
  when touched, backing away, winning duels); followers steer to their
  believed leader and leaders close in on recognised rivals.  See
  docs/leader_recognition.md.

v0.4
----
- Duels are resolved with local information only.  Each drone knows only
  its own timer; it counts down while a contacting neighbour is seen
  hovering, withdraws when its own timer runs out, and survives when the
  contact ends.  The previous central arbitration (one loser per contact
  group, chosen by comparing timers) is removed.  Departures are noticed
  after `perception_latency`; timers expiring within it both withdraw.
- Leaders no longer use the global leader count/positions: duplicate
  leaders settle it when they meet (validation phase or at the goal).

v0.3
----
- `follower_informed_fraction`, trajectory recording (`compare.py`).

v0.2
----
- Re-election seeds candidates on the goal side of the swarm
  (`reelection_seed_mode = "goal_side"`); the v0.1 peripheral seeding
  picked rear drones, whose leader trailed and split the swarm.
- Default rounds 1700 -> 1350 (seed 42: mission stable from 1257).

v0.1
----
- Re-election no longer re-seeds the just-demoted leader as a candidate
  (it was re-promoted as soon as its cooldown expired).
- Default rounds 3000 -> 1700 (seed 42: swarm in goal zone at ~1110,
  mission stable from 1624).
- Animation figure height doubled (15 x 16 in); swarm panel uses
  aspect='auto', so the y axis is stretched relative to x.

Changes vs v4.3
---------------
F8. Metric scale: distances and speeds are now coherent with a 2 m x 2 m
    drone. hard safety 5 m, contact 8 m, avoidance 12 m, preferred
    spacing 15 m, sensing 35 m, connectivity guard 30 m, tether stop
    30 m, arrival/anchor 8 m, swarm goal 40 m. Speeds in m/s: max 1.5,
    cruise 0.8, min creep 0.4.

F9. Mid-distance re-election experiment.  When the swarm centroid has
    closed `reelection_at_fraction` (default 0.5) of the initial
    distance to the goal, the current leader(s) are demoted, a few
    peripheral followers are promoted to CANDIDATE, and a cooldown
    prevents the ex-leader from re-promoting.  Enabled with
    `reelection_enabled = True`.

F10. Only `cb_ele_metrics.csv` is written, in the current working
     directory (`output_dir = "."`).

F11. Catch-up boost.  A follower whose nearest neighbour is farther than
     `preferred_spacing` scales its speed by `1 + gain * lag`, capped at
     `max_speed`.  Lagging drones can rejoin the swarm instead of being
     permanently left behind.

Server execution (v0.0)
-----------------------
S1. Headless: outside a Jupyter kernel the non-interactive Agg backend is
    selected, and the animation is written to `animation_file` in
    `output_dir` (MP4 via ffmpeg, GIF fallback via Pillow) instead of being
    displayed inline.
S2. Command-line entry point: `python cb_ele.py --help`.
"""

from dataclasses import dataclass, asdict, replace
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib

_IN_NOTEBOOK = "ipykernel" in sys.modules
if not _IN_NOTEBOOK and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.lines import Line2D


try:
    from scipy.spatial.distance import cdist as _scipy_cdist
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


CANDIDATE, PASSIVE, FOLLOWER, LEADER, CHALLENGER, DORMANT = range(6)
STATE_NAMES = ["CANDIDATE", "PASSIVE", "FOLLOWER", "LEADER", "CHALLENGER", "DORMANT"]
STATE_COLORS = ["#1f77b4", "#b7b7b7", "#17becf", "#d62728", "#ff7f0e", "#9467bd"]


# =============================================================================
# CONFIG
# =============================================================================

@dataclass(frozen=True)
class Config:
    # World and display ------------------------------------------------------
    n_agents: int = 30
    width: float = 1000.0
    height: float = 100.0
    rounds: int = 1350                       # v0.2: mission stable from 1257 (seed 42)
    seed: int = 42
    fps: int = 20
    animation_stride: int = 12
    embed_limit_mb: int = 180
    prefer_html5_video: bool = True

    # Physical / information scale -------------------------------------------
    # Drone spans 2 m x 2 m.  All distances in metres.
    drone_span: float = 2.0
    hard_safety_distance: float = 5.0        # 2.5 drone spans, safe gap
    election_radius: float = 8.0             # "in contact"
    avoidance_radius: float = 12.0           # too close
    preferred_spacing: float = 15.0          # comfortable flocking distance
    sensing_radius: float = 35.0             # must be > 2 x preferred_spacing

    # Election state machine -------------------------------------------------
    candidate_silent_rounds: int = 6
    initial_leader_validation_rounds: int = 30
    leader_audit_period: int = 90
    leader_audit_rounds: int = 0
    duel_timer_min: float = 1.0
    duel_timer_max: float = 8.0
    # v0.4: local duel rules. A drone sees a neighbour as hovering when its
    # observed speed is below this; it notices a departure after the latency.
    hover_speed_threshold: float = 0.05      # m/s
    perception_latency: float = 0.1          # s (fraction of a 1 s round)

    # v0.5: behavioural leader recognition (no signal, no IDs exchanged).
    # Each drone scores every neighbour it can see from protocol behaviour:
    # freezing when touched (+), backing away when touched (-), and staying
    # put while the duel partner leaves, i.e. winning a duel (++).
    # "off": no recognition; "observe": scores only (for accuracy);
    # "follow": followers steer to their believed leader and a leader that
    # recognises another leader closes in to duel it.
    leader_recognition: str = "follow"
    recognition_decay: float = 0.995
    recognition_win_weight: float = 3.0
    recognition_freeze_weight: float = 1.0
    recognition_retreat_weight: float = 2.0
    recognition_slow_weight: float = 0.5     # v0.6: per-round evidence for
                                             # moving slower than the observer's
                                             # visible neighbours (the leader's
                                             # gate/tether make it wait)
    recognition_threshold: float = 5.0
    recognition_cap: float = 50.0
    recognition_follow_weight: float = 3.0
    passive_backoff_min: int = 4
    passive_backoff_max: int = 12

    # Failure / challenger watchdog -----------------------------------------
    collision_window: int = 12
    collision_rate_threshold: float = 0.35
    challenger_coherence_threshold: float = 0.30
    challenger_persistence_rounds: int = 8
    challenger_delay_min: int = 3
    challenger_delay_max: int = 7

    # Motion controller (speeds in m/s, rounds are seconds) ------------------
    max_speed: float = 1.5
    candidate_speed: float = 1.0
    passive_retreat_speed: float = 1.0
    challenger_speed: float = 0.8
    alignment_weight: float = 2.5
    cohesion_weight: float = 0.50
    separation_weight: float = 2.2
    follower_goal_weight: float = 2.0
    follower_informed_fraction: float = 1.0  # v0.3: share of followers that
                                             # steer to the goal (leader always does)
    follower_catchup_gain: float = 2.0       # v4.4: lagging-follower speed boost

    connectivity_guard_radius: float = 30.0
    connectivity_guard_neighbors: int = 2
    connectivity_projection_iterations: int = 5
    follower_stop_speed: float = 0.15
    follower_min_creep: float = 0.40

    motion_noise: float = 0.02
    leader_alignment_start: float = 0.50
    leader_alignment_full: float = 0.88
    leader_cruise_speed: float = 0.80
    leader_min_gate: float = 0.30

    leader_connectivity_tether_enabled: bool = True
    leader_tether_full_distance: float = 15.0
    leader_tether_stop_distance: float = 30.0

    goal_arrival_radius: float = 8.0
    leader_anchor_radius: float = 8.0
    leader_loiter_radius: float = 5.0
    leader_loiter_radial_gain: float = 1.4
    swarm_goal_radius: float = 40.0

    # Mid-distance re-election experiment -----------------------------------
    reelection_enabled: bool = True
    reelection_at_fraction: float = 0.5      # fire when this fraction of the
                                             # initial centroid distance remains
    reelection_demote_target: str = "FOLLOWER"   # "DORMANT" | "FOLLOWER" | "CANDIDATE"
    reelection_seed_candidates: int = 5
    reelection_seed_mode: str = "goal_side"  # v0.2: "goal_side" | "peripheral"
    reelection_block_ex_leader_rounds: int = 180

    # Start area (v0.7): depth along x grows with N so the start density is
    # that of 30 drones in 20 m x 90 m (60 m^2 per drone) at every size.
    spawn_area_per_agent: float = 60.0

    # Goal settling (v0.7): every drone carries the goal (any drone may be
    # elected leader).  Inside this radius a drone starts no new election
    # (no challenger watchdog, candidates/challengers revert to followers)
    # and informed followers stop pushing towards the goal point.  0 = off.
    goal_settle_radius: float = 100.0

    # Performance knobs ------------------------------------------------------
    hard_safety_iterations: int = 80
    use_scipy_cdist: bool = True
    verbose_timing: bool = True

    # Failure injection ------------------------------------------------------
    leader_failure_round: int | None = None

    # Notebook output --------------------------------------------------------
    show_inline: bool = True                 # only honoured inside Jupyter
    show_metrics_table: bool = False

    # Server output ----------------------------------------------------------
    save_animation: bool = True
    animation_file: str = "cb_ele_animation.gif"   # .mp4 also supported (ffmpeg)

    # CSV output -------------------------------------------------------------
    output_dir: str = "."                    # v4.4: same folder as the notebook
    csv_prefix: str = "cb_ele"
    save_csv: bool = True
    save_summary_csv: bool = False           # v4.4: only metrics by default
    save_events_csv: bool = False
    save_frames_npz: bool = False


CONFIG = Config()


def _as_config(config):
    if isinstance(config, dict):
        return Config(**{**asdict(CONFIG), **config})
    return config


def initial_positions(rng, config):
    """Start positions shared by CB-ELE and the baselines (same RNG draws)."""
    n = config.n_agents
    span_y = config.height - 10.0
    depth = config.spawn_area_per_agent * n / span_y
    return np.column_stack((
        rng.uniform(5.0, 5.0 + depth, n),
        rng.uniform(5.0, config.height - 5.0, n),
    ))


# =============================================================================
# LOW-LEVEL KERNELS
# =============================================================================

def unit(vector):
    vector = np.asarray(vector, dtype=float)
    magnitude = np.linalg.norm(vector)
    return vector / magnitude if magnitude > 1e-12 else np.zeros_like(vector)


def pairwise_distances(position, use_scipy=True):
    if use_scipy and _HAS_SCIPY:
        return _scipy_cdist(position, position)
    diff = position[:, None, :] - position[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def components(adjacency):
    visited = np.zeros(len(adjacency), dtype=bool)
    result = []
    for start in range(len(adjacency)):
        if visited[start]:
            continue
        stack = [start]
        group = []
        visited[start] = True
        while stack:
            node = stack.pop()
            group.append(node)
            for neighbor in np.flatnonzero(adjacency[node]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        result.append(group)
    return result


def vectorized_local_coherence(heading, neighbors):
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    adj = neighbors.astype(float)
    denom = adj.sum(axis=1) + 1.0
    mean_cos = (cos_h + adj @ cos_h) / denom
    mean_sin = (sin_h + adj @ sin_h) / denom
    return np.sqrt(mean_cos * mean_cos + mean_sin * mean_sin)


# =============================================================================
# HARD SAFETY
# =============================================================================

if _HAS_NUMBA:
    @_njit(cache=True, fastmath=True)
    def _enforce_hard_safety_numba(position, width, height, r, iterations):
        n = len(position)
        r2 = r * r
        for _ in range(iterations):
            changed = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    dx = position[i, 0] - position[j, 0]
                    dy = position[i, 1] - position[j, 1]
                    d2 = dx * dx + dy * dy
                    if d2 < r2:
                        d = np.sqrt(d2) if d2 > 1e-18 else 1e-9
                        push = 0.5 * (r - d)
                        ux = dx / d
                        uy = dy / d
                        position[i, 0] += push * ux
                        position[i, 1] += push * uy
                        position[j, 0] -= push * ux
                        position[j, 1] -= push * uy
                        changed = True
            for i in range(n):
                if position[i, 0] < 0.0:
                    position[i, 0] = 0.0
                elif position[i, 0] > width:
                    position[i, 0] = width
                if position[i, 1] < 0.0:
                    position[i, 1] = 0.0
                elif position[i, 1] > height:
                    position[i, 1] = height
            if not changed:
                break
        return position


def enforce_hard_safety(position, config, iterations=None):
    if iterations is None:
        iterations = config.hard_safety_iterations
    if _HAS_NUMBA:
        return _enforce_hard_safety_numba(
            position, config.width, config.height,
            config.hard_safety_distance, iterations,
        )
    r = config.hard_safety_distance
    r2 = r * r
    for _ in range(iterations):
        diff = position[:, None, :] - position[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        np.fill_diagonal(d2, np.inf)
        overlap_mask = d2 < r2
        if not overlap_mask.any():
            break
        d = np.sqrt(np.maximum(d2, 1e-18))
        overlap = np.where(overlap_mask, (r - d) * 0.5, 0.0)
        unit_vec = diff / d[..., None]
        correction = np.einsum("ij,ijk->ik", overlap, unit_vec)
        position = position + correction
        np.clip(position[:, 0], 0.0, config.width, out=position[:, 0])
        np.clip(position[:, 1], 0.0, config.height, out=position[:, 1])
    return position


# =============================================================================
# LOCAL FORCES
# =============================================================================

def local_alignment(agent, heading, neighbors):
    ids = np.flatnonzero(neighbors[agent])
    sample = np.concatenate(([heading[agent]], heading[ids]))
    vector = np.array([np.cos(sample).mean(), np.sin(sample).mean()])
    return unit(vector)


def cohesion(agent, position, distance, neighbors, config):
    ids = np.flatnonzero(neighbors[agent])
    if not len(ids):
        return np.zeros(2)
    displacement = position[ids] - position[agent]
    d = distance[agent, ids]
    activation = np.clip(
        (d - config.preferred_spacing)
        / max(config.sensing_radius - config.preferred_spacing, 1e-9),
        0.0, 1.0,
    )
    return unit(np.sum(displacement * activation[:, None], axis=0))


def separation(agent, position, distance, radius):
    ids = np.flatnonzero((distance[agent] > 0) & (distance[agent] <= radius))
    force = np.zeros(2)
    for other in ids:
        d = max(distance[agent, other], 1e-9)
        force += ((radius - d) / radius) * (position[agent] - position[other]) / d
    return unit(force)


# =============================================================================
# CONNECTIVITY PROJECTION
# =============================================================================

def preserve_local_connectivity(position, distance, neighbors, velocity, movable, config):
    velocity = velocity.copy()
    n = len(position)
    protected = np.zeros_like(neighbors, dtype=bool)
    for i in range(n):
        ids = np.flatnonzero(neighbors[i])
        if len(ids):
            order = ids[np.argsort(distance[i, ids])]
            chosen = order[:config.connectivity_guard_neighbors]
            protected[i, chosen] = True
    protected |= protected.T

    soft_guard = 0.8 * config.connectivity_guard_radius
    active = protected & (distance >= soft_guard)
    pairs = np.argwhere(np.triu(active, k=1))
    if len(pairs) == 0:
        return velocity

    max_speed = config.max_speed
    for _ in range(config.connectivity_projection_iterations):
        changed = False
        for i, j in pairs:
            d = distance[i, j]
            if d < soft_guard:
                continue
            direction = (position[j] - position[i]) / max(d, 1e-9)
            separating_speed = float(np.dot(velocity[j] - velocity[i], direction))
            if separating_speed <= 0.0:
                continue
            mi, mj = movable[i], movable[j]
            if mi and mj:
                correction = 0.5 * separating_speed * direction
                velocity[i] += correction
                velocity[j] -= correction
            elif mi:
                velocity[i] += separating_speed * direction
            elif mj:
                velocity[j] -= separating_speed * direction
            changed = True
        speed = np.linalg.norm(velocity, axis=1)
        too_fast = speed > max_speed
        velocity[too_fast] *= (max_speed / speed[too_fast])[:, None]
        if not changed:
            break
    return velocity


# =============================================================================
# GRAPH METRICS
# =============================================================================

def graph_metrics(neighbors, state):
    groups = components(neighbors)
    sizes = [len(g) for g in groups]
    leader_counts = [
        int(np.sum(state[np.asarray(g, dtype=int)] == LEADER)) for g in groups
    ]
    reachable = sum(size for size, leaders in zip(sizes, leader_counts) if leaders > 0)
    return {
        "component_count": len(groups),
        "largest_component_fraction": max(sizes) / len(state),
        "leader_reachable_fraction": reachable / len(state),
        "single_leader_components": int(sum(c == 1 for c in leader_counts)),
        "components_without_leader": int(sum(c == 0 for c in leader_counts)),
        "components_with_multiple_leaders": int(sum(c > 1 for c in leader_counts)),
    }


def leader_is_auditing(age, distance_to_goal, config):
    if distance_to_goal <= config.goal_arrival_radius:
        return False
    if age <= config.initial_leader_validation_rounds:
        return True
    elapsed = age - config.initial_leader_validation_rounds - 1
    return elapsed % config.leader_audit_period < config.leader_audit_rounds


def leader_connectivity_tether(agent, distance, neighbors, config):
    if not config.leader_connectivity_tether_enabled:
        return 1.0
    ids = np.flatnonzero(neighbors[agent])
    if not len(ids):
        return 0.0
    nearest = float(np.min(distance[agent, ids]))
    return float(np.clip(
        (config.leader_tether_stop_distance - nearest)
        / max(config.leader_tether_stop_distance
              - config.leader_tether_full_distance, 1e-9),
        0.0, 1.0,
    ))


def trigger_reelection(state, confidence, leader_age, anomaly_counter,
                       state_age, duel_active, duel_timer, collision_history,
                       promotion_cooldown, position,
                       event_log, round_number, rng, config, goal=None):
    """v4.4: mid-run re-election. Demote current leader(s), seed candidates."""
    target_map = {"DORMANT": DORMANT, "FOLLOWER": FOLLOWER, "CANDIDATE": CANDIDATE}
    target = target_map.get(config.reelection_demote_target, FOLLOWER)

    leaders = np.flatnonzero(state == LEADER)
    for i in leaders:
        state[i] = target
        confidence[i] = 0
        leader_age[i] = 0
        anomaly_counter[i] = 0
        state_age[i] = 0
        duel_active[i] = False
        duel_timer[i] = np.inf
        collision_history[:, i] = False
        promotion_cooldown[i] = config.reelection_block_ex_leader_rounds
        event_log.append((round_number, int(i), "reelection_demote"))

    if config.reelection_seed_candidates > 0:
        # v0.1: never re-seed the leader(s) just demoted
        eligible = (state == FOLLOWER)
        eligible[leaders] = False
        followers = np.flatnonzero(eligible)
        if len(followers):
            centroid = position[followers].mean(axis=0)
            offset = position[followers] - centroid
            if config.reelection_seed_mode == "goal_side" and goal is not None:
                # v0.2: front of the swarm, i.e. farthest along centroid->goal
                d = offset @ unit(goal - centroid)
            else:
                d = np.linalg.norm(offset, axis=1)
            order = np.argsort(-d)                      # farthest first
            seeds = followers[order[:config.reelection_seed_candidates]]
            for i in seeds:
                state[i] = CANDIDATE
                confidence[i] = 0
                state_age[i] = 0
                duel_active[i] = False
                duel_timer[i] = np.inf
                collision_history[:, i] = False
                event_log.append((round_number, int(i), "reelection_seed"))


# =============================================================================
# METRICS
# =============================================================================

METRIC_KEYS = [
    "candidate_count", "passive_count", "follower_count", "leader_count",
    "challenger_count", "dormant_count", "contender_count",
    "leaders_auditing", "election_contacts", "hard_safety_violations",
    "minimum_distance", "component_count", "largest_component_fraction",
    "leader_reachable_fraction", "single_leader_components",
    "components_without_leader", "components_with_multiple_leaders",
    "heading_variance", "goal_alignment", "mean_local_coherence",
    "centroid_distance", "mean_speed", "normalized_mean_speed",
    "moving_fraction", "centroid_progress", "leader_goal_distance",
    "max_anomaly_counter", "swarm_goal_reached", "leader_at_goal",
    "mission_success",
    "leader_gate", "leader_tether", "leader_agreement",
]

METRIC_BOOL = {"swarm_goal_reached", "leader_at_goal", "mission_success"}


def compute_metrics_v4(
    position, previous_centroid, heading, state, goal,
    election_contacts, neighbors, audit_mask, anomaly_counter,
    observed_speed, local_coherence_all, distance, config,
    leader_gate=np.nan, leader_tether=np.nan, leader_agreement=np.nan,
):
    n = len(state)
    to_goal = goal[None, :] - position
    to_goal /= np.maximum(np.linalg.norm(to_goal, axis=1, keepdims=True), 1e-9)
    leaders = np.flatnonzero(state == LEADER)
    centroid = position.mean(axis=0)
    old_distance = float(np.linalg.norm(previous_centroid - goal))
    new_distance = float(np.linalg.norm(centroid - goal))
    leader_goal_distance = (
        float(np.min(np.linalg.norm(position[leaders] - goal, axis=1)))
        if len(leaders) else np.nan
    )
    graph = graph_metrics(neighbors, state)
    leader_at_goal = bool(
        len(leaders) == 1
        and leader_goal_distance <= config.goal_arrival_radius
    )
    mission_success = bool(
        leader_at_goal
        and new_distance <= config.swarm_goal_radius
        and graph["component_count"] == 1
        and graph["components_without_leader"] == 0
        and graph["components_with_multiple_leaders"] == 0
    )
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    mean_cos = cos_h.mean()
    mean_sin = sin_h.mean()
    heading_variance = float(np.clip(
        1.0 - np.sqrt(mean_cos * mean_cos + mean_sin * mean_sin), 0.0, 1.0
    ))
    goal_alignment = float(np.mean(cos_h * to_goal[:, 0] + sin_h * to_goal[:, 1]))
    hard_violations = int(np.triu(distance < config.hard_safety_distance - 1e-6, 1).sum())
    dist_no_self = distance.copy()
    np.fill_diagonal(dist_no_self, np.inf)
    minimum_distance = float(dist_no_self.min())
    contacts = int(np.triu(election_contacts, 1).sum())
    result = {
        "candidate_count": int(np.sum(state == CANDIDATE)),
        "passive_count": int(np.sum(state == PASSIVE)),
        "follower_count": int(np.sum(state == FOLLOWER)),
        "leader_count": int(np.sum(state == LEADER)),
        "challenger_count": int(np.sum(state == CHALLENGER)),
        "dormant_count": int(np.sum(state == DORMANT)),
        "contender_count": int(np.sum((state == CANDIDATE) | (state == LEADER))),
        "leaders_auditing": int(np.sum(audit_mask & (state == LEADER))),
        "election_contacts": contacts,
        "hard_safety_violations": hard_violations,
        "minimum_distance": minimum_distance,
        "heading_variance": heading_variance,
        "goal_alignment": goal_alignment,
        "mean_speed": float(np.mean(observed_speed)),
        "normalized_mean_speed": float(np.mean(observed_speed) / max(config.max_speed, 1e-9)),
        "moving_fraction": float(np.mean(observed_speed >= config.follower_stop_speed)),
        "centroid_distance": new_distance,
        "centroid_progress": float(old_distance - new_distance),
        "leader_goal_distance": leader_goal_distance,
        "leader_at_goal": leader_at_goal,
        "mean_local_coherence": float(np.mean(local_coherence_all)),
        "max_anomaly_counter": int(anomaly_counter.max()),
        "swarm_goal_reached": bool(new_distance <= config.swarm_goal_radius),
        "mission_success": mission_success,
        "leader_gate": float(leader_gate),
        "leader_tether": float(leader_tether),
        "leader_agreement": float(leader_agreement),
    }
    result.update(graph)
    result["leader_ids"] = (
        ",".join(map(str, leaders.tolist())) if len(leaders) else "-"
    )
    return result


# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation(config=CONFIG, return_trajectory=False):
    """Run CB-ELE.  With `return_trajectory`, also return per-round
    positions, realised velocities and states (see `compare.py`)."""
    config = _as_config(config)
    t_start = time.perf_counter()

    rng = np.random.default_rng(config.seed)
    n = config.n_agents
    goal = np.array([config.width - 5.0, config.height / 2.0])
    position = initial_positions(rng, config)
    heading = rng.uniform(-np.pi, np.pi, n)

    state = np.full(n, CANDIDATE, dtype=int)
    confidence = np.zeros(n, dtype=int)
    backoff = np.zeros(n, dtype=int)
    challenger_timer = np.zeros(n, dtype=int)
    anomaly_counter = np.zeros(n, dtype=int)
    state_age = np.zeros(n, dtype=int)
    leader_age = np.zeros(n, dtype=int)
    observed_speed = np.full(n, config.candidate_speed, dtype=float)
    duel_active = np.zeros(n, dtype=bool)
    duel_timer = np.full(n, np.inf)
    collision_history = np.zeros((config.collision_window, n), dtype=bool)
    promotion_cooldown = np.zeros(n, dtype=int)     # v4.4
    # v0.3: informed followers; separate RNG keeps the main stream unchanged
    informed = np.zeros(n, dtype=bool)
    informed_rng = np.random.default_rng([config.seed, 1])
    informed[informed_rng.choice(
        n, size=int(round(config.follower_informed_fraction * n)),
        replace=False)] = True

    # v0.5: behavioural leader recognition state
    score = np.zeros((n, n))                       # score[i, j]: i's view of j
    belief = np.full(n, -1)                        # i's believed leader
    contacts_prev = np.zeros((n, n), dtype=bool)   # contacts at last round start
    hover_prev = np.zeros(n, dtype=bool)           # hovering two motions ago
    recog_precision = np.full(config.rounds, np.nan)
    recog_recall = np.full(config.rounds, np.nan)

    # v4.4: baseline centroid distance, for the mid-run reelection trigger
    initial_centroid_distance = float(np.linalg.norm(position.mean(axis=0) - goal))
    reelection_fired = False

    display_rounds = set(range(0, config.rounds, max(1, config.animation_stride)))
    display_rounds.add(config.rounds - 1)
    display_frames = []

    metric_matrix = np.zeros((config.rounds, len(METRIC_KEYS)), dtype=float)
    metric_leader_ids = ["-"] * config.rounds
    if return_trajectory:
        traj_position = np.zeros((config.rounds, n, 2))
        traj_velocity = np.zeros((config.rounds, n, 2))
        traj_state = np.zeros((config.rounds, n), dtype=np.int8)
    event_log = []
    leader_transitions = []

    use_scipy = config.use_scipy_cdist and _HAS_SCIPY

    for round_number in range(config.rounds):
        previous_centroid = position.mean(axis=0).copy()
        distance = pairwise_distances(position, use_scipy=use_scipy)
        np.fill_diagonal(distance, np.inf)
        neighbors = distance <= config.sensing_radius
        election_contacts = distance <= config.election_radius
        contact_now = election_contacts.any(axis=1)

        slot = round_number % config.collision_window
        collision_history[slot] = contact_now
        sample_count = min(round_number + 1, config.collision_window)
        collision_rate = collision_history[:sample_count].mean(axis=0)

        # ------- v4.4: mid-distance re-election trigger -------------------
        if (config.reelection_enabled and not reelection_fired):
            current_distance = float(np.linalg.norm(
                position.mean(axis=0) - goal))
            if current_distance <= initial_centroid_distance * config.reelection_at_fraction:
                trigger_reelection(
                    state, confidence, leader_age, anomaly_counter,
                    state_age, duel_active, duel_timer, collision_history,
                    promotion_cooldown, position,
                    event_log, round_number, rng, config, goal=goal,
                )
                reelection_fired = True
        promotion_cooldown = np.maximum(promotion_cooldown - 1, 0)
        # ------------------------------------------------------------------

        current_leader_ids = tuple(int(x) for x in np.flatnonzero(state == LEADER))
        if not leader_transitions or leader_transitions[-1][1] != current_leader_ids:
            leader_transitions.append((round_number, current_leader_ids))

        if (config.leader_failure_round is not None
                and round_number == config.leader_failure_round):
            failed = np.flatnonzero(state == LEADER)
            state[failed] = DORMANT
            leader_age[failed] = 0
            duel_active[failed] = False
            duel_timer[failed] = np.inf
            for i in failed:
                event_log.append((round_number, int(i), "leader_failure"))

        # --- Duel handshake (v0.4: local rules only) ---
        # Each drone knows only its own timer.  It counts down while at least
        # one contacting neighbour is seen hovering; it withdraws when its own
        # timer runs out; it survives when contact ends.  A neighbour that
        # withdraws starts moving at its expiry instant and is noticed
        # `perception_latency` later, so the later timer pauses instead of
        # expiring.  Timers expiring within the latency both withdraw.
        hovering = observed_speed < config.hover_speed_threshold

        if config.leader_recognition != "off":
            # Events of the last motion step, as seen from outside.
            touched = contacts_prev.any(axis=1)
            freeze = touched & hovering
            retreat = touched & ~hovering
            win = (contacts_prev & hover_prev[:, None] & hover_prev[None, :]
                   & hovering[:, None] & ~hovering[None, :])        # j stayed, k left
            visible = neighbors.astype(float)
            evidence = (config.recognition_freeze_weight * freeze
                        - config.recognition_retreat_weight * retreat)
            # Slowness relative to what observer i sees around it.
            n_vis = np.maximum(visible.sum(axis=1), 1.0)
            reference = (visible @ observed_speed) / n_vis
            slowness = (reference[:, None] - observed_speed[None, :]) / config.max_speed
            score = config.recognition_decay * score + visible * (
                evidence[None, :]
                + config.recognition_slow_weight * slowness
                + config.recognition_win_weight * (visible @ win.T.astype(float)))
            np.clip(score, 0.0, config.recognition_cap, out=score)
            np.fill_diagonal(score, 0.0)
            masked = np.where(neighbors, score, -1.0)
            best = masked.argmax(axis=1)
            belief = np.where(masked[np.arange(n), best]
                              >= config.recognition_threshold, best, -1)
            followers = state == FOLLOWER
            is_leader = state == LEADER
            sees_leader = followers & (neighbors & is_leader[None, :]).any(axis=1)
            has_belief = followers & (belief >= 0)
            correct = has_belief & is_leader[np.maximum(belief, 0)]
            if has_belief.any():
                recog_precision[round_number] = correct.sum() / has_belief.sum()
            if sees_leader.any():
                recog_recall[round_number] = (correct & sees_leader).sum() / sees_leader.sum()
        contacts_prev = election_contacts.copy()
        hover_prev = hovering.copy()
        for i in range(n):
            if state[i] in (CANDIDATE, LEADER):
                if contact_now[i] and not duel_active[i]:
                    duel_active[i] = True
                    duel_timer[i] = rng.uniform(config.duel_timer_min, config.duel_timer_max)
                    event_log.append((round_number, i, "duel_start"))
                elif not contact_now[i] and duel_active[i]:
                    duel_active[i] = False
                    duel_timer[i] = np.inf
                    event_log.append((round_number, i, "duel_survive"))

        duelers = [i for i in range(n)
                   if duel_active[i] and state[i] in (CANDIDATE, LEADER)]
        departure = {}                      # drone -> instant it starts leaving
        losers = []
        for i in sorted(duelers, key=lambda k: duel_timer[k]):
            partners = [j for j in np.flatnonzero(election_contacts[i])
                        if hovering[j]]
            if not partners:
                continue                    # nobody holding: countdown paused
            seen_until = min(1.0, max(
                departure[j] + config.perception_latency if j in departure else 1.0
                for j in partners))
            if duel_timer[i] <= seen_until:
                departure[i] = max(duel_timer[i], 0.0)
                losers.append(i)
            else:
                duel_timer[i] -= seen_until

        for loser in losers:
            previous = state[loser]
            state[loser] = PASSIVE if previous == CANDIDATE else FOLLOWER
            backoff[loser] = rng.integers(config.passive_backoff_min,
                                          config.passive_backoff_max + 1)
            confidence[loser] = 0
            leader_age[loser] = 0
            anomaly_counter[loser] = 0
            duel_active[loser] = False
            duel_timer[loser] = np.inf
            event = "candidate_withdraw" if previous == CANDIDATE else "leader_withdraw"
            event_log.append((round_number, int(loser), event))

        # --- State transitions ---
        previous_state = state.copy()
        settled = ((config.goal_settle_radius > 0)
                   & (np.linalg.norm(position - goal, axis=1)
                      <= config.goal_settle_radius))
        coherence_all = vectorized_local_coherence(heading, neighbors)

        for i in range(n):
            has_neighbors = bool(neighbors[i].any())
            coherence = float(coherence_all[i])

            if settled[i] and previous_state[i] in (CANDIDATE, CHALLENGER, DORMANT):
                # v0.7: arrived -> no new elections at the goal
                state[i] = FOLLOWER
                confidence[i] = 0
                challenger_timer[i] = 0
                anomaly_counter[i] = 0
                duel_active[i] = False
                duel_timer[i] = np.inf
                event_log.append((round_number, i, "goal_settle"))
            elif previous_state[i] == CANDIDATE:
                if not has_neighbors:
                    state[i], confidence[i] = DORMANT, 0
                elif promotion_cooldown[i] > 0:      # v4.4: ex-leader cooldown
                    confidence[i] = 0
                elif not contact_now[i] and not duel_active[i]:
                    confidence[i] += 1
                    if confidence[i] >= config.candidate_silent_rounds:
                        state[i], confidence[i] = LEADER, 0
                        leader_age[i] = 0
                        anomaly_counter[i] = 0
                        event_log.append((round_number, i, "leader_promote"))
                else:
                    confidence[i] = 0
            elif previous_state[i] == PASSIVE:
                if not has_neighbors:
                    state[i], backoff[i] = DORMANT, 0
                else:
                    backoff[i] -= 1
                    if backoff[i] <= 0:
                        state[i] = FOLLOWER
                        collision_history[:, i] = False
                        anomaly_counter[i] = 0
            elif previous_state[i] == FOLLOWER:
                if not has_neighbors:
                    state[i] = DORMANT
                    anomaly_counter[i] = 0
                else:
                    anomalous = (
                        not settled[i]
                        and state_age[i] >= config.collision_window
                        and collision_rate[i] >= config.collision_rate_threshold
                        and coherence <= config.challenger_coherence_threshold
                    )
                    anomaly_counter[i] = anomaly_counter[i] + 1 if anomalous else 0
                    if anomaly_counter[i] >= config.challenger_persistence_rounds:
                        state[i] = CHALLENGER
                        challenger_timer[i] = rng.integers(
                            config.challenger_delay_min, config.challenger_delay_max + 1)
                        anomaly_counter[i] = 0
                        event_log.append((round_number, i, "challenger_enter"))
            elif previous_state[i] == CHALLENGER:
                if not has_neighbors:
                    state[i], challenger_timer[i] = DORMANT, 0
                else:
                    challenger_timer[i] -= 1
                    if challenger_timer[i] <= 0:
                        state[i], confidence[i] = CANDIDATE, 0
                        event_log.append((round_number, i, "candidate_reenter"))
            elif previous_state[i] == DORMANT and has_neighbors:
                state[i] = CHALLENGER
                challenger_timer[i] = rng.integers(
                    config.challenger_delay_min, config.challenger_delay_max + 1)
                event_log.append((round_number, i, "dormant_reconnect"))

        changed = state != previous_state
        state_age[changed] = 0
        state_age[~changed] += 1
        leader_age[state == LEADER] += 1
        leader_age[state != LEADER] = 0

        # --- Motion controller ---
        desired = np.zeros_like(position)
        speed = np.zeros(n)
        audit_mask = np.zeros(n, dtype=bool)
        noise_allowed = np.ones(n, dtype=bool)

        log_gate = np.nan
        log_tether = np.nan
        log_agreement = np.nan

        for i in range(n):
            ids = np.flatnonzero(neighbors[i])
            close_avoidance = separation(i, position, distance, config.avoidance_radius)
            keep_connected = cohesion(i, position, distance, neighbors, config)

            if state[i] in (CANDIDATE, LEADER) and duel_active[i]:
                desired[i] = np.array([np.cos(heading[i]), np.sin(heading[i])])
                speed[i] = 0.0
                noise_allowed[i] = False
            elif state[i] == CANDIDATE:
                desired[i] = (unit(position[ids].mean(axis=0) - position[i])
                              if len(ids) else np.zeros(2))
                speed[i] = config.candidate_speed
            elif state[i] == PASSIVE:
                desired[i] = close_avoidance
                speed[i] = config.passive_retreat_speed
            elif state[i] == FOLLOWER:
                if contact_now[i]:
                    desired[i] = close_avoidance
                    speed[i] = config.passive_retreat_speed
                else:
                    speed_sample = np.concatenate(([observed_speed[i]], observed_speed[ids]))
                    local_speed = float(np.clip(speed_sample.mean(), 0.0, config.max_speed))
                    goal_direction = unit(goal - position[i])
                    desired[i] = (
                        config.alignment_weight * local_alignment(i, heading, neighbors)
                        + config.cohesion_weight * keep_connected
                        + config.separation_weight * close_avoidance
                        + config.follower_goal_weight * informed[i]
                        * (not settled[i]) * goal_direction
                    )
                    b = belief[i]
                    if config.leader_recognition == "follow" and b >= 0:
                        to_b = position[b] - position[i]
                        d_b = float(np.linalg.norm(to_b))
                        pull = np.clip((d_b - config.preferred_spacing)
                                       / config.preferred_spacing, 0.0, 1.0)
                        desired[i] += config.recognition_follow_weight * (
                            pull * unit(to_b)
                            + np.array([np.cos(heading[b]), np.sin(heading[b])]))
                    # *** v4.4 FIX: catch-up boost for lagging followers ***
                    if len(ids):
                        nn_dist = float(np.min(distance[i, ids]))
                        lag = max(0.0, (nn_dist - config.preferred_spacing)
                                       / max(config.preferred_spacing, 1e-9))
                        boost = 1.0 + config.follower_catchup_gain * lag
                        local_speed = min(config.max_speed, local_speed * boost)
                    local_speed = max(local_speed, config.follower_min_creep)
                    speed[i] = local_speed
            elif state[i] == LEADER:
                # v0.4: no global leader count; duplicate leaders meet during
                # validation or at the goal and settle it in a duel.
                distance_to_goal = float(np.linalg.norm(goal - position[i]))
                auditing = leader_is_auditing(leader_age[i], distance_to_goal, config)
                audit_mask[i] = auditing
                noise_allowed[i] = False
                goal_direction = unit(goal - position[i])
                goal_angle = np.arctan2(goal_direction[1], goal_direction[0])
                if distance_to_goal <= config.leader_anchor_radius:
                    heading[i] = goal_angle
                    desired[i] = np.array([np.cos(heading[i]), np.sin(heading[i])])
                    speed[i] = 0.0
                    log_gate = 1.0
                    log_tether = 1.0
                    log_agreement = 1.0
                elif (config.leader_recognition == "follow" and belief[i] >= 0
                      and not auditing):
                    # v0.5: a recognised rival leader -> close in and duel
                    heading[i] = goal_angle
                    desired[i] = unit(position[belief[i]] - position[i])
                    speed[i] = config.leader_cruise_speed
                    log_gate = log_tether = log_agreement = np.nan
                elif auditing and len(ids):
                    heading[i] = goal_angle
                    desired[i] = unit(position[ids].mean(axis=0) - position[i])
                    speed[i] = config.candidate_speed
                    log_gate = 1.0
                    log_tether = 1.0
                    log_agreement = 0.0
                else:
                    heading[i] = goal_angle
                    desired[i] = goal_direction
                    agreement = (
                        float(np.mean(np.cos(heading[ids] - goal_angle)))
                        if len(ids) else 0.0
                    )
                    gate = np.clip((agreement + 1.0) / 2.0, config.leader_min_gate, 1.0)
                    tether = leader_connectivity_tether(i, distance, neighbors, config)
                    speed[i] = config.leader_cruise_speed * gate * tether
                    log_gate = float(gate)
                    log_tether = float(tether)
                    log_agreement = float(agreement)
            elif state[i] == CHALLENGER:
                desired[i] = (unit(position[ids].mean(axis=0) - position[i])
                              if len(ids) else np.zeros(2))
                speed[i] = config.challenger_speed
            else:
                desired[i] = np.array([np.cos(heading[i]), np.sin(heading[i])])
                speed[i] = 0.0
                noise_allowed[i] = False

        noise = rng.normal(0.0, config.motion_noise, desired.shape)
        desired[noise_allowed] += noise[noise_allowed]
        norms = np.linalg.norm(desired, axis=1)
        moving = norms > 1e-9
        desired[moving] /= norms[moving, None]
        velocity = speed[:, None] * desired

        movable = (
            (state != DORMANT)
            & ~(((state == CANDIDATE) | (state == LEADER)) & duel_active)
            & (speed > 1e-9)
        )
        velocity = preserve_local_connectivity(
            position, distance, neighbors, velocity, movable, config
        )
        actual_speed = np.linalg.norm(velocity, axis=1)
        moving_velocity = actual_speed > 1e-9
        heading[moving_velocity] = np.arctan2(
            velocity[moving_velocity, 1], velocity[moving_velocity, 0]
        )

        leader_mask = state == LEADER
        if leader_mask.any():                       # v0.4: per leader, local
            dx = goal[0] - position[leader_mask, 0]
            dy = goal[1] - position[leader_mask, 1]
            safe = np.hypot(dx, dy) > 1e-6
            if safe.any():
                heading[np.flatnonzero(leader_mask)[safe]] = np.arctan2(dy[safe], dx[safe])

        position_before = position
        position = position + velocity
        observed_speed = actual_speed.copy()

        position = enforce_hard_safety(position, config)

        at_x_wall = (position[:, 0] <= 0.0) | (position[:, 0] >= config.width)
        at_y_wall = (position[:, 1] <= 0.0) | (position[:, 1] >= config.height)
        heading[at_x_wall] = np.pi - heading[at_x_wall]
        heading[at_y_wall] = -heading[at_y_wall]

        if return_trajectory:
            traj_position[round_number] = position
            traj_velocity[round_number] = position - position_before
            traj_state[round_number] = state

        distance = pairwise_distances(position, use_scipy=use_scipy)
        np.fill_diagonal(distance, np.inf)
        neighbors = distance <= config.sensing_radius
        election_contacts = distance <= config.election_radius
        coherence_all = vectorized_local_coherence(heading, neighbors)

        metrics = compute_metrics_v4(
            position, previous_centroid, heading, state, goal,
            election_contacts, neighbors, audit_mask, anomaly_counter,
            observed_speed, coherence_all, distance, config,
            leader_gate=log_gate, leader_tether=log_tether,
            leader_agreement=log_agreement,
        )

        for k, key in enumerate(METRIC_KEYS):
            metric_matrix[round_number, k] = float(metrics[key])
        metric_leader_ids[round_number] = metrics["leader_ids"]

        if round_number in display_rounds:
            display_frames.append({
                "round": round_number,
                "position": position.copy(),
                "heading": heading.copy(),
                "state": state.copy(),
                "duel_active": duel_active.copy(),
                "audit_mask": audit_mask.copy(),
                "metrics": metrics.copy(),
            })

    table = pd.DataFrame(metric_matrix, columns=METRIC_KEYS)
    table["round"] = np.arange(config.rounds)
    table["leader_ids"] = metric_leader_ids
    table["recognition_precision"] = recog_precision
    table["recognition_recall"] = recog_recall
    for key in METRIC_BOOL:
        table[key] = table[key].astype(bool)

    events_df = pd.DataFrame(event_log, columns=["round", "agent", "event"])
    transitions_df = pd.DataFrame(
        [{"round": r, "leader_ids": ",".join(map(str, ids)) if ids else "-"}
         for r, ids in leader_transitions]
    )

    if config.verbose_timing:
        elapsed = time.perf_counter() - t_start
        print(f"[v4.4] Simulation: {config.rounds} rounds, "
              f"{config.n_agents} agents, {elapsed:.2f} s "
              f"({1000*elapsed/config.rounds:.2f} ms/round)")
        print(f"[v4.4] Leader transitions: {len(leader_transitions)}")
        if config.reelection_enabled:
            print(f"[v4.4] Re-election fired: {reelection_fired}")

    if return_trajectory:
        trajectory = {"position": traj_position, "velocity": traj_velocity,
                      "state": traj_state, "goal": goal}
        return display_frames, goal, table, events_df, transitions_df, trajectory
    return display_frames, goal, table, events_df, transitions_df


# =============================================================================
# SUMMARY
# =============================================================================

def stable_unique_round(metrics_table):
    good = (
        (metrics_table["components_without_leader"] == 0)
        & (metrics_table["components_with_multiple_leaders"] == 0)
    ).to_numpy()
    suffix_all = np.logical_and.accumulate(good[::-1])[::-1]
    indices = np.flatnonzero(suffix_all)
    return int(metrics_table.iloc[indices[0]]["round"]) if len(indices) else None


def stable_true_round(mask, rounds):
    values = np.asarray(mask, dtype=bool)
    suffix_all = np.logical_and.accumulate(values[::-1])[::-1]
    indices = np.flatnonzero(suffix_all)
    return int(rounds[indices[0]]) if len(indices) else None


def longest_true_streak(mask):
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def summarize_run(metrics_table, config=CONFIG, event_table=None,
                  transitions_table=None):
    config = _as_config(config)
    final = metrics_table.iloc[-1]
    convergence_round = stable_unique_round(metrics_table)
    stable_mission = stable_true_round(
        metrics_table["mission_success"], metrics_table["round"].to_numpy())
    post_election = (int(final["round"] - convergence_round)
                     if convergence_round is not None else 0)
    post = (metrics_table[metrics_table["round"] >= convergence_round]
            if convergence_round is not None else metrics_table.iloc[0:0])
    return pd.Series({
        "final_leaders": int(final["leader_count"]),
        "final_components": int(final["component_count"]),
        "one_leader_per_component": bool(
            final["components_without_leader"] == 0
            and final["components_with_multiple_leaders"] == 0),
        "stable_unique_round": convergence_round,
        "post_election_rounds_observed": post_election,
        "challenger_entries": (
            int((event_table["event"] == "challenger_enter").sum())
            if event_table is not None and len(event_table) else 0),
        "leader_transition_count": (
            int(len(transitions_table)) if transitions_table is not None else 0),
        "hard_safety_violations_total": int(
            metrics_table["hard_safety_violations"].sum()),
        "minimum_distance_seen": float(metrics_table["minimum_distance"].min()),
        "final_heading_variance": float(final["heading_variance"]),
        "final_goal_alignment": float(final["goal_alignment"]),
        "mean_post_election_progress": (
            float(post["centroid_progress"].mean()) if len(post) else np.nan),
        "final_goal_distance": float(final["centroid_distance"]),
        "final_leader_goal_distance": float(final["leader_goal_distance"]),
        "final_leader_at_goal": bool(final["leader_at_goal"]),
        "final_mission_success": bool(final["mission_success"]),
        "stable_mission_round": stable_mission,
        "longest_mission_success_streak": longest_true_streak(
            metrics_table["mission_success"]),
        "final_leader_reachable_fraction": float(final["leader_reachable_fraction"]),
    })


# =============================================================================
# CSV OUTPUT
# =============================================================================

def save_results_csv(metrics_table, event_table, summary_series, goal, config,
                     transitions_table=None):
    """v4.4: only the metrics CSV is written, in `config.output_dir`."""
    os.makedirs(config.output_dir, exist_ok=True)
    prefix = config.csv_prefix
    paths = {}

    metrics_path = os.path.join(config.output_dir, f"{prefix}_metrics.csv")
    metrics_table.to_csv(metrics_path, index=False, float_format="%.6f")
    paths["metrics"] = metrics_path

    return paths


def print_saved_paths(paths):
    print("\nCSV outputs:")
    for kind, path in paths.items():
        size_kb = os.path.getsize(path) / 1024.0
        print(f"  {kind:20s}  {path}  ({size_kb:.1f} KB)")


# =============================================================================
# ANIMATION
# =============================================================================

def save_animation_file(anim, config):
    """Write the animation to disk; MP4 via ffmpeg, GIF fallback via Pillow."""
    os.makedirs(config.output_dir, exist_ok=True)
    path = os.path.join(config.output_dir, config.animation_file)
    t_start = time.perf_counter()
    if not path.lower().endswith(".gif"):
        try:
            anim.save(path, writer=FFMpegWriter(fps=config.fps, bitrate=2400))
            print(f"\nAnimation written to {path} "
                  f"({time.perf_counter() - t_start:.1f} s)")
            return path
        except (FileNotFoundError, RuntimeError, OSError) as e:
            print(f"\nffmpeg unavailable ({e}); falling back to GIF.")
            path = os.path.splitext(path)[0] + ".gif"
    anim.save(path, writer=PillowWriter(fps=config.fps))
    print(f"\nAnimation written to {path} "
          f"({time.perf_counter() - t_start:.1f} s)")
    return path


def create_animation(config=CONFIG):
    config = _as_config(config)

    display_frames, goal, table, events, transitions = run_simulation(config)

    if config.save_csv:
        summary = summarize_run(table, config, events, transitions)
        paths = save_results_csv(table, events, summary, goal, config, transitions)
        print_saved_paths(paths)
        print("\nRun summary:")
        print(summary.to_string())

    plt.rcParams["animation.embed_limit"] = config.embed_limit_mb
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    fig = plt.figure(figsize=(22, 10), dpi=80, constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.9, 1.0])
    swarm_ax = fig.add_subplot(grid[0, :])
    state_ax = fig.add_subplot(grid[1, 0])
    performance_ax = fig.add_subplot(grid[1, 1])

    swarm_ax.set(xlim=(-4, config.width + 4), ylim=(-2, config.height + 2),
                 aspect="auto", xlabel="x (m)", ylabel="y (m)")
    swarm_ax.set_facecolor("#fafafa")
    swarm_ax.grid(True, linestyle=":", alpha=0.25)
    swarm_ax.scatter(*goal, marker="X", s=210, color="limegreen", edgecolors="black")
    swarm_ax.add_patch(plt.Circle(goal, config.swarm_goal_radius, fill=False,
                                  color="green", linestyle="--", alpha=0.35))
    swarm_ax.text(goal[0] + 2.0, goal[1] + 1.0, "GOAL",
                  color="green", weight="bold")

    first = display_frames[0]
    agents = swarm_ax.scatter(
        first["position"][:, 0], first["position"][:, 1], s=70,
        c=[STATE_COLORS[x] for x in first["state"]],
        edgecolors="black", linewidths=0.6)
    arrows = swarm_ax.quiver(
        first["position"][:, 0], first["position"][:, 1],
        np.cos(first["heading"]), np.sin(first["heading"]),
        color="black", alpha=0.55, scale=50, width=0.002)
    leader_halos = swarm_ax.scatter([], [], facecolors="none",
                                     edgecolors="#d62728", linewidths=2.5, s=225)
    duel_rings = swarm_ax.scatter([], [], facecolors="none",
                                   edgecolors="#ffbf00", linewidths=1.8, s=165)
    audit_rings = swarm_ax.scatter([], [], facecolors="none",
                                    edgecolors="#2ca02c", linewidths=1.8,
                                    linestyles="--", s=285)
    status_text = swarm_ax.text(
        0.012, 0.97, "", transform=swarm_ax.transAxes, va="top",
        family="monospace", fontsize=7.6,
        bbox=dict(facecolor="white", alpha=0.92, edgecolor="#888",
                  boxstyle="round"), zorder=20)
    handles = [Line2D([0], [0], marker="o", linestyle="None",
                      markerfacecolor=c, markeredgecolor="black",
                      label=n) for n, c in zip(STATE_NAMES, STATE_COLORS)]
    handles += [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none",
               markeredgecolor="#ffbf00", label="DUEL"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none",
               markeredgecolor="#2ca02c", label="LEADER VALIDATION"),
    ]
    swarm_ax.legend(handles=handles, ncol=4, loc="upper right", fontsize=6.7)
    swarm_ax.set_title(
        f"CB-ELE v4.4 — {config.width:.0f} m domain — "
        "mid-distance re-election, catch-up boost",
        weight="bold")

    rounds = table["round"].to_numpy()
    state_specs = [
        ("candidate_count", STATE_COLORS[CANDIDATE], "Candidates"),
        ("passive_count", STATE_COLORS[PASSIVE], "Passive"),
        ("follower_count", STATE_COLORS[FOLLOWER], "Followers"),
        ("leader_count", STATE_COLORS[LEADER], "Leaders"),
        ("challenger_count", STATE_COLORS[CHALLENGER], "Challengers"),
    ]
    state_lines = [state_ax.plot([], [], color=c, label=l)[0]
                   for _, c, l in state_specs]
    state_cursor = state_ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    state_ax.set(xlim=(0, config.rounds - 1), ylim=(0, config.n_agents + 1),
                 xlabel="Round", ylabel="Count", title="Protocol states")
    state_ax.grid(True, linestyle=":", alpha=0.3)
    state_ax.legend(fontsize=6.7)

    performance_specs = [
        ("heading_variance", "purple", "Heading variance"),
        ("goal_alignment", "green", "Goal alignment"),
        ("normalized_mean_speed", "black", "Normalized speed"),
        ("largest_component_fraction", "blue", "Largest component"),
        ("leader_reachable_fraction", "orange", "Leader reachable"),
    ]
    performance_lines = [performance_ax.plot([], [], color=c, label=l)[0]
                         for _, c, l in performance_specs]
    performance_cursor = performance_ax.axvline(0, color="black",
                                                 linestyle="--", alpha=0.5)
    performance_ax.set(xlim=(0, config.rounds - 1), ylim=(-1.05, 1.05),
                       xlabel="Round", ylabel="Normalized value",
                       title="Coordination and connectivity")
    performance_ax.grid(True, linestyle=":", alpha=0.3)
    performance_ax.legend(fontsize=6.7, loc="lower right")

    def update(frame_idx):
        f = display_frames[frame_idx]
        m = f["metrics"]
        agents.set_offsets(f["position"])
        agents.set_facecolors([STATE_COLORS[x] for x in f["state"]])
        arrows.set_offsets(f["position"])
        arrows.set_UVC(np.cos(f["heading"]), np.sin(f["heading"]))
        leader_mask = f["state"] == LEADER
        leader_halos.set_offsets(f["position"][leader_mask])
        leader_halos.set_visible(bool(leader_mask.any()))
        duel_rings.set_offsets(f["position"][f["duel_active"]])
        duel_rings.set_visible(bool(f["duel_active"].any()))
        audit_rings.set_offsets(f["position"][f["audit_mask"]])
        audit_rings.set_visible(bool(f["audit_mask"].any()))
        status_text.set_text(
            f"t={f['round']:4d}  C/P/F/L/CH="
            f"{m['candidate_count']:2d}/{m['passive_count']:2d}/"
            f"{m['follower_count']:2d}/{m['leader_count']:2d}/"
            f"{m['challenger_count']:2d}\n"
            f"leaders={m['leader_ids']}  "
            f"validation={m['leaders_auditing']:2d}  "
            f"contacts={m['election_contacts']:2d}\n"
            f"hard viol={m['hard_safety_violations']:2d}  "
            f"min d={m['minimum_distance']:.2f} m\n"
            f"components={m['component_count']:2d}  "
            f"leader reach={m['leader_reachable_fraction']:.2f}\n"
            f"HV={m['heading_variance']:.3f}  "
            f"GA={m['goal_alignment']:+.3f}  "
            f"speed={m['mean_speed']:.3f}  "
            f"goal d={m['centroid_distance']:.1f} m\n"
            f"gate={m['leader_gate']:.2f}  "
            f"tether={m['leader_tether']:.2f}  "
            f"agree={m['leader_agreement']:+.2f}  "
            f"mission={m['mission_success']}"
        )
        end = f["round"] + 1
        for line, (col, _, _) in zip(state_lines, state_specs):
            line.set_data(rounds[:end], table[col].to_numpy()[:end])
        for line, (col, _, _) in zip(performance_lines, performance_specs):
            line.set_data(rounds[:end], table[col].to_numpy()[:end])
        state_cursor.set_xdata([f["round"]] * 2)
        performance_cursor.set_xdata([f["round"]] * 2)
        return (agents, arrows, leader_halos, duel_rings, audit_rings,
                status_text, *state_lines, *performance_lines,
                state_cursor, performance_cursor)

    anim = FuncAnimation(fig, update, frames=len(display_frames),
                         interval=1000 * config.animation_stride / config.fps,
                         blit=False)
    if config.save_animation:
        save_animation_file(anim, config)
    if config.show_inline and _IN_NOTEBOOK:
        from IPython.display import HTML, display
        if config.prefer_html5_video:
            try:
                display(HTML(anim.to_html5_video()))
            except Exception as e:
                print(f"HTML5 video unavailable ({e}); using JS frames.")
                display(HTML(anim.to_jshtml(default_mode="once")))
        else:
            display(HTML(anim.to_jshtml(default_mode="once")))
    plt.close(fig)
    return anim, display_frames, table, events, transitions


def run_seed_sweep(seeds=range(10), config=CONFIG, rounds=None):
    config = _as_config(config)
    rows = []
    base = asdict(config)
    for seed in seeds:
        settings = {**base, "seed": int(seed), "show_inline": False,
                    "save_csv": False, "verbose_timing": False}
        if rounds is not None:
            settings["rounds"] = int(rounds)
        cfg = Config(**settings)
        _, _, table, event_table, trans_table = run_simulation(cfg)
        summary = summarize_run(table, cfg, event_table, trans_table).to_dict()
        summary["seed"] = int(seed)
        rows.append(summary)
    columns = ["seed"] + [k for k in rows[0] if k != "seed"]
    result = pd.DataFrame(rows)[columns]
    if config.save_csv:
        os.makedirs(config.output_dir, exist_ok=True)
        sweep_path = os.path.join(config.output_dir,
                                  f"{config.csv_prefix}_sweep.csv")
        result.to_csv(sweep_path, index=False)
        print(f"Sweep results written to {sweep_path}")
    return result


# =============================================================================
# MAIN
# =============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CB-ELE v4.4 UAV-swarm leader election (headless runner).")
    parser.add_argument("--rounds", type=int, default=CONFIG.rounds)
    parser.add_argument("--n-agents", type=int, default=CONFIG.n_agents)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--output-dir", default=CONFIG.output_dir)
    parser.add_argument("--animation-file", default=CONFIG.animation_file,
                        help="output name; .gif (default) or .mp4 (needs ffmpeg)")
    parser.add_argument("--no-animation", action="store_true",
                        help="run the simulation and write CSV only")
    parser.add_argument("--no-reelection", action="store_true",
                        help="disable the mid-distance re-election experiment")
    parser.add_argument("--recognition", choices=["off", "observe", "follow"],
                        default=CONFIG.leader_recognition,
                        help="behavioural leader recognition (v0.5)")
    parser.add_argument("--informed", type=float,
                        default=CONFIG.follower_informed_fraction,
                        help="fraction of followers that steer to the goal")
    parser.add_argument("--sweep", type=int, metavar="N", default=0,
                        help="run a seed sweep over seeds 0..N-1 instead")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = replace(
        CONFIG,
        rounds=args.rounds,
        n_agents=args.n_agents,
        seed=args.seed,
        output_dir=args.output_dir,
        animation_file=args.animation_file,
        reelection_enabled=not args.no_reelection,
        leader_recognition=args.recognition,
        follower_informed_fraction=args.informed,
        show_inline=False,
    )
    if args.sweep > 0:
        print(run_seed_sweep(range(args.sweep), config).to_string())
    elif args.no_animation:
        _, goal, table, events, transitions = run_simulation(config)
        summary = summarize_run(table, config, events, transitions)
        if config.save_csv:
            print_saved_paths(save_results_csv(
                table, events, summary, goal, config, transitions))
        print("\nRun summary:")
        print(summary.to_string())
    else:
        create_animation(config)
    print("\nDone.")


if __name__ == "__main__":
    main()
