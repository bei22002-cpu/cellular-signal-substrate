"""
PHASE9_COUNTERFACTUAL_WORLD_MODEL_AND_SCENARIO_SELECTION

Extends Phase 8 with scenario_a / scenario_b counterfactual channels, local world
modeling (world_state_estimate, external_error), and scenario_preference-driven
control — deterministic, fully local, single-file, tkinter.
"""

from __future__ import annotations

import time
import tkinter as tk
from collections import Counter, defaultdict, deque
from dataclasses import dataclass


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@dataclass(frozen=True)
class WorldParams:
    width: int = 50
    height: int = 50
    diffusion_coeff: float = 0.14
    decay_coeff: float = 0.990
    epsilon: float = 1e-6
    memory_decay: float = 0.97
    memory_gain: float = 0.10
    memory_cap: float = 5.0
    role_high_threshold: float = 1.42
    role_low_threshold: float = 0.42
    amplifier_boost: float = 1.06
    amplifier_adaptation_scale: float = 0.02
    dampener_factor: float = 0.88
    dampener_low_path_scale: float = 0.10
    dampener_neg_adapt_scale: float = 0.07
    router_bias_base: float = 1.02
    router_path_weight_scale: float = 0.13
    router_retain: float = 0.06
    memory_trace_scale: float = 0.08
    gate_effect_scale: float = 0.11
    relay_scale: float = 0.14
    binder_scale: float = 0.10
    output_sensitivity: float = 1.00
    signal_cap: float = 50.0
    center_pulse_value: float = 10.0
    injection_value: float = 1.0
    scale_factor: float = 10.0
    cluster_threshold: float = 0.76
    cluster_score_decay: float = 0.985
    cluster_support_gain: float = 0.13
    cluster_score_cap: float = 10.0
    cluster_persistence_size_min: int = 3
    output_zone_width: int = 4
    output_weight: float = 1.5
    decision_margin: float = 0.20
    output_center_band: float = 0.22
    adaptation_decay: float = 0.995
    adaptation_gain: float = 0.038
    adaptation_cap: float = 2.0
    path_weight_decay: float = 0.992
    path_weight_gain: float = 0.026
    path_weight_min: float = 0.50
    path_weight_max: float = 3.0
    eligibility_decay: float = 0.92
    eligibility_gain: float = 0.12
    eligibility_cap: float = 3.0
    eligibility_after_reward_decay: float = 0.55
    reward_gain_scale: float = 1.0
    wt_reinforce_scale: float = 0.014
    gb_reinforce_scale: float = 0.011
    relay_reinforce_scale: float = 0.018
    bind_reinforce_scale: float = 0.015
    working_trace_decay: float = 0.94
    working_trace_gain: float = 0.085
    working_trace_cap: float = 6.0
    memory_role_wt_decay_bonus: float = 0.985
    local_memory_neighbor_gain: float = 0.038
    gate_bias_decay: float = 0.97
    gate_bias_gain: float = 0.042
    gate_bias_min: float = -1.0
    gate_bias_max: float = 1.0
    relay_bias_decay: float = 0.96
    relay_bias_gain: float = 0.05
    relay_bias_cap: float = 1.5
    bind_affinity_decay: float = 0.97
    bind_affinity_gain: float = 0.04
    bind_affinity_cap: float = 2.0
    tissue_score_cap: float = 15.0
    tissue_merge_flow_threshold: float = 0.15
    tissue_role_compat_bonus: float = 0.35
    layer_input_max_x_frac: float = 0.22
    layer_inter_max_x_frac: float = 0.45
    layer_context_max_x_frac: float = 0.72
    router_eligibility_threshold: float = 0.40
    router_path_weight_threshold: float = 1.0
    gate_imbalance_threshold: float = 0.17
    memory_role_wt_threshold: float = 1.75
    memory_role_mem_threshold: float = 1.08
    episode_steps: int = 55
    rest_steps: int = 14
    rest_eligibility_decay: float = 0.88
    seq_phase1_steps: int = 40
    seq_gap_steps: int = 10
    seq_phase2_steps: int = 40
    seq_gap_long: int = 28
    path_component_threshold: float = 1.0
    input_zone_x_max: int = 3
    # Phase 7 meta-control
    goal_signal_decay: float = 0.94
    goal_signal_gain: float = 0.22
    goal_signal_cap: float = 2.5
    goal_scale: float = 0.08
    goal_neighbor_mix: float = 0.14
    meta_state_decay: float = 0.96
    meta_state_gain: float = 0.12
    meta_state_cap: float = 3.0
    control_weight_decay: float = 0.97
    control_weight_gain: float = 0.06
    control_weight_cap: float = 2.0
    stability_bias_decay: float = 0.985
    exploration_bias_decay: float = 0.98
    stab_expl_gain: float = 0.04
    meta_path_mod_scale: float = 0.035
    adaptive_switch_step: int = 120
    disturbance_period: int = 15
    disturbance_amp: float = 0.35
    # Phase 8 predictive self-model
    prediction_error_decay: float = 0.92
    prediction_error_cap: float = 3.0
    model_confidence_decay: float = 0.97
    model_confidence_gain: float = 0.08
    model_confidence_cap: float = 1.0
    prediction_bias: float = 0.06
    prediction_error_adapt_scale: float = 0.032
    pred_path_damp_scale: float = 0.02
    high_confidence_threshold: float = 0.55
    # Phase 9 counterfactual / world model
    scenario_a_bias: float = 1.012
    scenario_b_bias: float = 0.988
    scenario_pref_scale: float = 0.07
    world_state_decay: float = 0.96
    world_state_gain: float = 0.08
    world_state_corr: float = 0.06
    external_error_decay: float = 0.92
    consequence_trace_decay: float = 0.97
    consequence_gain: float = 0.045
    env_target_velocity: int = 1
    env_context_period: int = 95
    obstacle_cycle: int = 22


@dataclass(slots=True)
class Cell:
    signal: float = 0.0
    memory: float = 0.0
    role: int = 0
    cluster_id: int = 0
    cluster_score: float = 0.0
    adaptation: float = 0.0
    path_weight: float = 1.0
    eligibility: float = 0.0
    working_trace: float = 0.0
    gate_bias: float = 0.0
    tissue_id: int = 0
    tissue_score: float = 0.0
    layer_id: int = 0
    relay_bias: float = 0.5
    bind_affinity: float = 0.5
    goal_signal: float = 0.0
    meta_state: float = 0.0
    control_weight: float = 0.5
    stability_bias: float = 0.5
    exploration_bias: float = 0.5
    predicted_signal: float = 0.0
    prediction_error: float = 0.0
    simulation_flag: float = 0.0
    model_confidence: float = 0.5
    temporal_phase: int = 0
    scenario_a_signal: float = 0.0
    scenario_b_signal: float = 0.0
    world_state_estimate: float = 0.5
    external_error: float = 0.0
    scenario_score_a: float = 0.0
    scenario_score_b: float = 0.0
    scenario_preference: float = 0.0
    consequence_trace: float = 0.5


class Phase9World:
    MODE_CENTER_PULSE = "CENTER_PULSE"
    MODE_LEFT_STREAM = "LEFT_STREAM"
    MODE_CLEAR = "CLEAR"
    MODE_INPUT_A = "INPUT_A"
    MODE_INPUT_B = "INPUT_B"
    MODE_INPUT_AB = "INPUT_AB"
    TASK_A_THEN_B = "A_THEN_B"
    TASK_B_THEN_A = "B_THEN_A"
    TASK_A_GAP_B = "A_GAP_B"
    TASK_B_GAP_A = "B_GAP_A"
    TASK_COND_ROUTE = "COND_ROUTE"
    TASK_DELAY_MATCH = "DELAY_MATCH"
    TASK_MULTI_PATH = "MULTI_PATH"
    TASK_ADAPTIVE = "ADAPTIVE"
    TASK_COMPETING = "COMPETING"
    TASK_DISTURBANCE = "DISTURBANCE"
    TASK_DELAYED_RESPONSE = "DELAYED_RESPONSE"
    TASK_PATTERN_CONTINUATION = "PATTERN_CONTINUATION"
    TASK_INTERRUPTED_SIGNAL = "INTERRUPTED_SIGNAL"
    TASK_NOISE_FILTER = "NOISE_FILTER"
    TASK_MOVING_TARGET = "MOVING_TARGET"
    TASK_OBSTACLE_NAV = "OBSTACLE_NAV"
    TASK_CONTEXT_SWITCH = "CONTEXT_SWITCH"
    TASK_EXPLORE_EXPLOIT = "EXPLORE_EXPLOIT"

    RUN_FREE = "FREE_RUN"
    RUN_TRAINING = "TRAINING"
    RUN_TRAINING_SEQUENTIAL = "TRAINING_SEQUENTIAL"
    RUN_TRAIN_COND = "TRAIN_COND"
    RUN_TRAIN_DELAY = "TRAIN_DELAY"
    RUN_TRAIN_PHASE7 = "TRAIN_PHASE7"
    RUN_TRAIN_PHASE8 = "TRAIN_PHASE8"
    RUN_TRAIN_PHASE9 = "TRAIN_PHASE9"
    RUN_EVAL_SEQUENTIAL = "EVAL_SEQUENTIAL"

    LAYER_INPUT = 0
    LAYER_INTERMEDIATE = 1
    LAYER_CONTEXT = 2
    LAYER_OUTPUT = 3

    ROLE_NEUTRAL = 0
    ROLE_AMPLIFIER = 1
    ROLE_DAMPENER = 2
    ROLE_ROUTER = 3
    ROLE_OUTPUT = 4
    ROLE_MEMORY = 5
    ROLE_GATE = 6
    ROLE_BINDER = 7
    ROLE_RELAY = 8

    def __init__(self, params: WorldParams | None = None) -> None:
        self.p = params or WorldParams()
        self.grid_current = [[Cell() for _ in range(self.p.width)] for _ in range(self.p.height)]
        self.grid_next = [[Cell() for _ in range(self.p.width)] for _ in range(self.p.height)]
        self.mode = self.MODE_CENTER_PULSE
        self.current_task = self.TASK_A_THEN_B
        self.run_mode = self.RUN_FREE
        self.training_mode_idx = 0
        self.step_count = 0
        self.reinforcement_enabled = True
        self.training_phase = 0
        self.phase_step = 0
        self.seq_phase = 0
        self.seq_step = 0
        self._seq_gap_len = 0
        self.delay_match_first_is_a = True

        self.cluster_sizes: dict[int, int] = {}
        self.cluster_dominant_role: dict[int, int] = {}
        self.cluster_avg_wt: dict[int, float] = {}
        self.cluster_avg_cs: dict[int, float] = {}
        self.cluster_transition_flow_score = 0.0

        self.tissue_sizes: dict[int, int] = {}
        self.tissue_cluster_count: dict[int, int] = {}
        self.tissue_avg_score: dict[int, float] = {}
        self.inter_tissue_edge_count = 0
        self.average_inter_tissue_flow = 0.0
        self.strongest_tissue_path_score = 0.0

        self.output_top_value = 0.0
        self.output_bottom_value = 0.0
        self.output_context_value = 0.0
        self.current_decision = "NONE"

        self.total_episodes = 0
        self.correct_episodes = 0
        self.sequential_correct = 0
        self.sequential_total = 0
        self.conditional_correct = 0
        self.conditional_total = 0
        self.delay_match_correct = 0
        self.delay_match_total = 0
        self.reward_history: deque[float] = deque(maxlen=80)
        self.last_episode_reward = 0.0
        self.routing_bias_score = 0.0
        self.last_delta_b_after_a = 0.0
        self.last_delta_b_cold = 0.0
        self.context_dependency_score = 0.0
        self.working_memory_retention_score = 0.0
        self.hierarchical_dependency_score = 0.0

        self.active_path_count = 0
        self.largest_path_size = 0
        self.strongest_path_score = 0.0
        self.path_to_top_exists = False
        self.path_to_bottom_exists = False
        self.persistent_cluster_count = 0
        self.memory_cluster_count = 0
        self.gate_cluster_count = 0
        self.routing_cluster_count = 0
        self.binder_cluster_count = 0
        self.relay_cluster_count = 0
        self.active_tissue_count = 0
        self.persistent_tissue_count = 0
        self.memory_tissue_count = 0
        self.relay_tissue_count = 0
        self.gate_tissue_count = 0
        self.output_tissue_count = 0

        self.layer_activity = [0.0, 0.0, 0.0, 0.0]
        self._prev_total_signal = 0.0
        self._feed_output_error = 0.0
        self._feed_instability = 0.0
        self._last_reward_ma = 0.0
        self._path_weight_mean_prev = 1.0
        self.pathway_reconfiguration_rate = 0.0
        self.strategy_variation_score = 0.0
        self.recovery_time_after_disturbance = 0.0
        self._disturbance_step_marker = -1
        self._steps_since_disturb = 0
        self._episode_path_signatures: deque[tuple[int, int, str]] = deque(maxlen=32)
        self._prev_pref_sign = 0.0
        self._prev_pref_val = 0.0
        self.route_switch_latency = 0.0
        self.world_model_error_ema = 0.0
        self.output_accuracy_ema = 0.0
        self.reset(self.MODE_CENTER_PULSE, full=True)

    def layer_from_x(self, x: int) -> int:
        w = self.p.width
        f = x / max(w - 1, 1)
        if f < self.p.layer_input_max_x_frac:
            return self.LAYER_INPUT
        if f < self.p.layer_inter_max_x_frac:
            return self.LAYER_INTERMEDIATE
        if f < self.p.layer_context_max_x_frac:
            return self.LAYER_CONTEXT
        return self.LAYER_OUTPUT

    def _zero_cell(self, c: Cell, full: bool) -> None:
        c.signal = 0.0
        c.memory = 0.0
        c.role = self.ROLE_NEUTRAL
        c.cluster_id = 0
        c.cluster_score = 0.0
        c.tissue_id = 0
        c.tissue_score = 0.0
        c.layer_id = 0
        if full:
            c.adaptation = 0.0
            c.path_weight = 1.0
            c.eligibility = 0.0
            c.working_trace = 0.0
            c.gate_bias = 0.0
            c.relay_bias = 0.5
            c.bind_affinity = 0.5
            c.goal_signal = 0.0
            c.meta_state = 0.0
            c.control_weight = 0.5
            c.stability_bias = 0.5
            c.exploration_bias = 0.5
            c.predicted_signal = 0.0
            c.prediction_error = 0.0
            c.simulation_flag = 0.0
            c.model_confidence = 0.5
            c.temporal_phase = 0
            c.scenario_a_signal = 0.0
            c.scenario_b_signal = 0.0
            c.world_state_estimate = 0.5
            c.external_error = 0.0
            c.scenario_score_a = 0.0
            c.scenario_score_b = 0.0
            c.scenario_preference = 0.0
            c.consequence_trace = 0.5

    def reset(self, mode: str, full: bool = True) -> None:
        self.mode = mode
        self.step_count = 0
        self.seq_phase = 0
        self.seq_step = 0
        if full:
            self.training_phase = 0
            self.phase_step = 0
        for y in range(self.p.height):
            for x in range(self.p.width):
                self._zero_cell(self.grid_current[y][x], full=full)
                self._zero_cell(self.grid_next[y][x], full=full)
        if mode == self.MODE_CENTER_PULSE:
            cx, cy = self.p.width // 2, self.p.height // 2
            self.grid_current[cy][cx].signal = self.p.center_pulse_value
        if full:
            self._prev_total_signal = 0.0
            self._path_weight_mean_prev = 1.0
            self._episode_path_signatures.clear()
            self.strategy_variation_score = 0.0
            self.recovery_time_after_disturbance = 0.0
        self._set_seq_gap_for_task()
        self._post_step_analysis()

    def _set_seq_gap_for_task(self) -> None:
        if self.current_task in (
            self.TASK_A_GAP_B,
            self.TASK_B_GAP_A,
            self.TASK_DELAY_MATCH,
            self.TASK_DELAYED_RESPONSE,
            self.TASK_INTERRUPTED_SIGNAL,
        ):
            self._seq_gap_len = self.p.seq_gap_long
        else:
            self._seq_gap_len = self.p.seq_gap_steps

    def partial_reset_signals_only(self) -> None:
        for y in range(self.p.height):
            for x in range(self.p.width):
                c = self.grid_current[y][x]
                c.signal = 0.0
                c.cluster_id = 0
                c.tissue_id = 0
                c.eligibility *= 0.90
                c.working_trace *= 0.92
                c.goal_signal *= 0.85
                c.meta_state *= 0.92
                c.predicted_signal *= 0.88
                c.prediction_error *= 0.9
                c.model_confidence = clamp(c.model_confidence * 0.96 + 0.02, 0.0, 1.0)
                c.scenario_a_signal *= 0.88
                c.scenario_b_signal *= 0.88
                c.world_state_estimate = clamp(c.world_state_estimate * 0.97 + 0.015, 0.0, 1.0)
                c.consequence_trace *= 0.95
                self.grid_next[y][x].signal = 0.0
                self.grid_next[y][x].cluster_id = 0
        self._post_step_analysis()

    def inject_signal_left_stream(self) -> None:
        if self.p.width < 2:
            return
        v = self.p.injection_value
        for y in range(self.p.height):
            self.grid_current[y][1].signal += v

    def inject_input_a(self) -> None:
        if self.p.width < 2:
            return
        v = self.p.injection_value
        y_max = max(1, self.p.height // 3)
        for y in range(0, y_max):
            self.grid_current[y][1].signal += v

    def _apply_disturbance_perturbation(self) -> None:
        if self.mode != self.TASK_DISTURBANCE:
            return
        if self.step_count % self.p.disturbance_period != 0:
            return
        self._disturbance_step_marker = self.step_count
        amp = self.p.disturbance_amp * self.p.injection_value
        w, h = self.p.width, self.p.height
        for y in range(h):
            for x in range(w):
                if (x + y + self.step_count) % 9 == 0:
                    sgn = 1.0 if (x + y) % 2 == 0 else -1.0
                    self.grid_current[y][x].signal += amp * sgn

    def inject_input_b(self) -> None:
        if self.p.width < 2:
            return
        v = self.p.injection_value
        y_min = (2 * self.p.height) // 3
        for y in range(y_min, self.p.height):
            self.grid_current[y][1].signal += v

    def inject_at(self, x: int, y: int, amount: float | None = None) -> None:
        if 0 <= x < self.p.width and 0 <= y < self.p.height:
            self.grid_current[y][x].signal += self.p.injection_value if amount is None else amount

    def _inject_interrupt_noise(self) -> None:
        w, h = self.p.width, self.p.height
        if w < 2:
            return
        v = self.p.injection_value
        for y in range(h):
            for x in range(1, min(5, w)):
                if (x + y + self.step_count) % 3 != 0:
                    sgn = 1.0 if (x + y + self.step_count) % 2 == 0 else -1.0
                    self.grid_current[y][x].signal += 0.45 * v * sgn

    def _inject_phase8_noise(self, amp: float) -> None:
        w, h = self.p.width, self.p.height
        if w < 2:
            return
        v = self.p.injection_value
        for y in range(h):
            for x in range(1, min(6, w)):
                q = ((x * 13 + y * 7 + self.step_count * 11) % 7) - 3
                self.grid_current[y][x].signal += amp * v * q * 0.2

    def _sequential_inject(self) -> None:
        t = self.current_task
        if t == self.TASK_DELAY_MATCH:
            if self.seq_phase == 0:
                (self.inject_input_a if self.delay_match_first_is_a else self.inject_input_b)()
            elif self.seq_phase == 2:
                (self.inject_input_a if self.delay_match_first_is_a else self.inject_input_b)()
            return
        if t == self.TASK_COMPETING:
            if self.seq_phase != 1:
                self.inject_input_a()
                self.inject_input_b()
            return
        if t == self.TASK_INTERRUPTED_SIGNAL:
            if self.seq_phase == 0:
                self.inject_input_a()
            elif self.seq_phase == 1:
                self._inject_interrupt_noise()
            else:
                self.inject_input_b()
            return
        if t == self.TASK_NOISE_FILTER:
            if self.seq_phase != 1:
                self.inject_input_a()
                self._inject_phase8_noise(0.22)
            return
        phase_a = (
            self.TASK_A_THEN_B,
            self.TASK_A_GAP_B,
            self.TASK_COND_ROUTE,
            self.TASK_MULTI_PATH,
            self.TASK_ADAPTIVE,
            self.TASK_DISTURBANCE,
            self.TASK_DELAYED_RESPONSE,
            self.TASK_PATTERN_CONTINUATION,
            self.TASK_MOVING_TARGET,
            self.TASK_OBSTACLE_NAV,
            self.TASK_CONTEXT_SWITCH,
            self.TASK_EXPLORE_EXPLOIT,
        )
        if self.seq_phase == 0:
            if t in phase_a:
                self.inject_input_a()
            else:
                self.inject_input_b()
        elif self.seq_phase == 1:
            pass
        else:
            if t in phase_a:
                self.inject_input_b()
            else:
                self.inject_input_a()

    def _injection_free(self) -> None:
        m = self.mode
        if m == self.MODE_LEFT_STREAM:
            self.inject_signal_left_stream()
        elif m == self.MODE_INPUT_A:
            self.inject_input_a()
        elif m == self.MODE_INPUT_B:
            self.inject_input_b()
        elif m == self.MODE_INPUT_AB:
            self.inject_input_a()
            self.inject_input_b()

    def _apply_injection(self) -> None:
        seq_modes = (
            self.TASK_A_THEN_B,
            self.TASK_B_THEN_A,
            self.TASK_A_GAP_B,
            self.TASK_B_GAP_A,
            self.TASK_COND_ROUTE,
            self.TASK_DELAY_MATCH,
            self.TASK_MULTI_PATH,
            self.TASK_ADAPTIVE,
            self.TASK_COMPETING,
            self.TASK_DISTURBANCE,
            self.TASK_DELAYED_RESPONSE,
            self.TASK_PATTERN_CONTINUATION,
            self.TASK_INTERRUPTED_SIGNAL,
            self.TASK_NOISE_FILTER,
            self.TASK_MOVING_TARGET,
            self.TASK_OBSTACLE_NAV,
            self.TASK_CONTEXT_SWITCH,
            self.TASK_EXPLORE_EXPLOIT,
        )
        if self.mode in seq_modes:
            self._sequential_inject()
            return
        if self.run_mode == self.RUN_TRAINING and self.mode not in seq_modes:
            if self.training_phase == 0:
                self.inject_input_a()
            elif self.training_phase == 2:
                self.inject_input_b()
            return
        self._injection_free()

    def _cross_cluster_flux(self, x: int, y: int, cur: list[list[Cell]]) -> float:
        h, w = self.p.height, self.p.width
        c = cur[y][x]
        s = 0.0
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                n = cur[ny][nx]
                if n.cluster_id != c.cluster_id and c.cluster_id > 0 and n.cluster_id > 0:
                    s += min(abs(c.signal), abs(n.signal))
        return s

    def _cross_cluster_flux_pred(self, x: int, y: int, cur: list[list[Cell]]) -> float:
        h, w = self.p.height, self.p.width
        c = cur[y][x]
        s = 0.0
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                n = cur[ny][nx]
                if n.cluster_id != c.cluster_id and c.cluster_id > 0 and n.cluster_id > 0:
                    s += min(abs(c.predicted_signal), abs(n.predicted_signal))
        return s

    def _cross_cluster_flux_attr(self, x: int, y: int, cur: list[list[Cell]], attr: str) -> float:
        h, w = self.p.height, self.p.width
        c = cur[y][x]
        s = 0.0
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                n = cur[ny][nx]
                if n.cluster_id != c.cluster_id and c.cluster_id > 0 and n.cluster_id > 0:
                    s += min(abs(getattr(c, attr)), abs(getattr(n, attr)))
        return s

    def env_moving_target_y(self) -> int:
        h = self.p.height
        return int((self.step_count * self.p.env_target_velocity) % max(h, 1))

    def _local_env_sensed(self, y: int, x: int) -> float:
        h, w = self.p.height, self.p.width
        ty = self.env_moving_target_y()
        dist = abs(y - ty) / max(h, 1)
        base = clamp(1.0 - dist, 0.0, 1.0)
        cx = (self.step_count // max(1, self.p.obstacle_cycle)) % max(w, 1)
        obs = 1.0 if abs(x - int(cx)) <= 2 and y > h // 5 else 0.0
        return clamp(base * (1.0 - 0.42 * obs), 0.0, 1.0)

    def env_prefer_top(self) -> bool:
        return (self.step_count // max(1, self.p.env_context_period)) % 2 == 0

    def _evolve_channel_value(
        self,
        cs: float,
        up: float,
        down: float,
        left: float,
        right: float,
        cr: int,
        ada: float,
        pw: float,
        cwt: float,
        gb: float,
        rbi: float,
        baf: float,
        cross_flux: float,
        sig_cap: float,
        wt_cap: float,
        gb_lo: float,
        gb_hi: float,
        rb_cap: float,
        ba_cap: float,
        eps: float,
        bias_mult: float = 1.0,
    ) -> float:
        d, decay = self.p.diffusion_coeff, self.p.decay_coeff
        neighbor_avg = (up + down + left + right) / 4.0
        v = (cs + d * (neighbor_avg - cs)) * decay
        pw_cl = clamp(pw, self.p.path_weight_min, self.p.path_weight_max)
        pw_max = self.p.path_weight_max
        amp_boost = self.p.amplifier_boost
        amp_ada = self.p.amplifier_adaptation_scale
        damp_base = self.p.dampener_factor
        damp_lp = self.p.dampener_low_path_scale
        damp_na = self.p.dampener_neg_adapt_scale
        rb = self.p.router_bias_base
        rpw = self.p.router_path_weight_scale
        router_retain = self.p.router_retain
        mts = self.p.memory_trace_scale
        ges = self.p.gate_effect_scale
        rs = self.p.relay_scale
        bs = self.p.binder_scale
        out_sense = self.p.output_sensitivity

        if cr == self.ROLE_AMPLIFIER:
            v *= amp_boost + clamp(ada, -1.0, 1.0) * amp_ada
        elif cr == self.ROLE_DAMPENER:
            low_pw = clamp((pw_max - pw) / max(pw_max, 1e-6), 0.0, 1.0)
            neg_a = clamp(-ada, 0.0, 2.0)
            v *= damp_base * (1.0 + damp_lp * low_pw) * (1.0 + damp_na * neg_a)
        elif cr == self.ROLE_ROUTER:
            mult = 1.0 + pw_cl * rpw
            v = (v * mult * rb) + router_retain * (cs - v)
        elif cr == self.ROLE_MEMORY:
            v *= 1.0 + clamp(cwt, 0.0, wt_cap) * mts
        elif cr == self.ROLE_GATE:
            v *= 1.0 + clamp(gb, gb_lo, gb_hi) * ges
        elif cr == self.ROLE_RELAY:
            v *= 1.0 + clamp(rbi, 0.0, rb_cap) * rs
            v = (v * 1.02) + 0.04 * cross_flux / (sig_cap + 1e-6)
        elif cr == self.ROLE_BINDER:
            v *= 1.0 + clamp(baf, 0.0, ba_cap) * bs * 0.1
        elif cr == self.ROLE_OUTPUT:
            v *= out_sense

        v *= bias_mult
        v = clamp(v, -sig_cap, sig_cap)
        if abs(v) < eps:
            v = 0.0
        return v

    def roles_compatible(self, r1: int, r2: int) -> bool:
        if r1 == r2:
            return True
        a, b = (r1, r2) if r1 < r2 else (r2, r1)
        pairs = {(1, 8), (3, 5), (3, 7), (3, 8), (5, 6), (6, 7)}
        return (a, b) in pairs or abs(r1 - r2) <= 1

    def role_update_policy(
        self,
        x: int,
        y: int,
        new_memory: float,
        new_signal: float,
        new_wt: float,
        cross_flux: float,
        cur: list[list[Cell]],
    ) -> int:
        c = cur[y][x]
        el, pw, ada = c.eligibility, c.path_weight, c.adaptation
        ba = c.bind_affinity
        wt = new_wt
        rb = c.relay_bias

        if x >= self.p.width - self.p.output_zone_width and new_memory >= self.p.role_high_threshold:
            return self.ROLE_OUTPUT

        if ba >= 1.2 and c.cluster_score >= 0.8 and new_memory >= self.p.cluster_threshold:
            return self.ROLE_BINDER

        if cross_flux >= 0.4 * self.p.signal_cap and pw >= self.p.router_path_weight_threshold and el >= 0.35:
            return self.ROLE_RELAY

        if wt >= self.p.memory_role_wt_threshold and new_memory >= self.p.memory_role_mem_threshold:
            return self.ROLE_MEMORY

        if el >= self.p.router_eligibility_threshold and pw >= self.p.router_path_weight_threshold:
            return self.ROLE_ROUTER

        c_sig = c.signal
        up = cur[y - 1][x].signal if y > 0 else c_sig
        down = cur[y + 1][x].signal if y < self.p.height - 1 else c_sig
        imb = (up - down) / (self.p.signal_cap + 1e-6)
        if abs(imb) >= self.p.gate_imbalance_threshold and wt >= 0.35:
            return self.ROLE_GATE

        if new_memory >= self.p.role_high_threshold:
            gm = sum(
                abs(cur[y + dy][x + dx].signal - c_sig)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= y + dy < self.p.height and 0 <= x + dx < self.p.width
            ) * 0.25
            if gm >= 0.30:
                return self.ROLE_ROUTER
            return self.ROLE_AMPLIFIER

        if new_memory <= self.p.role_low_threshold and ada <= 0.22:
            return self.ROLE_DAMPENER
        return self.ROLE_NEUTRAL

    def compute_next_grid(self) -> None:
        w, h = self.p.width, self.p.height
        d, decay, eps = self.p.diffusion_coeff, self.p.decay_coeff, self.p.epsilon
        sig_cap = self.p.signal_cap
        mem_decay, mem_gain, mem_cap = self.p.memory_decay, self.p.memory_gain, self.p.memory_cap
        cs_decay, cs_gain, cs_cap = self.p.cluster_score_decay, self.p.cluster_support_gain, self.p.cluster_score_cap
        cluster_thr = self.p.cluster_threshold
        el_decay, el_gain, el_cap = self.p.eligibility_decay, self.p.eligibility_gain, self.p.eligibility_cap
        wt_decay, wt_gain, wt_cap = self.p.working_trace_decay, self.p.working_trace_gain, self.p.working_trace_cap
        mem_wt_bonus = self.p.memory_role_wt_decay_bonus
        nbr_wt_gain = self.p.local_memory_neighbor_gain
        gb_decay, gb_gain = self.p.gate_bias_decay, self.p.gate_bias_gain
        gb_lo, gb_hi = self.p.gate_bias_min, self.p.gate_bias_max
        rb_decay, rb_gain = self.p.relay_bias_decay, self.p.relay_bias_gain
        rb_cap = self.p.relay_bias_cap
        ba_decay, ba_gain = self.p.bind_affinity_decay, self.p.bind_affinity_gain
        ba_cap = self.p.bind_affinity_cap

        amp_boost = self.p.amplifier_boost
        amp_ada = self.p.amplifier_adaptation_scale
        damp_base = self.p.dampener_factor
        damp_lp = self.p.dampener_low_path_scale
        damp_na = self.p.dampener_neg_adapt_scale
        rb = self.p.router_bias_base
        rpw = self.p.router_path_weight_scale
        pw_max = self.p.path_weight_max
        router_retain = self.p.router_retain
        mts = self.p.memory_trace_scale
        ges = self.p.gate_effect_scale
        rs = self.p.relay_scale
        bs = self.p.binder_scale
        out_sense = self.p.output_sensitivity

        cur, nxt = self.grid_current, self.grid_next

        for y in range(h):
            row_cur, row_nxt = cur[y], nxt[y]
            for x in range(w):
                c = row_cur[x]
                cs, cm, cr, ccs = c.signal, c.memory, c.role, c.cluster_score
                ada, pw, elig = c.adaptation, c.path_weight, c.eligibility
                cwt, gb = c.working_trace, c.gate_bias
                rbi, baf = c.relay_bias, c.bind_affinity
                gs, ms, cw, sb, eb = c.goal_signal, c.meta_state, c.control_weight, c.stability_bias, c.exploration_bias
                lid = self.layer_from_x(x)

                up = cur[y - 1][x].signal if y > 0 else cs
                down = cur[y + 1][x].signal if y < h - 1 else cs
                left = row_cur[x - 1].signal if x > 0 else cs
                right = row_cur[x + 1].signal if x < w - 1 else cs

                neighbor_avg = (up + down + left + right) / 4.0
                new_signal = (cs + d * (neighbor_avg - cs)) * decay
                pw_cl = clamp(pw, self.p.path_weight_min, pw_max)
                cross_flux = self._cross_cluster_flux(x, y, cur)
                cross_flux_pred = self._cross_cluster_flux_pred(x, y, cur)
                ps = c.predicted_signal
                up_p = cur[y - 1][x].predicted_signal if y > 0 else ps
                down_p = cur[y + 1][x].predicted_signal if y < h - 1 else ps
                left_p = row_cur[x - 1].predicted_signal if x > 0 else ps
                right_p = row_cur[x + 1].predicted_signal if x < w - 1 else ps
                gmod = clamp(1.0 + gs * self.p.goal_scale, 0.55, 1.45)
                vp = self._evolve_channel_value(
                    ps,
                    up_p,
                    down_p,
                    left_p,
                    right_p,
                    cr,
                    ada,
                    pw,
                    cwt,
                    gb,
                    rbi,
                    baf,
                    cross_flux_pred,
                    sig_cap,
                    wt_cap,
                    gb_lo,
                    gb_hi,
                    rb_cap,
                    ba_cap,
                    eps,
                )
                vp = clamp(vp * gmod, -sig_cap, sig_cap)

                sa_sig = c.scenario_a_signal
                up_sa = cur[y - 1][x].scenario_a_signal if y > 0 else sa_sig
                down_sa = cur[y + 1][x].scenario_a_signal if y < h - 1 else sa_sig
                left_sa = row_cur[x - 1].scenario_a_signal if x > 0 else sa_sig
                right_sa = row_cur[x + 1].scenario_a_signal if x < w - 1 else sa_sig
                sb_sig = c.scenario_b_signal
                up_sb = cur[y - 1][x].scenario_b_signal if y > 0 else sb_sig
                down_sb = cur[y + 1][x].scenario_b_signal if y < h - 1 else sb_sig
                left_sb = row_cur[x - 1].scenario_b_signal if x > 0 else sb_sig
                right_sb = row_cur[x + 1].scenario_b_signal if x < w - 1 else sb_sig
                cross_flux_sa = self._cross_cluster_flux_attr(x, y, cur, "scenario_a_signal")
                cross_flux_sb = self._cross_cluster_flux_attr(x, y, cur, "scenario_b_signal")
                va = self._evolve_channel_value(
                    sa_sig,
                    up_sa,
                    down_sa,
                    left_sa,
                    right_sa,
                    cr,
                    ada,
                    pw,
                    cwt,
                    gb,
                    rbi,
                    baf,
                    cross_flux_sa,
                    sig_cap,
                    wt_cap,
                    gb_lo,
                    gb_hi,
                    rb_cap,
                    ba_cap,
                    eps,
                    self.p.scenario_a_bias,
                )
                va = clamp(va * gmod, -sig_cap, sig_cap)
                vb = self._evolve_channel_value(
                    sb_sig,
                    up_sb,
                    down_sb,
                    left_sb,
                    right_sb,
                    cr,
                    ada,
                    pw,
                    cwt,
                    gb,
                    rbi,
                    baf,
                    cross_flux_sb,
                    sig_cap,
                    wt_cap,
                    gb_lo,
                    gb_hi,
                    rb_cap,
                    ba_cap,
                    eps,
                    self.p.scenario_b_bias,
                )
                vb = clamp(vb * gmod, -sig_cap, sig_cap)
                var_a = ((sa_sig - up_sa) ** 2 + (sa_sig - down_sa) ** 2 + (sa_sig - left_sa) ** 2 + (sa_sig - right_sa) ** 2) * 0.25
                var_b = ((sb_sig - up_sb) ** 2 + (sb_sig - down_sb) ** 2 + (sb_sig - left_sb) ** 2 + (sb_sig - right_sb) ** 2) * 0.25
                norm_va = clamp(var_a / (sig_cap * sig_cap + 1e-6), 0.0, 2.0)
                norm_vb = clamp(var_b / (sig_cap * sig_cap + 1e-6), 0.0, 2.0)
                sensed_env = self._local_env_sensed(y, x)
                score_a = clamp(
                    0.34 * (1.0 - min(1.0, norm_va))
                    + 0.24 * sb
                    + 0.22 * c.consequence_trace
                    + 0.12 * (1.0 - abs(va - cs) / (sig_cap + 1e-6))
                    + 0.08 * sensed_env * sb,
                    0.0,
                    1.0,
                )
                score_b = clamp(
                    0.34 * (1.0 - min(1.0, norm_vb))
                    + 0.24 * eb
                    + 0.22 * c.consequence_trace
                    + 0.12 * (1.0 - abs(vb - cs) / (sig_cap + 1e-6))
                    + 0.08 * sensed_env * eb,
                    0.0,
                    1.0,
                )
                pref = clamp(score_a - score_b, -1.0, 1.0)

                if cr == self.ROLE_AMPLIFIER:
                    new_signal *= amp_boost + clamp(ada, -1.0, 1.0) * amp_ada
                elif cr == self.ROLE_DAMPENER:
                    low_pw = clamp((pw_max - pw) / max(pw_max, 1e-6), 0.0, 1.0)
                    neg_a = clamp(-ada, 0.0, 2.0)
                    new_signal *= damp_base * (1.0 + damp_lp * low_pw) * (1.0 + damp_na * neg_a)
                elif cr == self.ROLE_ROUTER:
                    mult = 1.0 + pw_cl * rpw
                    new_signal = (new_signal * mult * rb) + router_retain * (cs - new_signal)
                elif cr == self.ROLE_MEMORY:
                    new_signal *= 1.0 + clamp(cwt, 0.0, wt_cap) * mts
                elif cr == self.ROLE_GATE:
                    new_signal *= 1.0 + clamp(gb, gb_lo, gb_hi) * ges
                elif cr == self.ROLE_RELAY:
                    new_signal *= 1.0 + clamp(rbi, 0.0, rb_cap) * rs
                    new_signal = (new_signal * 1.02) + 0.04 * cross_flux / (sig_cap + 1e-6)
                elif cr == self.ROLE_BINDER:
                    new_signal *= 1.0 + clamp(baf, 0.0, ba_cap) * bs * 0.1
                elif cr == self.ROLE_OUTPUT:
                    new_signal *= out_sense

                new_signal = clamp(new_signal, -sig_cap, sig_cap)
                new_signal = clamp(new_signal * gmod, -sig_cap, sig_cap)
                pred_mod = clamp(1.0 + c.model_confidence * self.p.prediction_bias, 0.82, 1.18)
                new_signal = clamp(new_signal * pred_mod, -sig_cap, sig_cap)
                scen_mod = clamp(1.0 + pref * self.p.scenario_pref_scale, 0.86, 1.14)
                new_signal = clamp(new_signal * scen_mod, -sig_cap, sig_cap)
                if abs(new_signal) < eps:
                    new_signal = 0.0

                err_raw = new_signal - c.predicted_signal
                pe_mag = clamp(abs(err_raw) / (sig_cap + 1e-6), 0.0, 2.0)
                new_pe = clamp(
                    c.prediction_error * self.p.prediction_error_decay + pe_mag,
                    0.0,
                    self.p.prediction_error_cap,
                )
                new_mconf = clamp(
                    c.model_confidence * self.p.model_confidence_decay
                    + self.p.model_confidence_gain * (1.0 / (1.0 + new_pe)),
                    0.0,
                    self.p.model_confidence_cap,
                )
                sim_flag = clamp(abs(vp - new_signal) / (2.0 * sig_cap + 1e-6), 0.0, 1.0)
                pe = min(pe_mag, 1.0)

                ext_err = sensed_env - c.world_state_estimate
                new_wse = clamp(
                    c.world_state_estimate * self.p.world_state_decay
                    + self.p.world_state_gain * sensed_env
                    + self.p.world_state_corr * ext_err,
                    0.0,
                    1.0,
                )
                new_ext_err = clamp(
                    c.external_error * self.p.external_error_decay + abs(ext_err),
                    0.0,
                    2.0,
                )
                new_ct = clamp(
                    c.consequence_trace * self.p.consequence_trace_decay
                    + self.p.consequence_gain
                    * (1.0 - new_pe / (self.p.prediction_error_cap + 1e-6))
                    * (0.5 + 0.5 * abs(pref)),
                    0.0,
                    1.0,
                )

                new_memory = clamp(cm * mem_decay + abs(new_signal) * mem_gain, 0.0, mem_cap)
                nbr_act = abs(up) + abs(down) + abs(left) + abs(right)
                local_routing = 0.25 * clamp(nbr_act / (4.0 * sig_cap + 1e-6), 0.0, 1.0)
                new_elig = clamp(elig * el_decay + abs(new_signal) * el_gain + local_routing, 0.0, el_cap)

                mem_nbr_wt = 0.0
                same_tissue_nbr = 0
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        nc = cur[ny][nx]
                        if nc.role == self.ROLE_MEMORY:
                            mem_nbr_wt += nc.working_trace
                        if c.tissue_id > 0 and nc.tissue_id == c.tissue_id:
                            same_tissue_nbr += 1
                mem_nbr_wt *= 0.25

                eff_wt_decay = wt_decay * (mem_wt_bonus if cr == self.ROLE_MEMORY else 1.0)
                new_wt = cwt * eff_wt_decay + abs(new_signal) * wt_gain + mem_nbr_wt * nbr_wt_gain
                new_wt = clamp(new_wt, 0.0, wt_cap)

                imb = (up - down) / (sig_cap + 1e-6)
                new_gb = clamp(
                    gb * gb_decay + gb_gain * clamp(imb, -1.0, 1.0) * (new_wt / (wt_cap + 1e-6)),
                    gb_lo,
                    gb_hi,
                )

                new_rbi = clamp(
                    rbi * rb_decay + rb_gain * (cross_flux / (sig_cap + 1e-6)) + 0.02 * abs(new_signal) / sig_cap,
                    0.0,
                    rb_cap,
                )
                new_baf = clamp(
                    baf * ba_decay + ba_gain * (same_tissue_nbr / 4.0) * (ccs / (cs_cap + 1e-6)),
                    0.0,
                    ba_cap,
                )

                same_role = 0
                if new_memory >= cluster_thr:
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            n = cur[ny][nx]
                            if n.role == cr and n.memory >= cluster_thr:
                                same_role += 1
                local_support = (same_role / 4.0) * (new_memory / (mem_cap + 1e-6))
                new_cs = clamp(ccs * cs_decay + local_support * cs_gain, 0.0, cs_cap)

                ng = 0.0
                ngn = 0
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        ng += cur[ny][nx].goal_signal
                        ngn += 1
                ng /= max(ngn, 1)

                local_inst = clamp((abs(up - down) + abs(left - right)) / (4.0 * sig_cap + 1e-6), 0.0, 1.0)
                local_err = clamp(abs(new_signal - cm) / (mem_cap + 1e-6), 0.0, 1.0)
                mem_sig_mismatch = clamp(abs(abs(new_signal) - cm) / (mem_cap + 1e-6), 0.0, 1.0)
                ox_w = 0.55 + 0.45 * (float(x) / max(w - 1, 1))
                feed_local = self._feed_output_error * ox_w + self._feed_instability * 0.45

                gdec = self.p.goal_signal_decay
                ggain = self.p.goal_signal_gain
                gcap = self.p.goal_signal_cap
                gmix = self.p.goal_neighbor_mix
                new_goal = (
                    gs * gdec
                    + ggain * (local_err + 0.35 * local_inst + 0.22 * mem_sig_mismatch + 0.2 * feed_local)
                    + gmix * (ng - gs)
                )
                new_goal = clamp(new_goal, -gcap, gcap)

                sig_var = ((up - cs) ** 2 + (down - cs) ** 2 + (left - cs) ** 2 + (right - cs) ** 2) * 0.25
                sig_var_norm = clamp(sig_var / (sig_cap * sig_cap + 1e-6), 0.0, 1.0)
                tsc_n = clamp(c.tissue_score / (self.p.tissue_score_cap + 1e-6), 0.0, 1.0)
                new_meta = clamp(
                    ms * self.p.meta_state_decay
                    + self.p.meta_state_gain
                    * (
                        0.34 * (new_wt / (wt_cap + 1e-6))
                        + 0.2 * (new_cs / (cs_cap + 1e-6))
                        + 0.36 * sig_var_norm
                        + 0.1 * tsc_n
                    ),
                    -self.p.meta_state_cap,
                    self.p.meta_state_cap,
                )

                cw_cap = self.p.control_weight_cap
                cw_decay = self.p.control_weight_decay
                cw_gain = self.p.control_weight_gain
                perf_gap = max(0.0, -self._last_reward_ma)
                new_cw = clamp(
                    cw * cw_decay
                    + cw_gain
                    * (
                        (abs(new_meta) + abs(new_goal)) * 0.07
                        + 0.04 * perf_gap
                        + 0.02 * (1.0 if c.tissue_id > 0 else 0.0)
                        + 0.06 * new_mconf * (1.0 - pe)
                    ),
                    0.0,
                    cw_cap,
                )

                sb_decay = self.p.stability_bias_decay
                eb_decay = self.p.exploration_bias_decay
                se_gain = self.p.stab_expl_gain
                consistency = 1.0 - clamp(local_err + 0.45 * local_inst, 0.0, 1.0)
                new_sb = clamp(
                    sb * sb_decay + se_gain * consistency * (1.0 + 0.25 * max(0.0, self._last_reward_ma)),
                    0.0,
                    1.0,
                )
                new_eb = clamp(
                    eb * eb_decay
                    + se_gain * clamp(local_err + 0.35 * self._feed_output_error + 0.35 * self._feed_instability, 0.0, 1.0)
                    * (1.0 - 0.35 * sb),
                    0.0,
                    1.0,
                )

                pw_lo, pw_hi = self.p.path_weight_min, self.p.path_weight_max
                pw_d, pw_g = self.p.path_weight_decay, self.p.path_weight_gain
                elig_norm = new_elig / (el_cap + 1e-6)
                mps = self.p.meta_path_mod_scale
                explore_push = (new_eb - 0.5) * mps
                stability_pull = (new_sb - 0.5) * mps
                goal_steering = clamp(new_goal, -gcap, gcap) * mps * 0.06
                cw_norm = cw / (cw_cap + 1e-6)
                new_pw = clamp(
                    pw * pw_d
                    + pw_g * elig_norm * abs(new_signal) / (sig_cap + 1e-6) * (1.0 + explore_push - stability_pull + goal_steering) * (0.55 + 0.45 * cw_norm),
                    pw_lo,
                    pw_hi,
                )
                new_pw = clamp(
                    new_pw - self.p.pred_path_damp_scale * pe * (1.0 - 0.65 * c.model_confidence),
                    pw_lo,
                    pw_hi,
                )

                meta_gb = mps * (new_meta * 0.06 + new_goal * (new_eb - sb) * 0.08)
                meta_rb = mps * (new_goal * new_eb * 0.28 - new_sb * 0.1)
                new_gb = clamp(new_gb + meta_gb, gb_lo, gb_hi)
                new_rbi = clamp(new_rbi + meta_rb, 0.0, rb_cap)
                new_gb = clamp(new_gb - mps * (1.0 - c.model_confidence) * pe * 0.45, gb_lo, gb_hi)
                new_rbi = clamp(new_rbi + mps * (c.model_confidence - 0.5) * (1.0 - pe) * 0.5, 0.0, rb_cap)

                new_role = self.role_update_policy(x, y, new_memory, new_signal, new_wt, cross_flux, cur)
                if new_eb > 0.72 and abs(new_goal) > 0.85 and new_role == self.ROLE_NEUTRAL and new_elig > 0.42:
                    new_role = self.ROLE_ROUTER
                if new_sb > 0.78 and new_role == self.ROLE_ROUTER and new_elig < 0.22:
                    new_role = self.ROLE_NEUTRAL

                n = row_nxt[x]
                n.signal, n.memory, n.role, n.cluster_score = new_signal, new_memory, new_role, new_cs
                n.cluster_id = 0
                n.tissue_id, n.tissue_score = 0, 0.0
                ada_cap = self.p.adaptation_cap
                new_ada = clamp(
                    ada - self.p.prediction_error_adapt_scale * pe * (1.0 - 0.5 * c.model_confidence),
                    -ada_cap,
                    ada_cap,
                )
                n.adaptation, n.path_weight, n.eligibility = new_ada, new_pw, new_elig
                n.working_trace, n.gate_bias = new_wt, new_gb
                n.relay_bias, n.bind_affinity = new_rbi, new_baf
                n.layer_id = lid
                n.goal_signal, n.meta_state = new_goal, new_meta
                n.control_weight, n.stability_bias, n.exploration_bias = new_cw, new_sb, new_eb
                n.predicted_signal = vp
                n.prediction_error = new_pe
                n.simulation_flag = sim_flag
                n.model_confidence = new_mconf
                n.temporal_phase = self.step_count % 4
                n.scenario_a_signal = va
                n.scenario_b_signal = vb
                n.scenario_score_a = score_a
                n.scenario_score_b = score_b
                n.scenario_preference = pref
                n.world_state_estimate = new_wse
                n.external_error = new_ext_err
                n.consequence_trace = new_ct

    def swap_buffers(self) -> None:
        self.grid_current, self.grid_next = self.grid_next, self.grid_current

    def apply_episode_reinforcement(self, reward: float) -> None:
        if not self.reinforcement_enabled:
            return
        r = reward * self.p.reward_gain_scale
        ada_d, ada_g, ada_cap = self.p.adaptation_decay, self.p.adaptation_gain, self.p.adaptation_cap
        pw_d, pw_g = self.p.path_weight_decay, self.p.path_weight_gain
        pw_lo, pw_hi = self.p.path_weight_min, self.p.path_weight_max
        el_post = self.p.eligibility_after_reward_decay
        wt_cap = self.p.working_trace_cap
        gb_lo, gb_hi = self.p.gate_bias_min, self.p.gate_bias_max
        rb_cap = self.p.relay_bias_cap
        ba_cap = self.p.bind_affinity_cap

        for y in range(self.p.height):
            for x in range(self.p.width):
                c = self.grid_current[y][x]
                e = c.eligibility
                perr = abs(c.prediction_error) / (self.p.prediction_error_cap + 1e-6)
                c.adaptation = clamp(
                    c.adaptation * ada_d + r * e * ada_g - perr * e * self.p.prediction_error_adapt_scale,
                    -ada_cap,
                    ada_cap,
                )
                c.path_weight = clamp(c.path_weight * pw_d + r * e * pw_g, pw_lo, pw_hi)
                c.working_trace = clamp(c.working_trace + r * e * self.p.wt_reinforce_scale, 0.0, wt_cap)
                c.gate_bias = clamp(c.gate_bias + r * e * self.p.gb_reinforce_scale, gb_lo, gb_hi)
                c.relay_bias = clamp(c.relay_bias + r * e * self.p.relay_reinforce_scale, 0.0, rb_cap)
                c.bind_affinity = clamp(
                    c.bind_affinity + r * e * self.p.bind_reinforce_scale * (c.cluster_score / (self.p.cluster_score_cap + 1e-6)),
                    0.0,
                    ba_cap,
                )
                c.eligibility = min(c.eligibility * el_post, self.p.eligibility_cap)

        self.last_episode_reward = reward
        self.reward_history.append(reward)
        self.total_episodes += 1
        if reward > 0:
            self.correct_episodes += 1
        self.routing_bias_score += reward

    def _reward_episode(self) -> float:
        top, bot = self.output_top_value, self.output_bottom_value
        ctx = self.output_context_value
        m = self.p.decision_margin
        t = self.current_task
        h = self.p.height
        if t == self.TASK_MOVING_TARGET:
            ty = self.env_moving_target_y()
            want_top = ty < h // 2
            if want_top:
                if top > bot + m:
                    return 1.0
                if bot > top + m:
                    return -1.0
                return 0.0
            if bot > top + m:
                return 1.0
            if top > bot + m:
                return -1.0
            return 0.0
        if t == self.TASK_OBSTACLE_NAV:
            if top > bot + m:
                return 1.0
            if bot > top + m:
                return -1.0
            return 0.0
        if t == self.TASK_CONTEXT_SWITCH:
            if self.env_prefer_top():
                if top > bot + m:
                    return 1.0
                if bot > top + m:
                    return -1.0
                return 0.0
            if bot > top + m:
                return 1.0
            if top > bot + m:
                return -1.0
            return 0.0
        if t == self.TASK_EXPLORE_EXPLOIT:
            explore = (self.step_count // 30) % 2 == 0
            if explore:
                return 1.0 if top > bot + m else (-1.0 if bot > top + m else 0.0)
            return 1.0 if bot > top + m else (-1.0 if top > bot + m else 0.0)
        if t == self.TASK_MULTI_PATH:
            if max(abs(top), abs(bot)) < 0.5:
                return 0.0
            return 1.0 if self.current_decision in ("TOP", "BOTTOM") else 0.0
        if t == self.TASK_COMPETING:
            tot = abs(top) + abs(bot) + 1e-6
            balance = abs(top - bot) / tot
            if min(abs(top), abs(bot)) < 0.4:
                return -0.2
            return 1.0 if balance < 0.38 else (-0.4 if balance > 0.72 else 0.0)
        if t in (self.TASK_ADAPTIVE, self.TASK_DISTURBANCE):
            if top > bot + m:
                return 1.0
            if bot > top + m:
                return -1.0
            return 0.0
        if t == self.TASK_DELAY_MATCH:
            match = self.delay_match_first_is_a
            if match:
                return 1.0 if ctx > max(top, bot) * 0.45 else (-0.5 if abs(top - bot) < m else 0.0)
            return 1.0 if bot > top + m else (-1.0 if top > bot + m else 0.0)
        if t in (
            self.TASK_DELAYED_RESPONSE,
            self.TASK_PATTERN_CONTINUATION,
            self.TASK_INTERRUPTED_SIGNAL,
        ):
            if top > bot + m:
                return 1.0
            if bot > top + m:
                return -1.0
            return 0.0
        if t == self.TASK_NOISE_FILTER:
            if max(abs(top), abs(bot)) < 0.35:
                return 0.0
            return 1.0 if self.current_decision in ("TOP", "BOTTOM") else 0.0
        if t in (self.TASK_A_THEN_B, self.TASK_A_GAP_B, self.TASK_COND_ROUTE):
            if top > bot + m:
                return 1.0
            if bot > top + m:
                return -1.0
            return 0.0
        if t in (self.TASK_B_THEN_A, self.TASK_B_GAP_A):
            if bot > top + m:
                return 1.0
            if top > bot + m:
                return -1.0
            return 0.0
        return 0.0

    def _snapshot_context_probe(self) -> None:
        d = self.output_top_value - self.output_bottom_value
        if self.current_task in (self.TASK_A_THEN_B, self.TASK_A_GAP_B, self.TASK_COND_ROUTE):
            self.last_delta_b_after_a = d
        if self.mode == self.MODE_INPUT_B and self.run_mode == self.RUN_FREE:
            self.last_delta_b_cold = d
        self.context_dependency_score = abs(self.last_delta_b_after_a - self.last_delta_b_cold)

    def _advance_sequential(self) -> None:
        self.seq_step += 1
        t1, g2, t2 = self.p.seq_phase1_steps, self._seq_gap_len, self.p.seq_phase2_steps
        if self.seq_phase == 0 and self.seq_step >= t1:
            self.seq_phase, self.seq_step = 1, 0
        elif self.seq_phase == 1 and self.seq_step >= g2:
            self.seq_phase, self.seq_step = 2, 0
        elif self.seq_phase == 2 and self.seq_step >= t2:
            rew = self._reward_episode()
            self.sequential_total += 1
            if rew > 0:
                self.sequential_correct += 1
            if self.current_task == self.TASK_COND_ROUTE:
                self.conditional_total += 1
                if rew > 0:
                    self.conditional_correct += 1
            if self.current_task == self.TASK_DELAY_MATCH:
                self.delay_match_total += 1
                if rew > 0:
                    self.delay_match_correct += 1
            self.apply_episode_reinforcement(rew)
            self._snapshot_context_probe()
            sig = (
                1 if self.path_to_top_exists else 0,
                1 if self.path_to_bottom_exists else 0,
                self.current_decision,
            )
            self._episode_path_signatures.append(sig)
            if len(self._episode_path_signatures) >= 2:
                self.strategy_variation_score = len(set(self._episode_path_signatures)) / len(self._episode_path_signatures)
            self.partial_reset_signals_only()
            self.seq_phase, self.seq_step = 0, 0
            self._advance_training_task()

    def _advance_training_task(self) -> None:
        if self.run_mode not in (
            self.RUN_TRAINING_SEQUENTIAL,
            self.RUN_TRAIN_COND,
            self.RUN_TRAIN_DELAY,
            self.RUN_TRAIN_PHASE7,
            self.RUN_TRAIN_PHASE8,
            self.RUN_TRAIN_PHASE9,
        ):
            return
        order = [
            self.TASK_A_THEN_B,
            self.TASK_B_THEN_A,
            self.TASK_COND_ROUTE,
            self.TASK_A_GAP_B,
            self.TASK_DELAY_MATCH,
            self.TASK_MULTI_PATH,
            self.TASK_COMPETING,
            self.TASK_ADAPTIVE,
            self.TASK_DISTURBANCE,
        ]
        phase7_order = [
            self.TASK_MULTI_PATH,
            self.TASK_COMPETING,
            self.TASK_ADAPTIVE,
            self.TASK_DISTURBANCE,
        ]
        phase8_order = [
            self.TASK_DELAYED_RESPONSE,
            self.TASK_PATTERN_CONTINUATION,
            self.TASK_INTERRUPTED_SIGNAL,
            self.TASK_NOISE_FILTER,
        ]
        phase9_order = [
            self.TASK_MOVING_TARGET,
            self.TASK_OBSTACLE_NAV,
            self.TASK_CONTEXT_SWITCH,
            self.TASK_EXPLORE_EXPLOIT,
        ]
        if self.run_mode == self.RUN_TRAIN_COND:
            self.current_task = self.TASK_COND_ROUTE
        elif self.run_mode == self.RUN_TRAIN_DELAY:
            self.current_task = self.TASK_DELAY_MATCH
        elif self.run_mode == self.RUN_TRAIN_PHASE7:
            try:
                j = phase7_order.index(self.current_task)
                self.current_task = phase7_order[(j + 1) % len(phase7_order)]
            except ValueError:
                self.current_task = phase7_order[0]
        elif self.run_mode == self.RUN_TRAIN_PHASE8:
            try:
                j = phase8_order.index(self.current_task)
                self.current_task = phase8_order[(j + 1) % len(phase8_order)]
            except ValueError:
                self.current_task = phase8_order[0]
        elif self.run_mode == self.RUN_TRAIN_PHASE9:
            try:
                j = phase9_order.index(self.current_task)
                self.current_task = phase9_order[(j + 1) % len(phase9_order)]
            except ValueError:
                self.current_task = phase9_order[0]
        else:
            try:
                i = order.index(self.current_task)
                self.current_task = order[(i + 1) % len(order)]
            except ValueError:
                pass
        self.mode = self.current_task
        self._set_seq_gap_for_task()

    def _training_advance_phase4(self) -> None:
        T, R = self.p.episode_steps, self.p.rest_steps
        self.phase_step += 1
        if self.training_phase in (0, 2):
            if self.phase_step >= T:
                task = self.MODE_INPUT_A if self.training_phase == 0 else self.MODE_INPUT_B
                top, bot = self.output_top_value, self.output_bottom_value
                m = self.p.decision_margin
                rew = (
                    (1.0 if top > bot + m else (-1.0 if bot > top + m else 0.0))
                    if task == self.MODE_INPUT_A
                    else (1.0 if bot > top + m else (-1.0 if top > bot + m else 0.0))
                )
                self.apply_episode_reinforcement(rew)
                self.partial_reset_signals_only()
                self.training_phase = 1 if self.training_phase == 0 else 3
                self.phase_step = 0
            return
        if self.phase_step >= R:
            for y in range(self.p.height):
                for x in range(self.p.width):
                    self.grid_current[y][x].eligibility *= self.p.rest_eligibility_decay
            self.training_phase = (self.training_phase + 1) % 4
            self.phase_step = 0

    def step(self) -> None:
        self._apply_injection()
        self._apply_disturbance_perturbation()
        self.compute_next_grid()
        self.swap_buffers()
        self.step_count += 1
        self._post_step_analysis()
        seq_modes = (
            self.TASK_A_THEN_B,
            self.TASK_B_THEN_A,
            self.TASK_A_GAP_B,
            self.TASK_B_GAP_A,
            self.TASK_COND_ROUTE,
            self.TASK_DELAY_MATCH,
            self.TASK_MULTI_PATH,
            self.TASK_ADAPTIVE,
            self.TASK_COMPETING,
            self.TASK_DISTURBANCE,
            self.TASK_DELAYED_RESPONSE,
            self.TASK_PATTERN_CONTINUATION,
            self.TASK_INTERRUPTED_SIGNAL,
            self.TASK_NOISE_FILTER,
            self.TASK_MOVING_TARGET,
            self.TASK_OBSTACLE_NAV,
            self.TASK_CONTEXT_SWITCH,
            self.TASK_EXPLORE_EXPLOIT,
        )
        if self.mode in seq_modes:
            self._advance_sequential()
        elif self.run_mode == self.RUN_TRAINING:
            self._training_advance_phase4()

    def _compute_phase7_feedback(self) -> None:
        top, bot = self.output_top_value, self.output_bottom_value
        tot = abs(top) + abs(bot) + 1e-6
        self._feed_output_error = abs(top - bot) / tot
        rh = list(self.reward_history)
        if len(rh) >= 2:
            self._last_reward_ma = sum(rh[-10:]) / min(10, len(rh))
            self._feed_output_error += 0.45 * max(0.0, -self._last_reward_ma)
        else:
            self._last_reward_ma = 0.0
        self._feed_output_error = clamp(self._feed_output_error, 0.0, 2.5)

        s_mass = 0.0
        for y in range(self.p.height):
            for x in range(self.p.width):
                s_mass += abs(self.grid_current[y][x].signal)
        n = float(self.p.width * self.p.height)
        denom = n * self.p.signal_cap
        delta = abs(s_mass - self._prev_total_signal) / (denom + 1e-6)
        self._feed_instability = clamp(delta, 0.0, 1.0)

        tflow = self.average_inter_tissue_flow / (self.p.signal_cap + 1e-6)
        self._feed_output_error = clamp(self._feed_output_error + 0.12 * min(1.0, tflow), 0.0, 2.5)

        pw_mean = 0.0
        for y in range(self.p.height):
            for x in range(self.p.width):
                pw_mean += self.grid_current[y][x].path_weight
        pw_mean /= n
        self.pathway_reconfiguration_rate = abs(pw_mean - self._path_weight_mean_prev) * 120.0
        self._path_weight_mean_prev = pw_mean

        if self._disturbance_step_marker >= 0:
            self._steps_since_disturb = self.step_count - self._disturbance_step_marker
            if self._feed_instability < 0.07 and self._steps_since_disturb > 3:
                self.recovery_time_after_disturbance = 0.88 * self.recovery_time_after_disturbance + 0.12 * float(
                    self._steps_since_disturb
                )
                self._disturbance_step_marker = -1

        self._prev_total_signal = s_mass

        wme = 0.0
        mp = 0.0
        for y in range(self.p.height):
            for x in range(self.p.width):
                c = self.grid_current[y][x]
                wme += abs(c.external_error)
                mp += c.scenario_preference
        wme /= n
        mp /= n
        self.world_model_error_ema = 0.9 * self.world_model_error_ema + 0.1 * wme
        self.route_switch_latency = 0.88 * self.route_switch_latency + 0.12 * abs(mp - self._prev_pref_val)
        self._prev_pref_val = mp
        acc = 1.0 if abs(top - bot) > self.p.decision_margin else 0.0
        self.output_accuracy_ema = 0.94 * self.output_accuracy_ema + 0.06 * acc

    def _output_split_y(self) -> int:
        h = self.p.height
        if self.current_task == self.TASK_MOVING_TARGET:
            return clamp(self.env_moving_target_y(), 1, h - 2)
        if self.current_task == self.TASK_CONTEXT_SWITCH:
            return h // 4 if self.env_prefer_top() else (3 * h) // 4
        return h // 2

    def _post_step_analysis(self) -> None:
        self.run_cluster_analysis()
        self.run_tissue_analysis()
        self.compute_output_readout()
        self.run_path_analysis()
        self._cluster_typing_and_flow()
        self._layer_activity()
        self._wm_metric()
        self._hierarchical_score()
        self._compute_phase7_feedback()

    def _wm_metric(self) -> None:
        n = self.p.width * self.p.height
        s = sum(self.grid_current[y][x].working_trace for y in range(self.p.height) for x in range(self.p.width))
        self.working_memory_retention_score = (s / n) / self.p.working_trace_cap if self.p.working_trace_cap > 0 else 0.0

    def _hierarchical_score(self) -> None:
        la = self.layer_activity
        mid = 0.5 * (la[1] + la[2])
        inp = la[0] + 1e-6
        self.hierarchical_dependency_score = mid / inp if inp > 0 else 0.0

    def _layer_activity(self) -> None:
        self.layer_activity = [0.0, 0.0, 0.0, 0.0]
        for y in range(self.p.height):
            for x in range(self.p.width):
                c = self.grid_current[y][x]
                lid = self.layer_from_x(x)
                self.layer_activity[lid] += abs(c.signal) + 0.1 * c.working_trace

    def is_cluster_active(self, c: Cell) -> bool:
        if c.memory < self.p.cluster_threshold:
            return False
        return abs(c.signal) > self.p.epsilon or c.memory > self.p.cluster_threshold

    def run_cluster_analysis(self) -> None:
        w, h = self.p.width, self.p.height
        cur = self.grid_current
        for y in range(h):
            for x in range(w):
                cur[y][x].cluster_id = 0

        next_id = 1
        sizes: dict[int, int] = {}
        sum_mem: dict[int, float] = {}
        sum_wt: dict[int, float] = {}
        sum_cs: dict[int, float] = {}
        roles_votes: dict[int, list[int]] = {}
        q: deque[tuple[int, int]] = deque()

        for y0 in range(h):
            for x0 in range(w):
                c0 = cur[y0][x0]
                if c0.cluster_id != 0 or not self.is_cluster_active(c0):
                    continue
                role = c0.role
                cid = next_id
                next_id += 1
                q.clear()
                q.append((x0, y0))
                c0.cluster_id = cid
                roles_votes[cid] = []
                while q:
                    x, y = q.popleft()
                    c = cur[y][x]
                    sizes[cid] = sizes.get(cid, 0) + 1
                    sum_mem[cid] = sum_mem.get(cid, 0.0) + c.memory
                    sum_wt[cid] = sum_wt.get(cid, 0.0) + c.working_trace
                    sum_cs[cid] = sum_cs.get(cid, 0.0) + c.cluster_score
                    roles_votes[cid].append(c.role)
                    for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
                        if 0 <= nx < w and 0 <= ny < h:
                            n = cur[ny][nx]
                            if n.cluster_id == 0 and n.role == role and self.is_cluster_active(n):
                                n.cluster_id = cid
                                q.append((nx, ny))

        self.cluster_sizes = dict(sizes)
        self.cluster_dominant_role = {}
        self.cluster_avg_wt = {}
        self.cluster_avg_cs = {}
        persist = mem_c = gate_c = route_c = bind_c = rel_c = 0
        for cid, sz in sizes.items():
            self.cluster_avg_wt[cid] = sum_wt[cid] / sz
            self.cluster_avg_cs[cid] = sum_cs[cid] / sz
            ctr = Counter(roles_votes.get(cid, []))
            dom = ctr.most_common(1)[0][0] if ctr else self.ROLE_NEUTRAL
            self.cluster_dominant_role[cid] = dom
            if sz >= self.p.cluster_persistence_size_min:
                persist += 1
            if dom == self.ROLE_MEMORY:
                mem_c += 1
            elif dom == self.ROLE_GATE:
                gate_c += 1
            elif dom == self.ROLE_ROUTER:
                route_c += 1
            elif dom == self.ROLE_BINDER:
                bind_c += 1
            elif dom == self.ROLE_RELAY:
                rel_c += 1

        self.cluster_avg_memory = {cid: sum_mem[cid] / sizes[cid] for cid in sizes}
        self.persistent_cluster_count = persist
        self.memory_cluster_count = mem_c
        self.gate_cluster_count = gate_c
        self.routing_cluster_count = route_c
        self.binder_cluster_count = bind_c
        self.relay_cluster_count = rel_c

    def run_tissue_analysis(self) -> None:
        w, h = self.p.width, self.p.height
        cur = self.grid_current
        cluster_ids = [cid for cid in self.cluster_sizes if cid > 0]
        parent: dict[int, int] = {cid: cid for cid in cluster_ids}

        def find(a: int) -> int:
            if parent[a] != a:
                parent[a] = find(parent[a])
            return parent[a]

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        edges: list[tuple[int, int, float]] = []
        for y in range(h):
            for x in range(w):
                c = cur[y][x]
                for dx, dy in ((1, 0), (0, 1)):
                    nx, ny = x + dx, y + dy
                    if nx < w and ny < h:
                        n = cur[ny][nx]
                        if c.cluster_id > 0 and n.cluster_id > 0 and c.cluster_id != n.cluster_id:
                            ca, cb = c.cluster_id, n.cluster_id
                            flow = min(abs(c.signal), abs(n.signal))
                            r1 = self.cluster_dominant_role.get(ca, 0)
                            r2 = self.cluster_dominant_role.get(cb, 0)
                            compat = (self.p.tissue_role_compat_bonus if self.roles_compatible(r1, r2) else 0.0) + min(
                                1.0, flow / (self.p.signal_cap + 1e-6)
                            )
                            if compat >= 0.55 or flow >= self.p.tissue_merge_flow_threshold * self.p.signal_cap:
                                union(ca, cb)
                            edges.append((ca, cb, flow))

        roots: dict[int, int] = {}
        next_tid = 1
        cluster_to_tissue: dict[int, int] = {}
        for cid in cluster_ids:
            r = find(cid)
            if r not in roots:
                roots[r] = next_tid
                next_tid += 1
            cluster_to_tissue[cid] = roots[r]

        tissue_clusters: dict[int, list[int]] = defaultdict(list)
        for cid in cluster_ids:
            tid = cluster_to_tissue[cid]
            tissue_clusters[tid].append(cid)

        self.tissue_sizes = {}
        self.tissue_cluster_count = {}
        self.tissue_avg_score = {}
        persist_t = 0
        mem_t = gate_t = relay_t = out_t = 0
        for tid, cids in tissue_clusters.items():
            tot = sum(self.cluster_sizes.get(c, 0) for c in cids)
            self.tissue_sizes[tid] = tot
            self.tissue_cluster_count[tid] = len(cids)
            sc = sum(self.cluster_avg_cs.get(c, 0.0) for c in cids) / max(len(cids), 1)
            self.tissue_avg_score[tid] = clamp(sc, 0.0, self.p.tissue_score_cap)
            if tot >= self.p.cluster_persistence_size_min * 2:
                persist_t += 1
            doms = [self.cluster_dominant_role.get(c, 0) for c in cids]
            mc = Counter(doms).most_common(1)[0][0]
            if mc == self.ROLE_MEMORY:
                mem_t += 1
            elif mc == self.ROLE_GATE:
                gate_t += 1
            elif mc == self.ROLE_RELAY:
                relay_t += 1
            elif mc == self.ROLE_OUTPUT:
                out_t += 1

        self.active_tissue_count = len(tissue_clusters)
        self.persistent_tissue_count = persist_t
        self.memory_tissue_count = mem_t
        self.gate_tissue_count = gate_t
        self.relay_tissue_count = relay_t
        self.output_tissue_count = out_t

        for y in range(h):
            for x in range(w):
                c = cur[y][x]
                cid = c.cluster_id
                if cid > 0 and cid in cluster_to_tissue:
                    tid = cluster_to_tissue[cid]
                    c.tissue_id = tid
                    c.tissue_score = self.tissue_avg_score.get(tid, 0.0)
                else:
                    c.tissue_id = 0
                    c.tissue_score = 0.0

        inter_flow = 0.0
        inter_edges = 0
        for ca, cb, fl in edges:
            ta = cluster_to_tissue.get(ca, 0)
            tb = cluster_to_tissue.get(cb, 0)
            if ta > 0 and tb > 0 and ta != tb:
                inter_edges += 1
                inter_flow += fl
        self.inter_tissue_edge_count = inter_edges // 2
        self.average_inter_tissue_flow = inter_flow / max(inter_edges, 1)
        self.strongest_tissue_path_score = inter_flow

    def _cluster_typing_and_flow(self) -> None:
        w, h = self.p.width, self.p.height
        cur = self.grid_current
        flow = 0.0
        for y in range(h):
            for x in range(w):
                c = cur[y][x]
                for dx, dy in ((1, 0), (0, 1)):
                    nx, ny = x + dx, y + dy
                    if nx < w and ny < h:
                        n = cur[ny][nx]
                        if c.cluster_id != n.cluster_id and c.cluster_id > 0 and n.cluster_id > 0:
                            flow += min(abs(c.signal), abs(n.signal))
        self.cluster_transition_flow_score = flow

    def compute_output_readout(self) -> None:
        w, h = self.p.width, self.p.height
        oz = max(1, min(self.p.output_zone_width, w))
        x0 = w - oz
        split = self._output_split_y()
        ow = self.p.output_weight
        band = max(1, int(h * self.p.output_center_band))
        y_lo = (h - band) // 2
        y_hi = y_lo + band
        top = bot = ctx = 0.0
        for y in range(h):
            for x in range(x0, w):
                c = self.grid_current[y][x]
                wt = ow if c.role == self.ROLE_OUTPUT else 1.0
                v = c.signal * wt
                if y < split:
                    top += v
                else:
                    bot += v
                if y_lo <= y < y_hi:
                    ctx += v
        self.output_top_value = top
        self.output_bottom_value = bot
        self.output_context_value = ctx
        m = self.p.decision_margin
        if top > bot + m:
            self.current_decision = "TOP"
        elif bot > top + m:
            self.current_decision = "BOTTOM"
        elif top <= m and bot <= m:
            self.current_decision = "NONE"
        else:
            self.current_decision = "BOTH"

    def run_path_analysis(self) -> None:
        w, h = self.p.width, self.p.height
        thr = self.p.path_component_threshold
        cur = self.grid_current
        visited = [[False] * w for _ in range(h)]
        ix = min(self.p.input_zone_x_max, w - 1)
        oz = max(1, min(self.p.output_zone_width, w))
        ox0 = w - oz
        split = self._output_split_y()
        active = largest = 0
        strongest = 0.0
        any_top = any_bot = False
        q: deque[tuple[int, int]] = deque()
        for y0 in range(h):
            for x0 in range(w):
                if visited[y0][x0]:
                    continue
                c0 = cur[y0][x0]
                if c0.path_weight < thr:
                    continue
                active += 1
                q.clear()
                q.append((x0, y0))
                visited[y0][x0] = True
                size = 0
                score_sum = 0.0
                touches_in = touches_top_out = touches_bot_out = False
                while q:
                    x, y = q.popleft()
                    c = cur[y][x]
                    size += 1
                    score_sum += c.path_weight * (1.0 + c.eligibility)
                    if x <= ix:
                        touches_in = True
                    if x >= ox0:
                        if y < split:
                            touches_top_out = True
                        else:
                            touches_bot_out = True
                    for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx]:
                            if cur[ny][nx].path_weight >= thr:
                                visited[ny][nx] = True
                                q.append((nx, ny))
                largest = max(largest, size)
                strongest = max(strongest, score_sum)
                if touches_in and touches_top_out:
                    any_top = True
                if touches_in and touches_bot_out:
                    any_bot = True
        self.active_path_count = active
        self.largest_path_size = largest
        self.strongest_path_score = strongest
        self.path_to_top_exists = any_top
        self.path_to_bottom_exists = any_bot

    def metrics(self) -> dict:
        n = float(self.p.width * self.p.height)
        sums = defaultdict(float)
        maxs = defaultdict(float)
        counts = defaultdict(int)
        min_gb = 1e9
        max_gb = -1e9
        tissue_ctrl: dict[int, float] = defaultdict(float)
        for y in range(self.p.height):
            for x in range(self.p.width):
                c = self.grid_current[y][x]
                sums["sig"] += c.signal
                maxs["sig"] = max(maxs["sig"], c.signal)
                sums["mem"] += c.memory
                maxs["mem"] = max(maxs["mem"], c.memory)
                sums["wt"] += c.working_trace
                maxs["wt"] = max(maxs["wt"], c.working_trace)
                sums["gb"] += c.gate_bias
                min_gb = min(min_gb, c.gate_bias)
                max_gb = max(max_gb, c.gate_bias)
                sums["rb"] += c.relay_bias
                sums["ba"] += c.bind_affinity
                sums["goal"] += c.goal_signal
                maxs["goal"] = max(maxs["goal"], abs(c.goal_signal))
                sums["meta"] += c.meta_state
                sums["meta_sq"] += c.meta_state * c.meta_state
                sums["cw"] += c.control_weight
                maxs["cw"] = max(maxs["cw"], c.control_weight)
                sums["sb"] += c.stability_bias
                sums["eb"] += c.exploration_bias
                sums["pred"] += c.predicted_signal
                sums["pred_err"] += c.prediction_error
                maxs["pred_err"] = max(maxs["pred_err"], c.prediction_error)
                sums["mconf"] += c.model_confidence
                sums["sim_f"] += c.simulation_flag
                sums["div"] += abs(c.signal - c.predicted_signal)
                sums["pred_inf"] += abs(c.predicted_signal) * c.path_weight
                sums["pw_sum"] += c.path_weight
                sums["scen_div"] += abs(c.scenario_a_signal - c.scenario_b_signal)
                sums["pref_abs"] += abs(c.scenario_preference)
                sums["pref"] += c.scenario_preference
                sums["adapt_sum"] += abs(c.adaptation)
                if c.model_confidence >= self.p.high_confidence_threshold:
                    counts["hiconf"] += 1
                tid = c.tissue_id
                if tid > 0:
                    tissue_ctrl[tid] += c.control_weight
                r = c.role
                counts[r] += 1

        def rc(r: int) -> int:
            return counts.get(r, 0)

        avg_meta = sums["meta"] / n
        meta_var = max(0.0, sums["meta_sq"] / n - avg_meta * avg_meta)
        ctrl_vals = list(tissue_ctrl.values()) if tissue_ctrl else [0.0]
        ctrl_entropy = 0.0
        sctrl = sum(ctrl_vals) + 1e-9
        for v in ctrl_vals:
            p = v / sctrl
            if p > 1e-9:
                ctrl_entropy -= p * (p ** 0.5)

        return {
            "total_signal": sums["sig"],
            "max_signal": maxs["sig"],
            "avg_memory": sums["mem"] / n,
            "max_memory": maxs["mem"],
            "avg_working_trace": sums["wt"] / n,
            "max_working_trace": maxs["wt"],
            "avg_gate_bias": sums["gb"] / n,
            "min_gate_bias": min_gb if min_gb < 1e8 else 0.0,
            "max_gate_bias": max_gb if max_gb > -1e8 else 0.0,
            "avg_relay_bias": sums["rb"] / n,
            "avg_bind_affinity": sums["ba"] / n,
            "neutral_count": rc(0),
            "amplifier_count": rc(1),
            "dampener_count": rc(2),
            "router_count": rc(3),
            "output_count": rc(4),
            "memory_role_count": rc(5),
            "gate_role_count": rc(6),
            "binder_role_count": rc(7),
            "relay_role_count": rc(8),
            "active_cluster_count": len(self.cluster_sizes),
            "persistent_cluster_count": self.persistent_cluster_count,
            "largest_cluster_size": max(self.cluster_sizes.values()) if self.cluster_sizes else 0,
            "active_tissue_count": self.active_tissue_count,
            "persistent_tissue_count": self.persistent_tissue_count,
            "largest_tissue_size": max(self.tissue_sizes.values()) if self.tissue_sizes else 0,
            "average_tissue_size": (
                sum(self.tissue_sizes.values()) / len(self.tissue_sizes) if self.tissue_sizes else 0.0
            ),
            "max_tissue_score": max(self.tissue_avg_score.values()) if self.tissue_avg_score else 0.0,
            "memory_cluster_count": self.memory_cluster_count,
            "gate_cluster_count": self.gate_cluster_count,
            "routing_cluster_count": self.routing_cluster_count,
            "memory_tissue_count": self.memory_tissue_count,
            "relay_tissue_count": self.relay_tissue_count,
            "gate_tissue_count": self.gate_tissue_count,
            "output_tissue_count": self.output_tissue_count,
            "input_layer_activity": self.layer_activity[0],
            "intermediate_layer_activity": self.layer_activity[1],
            "context_layer_activity": self.layer_activity[2],
            "output_layer_activity": self.layer_activity[3],
            "inter_tissue_edge_count": self.inter_tissue_edge_count,
            "average_inter_tissue_flow": self.average_inter_tissue_flow,
            "strongest_tissue_path_score": self.strongest_tissue_path_score,
            "output_top_value": self.output_top_value,
            "output_bottom_value": self.output_bottom_value,
            "output_context_value": self.output_context_value,
            "current_decision": self.current_decision,
            "total_episodes": self.total_episodes,
            "sequential_task_accuracy": self.sequential_correct / max(self.sequential_total, 1),
            "conditional_routing_accuracy": self.conditional_correct / max(self.conditional_total, 1),
            "delayed_match_accuracy": self.delay_match_correct / max(self.delay_match_total, 1),
            "avg_reward": sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0.0,
            "avg_reward_last_n": sum(list(self.reward_history)[-10:]) / min(10, len(self.reward_history)) if self.reward_history else 0.0,
            "context_dependency_score": self.context_dependency_score,
            "hierarchical_dependency_score": self.hierarchical_dependency_score,
            "working_memory_retention_score": self.working_memory_retention_score,
            "cluster_transition_flow_score": self.cluster_transition_flow_score,
            "active_path_count": self.active_path_count,
            "largest_path_size": self.largest_path_size,
            "strongest_path_score": self.strongest_path_score,
            "path_to_top": self.path_to_top_exists,
            "path_to_bottom": self.path_to_bottom_exists,
            "rolling_accuracy": (
                sum(1 for r in list(self.reward_history)[-20:] if r > 0) / min(20, len(self.reward_history))
                if self.reward_history
                else 0.0
            ),
            "routing_bias_score": self.routing_bias_score,
            "average_goal_signal": sums["goal"] / n,
            "max_goal_signal": maxs["goal"],
            "average_meta_state": avg_meta,
            "meta_state_variance": meta_var,
            "max_control_weight": maxs["cw"],
            "average_control_weight": sums["cw"] / n,
            "control_weight_entropy_proxy": ctrl_entropy,
            "control_weight_distribution": dict(sorted(tissue_ctrl.items(), key=lambda kv: -kv[1])[:8]),
            "stability_bias_avg": sums["sb"] / n,
            "exploration_bias_avg": sums["eb"] / n,
            "strategy_variation_score": self.strategy_variation_score,
            "recovery_time_after_disturbance": self.recovery_time_after_disturbance,
            "pathway_reconfiguration_rate": self.pathway_reconfiguration_rate,
            "feed_output_error": self._feed_output_error,
            "feed_instability": self._feed_instability,
            "average_prediction_error": sums["pred_err"] / n,
            "max_prediction_error": maxs["pred_err"],
            "average_model_confidence": sums["mconf"] / n,
            "high_confidence_cell_count": counts.get("hiconf", 0),
            "prediction_accuracy_score": 1.0 / (1.0 + sums["pred_err"] / n),
            "predictive_influence_score": sums["pred_inf"] / (sums["pw_sum"] + 1e-9),
            "simulation_stability_score": 1.0 - sums["sim_f"] / n,
            "real_vs_predicted_divergence": sums["div"] / n,
            "average_predicted_signal": sums["pred"] / n,
            "scenario_divergence": sums["scen_div"] / n,
            "scenario_preference_mean": sums["pref"] / n,
            "average_scenario_preference_abs": sums["pref_abs"] / n,
            "world_model_error": self.world_model_error_ema,
            "output_accuracy_ema": self.output_accuracy_ema,
            "route_switch_latency": self.route_switch_latency,
            "average_adaptation_magnitude": sums["adapt_sum"] / n,
        }


class Phase9App:
    VIS_SIGNAL, VIS_ROLE, VIS_MEMORY, VIS_CLUSTER, VIS_TISSUE = 0, 1, 2, 3, 4
    VIS_PATH, VIS_ELIG, VIS_WT, VIS_GATE, VIS_RELAY, VIS_LAYER, VIS_TASK = 5, 6, 7, 8, 9, 10, 11

    def __init__(self, world: Phase9World) -> None:
        self.world = world
        self.running = False
        self.show_metrics = True
        self.vis_mode = 0
        self.emph = {"out": False, "ep": False, "wm": False, "gate": False, "tissue": False, "layer": False, "flow": False}

        self.root = tk.Tk()
        self.root.title("PHASE9: Counterfactual scenarios + world model (deterministic)")
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.frame, highlightthickness=0, bg="black")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.status_var = tk.StringVar(value="")
        tk.Label(self.frame, textvariable=self.status_var, anchor="w", justify="left", font=("Consolas", 7)).pack(side="bottom", fill="x")
        self._cell_w = self._cell_h = 10
        self._rect_ids: list[list[int]] = []
        self._last_rgb: list[list[tuple[int, int, int]]] = []
        self._fps_ts = time.perf_counter()
        self._fps = 0.0

        self.root.bind("<space>", lambda e: setattr(self, "running", not self.running) or self._rs())
        self.root.bind("r", lambda e: self._rst(Phase9World.MODE_CENTER_PULSE))
        self.root.bind("s", lambda e: self._rst(Phase9World.MODE_LEFT_STREAM))
        self.root.bind("c", lambda e: self._rst(Phase9World.MODE_CLEAR))
        self.root.bind("v", lambda e: setattr(self, "vis_mode", (self.vis_mode + 1) % 24) or self.render(True))
        self.root.bind("m", lambda e: setattr(self, "show_metrics", not self.show_metrics) or self._rs())
        self.root.bind("n", lambda e: (self.world.step(), self.render(False)) if not self.running else None)
        self.root.bind("p", lambda e: (self.world.partial_reset_signals_only(), self.render(True)))
        self.root.bind("x", lambda e: (self.world.reset(self.world.mode, full=True), self.render(True)))
        self.root.bind("t", self._cycle_train)
        self.root.bind("e", lambda e: (setattr(self.world, "run_mode", Phase9World.RUN_EVAL_SEQUENTIAL), setattr(self.world, "reinforcement_enabled", False)) or self._rs())
        self.root.bind("h", lambda e: self._tog("tissue"))
        self.root.bind("l", lambda e: self._tog("layer"))
        self.root.bind("k", lambda e: self._tog("flow"))
        self.root.bind("w", lambda e: self._tog("wm"))
        self.root.bind("g", lambda e: self._tog("gate"))
        self.root.bind("o", lambda e: self._tog("out"))
        self.root.bind("u", lambda e: self._tog("ep"))
        self.root.bind("4", lambda e: self._task(Phase9World.TASK_A_THEN_B))
        self.root.bind("5", lambda e: self._task(Phase9World.TASK_B_THEN_A))
        self.root.bind("6", lambda e: self._task(Phase9World.TASK_A_GAP_B))
        self.root.bind("7", lambda e: self._task(Phase9World.TASK_B_GAP_A))
        self.root.bind("8", lambda e: self._task(Phase9World.TASK_MULTI_PATH))
        self.root.bind("9", lambda e: self._task(Phase9World.TASK_COND_ROUTE))
        self.root.bind("0", lambda e: self._task(Phase9World.TASK_DELAY_MATCH))
        self.root.bind("q", lambda e: self._task(Phase9World.TASK_MULTI_PATH))
        self.root.bind("z", lambda e: self._task(Phase9World.TASK_ADAPTIVE))
        self.root.bind("j", lambda e: self._task(Phase9World.TASK_COMPETING))
        self.root.bind("d", lambda e: self._task(Phase9World.TASK_DISTURBANCE))
        self.root.bind(",", lambda e: self._task(Phase9World.TASK_DELAYED_RESPONSE))
        self.root.bind(".", lambda e: self._task(Phase9World.TASK_PATTERN_CONTINUATION))
        self.root.bind(";", lambda e: self._task(Phase9World.TASK_INTERRUPTED_SIGNAL))
        self.root.bind("'", lambda e: self._task(Phase9World.TASK_NOISE_FILTER))
        self.root.bind("<F1>", lambda e: self._task(Phase9World.TASK_MOVING_TARGET))
        self.root.bind("<F2>", lambda e: self._task(Phase9World.TASK_OBSTACLE_NAV))
        self.root.bind("<F3>", lambda e: self._task(Phase9World.TASK_CONTEXT_SWITCH))
        self.root.bind("<F4>", lambda e: self._task(Phase9World.TASK_EXPLORE_EXPLOIT))
        self.root.bind("1", lambda e: self._free(Phase9World.MODE_INPUT_A))
        self.root.bind("2", lambda e: self._free(Phase9World.MODE_INPUT_B))
        self.root.bind("3", lambda e: self._free(Phase9World.MODE_INPUT_AB))
        self.canvas.bind("<Button-1>", lambda e: (self.world.inject_at(int(e.x // self._cell_w), int(e.y // self._cell_h)), self.render(False)))
        self.root.bind("<Configure>", lambda e: (self._init(), self.render(True)))
        self._init()
        self.render(True)
        self._tick()

    def _tog(self, k: str) -> None:
        self.emph[k] = not self.emph[k]
        self._rs()

    def _cycle_train(self, _e=None) -> None:
        self.world.training_mode_idx = (self.world.training_mode_idx + 1) % 8
        modes = [
            Phase9World.RUN_FREE,
            Phase9World.RUN_TRAINING_SEQUENTIAL,
            Phase9World.RUN_TRAIN_COND,
            Phase9World.RUN_TRAIN_DELAY,
            Phase9World.RUN_TRAIN_PHASE7,
            Phase9World.RUN_TRAIN_PHASE8,
            Phase9World.RUN_TRAIN_PHASE9,
            Phase9World.RUN_TRAINING,
        ]
        self.world.run_mode = modes[self.world.training_mode_idx]
        if self.world.run_mode == Phase9World.RUN_FREE:
            pass
        elif self.world.run_mode == Phase9World.RUN_EVAL_SEQUENTIAL:
            pass
        elif self.world.run_mode == Phase9World.RUN_TRAIN_PHASE7:
            self.world.reinforcement_enabled = True
            if self.world.current_task not in (
                Phase9World.TASK_MULTI_PATH,
                Phase9World.TASK_COMPETING,
                Phase9World.TASK_ADAPTIVE,
                Phase9World.TASK_DISTURBANCE,
            ):
                self.world.current_task = Phase9World.TASK_MULTI_PATH
            self.world.mode = self.world.current_task
        elif self.world.run_mode == Phase9World.RUN_TRAIN_PHASE8:
            self.world.reinforcement_enabled = True
            if self.world.current_task not in (
                Phase9World.TASK_DELAYED_RESPONSE,
                Phase9World.TASK_PATTERN_CONTINUATION,
                Phase9World.TASK_INTERRUPTED_SIGNAL,
                Phase9World.TASK_NOISE_FILTER,
            ):
                self.world.current_task = Phase9World.TASK_DELAYED_RESPONSE
            self.world.mode = self.world.current_task
        elif self.world.run_mode == Phase9World.RUN_TRAIN_PHASE9:
            self.world.reinforcement_enabled = True
            if self.world.current_task not in (
                Phase9World.TASK_MOVING_TARGET,
                Phase9World.TASK_OBSTACLE_NAV,
                Phase9World.TASK_CONTEXT_SWITCH,
                Phase9World.TASK_EXPLORE_EXPLOIT,
            ):
                self.world.current_task = Phase9World.TASK_MOVING_TARGET
            self.world.mode = self.world.current_task
        else:
            self.world.reinforcement_enabled = True
            self.world.mode = self.world.current_task
        self._rs()

    def _rst(self, mode: str) -> None:
        self.world.run_mode = Phase9World.RUN_FREE
        self.world.reset(mode, full=True)
        self.render(True)

    def _task(self, task: str) -> None:
        self.world.run_mode = Phase9World.RUN_FREE
        self.world.current_task = task
        self.world._set_seq_gap_for_task()
        self.world.seq_phase = 0
        self.world.seq_step = 0
        self.world.reset(task, full=False)
        self.render(True)

    def _free(self, mode: str) -> None:
        self.world.run_mode = Phase9World.RUN_FREE
        self.world.mode = mode
        self.world.reset(mode, full=False)
        self.render(True)

    def _init(self) -> None:
        w, h = self.world.p.width, self.world.p.height
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self._cell_w, self._cell_h = max(1, cw // w), max(1, ch // h)
        self.canvas.delete("all")
        self._rect_ids = [[0] * w for _ in range(h)]
        self._last_rgb = [[(-1, -1, -1)] * w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                self._rect_ids[y][x] = self.canvas.create_rectangle(
                    x * self._cell_w, y * self._cell_h, (x + 1) * self._cell_w, (y + 1) * self._cell_h, outline="", fill="#000"
                )

    def _rs(self) -> None:
        self.status_var.set(self._status_text())

    def _status_text(self) -> str:
        m = self.world.metrics()
        st = "run" if self.running else "paus"
        vm = "sig role mem clu tis path elig wt g_gate relay lay task goal meta ctrl stabXexpl pred perr mconf simf scenA scenB pref wse ext".split()
        if not self.show_metrics:
            return f"{st} step={self.world.step_count} {vm[self.vis_mode]}"
        h = ""
        if self.emph["out"]:
            h += f"dec={m['current_decision']} t/b={m['output_top_value']:.1f}/{m['output_bottom_value']:.1f} | "
        if self.emph["tissue"]:
            h += f"tis={m['active_tissue_count']} persT={m['persistent_tissue_count']} maxT={m['largest_tissue_size']} | "
        if self.emph["layer"]:
            h += f"L_in={m['input_layer_activity']:.0f} L_mid={m['intermediate_layer_activity']:.0f} L_ctx={m['context_layer_activity']:.0f} | "
        if self.emph["flow"]:
            h += f"e_tis={m['inter_tissue_edge_count']} flow={m['average_inter_tissue_flow']:.2f} hi={m['hierarchical_dependency_score']:.2f} | "
        return h + f"{st} tissues={m['active_tissue_count']} clusters={m['active_cluster_count']} R={m['relay_role_count']} B={m['binder_role_count']} | {vm[self.vis_mode]}"

    def role_c(self, r: int) -> tuple[int, int, int]:
        w = self.world
        return {
            0: (110, 110, 110),
            1: (0, 200, 0),
            2: (220, 0, 0),
            3: (50, 100, 255),
            4: (240, 220, 0),
            5: (0, 220, 220),
            6: (220, 0, 220),
            7: (255, 140, 0),
            8: (160, 80, 255),
        }.get(r, (80, 80, 80))

    def rgb(self, c: Cell) -> tuple[int, int, int]:
        p = self.world.p
        vm = self.vis_mode
        if vm == 0:
            g = int(clamp(c.signal / p.scale_factor, 0, 1) * 255)
            return (g, g, g)
        if vm == 1:
            return self.role_c(c.role)
        if vm == 2:
            g = int(clamp(c.memory / p.memory_cap, 0, 1) * 255)
            return (g, g, g)
        if vm == 3:
            cid = c.cluster_id
            if cid <= 0:
                return (0, 0, 0)
            return (50 + (cid * 88) % 200, 50 + (cid * 55) % 200, 50 + (cid * 33) % 200)
        if vm == 4:
            tid = c.tissue_id
            if tid <= 0:
                return (10, 10, 10)
            return (40 + (tid * 77) % 215, 40 + (tid * 43) % 215, 40 + (tid * 91) % 215)
        if vm == 5:
            g = int(clamp((c.path_weight - p.path_weight_min) / (p.path_weight_max - p.path_weight_min), 0, 1) * 255)
            return (g, g, g)
        if vm == 6:
            g = int(clamp(c.eligibility / p.eligibility_cap, 0, 1) * 255)
            return (g, g, g)
        if vm == 7:
            g = int(clamp(c.working_trace / p.working_trace_cap, 0, 1) * 255)
            return (g, g, min(255, g + 50))
        if vm == 8:
            t = (c.gate_bias - p.gate_bias_min) / (p.gate_bias_max - p.gate_bias_min + 1e-9)
            return (int(80 + 175 * (1 - t)), int(80 + 100 * t), 100) if c.gate_bias >= 0 else (int(80 + 175 * t), 90, 120)
        if vm == 9:
            g = int(clamp(c.relay_bias / p.relay_bias_cap, 0, 1) * 255)
            return (g // 2, g // 3, g)
        if vm == 12:
            g = int(clamp((c.goal_signal + p.goal_signal_cap) / (2.0 * p.goal_signal_cap + 1e-9), 0, 1) * 255)
            return (g, min(255, g // 2 + 40), 255 - g)
        if vm == 13:
            g = int(clamp((c.meta_state + p.meta_state_cap) / (2.0 * p.meta_state_cap + 1e-9), 0, 1) * 255)
            return (min(255, 80 + g // 2), g, 255 - g)
        if vm == 14:
            g = int(clamp(c.control_weight / p.control_weight_cap, 0, 1) * 255)
            return (g // 2, min(255, g + 30), g // 3)
        if vm == 15:
            se = c.stability_bias + c.exploration_bias + 1e-9
            r = int(255 * (c.stability_bias / se))
            b = int(255 * (c.exploration_bias / se))
            return (r, 80, b)
        if vm == 16:
            g = int(clamp(abs(c.predicted_signal) / p.scale_factor, 0, 1) * 255)
            return (g // 2, g, min(255, g + 80))
        if vm == 17:
            g = int(clamp(c.prediction_error / p.prediction_error_cap, 0, 1) * 255)
            return (g, g // 4, 255 - g)
        if vm == 18:
            g = int(clamp(c.model_confidence, 0, 1) * 255)
            return (g // 3, min(255, g + 40), g // 2)
        if vm == 19:
            g = int(clamp(c.simulation_flag, 0, 1) * 255)
            return (g, 255 - g, g // 2)
        if vm == 20:
            g = int(clamp(abs(c.scenario_a_signal) / p.scale_factor, 0, 1) * 255)
            return (g, g // 2, min(255, g + 60))
        if vm == 21:
            g = int(clamp(abs(c.scenario_b_signal) / p.scale_factor, 0, 1) * 255)
            return (min(255, g + 40), g // 2, g)
        if vm == 22:
            g = int(clamp((c.scenario_preference + 1.0) / 2.0, 0, 1) * 255)
            return (g, 255 - g, 100)
        if vm == 23:
            g1 = int(clamp(c.world_state_estimate, 0, 1) * 255)
            g2 = int(clamp(c.external_error / 2.0, 0, 1) * 255)
            return (g2, g1, 255 - g2)
        lid = c.layer_id
        tint = [(40, 60, 100), (60, 100, 60), (100, 80, 40), (120, 100, 140)]
        tr, tg, tb = tint[lid % 4]
        g = int(clamp(c.signal / p.scale_factor, 0, 1) * 200)
        return (min(255, tr + g // 4), min(255, tg + g // 4), min(255, tb + g // 4))

    def render(self, force: bool) -> None:
        w, h = self.world.p.width, self.world.p.height
        cur = self.world.grid_current
        oz = max(1, min(self.world.p.output_zone_width, w))
        x0o = w - oz
        sp = self.world._output_split_y()
        for y in range(h):
            for x in range(w):
                c = cur[y][x]
                if self.vis_mode == 11:
                    g = int(clamp(c.signal / self.world.p.scale_factor, 0, 1) * 255)
                    if x >= x0o:
                        rgb = (min(255, g + 60), g // 2, g // 2) if y < sp else (g // 2, min(255, g + 60), g // 2)
                    elif x <= self.world.p.input_zone_x_max:
                        rgb = (g // 3, g // 3, min(255, g + 40))
                    else:
                        rgb = (g, g, g)
                else:
                    rgb = self.rgb(c)
                if force or rgb != self._last_rgb[y][x]:
                    self._last_rgb[y][x] = rgb
                    r, g, b = rgb
                    self.canvas.itemconfigure(self._rect_ids[y][x], fill=f"#{r:02x}{g:02x}{b:02x}")
        self._rs()

    def _tick(self) -> None:
        if self.running:
            self.world.step()
            self.render(False)
        t = time.perf_counter()
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(t - self._fps_ts, 1e-6))
        self._fps_ts = t
        self.root.after(16, self._tick)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    world = Phase9World(WorldParams())
    app = Phase9App(world)
    app.run()


if __name__ == "__main__":
    main()
