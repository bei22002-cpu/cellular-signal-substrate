#!/usr/bin/env python3
"""
Empirical harness for Phase9World (cellular_signal_substrate.py).

Run: python validation_phase9.py

Tweaks grid and diffusion so output readout is non-trivial within a few hundred
steps; default 50x50 + episode_steps=55 often yields zero output-zone signal
(transport-limited), which would falsely imply a broken readout.
"""

from __future__ import annotations

from dataclasses import replace

from cellular_signal_substrate import Phase9World, WorldParams

# Compact grid + moderate diffusion: signal reaches output columns in ~200 steps.
FAST = WorldParams(
    width=14,
    height=14,
    diffusion_coeff=0.38,
    decay_coeff=0.996,
    seq_phase1_steps=18,
    seq_gap_steps=4,
    seq_phase2_steps=18,
)


def run_world(p: WorldParams, mode: str, steps: int) -> Phase9World:
    w = Phase9World(p)
    w.run_mode = w.RUN_FREE
    w.mode = mode
    if mode in (
        w.TASK_MOVING_TARGET,
        w.TASK_OBSTACLE_NAV,
        w.TASK_CONTEXT_SWITCH,
        w.TASK_A_THEN_B,
        w.TASK_B_THEN_A,
        w.TASK_EXPLORE_EXPLOIT,
    ):
        w.current_task = mode
    w.reset(mode, full=False)
    for _ in range(steps):
        w.step()
    return w


def grid_signal_fingerprint(W: Phase9World) -> tuple[float, ...]:
    return tuple(W.grid_current[y][x].signal for y in range(W.p.height) for x in range(W.p.width))


def max_scenario_delta(W: Phase9World) -> float:
    m = 0.0
    for y in range(W.p.height):
        for x in range(W.p.width):
            c = W.grid_current[y][x]
            m = max(m, abs(c.scenario_a_signal - c.scenario_b_signal))
    return m


def main() -> None:
    p0 = FAST
    print("=== Determinism (identical IC + schedule) ===")
    a = run_world(p0, Phase9World.MODE_INPUT_A, 100)
    b = run_world(p0, Phase9World.MODE_INPUT_A, 100)
    print("fingerprints_equal", grid_signal_fingerprint(a) == grid_signal_fingerprint(b))

    print("\n=== Free-run INPUT_A (mechanism smoke) ===")
    W = run_world(p0, Phase9World.MODE_INPUT_A, 220)
    m = W.metrics()
    print("decision", m["current_decision"], "top", m["output_top_value"], "bot", m["output_bottom_value"])
    print(
        "scenario_divergence_mean",
        m["scenario_divergence"],
        "max_cell_scen_delta",
        round(max_scenario_delta(W), 4),
    )
    print("prediction_accuracy_score", m["prediction_accuracy_score"], "avg_pred_err", m["average_prediction_error"])
    print("world_model_error_ema", m["world_model_error"], "clusters", m["active_cluster_count"], "tissues", m["active_tissue_count"])

    print("\n=== Ablations (same mode/steps) ===")
    ablations = {
        "full": p0,
        "scenario_pref_off": replace(p0, scenario_pref_scale=0.0),
        "prediction_bias_off": replace(p0, prediction_bias=0.0),
        "goal_off": replace(p0, goal_scale=0.0, goal_signal_gain=0.0, goal_neighbor_mix=0.0),
        "world_gain_off": replace(p0, world_state_gain=0.0, world_state_corr=0.0),
        "equal_scenario_bias": replace(p0, scenario_a_bias=1.0, scenario_b_bias=1.0),
    }
    for label, pr in ablations.items():
        W = run_world(pr, Phase9World.MODE_INPUT_A, 220)
        m = W.metrics()
        print(
            label,
            "dec",
            m["current_decision"],
            "top",
            round(m["output_top_value"], 2),
            "bot",
            round(m["output_bottom_value"], 2),
            "wme",
            round(m["world_model_error"], 4),
        )

    print("\n=== Task suite (800 steps each, sequential scoring) ===")
    tasks = (
        Phase9World.TASK_MOVING_TARGET,
        Phase9World.TASK_OBSTACLE_NAV,
        Phase9World.TASK_CONTEXT_SWITCH,
        Phase9World.TASK_A_THEN_B,
        Phase9World.TASK_B_THEN_A,
        Phase9World.TASK_EXPLORE_EXPLOIT,
    )
    for t in tasks:
        W = run_world(p0, t, 800)
        m = W.metrics()
        print(
            t,
            "sequential_task_accuracy",
            round(m["sequential_task_accuracy"], 4),
            "episodes",
            W.sequential_total,
            "rolling_accuracy",
            round(m["rolling_accuracy"], 4),
        )

    print("\n=== Transport note: default-ish 32x32, default diffusion ===")
    slow = WorldParams(width=32, height=32)
    Ws = run_world(slow, Phase9World.MODE_INPUT_A, 120)
    m = Ws.metrics()
    print("steps=120 output_top", m["output_top_value"], "max_signal", m["max_signal"], "(often zero in output strip)")

    Wl = run_world(slow, Phase9World.MODE_LEFT_STREAM, 200)
    m = Wl.metrics()
    print("LEFT_STREAM steps=200 output_top", round(m["output_top_value"], 2), "decision", m["current_decision"])


if __name__ == "__main__":
    main()
