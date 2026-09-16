#!/usr/bin/env python3
"""
Full ablation + phase-proxy benchmark (compact grid for tractable runtime).

Writes: validation_phase9_suite_results.txt

Run: python validation_phase9_suite.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cellular_signal_substrate import Phase9World, WorldParams

# Same compact harness as validation_phase9.py
FAST = WorldParams(
    width=14,
    height=14,
    diffusion_coeff=0.38,
    decay_coeff=0.996,
    seq_phase1_steps=18,
    seq_gap_steps=4,
    seq_phase2_steps=18,
)

STEPS_TASK = 300
STEPS_INPUT_SANITY = 80


def run_task(p: WorldParams, task: str, steps: int) -> dict:
    w = Phase9World(p)
    w.run_mode = w.RUN_FREE
    w.mode = task
    w.current_task = task
    w.reset(task, full=False)
    # Optional time-series metrics for tasks intended to validate specific mechanisms.
    ext_abs_sum = 0.0
    pred_err_abs_sum = 0.0
    samples = 0
    # For regime shift: measure how many steps after the shift until decision matches new regime.
    shift_latency: int | None = None
    shift_at = max(1, p.seq_phase2_steps // 2)
    consec_match = 0
    for _ in range(steps):
        w.step()
        if task in (Phase9World.TASK_REGIME_SHIFT, Phase9World.TASK_DELAYED_TRAP):
            # Grid-level means are cheap on FAST; keep deterministic and direct.
            s_ext = 0.0
            s_pe = 0.0
            for yy in range(p.height):
                for xx in range(p.width):
                    c = w.grid_current[yy][xx]
                    s_ext += abs(c.external_error)
                    s_pe += abs(c.prediction_error)
            n = float(p.width * p.height)
            ext_abs_sum += s_ext / n
            pred_err_abs_sum += s_pe / n
            samples += 1

        if task == Phase9World.TASK_REGIME_SHIFT and shift_latency is None:
            if w.seq_phase == 2 and w.seq_step >= shift_at:
                want_top = w.env_prefer_top()
                want = "TOP" if want_top else "BOTTOM"
                if w.current_decision == want:
                    consec_match += 1
                else:
                    consec_match = 0
                if consec_match >= 3:
                    # Latency counted from first step after shift_at until stable match.
                    shift_latency = max(0, w.seq_step - shift_at)
    m = w.metrics()
    out = {
        "seq_acc": float(m["sequential_task_accuracy"]),
        "episodes": int(w.sequential_total),
        "rolling": float(m["rolling_accuracy"]),
        "scen_div": float(m["scenario_divergence"]),
        "pred_acc": float(m["prediction_accuracy_score"]),
        "pred_inf": float(m["predictive_influence_score"]),
        "wm_err": float(m["world_model_error"]),
        "instab": float(m["feed_instability"]),
        "clusters": int(m["active_cluster_count"]),
        "tissues": int(m["active_tissue_count"]),
        "hier": float(m["hierarchical_dependency_score"]),
    }
    if samples > 0:
        out["ext_err_abs_mean"] = ext_abs_sum / samples
        out["pred_err_abs_mean"] = pred_err_abs_sum / samples
    if shift_latency is not None:
        out["shift_latency"] = int(shift_latency)
    return out


def scenario_sanity(p: WorldParams) -> dict:
    w = Phase9World(p)
    w.run_mode = w.RUN_FREE
    w.reset(w.MODE_INPUT_A, full=False)
    for _ in range(STEPS_INPUT_SANITY):
        w.step()
    m = w.metrics()
    mx = 0.0
    for y in range(p.height):
        for x in range(p.width):
            c = w.grid_current[y][x]
            mx = max(mx, abs(c.scenario_a_signal - c.scenario_b_signal))
    return {
        "scenario_divergence": float(m["scenario_divergence"]),
        "max_cell_delta": mx,
        "pred_acc": float(m["prediction_accuracy_score"]),
        "wm_err": float(m["world_model_error"]),
    }


def md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


TASKS = [
    Phase9World.TASK_MOVING_TARGET,
    Phase9World.TASK_OBSTACLE_NAV,
    Phase9World.TASK_CONTEXT_SWITCH,
    Phase9World.TASK_A_THEN_B,
    Phase9World.TASK_B_THEN_A,
    Phase9World.TASK_EXPLORE_EXPLOIT,
    Phase9World.TASK_DELAYED_TRAP,
    Phase9World.TASK_REGIME_SHIFT,
]

ABLATIONS: list[tuple[str, WorldParams]] = [
    ("full", FAST),
    ("no_scenario_pref", replace(FAST, scenario_pref_scale=0.0)),
    ("no_scenario_excitation", replace(FAST, scenario_inject_coupling=0.0, scenario_from_signal_gain=0.0)),
    ("equal_scenario_bias", replace(FAST, scenario_a_bias=1.0, scenario_b_bias=1.0)),
    ("no_prediction_bias", replace(FAST, prediction_bias=0.0)),
    ("no_prediction_tracking", replace(FAST, prediction_track_signal=0.0)),
    ("no_world_update", replace(FAST, world_state_gain=0.0, world_state_corr=0.0)),
    ("no_goal_meta", replace(FAST, goal_scale=0.0, goal_signal_gain=0.0, goal_neighbor_mix=0.0, meta_state_gain=0.0, control_weight_gain=0.0)),
    ("freeze_adapt_path", replace(FAST, adaptation_gain=0.0, path_weight_gain=0.0, eligibility_gain=0.0)),
]

# Phase 7–9 stripped: no meta/pred/scenario/world/consequence gains (clusters still computed)
STRIPPED_789 = replace(
    FAST,
    goal_scale=0.0,
    goal_signal_gain=0.0,
    goal_neighbor_mix=0.0,
    meta_state_gain=0.0,
    control_weight_gain=0.0,
    prediction_bias=0.0,
    prediction_track_signal=0.0,
    scenario_pref_scale=0.0,
    scenario_inject_coupling=0.0,
    scenario_from_signal_gain=0.0,
    world_state_gain=0.0,
    world_state_corr=0.0,
    consequence_gain=0.0,
)


def main() -> None:
    lines: list[str] = []
    lines.append("# Phase 9 validation suite output (compact FAST grid)\n")
    lines.append(f"Grid: {FAST.width}x{FAST.height}, task_steps={STEPS_TASK}\n\n")

    lines.append("## Baseline task suite (Phase 9 full, FAST)\n")
    lines.append(
        md_row(
            [
                "task",
                "seq_acc",
                "episodes",
                "rolling",
                "scen_div",
                "pred_acc",
                "pred_inf",
                "wm_err",
                "instab",
                "shift_lat",
                "ext_err",
                "pred_err",
            ]
        )
        + "\n"
    )
    lines.append(md_row(["---"] * 12) + "\n")
    baseline: dict[str, dict] = {}
    for t in TASKS:
        r = run_task(FAST, t, STEPS_TASK)
        baseline[t] = r
        lines.append(
            md_row(
                [
                    t,
                    fmt(r["seq_acc"], 2),
                    str(r["episodes"]),
                    fmt(r["rolling"], 2),
                    fmt(r["scen_div"], 3),
                    fmt(r["pred_acc"], 3),
                    fmt(r["pred_inf"], 3),
                    fmt(r["wm_err"], 3),
                    fmt(r["instab"], 3),
                    str(r.get("shift_latency", "")),
                    fmt(float(r.get("ext_err_abs_mean", 0.0)), 3) if "ext_err_abs_mean" in r else "",
                    fmt(float(r.get("pred_err_abs_mean", 0.0)), 3) if "pred_err_abs_mean" in r else "",
                ]
            )
            + "\n"
        )

    lines.append("\n## Scenario counterfactual (FAST, MODE_INPUT_A)\n")
    ss = scenario_sanity(FAST)
    lines.append(
        md_row(
            [
                "scenario_divergence_mean",
                "max_cell|a-b|",
                "prediction_accuracy_score",
                "world_model_error_ema",
            ]
        )
        + "\n"
    )
    lines.append(md_row(["---"] * 4) + "\n")
    lines.append(
        md_row(
            [
                fmt(ss["scenario_divergence"], 6),
                fmt(ss["max_cell_delta"], 6),
                fmt(ss["pred_acc"], 6),
                fmt(ss["wm_err"], 6),
            ]
        )
        + "\n"
    )

    default50 = WorldParams()
    lines.append("\n## Scenario counterfactual (default 50x50, MODE_INPUT_A, 80 steps)\n")
    ss50 = scenario_sanity(default50)
    lines.append(
        md_row(
            [
                "scenario_divergence_mean",
                "max_cell|a-b|",
                "prediction_accuracy_score",
                "world_model_error_ema",
            ]
        )
        + "\n"
    )
    lines.append(md_row(["---"] * 4) + "\n")
    lines.append(
        md_row(
            [
                fmt(ss50["scenario_divergence"], 6),
                fmt(ss50["max_cell_delta"], 6),
                fmt(ss50["pred_acc"], 6),
                fmt(ss50["wm_err"], 6),
            ]
        )
        + "\n"
    )

    lines.append("\n## Ablations (delta vs full on same FAST grid)\n")
    for t in TASKS:
        lines.append(f"\n### {t}\n")
        lines.append(
            md_row(
                [
                    "variant",
                    "seq_acc",
                    "rolling",
                    "scen_div",
                    "pred_acc",
                    "wm_err",
                    "d_seq",
                    "shift_lat",
                    "ext_err",
                    "pred_err",
                ]
            )
            + "\n"
        )
        lines.append(md_row(["---"] * 10) + "\n")
        b = baseline[t]["seq_acc"]
        for name, pr in ABLATIONS:
            r = run_task(pr, t, STEPS_TASK)
            d = r["seq_acc"] - b
            lines.append(
                md_row(
                    [
                        name,
                        fmt(r["seq_acc"], 2),
                        fmt(r["rolling"], 2),
                        fmt(r["scen_div"], 3),
                        fmt(r["pred_acc"], 3),
                        fmt(r["wm_err"], 3),
                        fmt(d, 2),
                        str(r.get("shift_latency", "")),
                        fmt(float(r.get("ext_err_abs_mean", 0.0)), 3) if "ext_err_abs_mean" in r else "",
                        fmt(float(r.get("pred_err_abs_mean", 0.0)), 3) if "pred_err_abs_mean" in r else "",
                    ]
                )
                + "\n"
            )

    lines.append("\n## Phase-proxy comparison (no historical Phase 3/6 binaries in repo)\n")
    lines.append(
        "**stripped_789**: Phase 7–9 control/scenario/world/consequence gains zeroed (clusters/tissues still computed).\n"
    )
    lines.append("**full** = FAST defaults (Phase 9 full).\n\n")
    lines.append(md_row(["proxy", "task", "seq_acc", "instab", "clusters", "tissues", "hier"]) + "\n")
    lines.append(md_row(["---"] * 7) + "\n")
    for pname, pr in [("stripped_789", STRIPPED_789), ("full", FAST)]:
        for t in TASKS:
            r = run_task(pr, t, STEPS_TASK)
            lines.append(
                md_row(
                    [
                        pname,
                        t,
                        fmt(r["seq_acc"], 2),
                        fmt(r["instab"], 3),
                        str(r["clusters"]),
                        str(r["tissues"]),
                        fmt(r["hier"], 3),
                    ]
                )
                + "\n"
            )

    out = Path(__file__).resolve().parent / "validation_phase9_suite_results.txt"
    text = "".join(lines)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
