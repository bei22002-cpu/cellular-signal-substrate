# Phase 9 Cellular Intelligence System – Validation Report

**Artifact:** `cellular_signal_substrate.py` (single-file `Phase9World` + `WorldParams`)  
**Harnesses:** `validation_phase9.py` (smoke + determinism + short ablations), `validation_phase9_suite.py` (full task ablations + phase proxy; writes `validation_phase9_suite_results.txt`)  
**Date:** 2026-04-15  

This report stays empirical: mechanisms are classified as *present in code*, *causally active under stated conditions*, and *measurable via named fields/metrics*. No claims about intelligence or consciousness.

---

## 1. Executive Summary

| Area | Verdict |
|------|---------|
| **Determinism** | Confirmed: identical parameters, mode, and step count produce identical full-grid signal fingerprints (`validation_phase9.py`). |
| **Signal propagation & readout** | Functional on default-ish grids within typical horizons after transport/injection fixes (e.g. 32×32, 120 steps → non-zero `output_top_value`). |
| **Clusters / tissues / layers** | Implemented and updated every step (`run_cluster_analysis`, `run_tissue_analysis`, `compute_output_readout`, `_layer_activity`); counts non-zero in compact-grid runs. |
| **Meta-control (goal, meta_state, control_weight, path_weight, …)** | Implemented in `compute_next_grid`; ablations show **metric** shifts (e.g. `scenario_divergence` when goal/meta gains are zeroed) even when `sequential_task_accuracy` is unchanged on the FAST benchmark. |
| **Prediction** | `predicted_signal` is blended toward `signal` via `prediction_track_signal`; compact harness shows **non-saturated** `prediction_accuracy_score` (see §7). |
| **Counterfactual channels** | Excited at injection (`_couple_scenario_channels_at`) and via `scenario_from_signal_gain`; **50×50** cold-start shows nonzero mean divergence and large max cell |a−b|; **FAST 14×14** can show ~0 mean divergence while still supporting tasks (geometry-dependent). |
| **World-state estimate** | Disabling `world_state_gain` / `world_state_corr` raises EMA `world_model_error` (reproducible on INPUT_A ablations and per-task ablations in §4). |
| **Task suite A–E (+ moving target)** | On **FAST**, **300 steps** per run (`validation_phase9_suite_results.txt`): MOVING_TARGET **0.71**; OBSTACLE_NAV **1.00**; CONTEXT_SWITCH **1.00**; A_THEN_B **1.00**; B_THEN_A **1.00**; EXPLORE_EXPLOIT **0.86** (7 episodes each). On **FAST 800 steps** (`validation_phase9.py`), prior run: all **1.00** except MOVING **0.70** and EXPLORE **0.90**—episode count differs; use harness label when comparing. |
| **Phase 3 vs 6 vs 9 binaries** | **Not present** in this workspace. §5 uses a **parameter proxy** (`stripped_789`) vs **full** FAST—not historical builds. |

---

## 2. System Overview

- **Model:** 2D regular grid of `Cell` structs with local neighbor coupling each step (`compute_next_grid`), deterministic update order (nested `y`,`x` loops), then buffer swap (`swap_buffers`).
- **Inputs:** `_apply_injection` chooses between sequential-task injections (`_sequential_inject`) or “free” modes (`_injection_free`) depending on `mode` and `run_mode`.
- **Readout:** `compute_output_readout` sums `signal` in the right-hand band, split at `_output_split_y()` (balanced for MOVING_TARGET/CONTEXT_SWITCH in this build; reward uses environment-aligned timing in `_reward_episode`).
- **Global metrics:** `metrics()` aggregates sums/maxima over the grid (prediction, scenarios, meta, tissues, paths, etc.).
- **Training bookkeeping:** `_advance_sequential` ends an episode, calls `_reward_episode`, increments `sequential_total` / `sequential_correct`, applies `apply_episode_reinforcement`, then `partial_reset_signals_only`.

Code anchors (diffusion + eastward transport term):

```1008:1015:c:\Users\jmbei\AI Cell theory\cellular_signal_substrate.py
                up = cur[y - 1][x].signal if y > 0 else cs
                down = cur[y + 1][x].signal if y < h - 1 else cs
                left = row_cur[x - 1].signal if x > 0 else cs
                right = row_cur[x + 1].signal if x < w - 1 else cs

                neighbor_avg = (up + down + left + right) / 4.0
                east = self.p.eastward_transport
                new_signal = (cs + d * (neighbor_avg - cs) + east * (right - left)) * decay
```

Scenario preference modulation on the main signal:

```1164:1165:c:\Users\jmbei\AI Cell theory\cellular_signal_substrate.py
                scen_mod = clamp(1.0 + pref * self.p.scenario_pref_scale, 0.86, 1.14)
                new_signal = clamp(new_signal * scen_mod, -sig_cap, sig_cap)
```

Readout aggregation:

```1808:1831:c:\Users\jmbei\AI Cell theory\cellular_signal_substrate.py
    def compute_output_readout(self) -> None:
        ...
        self.output_top_value = top
        self.output_bottom_value = bot
```

---

## 3. Mechanism Verification Results

| Mechanism | Present | Causally active | Measurable | Where / how |
|-----------|---------|-----------------|------------|-------------|
| **Local signal propagation** | Yes | Yes | `total_signal`, `max_signal`, per-cell `signal` | `compute_next_grid`: diffusion + `eastward_transport` + role multipliers. |
| **Role differentiation** | Yes | Yes | `*_count` in `metrics()` | `role_update_policy` → `new_role` written each step. |
| **Memory persistence** | Yes | Yes | `avg_memory`, `max_memory` | `new_memory` from `memory_decay` / `memory_gain`. |
| **Cluster formation** | Yes | Yes | `active_cluster_count`, … | `run_cluster_analysis` flood-fill on `is_cluster_active`. |
| **Tissue formation** | Yes | Yes | `active_tissue_count`, `inter_tissue_edge_count`, … | `run_tissue_analysis`. |
| **Hierarchical layering** | Yes | Partial | `layer_activity`, `hierarchical_dependency_score` | `layer_from_x` buckets; score derived from activity sums. |
| **Meta-control** | Yes | Partial | `average_goal_signal`, `average_control_weight`, … | Goal/meta/control_weight updates in `compute_next_grid`; ablation `no_goal_meta` changes `scenario_divergence` on several tasks (§4). |
| **Prediction** | Yes | Yes | `prediction_accuracy_score`, `average_prediction_error`, `predictive_influence_score` | `predicted_signal` evolved then blended with `prediction_track_signal`; `pred_mod` on `new_signal`. |
| **Counterfactual (scenario_a/b)** | Yes | Yes | `scenario_divergence`, max \|a−b\| scan | Injection coupling `_couple_scenario_channels_at`; leak `scenario_from_signal_gain`; modulation `scen_mod`. |
| **World-state estimation** | Yes | Yes | `world_model_error` | `_local_env_sensed` → `world_state_estimate` / `external_error`. |
| **Consequence tracking** | Yes | Indirect | Via `consequence_trace` in scenario scores | `consequence_trace` update block; no single dedicated metric key. |

**Superficial vs doing work:**  
- **Always doing work:** signal propagation, memory, readout, episodic bookkeeping.  
- **Conditionally “visible” on metrics:** scenario **mean** divergence can be ~0 on small grids while max-cell divergence is nonzero elsewhere; **task accuracy** can be unchanged under some ablations even when auxiliary metrics move—so **causal necessity** must be judged per metric, not only `sequential_task_accuracy`.

---

## 4. Ablation Results

### 4.1 Free-run `MODE_INPUT_A` (compact harness)

**Protocol:** `validation_phase9.FAST`, `RUN_FREE`, `MODE_INPUT_A`, **220 steps** (`validation_phase9.py`).

| Variant | `current_decision` | `output_top` (rounded) | `output_bottom` | `world_model_error` EMA |
|---------|-------------------|------------------------|-----------------|-------------------------|
| full | TOP | 2061.18 | 2059.91 | 1.846 |
| `scenario_pref_scale=0` | **BOTH** | 2100.00 | 2100.00 | 1.846 |
| `prediction_bias=0` | TOP | 2061.39 | 2060.30 | 1.846 |
| `world_state_gain=world_state_corr=0` | TOP | 2061.18 | 2059.91 | **1.997** |

**Takeaway:** `scenario_pref_scale` is **causal for tie-breaking** on near-symmetric readouts; world gains are **causal** for EMA world-model error.

### 4.2 Per-task ablations (FAST, 300 steps, full task modes)

**Source:** `validation_phase9_suite_results.txt` (re-run: `python validation_phase9_suite.py`).

For each task, variants include: `full`, `no_scenario_pref`, `no_scenario_excitation`, `equal_scenario_bias`, `no_prediction_bias`, `no_prediction_tracking`, `no_world_update`, `no_goal_meta`, `freeze_adapt_path`.

**Observed patterns (quantitative, file-backed):**

| Finding | Evidence |
|---------|----------|
| **`sequential_task_accuracy` often invariant** | For MOVING_TARGET and most tasks, `d_seq = 0` across listed ablations except **EXPLORE_EXPLOIT** under `no_goal_meta` (**d_seq = +0.14**). |
| **Scenario divergence responds to excitation** | e.g. OBSTACLE_NAV: `scen_div` **0.283** full → **0.000** under `no_scenario_excitation`. |
| **Equal scenario biases collapse divergence** | OBSTACLE_NAV: `scen_div` **0.000** under `equal_scenario_bias`. |
| **World update ablation** | `wm_err` rises **~1.864 → ~1.995** consistently (`no_world_update`). |
| **Goal/meta ablation** | `scen_div` can jump sharply (e.g. MOVING 0→**2.904**; OBSTACLE 0.283→**5.050**)—indicates **strong coupling** between meta/scenario fields and the divergence metric, even when accuracy is unchanged. |

**Classification (for this benchmark):**

| Mechanism | Role on FAST 300-step task benchmark |
|-----------|--------------------------------------|
| `scenario_pref_scale` | Often **non-essential** for accuracy; **essential** for free-run tie break (§4.1). |
| Scenario excitation + bias | **Measurable** (divergence); **task accuracy** often unchanged. |
| Prediction tracking/bias | Changes `pred_acc` / `scen_div` in places; **accuracy** mostly unchanged. |
| World update | **Measurable** on `wm_err`; accuracy unchanged here. |
| Goal/meta | Large **metric** side-effects; **one** accuracy change on EXPLORE. |
| Freeze adapt/path | **No** accuracy delta in this table. |

---

## 5. Phase Comparison

**Historical Phase 3 / Phase 6 executables are not in this repository.** No side-by-side “phase binaries” exist.

**Proxy used instead:** `stripped_789` = all Phase 7–9 *parameter gains* for goal/meta/prediction/scenario/world/consequence set to **0** on the same `FAST` grid (clusters/tissues still computed). Compared to **`full`** = default `FAST`.

**Excerpt (`validation_phase9_suite_results.txt`):**

| proxy | task | seq_acc | instab | clusters | tissues | hier |
|-------|------|---------|--------|----------|---------|------|
| stripped_789 | EXPLORE_EXPLOIT | 1.00 | 0.002 | 2 | 1 | 1.167 |
| full | EXPLORE_EXPLOIT | 0.86 | 0.000 | 6 | 1 | 1.164 |

Other tasks: **identical** `seq_acc` between `stripped_789` and `full` on this proxy (MOVING 0.71; others 1.00).

**Interpretation:** On this benchmark, **adding Phase 7–9 parameters does not uniformly improve** `sequential_task_accuracy`; in one case (EXPLORE_EXPLOIT) the stripped proxy scores **higher**. This does **not** imply “phases are useless”—it means **this proxy is not equivalent to historical Phase 3/6/9**, and the chosen task reward + injection schedule can saturate accuracy under multiple settings.

---

## 6. Task Performance

**Success criterion:** `_reward_episode` returns `>0` on episode end; reported: `sequential_task_accuracy = sequential_correct / sequential_total`.

### A. Moving target routing (`TASK_MOVING_TARGET`)

- **Criterion:** `_reward_episode` uses moving-target-aligned routing (`_output_split_y` balanced; env step alignment in `_reward_episode`).
- **FAST 300-step suite:** accuracy **0.71**, 7 episodes.

### B. Obstacle avoidance (`TASK_OBSTACLE_NAV`)

- **Criterion:** reward favors configured top/bottom margin (see `_reward_episode`).
- **FAST 300-step suite:** **1.00**.

### C. Context-based output switching (`TASK_CONTEXT_SWITCH`)

- **Criterion:** `env_prefer_top` vs readout edge (with step alignment).
- **FAST 300-step suite:** **1.00**.

### D. Sequential input A→B vs B→A (`TASK_A_THEN_B`, `TASK_B_THEN_A`)

- **Criterion:** phase-structured injections + margin rules.
- **FAST 300-step suite:** **1.00** each.

### E. Explore vs exploit (`TASK_EXPLORE_EXPLOIT`)

- **Criterion:** alternating explore/exploit phases vs readout.
- **FAST 300-step suite:** **0.86** (full) vs **1.00** under `stripped_789` proxy in §5.

**Caveat:** Do not compare **300-step** and **800-step** harness rows without labeling; episode counts differ.

### Added legitimacy tasks (mechanism-targeted)

These were added to reduce “accuracy saturation” ambiguity by introducing **additional measurable outcomes** beyond win/loss.

#### F. Delayed consequence trap (`TASK_DELAYED_TRAP`)

- **Definition:** Episode \(k\) reward uses `env_prefer_top()`. At the end of episode \(k\), the system’s `current_decision` sets the *next* episode preference by flipping it: if it chose `TOP`, then episode \(k+1\) prefers `BOTTOM` and vice-versa. This creates a deterministic **delayed consequence** loop across episodes without introducing any new learning mechanism.
- **Measured:** `seq_acc` and time-series means over the run:
  - `ext_err`: mean per-step mean(|`external_error`|) across the grid
  - `pred_err`: mean per-step mean(|`prediction_error`|)

FAST 300-step baseline shows `seq_acc=1.00` with `ext_err≈1.808`, `pred_err≈1.934` (`validation_phase9_suite_results.txt`).

#### G. Within-episode regime shift (`TASK_REGIME_SHIFT`)

- **Definition:** During sequential **phase 2**, `env_prefer_top()` flips deterministically at `shift_at = seq_phase2_steps//2`. Reward at the end of the episode uses the **post-shift** regime via the same routing rule as `TASK_CONTEXT_SWITCH`.
- **Added metric:** `shift_lat` = number of phase-2 steps after the shift until the decision matches the new regime for 3 consecutive steps (measured by the suite harness, not by the world itself).
- **Measured:** `seq_acc`, `shift_lat`, plus `ext_err`/`pred_err` means.

FAST 300-step baseline shows `seq_acc=1.00`, `shift_lat=6`, `ext_err≈1.808`, `pred_err≈2.093`.

---

## 7. Prediction Analysis

- **Compact INPUT_A (220-step, `validation_phase9.py` smoke):** `prediction_accuracy_score ≈ 0.808`, `average_prediction_error ≈ 0.237`.
- **FAST task runs (`validation_phase9_suite_results.txt`):** `predictive_influence_score` is **large** (order **10¹–10²**) because it scales with `path_weight` mass in the grid—it is **not** “~0” in this build.
- **Ablations:** `no_prediction_tracking` lowers `prediction_accuracy_score` (e.g. MOVING_TARGET **0.321 → 0.296**).

**Operational scores (definitions from `metrics()`):**

| Score | Meaning |
|-------|---------|
| `prediction_accuracy_score` | \(1 / (1 + \text{mean}(|prediction\_error|))\) — higher is better. |
| `predictive_influence_score` | \(\sum |predicted\_signal|\cdot path\_weight / \sum path\_weight\) — **not** normalized to [0,1]. |

**Claim discipline:** “Prediction error decreases monotonically with experience” is **not** established here (would require logged time series per run).

---

## 8. Counterfactual Analysis

### 8.1 Divergence

| Setting | `scenario_divergence` (mean) | max cell \|a−b\| |
|---------|-------------------------------|------------------|
| FAST 14×14, INPUT_A, 80 steps | **0.000000** | **0.000000** |
| Default 50×50, INPUT_A, 80 steps | **0.086233** | **11.414458** |

### 8.2 `scenario_preference` and routing

- **Routing influence** is always present when `scen_mod` multiplies `new_signal` (see §3 code anchor in prior version; `pred_mod` / `scen_mod` block in `compute_next_grid`).
- **Preference magnitude:** `average_scenario_preference_abs` available in `metrics()` (used in suite script for 50×50 row in earlier tooling).

### 8.3 Quantitative scores (definitions used in this report)

| Name | Definition used |
|------|-------------------|
| **scenario_divergence_score** | `metrics()["scenario_divergence"]` — mean \|sa−sb\| per cell. |
| **counterfactual_benefit_score** | Mean over tasks A–E (+ MOVING) of \|`seq_acc(full)` − `seq_acc(no_scenario_pref)`| = **0.00** on FAST 300-step suite (no accuracy benefit detected for removing preference alone). |

**Note:** Benefit for **free-run** tie-breaking remains in §4.1 (`TOP` vs `BOTH`).

**Mechanism-targeted task note:** `TASK_DELAYED_TRAP` was intended to surface delayed-consequence benefits from counterfactual preference. On FAST 300-step runs, `seq_acc` remains **1.00** across listed ablations; the clearest measurable differences remain in auxiliary metrics (e.g. `pred_acc`, `wm_err`, and `ext_err`/`pred_err` means), not in win/loss.

---

## 9. World Model Analysis

- **Mechanism:** `_local_env_sensed` → `world_state_estimate` / `external_error` updates in `compute_next_grid`.
- **Measurable effect:** `no_world_update` raises `world_model_error` EMA to **~1.995** vs **~1.864** on FAST task ablations.

**Not fully validated:** `external_error` monotonic decrease under a scripted environment switch (would require a dedicated time-series log).

**New time-series evidence (REGIME_SHIFT):** On `TASK_REGIME_SHIFT`, `no_world_update` increases both:\n\n- `wm_err`: **1.863 → 1.995**\n- `ext_err` (mean per-step mean(|external_error|)): **1.808 → 1.956**\n\nwhile leaving `seq_acc=1.00` and `shift_lat=6` unchanged on this FAST benchmark. This supports the narrow claim that the world-model update loop is **causally active** (it reduces its own error metrics), but does not yet show a **performance** advantage under these task settings.

---

## 10. Emergence Analysis

From **phase-proxy table** (`validation_phase9_suite_results.txt`):

- **Cluster/tissue counts** differ between `stripped_789` and `full` (e.g. EXPLORE_EXPLOIT: clusters 2 vs 6).
- **Inter-tissue coordination:** use `inter_tissue_edge_count`, `average_inter_tissue_flow` from `metrics()` (not duplicated in suite file—extend script if needed).

**Conclusion:** structure is **real** (recomputed graphs) and **can differ** across settings; **causal impact on task accuracy** is **not** proven by cluster counts alone on this benchmark.

---

## 11. Determinism Results

`grid_signal_fingerprint` equality for two `run_world(FAST, MODE_INPUT_A, 100)` runs: **`True`** (`validation_phase9.py`).

No `random` in the hot path; rules are fixed arithmetic.

---

## 12. Failure Modes

| Mode | Evidence / mechanism |
|------|----------------------|
| **Geometry-dependent scenario metrics** | Mean `scenario_divergence` can read ~0 on FAST while 50×50 is nonzero (§8.1). |
| **Transport / horizon** | If injection + steps cannot energize the output strip, readout-based rewards fail (mitigated by multi-column injection + `eastward_transport` in this build). |
| **Signal saturation** | `signal_cap` clips `signal` widely under strong drive. |
| **Metric coupling under ablation** | Disabling goal/meta can **inflate** `scenario_divergence` without improving task accuracy—interpret metrics jointly. |

---

## 13. Limitations

1. No historical Phase 3/6 builds for true phase comparison.  
2. **FAST vs default 50×50** yield different emergent metric magnitudes.  
3. **Tissue “usage” ablation** (skipping `run_tissue_analysis` or zeroing tissue-derived terms) is **not** run—would require a code fork or feature flag.  
4. Task accuracy can be **saturated** under the engineered injection schedule, masking ablation effects—supplement with auxiliary metrics (§4.2).  

---

## 14. Conclusions

**Working (supported):** deterministic dynamics; episodic scoring; transport/readout fixes; scenario channel excitation on larger grids; world-model gain ablation effect on EMA error; prediction tracking/bias effects on prediction metrics.

**Partially working / context-dependent:** meta-control and scenario preference as **task-accuracy** drivers on FAST (often **neutral**); counterfactual **divergence** as a task predictor (often **uncorrelated** with accuracy here).

**Not demonstrated:** monotonic “learning curves” for prediction/world error without dedicated logged protocols.

**Unique (engineering):** single-file, deterministic, multi-channel local update with explicit injection-coupled scenario fields, episodic reinforcement, and grid diagnostics (clusters/tissues/paths).

**Reproduction:**  
- `python validation_phase9.py`  
- `python validation_phase9_suite.py` → read `validation_phase9_suite_results.txt`
