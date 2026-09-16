# Cellular Signal Substrate (Phase 9)

Deterministic, local-rule cellular AI simulation with signal propagation, roles, clusters/tissues, prediction, counterfactual scenario channels, and a local world-state estimate.

Single-file core: `cellular_signal_substrate.py` (Python + tkinter). No third-party packages required.

## Requirements

- **Python 3.10+** (3.11+ recommended)
- **tkinter** (usually included with Python)
  - Windows: comes with the official Python installer (enable **tcl/tk** if prompted)
  - macOS: `brew install python-tk` if `import tkinter` fails
  - Linux (Debian/Ubuntu): `sudo apt install python3-tk`

## How to

### 1. Download via the terminal

**Option A — clone with git (recommended):**

```bash
git clone https://github.com/bei22002-cpu/cellular-signal-substrate.git
cd cellular-signal-substrate
```

**Option B — download ZIP (no git):**

```bash
# macOS / Linux
curl -L -o cellular-signal-substrate.zip https://github.com/bei22002-cpu/cellular-signal-substrate/archive/refs/heads/master.zip
unzip cellular-signal-substrate.zip
cd cellular-signal-substrate-master
```

```powershell
# Windows PowerShell
Invoke-WebRequest -Uri "https://github.com/bei22002-cpu/cellular-signal-substrate/archive/refs/heads/master.zip" -OutFile cellular-signal-substrate.zip
Expand-Archive .\cellular-signal-substrate.zip -DestinationPath .
cd cellular-signal-substrate-master
```

### 2. Check Python

```bash
python --version
python -c "import tkinter; print('tkinter OK')"
```

### 3. Run the interactive app

```bash
python cellular_signal_substrate.py
```

A window opens with the cellular grid and a status bar. Use keyboard controls:

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `n` | Single step (when paused) |
| `r` | Reset center pulse |
| `s` | Left-stream injection |
| `c` | Clear |
| `x` | Full reset |
| `p` | Partial signal reset |
| `v` | Cycle visualization mode |
| `m` | Toggle metrics overlay |
| `t` | Cycle training / run modes |
| `e` | Eval sequential (reinforcement off) |
| `1` / `2` / `3` | Free modes INPUT_A / INPUT_B / INPUT_AB |
| `4`–`0`, `q`, `z`, `j`, `d`, `,` `.` `;` `'` | Select tasks |
| `F1`–`F4` | Moving target / obstacle / context / explore-exploit |

## Run validation (no GUI)

Empirical harnesses used for the Phase 9 report:

```bash
# Smoke tests, determinism, short ablations
python validation_phase9.py

# Full task suite + ablations (writes validation_phase9_suite_results.txt)
# Can take several minutes on the compact FAST grid
python validation_phase9_suite.py
```

Report: [Phase9_VALIDATION_REPORT.md](Phase9_VALIDATION_REPORT.md)

## Project layout

| File | Purpose |
|------|---------|
| `cellular_signal_substrate.py` | Simulation + tkinter UI |
| `validation_phase9.py` | Compact validation harness |
| `validation_phase9_suite.py` | Ablation + phase-proxy suite |
| `validation_phase9_suite_results.txt` | Latest suite metrics |
| `Phase9_VALIDATION_REPORT.md` | Formal validation write-up |

## Notes

- The system is **deterministic**: same params, mode, and step count → same trajectories.
- Default grid is **50×50**. Validation scripts use a smaller **14×14** “FAST” config so runs finish faster.
- No `pip install` step — standard library only.
