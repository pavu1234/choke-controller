````markdown
# Autonomous Production Choke Controller

## Overview

This project develops and validates a **multivariable feedback control system** for autonomous management of oil well production through choke valve adjustment. The controller maintains production rate, wellhead pressure, and flowline pressure within operational bounds while optimizing recovery.

### Key Features

- **Physics-based simulator**: Nonlinear model of well dynamics including inflow, outflow, and pressure coupling
- **System identification**: Automated extraction of transfer functions from step test data
- **Multivariable control**: Decoupling + PID architecture for coordinated regulation
- **Closed-loop validation**: Simulation-based performance testing across operating scenarios
- **Robustness analysis**: Parameter sensitivity and model uncertainty quantification

---

## Architecture

```
┌─────────────┐
│  Setpoints  │
│ (Q, WHP, ...)  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│   Decoupler      │  Inverse model compensation
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  PID Regulators  │  (3 loops: Q, WHP, FLP)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐      ┌─────────────────┐
│ Choke Command    │─────▶│  Well Simulator │
│   (0–100%)       │      │  (Physics Model)│
└──────────────────┘      └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │  Outputs        │
                          │ (Q, WHP, FLP)   │
                          └────────┬────────┘
                                   │
                                   └──────────▶ [Feedback]
```

---

## Simulation Pipeline

### 1. **Step Test Analysis** (`01_step_tests.py`)

**Objective**: Characterize open-loop dynamics by applying choke step changes.

**Outputs**:
- 8-step sequence (±5% to ±45% changes)
- Transient responses for Q, WHP, FLP, BHP
- `data/step_test_responses.csv`
- `plots/step_tests_overview.png` + individual step plots

**Example**:
```python
python 01_step_tests.py
```

---

### 2. **Model Identification** (`02_model_identification.py`)

**Objective**: Fit first-order + dead time transfer functions to step responses.

**Model Structure**:
```
G(s) = K * exp(-s*Td) / (tau*s + 1)
```

Where:
- **K**: Steady-state gain (output change per % choke opening)
- **tau**: Time constant (~settling time / 3)
- **Td**: Transport delay (dead time)

**Outputs**:
- `data/model_identification_results.csv` (K, tau, Td for each step & output)
- `plots/model_identification_overview.png`

**Example**:
```python
python 02_model_identification.py
```

---

### 3. **Controller Design** (`03_controller_design.py`)

**Objective**: Synthesize multivariable decoupling + PID controller.

**Control Strategy**:

1. **Decoupling**: Compensate for cross-coupling between outputs
   - Choke primarily controls Q
   - Q affects WHP and FLP (through pressure feedback)
   - Pre-compensate using inverse model

2. **PID Loops**: Three independent feedback regulators
   ```
   u = Kp*e + Ki*∫e + Kd*de/dt
   ```
   - Q tracking: higher bandwidth (~0.01 hr⁻¹)
   - WHP regulation: moderate bandwidth
   - FLP regulation: moderate bandwidth

3. **Safety**: Hard limits prevent constraint violations
   - BHP < 150 psia → close choke
   - WHP > 400 psia → close choke
   - FLP > 300 psia → close choke

**Controller Gains**:
```python
Kp_Q  = 0.01   # Production rate proportional
Ki_Q  = 0.002  # Production rate integral

Kp_WHP = 0.05  # Pressure proportional
Ki_WHP = 0.01  # Pressure integral

# Similar for FLP
```

**Outputs**:
- `MultivariableController` class instance
- Console output of tuning parameters

**Example**:
```python
python 03_controller_design.py
```

---

### 4. **Closed-Loop Simulation** (`04_closed_loop_simulation.py`)

**Objective**: Validate controller performance in closed-loop across scenarios.

**Scenarios**:

1. **Nominal**: Setpoint tracking with step changes (0 → 48h @ 2500 bbl/hr, 48 → 72h @ 3000 bbl/hr, return)
2. **Degraded**: Reservoir depleted (BHP decline over time)
3. **Pressure Limit**: Operation near safety limits (WHP → 380 psia)
4. **Rate Ramp**: Progressive increase to maximum production (1500 → 4500 bbl/hr)

**Key Metrics**:
- Mean Absolute Error (MAE) for Q and WHP tracking
- Root Mean Square Error (RMSE)
- Constraint violation count (WHP, BHP limits)

**Outputs**:
- CSV files: `data/closed_loop_*.csv` (one per scenario)
- Plot: `plots/closed_loop_performance.png` (4-panel figure)

**Example**:
```python
python 04_closed_loop_simulation.py
```

**Example Output**:
```
Performance Metrics for Nominal:
  Q tracking:   MAE=45.2 bbl/hr, RMSE=78.5 bbl/hr
  WHP tracking: MAE=12.3 psia, RMSE=18.9 psia
  WHP > 400 psia violations: 0
  BHP < 150 psia violations: 0
```

---

### 5. **Robustness Analysis** (`05_robustness_analysis.py`)

**Objective**: Quantify controller sensitivity to model uncertainty and parameter variations.

**Analysis Methods**:

1. **Monte Carlo**: Vary model parameters (±10–50%) and run closed-loop simulations
2. **Sensitivity**: Compute ∂(performance) / ∂(parameter) for each gain
3. **Failure modes**: Identify operating conditions leading to constraint violations

**Outputs**:
- `plots/robustness_monte_carlo.png` (distribution of tracking errors)
- `plots/robustness_sensitivity.png` (parameter sensitivity heatmap)
- `data/robustness_results.csv`
- Console: Risk assessment (e.g., "5% probability of WHP > 400 psia")

**Example**:
```python
python 05_robustness_analysis.py
```

---

## File Structure

```
choke-controller/
├── 01_step_tests.py              # Step test generation & analysis
├── 02_model_identification.py    # Transfer function fitting
├── 03_controller_design.py       # Multivariable controller synthesis
├── 04_closed_loop_simulation.py  # Validation across scenarios
├── 05_robustness_analysis.py     # Parameter sensitivity & Monte Carlo
├── simulator.py                  # Well dynamics physics model
├── controller_design.py          # MultivariableController class (imported)
├── data/                         # Output CSV files
│   ├── step_test_responses.csv
│   ├── model_identification_results.csv
│   ├── closed_loop_nominal.csv
│   ├── closed_loop_degraded.csv
│   └── ...
├── plots/                        # Generated figures
│   ├── step_tests_overview.png
│   ├── step_test_step_01.png
│   ├── ...
│   ├── model_identification_overview.png
│   ├── closed_loop_performance.png
│   └── robustness_monte_carlo.png
└── README.md                     # This file
```

---

## Quick Start

### Prerequisites

```bash
pip install numpy scipy pandas matplotlib
```

### Run Full Pipeline

```bash
# Sequential execution (recommended)
python 01_step_tests.py              # Generate step test data (~5 min)
python 02_model_identification.py    # Fit models (~2 min)
python 03_controller_design.py       # Design controller (~1 min)
python 04_closed_loop_simulation.py  # Validate performance (~10 min)
python 05_robustness_analysis.py     # Robustness study (~15 min)
```

Or run individually by importing classes:

```python
from simulator import WellSimulator
from controller_design import MultivariableController

sim = WellSimulator()
controller = MultivariableController()

# Run custom scenario
for t in range(1000):
    choke_cmd = controller.compute_command(
        Q_meas=sim.Q, WHP_meas=sim.WHP, 
        FLP_meas=sim.FLP, BHP_meas=sim.BHP
    )
    Q, WHP, FLP, BHP = sim.step(choke_cmd)
```

---

## Key Design Decisions

### 1. **Sampling Time**
- Ts = 1 minute for step tests and model identification (physical realism)
- Ts = 1 second for closed-loop simulation (computational detail)

### 2. **Nonlinear Well Model**
- Quadratic pressure-drop relationship: ΔP ∝ Q² (inertia)
- Beta-law valve nonlinearity: Effective choke ∝ (position)^1.2
- Depletion: BHP declines slowly (~0.05 psia per 1000 bbl) as reservoir is produced

### 3. **Decoupling Strategy**
- Simplified inverse model (linear approximation)
- Additive compensation: `u_decoupled = u_base + u_comp`
- Limits sensitivity to unmodeled dynamics

### 4. **Safety Prioritization**
- Hard limits (BHP, WHP, FLP) override setpoint tracking
- Anti-windup: Integral terms clipped at ±100
- Redundant checks in compute_command() method

---

## Expected Results

### Step Test Phase
- **Time to steady state**: ~12 hours per step
- **Gain nonlinearity**: ±5% steps show different gains than ±45% steps (nonlinear)
- **Dead times**: Q response faster (<1h) than BHP response (>2h)

### Closed-Loop Performance

| Scenario        | Q Tracking MAE | WHP Tracking MAE | Constraint Violations |
|-----------------|----------------|------------------|-----------------------|
| Nominal         | ~40 bbl/hr     | ~10 psia         | 0                     |
| Degraded        | ~120 bbl/hr    | ~15 psia         | 0                     |
| Pressure Limit  | ~80 bbl/hr     | ~20 psia         | 0                     |
| Rate Ramp       | ~100 bbl/hr    | ~25 psia         | 0                     |

### Robustness
- **Gain uncertainty (±20%)**: MAE increases by ~30–40%
- **Time constant uncertainty (±30%)**: Slow settling but stable
- **Dead time uncertainty (±50%)**: Modest increase in overshoot (~5–8%)

---

## Troubleshooting

### Issue: "Data directory not found"
**Solution**: Scripts auto-create `data/` and `plots/` directories.

### Issue: "Constraint violations in closed-loop"
**Possible causes**:
1. Aggressive setpoints (WHP_ref > 350 psia)
2. Tuning gains too high (reduce Kp, Ki)
3. Reservoir depleted below BHP_min

**Fix**: Adjust setpoints or tune controller gains (see `03_controller_design.py`).

### Issue: "Model fit error too high"
**Possible causes**:
1. Dead time detection failed (change threshold in `identify_dead_time()`)
2. Data contains noise (add smoothing filter)

**Fix**: Inspect CSV output and tune fitting parameters.

---

## References

1. **Well Production Control**
   - Skogestad, S. (2004). "Control of Integrating Systems." *Industrial & Engineering Chemistry Research*.
   - Godhavn, J. M. (2006). "Control Requirements for Autonomous Subsea Production Systems." *SPE Prod Facilities*.

2. **Multivariable Control**
   - Skogestad, S., & Postlethwaite, I. (2005). *Multivariable Feedback Control: Analysis and Design* (2nd ed.).
   - Ogunnaike, B. A., & Ray, W. H. (1994). *Process Dynamics, Modeling, and Control*.

3. **System Identification**
   - Ljung, L. (1999). *System Identification: Theory for the User* (2nd ed.).
   - Söderström, T., & Stoica, P. (1989). *System Identification*.

---

## License

MIT License. See LICENSE file for details.

---

## Contact

For questions or contributions, contact the project maintainer.
````
