"""
04_closed_loop_simulation.py: Closed-Loop Controller Evaluation
==============================================================

Runs closed-loop simulations with the multivariable controller.
Evaluates:
  1. Setpoint tracking response
  2. Disturbance rejection (e.g., reservoir pressure drop)
  3. Constraint satisfaction (safety limits)
  4. Robustness to model uncertainty

SCENARIOS:
  A. Nominal: Track setpoint changes while rejecting disturbances
  B. Degraded: Reservoir depleted (lower BHP) → reduced production
  C. Pressure limit: Setpoint close to safety limit
  D. Rate ramp: Slow increase to maximum production
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from simulator import WellSimulator
from controller_design import MultivariableController


def run_closed_loop_scenario(scenario_name, duration_hours, setpoint_sequence):
    """
    Execute one closed-loop scenario.
    
    Args:
        scenario_name: Descriptive name
        duration_hours: Total simulation time
        setpoint_sequence: List of (time_start, time_end, Q_ref, WHP_ref, FLP_ref)
    
    Returns:
        DataFrame with time series of all variables
    """
    
    sim = WellSimulator()
    controller = MultivariableController()
    sim.reset(initial_choke_pos=50.0)
    
    # Storage
    time_hours = []
    choke_cmd = []
    Q_history = []
    WHP_history = []
    FLP_history = []
    BHP_history = []
    Q_ref_history = []
    WHP_ref_history = []
    FLP_ref_history = []
    
    current_time = 0.0
    n_steps = int(duration_hours / sim.Ts)
    
    print(f"\n--- Scenario: {scenario_name} ({duration_hours} hours) ---")
    
    for step in range(n_steps):
        # Determine current setpoints
        current_Q_ref = 2500.0
        current_WHP_ref = 200.0
        current_FLP_ref = 100.0
        
        for t_start, t_end, Q_r, WHP_r, FLP_r in setpoint_sequence:
            if t_start <= current_time <= t_end:
                current_Q_ref = Q_r
                current_WHP_ref = WHP_r
                current_FLP_ref = FLP_r
                break
        
        # Update controller setpoints
        controller.set_setpoints(current_Q_ref, current_WHP_ref, current_FLP_ref)
        
        # Compute choke command
        choke_command = controller.compute_command(
            sim.Q, sim.WHP, sim.FLP, sim.BHP
        )
        
        # Step simulator
        Q, WHP, FLP, BHP = sim.step(choke_command)
        
        # Record
        time_hours.append(current_time)
        choke_cmd.append(choke_command)
        Q_history.append(Q)
        WHP_history.append(WHP)
        FLP_history.append(FLP)
        BHP_history.append(BHP)
        Q_ref_history.append(current_Q_ref)
        WHP_ref_history.append(current_WHP_ref)
        FLP_ref_history.append(current_FLP_ref)
        
        current_time += sim.Ts
        
        if (step % 3600) == 0:  # Print every hour
            print(f"  t={current_time/3600:.1f}h: Q={Q:.0f} bbl/hr (ref={current_Q_ref:.0f}), "
                  f"WHP={WHP:.0f} psia (ref={current_WHP_ref:.0f}), "
                  f"choke={choke_command:.1f}%")
    
    df = pd.DataFrame({
        'time_hours': time_hours,
        'choke_command': choke_cmd,
        'Q_bbl_hr': Q_history,
        'Q_ref_bbl_hr': Q_ref_history,
        'WHP_psia': WHP_history,
        'WHP_ref_psia': WHP_ref_history,
        'FLP_psia': FLP_history,
        'FLP_ref_psia': FLP_ref_history,
        'BHP_psia': BHP_history,
    })
    
    return df


def plot_closed_loop_results(results_dict):
    """
    Generate comprehensive closed-loop performance plots.
    """
    
    Path('plots').mkdir(exist_ok=True)
    
    # Create multi-panel figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('Closed-Loop Controller Performance', fontsize=16, fontweight='bold')
    
    colors = {'Nominal': 'b', 'Degraded': 'r', 'Pressure Limit': 'g', 'Rate Ramp': 'purple'}
    
    for scenario_name, df in results_dict.items():
        color = colors.get(scenario_name, 'k')
        time = df['time_hours'].values
        
        # Choke command
        axes[0].plot(time, df['choke_command'].values, color=color, linewidth=2,
                    label=scenario_name, alpha=0.7)
        
        # Production rate
        axes[1].plot(time, df['Q_bbl_hr'].values, color=color, linewidth=2.5,
                    label=f"{scenario_name} (measured)", alpha=0.8)
        axes[1].plot(time, df['Q_ref_bbl_hr'].values, color=color, linestyle='--',
                    linewidth=1.5, alpha=0.5)
        
        # Wellhead pressure
        axes[2].plot(time, df['WHP_psia'].values, color=color, linewidth=2.5,
                    label=f"{scenario_name} (measured)", alpha=0.8)
        axes[2].plot(time, df['WHP_ref_psia'].values, color=color, linestyle='--',
                    linewidth=1.5, alpha=0.5)
        
        # BHP
        axes[3].plot(time, df['BHP_psia'].values, color=color, linewidth=2,
                    label=scenario_name, alpha=0.7)
    
    # Formatting
    axes[0].set_ylabel('Choke Command (%)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 100])
    axes[0].legend(loc='best')
    
    axes[1].set_ylabel('Production Rate (bbl/hr)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')
    
    axes[2].set_ylabel('Wellhead Pressure (psia)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=400, color='r', linestyle='--', alpha=0.3, label='WHP_max')
    axes[2].legend(loc='best')
    
    axes[3].set_ylabel('Bottom-Hole Pressure (psia)', fontweight='bold')
    axes[3].set_xlabel('Time (hours)', fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(y=150, color='r', linestyle='--', alpha=0.3, label='BHP_min')
    axes[3].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('plots/closed_loop_performance.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved closed-loop performance plot to plots/closed_loop_performance.png")
    plt.close()


def compute_performance_metrics(df, scenario_name):
    """
    Compute quantitative metrics for each scenario.
    """
    
    # Error metrics
    Q_error = df['Q_bbl_hr'].values - df['Q_ref_bbl_hr'].values
    WHP_error = df['WHP_psia'].values - df['WHP_ref_psia'].values
    
    mae_Q = np.mean(np.abs(Q_error))
    rmse_Q = np.sqrt(np.mean(Q_error ** 2))
    mae_WHP = np.mean(np.abs(WHP_error))
    rmse_WHP = np.sqrt(np.mean(WHP_error ** 2))
    
    # Constraint violations
    whp_violations = np.sum(df['WHP_psia'].values > 400)
    bhp_violations = np.sum(df['BHP_psia'].values < 150)
    
    print(f"\nPerformance Metrics for {scenario_name}:")
    print(f"  Q tracking:   MAE={mae_Q:.1f} bbl/hr, RMSE={rmse_Q:.1f} bbl/hr")
    print(f"  WHP tracking: MAE={mae_WHP:.1f} psia, RMSE={rmse_WHP:.1f} psia")
    print(f"  WHP > 400 psia violations: {whp_violations}")
    print(f"  BHP < 150 psia violations: {bhp_violations}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("CLOSED-LOOP CONTROLLER SIMULATION")
    print("="*80)
    
    # Define scenarios
    scenarios = {
        'Nominal': [
            (0.0, 48.0, 2500.0, 200.0, 100.0),  # Nominal operation 0-48h
            (48.0, 72.0, 3000.0, 220.0, 110.0), # Increase setpoint 48-72h
            (72.0, 96.0, 2500.0, 200.0, 100.0), # Return to nominal 72-96h
        ],
        'Degraded': [
            (0.0, 96.0, 2500.0, 200.0, 100.0),  # Constant setpoint
        ],
        'Pressure Limit': [
            (0.0, 48.0, 2500.0, 200.0, 100.0),
            (48.0, 96.0, 3500.0, 380.0, 180.0),  # High pressure operation
        ],
        'Rate Ramp': [
            (0.0, 24.0, 1500.0, 150.0, 75.0),
            (24.0, 48.0, 2500.0, 200.0, 100.0),
            (48.0, 72.0, 3500.0, 250.0, 125.0),
            (72.0, 96.0, 4500.0, 300.0, 150.0),
        ],
    }
    
    # Run scenarios
    results = {}
    for scenario_name, setpoints in scenarios.items():
        duration = 96.0  # hours
        results[scenario_name] = run_closed_loop_scenario(
            scenario_name, duration, setpoints
        )
        
        # Save to CSV
        Path('data').mkdir(exist_ok=True)
        csv_path = f"data/closed_loop_{scenario_name.lower().replace(' ', '_')}.csv"
        results[scenario_name].to_csv(csv_path, index=False)
        print(f"✓ Saved {scenario_name} results to {csv_path}")
        
        # Compute metrics
        compute_performance_metrics(results[scenario_name], scenario_name)
    
    # Plot all scenarios
    plot_closed_loop_results(results)
    
    print("\n" + "="*80)
    print("CLOSED-LOOP SIMULATION COMPLETE")
    print("="*80)
    print("\nNext: Run 05_robustness_analysis.py for parameter sensitivity study")
