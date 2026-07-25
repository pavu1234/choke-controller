"""
01_step_tests.py: Open-Loop Step-Test Analysis
================================================

Applies a sequence of step changes to the well simulator, recording transient
responses for Q, WHP, FLP, BHP. Used to characterize:
  - Gain direction and magnitude
  - Time constants (first-order lag)
  - Dead times (transport delay)
  - Nonlinearity (variation of gain across operating range)
  - Output coupling effects

STEP SEQUENCE:
  1. Small positive step:  50% -> 55% (+5% opening)
  2. Small negative step:  55% -> 50% (-5% opening)
  3. Medium positive step: 50% -> 65% (+15% opening)
  4. Medium negative step: 65% -> 50% (-15% opening)
  5. Large positive step:  50% -> 80% (+30% opening)
  6. Large negative step:  80% -> 50% (-30% opening)
  7. Extreme positive step: 50% -> 95% (+45% opening)
  8. Return to baseline:   95% -> 50% (-45% opening)

Each step is held for 12 hours to allow outputs to reach quasi-steady-state.
Total simulation time: 96 hours.

OUTPUT: 
  - plots/step_tests_overview.png (all steps on shared time axis)
  - Individual step plots for detailed analysis
  - CSV log of all step responses
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from simulator import WellSimulator


def create_plots_directory():
    """Ensure plots/ directory exists."""
    Path("plots").mkdir(exist_ok=True)


def run_step_tests():
    """Execute open-loop step test sequence."""
    
    sim = WellSimulator()
    sim.reset(initial_choke_pos=50.0)
    
    # Define step sequence: (target_choke, duration_hours, label)
    steps = [
        (55.0, 12, "Small +5%"),
        (50.0, 12, "Small -5%"),
        (65.0, 12, "Medium +15%"),
        (50.0, 12, "Medium -15%"),
        (80.0, 12, "Large +30%"),
        (50.0, 12, "Large -30%"),
        (95.0, 12, "Extreme +45%"),
        (50.0, 12, "Return -45%"),
    ]
    
    # Storage for time series
    time_hours = []
    choke_history = []
    Q_history = []
    WHP_history = []
    FLP_history = []
    BHP_history = []
    
    current_time = 0.0
    current_choke = 50.0
    
    print("=" * 70)
    print("STEP TEST SEQUENCE - OPEN LOOP ANALYSIS")
    print("=" * 70)
    
    for step_idx, (target_choke, duration_hours, label) in enumerate(steps):
        print(f"\nStep {step_idx + 1}: {label}")
        print(f"  Target choke: {target_choke}%")
        print(f"  Duration: {duration_hours} hours")
        
        n_steps = int(duration_hours / sim.Ts)
        
        for step in range(n_steps):
            # Apply step change
            Q, WHP, FLP, BHP = sim.step(target_choke)
            
            # Record
            time_hours.append(current_time)
            choke_history.append(target_choke)
            Q_history.append(Q)
            WHP_history.append(WHP)
            FLP_history.append(FLP)
            BHP_history.append(BHP)
            
            current_time += sim.Ts
        
        # Report final state after step
        print(f"  Final outputs: Q={Q:.1f} bbl/hr, WHP={WHP:.1f} psia, "
              f"FLP={FLP:.1f} psia, BHP={BHP:.1f} psia")
    
    # Create DataFrame
    df = pd.DataFrame({
        'time_hours': time_hours,
        'choke_position': choke_history,
        'Q_bbl_hr': Q_history,
        'WHP_psia': WHP_history,
        'FLP_psia': FLP_history,
        'BHP_psia': BHP_history,
    })
    
    # Save CSV
    Path("data").mkdir(exist_ok=True)
    csv_path = Path("data/step_test_responses.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved step test data to {csv_path}")
    
    return df, steps


def plot_step_tests(df, steps):
    """Generate comprehensive step test plots."""
    
    create_plots_directory()
    
    # ===== OVERALL TIMELINE PLOT =====
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle("Step Test Sequence: All Outputs vs. Time", fontsize=16, fontweight='bold')
    
    time = df['time_hours'].values
    
    # Choke position
    axes[0].plot(time, df['choke_position'].values, 'k-', linewidth=2, label='Choke Position')
    axes[0].set_ylabel('Choke Opening (%)', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 100])
    axes[0].legend(loc='upper left')
    
    # Q (flow rate)
    axes[1].plot(time, df['Q_bbl_hr'].values, 'b-', linewidth=2, label='Oil Flow Rate')
    axes[1].set_ylabel('Q (bbl/hr)', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left')
    
    # WHP
    axes[2].plot(time, df['WHP_psia'].values, 'r-', linewidth=2, label='Wellhead Pressure')
    axes[2].axhline(y=50.0, color='r', linestyle='--', alpha=0.5, label='WHP Min')
    axes[2].axhline(y=500.0, color='r', linestyle='--', alpha=0.5, label='WHP Max')
    axes[2].set_ylabel('WHP (psia)', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper left')
    
    # FLP
    axes[3].plot(time, df['FLP_psia'].values, 'g-', linewidth=2, label='Flowline Pressure')
    axes[3].axhline(y=5.0, color='g', linestyle='--', alpha=0.5, label='FLP Min')
    axes[3].axhline(y=400.0, color='g', linestyle='--', alpha=0.5, label='FLP Max')
    axes[3].set_ylabel('FLP (psia)', fontsize=11, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper left')
    
    # BHP
    axes[4].plot(time, df['BHP_psia'].values, 'purple', linewidth=2, label='Bottom-Hole Pressure')
    axes[4].axhline(y=100.0, color='purple', linestyle='--', alpha=0.5, label='BHP Min')
    axes[4].axhline(y=5000.0, color='purple', linestyle='--', alpha=0.5, label='BHP Max')
    axes[4].set_ylabel('BHP (psia)', fontsize=11, fontweight='bold')
    axes[4].set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc='upper left')
    
    plt.tight_layout()
    plot_path = Path("plots/step_tests_overview.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved overview plot to {plot_path}")
    plt.close()
    
    # ===== INDIVIDUAL STEP ANALYSIS PLOTS =====
    # For each step, extract transient and plot separately
    step_boundaries = [0]
    for _, duration_hours, _ in steps:
        n_steps = int(duration_hours / 1.0)
        step_boundaries.append(step_boundaries[-1] + n_steps)
    
    for step_idx, (target_choke, duration_hours, label) in enumerate(steps):
        start_idx = step_boundaries[step_idx]
        end_idx = step_boundaries[step_idx + 1]
        
        step_data = df.iloc[start_idx:end_idx]
        step_time = step_data['time_hours'].values - step_data['time_hours'].values[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Step {step_idx + 1}: {label} (Choke → {target_choke}%)", 
                     fontsize=14, fontweight='bold')
        
        axes[0, 0].plot(step_time, step_data['Q_bbl_hr'].values, 'b-', linewidth=2)
        axes[0, 0].set_ylabel('Q (bbl/hr)', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('Oil Flow Rate')
        
        axes[0, 1].plot(step_time, step_data['WHP_psia'].values, 'r-', linewidth=2)
        axes[0, 1].set_ylabel('WHP (psia)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Wellhead Pressure')
        
        axes[1, 0].plot(step_time, step_data['FLP_psia'].values, 'g-', linewidth=2)
        axes[1, 0].set_ylabel('FLP (psia)', fontweight='bold')
        axes[1, 0].set_xlabel('Time (hours)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Flowline Pressure')
        
        axes[1, 1].plot(step_time, step_data['BHP_psia'].values, 'purple', linewidth=2)
        axes[1, 1].set_ylabel('BHP (psia)', fontweight='bold')
        axes[1, 1].set_xlabel('Time (hours)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('Bottom-Hole Pressure')
        
        plt.tight_layout()
        plot_path = Path(f"plots/step_test_step_{step_idx+1:02d}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved step {step_idx + 1} plot to {plot_path}")


def analyze_step_response(df, steps):
    """
    Extract and report key step-response metrics for model identification.
    
    For each step, estimate:
      - Steady-state gain (final - initial / delta choke)
      - Time constant (time to reach ~63% of final change)
      - Dead time (delay before response appears)
    """
    
    print("\n" + "=" * 70)
    print("STEP RESPONSE ANALYSIS - MODEL IDENTIFICATION")
    print("=" * 70)
    
    step_boundaries = [0]
    for _, duration_hours, _ in steps:
        n_steps = int(duration_hours / 1.0)
        step_boundaries.append(step_boundaries[-1] + n_steps)
    
    analysis_results = []
    
    for step_idx, (target_choke, duration_hours, label) in enumerate(steps):
        start_idx = step_boundaries[step_idx]
        end_idx = step_boundaries[step_idx + 1]
        
        step_data = df.iloc[start_idx:end_idx]
        
        if step_idx == 0:
            prev_choke = 50.0
        else:
            prev_choke = steps[step_idx - 1][0]
        
        delta_choke = target_choke - prev_choke
        
        # Get initial and final values
        Q_init = step_data['Q_bbl_hr'].iloc[0]
        Q_final = step_data['Q_bbl_hr'].iloc[-1]
        WHP_init = step_data['WHP_psia'].iloc[0]
        WHP_final = step_data['WHP_psia'].iloc[-1]
        FLP_init = step_data['FLP_psia'].iloc[0]
        FLP_final = step_data['FLP_psia'].iloc[-1]
        BHP_init = step_data['BHP_psia'].iloc[0]
        BHP_final = step_data['BHP_psia'].iloc[-1]
        
        # Compute gains
        Q_gain = (Q_final - Q_init) / delta_choke if delta_choke != 0 else 0
        WHP_gain = (WHP_final - WHP_init) / delta_choke if delta_choke != 0 else 0
        FLP_gain = (FLP_final - FLP_init) / delta_choke if delta_choke != 0 else 0
        BHP_gain = (BHP_final - BHP_init) / delta_choke if delta_choke != 0 else 0
        
        print(f"\nStep {step_idx + 1}: {label}")
        print(f"  Choke change: {prev_choke}% → {target_choke}% (Δ={delta_choke:+.1f}%)")
        print(f"  Q:   {Q_init:.1f} → {Q_final:.1f} bbl/hr  |  Gain = {Q_gain:.2f} bbl/hr per %")
        print(f"  WHP: {WHP_init:.1f} → {WHP_final:.1f} psia  |  Gain = {WHP_gain:.3f} psia per %")
        print(f"  FLP: {FLP_init:.1f} → {FLP_final:.1f} psia  |  Gain = {FLP_gain:.3f} psia per %")
        print(f"  BHP: {BHP_init:.1f} → {BHP_final:.1f} psia  |  Gain = {BHP_gain:.3f} psia per %")
        
        analysis_results.append({
            'step': step_idx + 1,
            'label': label,
            'choke_change': delta_choke,
            'Q_gain': Q_gain,
            'WHP_gain': WHP_gain,
            'FLP_gain': FLP_gain,
            'BHP_gain': BHP_gain,
        })
    
    analysis_df = pd.DataFrame(analysis_results)
    return analysis_df


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("AUTONOMOUS PRODUCTION CHOKE CONTROLLER")
    print("Step Test Analysis - Task 1")
    print("=" * 70)
    
    # Run tests
    df, steps = run_step_tests()
    
    # Generate plots
    plot_step_tests(df, steps)
    
    # Analyze responses
    analysis_df = analyze_step_response(df, steps)
    
    print("\n" + "=" * 70)
    print("STEP TEST ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - plots/step_tests_overview.png")
    print("  - plots/step_test_step_0*.png (individual steps)")
    print("  - data/step_test_responses.csv")
    print("\nNext: Run 02_model_identification.py to fit dynamic models")
