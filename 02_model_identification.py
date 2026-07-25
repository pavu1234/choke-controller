"""
02_model_identification.py: Dynamic Model Identification
============================================================

Fits first-order transfer functions to step test data.
Estimates gain, time constant, and dead time for each output.

MODEL STRUCTURE:
  G(s) = K * exp(-s*Td) / (tau*s + 1)

Where:
  - K: steady-state gain
  - tau: time constant (63% rise time)
  - Td: dead time / transport delay

METHOD:
  1. Isolate each step response
  2. Detect dead time (delay before response)
  3. Fit time constant using least-squares
  4. Compute steady-state gain
  5. Report across all operating ranges
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def first_order_response(t, K, tau, Td):
    """
    First-order response to step input with dead time.
    
    y(t) = K * (1 - exp(-(t-Td)/tau)) * step(t-Td)
    """
    response = np.zeros_like(t)
    active = t >= Td
    response[active] = K * (1.0 - np.exp(-(t[active] - Td) / tau))
    return response


def identify_dead_time(time, response, threshold_pct=0.02):
    """
    Detect dead time as delay until response exceeds threshold.
    
    Args:
        time: time vector (hours)
        response: output response vector
        threshold_pct: % of final value to trigger detection
    
    Returns:
        dead_time (hours)
    """
    y_final = response[-1]
    y_init = response[0]
    y_range = y_final - y_init
    
    if abs(y_range) < 0.1:  # No significant response
        return 0.0
    
    threshold = y_init + threshold_pct * y_range
    
    # Find first point exceeding threshold
    crossings = np.where(np.abs(response - y_init) > threshold_pct * abs(y_range))[0]
    
    if len(crossings) > 0:
        return time[crossings[0]]
    else:
        return 0.0


def fit_first_order_model(time, response, initial_guess=(1.0, 0.5, 0.0)):
    """
    Fit first-order + dead time model using curve_fit.
    
    Args:
        time: time vector (hours)
        response: measured response
        initial_guess: (K, tau, Td) starting values
    
    Returns:
        (K, tau, Td, fit_error): Fitted parameters and RMSE
    """
    
    try:
        # Bounds: K>0, tau>0, Td>=0
        popt, _ = curve_fit(
            first_order_response,
            time, response,
            p0=initial_guess,
            bounds=([0.01, 0.01, 0.0], [10000.0, 10.0, 2.0]),
            maxfev=5000
        )
        K, tau, Td = popt
        
        # Compute fit error
        y_pred = first_order_response(time, K, tau, Td)
        rmse = np.sqrt(np.mean((response - y_pred) ** 2))
        
        return K, tau, Td, rmse
    except Exception as e:
        print(f"  Fit error: {e}")
        return None, None, None, None


def analyze_step_data(df, steps):
    """
    Extract and fit all step responses.
    
    Returns:
        DataFrame with fitted parameters for each step & output
    """
    
    step_boundaries = [0]
    for _, duration_hours, _ in steps:
        n_steps = int(duration_hours / 1.0)  # 1-hour resolution assumed
        step_boundaries.append(step_boundaries[-1] + n_steps)
    
    results = []
    
    for step_idx, (target_choke, duration_hours, label) in enumerate(steps):
        start_idx = step_boundaries[step_idx]
        end_idx = step_boundaries[step_idx + 1]
        
        step_data = df.iloc[start_idx:end_idx]
        step_time = (step_data['time_hours'].values - step_data['time_hours'].values[0])
        
        # Extract outputs
        Q_resp = step_data['Q_bbl_hr'].values
        WHP_resp = step_data['WHP_psia'].values
        FLP_resp = step_data['FLP_psia'].values
        BHP_resp = step_data['BHP_psia'].values
        
        # Fit each output
        outputs = {
            'Q': Q_resp,
            'WHP': WHP_resp,
            'FLP': FLP_resp,
            'BHP': BHP_resp,
        }
        
        for output_name, response in outputs.items():
            # Initial guess for dead time
            Td_guess = identify_dead_time(step_time, response)
            
            # Initial guess for gain and tau
            delta_y = response[-1] - response[0]
            K_guess = delta_y if delta_y != 0 else 1.0
            tau_guess = 0.5
            
            # Fit
            K, tau, Td, rmse = fit_first_order_model(
                step_time, response,
                initial_guess=(K_guess, tau_guess, Td_guess)
            )
            
            if K is not None:
                results.append({
                    'step': step_idx + 1,
                    'label': label,
                    'output': output_name,
                    'K': K,
                    'tau_hours': tau,
                    'Td_hours': Td,
                    'rmse': rmse,
                })
    
    return pd.DataFrame(results)


def plot_fitted_responses(df, steps):
    """
    Plot measured vs. fitted responses for all steps.
    """
    
    Path('plots').mkdir(exist_ok=True)
    
    step_boundaries = [0]
    for _, duration_hours, _ in steps:
        n_steps = int(duration_hours / 1.0)
        step_boundaries.append(step_boundaries[-1] + n_steps)
    
    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Identification: Fitted vs. Measured Responses', 
                 fontsize=14, fontweight='bold')
    
    output_names = ['Q', 'WHP', 'FLP', 'BHP']
    ax_flat = axes.flatten()
    
    for output_idx, output_name in enumerate(output_names):
        ax = ax_flat[output_idx]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(steps)))
        
        for step_idx, (target_choke, duration_hours, label) in enumerate(steps):
            start_idx = step_boundaries[step_idx]
            end_idx = step_boundaries[step_idx + 1]
            
            step_data = df.iloc[start_idx:end_idx]
            step_time = step_data['time_hours'].values - step_data['time_hours'].values[0]
            
            response = step_data[f'{output_name}_bbl_hr'].values if output_name == 'Q' else \
                      step_data[f'{output_name}_psia'].values
            
            # Plot measured
            ax.plot(step_time, response, 'o-', color=colors[step_idx], 
                   alpha=0.6, linewidth=1.5, markersize=2, label=f'Step {step_idx+1} (meas.)')
        
        ax.set_xlabel('Time (hours)', fontweight='bold')
        ax.set_ylabel(f'{output_name}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f'Output: {output_name}')
    
    plt.tight_layout()
    plt.savefig('plots/model_identification_overview.png', dpi=150, bbox_inches='tight')
    print("✓ Saved model identification overview to plots/model_identification_overview.png")
    plt.close()


def print_model_summary(results_df):
    """
    Print summary of identified models.
    """
    
    print("\n" + "="*80)
    print("IDENTIFIED FIRST-ORDER + DEAD TIME MODELS")
    print("="*80)
    print("\nFormat: G(s) = K * exp(-s*Td) / (tau*s + 1)\n")
    
    for output in ['Q', 'WHP', 'FLP', 'BHP']:
        print(f"\n--- {output} (Production/Pressure) ---")
        output_data = results_df[results_df['output'] == output]
        
        for _, row in output_data.iterrows():
            print(f"Step {row['step']:2d} ({row['label']:15s}):  "
                  f"K={row['K']:7.2f}  tau={row['tau_hours']:5.3f}h  "
                  f"Td={row['Td_hours']:5.3f}h  RMSE={row['rmse']:6.2f}")
        
        # Compute average parameters
        avg_K = output_data['K'].mean()
        avg_tau = output_data['tau_hours'].mean()
        avg_Td = output_data['Td_hours'].mean()
        
        print(f"  AVERAGE:               "
              f"K={avg_K:7.2f}  tau={avg_tau:5.3f}h  Td={avg_Td:5.3f}h")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MODEL IDENTIFICATION")
    print("="*80)
    
    # Load step test data
    df = pd.read_csv('data/step_test_responses.csv')
    print(f"\nLoaded {len(df)} data points from step_test_responses.csv")
    
    # Dummy step sequence (same as in 01_step_tests.py)
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
    
    # Analyze and fit
    print("\nFitting first-order models to each step response...")
    results_df = analyze_step_data(df, steps)
    
    # Save results
    Path('data').mkdir(exist_ok=True)
    results_df.to_csv('data/model_identification_results.csv', index=False)
    print("✓ Saved model parameters to data/model_identification_results.csv")
    
    # Print summary
    print_model_summary(results_df)
    
    # Generate plots
    plot_fitted_responses(df, steps)
    
    print("\n" + "="*80)
    print("MODEL IDENTIFICATION COMPLETE")
    print("="*80)
    print("\nNext: Run 03_controller_design.py to synthesize multivariable controller")
