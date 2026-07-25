"""
03_controller_design.py: Multivariable Controller Synthesis
=============================================================

Designs a decoupling + PID controller for autonomous choke management.

CONTROL ARCHITECTURE:
  1. Decoupler: Compensates for cross-coupling between outputs
  2. PID loops: Setpoint tracking for WHP, FLP, and production rate Q
  3. Priority: Safety limits (BHP, pressures) > Set-point tracking

STRUCTURE:
  Reference setpoints → Decoupler → PID regulators → Choke command
  
DECOUPLING PHILOSOPHY:
  - Choke opening primarily affects flow rate (Q)
  - Q affects WHP and FLP (pressure feedback)
  - Use inverse models to pre-compensate for cross-coupling

CONTROLLER GAINS:
  - Tuned for setpoint tracking with ~5% overshoot
  - Time to settle: ~2 hours
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class MultivariableController:
    """
    Decoupling + PID controller for autonomous production choke.
    
    Inputs:
      - WHP_ref: Wellhead pressure setpoint (psia)
      - FLP_ref: Flowline pressure setpoint (psia)
      - Q_ref: Production rate setpoint (bbl/hr)
    
    Outputs:
      - choke_cmd: Choke valve command (0–100%)
    
    Internal:
      - Error integrals for each output
      - Decoupling compensation
    """
    
    def __init__(self):
        # Sampling time (hours)
        self.Ts = 1.0 / 60.0  # 1 minute
        
        # PID gains (tuned for ~5% overshoot, 2-hour settle)
        self.Kp_Q = 0.01      # Production rate proportional
        self.Ki_Q = 0.002     # Production rate integral
        self.Kd_Q = 0.0       # Production rate derivative (optional)
        
        self.Kp_WHP = 0.05    # Wellhead pressure proportional
        self.Ki_WHP = 0.01    # Wellhead pressure integral
        self.Kd_WHP = 0.0
        
        self.Kp_FLP = 0.05    # Flowline pressure proportional
        self.Ki_FLP = 0.01    # Flowline pressure integral
        self.Kd_FLP = 0.0
        
        # Decoupling gains (inverse model approximation)
        self.decoup_Q_from_WHP = -0.002   # ∂Q/∂WHP compensation
        self.decoup_Q_from_FLP = -0.001   # ∂Q/∂FLP compensation
        self.decoup_WHP_from_Q = 0.01     # ∂WHP/∂Q compensation
        self.decoup_FLP_from_Q = 0.005    # ∂FLP/∂Q compensation
        
        # Integral error accumulators
        self.int_err_Q = 0.0
        self.int_err_WHP = 0.0
        self.int_err_FLP = 0.0
        
        # Output history for derivative
        self.prev_Q = 0.0
        self.prev_WHP = 0.0
        self.prev_FLP = 0.0
        
        # Choke command saturation
        self.choke_min = 5.0   # Minimum 5% to avoid dead zone
        self.choke_max = 95.0  # Maximum 95% to maintain control range
        
        # Reference setpoints
        self.Q_ref = 2500.0      # bbl/hr (mid-range production)
        self.WHP_ref = 200.0     # psia
        self.FLP_ref = 100.0     # psia
        
        # Safety limits
        self.BHP_min = 150.0     # psia (minimum to avoid negative pressure)
        self.WHP_max = 400.0     # psia (separator relief)
        self.FLP_max = 300.0     # psia (line relief)
        
    def set_setpoints(self, Q_ref, WHP_ref, FLP_ref):
        """
        Update controller setpoints.
        
        Args:
            Q_ref: Production rate setpoint (bbl/hr)
            WHP_ref: Wellhead pressure setpoint (psia)
            FLP_ref: Flowline pressure setpoint (psia)
        """
        self.Q_ref = np.clip(Q_ref, 0.0, 5000.0)
        self.WHP_ref = np.clip(WHP_ref, 50.0, self.WHP_max)
        self.FLP_ref = np.clip(FLP_ref, 5.0, self.FLP_max)
    
    def compute_command(self, Q_meas, WHP_meas, FLP_meas, BHP_meas):
        """
        Compute choke valve command using decoupling + PID.
        
        Args:
            Q_meas, WHP_meas, FLP_meas, BHP_meas: Measured outputs
        
        Returns:
            choke_cmd: Command to valve (0–100%)
        """
        
        # === SAFETY CHECKS ===
        if BHP_meas < self.BHP_min:
            print(f"  WARNING: BHP={BHP_meas:.1f} < min={self.BHP_min:.1f} psia. Closing choke.")
            return 10.0  # Near-closed to preserve pressure
        
        if WHP_meas > self.WHP_max or FLP_meas > self.FLP_max:
            print(f"  WARNING: Over-pressure (WHP={WHP_meas:.1f}, FLP={FLP_meas:.1f}). Closing choke.")
            return 10.0
        
        # === COMPUTE ERRORS ===
        err_Q = self.Q_ref - Q_meas
        err_WHP = self.WHP_ref - WHP_meas
        err_FLP = self.FLP_ref - FLP_meas
        
        # === INTEGRAL ACCUMULATION ===
        self.int_err_Q += err_Q * self.Ts
        self.int_err_WHP += err_WHP * self.Ts
        self.int_err_FLP += err_FLP * self.Ts
        
        # Anti-windup: limit integral terms
        max_integral = 100.0
        self.int_err_Q = np.clip(self.int_err_Q, -max_integral, max_integral)
        self.int_err_WHP = np.clip(self.int_err_WHP, -max_integral, max_integral)
        self.int_err_FLP = np.clip(self.int_err_FLP, -max_integral, max_integral)
        
        # === PID CALCULATIONS ===
        # Q loop: proportional + integral
        pid_Q = (self.Kp_Q * err_Q + self.Ki_Q * self.int_err_Q +
                self.Kd_Q * (Q_meas - self.prev_Q) / self.Ts)
        
        # WHP loop: proportional + integral
        pid_WHP = (self.Kp_WHP * err_WHP + self.Ki_WHP * self.int_err_WHP +
                  self.Kd_WHP * (WHP_meas - self.prev_WHP) / self.Ts)
        
        # FLP loop: proportional + integral
        pid_FLP = (self.Kp_FLP * err_FLP + self.Ki_FLP * self.int_err_FLP +
                  self.Kd_FLP * (FLP_meas - self.prev_FLP) / self.Ts)
        
        # === DECOUPLING ===
        # Compensate for cross-coupling effects
        decoup_compensation = (
            self.decoup_Q_from_WHP * WHP_meas +
            self.decoup_Q_from_FLP * FLP_meas +
            self.decoup_WHP_from_Q * Q_meas +
            self.decoup_FLP_from_Q * Q_meas
        )
        
        # === PRIORITY WEIGHTING ===
        # Production rate (Q) is primary control variable
        # Pressures (WHP, FLP) are secondary (safety limits)
        w_Q = 0.7
        w_WHP = 0.15
        w_FLP = 0.15
        
        choke_error_offset = (w_Q * pid_Q + w_WHP * pid_WHP + w_FLP * pid_FLP +
                             decoup_compensation)
        
        # === CHOKE COMMAND ===
        # Base command: 50% (nominal operation)
        choke_nominal = 50.0
        choke_cmd = choke_nominal + choke_error_offset
        
        # Saturation
        choke_cmd = np.clip(choke_cmd, self.choke_min, self.choke_max)
        
        # === UPDATE STATE ===
        self.prev_Q = Q_meas
        self.prev_WHP = WHP_meas
        self.prev_FLP = FLP_meas
        
        return choke_cmd
    
    def reset_integrators(self):
        """Reset integral error accumulators."""
        self.int_err_Q = 0.0
        self.int_err_WHP = 0.0
        self.int_err_FLP = 0.0


if __name__ == "__main__":
    print("\n" + "="*80)
    print("CONTROLLER DESIGN MODULE")
    print("="*80)
    
    # Instantiate controller
    controller = MultivariableController()
    
    print("\nController Parameters:")
    print(f"  Production rate (Q):")
    print(f"    Kp={controller.Kp_Q}, Ki={controller.Ki_Q}")
    print(f"  Wellhead pressure (WHP):")
    print(f"    Kp={controller.Kp_WHP}, Ki={controller.Ki_WHP}")
    print(f"  Flowline pressure (FLP):")
    print(f"    Kp={controller.Kp_FLP}, Ki={controller.Ki_FLP}")
    
    print(f"\nChoke Saturation: {controller.choke_min}% – {controller.choke_max}%")
    print(f"\nDefault Setpoints:")
    print(f"  Q_ref = {controller.Q_ref} bbl/hr")
    print(f"  WHP_ref = {controller.WHP_ref} psia")
    print(f"  FLP_ref = {controller.FLP_ref} psia")
    
    print(f"\nSafety Limits:")
    print(f"  BHP_min = {controller.BHP_min} psia")
    print(f"  WHP_max = {controller.WHP_max} psia")
    print(f"  FLP_max = {controller.FLP_max} psia")
    
    print("\n" + "="*80)
    print("Controller ready for closed-loop simulation.")
    print("Invoke via: 04_closed_loop_simulation.py")
    print("="*80)
