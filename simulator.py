"""
simulator.py: Well Production Dynamics Simulator
=====================================================

Physically-motivated nonlinear model of well response to choke openings.

STATE VARIABLES:
  - Reservoir pressure (BHP): ~3000 psia baseline
  - Choke valve position: 0–100%
  - Well production rate (Q): nonlinear function of BHP and choke
  - Surface pressures: WHP (wellhead), FLP (flowline)

KEY PHYSICS:
  1. Inflow (Darcy): Q ∝ (BHP - BHP_bubble) / μ * (1/re + 1/rw + skin)
  2. Outflow (valve): Depends on choke opening & downstream pressure
  3. Pressure dynamics: Material balance (BHP depletion from production)
  4. Time constants: Natural lag from fluid compressibility & pipe volumes
  5. Valve nonlinearity: Gain and lag vary with opening position

CALIBRATION TARGETS:
  - WHP range: 50–500 psia (typical separator pressure)
  - FLP range: 5–400 psia (lower pressure drop across choke at high choke %)
  - BHP range: 100–5000 psia (depletion + inflow matching)
  - Q range: 100–5000 bbl/hr (production plateau then decline)

SIMULATION SAMPLING: Ts = 1 minute (60 seconds)
"""

import numpy as np


class WellSimulator:
    """
    Nonlinear well simulator with choke control input.
    
    Outputs:
      - Q: oil production rate (bbl/hr)
      - WHP: wellhead pressure (psia)
      - FLP: flowline pressure (psia)
      - BHP: bottom-hole flowing pressure (psia)
    """
    
    def __init__(self):
        # Sampling time (minutes)
        self.Ts = 1.0 / 60.0  # 1 second
        
        # Initial reservoir state
        self.BHP_initial = 3000.0  # psia
        self.BHP_min = 100.0  # psia (bubble point)
        self.BHP_depletion_rate = 0.05  # psia/bbl produced
        
        # Pressure drop model: ΔP = a * Q^2 / (choke_opening / 100)^2 + b * Q
        # (includes Darcy + acceleration effects)
        self.pressure_drop_a = 0.001  # quadratic term coefficient
        self.pressure_drop_b = 0.01   # linear term coefficient
        
        # Valve flow coefficient (reduces effective choke opening at low %)
        self.valve_beta = 1.2  # nonlinearity exponent
        
        # Time constants (first-order lags)
        self.tau_Q = 0.5  # hours (~30 min for flow rate response)
        self.tau_WHP = 0.25  # hours (~15 min for pressure response)
        self.tau_FLP = 0.1  # hours (~6 min for flowline response)
        self.tau_BHP = 2.0  # hours (~2 hr for reservoir response)
        
        # Steady-state parameters
        self.J_index = 1.0  # productivity index (bbl/hr per psia)
        self.separator_pressure = 50.0  # psia (downstream of choke)
        
        # State
        self.BHP = self.BHP_initial
        self.Q = 0.0
        self.WHP = 50.0
        self.FLP = 5.0
        self.Q_cumulative = 0.0  # for BHP depletion
    
    def reset(self, initial_choke_pos=50.0):
        """Reset simulator to baseline state."""
        self.BHP = self.BHP_initial
        self.Q = self._calculate_production_rate(self.BHP, initial_choke_pos)
        self.WHP = 50.0
        self.FLP = 5.0
        self.Q_cumulative = 0.0
    
    def step(self, choke_opening):
        """
        Advance simulator one time step with constant choke opening.
        
        Args:
            choke_opening: Choke valve opening (0–100%)
        
        Returns:
            (Q, WHP, FLP, BHP): Current outputs after time step
        """
        
        # Limit choke opening
        choke_opening = np.clip(choke_opening, 0.0, 100.0)
        
        # === INFLOW: Production rate from reservoir ===
        Q_ss = self._calculate_production_rate(self.BHP, choke_opening)
        
        # First-order lag for flow rate response
        self.Q += (Q_ss - self.Q) * (self.Ts / self.tau_Q)
        self.Q = np.clip(self.Q, 0.0, 5000.0)  # Limit to max capacity
        
        # === RESERVOIR DEPLETION ===
        # BHP declines as we produce (simplified depletion)
        self.Q_cumulative += self.Q * self.Ts  # Accumulate production
        BHP_depleted = self.BHP_initial - self.BHP_depletion_rate * (self.Q_cumulative / 1000.0)
        BHP_depleted = np.clip(BHP_depleted, self.BHP_min, self.BHP_initial)
        
        # Slow approach to depletion (time constant for reservoir)
        self.BHP += (BHP_depleted - self.BHP) * (self.Ts / self.tau_BHP)
        
        # === PRESSURE DYNAMICS ===
        # Wellhead pressure: decreases with higher flow (higher friction loss)
        # Simplified: WHP = separator_pressure + friction_drop(Q)
        WHP_ss = self.separator_pressure + self._friction_pressure_drop(self.Q)
        self.WHP += (WHP_ss - self.WHP) * (self.Ts / self.tau_WHP)
        self.WHP = np.clip(self.WHP, 50.0, 500.0)
        
        # Flowline pressure: depends on choke flow characteristics
        # FLP = downstream pressure + pressure rise due to choke restriction
        choke_delta_P = self._choke_pressure_drop(self.Q, choke_opening)
        FLP_ss = self.separator_pressure + choke_delta_P * 0.5  # Half the total drop
        self.FLP += (FLP_ss - self.FLP) * (self.Ts / self.tau_FLP)
        self.FLP = np.clip(self.FLP, 5.0, 400.0)
        
        return self.Q, self.WHP, self.FLP, self.BHP
    
    def _calculate_production_rate(self, BHP, choke_opening):
        """
        Steady-state production rate based on BHP and choke opening.
        
        Nonlinear relationships:
          1. Inflow limited by BHP (reservoir pressure)
          2. Outflow limited by choke opening
        """
        
        if BHP <= self.BHP_min:
            return 0.0
        
        # Inflow: Darcy law (simplified)
        inflow_max = self.J_index * (BHP - self.BHP_min)
        
        # Outflow: Limited by choke valve
        # Effective opening: β-law nonlinearity (low openings are restrictive)
        effective_choke = (choke_opening / 100.0) ** self.valve_beta
        
        if effective_choke < 0.01:
            return 0.0
        
        # Combined effect: whichever is more restrictive
        Q_choke_limit = 5000.0 * effective_choke  # Max at 100% opening
        
        Q_ss = min(inflow_max, Q_choke_limit)
        
        return np.clip(Q_ss, 0.0, 5000.0)
    
    def _friction_pressure_drop(self, Q):
        """
        Pressure drop due to friction losses in wellbore/tubing.
        Increases quadratically with flow rate.
        """
        return 0.005 * Q ** 1.8
    
    def _choke_pressure_drop(self, Q, choke_opening):
        """
        Pressure drop across choke valve.
        
        ΔP_choke ∝ Q^2 / (effective_choke_area)^2
        At low choke %, valve is highly restrictive (large ΔP).
        """
        
        if Q < 1.0:
            return 1.0  # Minimum pressure to maintain flow
        
        effective_choke = (choke_opening / 100.0) ** self.valve_beta
        
        if effective_choke < 0.01:
            return 1000.0  # Very high pressure drop at near-closed
        
        # ΔP = a*Q^2 / effective_choke^2 + b*Q
        delta_P = (self.pressure_drop_a * Q ** 2 / (effective_choke ** 2) +
                   self.pressure_drop_b * Q)
        
        return np.clip(delta_P, 1.0, 1000.0)
