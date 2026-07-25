"""
Placeholder Well Simulator
===========================

A realistic simulator for a single naturally flowing oil well, with production
choke as the primary control variable. Implements first-order-plus-dead-time
(FOPDT) dynamics between choke opening and flow/pressure outputs, with
physically plausible nonlinearity and interaction effects.

SIMULATOR ASSUMPTIONS:
- Single well, no commingling
- Natural flow (no artificial lift)
- Constant GOR, water cut, fluid properties
- Quasi-static surface flow (no slugging)
- Choke is the sole production control

OUTPUTS:
- Q: oil flow rate (bbl/hr)
- WHP: wellhead pressure (psia)
- FLP: flowline pressure (psia)
- BHP: bottom-hole pressure (psia)

CONTROL INTERVAL: Ts = 1 hour
"""

import numpy as np
from typing import Tuple


class WellSimulator:
    """
    Placeholder simulator for a naturally flowing oil well.
    
    Implements first-order-plus-dead-time (FOPDT) dynamics for each output
    with plausible nonlinearity in the choke-to-flow gain and pressure response.
    """
    
    def __init__(self):
        """Initialize simulator with well model parameters."""
        
        # ===== WELL MODEL PARAMETERS =====
        # These represent a typical midstream naturally flowing well
        # with surface separation at ~150 psia and modest backpressure from flowline.
        
        # Choke-to-flow gain (Q output)
        # Nonlinear: gain decreases at high choke openings (pressure-squared dependence)
        self.Q_gain_base = 5.8  # bbl/hr per % choke opening (at mid-range opening)
        self.Q_nonlinearity = 0.02  # mild gain reduction at high openings
        
        # Choke-to-pressure gain (negative: higher choke -> lower pressure)
        self.WHP_gain = -0.12  # psia per % choke opening
        self.FLP_gain = -0.08
        self.BHP_gain = -0.06
        
        # Time constants (hours) - first-order lag in response to choke move
        self.Q_tau = 0.8  # Flow responds quickest (fluid in tubing)
        self.WHP_tau = 1.2
        self.FLP_tau = 1.5
        self.BHP_tau = 2.0  # BHP responds slowest (deep reservoir pressure transient)
        
        # Dead times (hours) - delay before choke move propagates
        self.Q_dead_time = 0.25
        self.WHP_dead_time = 0.3
        self.FLP_dead_time = 0.35
        self.BHP_dead_time = 0.4
        
        # Steady-state baseline outputs at 50% choke opening (neutral operating point)
        self.Q_ss_baseline = 100.0  # bbl/hr
        self.WHP_ss_baseline = 150.0  # psia
        self.FLP_ss_baseline = 120.0  # psia
        self.BHP_ss_baseline = 1500.0  # psia
        
        # Safe operating limits (hard constraints)
        self.WHP_min, self.WHP_max = 50.0, 500.0  # psia
        self.FLP_min, self.FLP_max = 5.0, 400.0   # psia
        self.BHP_min, self.BHP_max = 100.0, 5000.0  # psia
        
        # Control interval
        self.Ts = 1.0  # hour
        
        # ===== INTERNAL STATE (FOPDT filters) =====
        # Each output is generated via FOPDT: apply first-order lag + dead time to choke input
        self.choke_history = [50.0] * 20  # circular buffer of past choke positions for dead time
        self.choke_index = 0
        
        # State for first-order filtering (low-pass on steady-state output)
        self.Q_filter_state = self.Q_ss_baseline
        self.WHP_filter_state = self.WHP_ss_baseline
        self.FLP_filter_state = self.FLP_ss_baseline
        self.BHP_filter_state = self.BHP_ss_baseline
        
        # Current outputs
        self.Q = self.Q_ss_baseline
        self.WHP = self.WHP_ss_baseline
        self.FLP = self.FLP_ss_baseline
        self.BHP = self.BHP_ss_baseline
        
        # Time tracking
        self.time = 0.0
        self.step_count = 0
    
    def _apply_fopdt(
        self,
        current_state: float,
        target: float,
        tau: float,
        dead_time: float,
    ) -> float:
        """
        Apply first-order-plus-dead-time (FOPDT) dynamics.
        
        Implements:
            d(state)/dt = (target - state) / tau
        integrated over Ts with exponential update.
        
        Dead time is applied by indexing into choke history.
        
        Args:
            current_state: current filter state
            target: desired steady-state output (from choke position + gain)
            tau: time constant (hours)
            dead_time: dead time (hours)
        
        Returns:
            Updated filter state
        """
        # Discretize first-order: x[k+1] = x[k] + (target - x[k]) * (1 - exp(-Ts/tau))
        alpha = 1.0 - np.exp(-self.Ts / tau)
        new_state = current_state + (target - current_state) * alpha
        return new_state
    
    def _choke_delayed(self, dead_time: float) -> float:
        """
        Return delayed choke position from history.
        
        Args:
            dead_time: dead time in hours
        
        Returns:
            Delayed choke position (%)
        """
        # Convert dead time to steps
        delay_steps = int(np.round(dead_time / self.Ts))
        delay_steps = min(delay_steps, len(self.choke_history) - 1)
        
        # Index into circular buffer
        idx = (self.choke_index - delay_steps) % len(self.choke_history)
        return self.choke_history[idx]
    
    def _compute_gains(self, choke_pos: float) -> Tuple[float, float, float, float]:
        """
        Compute nonlinear choke-to-output gains at current operating point.
        
        Q gain is reduced at high choke openings (pressure-squared flow relationship).
        Pressure gains are assumed roughly linear over operating range.
        
        Args:
            choke_pos: current choke position (%)
        
        Returns:
            (Q_gain, WHP_gain, FLP_gain, BHP_gain)
        """
        # Q gain reduction at high choke (well becomes choke-limited at wide-open)
        choke_fraction = choke_pos / 100.0
        Q_gain = self.Q_gain_base * (1.0 - self.Q_nonlinearity * choke_fraction**2)
        
        # Pressure gains are roughly constant (linear assumption is reasonable)
        WHP_gain = self.WHP_gain
        FLP_gain = self.FLP_gain
        BHP_gain = self.BHP_gain
        
        return Q_gain, WHP_gain, FLP_gain, BHP_gain
    
    def step(self, choke_position: float) -> Tuple[float, float, float, float]:
        """
        Advance simulator by one control interval (1 hour).
        
        Updates internal state using FOPDT dynamics, enforces output limits,
        and returns current outputs.
        
        Args:
            choke_position: desired choke opening (%)
                Must be in [0, 100]; will be clipped if out of range.
        
        Returns:
            (Q, WHP, FLP, BHP)
                Q: oil flow rate (bbl/hr)
                WHP: wellhead pressure (psia)
                FLP: flowline pressure (psia)
                BHP: bottom-hole pressure (psia)
        """
        # Enforce choke bounds
        choke_position = np.clip(choke_position, 0.0, 100.0)
        
        # Store in history for dead-time lookup
        self.choke_history[self.choke_index] = choke_position
        self.choke_index = (self.choke_index + 1) % len(self.choke_history)
        
        # Get delayed choke positions for each output
        choke_Q = self._choke_delayed(self.Q_dead_time)
        choke_WHP = self._choke_delayed(self.WHP_dead_time)
        choke_FLP = self._choke_delayed(self.FLP_dead_time)
        choke_BHP = self._choke_delayed(self.BHP_dead_time)
        
        # Compute nonlinear gains at current choke position
        Q_gain, WHP_gain, FLP_gain, BHP_gain = self._compute_gains(choke_Q)
        
        # Compute steady-state targets from baseline + choke deviation
        Q_target = self.Q_ss_baseline + Q_gain * (choke_Q - 50.0)
        WHP_target = self.WHP_ss_baseline + WHP_gain * (choke_WHP - 50.0)
        FLP_target = self.FLP_ss_baseline + FLP_gain * (choke_FLP - 50.0)
        BHP_target = self.BHP_ss_baseline + BHP_gain * (choke_BHP - 50.0)
        
        # Enforce lower bounds on outputs (physical limits)
        Q_target = max(Q_target, 0.0)
        WHP_target = max(WHP_target, self.WHP_min)
        FLP_target = max(FLP_target, self.FLP_min)
        BHP_target = max(BHP_target, self.BHP_min)
        
        # Apply first-order filtering (FOPDT dynamics)
        self.Q_filter_state = self._apply_fopdt(
            self.Q_filter_state, Q_target, self.Q_tau, self.Q_dead_time
        )
        self.WHP_filter_state = self._apply_fopdt(
            self.WHP_filter_state, WHP_target, self.WHP_tau, self.WHP_dead_time
        )
        self.FLP_filter_state = self._apply_fopdt(
            self.FLP_filter_state, FLP_target, self.FLP_tau, self.FLP_dead_time
        )
        self.BHP_filter_state = self._apply_fopdt(
            self.BHP_filter_state, BHP_target, self.BHP_tau, self.BHP_dead_time
        )
        
        # Clip to hard limits
        self.Q = np.clip(self.Q_filter_state, 0.0, 300.0)
        self.WHP = np.clip(self.WHP_filter_state, self.WHP_min, self.WHP_max)
        self.FLP = np.clip(self.FLP_filter_state, self.FLP_min, self.FLP_max)
        self.BHP = np.clip(self.BHP_filter_state, self.BHP_min, self.BHP_max)
        
        # Advance time
        self.time += self.Ts
        self.step_count += 1
        
        return self.Q, self.WHP, self.FLP, self.BHP
    
    def reset(self, initial_choke_pos: float = 50.0):
        """
        Reset simulator to initial condition.
        
        Args:
            initial_choke_pos: initial choke opening (%), default 50% (mid-range)
        """
        self.choke_history = [initial_choke_pos] * len(self.choke_history)
        self.choke_index = 0
        self.Q_filter_state = self.Q_ss_baseline
        self.WHP_filter_state = self.WHP_ss_baseline
        self.FLP_filter_state = self.FLP_ss_baseline
        self.BHP_filter_state = self.BHP_ss_baseline
        self.time = 0.0
        self.step_count = 0
        
        # Do one step to initialize outputs at baseline
        self.step(initial_choke_pos)
    
    def get_limits(self) -> dict:
        """Return pressure limits for use by controller."""
        return {
            "WHP": (self.WHP_min, self.WHP_max),
            "FLP": (self.FLP_min, self.FLP_max),
            "BHP": (self.BHP_min, self.BHP_max),
        }


if __name__ == "__main__":
    # Quick sanity check: run simulator at fixed choke, verify steady-state
    sim = WellSimulator()
    print("Well Simulator Initialized")
    print(f"Baseline outputs (choke=50%): Q={sim.Q:.1f} bbl/hr, "
          f"WHP={sim.WHP:.1f} psia, FLP={sim.FLP:.1f} psia, BHP={sim.BHP:.1f} psia")
    
    # Step to 60% choke and observe transient
    print("\nStepping to 60% choke opening:")
    for i in range(10):
        Q, WHP, FLP, BHP = sim.step(60.0)
        print(f"Step {i+1}: Q={Q:.1f}, WHP={WHP:.1f}, FLP={FLP:.1f}, BHP={BHP:.1f}")
